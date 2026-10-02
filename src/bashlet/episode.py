"""Task-independent bash interaction: instruction + workspace + model → Episode."""

import hashlib
import json
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from . import protocol
from .config import ExperimentConfig, config_record, load_config
from .models import (
    ContextExhausted,
    EpisodeResult,
    InteractionBudget,
    InvalidAction,
    Isolation,
    Model,
    ModelBackend,
    ModelCall,
    ModelResponseError,
    SamplingParams,
    Step,
    Usage,
)
from .sandbox import Sandbox
from .spend import BudgetRefused
from .workspace import capture_workspace

_PROBE_OUTPUT_BYTES = 16_384


def _verify_isolation(sandbox: Sandbox, run_dir: Path) -> Isolation:
    probe_dir = run_dir / "probe"
    probe_dir.mkdir()
    canary = run_dir / "private" / "host-private.txt"
    canary.write_text("host-private-canary")
    paths = {
        "host_private": str(canary),
        "host_log": str(run_dir / "run.json"),
    }
    namespaces = {
        name: os.readlink(f"/proc/self/ns/{name}")
        for name in ("mnt", "pid", "net", "user", "ipc", "uts")
    }
    readonly_mounts = [
        "/etc",
        "/opt/bin",
        sandbox.environment.python.target,
        *(mount.target for mount in sandbox.environment.ro_bind),
    ]
    code = f"""
import grp, json, os, pathlib, platform, pwd, socket, subprocess, sys
import pytest
checks = {{}}
for name, host_id in {namespaces!r}.items():
    checks['namespace_' + name] = os.readlink('/proc/self/ns/' + name) != host_id
expected = {sandbox.variables!r}
checks['environment'] = (
    set(os.environ) <= set(expected) | {{'LC_CTYPE', 'PWD'}}
    and all(os.environ.get(key) == value for key, value in expected.items())
)
checks['working_directory'] = os.environ.get('PWD') == '/workspace'
for name, path in {paths!r}.items():
    for prefix in ('', '/proc/1/root'):
        try:
            pathlib.Path(prefix + path).read_bytes()
        except OSError:
            checks['denied_' + name + prefix] = True
        else:
            checks['denied_' + name + prefix] = False
pathlib.Path('/workspace/writable').write_text('ok')
checks['workspace_writable'] = pathlib.Path('/workspace/writable').read_text() == 'ok'
home = pathlib.Path(os.environ['HOME'])
(home / 'writable').write_text('ok')
checks['home_writable'] = (home / 'writable').read_text() == 'ok'
checks['account'] = pwd.getpwuid(os.getuid()).pw_name == 'agent'
checks['group'] = grp.getgrgid(os.getgid()).gr_name == 'agent'
checks['localhost'] = socket.gethostbyname('localhost') == '127.0.0.1'
for mount in {readonly_mounts!r}:
    checks['readonly_' + mount] = bool(os.statvfs(mount).f_flag & os.ST_RDONLY)
checks['python_version'] = (
    platform.python_version() == {sandbox.environment.python_version!r}
)
checks['pytest_version'] = pytest.__version__ == {sandbox.environment.pytest_version!r}
status = pathlib.Path('/proc/self/status').read_text()
checks['capabilities_dropped'] = 'CapEff:\\t0000000000000000' in status
tools = {{
    'awk': (['awk', 'BEGIN {{print 7}}'], '7\\n'),
    'sed': (['sed', 's/old/new/'], 'new\\n'),
    'grep': (['grep', 'old'], 'old\\n'),
    'git': (['git', 'init', '-q', str(home / 'repo')], ''),
    'pytest': (['pytest', '--version'], 'pytest ' + pytest.__version__ + '\\n'),
}}
for name, (argv, output) in tools.items():
    try:
        tool = subprocess.run(argv, input='old\\n', capture_output=True, text=True)
        checks['tool_' + name] = tool.returncode == 0 and tool.stdout == output
    except OSError:
        checks['tool_' + name] = False
report = {{'checks': checks, 'python': sys.version, 'pytest': pytest.__version__}}
print(json.dumps(report))
"""
    # Capability checks must not fail just because the episode has a tiny output budget.
    probe = Sandbox(
        replace(
            sandbox.environment,
            timeout_seconds=sandbox.environment.probe_timeout_seconds,
            output_bytes=_PROBE_OUTPUT_BYTES,
        )
    )
    observation = probe.execute(probe_dir, [probe.python, "-I", "-B", "-c", code])
    if observation.status == "interrupted":
        raise KeyboardInterrupt
    if observation.status != "ok":
        return Isolation(False, reason=f"{observation.status}: {observation.stderr}")
    try:
        report = json.loads(observation.stdout)
        checks = report["checks"]
        if not checks or not all(value is True for value in checks.values()):
            return Isolation(False, checks, "An isolation access probe failed")
        with open(sandbox.bwrap, "rb") as executable:
            executable_hash = hashlib.file_digest(executable, "sha256").hexdigest()
        return Isolation(
            True,
            checks,
            runtime={
                "task_python": report["python"],
                "task_pytest": report["pytest"],
                "python_environment": sandbox.environment.python.source,
                "kernel": os.uname().release,
                "bwrap_sha256": executable_hash,
            },
        )
    except (ValueError, KeyError, TypeError):
        return Isolation(False, reason="Invalid isolation probe response")


class _SamplerError(Exception):
    """The sampling call raised instead of returning a Generation."""


def _sample(model: Model, context: list[int], sampling: SamplingParams) -> ModelCall:
    started = time.monotonic()
    try:
        generation = model.sample(tuple(context), sampling)
    except (ModelResponseError, ContextExhausted, BudgetRefused):
        raise
    except Exception as error:
        # The call itself failed (service, network, Adapter): not model output.
        raise _SamplerError(f"{type(error).__name__}: {error}") from error
    duration = time.monotonic() - started
    generation.validate(sampling)
    try:
        text = model.encoding.decode(generation.tokens)
        reply = model.encoding.reply(generation.tokens)
    except Exception as error:
        raise ModelResponseError(
            f"Decoding failed: {type(error).__name__}: {error}", generation
        ) from error
    return ModelCall(
        model.identity, sampling, len(context), generation, text, duration, reply
    )


def _budget_exhausted_reason(budget: InteractionBudget, usage: Usage) -> str:
    if usage.steps >= budget.max_steps:
        return f"Reached max_steps={budget.max_steps}"
    if usage.generated_tokens >= budget.max_generated_tokens:
        return f"Reached max_generated_tokens={budget.max_generated_tokens}"
    if usage.elapsed_seconds >= budget.max_seconds:
        return f"Reached max_seconds={budget.max_seconds}"
    return ""


def save_record(
    result: EpisodeResult,
    condition: ExperimentConfig,
    extra: dict[str, object] | None = None,
) -> None:
    """Persist operator-only evidence; no record is mounted into the workspace."""
    sandbox = Sandbox(condition.environment)
    record = (
        asdict(result)
        | {
            "schema_version": 10,
            "config": config_record(condition),
            "fingerprints": protocol.fingerprints(condition),
            "protocol": protocol.describe(condition.protocol),
            "sampling": asdict(condition.model.sampling),
            "isolation_policy": "linux-bwrap-v1",
            "effective_resource_limits": sandbox.resource_limits,
            "isolation_probe_limits": {
                "timeout_seconds": condition.environment.probe_timeout_seconds,
                "output_bytes": _PROBE_OUTPUT_BYTES,
            },
            "host_python": sys.version,
            "bwrap": condition.environment.bwrap,
        }
        | (extra or {})
    )
    pending = result.run_dir / "run.json.tmp"
    pending.write_text(json.dumps(record, default=str, ensure_ascii=True, indent=2))
    pending.replace(result.run_dir / "run.json")


def run_episode(
    *,
    instruction: str,
    workspace: Path,
    model: Model,
    config: ExperimentConfig | None = None,
    run_dir: Path | None = None,
    on_update: Callable[[EpisodeResult], None] | None = None,
) -> EpisodeResult:
    """Run one Episode on an existing workspace, without Task data or grading.

    ``config`` is the resolved versioned configuration. ``run_dir`` owns the
    operator-only records and blobs; by default a unique directory under runs/.
    ``on_update`` can persist an experiment record or render progress for a CLI.
    The workspace is modified in place and remains available after termination.
    """
    condition = config if config is not None else load_config()
    run_dir = (run_dir or Path("runs") / uuid.uuid4().hex).resolve()
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    (run_dir / "private").mkdir(exist_ok=True, mode=0o700)
    workspace = workspace.resolve()
    sandbox = Sandbox(condition.environment)
    budget, sampling = condition.budget, condition.model.sampling
    backend = model if isinstance(model, ModelBackend) else None
    result = EpisodeResult(
        run_dir=run_dir,
        workspace=workspace,
        limits=sandbox.limits,
        isolation=Isolation(False, reason="Not yet checked"),
        policy=model.identity,
        encoding=model.encoding.identity,
        encoding_details=model.encoding.describe(),
        budget=budget,
    )

    def save() -> None:
        if backend is not None:
            result.policy_details = backend.describe()
        if on_update is None:
            save_record(result, condition)
        else:
            on_update(result)

    started: float | None = None
    recording_step: int | None = None
    try:
        save()
        result.initial_workspace_manifest = capture_workspace(workspace, run_dir)
        result.isolation = _verify_isolation(sandbox, run_dir)
        if not result.isolation.passed:
            result.status = "infrastructure_error"
            result.reason = result.isolation.reason
            return result
        if backend is not None:
            try:
                backend.probe()
            except (ModelResponseError, ContextExhausted, BudgetRefused):
                raise
            except Exception as error:
                raise _SamplerError(f"Capability probe failed: {error}") from error
        started = time.monotonic()

        def record_step(step: Step) -> None:
            nonlocal recording_step
            result.usage.elapsed_seconds = time.monotonic() - started
            recording_step = len(result.trajectory)
            # Preserve action/model/output evidence before interruptible file I/O.
            result.trajectory.append(replace(step, usage=replace(result.usage)))
            manifest = capture_workspace(workspace, run_dir)
            result.usage.elapsed_seconds = time.monotonic() - started
            result.trajectory[recording_step] = replace(
                step, workspace_manifest=manifest, usage=replace(result.usage)
            )
            recording_step = None

        context = result.context_tokens
        context.extend(
            model.encoding.prompt(condition.protocol.system_prompt, instruction)
        )
        consecutive_invalid = 0
        while True:
            result.usage.elapsed_seconds = time.monotonic() - started
            reason = _budget_exhausted_reason(budget, result.usage)
            if reason:
                result.status, result.reason = "budget_exhausted", reason
                result.budget_kind = "interaction"
                return result
            remaining = budget.max_generated_tokens - result.usage.generated_tokens
            if backend is not None:
                context_remaining = backend.max_context_tokens - len(context)
                if context_remaining <= 0:
                    raise ContextExhausted(
                        f"Reached model context limit={backend.max_context_tokens}"
                    )
                remaining = min(remaining, context_remaining)
            request = replace(sampling, max_tokens=min(sampling.max_tokens, remaining))
            call = _sample(model, context, request)
            context.extend(call.generation.tokens)
            result.usage.generated_tokens += len(call.generation.tokens)
            if call.generation.stop_reason == "context":
                result.usage.steps += 1
                reason = "Generation reached the model context limit"
                record_step(Step(InvalidAction(reason), model_call=call))
                result.status, result.reason = "context_exhausted", reason
                return result
            assert call.reply is not None
            action = (
                protocol.parse_action(
                    call.reply.action_text, call.generation.stop_reason == "length"
                )
                if call.reply.thinking_complete
                else InvalidAction("Native thinking did not reach its end marker")
            )
            result.usage.steps += 1
            if isinstance(action, InvalidAction):
                result.usage.invalid_actions += 1
                consecutive_invalid += 1
                feedback = protocol.render_invalid(
                    action, condition.protocol.invalid_action_format
                )
                record_step(Step(action, None, call, feedback))
                if consecutive_invalid >= budget.max_consecutive_invalid:
                    result.status = "format_error"
                    result.reason = (
                        f"{consecutive_invalid} consecutive invalid actions: "
                        + action.reason
                    )
                    return result
            else:
                consecutive_invalid = 0
                observation = sandbox.execute(
                    workspace,
                    ["/bin/bash", "--noprofile", "--norc", "-c", action.command],
                    merge_output=True,
                )
                if observation.status == "interrupted":
                    record_step(Step(action, observation, call))
                    result.status, result.reason = "interrupted", "Episode interrupted"
                    return result
                if observation.status == "infrastructure_error":
                    record_step(Step(action, observation, call))
                    result.status = "infrastructure_error"
                    result.reason = observation.output
                    return result
                submitted = protocol.submission_text(
                    observation, condition.protocol.submission_marker
                )
                if submitted is not None:
                    record_step(Step(action, observation, call))
                    result.status = "submitted"
                    result.submission_text = submitted
                    return result
                feedback = protocol.render_observation(observation, condition.protocol)
                record_step(Step(action, observation, call, feedback))
            # Feedback is appended only after its action has actually run.
            context.extend(
                model.encoding.continuation(call.generation.tokens, feedback)
            )
            save()
    except BudgetRefused as error:
        result.status, result.reason = "budget_exhausted", str(error)
        result.budget_kind = "spend"
        return result
    except ContextExhausted as error:
        result.status, result.reason = "context_exhausted", str(error)
        return result
    except ModelResponseError as error:
        # Rejected output is kept for audit but never enters context or executes.
        result.rejected_generation = error.generation
        if started is not None and error.generation is not None:
            result.usage.generated_tokens += len(error.generation.tokens)
        result.status, result.reason = "model_error", str(error)
        return result
    except _SamplerError as error:
        result.status, result.reason = "infrastructure_error", str(error)
        return result
    except OSError as error:
        result.status = "infrastructure_error"
        result.reason = str(error)
        return result
    except KeyboardInterrupt:
        result.status = "interrupted"
        result.reason = "Episode interrupted"
        if recording_step is not None:
            manifest = capture_workspace(workspace, run_dir)
            if started is not None:
                result.usage.elapsed_seconds = time.monotonic() - started
            result.trajectory[recording_step] = replace(
                result.trajectory[recording_step],
                workspace_manifest=manifest,
                usage=replace(result.usage),
            )
        elif started is None:
            result.initial_workspace_manifest = capture_workspace(workspace, run_dir)
        return result
    finally:
        if started is not None:
            result.usage.elapsed_seconds = time.monotonic() - started
        save()
