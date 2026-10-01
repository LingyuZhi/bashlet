"""Task-independent bash interaction: instruction + workspace + model → Episode."""

import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

from . import protocol
from .config import ExperimentConfig, load_config
from .models import (
    EpisodeResult,
    Generation,
    InteractionBudget,
    InvalidAction,
    Isolation,
    Model,
    ModelCall,
    SamplingParams,
    Step,
    Submit,
    Usage,
)
from .sandbox import Sandbox
from .workspace import capture_workspace

_PROBE_TIMEOUT_SECONDS = 5.0
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
    code = f"""
import json, os, pathlib, sys
checks = {{}}
for name, host_id in {namespaces!r}.items():
    checks['namespace_' + name] = os.readlink('/proc/self/ns/' + name) != host_id
checks['environment'] = set(os.environ) <= {{'PATH', 'LANG', 'LC_CTYPE', 'PWD'}}
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
print(json.dumps({{'checks': checks, 'python': sys.version}}))
"""
    # Capability checks must not fail just because the episode has a tiny output budget.
    probe = Sandbox(
        replace(
            sandbox.environment,
            timeout_seconds=_PROBE_TIMEOUT_SECONDS,
            output_bytes=_PROBE_OUTPUT_BYTES,
        )
    )
    observation = probe.execute(probe_dir, ["/usr/bin/python3", "-I", "-c", code])
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
                "kernel": os.uname().release,
                "bwrap_sha256": executable_hash,
            },
        )
    except (ValueError, KeyError, TypeError):
        return Isolation(False, reason="Invalid isolation probe response")


class _SamplerError(Exception):
    """The sampling call raised instead of returning a Generation."""


class _ModelError(Exception):
    """The returned Generation broke the sampling contract or cannot be decoded."""

    def __init__(self, reason: str, generation: Generation | None = None) -> None:
        super().__init__(reason)
        self.generation = generation


def _sample(model: Model, context: list[int], sampling: SamplingParams) -> ModelCall:
    started = time.monotonic()
    try:
        generation = model.sample(tuple(context), sampling)
    except Exception as error:
        # The call itself failed (service, network, Adapter): not model output.
        raise _SamplerError(f"{type(error).__name__}: {error}") from error
    duration = time.monotonic() - started
    tokens, logprobs = generation.tokens, generation.logprobs
    problem = ""
    if generation.source not in ("sampled", "replayed"):
        problem = f"Unknown generation source {generation.source!r}"
    elif generation.source == "replayed" and logprobs is not None:
        problem = "Replayed generations cannot carry sampling logprobs"
    elif generation.source == "sampled" and logprobs is None:
        problem = "Sampled generations require per-token logprobs"
    elif (
        generation.source == "sampled"
        and logprobs is not None
        and len(tokens) != len(logprobs)
    ):
        problem = f"{len(tokens)} tokens but {len(logprobs)} logprobs"
    elif len(tokens) > sampling.max_tokens:
        problem = f"{len(tokens)} tokens exceed max_tokens={sampling.max_tokens}"
    elif not all(type(token) is int and token >= 0 for token in tokens):
        problem = "Token IDs must be non-negative integers"
    elif logprobs is not None and not all(
        math.isfinite(value) and value <= 0 for value in logprobs
    ):
        problem = "Logprobs must be finite and non-positive"
    elif generation.stop_reason not in ("stop", "length"):
        problem = f"Unknown stop reason {generation.stop_reason!r}"
    if problem:
        raise _ModelError(problem, generation)
    try:
        text = model.encoding.decode(tokens)
    except Exception as error:
        raise _ModelError(
            f"Decoding failed: {type(error).__name__}: {error}", generation
        ) from error
    return ModelCall(model.identity, sampling, len(context), generation, text, duration)


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
            "schema_version": 6,
            "config": asdict(condition),
            "fingerprints": protocol.fingerprints(condition),
            "protocol": protocol.describe(condition.protocol),
            "sampling": asdict(condition.model.sampling),
            "isolation_policy": "linux-bwrap-v1",
            "effective_resource_limits": sandbox.resource_limits,
            "isolation_probe_limits": {
                "timeout_seconds": _PROBE_TIMEOUT_SECONDS,
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
    result = EpisodeResult(
        run_dir=run_dir,
        workspace=workspace,
        limits=sandbox.limits,
        isolation=Isolation(False, reason="Not yet checked"),
        policy=model.identity,
        encoding=model.encoding.identity,
        budget=budget,
    )

    def save() -> None:
        if on_update is None:
            save_record(result, condition)
        else:
            on_update(result)

    save()
    started: float | None = None
    try:
        result.initial_workspace_manifest = capture_workspace(workspace, run_dir)
        result.isolation = _verify_isolation(sandbox, run_dir)
        if not result.isolation.passed:
            result.status = "infrastructure_error"
            result.reason = result.isolation.reason
            return result
        started = time.monotonic()

        def record_step(step: Step) -> None:
            result.trajectory.append(
                replace(step, workspace_manifest=capture_workspace(workspace, run_dir))
            )

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
                return result
            remaining = budget.max_generated_tokens - result.usage.generated_tokens
            request = replace(sampling, max_tokens=min(sampling.max_tokens, remaining))
            try:
                call = _sample(model, context, request)
            except _ModelError as error:
                # Rejected output is kept for audit but never enters context.
                result.rejected_generation = error.generation
                if error.generation is not None:
                    result.usage.generated_tokens += len(error.generation.tokens)
                result.status, result.reason = "model_error", str(error)
                return result
            except _SamplerError as error:
                result.status = "infrastructure_error"
                result.reason = f"Model call failed: {error}"
                return result
            context.extend(call.generation.tokens)
            result.usage.generated_tokens += len(call.generation.tokens)
            action = protocol.parse_action(
                call.text, call.generation.stop_reason == "length"
            )
            result.usage.steps += 1
            if isinstance(action, Submit):
                record_step(Step(action, None, call))
                save()
                result.status = "submitted"
                result.submission_text = ""
                return result
            if isinstance(action, InvalidAction):
                result.usage.invalid_actions += 1
                consecutive_invalid += 1
                if consecutive_invalid >= budget.max_consecutive_invalid:
                    record_step(Step(action, None, call))
                    result.status = "invalid_action"
                    result.reason = (
                        f"{consecutive_invalid} consecutive invalid actions: "
                        + action.reason
                    )
                    return result
                feedback = protocol.render_invalid(
                    action, condition.protocol.invalid_action_format
                )
                record_step(Step(action, None, call, feedback))
            else:
                consecutive_invalid = 0
                observation = sandbox.execute(
                    workspace,
                    ["/bin/bash", "--noprofile", "--norc", "-c", action.command],
                )
                if observation.status == "infrastructure_error":
                    record_step(Step(action, observation, call))
                    result.status = "infrastructure_error"
                    result.reason = observation.stderr
                    return result
                feedback = protocol.render_observation(
                    observation, condition.protocol.observation_template
                )
                record_step(Step(action, observation, call, feedback))
            # Feedback is appended only after its action has actually run.
            context.extend(
                model.encoding.continuation(call.generation.tokens, feedback)
            )
            save()
    except OSError as error:
        result.status = "infrastructure_error"
        result.reason = str(error)
        return result
    except KeyboardInterrupt:
        result.status = "interrupted"
        result.reason = "Episode interrupted"
        return result
    finally:
        if started is not None:
            result.usage.elapsed_seconds = time.monotonic() - started
        save()
