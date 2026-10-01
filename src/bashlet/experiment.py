"""One authoritative entry point: task → actions → real observations → grade."""

import hashlib
import json
import keyword
import math
import os
import stat
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from . import protocol
from .analysis import analyze_episode
from .config import load_config
from .models import (
    CaseResult,
    ExperimentResult,
    Generation,
    Grade,
    InteractionBudget,
    InvalidAction,
    Isolation,
    Limits,
    Model,
    ModelCall,
    SamplingParams,
    Step,
    Submission,
    Submit,
    Task,
    Usage,
    WorkspaceEntry,
)
from .sandbox import Sandbox
from .workspace import capture_workspace

_GRADING_DRIVER = """
import contextlib, importlib.util, json, sys
request = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    spec = importlib.util.spec_from_file_location('solution', '/workspace/solution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answer = getattr(module, request['function'])(*request['args'])
print(json.dumps(answer, allow_nan=False))
"""

_PROBE_TIMEOUT_SECONDS = 5.0
_PROBE_OUTPUT_BYTES = 16_384


def _verify_isolation(sandbox: Sandbox, run_dir: Path) -> Isolation:
    probe_dir = run_dir / "probe"
    probe_dir.mkdir()
    canary = run_dir / "private" / "host-private.txt"
    canary.write_text("host-private-canary")
    paths = {
        "host_private": str(canary),
        "reference_and_grading_material": str(run_dir / "private" / "task.json"),
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


def _read_solution(workspace: Path, limit: int) -> bytes:
    descriptor = os.open(
        workspace / "solution.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("solution.py must be a regular file")
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError("solution.py exceeds the submission size limit")
    return content


def _snapshot(workspace: Path, run_dir: Path, limit: int) -> Submission:
    content = _read_solution(workspace, limit)
    snapshot_dir = run_dir / "submission"
    snapshot_dir.mkdir()
    target = snapshot_dir / "solution.py"
    target.write_bytes(content)
    target.chmod(0o400)
    return Submission(
        str(target.relative_to(run_dir)),
        hashlib.sha256(content).hexdigest(),
        len(content),
    )


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


def _grade(task: Task, sandbox: Sandbox, run_dir: Path) -> Grade:
    results: list[CaseResult] = []
    for case in task.grading_cases:
        observation = sandbox.execute(
            run_dir / "submission",
            ["/usr/bin/python3", "-I", "-c", _GRADING_DRIVER],
            stdin=json.dumps(
                {"function": task.function_name, "args": case.args}
            ).encode(),
            readonly=True,
        )
        if observation.status == "infrastructure_error":
            results.append(CaseResult(False, observation.status, observation.stderr))
            break
        if observation.status != "ok":
            results.append(CaseResult(False, observation.status))
        else:
            try:
                passed = json.loads(observation.stdout) == case.expected
                results.append(
                    CaseResult(passed, "passed" if passed else "wrong_answer")
                )
            except (ValueError, RecursionError):
                results.append(CaseResult(False, "invalid_output"))
    passed_cases = sum(result.passed for result in results)
    return Grade(
        passed_cases == len(task.grading_cases),
        passed_cases,
        len(task.grading_cases),
        tuple(results),
    )


def run_experiment(
    *,
    task: Task,
    model: Model,
    output_dir: Path = Path("runs"),
    config: Sequence[Path] = (),
    overrides: Sequence[str] = (),
) -> ExperimentResult:
    """Run one Episode from a sampling model or pre-written replay replies.

    Protocol text, the execution environment, the interaction budget and sampling
    parameters come from the versioned config. The caller still supplies the
    model adapter. A model only ever receives the rendered system prompt, task
    prompt and real feedback from earlier steps.

    The returned record is operator-only. It is not an agent observation.
    Grading v1 calls a named function with JSON positional arguments and compares
    its JSON result in the host; neither reference code nor expected values enter
    any student process. A submission always ends the episode.
    """
    condition = load_config(config, overrides)
    limits = Limits(
        timeout_seconds=condition.environment.timeout_seconds,
        output_bytes=condition.environment.output_bytes,
        submission_bytes=condition.environment.submission_bytes,
        memory_bytes=condition.environment.memory_bytes,
        file_bytes=condition.environment.file_bytes,
    )
    budget = condition.budget
    sampling = condition.model.sampling
    if not task.task_id or not task.grading_cases:
        raise ValueError("A task identity and at least one grading case are required")
    if not task.function_name.isidentifier() or keyword.iskeyword(task.function_name):
        raise ValueError("The task function must be a Python identifier")
    task_json = json.dumps(asdict(task), ensure_ascii=True, allow_nan=False, indent=2)
    run_dir = output_dir.resolve() / uuid.uuid4().hex
    run_dir.mkdir(parents=True, mode=0o700)
    private_dir = run_dir / "private"
    private_dir.mkdir(mode=0o700)
    (private_dir / "task.json").write_text(task_json)
    result = ExperimentResult(
        run_dir,
        task.task_id,
        hashlib.sha256(task_json.encode()).hexdigest(),
        limits,
        Isolation(False, reason="Not yet checked"),
        policy=model.identity,
        encoding=model.encoding.identity,
        budget=budget,
    )
    sandbox = Sandbox(condition.environment)

    def save() -> None:
        record = asdict(result) | {
            "schema_version": 5,
            "config": asdict(condition),
            "fingerprints": protocol.fingerprints(condition),
            "protocol": protocol.describe(condition.protocol),
            "sampling": asdict(sampling),
            "grading_protocol": "json-function-v1",
            "isolation_policy": "linux-bwrap-v1",
            "effective_resource_limits": sandbox.resource_limits,
            "isolation_probe_limits": {
                "timeout_seconds": _PROBE_TIMEOUT_SECONDS,
                "output_bytes": _PROBE_OUTPUT_BYTES,
            },
            "host_python": sys.version,
            "bwrap": condition.environment.bwrap,
        }
        pending = run_dir / "run.json.tmp"
        # Linux paths can contain surrogate-escaped bytes; JSON must escape them.
        pending.write_text(json.dumps(record, default=str, ensure_ascii=True, indent=2))
        pending.replace(run_dir / "run.json")

    save()
    started: float | None = None
    feedback_materials: dict[str, WorkspaceEntry] = {}
    try:
        result.isolation = _verify_isolation(sandbox, run_dir)
        if not result.isolation.passed:
            result.status = "infrastructure_error"
            result.reason = result.isolation.reason
            return result
        # The Interaction Budget clock covers the episode, not the probe.
        started = time.monotonic()
        workspace = run_dir / "workspace"
        workspace.mkdir()
        (workspace / "task.md").write_text(task.prompt)
        (workspace / "feedback_tests.py").write_text(task.feedback_tests)
        result.initial_workspace_manifest = capture_workspace(workspace, run_dir)
        feedback_materials = {
            "feedback_tests.py": result.initial_workspace_manifest["feedback_tests.py"]
        }

        def record_step(step: Step) -> None:
            result.trajectory.append(
                replace(step, workspace_manifest=capture_workspace(workspace, run_dir))
            )

        context = result.context_tokens
        context.extend(
            model.encoding.prompt(condition.protocol.system_prompt, task.prompt)
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
                try:
                    result.submission = _snapshot(
                        workspace, run_dir, limits.submission_bytes
                    )
                except (OSError, ValueError) as error:
                    result.status = "invalid_submission"
                    result.reason = str(error)
                    return result
                result.grade = _grade(task, sandbox, run_dir)
                result.reason = next(
                    (case.reason for case in result.grade.cases if case.reason), ""
                )
                result.status = (
                    "infrastructure_error"
                    if any(
                        case.status == "infrastructure_error"
                        for case in result.grade.cases
                    )
                    else "completed"
                )
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
    finally:
        if started is not None:
            result.usage.elapsed_seconds = time.monotonic() - started
        analyze_episode(result, task, feedback_materials)
        save()
