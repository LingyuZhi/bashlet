"""One authoritative entry point: task → real observations → snapshot → grade."""

import hashlib
import json
import keyword
import math
import os
import stat
import sys
import uuid
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

from .models import (
    Bash,
    CaseResult,
    ExperimentResult,
    Grade,
    Isolation,
    Limits,
    Step,
    Submission,
    Submit,
    Task,
)
from .sandbox import Sandbox

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

_DEFAULT_LIMITS = Limits()
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
            sandbox.limits,
            timeout_seconds=_PROBE_TIMEOUT_SECONDS,
            output_bytes=_PROBE_OUTPUT_BYTES,
        ),
        sandbox.bwrap,
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


def _snapshot(workspace: Path, run_dir: Path, limit: int) -> Submission:
    descriptor = os.open(
        workspace / "solution.py", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    )
    with os.fdopen(descriptor, "rb") as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError("solution.py must be a regular file")
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError("solution.py exceeds the submission size limit")
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
    actions: Sequence[Bash | Submit],
    output_dir: Path = Path("runs"),
    limits: Limits = _DEFAULT_LIMITS,
    bwrap: str = "/usr/bin/bwrap",
) -> ExperimentResult:
    """Run trusted scripted actions; only Bash commands enter the task sandbox.

    The returned record is operator-only. It is not an agent observation.
    Grading v1 calls a named function with JSON positional arguments and compares
    its JSON result in the host; neither reference code nor expected values enter
    any student process. A submission always ends the episode.
    """
    if not task.task_id or not task.grading_cases:
        raise ValueError("A task identity and at least one grading case are required")
    if not task.function_name.isidentifier() or keyword.iskeyword(task.function_name):
        raise ValueError("The task function must be a Python identifier")
    if any(not math.isfinite(value) or value <= 0 for value in asdict(limits).values()):
        raise ValueError("Execution limits must be finite and positive")
    task_json = json.dumps(asdict(task), ensure_ascii=False, allow_nan=False, indent=2)
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
    )
    sandbox = Sandbox(limits, bwrap)

    def save() -> None:
        record = asdict(result) | {
            "schema_version": 1,
            "mode": "scripted",
            "grading_protocol": "json-function-v1",
            "isolation_policy": "linux-bwrap-v1",
            "effective_resource_limits": sandbox.resource_limits,
            "isolation_probe_limits": {
                "timeout_seconds": _PROBE_TIMEOUT_SECONDS,
                "output_bytes": _PROBE_OUTPUT_BYTES,
            },
            "host_python": sys.version,
            "bwrap": bwrap,
            "action_plan": [
                {"kind": "bash", "command": action.command}
                if isinstance(action, Bash)
                else {"kind": "submit"}
                for action in actions
            ],
        }
        pending = run_dir / "run.json.tmp"
        pending.write_text(
            json.dumps(record, default=str, ensure_ascii=False, indent=2)
        )
        pending.replace(run_dir / "run.json")

    save()
    try:
        result.isolation = _verify_isolation(sandbox, run_dir)
        if not result.isolation.passed:
            result.status = "infrastructure_error"
            result.reason = result.isolation.reason
            return result
        workspace = run_dir / "workspace"
        workspace.mkdir()
        (workspace / "task.md").write_text(task.prompt)
        (workspace / "feedback_tests.py").write_text(task.feedback_tests)
        for action in actions:
            if isinstance(action, Bash):
                observation = sandbox.execute(
                    workspace,
                    ["/bin/bash", "--noprofile", "--norc", "-c", action.command],
                )
                result.trajectory.append(Step(action, observation))
                save()
                if observation.status == "infrastructure_error":
                    result.status = "infrastructure_error"
                    result.reason = observation.stderr
                    return result
            else:
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
        result.status = "not_submitted"
        return result
    except OSError as error:
        result.status = "infrastructure_error"
        result.reason = str(error)
        return result
    finally:
        save()
