"""Task preparation, task-independent interaction, submission grading and analysis."""

import hashlib
import json
import keyword
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

from .analysis import analyze_episode
from .config import load_config
from .episode import run_episode, save_record
from .models import (
    CaseResult,
    EpisodeResult,
    ExperimentResult,
    FunctionGrader,
    Grade,
    Model,
    Task,
)
from .sandbox import Sandbox
from .workspace import snapshot_workspace, validate_workspace_path

_GRADING_DRIVER = """
import contextlib, importlib.util, json, sys
request = json.load(sys.stdin)
sys.path.insert(0, '/workspace')
with contextlib.redirect_stdout(sys.stderr):
    name = request['module'][:-3].replace('/', '.')
    source = '/workspace/' + request['module']
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    answer = getattr(module, request['function'])(*request['args'])
print(json.dumps(answer, allow_nan=False))
"""


def _validate_task(task: Task, environment_id: str) -> None:
    if not task.task_id or not task.grader.cases:
        raise ValueError("A task identity and at least one grading case are required")
    if task.environment_id != environment_id:
        raise ValueError(
            f"Task environment {task.environment_id!r} "
            f"does not match {environment_id!r}"
        )
    validate_workspace_path(task.grader.module_path)
    if not task.grader.module_path.endswith(".py"):
        raise ValueError("The grading module must be a Python file")
    function = task.grader.function_name
    if not function.isidentifier() or keyword.iskeyword(function):
        raise ValueError("The task function must be a Python identifier")
    for path, content in task.initial_files.items():
        validate_workspace_path(path)
        if not isinstance(content, str):
            raise ValueError(f"Initial file {path!r} must contain text")
        parents = Path(path).parents
        if any(str(parent) in task.initial_files for parent in parents):
            raise ValueError(f"Conflicting initial file path {path!r}")
    for path in task.implementation_paths:
        validate_workspace_path(path)


def _grade(grader: FunctionGrader, sandbox: Sandbox, run_dir: Path) -> Grade:
    workspace = run_dir / "private" / "grading-workspace"
    shutil.copytree(run_dir / "submission", workspace)
    results: list[CaseResult] = []
    for case in grader.cases:
        observation = sandbox.execute(
            workspace,
            ["/usr/bin/python3", "-I", "-B", "-c", _GRADING_DRIVER],
            stdin=json.dumps(
                {
                    "module": grader.module_path,
                    "function": grader.function_name,
                    "args": case.args,
                }
            ).encode(),
            readonly=True,
        )
        if observation.status == "infrastructure_error":
            results.append(CaseResult(False, observation.status, observation.stderr))
            break
        if observation.status != "ok":
            results.append(CaseResult(False, observation.status))
        elif observation.output_truncated:
            results.append(CaseResult(False, "output_limit"))
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
        passed_cases == len(grader.cases),
        passed_cases,
        len(grader.cases),
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
    """Prepare a Task, run its Episode, and grade only an explicit Submission.

    Initial files and instruction are public; the grader and reference remain
    host-only. Grading executes a read-only snapshot copy in a fresh sandbox per
    case and compares its JSON result on the host. The returned record and run
    directory are operator-only and never become student observations.
    """
    condition = load_config(config, overrides)
    _validate_task(task, condition.environment.id)
    task_json = json.dumps(asdict(task), ensure_ascii=True, allow_nan=False, indent=2)
    run_dir = output_dir.resolve() / uuid.uuid4().hex
    run_dir.mkdir(parents=True, mode=0o700)
    private = run_dir / "private"
    private.mkdir(mode=0o700)
    (private / "task.json").write_text(task_json)
    workspace = run_dir / "workspace"
    workspace.mkdir()
    for path, content in task.initial_files.items():
        target = workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    metadata: dict[str, object] = {
        "task_id": task.task_id,
        "task_sha256": hashlib.sha256(task_json.encode()).hexdigest(),
        "environment_id": task.environment_id,
        "grading_protocol": "json-function-v1",
        "candidates": [],
        "repair_attempts": [],
        "submission": None,
        "grade": None,
        "passed": False,
    }

    def save_progress(episode: EpisodeResult) -> None:
        save_record(episode, condition, metadata | {"episode_status": episode.status})

    episode = run_episode(
        instruction=task.prompt,
        workspace=workspace,
        model=model,
        config=condition,
        run_dir=run_dir,
        on_update=save_progress,
    )
    result = ExperimentResult(
        **{field.name: getattr(episode, field.name) for field in fields(EpisodeResult)},
        task_id=task.task_id,
        task_sha256=str(metadata["task_sha256"]),
        environment_id=task.environment_id,
        episode_status=episode.status,
    )
    try:
        if episode.status != "submitted":
            return result
        manifest = episode.trajectory[-1].workspace_manifest
        try:
            result.submission = snapshot_workspace(manifest, run_dir, result.limits)
        except (OSError, ValueError) as error:
            result.status, result.reason = "invalid_submission", str(error)
            return result
        result.grade = _grade(task.grader, Sandbox(condition.environment), run_dir)
        result.reason = next(
            (case.reason for case in result.grade.cases if case.reason), ""
        )
        result.status = (
            "infrastructure_error"
            if any(case.status == "infrastructure_error" for case in result.grade.cases)
            else "completed"
        )
        return result
    except OSError as error:
        result.status, result.reason = "infrastructure_error", str(error)
        return result
    finally:
        feedback_materials = {
            path: result.initial_workspace_manifest[path]
            for path in task.initial_files
            if path not in task.implementation_paths
            and path in result.initial_workspace_manifest
        }
        analyze_episode(result, task, feedback_materials)
        save_record(
            result,
            condition,
            {
                "grading_protocol": "json-function-v1",
                "passed": result.passed,
            },
        )
