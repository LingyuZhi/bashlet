"""Task preparation, task-independent interaction, submission grading and analysis."""

import hashlib
import json
import keyword
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, fields
from pathlib import Path

from .analysis import analyze_episode
from .config import ExperimentConfig, load_config
from .episode import run_episode, save_record
from .models import (
    CaseResult,
    CommandGrader,
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
    if not task.task_id:
        raise ValueError("A task identity is required")
    if task.environment_id != environment_id:
        raise ValueError(
            f"Task environment {task.environment_id!r} "
            f"does not match {environment_id!r}"
        )
    grader = task.grader
    if isinstance(grader, FunctionGrader):
        if not grader.cases:
            raise ValueError("At least one grading case is required")
        validate_workspace_path(grader.module_path)
        if not grader.module_path.endswith(".py"):
            raise ValueError("The grading module must be a Python file")
        function = grader.function_name
        if not function.isidentifier() or keyword.iskeyword(function):
            raise ValueError("The task function must be a Python identifier")
    else:
        if not grader.command.strip() or "\x00" in grader.command:
            raise ValueError("The grading command must be non-empty and contain no NUL")
        _validate_files(grader.hidden_files)
    _validate_files(task.initial_files)
    for path in task.implementation_paths:
        validate_workspace_path(path)


def _validate_files(files: dict[str, str]) -> None:
    for path, content in files.items():
        validate_workspace_path(path)
        if not isinstance(content, str):
            raise ValueError(f"Task file {path!r} must contain text")
        parents = Path(path).parents
        if any(str(parent) in files for parent in parents):
            raise ValueError(f"Conflicting task file path {path!r}")


def _grade(
    grader: FunctionGrader | CommandGrader, sandbox: Sandbox, run_dir: Path
) -> Grade:
    workspace = run_dir / "private" / "grading-workspace"
    shutil.copytree(run_dir / "submission", workspace)
    if isinstance(grader, CommandGrader):
        try:
            for path, content in grader.hidden_files.items():
                destination = workspace / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.unlink(missing_ok=True)
                destination.write_text(content)
        except (FileExistsError, IsADirectoryError, NotADirectoryError) as error:
            return Grade(
                False, 0, 1, (CaseResult(False, "grading_path_conflict", str(error)),)
            )
        observation = sandbox.execute(
            workspace,
            ["/bin/bash", "--noprofile", "--norc", "-c", grader.command],
            readonly=True,
        )
        if observation.status == "interrupted":
            raise KeyboardInterrupt
        passed = observation.status == "ok"
        command_result = CaseResult(
            passed,
            "passed" if passed else observation.status,
            observation.stderr if observation.status == "infrastructure_error" else "",
        )
        return Grade(passed, int(passed), 1, (command_result,))
    results: list[CaseResult] = []
    for case in grader.cases:
        observation = sandbox.execute(
            workspace,
            [sandbox.python, "-I", "-B", "-c", _GRADING_DRIVER],
            stdin=json.dumps(
                {
                    "module": grader.module_path,
                    "function": grader.function_name,
                    "args": case.args,
                }
            ).encode(),
            readonly=True,
        )
        if observation.status == "interrupted":
            raise KeyboardInterrupt
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
    config: ExperimentConfig | Sequence[Path] = (),
    overrides: Sequence[str] = (),
    on_update: Callable[[EpisodeResult], None] | None = None,
) -> ExperimentResult:
    """Prepare a Task, run its Episode, and grade only an explicit Submission.

    Initial files and instruction are public; the grader and reference remain
    host-only. Grading uses a read-only snapshot copy in fresh sandboxes. Function
    results are compared on the host; test commands pass on exit code zero and
    share a process with untrusted student code. The returned record and run
    directory are operator-only and never become student observations.
    """
    if isinstance(config, ExperimentConfig):
        if overrides:
            raise ValueError("Apply overrides before passing a resolved config")
        condition = config
    else:
        condition = load_config(config, overrides)
    _validate_task(task, condition.environment.id)
    grading_protocol = (
        "test-command-v1"
        if isinstance(task.grader, CommandGrader)
        else "json-function-v1"
    )
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
        "grading_protocol": grading_protocol,
        "candidates": [],
        "repair_attempts": [],
        "submission": None,
        "grade": None,
        "passed": False,
    }

    def save_progress(episode: EpisodeResult) -> None:
        save_record(episode, condition, metadata | {"episode_status": episode.status})
        if on_update is not None:
            on_update(episode)

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
    except KeyboardInterrupt:
        result.status, result.reason = "interrupted", "Grading interrupted"
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
                "grading_protocol": grading_protocol,
                "passed": result.passed,
            },
        )
        if on_update is not None:
            on_update(result)
