"""Task preparation, task-independent interaction, submission grading and analysis."""

import hashlib
import json
import keyword
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from .analysis import analyze_episode
from .config import DirectCodeConfig, ExperimentConfig, load_config
from .demonstrations import (
    ActionSupervision,
    SFTAdapter,
    export_demonstration,
    validate_sft_request,
)
from .episode import run_episode, save_record
from .models import (
    CaseResult,
    CommandGrader,
    EpisodeResult,
    ExperimentResult,
    FirstImplementation,
    FunctionGrader,
    Grade,
    GradingCase,
    Model,
    Task,
    TaskOrigin,
)
from .sandbox import Sandbox
from .tasks import validate_origin
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


def task_from_record(data: dict[str, Any]) -> Task:
    """Decode the trusted Task format shared by CLI and evaluation plans."""
    task_data = dict(data)
    grader_data = dict(task_data.pop("grader"))
    grader: FunctionGrader | CommandGrader
    if "command" in grader_data:
        grader = CommandGrader(**grader_data)
    else:
        cases = tuple(
            GradingCase(args=tuple(case["args"]), expected=case["expected"])
            for case in grader_data.pop("cases")
        )
        grader = FunctionGrader(**grader_data, cases=cases)
    if "implementation_paths" in task_data:
        task_data["implementation_paths"] = tuple(task_data["implementation_paths"])
    if task_data.get("origin") is not None:
        task_data["origin"] = TaskOrigin(**task_data["origin"])
        validate_origin(task_data["origin"])
    return Task(**task_data, grader=grader)


def serialize_task(task: Task) -> str:
    """Canonical private Task document used for both storage and identity hashes."""
    return json.dumps(asdict(task), ensure_ascii=True, allow_nan=False, indent=2)


def _validate_task(task: Task, environment_id: str) -> None:
    if not task.task_id:
        raise ValueError("A task identity is required")
    if task.origin is not None:
        validate_origin(task.origin)
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
    grader: FunctionGrader | CommandGrader,
    sandbox: Sandbox,
    run_dir: Path,
    *,
    submission: str = "submission",
    destination: str = "private/grading-workspace",
) -> Grade:
    workspace = run_dir / destination
    shutil.copytree(run_dir / submission, workspace)
    if isinstance(grader, CommandGrader):
        try:
            for path, content in grader.hidden_files.items():
                file_path = workspace / path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.unlink(missing_ok=True)
                file_path.write_text(content)
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


def _assess_first(
    result: ExperimentResult, task: Task, condition: ExperimentConfig
) -> FirstImplementation:
    assessment = FirstImplementation()
    if not result.candidates:
        return assessment
    candidate = result.candidates[0]
    assessment.candidate_index = candidate.index
    assessment.formed_at_step = candidate.formed_at_step
    manifest = result.trajectory[candidate.formed_at_step].workspace_manifest
    try:
        assessment.snapshot = snapshot_workspace(
            manifest,
            result.run_dir,
            result.limits,
            destination="private/first-implementation",
        )
        assessment.grade = _grade(
            task.grader,
            Sandbox(condition.environment),
            result.run_dir,
            submission=assessment.snapshot.path,
            destination="private/first-grading-workspace",
        )
        assessment.status = (
            "infrastructure_error"
            if any(c.status == "infrastructure_error" for c in assessment.grade.cases)
            else "graded"
        )
    except ValueError as error:
        assessment.status, assessment.reason = "invalid_snapshot", str(error)
    except OSError as error:
        assessment.status, assessment.reason = "infrastructure_error", str(error)
    except KeyboardInterrupt:
        assessment.status, assessment.reason = (
            "interrupted",
            "First grading interrupted",
        )
    return assessment


def run_experiment(
    *,
    task: Task,
    model: Model,
    output_dir: Path = Path("runs"),
    config: ExperimentConfig | Sequence[Path] = (),
    overrides: Sequence[str] = (),
    on_update: Callable[[EpisodeResult], None] | None = None,
    assess_first_implementation: bool = False,
    direct_code: DirectCodeConfig | None = None,
    sft_adapter: SFTAdapter | None = None,
    sft_supervision: Sequence[ActionSupervision] | None = None,
    sft_max_context_tokens: int = 32768,
) -> ExperimentResult:
    """Prepare a Task, run its Episode, and grade only an explicit Submission.

    Initial files and instruction are public; the grader and reference remain
    host-only. Grading uses a read-only snapshot copy in fresh sandboxes. Function
    results are compared on the host; test commands pass on exit code zero and
    share a process with untrusted student code. The returned record and run
    directory are operator-only and never become student observations.

    ``sft_adapter`` opts into demonstration validation and export after
    grading. Rejected demonstrations retain their run evidence but never reach
    the Adapter. Export errors propagate after the final run record is saved.
    """
    if isinstance(config, ExperimentConfig):
        if overrides:
            raise ValueError("Apply overrides before passing a resolved config")
        condition = config
    else:
        condition = load_config(config, overrides)
    _validate_task(task, condition.environment.id)
    if direct_code is not None and (
        sft_adapter is not None or assess_first_implementation
    ):
        raise ValueError(
            "Direct coding is an independent evaluation, without SFT export"
        )
    if sft_supervision is not None and sft_adapter is None:
        raise ValueError("SFT supervision requires an Adapter")
    if sft_adapter is not None:
        validate_sft_request(
            task, model, sft_adapter, executed=sft_supervision is not None
        )
    grading_protocol = (
        "test-command-v1"
        if isinstance(task.grader, CommandGrader)
        else "json-function-v1"
    )
    task_json = serialize_task(task)
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
        "task_type": task.task_type,
        "variant_id": task.variant_id,
        "feedback_start_id": task.feedback_start_id,
        "environment_variant_id": task.environment_variant_id,
        "task_origin": asdict(task.origin) if task.origin else None,
        "task_sha256": hashlib.sha256(task_json.encode()).hexdigest(),
        "environment_id": task.environment_id,
        "grading_protocol": grading_protocol,
        "candidates": [],
        "repair_attempts": [],
        "submission": None,
        "grade": None,
        "passed": False,
    }
    if direct_code is not None:
        metadata.update(evaluation_track="coding", direct_code=asdict(direct_code))

    def save_progress(episode: EpisodeResult) -> None:
        save_record(episode, condition, metadata | {"episode_status": episode.status})
        if on_update is not None:
            on_update(episode)

    if direct_code is not None:
        from .evaluation.direct import run_direct_code

        episode = run_direct_code(
            task, model, workspace, run_dir, condition, direct_code, save_progress
        )
    else:
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
        if assess_first_implementation:
            result.first_implementation = _assess_first(result, task, condition)
        final_metadata: dict[str, object] = {
            "grading_protocol": grading_protocol,
            "task_origin": asdict(task.origin) if task.origin else None,
            "task_type": task.task_type,
            "variant_id": task.variant_id,
            "feedback_start_id": task.feedback_start_id,
            "environment_variant_id": task.environment_variant_id,
            "passed": result.passed,
        }
        if direct_code is not None:
            final_metadata.update(
                evaluation_track="coding", direct_code=asdict(direct_code)
            )
        try:
            if sft_adapter is not None:
                # Seal actual grading and history before any SFT consumer sees it.
                save_record(result, condition, final_metadata)
                export_demonstration(
                    result,
                    task,
                    model,
                    condition,
                    sft_adapter,
                    supervision=sft_supervision,
                    max_context_tokens=sft_max_context_tokens,
                )
        finally:
            save_record(result, condition, final_metadata)
        if on_update is not None:
            on_update(result)
