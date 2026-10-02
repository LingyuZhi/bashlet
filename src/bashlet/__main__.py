"""Watch one isolated agent episode or experiment, or replay a saved run."""

import argparse
import json
import shutil
import signal
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from types import FrameType

from rich.console import Console

from .config import load_config, load_model
from .episode import run_episode, save_record
from .experiment import run_experiment
from .mbpp import mbpp_task_path, prepare_mbpp
from .models import (
    CommandGrader,
    EpisodeResult,
    ExperimentResult,
    FunctionGrader,
    GradingCase,
    Isolation,
    ModelBackend,
    Task,
    TaskOrigin,
)
from .sandbox import Sandbox
from .spend import DEFAULT_POLICY, BudgetRefused, SpendLedger, SpendPolicy
from .view import RunView
from .workspace import capture_workspace


def _interrupt(signum: int, frame: FrameType | None) -> None:
    # The first Ctrl-C cancels work; allow cleanup and saving to finish.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    raise KeyboardInterrupt


def _execution_options(parser: argparse.ArgumentParser) -> None:
    _view_options(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument(
        "-c",
        "--config",
        action="append",
        type=Path,
        default=[],
        help="YAML layered after the built-in default; later files win",
    )
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=[],
        metavar="key=value",
        help="Override a config value after all --config files",
    )


def _view_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="Display recorded thinking, which is collapsed by default",
    )


def _run_agent(options: argparse.Namespace, view: RunView) -> EpisodeResult:
    condition = load_config(options.config, options.overrides)
    model = load_model(condition.model)
    source = options.workdir.resolve()
    if not source.is_dir():
        raise ValueError(f"Work directory {source} must be a directory")
    output_dir = options.output_dir.resolve()
    run_dir = output_dir / uuid.uuid4().hex
    run_dir.mkdir(parents=True, mode=0o700)
    # A run under its source must never recursively copy itself.
    excluded = output_dir if output_dir != source else run_dir

    def ignore(directory: str, names: list[str]) -> set[str]:
        return {name for name in names if Path(directory) / name == excluded}

    workspace = run_dir / "workspace"

    def progress(result: EpisodeResult) -> None:
        save_record(result, condition, {"instruction": options.instruction})
        view.update(asdict(result))

    try:
        shutil.copytree(source, workspace, symlinks=True, ignore=ignore)
    except (KeyboardInterrupt, OSError) as error:
        workspace.mkdir(exist_ok=True)
        result = EpisodeResult(
            run_dir=run_dir,
            workspace=workspace,
            limits=Sandbox(condition.environment).limits,
            isolation=Isolation(False, reason="Workspace preparation did not finish"),
            policy=model.identity,
            policy_details=model.describe() if isinstance(model, ModelBackend) else {},
            encoding=model.encoding.identity,
            encoding_details=model.encoding.describe(),
            budget=condition.budget,
            status="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "infrastructure_error",
            reason="Workspace copy interrupted"
            if isinstance(error, KeyboardInterrupt)
            else str(error),
            initial_workspace_manifest=capture_workspace(workspace, run_dir),
        )
        progress(result)
        return result

    return run_episode(
        instruction=options.instruction,
        workspace=workspace,
        model=model,
        config=condition,
        run_dir=run_dir,
        on_update=progress,
    )


def _run_task(options: argparse.Namespace, view: RunView) -> ExperimentResult:
    task_path = options.task
    if options.task_id is not None:
        if options.split is None:
            raise ValueError("Task-set selection requires --split")
        task_path = mbpp_task_path(task_path, options.task_id, options.split)
    elif options.split is not None:
        raise ValueError("--split requires a task set and --task-id")
    data = json.loads(task_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Task file must contain a JSON object")
    condition = load_config(options.config, options.overrides)
    if "task" in data:
        unknown = set(data) - {"task", "replies"}
        if unknown:
            raise ValueError(f"Unknown request fields: {', '.join(sorted(unknown))}")
        if "replies" in data:
            replies = data["replies"]
            if not isinstance(replies, list) or not all(
                isinstance(reply, str) for reply in replies
            ):
                raise ValueError("Replay replies must be a list of strings")
            if "replies" in condition.model.parameters:
                raise ValueError(
                    "Supply replies in either the request or model.replies"
                )
            condition = replace(
                condition,
                model=replace(
                    condition.model,
                    parameters=condition.model.parameters | {"replies": replies},
                ),
            )
        data = data["task"]
    if not isinstance(data, dict):
        raise ValueError("task must be a JSON object")
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
    task = Task(**task_data, grader=grader)
    model = load_model(condition.model)
    return run_experiment(
        task=task,
        model=model,
        output_dir=options.output_dir,
        config=condition,
        on_update=lambda result: view.update(asdict(result)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch an isolated bashlet agent")
    commands = parser.add_subparsers(dest="command", required=True)
    agent = commands.add_parser(
        "agent", help="Run the Harness on a copy of a directory"
    )
    agent.add_argument("instruction")
    agent.add_argument("--workdir", type=Path, default=Path.cwd())
    _execution_options(agent)
    task = commands.add_parser(
        "task", help="Run one trusted task and display its grade"
    )
    task.add_argument(
        "task", type=Path, help="Task JSON or a legacy task/replies request"
    )
    task.add_argument("--task-id", help="Select a task from a prepared MBPP manifest")
    task.add_argument("--split", choices=("train", "dev", "test"))
    _execution_options(task)
    prepare = commands.add_parser(
        "prepare-mbpp", help="Import pinned MBPP and validate references in isolation"
    )
    prepare.add_argument("--source-lock", type=Path, required=True)
    prepare.add_argument("--source-file", type=Path)
    _execution_options(prepare)
    show = commands.add_parser("show", help="Replay a saved run without executing it")
    show.add_argument("run_dir", type=Path)
    _view_options(show)
    spend = commands.add_parser(
        "spend", help="Inspect or reconcile the shared cash ledger"
    )
    spend_commands = spend.add_subparsers(dest="spend_command", required=True)
    status = spend_commands.add_parser(
        "status", help="Show charges and pending ceilings"
    )
    reconcile = spend_commands.add_parser(
        "reconcile", help="Record verified completion and billing"
    )
    for command in (status, reconcile):
        command.add_argument("--ledger", type=Path, default=Path("runs/spend.sqlite3"))
        command.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    reconcile.add_argument("request_id")
    reconcile.add_argument("--actual-cny", required=True)
    reconcile.add_argument("--outcome", choices=("completed", "failed"), required=True)
    reconcile.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="JSON completion and billing evidence",
    )
    options = parser.parse_args()
    view = RunView(Console(), show_thinking=getattr(options, "show_thinking", False))
    previous_handler = signal.signal(signal.SIGINT, _interrupt)
    try:
        if options.command == "prepare-mbpp":
            manifest = prepare_mbpp(
                source_lock=options.source_lock,
                source_file=options.source_file,
                output_dir=options.output_dir,
                config=load_config(options.config, options.overrides),
            )
            print(manifest)
            return
        if options.command == "spend":
            ledger = SpendLedger(
                options.ledger, policy=SpendPolicy.read(options.policy)
            )
            if options.spend_command == "reconcile":
                evidence = json.loads(options.evidence.read_text())
                if not isinstance(evidence, dict):
                    raise ValueError("Billing evidence must be a JSON object")
                ledger.reconcile(
                    options.request_id,
                    actual_cny=options.actual_cny,
                    evidence=evidence,
                    outcome=options.outcome,
                )
            print(json.dumps(ledger.snapshot(), indent=2, ensure_ascii=True))
            return
        if options.command == "show":
            record = json.loads((options.run_dir / "run.json").read_text())
            if not isinstance(record, dict):
                raise ValueError("run.json must contain a JSON object")
            view.finish(record)
            return
        result = (
            _run_agent(options, view)
            if options.command == "agent"
            else _run_task(options, view)
        )
    except (OSError, ValueError, TypeError, KeyError, BudgetRefused) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    view.finish(asdict(result))
    if result.status == "interrupted":
        raise SystemExit(130)
    if result.status == "infrastructure_error":
        raise SystemExit(2)
    if isinstance(result, ExperimentResult):
        if not result.passed:
            raise SystemExit(1)
    elif result.status != "submitted":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
