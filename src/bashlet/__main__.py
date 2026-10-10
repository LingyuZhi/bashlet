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
from .demonstrations import prepare_sft
from .episode import run_episode, save_record
from .evaluation import evaluate
from .experiment import run_experiment, task_from_record
from .mbpp import prepare_mbpp
from .models import (
    EpisodeResult,
    ExperimentResult,
    Isolation,
    ModelBackend,
)
from .sandbox import Sandbox
from .spend import DEFAULT_POLICY, BudgetRefused, SpendLedger, SpendPolicy
from .tasks import task_path as select_task_path
from .tasks.preparation import DEFAULT_COHORT, prepare_harness
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
        task_path = select_task_path(task_path, options.task_id, options.split)
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
    task = task_from_record(data)
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
    task.add_argument("--task-id", help="Select a task from a prepared task manifest")
    task.add_argument("--split", choices=("train", "train_candidate", "dev", "test"))
    _execution_options(task)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate a fixed development task list"
    )
    evaluation.add_argument("plan", type=Path)
    evaluation.add_argument("--snapshot", required=True, help="Expected model identity")
    evaluation.add_argument("--monitor-config", type=Path)
    evaluation.add_argument("--training-run", type=Path)
    evaluation.add_argument(
        "--evidence-kind",
        choices=("experimental", "test-double"),
        default="experimental",
    )
    _execution_options(evaluation)
    acceptance = commands.add_parser(
        "evaluate-acceptance",
        help="Run or extend the frozen Harness and direct coding checks",
    )
    acceptance.add_argument("policy", type=Path)
    acceptance.add_argument("--snapshot", required=True)
    acceptance.add_argument(
        "--stage", choices=("screening", "confirmation"), default="screening"
    )
    acceptance.add_argument("--resume", type=Path)
    acceptance.add_argument(
        "--evidence-kind",
        choices=("experimental", "test-double"),
        default="experimental",
    )
    acceptance.add_argument("--monitor-config", type=Path)
    acceptance.add_argument("--training-run", type=Path)
    _execution_options(acceptance)
    prepare = commands.add_parser(
        "prepare-mbpp", help="Import pinned MBPP and validate references in isolation"
    )
    prepare.add_argument("--source-lock", type=Path, required=True)
    prepare.add_argument("--source-file", type=Path)
    _execution_options(prepare)
    harness_tasks = commands.add_parser(
        "prepare-harness", help="Freeze mixed-source tasks with real grading validation"
    )
    harness_tasks.add_argument("--mbpp-manifest", type=Path)
    harness_tasks.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    harness_tasks.add_argument("--workers", type=int, default=1)
    harness_tasks.add_argument("--collect-quality", action="store_true")
    _execution_options(harness_tasks)
    public_tasks = commands.add_parser(
        "import-public-tasks",
        help="Convert pinned public Python tasks to a private catalog",
    )
    public_tasks.add_argument("plan", type=Path)
    public_tasks.add_argument("--source-dir", type=Path)
    public_tasks.add_argument("--output-dir", type=Path, required=True)
    demonstrations = commands.add_parser(
        "prepare-sft",
        help="Export reviewed trajectories or executed training references",
    )
    demonstrations.add_argument("manifest", type=Path)
    demonstrations.add_argument(
        "--trajectories",
        type=Path,
        help="Reviewed recorded runs and optional Codex corrections",
    )
    demonstrations.add_argument("--max-context-tokens", type=int, default=32768)
    demonstrations.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Default: all verified training tasks",
    )
    _execution_options(demonstrations)
    collection = commands.add_parser(
        "collect-demonstrations", help="Collect bounded, explicitly sourced attempts"
    )
    collection.add_argument("plan", type=Path)
    _execution_options(collection)
    pilot_report = commands.add_parser(
        "report-collection-pilot", help="Report reviewed paired v3/v4 evidence"
    )
    pilot_report.add_argument("plan", type=Path)
    pilot_report.add_argument("--collection", action="append", type=Path, required=True)
    pilot_report.add_argument("--review", action="append", type=Path, required=True)
    pilot_report.add_argument("--output-dir", type=Path, required=True)
    freeze = commands.add_parser(
        "freeze-demonstrations", help="Replay, load and audit reviewed training data"
    )
    freeze.add_argument("collection", type=Path)
    freeze.add_argument("--review", type=Path, required=True)
    _execution_options(freeze)
    training = commands.add_parser(
        "train-sft", help="Train and checkpoint verified SFT demonstrations"
    )
    training.add_argument("training_config", type=Path)
    training.add_argument("--output-dir", type=Path, required=True)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--stop-after", type=int)
    opsd = commands.add_parser("train-opsd", help="Run a bounded, paired OPSD pilot")
    opsd.add_argument("pilot_config", type=Path)
    opsd.add_argument("--output-dir", type=Path, required=True)
    opsd.add_argument("--resume", action="store_true")
    opsd.add_argument("--preflight-config", type=Path)
    verify_opsd = commands.add_parser(
        "verify-opsd", help="Verify saved OPSD state and probabilities"
    )
    verify_opsd.add_argument("pilot_config", type=Path)
    verify_opsd.add_argument("--output-dir", type=Path, required=True)
    verify_opsd.add_argument(
        "--retry", action="store_true", help="Retry a previously failed verification"
    )
    verify_opsd.add_argument(
        "--native", action="store_true", help="Run native template and token checks"
    )
    opsd_watch = commands.add_parser(
        "watch-opsd", help="Monitor and safely resume the frozen expanded OPSD run"
    )
    opsd_watch.add_argument(
        "--config", type=Path, default=Path("configs/opsd-expanded-monitor-v1.yaml")
    )
    opsd_watch.add_argument(
        "--once", action="store_true", help="Record one monitoring cycle and exit"
    )
    selection = commands.add_parser(
        "select-sft", help="Select a saved SFT snapshot using development results"
    )
    selection.add_argument("run_dir", type=Path)
    selection.add_argument("--baseline", type=Path, required=True)
    selection.add_argument("--evaluation", type=Path, action="append", required=True)
    selection.add_argument("--benefit-review", type=Path, action="append", default=[])
    selection.add_argument(
        "--evidence-kind",
        choices=("experimental", "test-double"),
        default="experimental",
    )
    selection.add_argument("--output-dir", type=Path)
    selection.add_argument("--monitor-config", type=Path)
    export = commands.add_parser(
        "export-snapshot", help="Download an existing PyTRIO sampler as a LoRA adapter"
    )
    export.add_argument("training_run", type=Path)
    export.add_argument("--step", type=int, required=True)
    export.add_argument("--output-dir", type=Path, required=True)
    export.add_argument("--api-key-env", default="PYTRIO_API_KEY")
    export.add_argument("--dotenv", type=Path, default=Path(".env"))
    local = commands.add_parser(
        "prepare-local-snapshot", help="Verify and identify local Qwen3.5 weights"
    )
    local.add_argument("base_path", type=Path)
    local.add_argument("--model-revision", required=True)
    local.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--export", type=Path, dest="export_path")
    local.add_argument(
        "--merge",
        action="store_true",
        help="Merge LoRA in FP32, then round to base precision",
    )
    monitoring = commands.add_parser(
        "monitor", help="Import or sync experiment metrics"
    )
    monitor_commands = monitoring.add_subparsers(dest="monitor_command", required=True)
    historical = monitor_commands.add_parser("import-evaluation")
    historical.add_argument("evaluation", type=Path)
    historical.add_argument("--monitor-config", type=Path, required=True)
    historical.add_argument("--output-dir", type=Path, required=True)
    sync = monitor_commands.add_parser("sync")
    sync.add_argument("run_dir", type=Path)
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
    total = spend_commands.add_parser(
        "reconcile-total", help="Confirm cumulative spend; retain unknown request fees"
    )
    authorization = spend_commands.add_parser(
        "authorize-policy", help="Apply evidenced cash limits without resetting spend"
    )
    authorization.add_argument("new_policy", type=Path)
    authorization.add_argument("--evidence", type=Path, required=True)
    authorization.add_argument(
        "--allow-project-increase",
        action="store_true",
        help="Allow a project-cap increase explicitly listed in the evidence",
    )
    for command in (status, reconcile, total, authorization):
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
    total.add_argument("--actual-cny", required=True)
    total.add_argument("--evidence", type=Path, required=True)
    total.add_argument(
        "--group-actual-cny", action="append", default=[], metavar="GROUP=CNY"
    )
    options = parser.parse_args()
    view = RunView(Console(), show_thinking=getattr(options, "show_thinking", False))
    previous_handler = signal.signal(signal.SIGINT, _interrupt)
    try:
        if options.command == "verify-opsd":
            from .distillation.pilot import verify_opsd_checkpoints

            path = verify_opsd_checkpoints(
                options.pilot_config,
                options.output_dir,
                retry=options.retry,
                native=options.native,
            )
            record = json.loads(path.read_text())
            print(f"{record['status']}\n{path}")
            raise SystemExit(0 if record["status"] == "complete" else 2)
        if options.command == "watch-opsd":
            from .distillation.watchdog import run_watch

            run_watch(options.config, once=options.once)
            return
        if options.command == "train-opsd":
            from .distillation.pilot import run_opsd_pilot

            if options.preflight_config:
                from .distillation.preflight import run_with_preflight

                path = run_with_preflight(
                    options.pilot_config,
                    options.preflight_config,
                    options.output_dir,
                    resume=options.resume,
                )
            else:
                path = run_opsd_pilot(
                    options.pilot_config, options.output_dir, resume=options.resume
                )
            record = json.loads(path.read_text())
            print(f"{record['status']}\n{path}")
            raise SystemExit(0 if record["status"] == "complete" else 2)
        if options.command == "report-collection-pilot":
            from .curation.pilot import report_collection_pilot

            print(
                report_collection_pilot(
                    options.plan,
                    options.collection,
                    options.review,
                    options.output_dir,
                )
            )
            return
        if options.command == "evaluate-acceptance":
            from .evaluation.suite import evaluate_acceptance
            from .monitoring import MonitorConfig

            path = evaluate_acceptance(
                options.policy,
                snapshot=options.snapshot,
                evidence_kind=options.evidence_kind,
                condition=load_config(options.config, options.overrides),
                output_dir=options.output_dir,
                stage=options.stage,
                resume=options.resume,
                monitoring=MonitorConfig.read(options.monitor_config)
                if options.monitor_config
                else None,
                training_run=options.training_run,
            )
            print(path)
            raise SystemExit(
                0 if json.loads(path.read_text())["status"] == "complete" else 2
            )
        if options.command == "freeze-demonstrations":
            from .curation import freeze_demonstrations

            print(
                freeze_demonstrations(
                    options.collection,
                    options.review,
                    options.output_dir,
                    load_config(options.config, options.overrides),
                )
            )
            return
        if options.command == "collect-demonstrations":
            from .curation import collect_demonstrations

            print(
                collect_demonstrations(
                    options.plan,
                    options.output_dir,
                    load_config(options.config, options.overrides),
                )
            )
            return
        if options.command == "prepare-local-snapshot":
            from .snapshots import prepare_local_snapshot

            print(
                prepare_local_snapshot(
                    options.base_path,
                    options.model_revision,
                    options.output_dir,
                    export_path=options.export_path,
                    merge=options.merge,
                )
            )
            return
        if options.command == "export-snapshot":
            from .snapshots import export_snapshot

            print(
                export_snapshot(
                    options.training_run,
                    options.step,
                    options.output_dir,
                    api_key_env=options.api_key_env,
                    dotenv_path=options.dotenv,
                )
            )
            return
        if options.command == "select-sft":
            from .evaluation import select_sft
            from .monitoring import MonitorConfig

            print(
                select_sft(
                    options.run_dir,
                    options.baseline,
                    options.evaluation,
                    evidence_kind=options.evidence_kind,
                    reviews=options.benefit_review,
                    output_dir=options.output_dir,
                    monitoring=MonitorConfig.read(options.monitor_config)
                    if options.monitor_config
                    else None,
                )
            )
            return
        if options.command == "train-sft":
            from .training import train_sft

            print(
                train_sft(
                    options.training_config,
                    options.output_dir,
                    resume=options.resume,
                    stop_after=options.stop_after,
                )
            )
            return
        if options.command == "monitor":
            from .monitoring import MonitorConfig, import_evaluation, sync_monitor

            if options.monitor_command == "sync":
                sync_monitor(options.run_dir)
            else:
                import_evaluation(
                    options.evaluation,
                    options.output_dir,
                    MonitorConfig.read(options.monitor_config),
                )
            return
        if options.command == "evaluate":
            from .monitoring import MonitorConfig

            path = evaluate(
                plan_path=options.plan,
                snapshot=options.snapshot,
                evidence_kind=options.evidence_kind,
                condition=load_config(options.config, options.overrides),
                output_dir=options.output_dir,
                monitoring=MonitorConfig.read(options.monitor_config)
                if options.monitor_config
                else None,
                training_run=options.training_run,
            )
            record = json.loads(path.read_text())
            summary = record["summary"]
            print(
                f"{record['status']}: {summary['passed_tasks']}/"
                f"{summary['planned_tasks']} passed; "
                f"missing={summary['missing_task_ids']}\n{path}"
            )
            raise SystemExit(2 if record["status"] != "complete" else 0)
        if options.command == "prepare-harness":
            manifest = prepare_harness(
                mbpp_manifest=options.mbpp_manifest,
                cohort_path=options.cohort,
                workers=options.workers,
                collect_quality=options.collect_quality,
                output_dir=options.output_dir,
                config=load_config(options.config, options.overrides),
            )
            print(manifest)
            return
        if options.command == "import-public-tasks":
            from .tasks.public import prepare_public_catalog

            print(
                prepare_public_catalog(
                    plan_path=options.plan,
                    source_dir=options.source_dir,
                    output_dir=options.output_dir,
                )
            )
            return
        if options.command == "prepare-mbpp":
            manifest = prepare_mbpp(
                source_lock=options.source_lock,
                source_file=options.source_file,
                output_dir=options.output_dir,
                config=load_config(options.config, options.overrides),
            )
            print(manifest)
            return
        if options.command == "prepare-sft":
            manifest = prepare_sft(
                manifest_path=options.manifest,
                task_ids=options.task_id,
                output_dir=options.output_dir,
                config=load_config(options.config, options.overrides),
                trajectories_path=options.trajectories,
                max_context_tokens=options.max_context_tokens,
            )
            record = json.loads(manifest.read_text())
            print(f"{record['status']}: {record['counts']}\n{manifest}")
            return
        if options.command == "spend":
            ledger = SpendLedger(
                options.ledger, policy=SpendPolicy.read(options.policy)
            )
            if options.spend_command == "authorize-policy":
                evidence = json.loads(options.evidence.read_text())
                ledger.authorize_policy(
                    SpendPolicy.read(options.new_policy),
                    evidence=evidence,
                    allow_project_increase=options.allow_project_increase,
                )
            if options.spend_command in ("reconcile", "reconcile-total"):
                evidence = json.loads(options.evidence.read_text())
                if not isinstance(evidence, dict):
                    raise ValueError("Billing evidence must be a JSON object")
                if options.spend_command == "reconcile-total":
                    ledger.reconcile_total(
                        actual_cny=options.actual_cny,
                        evidence=evidence,
                        group_actual_cny=dict(
                            item.split("=", 1) for item in options.group_actual_cny
                        ),
                    )
                else:
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
