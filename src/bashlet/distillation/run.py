"""Run OPSD with frozen conditions, durable updates and local evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from bashlet.backends.journal import RequestJournal, summarize_requests
from bashlet.config import config_record
from bashlet.distillation import OPSDConfig, OPSDRun
from bashlet.distillation.config import (
    RunConfig,
    resolve_condition,
    validate_training_selection,
)
from bashlet.distillation.development import _decision, _evaluate_checkpoint
from bashlet.distillation.preflight import validate_preflight_evidence
from bashlet.distillation.pytrio import PyTRIOOPSD
from bashlet.harness.experiment import run_experiment, task_from_record
from bashlet.harness.models import Task
from bashlet.monitoring import (
    MonitorConfig,
    RunMonitor,
    request_metrics,
    write_document,
)
from bashlet.tasks import VERSIONED_CONVERSION, task_family, task_path, validate_task


def _requests(
    journal: RequestJournal,
    group: str,
    evaluation_group: str | None = None,
    *additional_groups: str,
) -> dict[str, Any]:
    snapshot = journal.snapshot()
    return summarize_requests(
        snapshot,
        [
            e
            for e in snapshot["entries"]
            if e["group"] in {group, evaluation_group, *additional_groups}
        ],
    )


def _visits(config: RunConfig) -> list[dict[str, Any]]:
    visits: list[dict[str, Any]] = []
    for epoch in range(1, config.epochs + 1):
        identities = list(config.train_task_ids)
        if config.shuffle_seed is not None:
            random.Random(config.shuffle_seed + epoch - 1).shuffle(identities)
        visits.extend(
            {
                "visit_id": f"epoch-{epoch}/{identity}",
                "epoch": epoch,
                "task_id": identity,
                "status": "not_started",
            }
            for identity in identities
        )
    return visits


def _visit_receipt(row: dict[str, Any], batch: dict[str, Any]) -> None:
    row.update(
        status=batch["status"],
        episode=batch["episode"],
        reason=batch["reason"],
        batch_index=batch["index"],
        supervised_tokens=batch["supervised_tokens"],
        nonzero_signal_tokens=batch.get("nonzero_signal_tokens"),
    )
    if "update" in batch:
        row.update(
            optimizer_step=batch["update"]["optimizer_step"],
            updated_policy=batch["update"]["student_policy"],
            training_state=batch["update"]["training_state"],
        )


def _training_progress(
    record: dict[str, Any], run: OPSDRun, epoch: int
) -> dict[str, Any]:
    rows = record["training_tasks"]
    sampled = [
        row
        for row in rows
        if row.get("episode")
        and run.record["batches"][row["batch_index"]]["interaction_steps"]
    ]
    updated = [row for row in rows if row.get("updated_policy")]
    return {
        "epoch": epoch,
        "planned_families": len(record["configuration"]["pilot"]["train_task_ids"]),
        "visited_families": len({row["task_id"] for row in sampled}),
        "updated_families": len({row["task_id"] for row in updated}),
        "episodes": len(run.record["batches"]),
        "trajectories": len(sampled),
        "optimizer_steps": run.record["optimizer_step"],
        "planned_updates": len(rows),
        "planned_visits": len(rows),
        "completed_visits": sum(row["status"] == "completed" for row in rows),
        "rejected_visits": sum(row["status"] == "rejected" for row in rows),
        "missing_visits": sum(row["status"] == "not_started" for row in rows),
        "supervised_tokens": sum(b["supervised_tokens"] for b in run.record["batches"]),
        "nonzero_signal_tokens": sum(
            b.get("nonzero_signal_tokens", 0) for b in run.record["batches"]
        ),
        "by_task_type": {
            kind: {
                "updated_families": len(
                    {row["task_id"] for row in updated if row["task_type"] == kind}
                ),
                "updates": sum(row["task_type"] == kind for row in updated),
                "supervised_tokens": sum(
                    row.get("supervised_tokens", 0)
                    for row in rows
                    if row["task_type"] == kind
                ),
                "nonzero_signal_tokens": sum(
                    row.get("nonzero_signal_tokens") or 0
                    for row in rows
                    if row["task_type"] == kind
                ),
            }
            for kind in ("short_program", "code_edit", "workspace")
        },
    }


def _execution_limit(
    source: Path | None,
    config_path: Path,
    config: RunConfig,
    root: Path,
    record: dict[str, Any],
) -> int | None:
    """Freeze a temporary execution prefix without changing the research plan."""
    previous = record.get("execution_limit")
    if source is None:
        if previous is not None:
            raise ValueError("Resume must retain the frozen execution limit")
        return None
    raw = source.read_bytes()
    settings = yaml.safe_load(raw)
    maximum = settings.get("max_updates") if isinstance(settings, dict) else None
    planned = len(config.train_task_ids) * config.epochs
    if (
        not isinstance(settings, dict)
        or settings.get("schema_version") != 1
        or type(maximum) is not int
        or not 0 < maximum < planned
        or maximum % len(config.train_task_ids)
        or not config.evaluate_each_epoch
        or not isinstance(settings.get("reason"), str)
        or not settings["reason"]
    ):
        raise ValueError("Execution limit requires a complete epoch prefix and dev")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    if settings.get("pilot_config_sha256") != config_hash:
        raise ValueError("Execution limit belongs to another frozen configuration")
    expected = {
        "config": str(source.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "pilot_config_sha256": config_hash,
        "max_updates": maximum,
        "planned_updates": planned,
        "reason": settings["reason"],
    }
    if previous is not None and previous != expected:
        raise ValueError("Resume must retain the frozen execution limit")
    if previous is None:
        training_path = root / "training/opsd.json"
        training = (
            json.loads(training_path.read_text()) if training_path.exists() else {}
        )
        if (
            record["optimizer_step"] != 0
            or any(row["status"] != "not_started" for row in record["training_tasks"])
            or training.get("optimizer_step", 0) != 0
            or training.get("batches")
            or training.get("pending_update") is not None
        ):
            raise ValueError("Execution limit must be frozen before formal training")
        record["execution_limit"] = expected
        record["execution_status"] = "incomplete"
    return maximum


def _load_training_tasks(
    config: RunConfig,
    manifest: dict[str, Any],
) -> tuple[list[Task], list[Task]]:
    """Reject family leakage across the whole training candidate pool."""
    tasks = [
        task_from_record(
            json.loads(
                task_path(Path(config.train_manifest), identity, "train").read_text()
            )
        )
        for identity in config.train_task_ids
    ]
    # Validate the frozen development manifest before creating any remote client.
    from bashlet.evaluation.harness import _tasks

    _, dev_tasks = _tasks(Path(config.dev_plan))
    if len({task_family(t) for t in dev_tasks}) != len(dev_tasks):
        raise ValueError("Development evaluation requires independent task families")
    families = {task_family(t) for t in tasks}
    if len(families) != len(tasks):
        raise ValueError("Pilot training requires independent task families")
    for task in tasks:
        assert task.origin is not None
        validate_task(task, harness=True)
    reserved = (
        {
            entry["family_id"]
            for entry in manifest["tasks"]
            if entry["assignment"]["role"] == "train_candidate"
        }
        if manifest["conversion"] == VERSIONED_CONVERSION
        else families
    )
    if any(t.origin is None or task_family(t) in reserved for t in dev_tasks):
        raise ValueError("Pilot training and development families must be disjoint")
    return tasks, dev_tasks


def _recover_completed_visit(
    row: dict[str, Any],
    run: OPSDRun,
) -> bool:
    """Recognize an update completed before the parent saved its receipt."""
    if row["status"] != "running":
        return False
    index = row.get("batch_index")
    batches = run.record["batches"]
    if type(index) is not int or not 0 <= index < len(batches):
        return False
    batch = batches[index]
    episode = json.loads((Path(batch["episode"]) / "run.json").read_text())
    if (
        "update" not in batch
        or episode["task_id"] != row["task_id"]
        or batch["student_policy"] != row.get("student_policy")
    ):
        return False
    _visit_receipt(row, batch)
    row.update(status="completed", recovered_completed_update=True)
    return True


def _train(
    config_path: Path,
    root: Path,
    resume: bool,
    execution_limit: Path | None,
    preflight_evidence: Path | None = None,
) -> Path:
    config = RunConfig.read(config_path)
    condition, backend_config = resolve_condition(config)
    training = OPSDConfig(**config.training)
    if not training.group or backend_config.group != training.group:
        raise ValueError("Student and teacher must share a request group")
    manifest = json.loads(Path(config.train_manifest).read_text())
    planned_visits = _visits(config)
    selection_sha256 = validate_training_selection(config, manifest)
    proof_path = preflight_evidence or (
        Path(config.preflight_evidence) if config.preflight_evidence else None
    )
    preflight_sha256 = validate_preflight_evidence(
        proof_path, condition, backend_config, training
    )
    tasks, dev_tasks = _load_training_tasks(config, manifest)
    monitoring = MonitorConfig(**config.monitoring)
    if config.evaluation is None:
        raise ValueError("OPSD requires explicit local evaluation settings")
    monitoring.credential()
    journal = RequestJournal(
        Path(config.journal_path) if config.journal_path else root / "requests.sqlite3"
    )
    evaluation_group = config.evaluation_group or training.group
    extra_groups = tuple(config.evaluation_groups.values())
    path = root / "pilot.json"
    frozen = {
        "pilot": asdict(config),
        "condition": config_record(condition),
        "dev_plan_sha256": hashlib.sha256(
            Path(config.dev_plan).read_bytes()
        ).hexdigest(),
        "train_manifest_sha256": hashlib.sha256(
            Path(config.train_manifest).read_bytes()
        ).hexdigest(),
    }
    frozen = json.loads(json.dumps(frozen))
    if selection_sha256 is not None:
        frozen["training_selection_sha256"] = selection_sha256
    if preflight_sha256 is not None:
        frozen["preflight_sha256"] = preflight_sha256
    tasks_by_id = {task.task_id: task for task in tasks}
    if path.exists():
        if not resume:
            raise ValueError("Pilot exists; use --resume to preserve progress")
        record = json.loads(path.read_text())
        previous = record["configuration"] | {
            "pilot": asdict(RunConfig(**record["configuration"]["pilot"]))
        }
        if previous != frozen:
            raise ValueError(
                "Pilot resume must preserve tasks, conditions and configuration"
            )
        if len(record["training_tasks"]) != len(planned_visits):
            raise ValueError("Pilot resume must retain every epoch visit")
        for row, planned in zip(record["training_tasks"], planned_visits, strict=True):
            if (
                row["task_id"] != planned["task_id"]
                or row.get("epoch", 1) != planned["epoch"]
                or row.get("visit_id", planned["visit_id"]) != planned["visit_id"]
            ):
                raise ValueError("Pilot resume must retain its fixed visit order")
            row.setdefault("epoch", planned["epoch"])
            row.setdefault("visit_id", planned["visit_id"])
    else:
        if resume:
            raise ValueError("No pilot exists to resume")
        record = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "configuration": frozen,
            "status": "preflight",
            "optimizer_step": 0,
            "training_tasks": planned_visits,
            "epochs_completed": 0,
            "training_complete": False,
            "main_result": None,
            "baseline": None,
            "evaluation": None,
            "dev_task_ids": [t.task_id for t in dev_tasks],
            "elapsed_seconds": 0.0,
            "requests": _requests(
                journal, training.group, evaluation_group, *extra_groups
            ),
        }
        write_document(path, record)
    maximum = _execution_limit(execution_limit, config_path, config, root, record)
    if record["status"] == "complete":
        return path
    for row in record["training_tasks"]:
        task = tasks_by_id[row["task_id"]]
        assert task.origin is not None
        row.setdefault("family_id", task_family(task))
        if task.assignment is not None:
            row.setdefault("task_assignment", asdict(task.assignment))
        row.setdefault("task_type", task.task_type or "short_program")
    record["configuration_path"] = str(config_path.resolve())
    record["run_dir"] = str(root)
    write_document(path, record)
    monitor = RunMonitor(
        root / "summary",
        monitoring,
        phase="opsd-pilot",
        metadata={
            "evidence_kind": "experimental",
            "base_model": backend_config.base_model,
            "server_model_revision": None,
            "training": asdict(training),
        },
        require_online=True,
    )
    # The W&B SDK has one active global run; each stage owns a distinct run.
    monitor.finish()
    started = time.monotonic()
    run = None
    try:
        adapter = PyTRIOOPSD(
            backend_config, journal=journal, output_dir=root / "training"
        )
        run = OPSDRun(
            root / "training",
            snapshot=adapter.snapshot,
            adapter=adapter,
            journal=journal,
            config=training,
            monitoring=monitoring,
            resume=(root / "training/opsd.json").exists(),
        )
        adapter.bind_run(run.record["run_id"])
        if record.get("execution_status") == "complete":
            from bashlet.distillation.development import development_curve

            epoch = record["epochs_completed"]
            checkpoint = record["epoch_checkpoints"][str(epoch)]
            if (
                run.record["optimizer_step"] != maximum
                or run.student.identity != checkpoint["snapshot"]
                or run.record["last_update"]["training_state"]
                != checkpoint["training_state"]
                or development_curve(record)[f"epoch-{epoch}"]["status"] != "complete"
            ):
                raise ValueError("Completed execution limit has inconsistent evidence")
            return path
        record["initial_snapshot"] = asdict(adapter.snapshot)
        record["training_run"] = str(run.path)
        record.pop("error_type", None)
        record.pop("reason", None)
        record["status"] = "baseline"
        write_document(path, record)
        baseline_stage = _evaluate_checkpoint(
            config=config,
            root=root,
            path=path,
            record=record,
            run=run,
            adapter=adapter,
            condition=condition,
            monitoring=monitoring,
            phase="baseline",
            snapshot=adapter.snapshot.identity,
            optimizer_step=0,
        )
        if baseline_stage["runs"]:
            record["baseline"] = next(iter(baseline_stage["runs"].values()))["path"]
        if baseline_stage["status"] != "complete":
            raise ValueError("Complete matched baseline is required before updating")
        write_document(path, record)
        for epoch in range(1, config.epochs + 1):
            epoch_rows = [
                row for row in record["training_tasks"] if row["epoch"] == epoch
            ]
            for row in epoch_rows:
                task = tasks_by_id[row["task_id"]]
                if _recover_completed_visit(row, run):
                    record["optimizer_step"] = run.record["optimizer_step"]
                    record["training_progress"] = _training_progress(record, run, epoch)
                    write_document(path, record)
                if row["status"] == "completed":
                    continue
                if row["status"] != "not_started":
                    raise ValueError(
                        "Interrupted training visit requires reconciliation"
                    )
                if maximum is not None and run.record["optimizer_step"] >= maximum:
                    break
                row.update(
                    status="running",
                    batch_index=len(run.record["batches"]),
                    student_policy=run.student.identity,
                    request_count_before_attempt=len(journal.snapshot()["entries"]),
                    family_id=task_family(task),
                    task_type=task.task_type or "short_program",
                )
                record["status"] = "training"
                write_document(path, record)
                result = run_experiment(
                    task=task,
                    model=run.student,
                    opsd_run=run,
                    config=condition,
                    output_dir=root / "rollouts",
                    assess_first_implementation=True,
                )
                batch = run.record["batches"][-1]
                _visit_receipt(row, batch)
                row["budget_kind"] = result.budget_kind
                record["optimizer_step"] = run.record["optimizer_step"]
                record["training_progress"] = _training_progress(record, run, epoch)
                write_document(path, record)
                print(
                    f"OPSD epoch={epoch} {task.task_id}: {row['status']}; "
                    f"step={record['optimizer_step']}",
                    flush=True,
                )
                if row["status"] != "completed":
                    raise ValueError(
                        "Pilot visit did not complete; stop before another update"
                    )
            if any(row["status"] != "completed" for row in epoch_rows):
                break
            record["epochs_completed"] = epoch
            last = epoch_rows[-1]
            record.setdefault("epoch_checkpoints", {})[str(epoch)] = {
                "snapshot": last["updated_policy"],
                "training_state": last["training_state"],
                "optimizer_step": last["optimizer_step"],
            }
            write_document(path, record)
            if config.evaluate_each_epoch and epoch < config.epochs:
                last = epoch_rows[-1]
                stage = _evaluate_checkpoint(
                    config=config,
                    root=root,
                    path=path,
                    record=record,
                    run=run,
                    adapter=adapter,
                    condition=condition,
                    monitoring=monitoring,
                    phase=f"epoch-{epoch}",
                    snapshot=last["updated_policy"],
                    optimizer_step=last["optimizer_step"],
                )
                if stage["status"] != "complete":
                    raise ValueError("Intermediate checkpoint evaluation is incomplete")
            if maximum is not None and epoch * len(config.train_task_ids) >= maximum:
                break
        run._roles()
        record["student_policy"] = run.student.identity
        record["teacher_policy"] = adapter.teacher_identity
        record["training_complete"] = (
            record["epochs_completed"] == config.epochs
            and run.record["optimizer_step"] == len(planned_visits)
            and all(row.get("updated_policy") for row in record["training_tasks"])
        )
        record["main_result"] = None
        if maximum is not None:
            epoch = maximum // len(config.train_task_ids)
            record["execution_status"] = (
                "complete"
                if run.record["optimizer_step"] == maximum
                and record["epochs_completed"] == epoch
                and record.get("evaluations", {})
                .get(f"epoch-{epoch}", {})
                .get("status")
                == "complete"
                else "incomplete"
            )
        if (maximum is not None or config.require_complete_training) and not record[
            "training_complete"
        ]:
            record["status"] = "incomplete"
            return path
        if record["training_complete"]:
            record["main_result"] = {
                "epoch": config.epochs,
                "snapshot": run.student.identity,
                "training_state": run.record["last_update"]["training_state"],
                "optimizer_step": run.record["optimizer_step"],
                "selection": "prespecified final epoch; no development selection",
            }
        stage = _evaluate_checkpoint(
            config=config,
            root=root,
            path=path,
            record=record,
            run=run,
            adapter=adapter,
            condition=condition,
            monitoring=monitoring,
            phase="final",
            snapshot=run.student.identity,
            optimizer_step=run.record["optimizer_step"],
        )
        if stage["runs"]:
            record["evaluation"] = next(iter(stage["runs"].values()))["path"]
        record["status"] = "complete" if stage["status"] == "complete" else "incomplete"
    except BaseException as error:
        if (
            run is not None
            and isinstance(error, ValueError)
            and str(error).startswith(
                (
                    "W&B startup failed;",
                    "W&B network unavailable;",
                    "W&B online verification required",
                )
            )
            and run.record["pending_update"] is None
        ):
            snapshot = journal.snapshot()
            for row in record["training_tasks"]:
                if (
                    row["status"] == "running"
                    and row.get("batch_index") == len(run.record["batches"])
                    and row.get("student_policy") == run.student.identity
                    and row.get("request_count_before_attempt")
                    == len(snapshot["entries"])
                    and not snapshot["pending_count"]
                ):
                    # begin() can fail before sampling. The unchanged journal
                    # and batch count prove this visit has no remote work to replay.
                    row.update(
                        status="not_started",
                        recovered_unstarted_visit=True,
                        recovery_reason="monitor_start_failed_before_sampling",
                    )
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        record.pop("reason", None)
        # Exception messages from external providers can contain private request data.
        if isinstance(error, ValueError):
            record["reason"] = str(error)
    finally:
        record["elapsed_seconds"] += time.monotonic() - started
        if run is not None:
            record["optimizer_step"] = run.record["optimizer_step"]
        record["requests"] = _requests(
            journal, training.group, evaluation_group, *extra_groups
        )
        _decision(record)
        write_document(path, record)
        monitor = RunMonitor.resume(root / "summary")
        monitor.log(
            f"status-{record['status']}-{record['optimizer_step']}",
            {
                "run/status": record["status"],
                "train/optimizer_step": record["optimizer_step"],
                "run/reason": record["decision"]["action"],
                **request_metrics(record["requests"]),
            },
        )
        for phase, point in record["development_curve"].items():
            monitor.log(
                f"development-{phase}-{point['status']}",
                {
                    "eval/phase": phase,
                    "train/optimizer_step": point["optimizer_step"],
                    **(
                        {"eval/mean_accuracy": point["mean_accuracy"]}
                        if point["mean_accuracy"] is not None
                        else {}
                    ),
                    "eval/status": point["status"],
                    "eval/planned_families": point["planned_families"],
                    "eval/planned_episodes": point["planned_episodes"],
                    "eval/missing_episodes": len(point["missing"]),
                },
            )
        monitor.finish()
    return path


def train_opsd(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    execution_limit: Path | None = None,
    preflight_evidence: Path | None = None,
) -> Path:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Pilot is already active") from None
        return _train(config_path, root, resume, execution_limit, preflight_evidence)


def verify_opsd_checkpoints(
    config_path: Path, output_dir: Path, *, native: bool = False, retry: bool = False
) -> Path:
    """Verify the saved training state without another optimizer update."""
    root = output_dir.resolve()
    config = RunConfig.read(config_path)
    _, backend_config = resolve_condition(config)
    if backend_config.source != "base":
        raise ValueError("Checkpoint parameters: only Base initialization is supported")
    training = OPSDConfig(**config.training)
    journal = RequestJournal(
        Path(config.journal_path) if config.journal_path else root / "requests.sqlite3"
    )
    path = root / "checkpoint-validation.json"
    with (root / "pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Pilot is already active") from None
        history = []
        if path.exists():
            previous = json.loads(path.read_text())
            if not retry or previous.get("status") not in ("started", "failed"):
                raise ValueError(
                    "Checkpoint validation exists; do not repeat remote probes"
                )
            workflow_path = root.parent / "workflow.json"
            if workflow_path.exists() and json.loads(workflow_path.read_text()).get(
                "preflight_sha256"
            ):
                raise ValueError("Accepted preflight evidence cannot be replaced")
            training_record = json.loads((root / "training/opsd.json").read_text())
            if (
                training_record["pending_update"] is not None
                or journal.snapshot()["pending_count"]
            ):
                raise ValueError(
                    "Checkpoint retry requires settled updates and requests"
                )
            history = list(previous.get("previous_attempts", []))
            directory = (
                root / "verification-attempts" / f"attempt-{len(history) + 1:04d}"
            )
            directory.mkdir(parents=True, exist_ok=False)
            archived = {}
            for source in (path, root / "training/checkpoint-scores.json"):
                if source.exists():
                    content = source.read_bytes()
                    destination = directory / source.name
                    destination.write_bytes(content)
                    archived[source.name] = {
                        "path": str(destination),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
            history.append(archived)
        elif retry:
            raise ValueError("No failed checkpoint validation exists to retry")
        attempt = len(history) + 1
        record: dict[str, Any] = {"status": "started"}
        record.update(verification_attempt=attempt, previous_attempts=history)
        write_document(path, record)
        try:
            adapter = PyTRIOOPSD(
                backend_config,
                journal=journal,
                output_dir=root / "training",
            )
            run = OPSDRun(
                root / "training",
                snapshot=adapter.snapshot,
                adapter=adapter,
                journal=journal,
                config=training,
                monitoring=MonitorConfig(**config.monitoring),
                resume=True,
            )
            adapter.bind_run(run.record["run_id"])
            record = adapter.verify_progress(run)
            if native and record["status"] == "complete":
                from bashlet.distillation.preflight import verify_native_preflight

                record["native"] = verify_native_preflight(
                    config_path, root, adapter, run
                )
        except BaseException as error:
            record.update(status="failed", error_type=type(error).__name__)
            if isinstance(error, ValueError) and str(error).startswith(
                (
                    "Native preflight",
                    "Truncated thinking",
                    "Real truncation",
                    "Checkpoint parameters",
                    "Checkpoint scoring",
                    "Checkpoint probabilities",
                )
            ):
                record["reason"] = str(error)
        finally:
            record.update(verification_attempt=attempt, previous_attempts=history)
            record["requests"] = _requests(
                journal,
                training.group,
                config.evaluation_group,
                *config.evaluation_groups.values(),
            )
            write_document(path, record)
            pilot_path = root / "pilot.json"
            pilot = json.loads(pilot_path.read_text())
            pilot["requests"] = record["requests"]
            pilot["checkpoint_validation"] = str(path)
            _decision(pilot)
            write_document(pilot_path, pilot)
            monitor = RunMonitor.resume(root / "summary")
            monitor.log(
                f"checkpoint-validation-attempt-{attempt}",
                {
                    "run/status": record["status"],
                    "checkpoint/verified": record["status"] == "complete",
                    **request_metrics(record["requests"]),
                },
            )
            monitor.finish()
    return path
