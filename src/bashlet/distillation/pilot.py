"""A resumable bounded pilot: matched baseline, real updates, paired evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import random
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from ..config import ExperimentConfig, config_record, load_config
from ..evaluation.harness import evaluate
from ..experiment import run_experiment, task_from_record
from ..monitoring import MonitorConfig, RunMonitor, spend_metrics, write_document
from ..spend import SpendLedger, SpendPolicy, summarize_cost
from ..tasks import task_path, validate_harness_origin
from . import OPSDConfig, OPSDRun
from .pytrio import PyTRIOOPSD, PyTRIOOPSDConfig


@dataclass(frozen=True)
class PilotConfig:
    train_manifest: str
    train_task_ids: list[str]
    dev_plan: str
    condition: list[str]
    ledger_path: str = "runs/spend.sqlite3"
    backend: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    evaluation_group: str | None = None
    stop_on_training_budget: bool = False
    defer_decision: bool = False
    epochs: int = 1
    shuffle_seed: int | None = None
    evaluation_seeds: list[int] = field(default_factory=list)
    evaluate_each_epoch: bool = False
    evaluation_groups: dict[str, str] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> PilotConfig:
        value = cls(**yaml.safe_load(path.read_text()))
        if not value.train_task_ids or len(set(value.train_task_ids)) != len(
            value.train_task_ids
        ):
            raise ValueError("Pilot needs unique fixed training task IDs")
        if type(value.epochs) is not int or value.epochs < 1:
            raise ValueError("Pilot epochs must be a positive integer")
        if value.shuffle_seed is not None and (
            type(value.shuffle_seed) is not int or value.shuffle_seed < 0
        ):
            raise ValueError("Pilot shuffle seed must be a nonnegative integer")
        if len(set(value.evaluation_seeds)) != len(value.evaluation_seeds) or any(
            type(seed) is not int or seed < 0 for seed in value.evaluation_seeds
        ):
            raise ValueError(
                "Pilot evaluation seeds must be distinct nonnegative integers"
            )
        if type(value.evaluate_each_epoch) is not bool:
            raise ValueError("Pilot epoch evaluation must be a boolean")
        if value.evaluation_groups and (
            set(value.evaluation_groups) != {"baseline", "intermediate", "final"}
            or any(
                not isinstance(group, str) or not group
                for group in value.evaluation_groups.values()
            )
        ):
            raise ValueError(
                "Pilot evaluation groups require baseline, intermediate and final"
            )
        return value


def _cost(
    ledger: SpendLedger,
    group: str,
    evaluation_group: str | None = None,
    *additional_groups: str,
) -> dict[str, Any]:
    snapshot = ledger.snapshot()
    return summarize_cost(
        snapshot,
        [
            e
            for e in snapshot["entries"]
            if e["group"] in {group, evaluation_group, *additional_groups}
        ],
    )


def _visits(config: PilotConfig) -> list[dict[str, Any]]:
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


def _training_budget_stop(
    config: PilotConfig,
    run: OPSDRun,
    batch: dict[str, Any],
    *,
    cash_exhausted: bool = False,
) -> bool:
    return bool(
        config.stop_on_training_budget
        and (batch["reason"] in {"budget_refused", "BudgetRefused"} or cash_exhausted)
        and run.record["pending_update"] is None
        and run.student.identity == run.record["student_policy"]
        and not _cost(run.ledger, run.config.group)["pending_upper_bound_cny"]
    )


def _evaluate_checkpoint(
    *,
    config: PilotConfig,
    root: Path,
    path: Path,
    record: dict[str, Any],
    run: OPSDRun,
    adapter: PyTRIOOPSD,
    condition: ExperimentConfig,
    monitoring: MonitorConfig,
    phase: str,
    snapshot: str,
    optimizer_step: int,
) -> dict[str, Any]:
    from ..evaluation.harness import _digest, _tasks

    allocation = (
        "baseline"
        if phase == "baseline"
        else "final"
        if phase == "final"
        else "intermediate"
    )
    group = config.evaluation_groups.get(
        allocation, config.evaluation_group or run.config.group
    )
    seeds: Sequence[int | None] = config.evaluation_seeds or [
        condition.model.sampling.seed
    ]
    expected = {"snapshot": snapshot, "optimizer_step": optimizer_step, "group": group}
    stages: dict[str, dict[str, Any]] = record.setdefault("evaluations", {})
    stage = stages.setdefault(phase, expected | {"status": "running", "runs": {}})
    if any(stage[key] != value for key, value in expected.items()):
        raise ValueError(
            "Checkpoint evaluation must retain its snapshot, step and allocation"
        )
    # A cash stop can leave the student at an already evaluated checkpoint.
    # Identical checkpoint/seed evidence supplies the final report once.
    if phase == "final" and config.evaluation_seeds and not stage["runs"]:
        for previous_phase, previous in stages.items():
            if (
                previous_phase != phase
                and previous["status"] == "complete"
                and previous["snapshot"] == snapshot
                and previous["optimizer_step"] == optimizer_step
                and set(previous["runs"]) == {str(seed) for seed in seeds}
            ):
                stage.update(
                    runs={seed: dict(ref) for seed, ref in previous["runs"].items()},
                    reused_from=previous_phase,
                )
                break
    write_document(path, record)
    plan, _ = _tasks(Path(config.dev_plan))
    for seed in seeds:
        resolved = replace(
            condition,
            model=replace(
                condition.model,
                sampling=replace(condition.model.sampling, seed=seed),
            ),
        )
        legacy = not config.evaluation_seeds and config.epochs == 1
        directory = (
            (root / ("baseline" if phase == "baseline" else "updated"))
            if legacy
            else root / "evaluation" / phase / f"seed-{seed}"
        )
        destination = directory / "evaluation.json"
        reference = stage["runs"].get(str(seed))
        if reference:
            destination = Path(reference["path"])
            directory = destination.parent
        if (
            reference
            and hashlib.sha256(destination.read_bytes()).hexdigest()
            != reference["sha256"]
        ):
            raise ValueError("Checkpoint evaluation evidence was changed")
        prior = json.loads(destination.read_text()) if destination.exists() else None
        if prior and (
            prior["snapshot"] != snapshot
            or prior["config"] != json.loads(json.dumps(config_record(resolved)))
            or prior["plan"] != plan
            or prior["comparison_sha256"] != _digest(prior["comparison"])
        ):
            raise ValueError(
                "Checkpoint evaluation resume must retain tasks and conditions"
            )
        if prior is None or prior["status"] != "complete":
            if run.student.identity != snapshot:
                raise ValueError(
                    "Incomplete checkpoint evaluation cannot use a later student"
                )
            evaluate(
                plan_path=Path(config.dev_plan),
                snapshot=snapshot,
                evidence_kind="experimental",
                condition=resolved,
                output_dir=root,
                run_dir=directory,
                model=adapter.evaluation_model(group=group),
                monitoring=monitoring,
                training_run=run.path,
                phase=phase + "-eval",
                resume=directory.exists(),
            )
            prior = json.loads(destination.read_text())
        stage["runs"][str(seed)] = {
            "path": str(destination),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "status": prior["status"],
        }
        write_document(path, record)
        if prior["status"] != "complete":
            stage["status"] = "incomplete"
            write_document(path, record)
            return stage
    stage["status"] = "complete"
    write_document(path, record)
    return stage


def _decision(record: dict[str, Any]) -> None:
    before_path, after_path = record.get("baseline"), record.get("evaluation")
    before = json.loads(Path(before_path).read_text()) if before_path else None
    after = json.loads(Path(after_path).read_text()) if after_path else None
    comparable = bool(
        before
        and after
        and before["status"] == after["status"] == "complete"
        and before["comparison_sha256"] == after["comparison_sha256"]
    )
    progressed = bool(record.get("optimizer_step", 0))
    difference = None
    if comparable:
        assert before is not None and after is not None
        difference = (
            after["summary"]["passed_tasks"] - before["summary"]["passed_tasks"]
        )
    seed_comparisons = {}
    stages = record.get("evaluations", {})
    if "baseline" in stages and "final" in stages:
        baseline, final = stages["baseline"], stages["final"]
        comparable = baseline["status"] == final["status"] == "complete"
        for seed, reference in baseline["runs"].items():
            old = json.loads(Path(reference["path"]).read_text())
            new_ref = final["runs"].get(seed)
            new = json.loads(Path(new_ref["path"]).read_text()) if new_ref else None
            matched = bool(
                new
                and old["status"] == new["status"] == "complete"
                and old["comparison_sha256"] == new["comparison_sha256"]
            )
            comparable = comparable and matched
            seed_comparisons[seed] = {
                "matched_complete": matched,
                "baseline": old["summary"],
                "updated": new["summary"] if new else None,
                "learning_signal": new["summary"]["passed_tasks"]
                - old["summary"]["passed_tasks"]
                if matched and new
                else None,
            }
        difference = (
            sum(item["learning_signal"] for item in seed_comparisons.values())
            if comparable
            else None
        )
    record["comparison"] = {
        "matched_complete": comparable,
        "baseline": before["summary"] if before else None,
        "updated": after["summary"] if after else None,
        "learning_signal": difference,
        "by_seed": seed_comparisons,
        "scope": "Fixed pilot development subset; no held-out acceptance",
    }
    if record["configuration"]["pilot"].get("defer_decision", False):
        conclusion, reason = (
            "review_results" if progressed and comparable else "insufficient_evidence",
            "Exploratory results; no numeric main-OPSD threshold was chosen",
        )
    elif not progressed or not comparable:
        conclusion, reason = (
            "repair_then_retry",
            "Incomplete update or matched development evidence",
        )
    elif record["comparison"]["learning_signal"] > 0:
        conclusion, reason = (
            "continue_opsd",
            "Positive pilot task signal; larger development confirmation required",
        )
    else:
        conclusion, reason = (
            "stop_investment",
            "No positive task signal in the matched pilot; loss/KL is insufficient",
        )
    record["decision"] = {"action": conclusion, "reason": reason}
    record["billing"] = {
        "teacher_category_verified": False,
        "cache_discount_assumed": False,
        "actual_cny": record["cost"]["actual_cny"],
        "sdk_scoring": "0.2.9 compute_logprobs uses prompt logprobs and max_tokens=1",
        "limitation": "SDK drops billing details; usage and ceilings are not bills",
    }
    record["alternative_9b"] = {
        "started": False,
        "teacher_identity": None,
        "compatibility": "unverified; separate conditional task required",
    }


def _pilot(config_path: Path, root: Path, resume: bool) -> Path:
    config = PilotConfig.read(config_path)
    backend_config = PyTRIOOPSDConfig(**config.backend)
    training = OPSDConfig(**config.training)
    if (
        not training.group
        or backend_config.group != training.group
        or backend_config.spend_config != training.spend_config
    ):
        raise ValueError("Pilot student and teacher must share a cash allocation")
    condition = load_config([Path(p) for p in config.condition])
    tasks = [
        task_from_record(
            json.loads(
                task_path(Path(config.train_manifest), identity, "train").read_text()
            )
        )
        for identity in config.train_task_ids
    ]
    # Validate the frozen development manifest before creating any remote client.
    from ..evaluation.harness import _tasks

    _, dev_tasks = _tasks(Path(config.dev_plan))
    families = {t.origin.family_id for t in tasks if t.origin is not None}
    if len(families) != len(tasks):
        raise ValueError("Pilot training requires independent task families")
    for task in tasks:
        assert task.origin is not None
        validate_harness_origin(task.origin)
    if any(t.origin is None or t.origin.family_id in families for t in dev_tasks):
        raise ValueError("Pilot training and development families must be disjoint")
    monitoring = MonitorConfig(**config.monitoring)
    monitoring.credential()
    ledger = SpendLedger(
        Path(config.ledger_path), policy=SpendPolicy.read(Path(training.spend_config))
    )
    evaluation_group = config.evaluation_group or training.group
    extra_groups = tuple(config.evaluation_groups.values())
    if any(
        group not in ledger.policy.groups
        for group in (training.group, evaluation_group, *extra_groups)
    ):
        raise ValueError("Pilot training and evaluation need named cash allocations")
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
    planned_visits = _visits(config)
    tasks_by_id = {task.task_id: task for task in tasks}
    if path.exists():
        if not resume:
            raise ValueError("Pilot exists; use --resume to preserve progress")
        record = json.loads(path.read_text())
        previous = record["configuration"] | {
            "pilot": asdict(PilotConfig(**record["configuration"]["pilot"]))
        }
        if previous != frozen:
            raise ValueError(
                "Pilot resume must preserve tasks, conditions and configuration"
            )
        if record["status"] == "complete":
            return path
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
            "baseline": None,
            "evaluation": None,
            "dev_task_ids": [t.task_id for t in dev_tasks],
            "elapsed_seconds": 0.0,
            "cost": _cost(ledger, training.group, evaluation_group, *extra_groups),
        }
        write_document(path, record)
    for row in record["training_tasks"]:
        task = tasks_by_id[row["task_id"]]
        assert task.origin is not None
        row.setdefault("family_id", task.origin.family_id)
        row.setdefault("task_type", task.task_type or "short_program")
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
            backend_config, ledger=ledger, output_dir=root / "training"
        )
        run = OPSDRun(
            root / "training",
            snapshot=adapter.snapshot,
            adapter=adapter,
            ledger=ledger,
            config=training,
            monitoring=monitoring,
            resume=(root / "training/opsd.json").exists(),
        )
        adapter.bind_run(run.record["run_id"])
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
                if record.get("training_stop"):
                    break
                task = tasks_by_id[row["task_id"]]
                if row["status"] == "running":
                    # The captured batch index identifies this visit, even when
                    # the same task was already updated in an earlier epoch.
                    index = row.get("batch_index")
                    candidates = (
                        run.record["batches"][index : index + 1]
                        if index is not None
                        else [
                            b
                            for b in run.record["batches"]
                            if "update" in b
                            and json.loads(
                                (Path(b["episode"]) / "run.json").read_text()
                            )["task_id"]
                            == task.task_id
                        ]
                    )
                    if len(candidates) == 1:
                        batch = candidates[0]
                        episode = json.loads(
                            (Path(batch["episode"]) / "run.json").read_text()
                        )
                        if (
                            "update" in batch
                            and episode["task_id"] == task.task_id
                            and batch["student_policy"]
                            == row.get("student_policy", batch["student_policy"])
                        ):
                            _visit_receipt(row, batch)
                            # Monitoring can interrupt after the optimizer and
                            # full training state were durably completed.
                            row["status"] = "completed"
                            row["recovered_completed_update"] = True
                            if _training_budget_stop(
                                config,
                                run,
                                batch,
                                cash_exhausted=episode.get("budget_kind") == "spend",
                            ):
                                record["training_stop"] = "budget_refused"
                            record["optimizer_step"] = run.record["optimizer_step"]
                            record["training_progress"] = _training_progress(
                                record, run, epoch
                            )
                            write_document(path, record)
                if record.get("training_stop"):
                    break
                if row["status"] == "completed":
                    continue
                if row["status"] != "not_started":
                    raise ValueError(
                        "Interrupted training visit requires reconciliation"
                    )
                row.update(
                    status="running",
                    batch_index=len(run.record["batches"]),
                    student_policy=run.student.identity,
                    request_count_before_attempt=len(ledger.snapshot()["entries"]),
                    family_id=task.origin.family_id if task.origin else None,
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
                if _training_budget_stop(
                    config, run, batch, cash_exhausted=result.budget_kind == "spend"
                ):
                    record["training_stop"] = "budget_refused"
                    write_document(path, record)
                    break
                if row["status"] != "completed":
                    raise ValueError(
                        "Pilot visit did not complete; stop before another update"
                    )
            if record.get("training_stop"):
                break
            record["epochs_completed"] = epoch
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
                    last_evaluation = json.loads(
                        Path(next(reversed(stage["runs"].values()))["path"]).read_text()
                    )
                    if (
                        config.stop_on_training_budget
                        and not _cost(
                            ledger, training.group, evaluation_group, *extra_groups
                        )["pending_upper_bound_cny"]
                        and any(
                            row.get("budget_kind") == "spend"
                            for row in last_evaluation["tasks"]
                        )
                    ):
                        record["training_stop"] = (
                            "intermediate_evaluation_budget_refused"
                        )
                        write_document(path, record)
                        break
                    raise ValueError("Intermediate checkpoint evaluation is incomplete")
        run._roles()
        record["student_policy"] = run.student.identity
        record["teacher_policy"] = adapter.teacher_identity
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
            snapshot = ledger.snapshot()
            for row in record["training_tasks"]:
                if (
                    row["status"] == "running"
                    and row.get("batch_index") == len(run.record["batches"])
                    and row.get("student_policy") == run.student.identity
                    and row.get("request_count_before_attempt")
                    == len(snapshot["entries"])
                    and not snapshot["pending_cny"]
                ):
                    # begin() can fail before sampling. The unchanged ledger
                    # and batch count prove this visit has no paid work to replay.
                    row.update(
                        status="not_started",
                        recovered_unstarted_visit=True,
                        recovery_reason="monitor_start_failed_before_sampling",
                    )
        record.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        # Exception messages from external providers can contain private request data.
        if isinstance(error, ValueError):
            record["reason"] = str(error)
    finally:
        record["elapsed_seconds"] += time.monotonic() - started
        if run is not None:
            record["optimizer_step"] = run.record["optimizer_step"]
        record["cost"] = _cost(ledger, training.group, evaluation_group, *extra_groups)
        _decision(record)
        write_document(path, record)
        monitor = RunMonitor.resume(root / "summary")
        monitor.log(
            f"status-{record['status']}-{record['optimizer_step']}",
            {
                "run/status": record["status"],
                "train/optimizer_step": record["optimizer_step"],
                "run/reason": record["decision"]["action"],
                **spend_metrics(record["cost"]),
            },
        )
        monitor.finish()
    return path


def run_opsd_pilot(
    config_path: Path, output_dir: Path, *, resume: bool = False
) -> Path:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Pilot is already active") from None
        return _pilot(config_path, root, resume)


def verify_opsd_checkpoints(config_path: Path, output_dir: Path) -> Path:
    """Verify the saved training state without another optimizer update."""
    root = output_dir.resolve()
    config = PilotConfig.read(config_path)
    training = OPSDConfig(**config.training)
    ledger = SpendLedger(
        Path(config.ledger_path), policy=SpendPolicy.read(Path(training.spend_config))
    )
    path = root / "checkpoint-validation.json"
    with (root / "pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Pilot is already active") from None
        if path.exists():
            raise ValueError("Checkpoint validation exists; do not repeat paid probes")
        record: dict[str, Any] = {"status": "started"}
        write_document(path, record)
        try:
            adapter = PyTRIOOPSD(
                PyTRIOOPSDConfig(**config.backend),
                ledger=ledger,
                output_dir=root / "training",
            )
            run = OPSDRun(
                root / "training",
                snapshot=adapter.snapshot,
                adapter=adapter,
                ledger=ledger,
                config=training,
                monitoring=MonitorConfig(**config.monitoring),
                resume=True,
            )
            adapter.bind_run(run.record["run_id"])
            record = adapter.verify_progress(run)
        except BaseException as error:
            record = {"status": "failed", "error_type": type(error).__name__}
        finally:
            record["cost"] = _cost(
                ledger,
                training.group,
                config.evaluation_group,
                *config.evaluation_groups.values(),
            )
            write_document(path, record)
            pilot_path = root / "pilot.json"
            pilot = json.loads(pilot_path.read_text())
            pilot["cost"] = record["cost"]
            pilot["checkpoint_validation"] = str(path)
            _decision(pilot)
            write_document(pilot_path, pilot)
            monitor = RunMonitor.resume(root / "summary")
            monitor.log(
                "checkpoint-validation",
                {
                    "run/status": record["status"],
                    "checkpoint/verified": record["status"] == "complete",
                    **spend_metrics(record["cost"]),
                },
            )
            monitor.finish()
    return path
