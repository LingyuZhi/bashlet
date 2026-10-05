"""A resumable bounded pilot: matched baseline, real updates, paired evaluation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import config_record, load_config
from ..evaluation.harness import evaluate
from ..experiment import run_experiment, task_from_record
from ..monitoring import MonitorConfig, RunMonitor, spend_metrics, write_document
from ..spend import SpendLedger, SpendPolicy, summarize_cost
from ..tasks import task_path
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

    @classmethod
    def read(cls, path: Path) -> PilotConfig:
        value = cls(**yaml.safe_load(path.read_text()))
        if not value.train_task_ids or len(set(value.train_task_ids)) != len(
            value.train_task_ids
        ):
            raise ValueError("Pilot needs unique fixed training task IDs")
        return value


def _cost(ledger: SpendLedger, group: str) -> dict[str, Any]:
    snapshot = ledger.snapshot()
    return summarize_cost(
        snapshot, [e for e in snapshot["entries"] if e["group"] == group]
    )


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
    record["comparison"] = {
        "matched_complete": comparable,
        "baseline": before["summary"] if before else None,
        "updated": after["summary"] if after else None,
        "learning_signal": difference,
        "scope": "Fixed pilot development subset; no held-out acceptance",
    }
    if not progressed or not comparable:
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
        training.group != "opsd-pilot"
        or backend_config.group != training.group
        or backend_config.spend_config != training.spend_config
    ):
        raise ValueError("All pilot stages must share the opsd-pilot cash allocation")
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
    if any(t.origin is None or t.origin.family_id in families for t in dev_tasks):
        raise ValueError("Pilot training and development families must be disjoint")
    monitoring = MonitorConfig(**config.monitoring)
    monitoring.credential()
    ledger = SpendLedger(
        Path(config.ledger_path), policy=SpendPolicy.read(Path(training.spend_config))
    )
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
    if path.exists():
        if not resume:
            raise ValueError("Pilot exists; use --resume to preserve progress")
        record = json.loads(path.read_text())
        if record["configuration"] != frozen:
            raise ValueError(
                "Pilot resume must preserve tasks, conditions and configuration"
            )
        if record["status"] == "complete":
            return path
    else:
        if resume:
            raise ValueError("No pilot exists to resume")
        record = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "configuration": frozen,
            "status": "preflight",
            "optimizer_step": 0,
            "training_tasks": [
                {"task_id": t.task_id, "status": "not_started"} for t in tasks
            ],
            "baseline": None,
            "evaluation": None,
            "dev_task_ids": [t.task_id for t in dev_tasks],
            "elapsed_seconds": 0.0,
            "cost": _cost(ledger, training.group),
        }
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
        record["status"] = "baseline"
        write_document(path, record)
        if (
            record["baseline"] is None
            or json.loads(Path(record["baseline"]).read_text())["status"] != "complete"
        ):
            record["baseline"] = str(
                evaluate(
                    plan_path=Path(config.dev_plan),
                    snapshot=adapter.snapshot.identity,
                    evidence_kind="experimental",
                    condition=condition,
                    output_dir=root,
                    run_dir=root / "baseline",
                    model=adapter.evaluation_model(),
                    monitoring=monitoring,
                    training_run=run.path,
                    phase="baseline-eval",
                    resume=(root / "baseline").exists(),
                )
            )
            write_document(path, record)
        baseline = json.loads(Path(record["baseline"]).read_text())
        if baseline["status"] != "complete":
            raise ValueError("Complete matched baseline is required before updating")
        for task, row in zip(tasks, record["training_tasks"], strict=True):
            if row["status"] == "running":
                completed = next(
                    (
                        b
                        for b in run.record["batches"]
                        if "update" in b
                        and json.loads((Path(b["episode"]) / "run.json").read_text())[
                            "task_id"
                        ]
                        == task.task_id
                    ),
                    None,
                )
                if completed:
                    row.update(
                        status="completed",
                        episode=completed["episode"],
                        reason=completed["reason"],
                        recovered_completed_update=True,
                    )
                    record["optimizer_step"] = run.record["optimizer_step"]
                    write_document(path, record)
            if row["status"] == "completed":
                continue
            if row["status"] != "not_started":
                raise ValueError("Interrupted training attempt requires reconciliation")
            row["status"] = "running"
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
            row.update(
                status=batch["status"],
                episode=str(result.run_dir),
                reason=batch["reason"],
            )
            record["optimizer_step"] = run.record["optimizer_step"]
            write_document(path, record)
            print(
                f"OPSD {task.task_id}: {row['status']}; "
                f"step={record['optimizer_step']}",
                flush=True,
            )
            if row["status"] != "completed":
                raise ValueError(
                    "Pilot batch did not complete; stop before another update"
                )
        record["student_policy"] = run.student.identity
        record["teacher_policy"] = adapter.teacher_identity
        record["evaluation"] = str(
            evaluate(
                plan_path=Path(config.dev_plan),
                snapshot=run.student.identity,
                evidence_kind="experimental",
                condition=condition,
                output_dir=root,
                run_dir=root / "updated",
                model=adapter.evaluation_model(),
                monitoring=monitoring,
                training_run=run.path,
                phase="dev-eval",
                resume=(root / "updated").exists(),
            )
        )
        updated = json.loads(Path(record["evaluation"]).read_text())
        record["status"] = (
            "complete" if updated["status"] == "complete" else "incomplete"
        )
    except BaseException as error:
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
        record["cost"] = _cost(ledger, training.group)
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
            record["cost"] = _cost(ledger, training.group)
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
