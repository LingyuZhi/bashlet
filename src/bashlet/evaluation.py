"""Fixed development cohorts, durable task states and post-episode reporting."""

import hashlib
import json
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, config_record, load_model
from .experiment import run_experiment, serialize_task, task_from_record
from .mbpp import mbpp_task_path
from .models import EpisodeResult, Model, ModelBackend, Task
from .monitoring import MonitorConfig, RunMonitor, code_revision, evaluation_metrics
from .protocol import fingerprints
from .spend import summarize_cost
from .vllm import VLLMModel

VERSION = "development-evaluation-v1"
RULES = {
    "first_implementation": "first-recorded-candidate-v1",
    "candidate_definition": (
        "Earliest post-command manifest with all declared implementation paths "
        "recorded as regular files within the implementation byte limit. Empty, "
        "syntactically invalid and unchanged initial implementations count. "
        "Grade the entire saved workspace at that step after interaction ends."
    ),
    "denominator": "all planned task IDs, including missing and failed tasks",
    "recovery_denominator": (
        "task IDs whose first implementation was independently graded and failed; "
        "the numerator requires a passing final submission. Cohorts are recorded "
        "per model and must not be treated as a shared paired failure set."
    ),
    "episode_attempts": 1,
    "retry_policy": (
        "No episode retries. Only the configured bounded backend request retries; "
        "all paid attempts use the same persistent ledger. Stop on infrastructure "
        "failure or interruption and retain every remaining task as not_started."
    ),
}


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode()
    ).hexdigest()


def _write(path: Path, record: dict[str, Any]) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(record, ensure_ascii=True, allow_nan=False, indent=2))
    pending.replace(path)


def _tasks(plan_path: Path) -> tuple[dict[str, Any], list[Task]]:
    plan = json.loads(plan_path.read_text())
    if not isinstance(plan, dict) or set(plan) != {
        "version",
        "name",
        "manifest",
        "split",
        "task_ids",
    }:
        raise ValueError(
            "Evaluation plan requires version, name, manifest, split, task_ids"
        )
    if (
        plan["version"] != VERSION
        or not isinstance(plan["name"], str)
        or not plan["name"]
    ):
        raise ValueError("Unknown evaluation version or empty name")
    if plan["split"] != "dev":
        raise ValueError(
            "Development evaluation requires split=dev; held-out is frozen"
        )
    task_ids = plan["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or not all(isinstance(item, str) and item for item in task_ids)
        or len(set(task_ids)) != len(task_ids)
    ):
        raise ValueError("A nonempty list of unique task IDs is required")
    manifest_path = (plan_path.parent / plan["manifest"]).resolve()
    tasks = [
        task_from_record(
            json.loads(mbpp_task_path(manifest_path, task_id, "dev").read_text())
        )
        for task_id in task_ids
    ]
    manifest = json.loads(manifest_path.read_text())
    return plan | {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source": manifest["source"],
        "conversion": manifest["conversion"],
    }, tasks


def _outcome(episode: dict[str, Any]) -> str:
    status = episode["status"]
    if status == "completed":
        return "passed" if episode["passed"] else "model_failure"
    if status == "invalid_submission":
        return "model_failure"
    if status == "format_error":
        return "protocol_error"
    return str(status)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [
        row["task_id"]
        for row in rows
        if row["outcome"]
        in {"not_started", "running", "infrastructure_error", "interrupted"}
        or row.get("budget_kind") == "spend"
    ]
    first_missing = [
        row["task_id"]
        for row in rows
        if row.get("first_implementation", {}).get("status")
        in {"infrastructure_error", "interrupted"}
    ]
    passed = sum(row["outcome"] == "passed" for row in rows)
    first_passed = sum(
        bool((row.get("first_implementation", {}).get("grade") or {}).get("passed"))
        for row in rows
    )
    failed_first = [
        row
        for row in rows
        if row.get("first_implementation", {}).get("status") == "graded"
        and not row["first_implementation"]["grade"]["passed"]
    ]
    recovered = [row["task_id"] for row in failed_first if row["outcome"] == "passed"]
    return {
        "planned_tasks": len(rows),
        "passed_tasks": passed,
        "final_pass_rate": passed / len(rows),
        "missing_task_ids": missing,
        "first_grading_missing_task_ids": first_missing,
        "first_implementation_passed": first_passed,
        "first_implementation_pass_rate": first_passed / len(rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "recovery": {
            "task_ids": [row["task_id"] for row in failed_first],
            "denominator": len(failed_first),
            "recovered_task_ids": recovered,
            "rate": len(recovered) / len(failed_first) if failed_first else None,
        },
        "typical_trajectories": {
            outcome: next(row["run"] for row in rows if row["outcome"] == outcome)
            for outcome in dict.fromkeys(row["outcome"] for row in rows)
            if outcome != "not_started"
        },
    }


def _cost(model: Model) -> dict[str, Any]:
    details: dict[str, Any] = (
        model.describe() if isinstance(model, ModelBackend) else {}
    )
    spend = details.get("spend", {})
    entries = [
        entry
        for entry in spend.get("entries", [])
        if entry["evidence"].get("sampling_run_cuid")
        == details.get("sampling_run_cuid")
        and details.get("sampling_run_cuid") is not None
    ]
    return summarize_cost(spend, entries)


def evaluate(
    *,
    plan_path: Path,
    snapshot: str,
    evidence_kind: str,
    condition: ExperimentConfig,
    output_dir: Path,
    monitoring: MonitorConfig | None = None,
    training_run: Path | None = None,
) -> Path:
    """CLI orchestration; every planned attempt uses the existing experiment seam."""
    plan, tasks = _tasks(plan_path)
    model = load_model(condition.model)
    if model.identity != snapshot:
        raise ValueError(
            f"Snapshot mismatch: expected {snapshot!r}, got {model.identity!r}"
        )
    details: dict[str, Any] = (
        model.describe() if isinstance(model, ModelBackend) else {}
    )
    local = isinstance(model, VLLMModel) and details.get("development_only") is False
    if evidence_kind == "experimental" and not (
        (
            details.get("backend") == "pytrio"
            and details.get("development_only") is False
        )
        or local
    ):
        raise ValueError(
            "Experimental evaluation requires PyTRIO or a verified local snapshot; "
            "use --evidence-kind test-double for development backends"
        )
    run_dir = output_dir.resolve() / uuid.uuid4().hex
    run_dir.mkdir(parents=True, mode=0o700)
    path = run_dir / "evaluation.json"
    configuration = config_record(condition)
    rows = [
        {
            "task_id": task.task_id,
            "task_sha256": hashlib.sha256(serialize_task(task).encode()).hexdigest(),
            "task_origin": asdict(task.origin) if task.origin else None,
            "feedback_command": task.feedback_command,
            "implementation_paths": list(task.implementation_paths),
            "outcome": "not_started",
            "run": None,
        }
        for task in tasks
    ]
    comparison = {
        "version": VERSION,
        "evidence_kind": evidence_kind,
        "rules": RULES,
        "tasks": [dict(row) for row in rows],
        "source": plan["source"],
        "conversion": plan["conversion"],
        "protocol": configuration["protocol"],
        "environment": configuration["environment"],
        "budget": configuration["budget"],
        "sampling": configuration["model"]["sampling"],
        "encoding": model.encoding.describe(),
    }
    record: dict[str, Any] = {
        "version": VERSION,
        "started_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision(),
        "status": "running",
        "evidence_kind": evidence_kind,
        "snapshot": snapshot,
        "snapshot_details": details,
        "plan": plan,
        "config": configuration,
        "comparison": comparison,
        "comparison_sha256": _digest(comparison),
        "tasks": rows,
    }

    def save() -> None:
        comparison["max_context_tokens"] = (
            model.max_context_tokens if isinstance(model, ModelBackend) else None
        )
        if local and isinstance(model, VLLMModel):
            from .snapshots import _encoding_condition

            runtime = model.describe()
            record["snapshot_details"] = runtime
            comparison["encoding"] = _encoding_condition(model.encoding.describe())
            comparison["inference"] = runtime.get("inference_conditions")
        record["comparison_sha256"] = _digest(comparison)
        record["summary"] = _summary(rows)
        record["cost"] = _cost(model)
        _write(path, record)

    save()  # All denominators are durable before any paid probe or task starts.
    monitor = None
    active_row = None
    try:
        if monitoring is not None:
            monitor = RunMonitor(
                run_dir,
                monitoring,
                phase="dev-eval",
                metadata={
                    "snapshot": snapshot,
                    "evidence_kind": evidence_kind,
                    "training_run": str(training_run.resolve())
                    if training_run
                    else None,
                    "comparison_sha256": record["comparison_sha256"],
                    "encoding": model.encoding.identity,
                    "fingerprints": fingerprints(condition),
                    "sampling": configuration["model"]["sampling"],
                    "code_revision": record["code_revision"],
                    "data_source": {
                        k: plan[k]
                        for k in ("source", "conversion", "split", "task_ids")
                    },
                },
                require_online=evidence_kind == "experimental",
            )
        for task, row in zip(tasks, rows, strict=True):
            active_row = row
            row["outcome"] = "running"
            save()

            def progress(episode: EpisodeResult, row: dict[str, Any] = row) -> None:
                row["run"] = str(episode.run_dir.relative_to(run_dir))
                save()

            result = run_experiment(
                task=task,
                model=model,
                config=condition,
                output_dir=run_dir / "episodes",
                on_update=progress,
                assess_first_implementation=True,
            )
            episode = json.loads((result.run_dir / "run.json").read_text())
            row.update(
                outcome=_outcome(episode),
                status=result.status,
                reason=result.reason,
                episode_status=result.episode_status,
                budget_kind=result.budget_kind,
                grade=episode["grade"],
                first_implementation=episode["first_implementation"],
                usage=episode["usage"],
                episode_sha256=hashlib.sha256(
                    (result.run_dir / "run.json").read_bytes()
                ).hexdigest(),
            )
            if (
                local
                and isinstance(model, VLLMModel)
                and model.describe().get("identity_verified") is not True
            ):
                row["outcome"] = "infrastructure_error"
            save()
            if monitor is not None:
                monitor.update_metadata(
                    {"comparison_sha256": record["comparison_sha256"]}
                )
                monitor.log(f"task-{task.task_id}", evaluation_metrics(record))
            print(f"{task.task_id}: {row['outcome']}", flush=True)
            if (
                row["outcome"] in {"infrastructure_error", "interrupted"}
                or result.budget_kind == "spend"
                or (
                    result.first_implementation is not None
                    and result.first_implementation.status
                    in {"infrastructure_error", "interrupted"}
                )
            ):
                break
    except (OSError, KeyboardInterrupt) as error:
        if active_row is not None:
            active_row.update(
                outcome="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "infrastructure_error",
                reason=str(error),
            )
        record["error_type"] = type(error).__name__
    finally:
        summary = _summary(rows)
        record["status"] = (
            "incomplete"
            if summary["missing_task_ids"] or summary["first_grading_missing_task_ids"]
            else "complete"
        )
        record["finished_at"] = datetime.now(UTC).isoformat()
        record["snapshot_details"] = (
            model.describe() if isinstance(model, ModelBackend) else {}
        )
        save()
        if monitor is not None:
            monitor.update_metadata({"comparison_sha256": record["comparison_sha256"]})
            monitor.log("finished", evaluation_metrics(record))
            monitor.finish()
    return path


def select_sft(run_dir: Path, baseline: Path, evaluations: list[Path]) -> Path:
    """Select by fixed-denominator dev success; retain separate restart artifacts."""
    training = json.loads((run_dir / "training.json").read_text())
    control = json.loads(baseline.read_text())
    local = control.get("snapshot_details", {}).get("backend") == "vllm"
    if local:
        _require_verified_local(control, baseline)
        details = control["snapshot_details"]
        if (
            details["snapshot_kind"] != "base"
            or details["base_model"] != training["metadata"]["base_model"]
        ):
            raise ValueError("SFT baseline must load the original local base snapshot")
        base_snapshot = f"vllm/{details['base_model']}@{details['model_revision']}"
    else:
        base_snapshot = f"pytrio/{training['metadata']['base_model']}@unversioned-base"
    if control.get("snapshot") != base_snapshot or any(
        path.resolve() == baseline.resolve() for path in evaluations
    ):
        raise ValueError(
            "SFT baseline must be the original instruction model evaluation"
        )
    if not evaluations or len(set(evaluations)) != len(evaluations):
        raise ValueError("Select from distinct development evaluations")
    checkpoints = {
        (
            c["sampler"]["path"]
            if local
            else f"pytrio/{training['metadata']['base_model']}@{c['sampler']['path']}"
        ): c
        for c in training["checkpoints"]
        if c["sampler"]["kind"] == "sampler_weights"
        and c["training_state"]["kind"] == "training_state"
    }
    records = [
        (path, json.loads(path.read_text())) for path in [baseline, *evaluations]
    ]
    for path, record in records:
        if local:
            _require_verified_local(record, path)
        if record["plan"]["split"] != "dev":
            raise ValueError("SFT selection requires development data only")
        if record["status"] != "complete" or record["evidence_kind"] != "experimental":
            raise ValueError("SFT selection requires complete experimental evaluations")
        if (
            record["comparison_sha256"] != _digest(record["comparison"])
            or record["comparison_sha256"] != control["comparison_sha256"]
            or record["plan"]["task_ids"] != control["plan"]["task_ids"]
            or record["summary"]["planned_tasks"] != len(control["plan"]["task_ids"])
        ):
            raise ValueError(
                "Changed comparison conditions require a matching baseline"
            )
    candidates = []
    seen = set()
    for path, record in records[1:]:
        snapshot = record["snapshot"]
        if local:
            adapter = record["snapshot_details"].get("adapter")
            if (
                not isinstance(adapter, dict)
                or adapter.get("training_record_sha256")
                != hashlib.sha256((run_dir / "training.json").read_bytes()).hexdigest()
            ):
                raise ValueError(
                    "Local evaluation must identify the saved SFT training record"
                )
            snapshot = adapter["checkpoint"]["path"]
            if (
                snapshot in checkpoints
                and adapter["optimizer_step"] != checkpoints[snapshot]["optimizer_step"]
            ):
                raise ValueError("Local evaluation checkpoint step mismatch")
        if snapshot not in checkpoints or snapshot in seen:
            raise ValueError(
                "Evaluation must load a distinct saved SFT sampler snapshot"
            )
        seen.add(snapshot)
        candidates.append(
            {
                "evaluation": str(path.resolve()),
                "evaluation_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "passed_tasks": record["summary"]["passed_tasks"],
                "planned_tasks": record["summary"]["planned_tasks"],
                **checkpoints[snapshot],
            }
        )
    selected = max(candidates, key=lambda c: (c["passed_tasks"], -c["optimizer_step"]))
    selection = {
        "rule": (
            "highest fixed-denominator development pass rate; ties prefer earlier step"
        ),
        "comparison_sha256": control["comparison_sha256"],
        "baseline": str(baseline.resolve()),
        "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "baseline_passed_tasks": control["summary"]["passed_tasks"],
        "candidates": candidates,
        "optimizer_step": selected["optimizer_step"],
        "student_snapshot": selected["sampler"]["path"],
        "frozen_teacher_snapshot": selected["sampler"]["path"],
        "sft_training_state": selected["training_state"],
        "optimizer_transition": "reset-for-opsd",
        "opsd_initialization": (
            "load selected SFT weights into a new optimizer; freeze teacher"
        ),
        "teacher_scoring_verified": False,
    }
    destination = run_dir / "selection.json"
    if destination.exists() and json.loads(destination.read_text()) != selection:
        raise ValueError(
            "Selection already exists; preserve it and use a new experiment"
        )
    _write(destination, selection)
    monitor_path = run_dir / "monitor.json"
    if monitor_path.exists():
        monitor = RunMonitor.resume(run_dir)
        monitor.log(
            "selection",
            {
                "checkpoint/selected_sampler": selection["student_snapshot"],
                "checkpoint/selected_optimizer_step": selection["optimizer_step"],
                "checkpoint/selected_training_state": selected["training_state"][
                    "path"
                ],
                "eval/selected_passed_tasks": selected["passed_tasks"],
                "eval/planned_tasks": selected["planned_tasks"],
                "eval/completed_tasks": selected["planned_tasks"],
            },
        )
        monitor.finish()
    return destination


def _require_verified_local(record: dict[str, Any], path: Path) -> None:
    details = record.get("snapshot_details", {})
    if (
        details.get("backend") != "vllm"
        or details.get("development_only") is not False
        or details.get("identity_verified") is not True
        or details.get("identity") != record.get("snapshot")
        or not details.get("snapshot_manifest_sha256")
        or details.get("capability_probe", {}).get("passed") is not True
        or not details.get("inference_conditions")
        or not details.get("load_proof")
        or record.get("status") != "complete"
        or not record.get("tasks")
    ):
        raise ValueError(
            "Local experimental evidence requires a verified snapshot identity"
        )
    for row in record.get("tasks", []):
        episode_path = path.parent / row["run"] / "run.json"
        if (
            hashlib.sha256(episode_path.read_bytes()).hexdigest()
            != row["episode_sha256"]
        ):
            raise ValueError("Local evaluation episode hash mismatch")
        episode = json.loads(episode_path.read_text())
        if (
            episode["policy"] != record["snapshot"]
            or episode["policy_details"].get("identity_verified") is not True
            or episode["policy_details"].get("snapshot_manifest_sha256")
            != details["snapshot_manifest_sha256"]
        ):
            raise ValueError(
                "Local evaluation episode requires a verified snapshot identity"
            )
