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
    return {
        "ledger": spend.get("ledger"),
        "request_ids": [entry["id"] for entry in entries],
        "charged_upper_bound_cny": round(
            sum(entry["charged_cny"] for entry in entries), 6
        ),
        "pending_upper_bound_cny": round(
            sum(entry["upper_cny"] for entry in entries if entry["state"] == "pending"),
            6,
        ),
        "actual_cny": round(sum(entry["charged_cny"] for entry in entries), 6)
        if entries and all(entry["billing_basis"] == "confirmed" for entry in entries)
        else None,
        "billing_note": "Unreconciled ceilings are not verified actual charges.",
    }


def evaluate(
    *,
    plan_path: Path,
    snapshot: str,
    evidence_kind: str,
    condition: ExperimentConfig,
    output_dir: Path,
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
    if evidence_kind == "experimental" and (
        details.get("backend") != "pytrio"
        or details.get("development_only") is not False
    ):
        raise ValueError(
            "Experimental evaluation requires PyTRIO; use --evidence-kind test-double"
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
        record["comparison_sha256"] = _digest(comparison)
        record["summary"] = _summary(rows)
        record["cost"] = _cost(model)
        _write(path, record)

    save()  # All denominators are durable before any paid probe or task starts.
    try:
        for task, row in zip(tasks, rows, strict=True):
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
            save()
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
        row.update(
            outcome="interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "infrastructure_error",
            reason=str(error),
        )
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
    return path
