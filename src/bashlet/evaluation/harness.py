"""Fixed development cohorts, durable task states and post-episode reporting."""

import hashlib
import json
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DirectCodeConfig, ExperimentConfig, config_record, load_model
from ..experiment import run_experiment, serialize_task, task_from_record
from ..models import EpisodeResult, Model, ModelBackend, Task
from ..monitoring import MonitorConfig, RunMonitor, code_revision, evaluation_metrics
from ..protocol import fingerprints
from ..spend import summarize_cost
from ..tasks import task_family, task_path
from ..vllm import VLLMModel, configured_request_policy

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


def effective_comparison(record: dict[str, Any]) -> dict[str, Any]:
    """Recover omitted legacy HTTP defaults from the saved constructor config."""
    comparison: dict[str, Any] = dict(record["comparison"])
    comparison["request_policy"] = configured_request_policy(record["config"]["model"])
    return comparison


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
            json.loads(task_path(manifest_path, task_id, "dev").read_text())
        )
        for task_id in task_ids
    ]
    families = [task_family(task) for task in tasks if task.origin is not None]
    if len(families) != len(set(families)):
        raise ValueError(
            "Evaluation requires independent task families; a family variant "
            "cannot add an independent task to the denominator"
        )
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
        "by_task_type": {
            task_type: {
                "planned_tasks": len(group),
                "passed_tasks": sum(row["outcome"] == "passed" for row in group),
                "final_pass_rate": sum(row["outcome"] == "passed" for row in group)
                / len(group),
                "outcomes": dict(Counter(row["outcome"] for row in group)),
                "protocol_errors": sum(row.get("protocol_errors", 0) for row in group),
                "command_problems": dict(
                    sum(
                        (Counter(row.get("command_problems", {})) for row in group),
                        Counter(),
                    )
                ),
                "submitted_tasks": sum(
                    row.get("episode_status") == "submitted" for row in group
                ),
                "terminations": dict(
                    Counter(row.get("episode_status", "not_started") for row in group)
                ),
                "typical_trajectories": {
                    outcome: next(
                        row["run"] for row in group if row["outcome"] == outcome
                    )
                    for outcome in dict.fromkeys(row["outcome"] for row in group)
                    if outcome != "not_started"
                },
            }
            for task_type in dict.fromkeys(
                row.get("task_type") or "unspecified" for row in rows
            )
            if (
                group := [
                    row
                    for row in rows
                    if (row.get("task_type") or "unspecified") == task_type
                ]
            )
        },
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


def _episode_row(row: dict[str, Any], path: Path, episode: dict[str, Any]) -> None:
    if episode["task_sha256"] != row["task_sha256"]:
        raise ValueError("Saved evaluation episode belongs to another task")
    row.update(
        outcome=_outcome(episode),
        status=episode["status"],
        reason=episode["reason"],
        episode_status=episode["episode_status"],
        budget_kind=episode.get("budget_kind"),
        grade=episode["grade"],
        first_implementation=episode.get("first_implementation") or {},
        usage=episode["usage"],
        episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    row["protocol_errors"] = sum(
        step["action"]["kind"] == "invalid" for step in episode["trajectory"]
    )
    row["command_problems"] = dict(
        Counter(
            step["observation"]["status"]
            for step in episode["trajectory"]
            if step.get("observation") and step["observation"]["status"] != "ok"
        )
    )
    row["diagnostics"] = {
        "verifications": sum(step["verification"] for step in episode["trajectory"]),
        "repair_attempts": len(episode["repair_attempts"]),
        "interaction_steps": len(episode["trajectory"]),
        "counts_are_not_task_benefit": True,
    }


def evaluate(
    *,
    plan_path: Path,
    snapshot: str,
    evidence_kind: str,
    condition: ExperimentConfig,
    output_dir: Path,
    monitoring: MonitorConfig | None = None,
    training_run: Path | None = None,
    direct_code: DirectCodeConfig | None = None,
    run_dir: Path | None = None,
    model: Model | None = None,
    phase: str | None = None,
    resume: bool = False,
    resume_running_episodes: bool = False,
    resume_elapsed_seconds: dict[str, float] | None = None,
) -> Path:
    """CLI orchestration; every planned attempt uses the existing experiment seam."""
    plan, tasks = _tasks(plan_path)
    track = "coding" if direct_code is not None else "harness"
    model = model if model is not None else load_model(condition.model)
    if model.identity != snapshot:
        raise ValueError(
            f"Snapshot mismatch: expected {snapshot!r}, got {model.identity!r}"
        )
    details: dict[str, Any] = (
        model.describe() if isinstance(model, ModelBackend) else {}
    )
    local = isinstance(model, VLLMModel) and details.get("development_only") is False
    if local and resume:
        assert isinstance(model, VLLMModel)
        model.probe()
        details = model.describe()
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
    run_dir = (run_dir or output_dir.resolve() / uuid.uuid4().hex).resolve()
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=resume)
    path = run_dir / "evaluation.json"
    configuration = config_record(condition)
    rows = [
        {
            "task_id": task.task_id,
            "task_sha256": hashlib.sha256(serialize_task(task).encode()).hexdigest(),
            "task_origin": asdict(task.origin) if task.origin else None,
            "task_assignment": asdict(task.assignment) if task.assignment else None,
            "task_type": task.task_type or "short_program",
            "variant_id": task.variant_id,
            "feedback_start_id": task.feedback_start_id,
            "environment_variant_id": task.environment_variant_id,
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
        "track": track,
        "request_policy": configured_request_policy(configuration["model"]),
    }
    if direct_code is not None:
        comparison["direct_code"] = asdict(direct_code)
        del comparison["protocol"]
    if local:
        from ..snapshots import encoding_condition

        comparison["encoding"] = encoding_condition(model.encoding.describe())
        if resume:
            assert isinstance(model, VLLMModel)
            comparison["inference"] = details.get("inference_conditions")
            comparison["max_context_tokens"] = model.max_context_tokens
    record: dict[str, Any] = {
        "version": VERSION,
        "track": track,
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
    previous_requests: set[str] = set()
    resumed_episodes: dict[str, Path] = {}
    if resume and path.exists():
        previous = json.loads(path.read_text())
        expected = json.loads(json.dumps(record))
        if any(
            previous.get(key) != expected[key]
            for key in (
                "version",
                "track",
                "snapshot",
                "plan",
                "config",
                "evidence_kind",
            )
        ) or any(
            previous["comparison"].get(key) != value
            for key, value in expected["comparison"].items()
        ):
            raise ValueError(
                "Evaluation resume must preserve snapshot, tasks and conditions"
            )
        if previous["status"] == "complete":
            return path
        record, rows, comparison = previous, previous["tasks"], previous["comparison"]
        previous_requests = set(previous["cost"]["request_ids"])
        for row in rows:
            if row["outcome"] != "running":
                continue
            episode_path = run_dir / row["run"] / "run.json" if row["run"] else None
            episode = (
                json.loads(episode_path.read_text())
                if episode_path and episode_path.exists()
                else None
            )
            if (
                resume_running_episodes
                and episode
                and episode["policy"] == snapshot
                and episode["status"] == "running"
                and direct_code is None
            ):
                assert episode_path is not None
                resumed_episodes[row["task_id"]] = episode_path.parent
            elif (
                episode
                and episode["policy"] == snapshot
                and episode["status"] not in {"running", "submitted"}
            ):
                assert episode_path is not None
                _episode_row(row, episode_path, episode)
            else:
                row.update(
                    outcome="interrupted",
                    reason="Unfinished attempt preserved on resume",
                )
        record["status"] = "running"
        record.pop("finished_at", None)
    elif resume and any(run_dir.iterdir()):
        raise ValueError("Evaluation recovery requires its existing record")

    def save() -> None:
        comparison["max_context_tokens"] = (
            model.max_context_tokens if isinstance(model, ModelBackend) else None
        )
        if local and isinstance(model, VLLMModel):
            from ..snapshots import encoding_condition

            runtime = model.describe()
            record["snapshot_details"] = runtime
            comparison["encoding"] = encoding_condition(model.encoding.describe())
            comparison["inference"] = runtime.get("inference_conditions")
        record["comparison_sha256"] = _digest(comparison)
        record["summary"] = _summary(rows)
        record["cost"] = _cost(model)
        if previous_requests and isinstance(model, ModelBackend):
            spend_details: dict[str, Any] = model.describe()
            spend = spend_details.get("spend", {})
            assert isinstance(spend, dict)
            selected_ids = previous_requests | set(record["cost"]["request_ids"])
            record["cost"] = summarize_cost(
                spend, [e for e in spend.get("entries", []) if e["id"] in selected_ids]
            )
        _write(path, record)

    save()  # All denominators are durable before any paid probe or task starts.
    monitor = None
    active_row = None
    try:
        if monitoring is not None:
            monitor = RunMonitor(
                run_dir,
                monitoring,
                phase=phase
                or ("coding-eval" if direct_code is not None else "dev-eval"),
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
                require_online=evidence_kind == "experimental" and not local,
            )
        for task, row in zip(tasks, rows, strict=True):
            if row["outcome"] != "not_started" and task.task_id not in resumed_episodes:
                continue
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
                assess_first_implementation=direct_code is None,
                direct_code=direct_code,
                resume_run_dir=resumed_episodes.get(task.task_id),
                resume_elapsed_seconds=(resume_elapsed_seconds or {}).get(task.task_id),
            )
            episode = json.loads((result.run_dir / "run.json").read_text())
            _episode_row(row, result.run_dir / "run.json", episode)
            if (
                local
                and isinstance(model, VLLMModel)
                and model.describe().get("identity_verified") is not True
            ):
                row["outcome"] = "infrastructure_error"
            save()
            active_row = None
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
            completed_tasks = len(rows) - len(summary["missing_task_ids"])
            monitor.log(
                f"finished-{completed_tasks}-{record['status']}",
                evaluation_metrics(record),
            )
            monitor.finish()
    return path


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
