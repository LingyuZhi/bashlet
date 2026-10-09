"""Frozen dual-track plans, three-seed extension and durable result reuse."""

import hashlib
import json
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import (
    DirectCodeConfig,
    ExperimentConfig,
    _require_positive_numbers,
    config_record,
)
from ..experiment import serialize_task
from ..models import FunctionGrader, InteractionBudget, SamplingParams
from ..monitoring import MonitorConfig, code_revision
from ..tasks import task_family
from ..vllm import configured_request_policy
from .harness import _digest, _tasks, _write, evaluate

VERSION = "dual-track-acceptance-v1"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_policy(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    policy = json.loads(path.read_text())
    required = {
        "version",
        "name",
        "harness_plan",
        "coding_plan",
        "variant_plan",
        "harness_counts",
        "coding_count",
        "seeds",
        "coding",
        "coding_sampling",
        "coding_budget",
        "rules",
    }
    if (
        not isinstance(policy, dict)
        or set(policy) != required
        or policy["version"] != VERSION
    ):
        raise ValueError("Unknown dual-track acceptance policy")
    seeds = policy["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != 3
    ):
        raise ValueError("Freeze exactly three distinct nonnegative sampling seeds")
    DirectCodeConfig(**policy["coding"])
    rules = policy["rules"]
    if (
        rules["confirmation_seeds"] != 3
        or rules["repeated_seeds"] != 2
        or not 0 < rules["regression_sign_test_alpha"] <= 0.05
        or type(rules["scene_min_gained_families"]) is not int
        or rules["scene_min_gained_families"] < 1
        or rules["scene_task_type"] != "workspace"
    ):
        raise ValueError("Invalid frozen confirmation or comparison rules")
    inputs = {}
    paths = {}
    family_sets: dict[str, set[str]] = {}
    for track, key in (
        ("harness", "harness_plan"),
        ("coding", "coding_plan"),
        ("variant", "variant_plan"),
    ):
        if policy[key] is None and track == "variant":
            continue
        plan_path = (path.parent / policy[key]).resolve()
        plan, tasks = _tasks(plan_path)
        family_sets[track] = {task_family(task) for task in tasks if task.origin}
        if len(family_sets[track]) != len(tasks):
            raise ValueError("Acceptance requires one independent family per track row")
        if (
            track == "harness"
            and dict(Counter(task.task_type or "short_program" for task in tasks))
            != policy["harness_counts"]
        ):
            raise ValueError("Frozen Harness type denominators differ")
        if track == "coding" and (
            len(tasks) != policy["coding_count"]
            or any(
                task.origin is None
                or task.origin.dataset != "google-research/mbpp"
                or not isinstance(task.grader, FunctionGrader)
                for task in tasks
            )
        ):
            raise ValueError(
                "Coding track requires the frozen verified MBPP development list"
            )
        paths[track] = plan_path
        inputs[track] = {
            "plan_sha256": file_hash(plan_path),
            "manifest_sha256": plan["manifest_sha256"],
            "task_ids": plan["task_ids"],
            "families": sorted(family_sets[track]),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "family_id": task_family(task),
                    "task_sha256": _digest(json.loads(serialize_task(task))),
                }
                for task in tasks
                if task.origin
            ],
        }
    if not family_sets.get("variant", set()) <= family_sets["harness"]:
        raise ValueError(
            "Environment variants must retain a frozen Harness task family"
        )
    return {
        "policy": policy,
        "inputs": inputs,
        "overlap_family_ids": sorted(family_sets["harness"] & family_sets["coding"]),
    }, paths


def coding_condition(
    condition: ExperimentConfig, policy: dict[str, Any]
) -> ExperimentConfig:
    resolved = replace(
        condition,
        model=replace(
            condition.model, sampling=SamplingParams(**policy["coding_sampling"])
        ),
        budget=InteractionBudget(**policy["coding_budget"]),
    )
    _require_positive_numbers(resolved)
    return resolved


def condition_identity(condition: ExperimentConfig) -> dict[str, Any]:
    record = config_record(condition)
    return {key: record[key] for key in ("protocol", "environment", "budget")} | {
        "sampling": {
            key: value
            for key, value in record["model"]["sampling"].items()
            if key != "seed"
        },
        "request_policy": configured_request_policy(record["model"]),
    }


def effective_conditions(record: dict[str, Any]) -> dict[str, Any]:
    conditions = {track: dict(values) for track, values in record["conditions"].items()}
    for track, values in conditions.items():
        if "request_policy" not in values:
            references = record["runs"].get(track, {})
            values["request_policy"] = {}
            for reference in references.values():
                try:
                    values["request_policy"] = configured_request_policy(
                        load_evaluation(reference)[1]["config"]["model"]
                    )
                    break
                except (OSError, ValueError, KeyError, TypeError):
                    continue
    return conditions


def load_evaluation(reference: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = Path(reference["path"])
    if file_hash(path) != reference["sha256"]:
        raise ValueError(
            "Evaluation evidence hash differs; completed results cannot be replaced"
        )
    record = json.loads(path.read_text())
    if record["comparison_sha256"] != _digest(record["comparison"]):
        raise ValueError("Evaluation comparison hash differs")
    if [row["task_id"] for row in record["tasks"]] != record["plan"]["task_ids"]:
        raise ValueError("Evaluation rows differ from the fixed denominator")
    return path, record


def suite_summary(record: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "overlap_family_ids": record["frozen"]["overlap_family_ids"],
        "seeds_are_independent_tasks": False,
        "variants_add_task_families": False,
    }
    for track, input_record in record["frozen"]["inputs"].items():
        evaluations = [
            load_evaluation(ref)[1] for ref in record["runs"][track].values()
        ]
        rows = [row for item in evaluations for row in item["tasks"]]
        summary[track] = {
            "task_families": len(input_record["families"]),
            "planned_attempts": len(input_record["task_ids"]) * len(record["seeds"]),
            "attempts": len(rows),
            "passed_attempts": sum(row["outcome"] == "passed" for row in rows),
            "by_seed": {
                str(item["config"]["model"]["sampling"]["seed"]): item["summary"]
                for item in evaluations
            },
        }
    return summary


def _seal_active(record: dict[str, Any]) -> None:
    active = record.get("active_attempt")
    if active is None:
        return
    path = Path(active["path"])
    if path.is_file():
        record["runs"][active["track"]][str(active["seed"])] = {
            "path": str(path),
            "sha256": file_hash(path),
        }
    else:
        record["attempt_gap"] = active
    record["active_attempt"] = None


def _complete_evidence(evaluation: dict[str, Any]) -> bool:
    return evaluation["status"] == "complete" and not any(
        row["outcome"] == "model_error" for row in evaluation["tasks"]
    )


def evaluate_acceptance(
    policy_path: Path,
    *,
    snapshot: str,
    evidence_kind: str,
    condition: ExperimentConfig,
    output_dir: Path,
    stage: str = "screening",
    resume: Path | None = None,
    monitoring: MonitorConfig | None = None,
    training_run: Path | None = None,
) -> Path:
    """Freeze before execution and extend each (snapshot, track, seed) at most once."""
    frozen, plans = read_policy(policy_path)
    policy = frozen["policy"]
    if stage not in {"screening", "confirmation"}:
        raise ValueError("Unknown acceptance stage")
    conditions = {
        "harness": condition_identity(condition),
        "coding": condition_identity(coding_condition(condition, policy)),
    }
    if resume is None:
        root = output_dir.resolve()
        root.mkdir(parents=True, mode=0o700, exist_ok=False)
        destination = root / "acceptance.json"
        record: dict[str, Any] = {
            "version": VERSION,
            "snapshot": snapshot,
            "evidence_kind": evidence_kind,
            "started_at": datetime.now(UTC).isoformat(),
            "code_revision": code_revision(),
            "frozen": frozen,
            "frozen_sha256": _digest(frozen),
            "conditions": conditions,
            "runs": {track: {} for track in plans},
            "status": "running",
            "training_run": str(training_run.resolve()) if training_run else None,
        }
    else:
        destination = resume.resolve()
        root = destination.parent
        record = json.loads(destination.read_text())
        if (
            record["version"] != VERSION
            or record["snapshot"] != snapshot
            or record["evidence_kind"] != evidence_kind
            or record["frozen_sha256"] != _digest(frozen)
            or _digest(effective_conditions(record)) != _digest(conditions)
        ):
            raise ValueError(
                "Resume requires the same snapshot, policy and comparison conditions"
            )
        _seal_active(record)
        _write(destination, record)
        if record.get("attempt_gap"):
            raise ValueError("An unfinished attempt has no evidence; retain the gap")
        for track, refs in record["runs"].items():
            for seed, reference in refs.items():
                _, prior = load_evaluation(reference)
                if not _complete_evidence(prior):
                    raise ValueError(
                        "Incomplete attempts cannot be retried as extra evidence"
                    )
                if (
                    prior["snapshot"] != snapshot
                    or prior["config"]["model"]["sampling"]["seed"] != int(seed)
                    or prior["plan"]["task_ids"] != frozen["inputs"][track]["task_ids"]
                ):
                    raise ValueError(
                        "Reused evidence differs from its frozen track and seed"
                    )
    seeds = policy["seeds"] if stage == "confirmation" else policy["seeds"][:1]
    # Confirmation cannot be shrunk by later requesting screening.
    record["seeds"] = list(dict.fromkeys([*record.get("seeds", []), *seeds]))
    record["stage"] = "confirmation" if len(record["seeds"]) == 3 else "screening"
    record["status"] = "running"
    _write(destination, record)
    try:
        for seed in record["seeds"]:
            for track, plan in plans.items():
                if str(seed) in record["runs"][track]:
                    continue
                resolved = (
                    coding_condition(condition, policy)
                    if track == "coding"
                    else condition
                )
                resolved = replace(
                    resolved,
                    model=replace(
                        resolved.model,
                        sampling=replace(resolved.model.sampling, seed=seed),
                    ),
                )
                episode_run_dir = root / track / f"seed-{seed}"
                record["active_attempt"] = {
                    "track": track,
                    "seed": seed,
                    "path": str(episode_run_dir / "evaluation.json"),
                }
                _write(destination, record)
                path = evaluate(
                    plan_path=plan,
                    snapshot=snapshot,
                    evidence_kind=evidence_kind,
                    condition=resolved,
                    output_dir=root / track,
                    monitoring=monitoring,
                    training_run=training_run,
                    direct_code=DirectCodeConfig(**policy["coding"])
                    if track == "coding"
                    else None,
                    run_dir=episode_run_dir,
                )
                record["runs"][track][str(seed)] = {
                    "path": str(path),
                    "sha256": file_hash(path),
                }
                record["active_attempt"] = None
                _write(destination, record)
                if not _complete_evidence(json.loads(path.read_text())):
                    return destination
    finally:
        _seal_active(record)
        record["summary"] = suite_summary(record)
        record["status"] = (
            "complete"
            if not record.get("attempt_gap")
            and all(
                len(refs) == len(record["seeds"])
                and all(
                    _complete_evidence(load_evaluation(ref)[1]) for ref in refs.values()
                )
                for refs in record["runs"].values()
            )
            else "incomplete"
        )
        record["finished_at"] = datetime.now(UTC).isoformat()
        _write(destination, record)
    return destination
