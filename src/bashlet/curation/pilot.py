"""Paired protocol reports from immutable runs and explicit reply reviews."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import _plan, _save, _sha

_PROTOCOLS = ("bash-block-v3", "bash-block-v4")
_COUNTS = (
    "planned_attempts",
    "started_attempts",
    "recorded_attempts",
    "unattempted",
    "valid_actions",
    "actions",
    "format_errors",
    "invented_observation_replies",
    "reviewed_replies",
    "submissions",
    "grading_passes",
    "quality_candidates",
    "budget_exhausted",
    "context_exhausted",
    "history_over_limit",
    "generated_tokens",
    "model_input_tokens",
)


def _condition(config: dict[str, Any]) -> dict[str, Any]:
    return config | {
        "protocol": {
            key: value
            for key, value in config["protocol"].items()
            if key not in ("name", "system_prompt")
        }
    }


def _metrics(
    root: Path,
    attempt: dict[str, Any],
    decision: dict[str, Any],
    selected: dict[str, Any],
    collection: dict[str, Any],
    policy: str,
) -> dict[str, Any]:
    counts = dict.fromkeys(_COUNTS, 0)
    counts.update(planned_attempts=1, started_attempts=1)
    if "run_dir" not in attempt:
        if decision["actions"] or decision["accept"]:
            raise ValueError("An unfinished attempt has no reviewable candidate")
        return counts | {"status": attempt["status"], "reason": "No sealed run"}
    run_dir = (root / attempt["run_dir"]).resolve()
    if not run_dir.is_relative_to(root / "episodes"):
        raise ValueError("Attempt evidence is outside the collection")
    run_path = run_dir / "run.json"
    if _sha(run_path) != attempt["run_sha256"]:
        raise ValueError("Attempt evidence hash differs")
    record = json.loads(run_path.read_text())
    if (
        record["config"] != collection["config"]
        or record["policy"] != policy
        or record["task_id"] != selected["task_id"]
        or record["task_sha256"] != selected["task_sha256"]
        or record["status"] != attempt["status"]
        or record["passed"] != attempt["passed"]
        or len(decision["actions"]) != len(record["trajectory"])
    ):
        raise ValueError("Attempt or review differs from its frozen provenance")
    counts.update(
        recorded_attempts=1,
        submissions=int(record["episode_status"] == "submitted"),
        grading_passes=int(record["passed"]),
        quality_candidates=int(decision["accept"]),
        budget_exhausted=int(record["status"] == "budget_exhausted"),
        context_exhausted=int(record["status"] == "context_exhausted"),
        generated_tokens=record["usage"]["generated_tokens"],
    )
    for step, action_review in zip(
        record["trajectory"], decision["actions"], strict=True
    ):
        call = step["model_call"]
        if call is None:
            raise ValueError("A paired model action must retain its generation")
        if call["generation"]["source"] != "sampled" or call["policy"] != policy:
            raise ValueError("Replay cannot count as a paired model attempt")
        review = action_review["observation_review"]
        if (
            type(review["invented"]) is not bool
            or not isinstance(review["reason"], str)
            or not review["reason"].strip()
            or review["reply_sha256"]
            != hashlib.sha256(call["text"].encode()).hexdigest()
        ):
            raise ValueError("Observation review must bind the original reply")
        valid = "command" in step["action"]
        counts["actions"] += 1
        counts["valid_actions"] += int(valid)
        counts["format_errors"] += int(
            not valid and call["generation"]["stop_reason"] != "context"
        )
        counts["reviewed_replies"] += 1
        counts["invented_observation_replies"] += int(review["invented"])
        counts["model_input_tokens"] += call["context_offset"]
    return counts | {
        "status": record["status"],
        "reason": record["reason"],
        "run": str(run_path),
        "run_sha256": attempt["run_sha256"],
        "history_tokens": len(record["context_tokens"]),
        "encoding": record["encoding"],
        "inference_conditions": record["policy_details"].get("inference_conditions"),
    }


def report_collection_pilot(
    plan_path: Path,
    collection_paths: Sequence[Path],
    review_paths: Sequence[Path],
    output_dir: Path,
) -> Path:
    """Keep both planned denominators, including incomplete collections.

    This entry point consumes evidence only. It does not sample, replay, infer
    quality from tags, or replace failed attempts. Demonstration eligibility is
    established separately by freeze_demonstrations and the training loader.
    """
    plan = _plan(plan_path)
    if len(collection_paths) != 2 or len(review_paths) != 2:
        raise ValueError("Supply one collection and review for each protocol")
    paired: dict[str, dict[str, Any]] = {
        item["task_id"]: dict(item) for item in plan["tasks"]
    }
    protocols: dict[str, Any] = {}
    artifacts: dict[str, str] = {str(plan_path.resolve()): _sha(plan_path)}
    conditions = []
    execution_conditions = []
    for path, review_path in zip(collection_paths, review_paths, strict=True):
        root = path.resolve().parent
        collection = json.loads(path.read_text())
        review = json.loads(review_path.read_text())
        name = collection["config"]["protocol"]["name"]
        if name not in _PROTOCOLS or name in protocols:
            raise ValueError("Exactly one v3 and one v4 collection is required")
        if (
            collection["schema_version"] != 1
            or collection["plan_sha256"] != _sha(plan_path)
            or _sha(root / "plan.json") != _sha(plan_path)
            or review["schema_version"] != 1
            or review["collection_sha256"] != _sha(path)
            or len(collection["attempts"]) > len(plan["tasks"])
            or len(review["attempts"]) != len(collection["attempts"])
            or (
                collection["status"] == "complete"
                and len(collection["attempts"]) != len(plan["tasks"])
            )
        ):
            raise ValueError("Collection or review differs from its frozen plan")
        conditions.append(_condition(collection["config"]))
        totals = Counter(dict.fromkeys(_COUNTS, 0))
        per_type: dict[str, Counter[str]] = {}
        statuses = Counter[str]()
        for index, selected in enumerate(plan["tasks"]):
            if index >= len(collection["attempts"]):
                row: dict[str, Any] = dict.fromkeys(_COUNTS, 0) | {
                    "planned_attempts": 1,
                    "unattempted": 1,
                    "status": "unattempted",
                    "reason": "The scheduled attempt was not started",
                }
            else:
                attempt = collection["attempts"][index]
                decision = review["attempts"][index]
                if (
                    attempt["attempt"] != index + 1
                    or decision["attempt"] != index + 1
                    or any(attempt[key] != value for key, value in selected.items())
                    or type(decision["accept"]) is not bool
                    or not isinstance(decision["reason"], str)
                    or not decision["reason"].strip()
                ):
                    raise ValueError("Review each scheduled family exactly once")
                row = _metrics(
                    root,
                    attempt,
                    decision,
                    selected,
                    collection,
                    plan["expected_policy"],
                )
            row["history_over_limit"] = int(
                row.get("history_tokens", 0) > plan["max_context_tokens"]
            )
            numeric = {key: row[key] for key in _COUNTS}
            totals.update(numeric)
            per_type.setdefault(selected["task_type"], Counter()).update(numeric)
            statuses.update([row["status"]])
            paired[selected["task_id"]][name] = row
            if "run" in row:
                artifacts[row["run"]] = row["run_sha256"]
                execution_conditions.append(
                    (row["encoding"], row["inference_conditions"])
                )
        protocols[name] = dict(totals) | {
            "collection_status": collection["status"],
            "per_type": per_type,
            "terminal_statuses": statuses,
            "snapshot_verified": collection.get("policy_details", {}).get(
                "identity_verified", False
            ),
        }
        artifacts.update(
            {
                str(path.resolve()): _sha(path),
                str(review_path.resolve()): _sha(review_path),
            }
        )
    if conditions[0] != conditions[1] or any(
        item != execution_conditions[0] for item in execution_conditions[1:]
    ):
        raise ValueError("Paired conditions may differ only in protocol name/prompt")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    destination = output_dir / "report.json"
    _save(
        destination,
        {
            "schema_version": 1,
            "planned_attempts": 2 * len(plan["tasks"]),
            "independent_families": len(plan["tasks"]),
            "max_context_tokens": plan["max_context_tokens"],
            "protocols": protocols,
            "paired": list(paired.values()),
            "sha256": artifacts,
            "on_policy_eligible": False,
            "limitations": [
                "One fixed-seed paired attempt per training family; this pilot is "
                "not independent capability evaluation.",
                "Invented observations and quality are explicit reply-bound reviews; "
                "tag counts alone are not behavioural evidence.",
                "Quality candidates are not loadable demonstrations. Replay, freeze "
                "and training-load results are separate evidence.",
            ],
        },
    )
    return destination
