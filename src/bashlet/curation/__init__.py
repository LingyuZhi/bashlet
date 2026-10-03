"""Bounded model collection and human-reviewed, executable Harness datasets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from ..config import ExperimentConfig, config_record, load_model
from ..demonstrations import ActionSupervision, prepare_sft
from ..experiment import run_experiment, task_from_record
from ..tasks import task_path, validate_harness_origin
from ..vllm import VLLMModel


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _plan(path: Path) -> dict[str, Any]:
    plan: dict[str, Any] = json.loads(path.read_text())
    for name, ceiling in (
        ("max_attempts", 60),
        ("max_accepted", 60),
        ("max_corrections", 3),
        ("max_context_tokens", 32768),
    ):
        value = plan.get(name)
        minimum = 0 if name == "max_corrections" else 1
        if type(value) is not int or not minimum <= value <= ceiling:
            raise ValueError(f"{name} must be an integer in [{minimum}, {ceiling}]")
    if plan["schema_version"] != 1:
        raise ValueError("Unsupported collection plan")
    selected = plan["tasks"]
    if (
        not selected
        or len(selected) > plan["max_attempts"]
        or len({item["family_id"] for item in selected}) != len(selected)
        or len({item["task_id"] for item in selected}) != len(selected)
    ):
        raise ValueError("Select distinct training families within max_attempts")
    manifest = Path(plan["task_manifest"])
    if _sha(manifest) != plan["task_manifest_sha256"]:
        raise ValueError("Frozen task manifest hash differs")
    for item in selected:
        path = task_path(manifest, item["task_id"], "train")
        task = task_from_record(json.loads(path.read_text()))
        assert task.origin is not None
        validate_harness_origin(task.origin)
        if (
            _sha(path) != item["task_sha256"]
            or task.origin.family_id != item["family_id"]
            or task.task_type != item["task_type"]
        ):
            raise ValueError("Selected task differs from the frozen training family")
    return plan


def collect_demonstrations(
    plan_path: Path, output_dir: Path, config: ExperimentConfig
) -> Path:
    """Execute exactly one bounded attempt per selected training family.

    An existing directory is never resumed or overwritten. Infrastructure errors
    leave an incomplete ledger; failed attempts are never silently replaced.
    """
    plan = _plan(plan_path)
    model = load_model(config.model)
    if not isinstance(model, VLLMModel) or model.identity != plan["expected_policy"]:
        raise ValueError("Collection requires the planned local vLLM base snapshot")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    (output_dir / "plan.json").write_bytes(plan_path.read_bytes())
    (output_dir / "source-manifest.json").write_bytes(
        Path(plan["task_manifest"]).read_bytes()
    )
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete",
        "started_at": datetime.now(UTC).isoformat(),
        "plan_sha256": _sha(output_dir / "plan.json"),
        "config": config_record(config),
        "attempts": [],
        "on_policy_eligible": False,
    }
    destination = output_dir / "collection.json"
    _save(destination, ledger)
    try:
        model.probe()
        details = model.describe()
        if (
            details.get("identity_verified") is not True
            or details.get("snapshot_kind") != "base"
        ):
            raise ValueError("Collection requires verified original base weights")
        ledger["policy_details"] = details
        for index, selected in enumerate(plan["tasks"]):
            item = dict(selected, attempt=index + 1, status="started")
            ledger["attempts"].append(item)
            _save(destination, ledger)
            task = task_from_record(
                json.loads(
                    task_path(
                        Path(plan["task_manifest"]), selected["task_id"], "train"
                    ).read_text()
                )
            )
            result = run_experiment(
                task=task,
                model=model,
                config=config,
                output_dir=output_dir / "episodes",
            )
            item.update(
                status=result.status,
                passed=result.passed,
                reason=result.reason,
                run_dir=str(result.run_dir.relative_to(output_dir)),
                run_sha256=_sha(result.run_dir / "run.json"),
                sampling=asdict(config.model.sampling),
                steps=len(result.trajectory),
            )
            _save(destination, ledger)
            print(
                f"{index + 1}/{len(plan['tasks'])} {task.task_id}: "
                f"{result.status}, passed={result.passed}",
                flush=True,
            )
            if result.status in ("infrastructure_error", "interrupted", "model_error"):
                raise ValueError(f"Collection stopped: {result.reason}")
        ledger["status"] = "complete"
        ledger["finished_at"] = datetime.now(UTC).isoformat()
    finally:
        _save(destination, ledger)
    return destination


def _supervision(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action.get("action_type"), str) or not action["action_type"]:
        raise ValueError("Every reviewed action needs an action_type")
    return asdict(
        ActionSupervision(action["supervise"], action["reason"], action["author"])
    )


def freeze_demonstrations(
    collection_path: Path,
    review_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> Path:
    """Replay human selections, load the real training inputs and freeze an audit.

    Quality is an explicit human judgement. This function checks provenance and
    reports those decisions; it never infers usefulness from a successful grade.
    """
    from ..training import load_demonstrations

    root = collection_path.resolve().parent
    plan = _plan(root / "plan.json")
    collection = json.loads(collection_path.read_text())
    review = json.loads(review_path.read_text())
    attempts = collection["attempts"]
    decisions = review["attempts"]
    if (
        collection["status"] != "complete"
        or review.get("collection_sha256") != _sha(collection_path)
        or collection["plan_sha256"] != _sha(root / "plan.json")
        or len(attempts) != len(plan["tasks"])
        or review["schema_version"] != 1
        or [item["attempt"] for item in decisions] != list(range(1, len(attempts) + 1))
    ):
        raise ValueError(
            "Freeze requires a complete collection and review of every attempt"
        )
    selected = []
    corrected = 0
    for index, (attempt, decision, task) in enumerate(
        zip(attempts, decisions, plan["tasks"], strict=True)
    ):
        source_dir = (root / attempt["run_dir"]).resolve()
        if not source_dir.is_relative_to(root / "episodes"):
            raise ValueError("Attempt evidence is outside the collection")
        if _sha(source_dir / "run.json") != attempt["run_sha256"]:
            raise ValueError("Attempt evidence hash differs")
        original = json.loads((source_dir / "run.json").read_text())
        if (
            attempt["attempt"] != index + 1
            or any(attempt[key] != value for key, value in task.items())
            or original["task_id"] != task["task_id"]
            or original["task_sha256"] != task["task_sha256"]
            or original["status"] != attempt["status"]
            or original["passed"] != attempt["passed"]
            or original["policy"] != plan["expected_policy"]
            or original["config"] != collection["config"]
            or any(
                step["model_call"] is not None
                and (
                    step["model_call"]["generation"]["source"] != "sampled"
                    or step["model_call"]["policy"] != plan["expected_policy"]
                )
                for step in original["trajectory"]
            )
        ):
            raise ValueError("Attempt differs from its planned sampled provenance")
        if (
            type(decision["accept"]) is not bool
            or not decision["reason"].strip()
            or len(decision["actions"]) != len(original["trajectory"])
        ):
            raise ValueError(
                "Every attempt and action needs an explicit quality review"
            )
        actions = [_supervision(action) for action in decision["actions"]]
        corrections = decision.get("corrections", [])
        if corrections:
            if (
                not decision["accept"]
                or not decision.get("correction_reason", "").strip()
            ):
                raise ValueError(
                    "Corrections require an accepted, justified coverage gap"
                )
            corrected += 1
        if decision["accept"]:
            selected.append(
                {
                    "task_id": task["task_id"],
                    "run_dir": str(source_dir),
                    "run_sha256": attempt["run_sha256"],
                    "actions": actions,
                    "corrections": [
                        {
                            "reply": correction["reply"],
                            "supervision": _supervision(correction["supervision"]),
                        }
                        for correction in corrections
                    ],
                }
            )
    if (
        not selected
        or len(selected) > plan["max_accepted"]
        or corrected > plan["max_corrections"]
    ):
        raise ValueError("Selection exceeds acceptance/correction bounds or is empty")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    for name, source in (
        ("plan.json", root / "plan.json"),
        ("collection.json", collection_path),
        ("quality-review.json", review_path),
    ):
        (output_dir / name).write_bytes(source.read_bytes())
    export_review = output_dir / "export-review.json"
    _save(export_review, {"schema_version": 1, "trajectories": selected})
    dataset = prepare_sft(
        manifest_path=Path(plan["task_manifest"]),
        output_dir=output_dir / "data",
        config=config,
        trajectories_path=export_review,
        max_context_tokens=plan["max_context_tokens"],
    )
    manifest = json.loads(dataset.read_text())
    # Loading rechecks each mask, shift, history, source and independent grade.
    if manifest["counts"]["accepted"]:
        load_demonstrations(dataset)
    rows = [
        json.loads(line)
        for line in (dataset.parent / "examples.jsonl").read_text().splitlines()
    ]
    by_task = {
        attempt["task_id"]: decision
        for attempt, decision in zip(attempts, decisions, strict=True)
    }
    outcomes = {item["task_id"]: item for item in manifest["tasks"]}
    report = _report(plan, attempts, by_task, outcomes, rows)
    _save(output_dir / "report.json", report)
    # Hash the generated data AND the original, immutable sampled evidence.
    artifacts = {
        str(path.relative_to(output_dir)): _sha(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    _save(
        output_dir / "freeze.json",
        {
            "schema_version": 1,
            "status": "frozen",
            "created_at": datetime.now(UTC).isoformat(),
            "sha256": artifacts,
            "source_runs": [
                {
                    "path": str(root / attempt["run_dir"] / "run.json"),
                    "sha256": attempt["run_sha256"],
                }
                for attempt in attempts
            ],
            "load_verified": bool(rows),
            "on_policy_eligible": False,
        },
    )
    return output_dir / "report.json"


def _report(
    plan: dict[str, Any],
    attempts: list[dict[str, Any]],
    reviews: dict[str, Any],
    outcomes: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tokens = Counter[str]()
    action_types: dict[str, Counter[str]] = {}
    per_type: dict[str, Counter[str]] = {}
    masked: dict[str, Counter[str]] = {}
    examples = []
    for attempt in attempts:
        per_type.setdefault(attempt["task_type"], Counter()).update(candidates=1)
    for row in rows:
        meta = row["metadata"]
        decision = reviews[meta["task_id"]]
        counts = {
            "context": meta["context_tokens"],
            "model_input": meta["context_tokens"] - 1,
            "supervised": meta["supervised_tokens"],
            "masked_generated": sum(
                len(a["generation"]["tokens"])
                for a in meta["action_supervision"]
                if not a["supervise"]
            ),
        }
        tokens.update(counts)
        per_type[meta["task_type"]].update(accepted=1, **counts)
        labels = decision["actions"] + [
            c["supervision"] for c in decision.get("corrections", [])
        ]
        for label, action in zip(labels, meta["action_supervision"], strict=True):
            generated = len(action["generation"]["tokens"])
            action_types.setdefault(label["action_type"], Counter()).update(
                actions=1,
                generated_tokens=generated,
                supervised_tokens=generated if action["supervise"] else 0,
            )
            if not action["supervise"]:
                masked.setdefault(action["reason"], Counter()).update(
                    actions=1, tokens=generated
                )
        examples.append(
            {
                "task_id": meta["task_id"],
                "family_id": meta["origin"]["family_id"],
                "task_type": meta["task_type"],
                **counts,
                "codex_corrected": bool(decision.get("corrections")),
                "actions": len(labels),
            }
        )
    accepted = {item["task_id"] for item in examples}
    rejected = [
        {
            **{
                key: attempt[key]
                for key in ("attempt", "task_id", "family_id", "task_type")
            },
            "stage": "export" if reviews[attempt["task_id"]]["accept"] else "quality",
            "reason": outcomes[attempt["task_id"]]["reason"]
            if reviews[attempt["task_id"]]["accept"]
            else reviews[attempt["task_id"]]["reason"],
        }
        for attempt in attempts
        if attempt["task_id"] not in accepted
    ]
    corrected = sum(item["codex_corrected"] for item in examples)
    return {
        "schema_version": 1,
        "dataset": "data/manifest.json",
        "on_policy_eligible": False,
        "counts": {
            "base_sampled_attempts": len(attempts),
            "sampled_tasks": len({a["task_id"] for a in attempts}),
            "sampled_families": len({a["family_id"] for a in attempts}),
            "accepted_trajectories": len(rows),
            "accepted_tasks": len(accepted),
            "accepted_families": len({a["family_id"] for a in examples}),
            "base_only_trajectories": len(rows) - corrected,
            "codex_correction_attempts": sum(
                bool(r.get("corrections")) for r in reviews.values()
            ),
            "codex_corrected_trajectories": corrected,
            "codex_authored_actions": sum(
                len(r.get("corrections", [])) for r in reviews.values()
            ),
            "replay_attempts": sum("run_dir" in item for item in outcomes.values()),
            "replay_passed": len(rows),
            "rejected": len(rejected),
        },
        "tokens": dict(tokens),
        "token_distribution": {
            key: {
                "min": min(values),
                "median": median(values),
                "max": max(values),
            }
            for key in ("model_input", "supervised")
            if (values := [e[key] for e in examples])
        },
        "per_type": per_type,
        "action_types": action_types,
        "masked_reasons": masked,
        "examples": examples,
        "rejected": rejected,
        "rejection_reasons": dict(Counter(item["reason"] for item in rejected)),
        "coverage_gaps": {
            kind: max(0, plan["target_per_type"] - counts["accepted"])
            for kind, counts in per_type.items()
        },
        "limitations": [
            "One fixed-seed attempt per training family; this selected dataset "
            "is not a model capability estimate.",
            "All training rows are static SFT; reconstruction and Codex corrections "
            "are replay, never on-policy samples.",
            "Exact replay excludes nondeterministic observations; missing families "
            "limit coverage and pilot interpretation.",
            "Action categories are exclusive primary roles assigned by the "
            "reviewer; code-bearing writes retain full-token supervision.",
        ],
    }
