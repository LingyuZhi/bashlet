"""Finite multi-source attempts and family-aware, immutable SFT collection."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, config_record, load_model
from ..demonstrations import prepare_sft
from ..encoding import QwenChatMLEncoding
from ..experiment import run_experiment, task_from_record
from ..models import Task
from ..replay import ReplayModel
from ..tasks import task_path
from ..vllm import VLLMModel
from . import _save, _sha, _supervision


def validate_sources(plan: dict[str, Any]) -> None:
    for name, ceiling in (
        ("max_attempts_per_family", 3),
        ("max_accepted_per_family", 2),
    ):
        value = plan.get(name)
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer in [1, {ceiling}]")
    for name, minimum, maximum in (
        ("max_elapsed_seconds", 1, 7200),
        ("gpu_seconds", 0, 7200),
    ):
        value = plan.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(f"{name} must be finite in [{minimum}, {maximum}]")
    if plan.get("paid_cny") != 0:
        raise ValueError("This collection supports zero new paid requests only")
    sources = plan.get("sources", [])
    if not sources or len({s["name"] for s in sources}) != len(sources):
        raise ValueError("Select distinct explicit generation sources")
    for source in sources:
        if not all(
            isinstance(source.get(key), str) and source[key].strip()
            for key in ("name", "policy", "generation_method", "author")
        ):
            raise ValueError("Sources need a policy, author and generation method")
        if source["kind"] == "authored":
            if source["author"] != "codex" or not source["policy"].startswith("codex/"):
                raise ValueError(
                    "Authored sources must preserve explicit Codex identity"
                )
            artifact = Path(source["artifact"])
            if _sha(artifact) != source["artifact_sha256"]:
                raise ValueError("Authored source artifact hash differs")
            data = json.loads(artifact.read_text())
            allowed = {task["task_id"] for task in plan["tasks"]}
            if data["schema_version"] != 1 or any(
                item["task_id"] not in allowed
                or not isinstance(item["replies"], list)
                or not item["replies"]
                or not all(isinstance(reply, str) for reply in item["replies"])
                for item in data["attempts"]
            ):
                raise ValueError(
                    "Authored records must identify selected training tasks"
                )
        elif source["kind"] == "sampled":
            if (
                source["author"] != "model"
                or plan["gpu_seconds"] <= 0
                or type(source.get("max_attempts_per_family")) is not int
                or not 1 <= source["max_attempts_per_family"] <= 2
            ):
                raise ValueError("Local sampling requires a bounded GPU stage")
        else:
            raise ValueError("Unknown generation source kind")


def collect(
    plan_path: Path, plan: dict[str, Any], output_dir: Path, config: ExperimentConfig
) -> Path:
    base = load_model(config.model)
    if not isinstance(base.encoding, QwenChatMLEncoding):
        raise ValueError("Collection requires the student's Qwen ChatML encoding")
    if any(s["kind"] == "sampled" for s in plan["sources"]):
        if not isinstance(base, VLLMModel):
            raise ValueError("Sampled collection requires verified local vLLM weights")
        if any(
            s["policy"] != base.identity
            for s in plan["sources"]
            if s["kind"] == "sampled"
        ):
            raise ValueError("Sampled source policy differs from its planned snapshot")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    (output_dir / "plan.json").write_bytes(plan_path.read_bytes())
    (output_dir / "source-manifest.json").write_bytes(
        Path(plan["task_manifest"]).read_bytes()
    )
    authored: dict[str, list[dict[str, Any]]] = {}
    for index, source in enumerate(plan["sources"]):
        if source["kind"] == "authored":
            data = Path(source["artifact"]).read_bytes()
            (output_dir / f"source-{index}.json").write_bytes(data)
            authored[source["name"]] = json.loads(data)["attempts"]
    started = time.monotonic()
    ledger: dict[str, Any] = {
        "schema_version": 2,
        "status": "incomplete",
        "started_at": datetime.now(UTC).isoformat(),
        "plan_sha256": _sha(output_dir / "plan.json"),
        "config": config_record(config),
        "attempts": [],
        "stop_reason": "source_exhausted",
        "on_policy_eligible": False,
        "sources": plan["sources"],
    }
    destination = output_dir / "collection.json"
    _save(destination, ledger)
    passing = Counter[str]()
    try:
        if isinstance(base, VLLMModel) and any(
            s["kind"] == "sampled" for s in plan["sources"]
        ):
            base.probe()
            if base.describe().get("identity_verified") is not True:
                raise ValueError("Unverified local source snapshot")
            ledger["policy_details"] = base.describe()
        for selected in plan["tasks"]:
            family = selected["family_id"]
            family_attempts = 0
            for source in plan["sources"]:
                recipes = (
                    [
                        item
                        for item in authored[source["name"]]
                        if item["task_id"] == selected["task_id"]
                    ]
                    if source["kind"] == "authored"
                    else [{}] * source["max_attempts_per_family"]
                )
                for recipe in recipes:
                    elapsed = time.monotonic() - started
                    if elapsed >= plan["max_elapsed_seconds"] or (
                        source["kind"] == "sampled" and elapsed >= plan["gpu_seconds"]
                    ):
                        ledger["stop_reason"] = "time_limit"
                        return destination
                    if len(ledger["attempts"]) >= plan["max_attempts"]:
                        ledger["stop_reason"] = "attempt_limit"
                        ledger["status"] = "complete"
                        return destination
                    if sum(passing.values()) >= plan["max_accepted"]:
                        ledger["stop_reason"] = "candidate_limit"
                        ledger["status"] = "complete"
                        return destination
                    if (
                        family_attempts >= plan["max_attempts_per_family"]
                        or passing[family] >= plan["max_accepted_per_family"]
                    ):
                        break
                    family_attempts += 1
                    identity = uuid.uuid4().hex
                    remaining = plan["max_elapsed_seconds"] - elapsed
                    if source["kind"] == "sampled":
                        remaining = min(remaining, plan["gpu_seconds"] - elapsed)
                    execution_config = replace(
                        config,
                        budget=replace(
                            config.budget,
                            max_seconds=min(config.budget.max_seconds, remaining),
                        ),
                    )
                    item = dict(
                        selected,
                        attempt=len(ledger["attempts"]) + 1,
                        attempt_id=identity,
                        demonstration_id=identity,
                        family_attempt=family_attempts,
                        status="started",
                        execution_config=config_record(execution_config),
                        provenance={
                            key: source[key]
                            for key in (
                                "name",
                                "kind",
                                "policy",
                                "author",
                                "generation_method",
                            )
                        },
                    )
                    if source["kind"] == "authored":
                        item["provenance"].update(
                            artifact_sha256=source["artifact_sha256"],
                            original_replies=recipe["replies"],
                        )
                    ledger["attempts"].append(item)
                    _save(destination, ledger)
                    task = task_from_record(
                        json.loads(
                            task_path(
                                Path(plan["task_manifest"]),
                                selected["task_id"],
                                "train",
                            ).read_text()
                        )
                    )
                    model = (
                        ReplayModel(
                            [
                                base.encoding.replay_reply(reply)
                                for reply in recipe["replies"]
                            ],
                            encoding=base.encoding,
                            identity=source["policy"],
                        )
                        if source["kind"] == "authored"
                        else base
                    )
                    result = run_experiment(
                        task=task,
                        model=model,
                        config=execution_config,
                        output_dir=output_dir / "episodes",
                    )
                    item.update(
                        status=result.status,
                        passed=result.passed,
                        reason=result.reason,
                        run_dir=str(result.run_dir.relative_to(output_dir)),
                        run_sha256=_sha(result.run_dir / "run.json"),
                        steps=len(result.trajectory),
                    )
                    passing[family] += int(result.passed)
                    _save(destination, ledger)
                    print(
                        f"{item['attempt']} {task.task_id}: {result.status}, "
                        f"passed={result.passed}",
                        flush=True,
                    )
                    if result.status in (
                        "infrastructure_error",
                        "interrupted",
                        "model_error",
                    ):
                        ledger["stop_reason"] = result.status
                        return destination
        ledger["status"] = "complete"
        ledger["stop_reason"] = (
            "attempt_limit"
            if len(ledger["attempts"]) == plan["max_attempts"]
            else "candidate_limit"
            if sum(passing.values()) == plan["max_accepted"]
            else "source_exhausted"
        )
    finally:
        ledger["elapsed_seconds"] = time.monotonic() - started
        ledger["finished_at"] = datetime.now(UTC).isoformat()
        _save(destination, ledger)
    return destination


def _evidence(
    root: Path,
    collection: dict[str, Any],
    attempt: dict[str, Any],
    decision: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any] | None:
    if (
        any(attempt[key] != value for key, value in selected.items())
        or decision["attempt_id"] != attempt["attempt_id"]
        or type(decision["accept"]) is not bool
        or not decision["reason"].strip()
    ):
        raise ValueError("Attempt or review differs from the selected family")
    if "run_dir" not in attempt:
        if decision["accept"] or decision["actions"]:
            raise ValueError("Unfinished attempt has no reviewable actions")
        return None
    source_dir = (root / attempt["run_dir"]).resolve()
    if (
        not source_dir.is_relative_to(root / "episodes")
        or _sha(source_dir / "run.json") != attempt["run_sha256"]
    ):
        raise ValueError("Attempt evidence is outside the collection or hash differs")
    original: dict[str, Any] = json.loads((source_dir / "run.json").read_text())
    provenance = attempt["provenance"]
    if (
        original["config"] != attempt.get("execution_config", collection["config"])
        or original["policy"] != provenance["policy"]
        or original["task_id"] != selected["task_id"]
        or original["task_sha256"] != selected["task_sha256"]
        or original["status"] != attempt["status"]
        or original["passed"] != attempt["passed"]
        or len(original["trajectory"]) != len(decision["actions"])
    ):
        raise ValueError("Attempt differs from its executed provenance")
    expected_source = "sampled" if provenance["kind"] == "sampled" else "replayed"
    for step, label in zip(original["trajectory"], decision["actions"], strict=True):
        call = step["model_call"]
        if (
            call is None
            or call["policy"] != provenance["policy"]
            or call["generation"]["source"] != expected_source
            or label["author"] != provenance["author"]
        ):
            raise ValueError(
                "Action source or author differs from its executed provenance"
            )
        _supervision(label)
    return original


def _protected_reason(
    record: dict[str, Any], decision: dict[str, Any], task: Task
) -> str:
    protected = set(task.initial_files) - set(task.implementation_paths)
    for step in record["trajectory"]:
        for path in protected:
            actual = step["workspace_manifest"].get(path, {})
            if (
                actual.get("sha256")
                != hashlib.sha256(task.initial_files[path].encode()).hexdigest()
            ):
                return "Task input or feedback materials were modified"
    if task.task_type == "workspace" and decision["accept"]:
        usage = decision.get("workspace_use", {})
        read, write = usage.get("read_step"), usage.get("implementation_step")
        steps = record["trajectory"]
        if (
            type(read) is not int
            or type(write) is not int
            or not 0 <= read < write < len(steps)
            or not usage.get("reason", "").strip()
            or not usage.get("paths")
            or any(path not in protected for path in usage["paths"])
            or not steps[read].get("observation")
            or steps[read]["observation"]["status"] != "ok"
            or not steps[read]["observation"]["output"].strip()
        ):
            return "Workspace use needs a reviewed read-to-implementation dependency"
    return ""


def _recovery(record: dict[str, Any], decision: dict[str, Any]) -> bool:
    """Bind custom public checks to actual failure, changed code and verification."""
    proof = decision.get("recovery")
    if proof is None:
        return any(repair["passed"] is True for repair in record["repair_attempts"])
    failed, repaired, verified = (
        proof.get(key) for key in ("failure_step", "repair_step", "verification_step")
    )
    steps = record["trajectory"]
    if (
        any(type(value) is not int for value in (failed, repaired, verified))
        or not 0 <= failed < repaired < verified < len(steps)
        or not proof.get("reason", "").strip()
    ):
        raise ValueError(
            "Recovery review needs ordered real failure and verification steps"
        )
    failure, verification = steps[failed]["observation"], steps[verified]["observation"]
    if (
        failure is None
        or failure["status"] != "command_failed"
        or failure["exit_code"] in (None, 0)
        or verification is None
        or verification["status"] != "ok"
        or steps[failed]["candidate"] is None
        or steps[repaired]["candidate"] == steps[failed]["candidate"]
        or steps[verified]["candidate"] != steps[repaired]["candidate"]
        or record["episode_status"] != "submitted"
        or not record["passed"]
    ):
        raise ValueError(
            "Recovery review does not match an executed repair "
            "and successful submission"
        )
    return True


def freeze(
    collection_path: Path,
    review_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    plan: dict[str, Any],
) -> Path:
    from ..training import load_demonstrations

    root = collection_path.resolve().parent
    collection = json.loads(collection_path.read_text())
    review = json.loads(review_path.read_text())
    attempts, decisions = collection["attempts"], review["attempts"]
    if (
        collection["schema_version"] != 2
        or collection["sources"] != plan["sources"]
        or review["schema_version"] != 2
        or review["collection_sha256"] != _sha(collection_path)
        or collection["plan_sha256"] != _sha(root / "plan.json")
        or len(attempts) != len(decisions)
        or len(attempts) > plan["max_attempts"]
        or len({a["attempt_id"] for a in attempts}) != len(attempts)
        or len({a["demonstration_id"] for a in attempts}) != len(attempts)
    ):
        raise ValueError("Collection and complete review must bind the frozen plan")
    selected = {item["task_id"]: item for item in plan["tasks"]}
    family_attempts = Counter[str]()
    candidates = []
    rejected = []
    corrections = 0
    records = {}
    sources = {source["name"]: source for source in plan["sources"]}
    for index, (attempt, decision) in enumerate(zip(attempts, decisions, strict=True)):
        provenance = attempt["provenance"]
        source = sources.get(provenance["name"])
        if source is None or any(
            provenance.get(key) != source.get(key)
            for key in (
                "name",
                "kind",
                "policy",
                "author",
                "generation_method",
                "artifact_sha256",
            )
        ):
            raise ValueError("Attempt source differs from the frozen source admission")
        task_info = selected[attempt["task_id"]]
        family = task_info["family_id"]
        family_attempts[family] += 1
        if (
            attempt["attempt"] != index + 1
            or attempt["family_attempt"] != family_attempts[family]
            or family_attempts[family] > plan["max_attempts_per_family"]
        ):
            raise ValueError("Family attempts exceed the frozen finite bound")
        record = _evidence(root, collection, attempt, decision, task_info)
        reason = decision["reason"] if not decision["accept"] else ""
        if record is not None:
            records[attempt["demonstration_id"]] = record
            task = task_from_record(
                json.loads(
                    task_path(
                        Path(plan["task_manifest"]), attempt["task_id"], "train"
                    ).read_text()
                )
            )
            reason = reason or _protected_reason(record, decision, task)
        if reason or record is None:
            rejected.append(
                {
                    "attempt_id": attempt["attempt_id"],
                    "demonstration_id": attempt["demonstration_id"],
                    **task_info,
                    "stage": "quality",
                    "reason": reason or "No sealed run",
                }
            )
            continue
        additions = decision.get("corrections", [])
        corrections += bool(additions)
        if additions and not decision.get("correction_reason", "").strip():
            raise ValueError(
                "A correction needs a reason tied to the real failure prefix"
            )
        candidates.append(
            {
                "demonstration_id": attempt["demonstration_id"],
                "task_id": attempt["task_id"],
                "run_dir": str(root / attempt["run_dir"]),
                "run_sha256": attempt["run_sha256"],
                "provenance": attempt["provenance"],
                "complementary_reason": decision.get("complementary_reason", ""),
                "actions": [_supervision(action) for action in decision["actions"]],
                "corrections": [
                    {
                        "reply": addition["reply"],
                        "supervision": _supervision(addition["supervision"]),
                    }
                    for addition in additions
                ],
            }
        )
    if len(candidates) > plan["max_accepted"] or corrections > plan["max_corrections"]:
        raise ValueError(
            "Reviewed selection exceeds the acceptance or correction bound"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    for name, source in (
        ("plan.json", root / "plan.json"),
        ("collection.json", collection_path),
        ("quality-review.json", review_path),
    ):
        (output_dir / name).write_bytes(source.read_bytes())
    for index, source in enumerate(plan["sources"]):
        if source["kind"] == "authored":
            original = root / f"source-{index}.json"
            if _sha(original) != source["artifact_sha256"]:
                raise ValueError("Preserved authored source artifact hash differs")
            (output_dir / original.name).write_bytes(original.read_bytes())
    rows = []
    outcomes = []
    if candidates:
        export_review = output_dir / "export-review.json"
        _save(export_review, {"schema_version": 2, "trajectories": candidates})
        dataset = prepare_sft(
            manifest_path=Path(plan["task_manifest"]),
            output_dir=output_dir / "data",
            config=config,
            trajectories_path=export_review,
            max_context_tokens=plan["max_context_tokens"],
        )
        manifest = json.loads(dataset.read_text())
        outcomes = manifest["tasks"]
        if manifest["counts"]["accepted"]:
            load_demonstrations(dataset)
        rows = [
            json.loads(line)
            for line in (dataset.parent / "examples.jsonl").read_text().splitlines()
        ]
    for outcome in outcomes:
        if outcome["status"] != "accepted":
            rejected.append(
                {
                    "demonstration_id": outcome["demonstration_id"],
                    "task_id": outcome["task_id"],
                    "stage": "export",
                    "reason": outcome["reason"],
                }
            )
    report = _report(
        plan, collection, decisions, records, rows, rejected, len(candidates), outcomes
    )
    _save(output_dir / "report.json", report)
    _save(
        output_dir / "freeze.json",
        {
            "schema_version": 2,
            "status": "frozen",
            "created_at": datetime.now(UTC).isoformat(),
            "sha256": {
                str(path.relative_to(output_dir)): _sha(path)
                for path in sorted(output_dir.rglob("*"))
                if path.is_file() and not path.is_symlink()
            },
            "source_runs": [
                {
                    "path": str(root / a["run_dir"] / "run.json"),
                    "sha256": a["run_sha256"],
                }
                for a in attempts
                if "run_dir" in a
            ],
            "load_verified": bool(rows),
            "on_policy_eligible": False,
        },
    )
    return output_dir / "report.json"


def _report(
    plan: dict[str, Any],
    collection: dict[str, Any],
    decisions: list[dict[str, Any]],
    records: dict[str, Any],
    rows: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    candidates: int,
    outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    attempts = collection["attempts"]
    reviews = {
        a["demonstration_id"]: d for a, d in zip(attempts, decisions, strict=True)
    }
    accepted_families: set[str] = set()
    tokens = Counter[str]()
    per_type = {
        kind: Counter[str]() for kind in ("short_program", "code_edit", "workspace")
    }
    per_source: dict[str, Counter[str]] = {}
    families_by_type: dict[str, set[str]] = {kind: set() for kind in per_type}
    action_types: dict[str, Counter[str]] = {}
    masked: dict[str, Counter[str]] = {}
    examples = []
    source_families: dict[str, set[str]] = {}
    dispositions = {a["demonstration_id"]: a for a in attempts}
    quality_ids = {item["demonstration_id"] for item in outcomes}
    for attempt in attempts:
        per_type[attempt["task_type"]].update(attempts=1)
        per_source.setdefault(attempt["provenance"]["name"], Counter()).update(
            attempts=1
        )
        if attempt["demonstration_id"] in quality_ids:
            per_type[attempt["task_type"]].update(quality_candidates=1)
            per_source[attempt["provenance"]["name"]].update(quality_candidates=1)
    for row in rows:
        meta = row["metadata"]
        identity = meta["demonstration_id"]
        decision = reviews[identity]
        record = json.loads((Path(meta["run_dir"]) / "run.json").read_text())
        family = meta["origin"]["family_id"]
        recovered = _recovery(record, decision)
        counts = {
            "context": meta["context_tokens"],
            "model_input": meta["context_tokens"] - 1,
            "supervised": meta["supervised_tokens"],
            "masked_generated": 0,
        }
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
                counts["masked_generated"] += generated
                masked.setdefault(action["reason"], Counter()).update(
                    actions=1, tokens=generated
                )
        tokens.update(counts)
        accepted_families.add(family)
        families_by_type[meta["task_type"]].add(family)
        per_type[meta["task_type"]].update(
            accepted=1, recovery=int(recovered), **counts
        )
        name = meta["provenance"]["name"]
        per_source[name].update(accepted=1, recovery=int(recovered), **counts)
        source_families.setdefault(name, set()).add(family)
        examples.append(
            {
                "demonstration_id": identity,
                "task_id": meta["task_id"],
                "family_id": family,
                "task_type": meta["task_type"],
                "provenance": meta["provenance"],
                "recovery": recovered,
                "workspace_use": decision.get("workspace_use"),
                **counts,
            }
        )
    for kind, families in families_by_type.items():
        per_type[kind]["accepted_families"] = len(families)
    for name, counts in per_source.items():
        counts["accepted_families"] = len(source_families.get(name, set()))
    for rejection in rejected:
        attempt = dispositions[rejection["demonstration_id"]]
        per_type[attempt["task_type"]].update(rejected=1)
        per_source[attempt["provenance"]["name"]].update(rejected=1)
    recovered_count = sum(e["recovery"] for e in examples)
    sampled = sum(a["provenance"]["kind"] == "sampled" for a in attempts)
    gaps = {
        kind: max(0, plan["target_per_type"] - len(families))
        for kind, families in families_by_type.items()
    }
    return {
        "schema_version": 2,
        "dataset": "data/manifest.json" if rows else None,
        "on_policy_eligible": False,
        "target_met": not any(gaps.values()),
        "counts": {
            "planned_families": len(plan["tasks"]),
            "collection_attempts": len(attempts),
            "attempted_families": len({a["family_id"] for a in attempts}),
            "base_sampled_attempts": sampled,
            "reference_constructed_attempts": sum(
                a["provenance"]["kind"] == "authored"
                and a["provenance"]["generation_method"]
                not in ("initial_state_audit", "correction")
                for a in attempts
            ),
            "initial_state_audit_attempts": sum(
                a["provenance"]["generation_method"] == "initial_state_audit"
                for a in attempts
            ),
            "correction_attempts": sum(bool(d.get("corrections")) for d in decisions),
            "quality_candidates": candidates,
            "accepted_trajectories": len(rows),
            "accepted_tasks": len({e["task_id"] for e in examples}),
            "accepted_families": len(accepted_families),
            "recovery_trajectories": recovered_count,
            "replay_attempts": sum("run_dir" in item for item in outcomes),
            "replay_passed": len(rows),
            "rejected": len(rejected),
        },
        "tokens": dict(tokens),
        "per_type": per_type,
        "per_source": per_source,
        "action_types": action_types,
        "masked_reasons": masked,
        "examples": examples,
        "rejected": rejected,
        "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
        "coverage_gaps": gaps,
        "unattempted_families": [
            t["family_id"]
            for t in plan["tasks"]
            if t["family_id"] not in {a["family_id"] for a in attempts}
        ],
        "recovery": {
            "target_min": 0.20,
            "target_max": 0.25,
            "fraction": recovered_count / len(rows) if rows else None,
        },
        "collection_status": collection["status"],
        "stop_reason": collection["stop_reason"],
        "limitations": [
            "Static SFT and reconstruction are not on-policy or teacher scoring.",
            "Quality and workspace dependence require explicit human action review.",
            "Recovery targets never reject high-quality natural recovery.",
            "Exact replay preserves original observations; "
            "rejected replays remain separate.",
        ],
    }
