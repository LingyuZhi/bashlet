"""Paired checkpoint evaluations and fixed-denominator development curves."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bashlet.config import ExperimentConfig
from bashlet.distillation import OPSDRun
from bashlet.distillation.pytrio import PyTRIOOPSD
from bashlet.monitoring import MonitorConfig, write_document

if TYPE_CHECKING:
    from bashlet.distillation.config import RunConfig


def _evaluate_checkpoint(
    *,
    config: RunConfig,
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
    from bashlet.evaluation.checkpoints import evaluate_checkpoint

    if config.evaluation is None:
        raise ValueError("OPSD requires explicit local evaluation settings")
    source = Path(record["configuration_path"])
    executor = config.evaluation.get("executor", [])
    if executor:
        command = [
            "uv",
            "run",
            "--locked",
            "bashlet",
            "evaluate-checkpoint",
            str(source),
            "--run-dir",
            str(root),
            "--phase",
            phase,
        ]
        remote = "cd " + shlex.quote(str(Path.cwd())) + " && " + shlex.join(command)
        completed = subprocess.run(
            [*executor, "bash -lc " + shlex.quote(remote)], check=False
        )
        destination = root / "evaluation" / phase / "stage.json"
        if completed.returncode:
            raise ValueError("Local checkpoint evaluation did not complete")
    else:
        destination = evaluate_checkpoint(source, phase=phase, run_dir=root)
    stage: dict[str, Any] = json.loads(destination.read_text())
    if stage["snapshot"] != snapshot or stage["optimizer_step"] != optimizer_step:
        raise ValueError("Local evaluation used another saved checkpoint")
    record.setdefault("evaluations", {})[phase] = stage
    record["development_curve"] = development_curve(record)
    write_document(path, record)
    return stage


def _decision(record: dict[str, Any]) -> None:
    record["development_curve"] = development_curve(record)
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
    curve = record["development_curve"]
    means = [
        curve.get(phase, {}).get("mean_accuracy") for phase in ("baseline", "final")
    ]
    record["comparison"]["mean_accuracy_difference"] = (
        means[1] - means[0]
        if comparable and all(m is not None for m in means)
        else None
    )
    record["decision"] = {
        "action": "review_results"
        if progressed and comparable
        else "insufficient_evidence",
        "reason": "Report paired evidence for research review",
    }
    record["billing"] = {
        "teacher_category_verified": False,
        "cache_discount_assumed": False,
        "actual_cny": record["cost"]["actual_cny"],
        "sdk_scoring": "0.2.9 compute_logprobs uses prompt logprobs and max_tokens=1",
        "limitation": "SDK drops billing details; usage and ceilings are not bills",
    }


def development_curve(record: dict[str, Any]) -> dict[str, Any]:
    """Average fixed seeds within each independent family, retaining every gap."""
    configuration = record["configuration"]
    seeds = configuration["pilot"]["evaluation_seeds"] or [
        configuration["condition"]["model"]["sampling"]["seed"]
    ]
    ids = record["dev_task_ids"]
    curve = {}
    for phase, stage in record.get("evaluations", {}).items():
        runs = {}
        for seed, reference in stage["runs"].items():
            raw = Path(reference["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError("Development evidence changed")
            evaluation = json.loads(raw)
            runs[seed] = {row["task_id"]: row for row in evaluation["tasks"]}
        families = []
        missing = []
        for identity in ids:
            outcomes = {}
            metadata: dict[str, Any] = {}
            for seed in seeds:
                row = runs.get(str(seed), {}).get(identity, {})
                metadata = row or metadata
                outcome = row.get("outcome", "not_started")
                unavailable = (
                    outcome
                    in {"not_started", "running", "interrupted", "infrastructure_error"}
                    or row.get("budget_kind") == "spend"
                )
                outcomes[str(seed)] = None if unavailable else int(outcome == "passed")
                if unavailable:
                    missing.append({"task_id": identity, "seed": seed})
            success = sum(value or 0 for value in outcomes.values())
            origin = (
                metadata.get("task_assignment") or metadata.get("task_origin") or {}
            )
            families.append(
                {
                    "task_id": identity,
                    "family_id": origin.get("family_id", identity),
                    "task_type": metadata.get("task_type", "unknown"),
                    "by_seed": outcomes,
                    "mean_accuracy": success / len(seeds)
                    if all(value is not None for value in outcomes.values())
                    else None,
                    "observed_success_lower_bound": success / len(seeds),
                }
            )
        complete = not missing and stage["status"] == "complete"
        lower = sum(f["observed_success_lower_bound"] for f in families) / len(ids)
        curve[phase] = {
            "snapshot": stage["snapshot"],
            "optimizer_step": stage["optimizer_step"],
            "status": "complete" if complete else "incomplete",
            "planned_families": len(ids),
            "planned_episodes": len(ids) * len(seeds),
            "evaluation_seeds": seeds,
            "mean_accuracy": lower if complete else None,
            "observed_success_lower_bound": lower,
            "missing": missing,
            "families": families,
            "by_task_type": {
                kind: {
                    "planned_families": len(group),
                    "mean_accuracy": sum(f["mean_accuracy"] for f in group) / len(group)
                    if all(f["mean_accuracy"] is not None for f in group)
                    else None,
                }
                for kind in {f["task_type"] for f in families}
                if (group := [f for f in families if f["task_type"] == kind])
            },
            "aggregation": "mean over fixed seeds per family, then equal family mean",
        }
    return curve
