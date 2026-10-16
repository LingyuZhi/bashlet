"""Paired checkpoint evaluations and fixed-denominator development curves."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config import ExperimentConfig, config_record
from ..evaluation.harness import evaluate
from ..monitoring import MonitorConfig, RunMonitor, write_document
from . import OPSDRun
from .pytrio import PyTRIOOPSD

if TYPE_CHECKING:
    from .pilot import PilotConfig


def _background_stage(
    root: Path, record: dict[str, Any], phase: str, snapshot: str, step: int,
) -> dict[str, Any] | None:
    """Admit an explicitly configured independent checkpoint evaluation job."""
    binding_path = root / "development-execution.json"
    previous = record.get("development_execution")
    if not binding_path.exists():
        if previous is not None:
            raise ValueError("Background development execution binding is missing")
        return None
    from .pilot import PilotConfig

    binding = json.loads(binding_path.read_text())
    config_path = Path(binding["config"])
    if (
        hashlib.sha256(config_path.read_bytes()).hexdigest() != binding["sha256"]
        or previous is not None and previous != binding
    ):
        raise ValueError("Background development configuration changed")
    import yaml

    settings = yaml.safe_load(config_path.read_text())
    pilot_path = Path(settings["pilot_config"])
    if (
        settings.get("schema_version") != 1
        or Path(settings["pilot_run"]).resolve() != root.resolve()
        or hashlib.sha256(pilot_path.read_bytes()).hexdigest()
        != settings["pilot_config_sha256"]
        or json.loads(json.dumps(asdict(PilotConfig.read(pilot_path))))
        != record["configuration"]["pilot"]
    ):
        raise ValueError("Background development belongs to another pilot")
    routing = root / "evaluation-routing.json"
    if "evaluation_routing_sha256" in settings:
        if hashlib.sha256(routing.read_bytes()).hexdigest() != settings[
            "evaluation_routing_sha256"
        ]:
            raise ValueError("Background development evaluation routing changed")
        route = json.loads(routing.read_text())
        if hashlib.sha256(Path(route["config"]).read_bytes()).hexdigest() != route[
            "sha256"
        ]:
            raise ValueError("Local evaluation configuration changed")
    record["development_execution"] = binding
    receipt_path = settings["job_receipts"].get(phase)
    if receipt_path is None:
        return None
    if not phase.startswith("epoch-"):
        raise ValueError("Only saved epoch checkpoints can run in the background")
    receipt = json.loads(Path(receipt_path).read_text())
    if (
        receipt.get("schema_version") != 1 or receipt.get("phase") != phase
        or receipt.get("snapshot") != snapshot or receipt.get("optimizer_step") != step
        or not isinstance(receipt.get("session"), str) or not receipt["session"]
    ):
        raise ValueError("Background development job has another checkpoint")
    job = {
        "receipt": str(receipt_path), "session": receipt["session"],
        "stage": receipt["stage"], "snapshot": snapshot, "optimizer_step": step,
    }
    known = record.setdefault("background_development", {}).setdefault(phase, job)
    if known != job:
        raise ValueError("Background development job identity changed")
    destination = Path(job["stage"])
    stage: dict[str, Any] = (
        json.loads(destination.read_text()) if destination.exists()
        else {
            "snapshot": snapshot, "optimizer_step": step,
            "status": "running", "runs": {},
        }
    )
    if stage["snapshot"] != snapshot or stage["optimizer_step"] != step:
        raise ValueError("Background development stage has another checkpoint")
    return stage


def _wait_for_background(
    root: Path, path: Path, record: dict[str, Any],
) -> None:
    """Finish independent jobs before switching the service or reporting results."""
    for phase, job in record.get("background_development", {}).items():
        while True:
            stage = _background_stage(
                root, record, phase, job["snapshot"], job["optimizer_step"],
            )
            if stage is None:
                raise ValueError("Background development configuration is missing")
            record.setdefault("evaluations", {})[phase] = stage
            record["development_curve"] = development_curve(record)
            write_document(path, record)
            if stage["status"] == "complete":
                break
            result = subprocess.run(
                ["tmux", "list-panes", "-t", job["session"], "-F", "#{pane_dead}"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode or result.stdout.strip() != "0":
                raise ValueError(
                    "Background development worker exited "
                    "before completing its checkpoint"
                )
            time.sleep(15)


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

    background = _background_stage(root, record, phase, snapshot, optimizer_step)
    if background is not None:
        record.setdefault("evaluations", {})[phase] = background
        record["development_curve"] = development_curve(record)
        write_document(path, record)
        return background
    if phase != "baseline":
        _wait_for_background(root, path, record)
    routing = root / "evaluation-routing.json"
    if routing.exists():
        from ..snapshots.manifest import file_sha256

        receipt = json.loads(routing.read_text())
        local_config = Path(receipt["config"])
        if file_sha256(local_config) != receipt["sha256"]:
            raise ValueError("Local evaluation routing configuration changed")
        import yaml

        settings = yaml.safe_load(local_config.read_text())
        destination = Path(settings["output_dir"]).resolve() / phase / "stage.json"
        if phase == "baseline" and record.get("development_execution"):
            cached: dict[str, Any] = json.loads(destination.read_text())
            if (
                cached != record.get("evaluations", {}).get(phase)
                or cached.get("snapshot") != snapshot
                or cached.get("optimizer_step") != optimizer_step
                or cached.get("routing_sha256") != receipt["sha256"]
                or cached.get("status") != "complete"
                or set(cached.get("runs", {}))
                != {str(seed) for seed in config.evaluation_seeds}
            ):
                raise ValueError("Cached baseline differs from its completed receipt")
            # Verify every pinned report without taking the local evaluator's lock.
            record["development_curve"] = development_curve(record)
            if record["development_curve"][phase]["status"] != "complete":
                raise ValueError("Background development requires a complete baseline")
            write_document(path, record)
            return cached
        command = [
            "uv",
            "run",
            "--locked",
            "bashlet",
            "evaluate-checkpoint",
            str(local_config.resolve()),
            "--phase",
            phase,
        ]
        executor = settings.get("executor", [])
        if executor:
            remote = "cd " + shlex.quote(str(Path.cwd())) + " && " + shlex.join(command)
            command = [*executor, "bash -lc " + shlex.quote(remote)]
        completed = subprocess.run(command, check=False)
        if not destination.exists():
            raise ValueError("Local checkpoint evaluation did not produce evidence")
        stage: dict[str, Any] = json.loads(destination.read_text())
        if stage["snapshot"] != snapshot or stage["optimizer_step"] != optimizer_step:
            raise ValueError(
                "Local checkpoint evaluation used another saved checkpoint"
            )
        if completed.returncode and stage["status"] == "complete":
            raise ValueError("Local checkpoint evaluator exited abnormally")
        record.setdefault("evaluations", {})[phase] = stage
        record["development_curve"] = development_curve(record)
        write_document(path, record)
        return stage

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
                # The parent summary verified this same W&B project before
                # any remote work. A later network outage can stay offline.
                require_online=False,
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
    record["development_curve"] = development_curve(record)
    write_document(path, record)
    summary = RunMonitor.resume(root / "summary")
    point = record["development_curve"][phase]
    summary.log(
        f"development-{phase}-complete",
        {
            "eval/phase": phase,
            "train/optimizer_step": optimizer_step,
            "eval/mean_accuracy": point["mean_accuracy"],
            "eval/planned_families": point["planned_families"],
            "eval/planned_episodes": point["planned_episodes"],
            "eval/missing_episodes": 0,
        },
    )
    summary.finish()
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
