"""In-memory evaluator for orchestration tests; production uses local vLLM.

The fake SDK provides deterministic model samples here. Production checkpoint
export, server identity and local evaluation are exercised in test_local_checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from bashlet.config import ExperimentConfig, config_record
from bashlet.distillation import OPSDRun
from bashlet.distillation.config import RunConfig
from bashlet.distillation.development import development_curve
from bashlet.distillation.pytrio import PyTRIOOPSD
from bashlet.evaluation.harness import evaluate
from bashlet.monitoring import MonitorConfig, RunMonitor, write_document


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
    from bashlet.evaluation.harness import _digest, _tasks

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
    # A resumed run may have already evaluated the selected checkpoint.
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
            model = adapter._model(snapshot.split("@", 1)[1], group=group)
            model.operation = "evaluation"
            model.verify_loaded_sampler()
            evaluate(
                plan_path=Path(config.dev_plan),
                snapshot=snapshot,
                evidence_kind="test-double",
                condition=resolved,
                output_dir=root,
                run_dir=directory,
                model=model,
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
