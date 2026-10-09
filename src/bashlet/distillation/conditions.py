"""Resolve training encoding and enforce the published cohort and visit plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from ..config import ExperimentConfig, load_config
from ..tasks.freeze import validate_selection
from .pytrio import PyTRIOOPSDConfig

if TYPE_CHECKING:
    from .pilot import PilotConfig


def resolve_condition(config: PilotConfig) -> tuple[ExperimentConfig, PyTRIOOPSDConfig]:
    backend = PyTRIOOPSDConfig(**config.backend)
    explicit = any(
        "thinking" in yaml.safe_load(Path(p).read_text()).get("model", {})
        for p in config.condition
    )
    condition = load_config(
        [Path(p) for p in config.condition],
        [] if explicit else [f"model.thinking={backend.thinking}"],
    )
    thinking = condition.model.parameters["thinking"]
    if "thinking" in config.backend and backend.thinking != thinking:
        raise ValueError("Student and Teacher thinking settings must match")
    return condition, replace(backend, thinking=thinking)


def validate_training_selection(
    config: PilotConfig,
    condition: ExperimentConfig,
    backend: PyTRIOOPSDConfig,
    manifest: dict[str, Any],
    visits: list[dict[str, Any]],
) -> str | None:
    if config.training_selection is None:
        return None
    root = Path(config.train_manifest).resolve().parent
    validate_selection(root, manifest)
    path = Path(config.training_selection).resolve()
    artifact = manifest.get("training_selection", {})
    if path != root / artifact.get("file", ""):
        raise ValueError("Frozen training selection must belong to the manifest")
    raw = path.read_bytes()
    selected = json.loads(raw)
    seeds = selected["seeds"]
    if (
        hashlib.sha256(raw).hexdigest() != artifact["sha256"]
        or config.train_task_ids != selected["task_ids"]
        or config.epochs != len(selected["epochs"])
        or config.shuffle_seed != seeds["epoch_shuffle"]
        or config.evaluation_seeds != seeds["evaluation"]
        or backend.seed != seeds["training"]
        or condition.model.sampling.seed != seeds["training"]
        or not config.evaluate_each_epoch
        or not config.require_complete_training
        or backend.source != "base"
        or not backend.thinking
        or [
            [v["task_id"] for v in visits if v["epoch"] == epoch]
            for epoch in range(1, config.epochs + 1)
        ]
        != selected["epochs"]
    ):
        raise ValueError("Frozen training subset, epoch order or seeds differ")
    from ..evaluation.harness import _tasks

    plan, _ = _tasks(Path(config.dev_plan))
    if (
        (Path(config.dev_plan).parent / plan["manifest"]).resolve()
        != Path(config.train_manifest).resolve()
        or plan["split"] != "dev"
        or plan["task_ids"] != manifest["cohort"]["development"]["task_ids"]
    ):
        raise ValueError("Frozen training requires the complete new development split")
    sampling = condition.model.sampling
    if (
        condition.protocol.name != "bash-block-v6"
        or condition.budget.max_steps != 20
        or condition.budget.max_generated_tokens != 16384
        or condition.budget.max_seconds != 600
        or sampling.max_tokens != 1024
        or sampling.temperature != 1.0
        or sampling.top_p != 1.0
    ):
        raise ValueError(
            "Frozen training requires the shared v6 interaction conditions"
        )
    return hashlib.sha256(raw).hexdigest()
