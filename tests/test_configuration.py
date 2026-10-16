"""Configuration composition and run/data separation at public loading seams."""

# ruff: noqa: F811
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml
from test_cohort_freeze import frozen_cohort  # noqa: F401

from bashlet.config import load_config, read_yaml
from bashlet.distillation.config import (
    RunConfig,
    resolve_condition,
    validate_training_selection,
)


def test_references_resolve_at_their_source_and_unknown_keys_still_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    parent = shared / "conditions.yaml"
    parent.write_text(
        yaml.safe_dump(
            {
                "budget": {"max_steps": 7},
                "environment": {"python": {"source": "python"}},
                "model": {"tokenizer": "./tokenizer"},
            }
        )
    )
    child = tmp_path / "experiment.yaml"
    child.write_text(
        "extends: shared/conditions.yaml\nbudget:\n  max_generated_tokens: 1234\n"
    )
    monkeypatch.chdir("/tmp")
    config = load_config([child])
    assert config.budget.max_steps == 7
    assert config.budget.max_generated_tokens == 1234
    assert config.environment.python.source == str(shared / "python")
    assert config.model.parameters["tokenizer"] == str(shared / "tokenizer")
    saved = asdict(config)
    parent.write_text("budget:\n  max_steps: 9\n")
    assert asdict(config) == saved
    assert load_config([child]).budget.max_steps == 9
    parent.write_text("budget:\n  typo: 9\n")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config([child])


def test_inheritance_rejects_cycles_and_invalid_references(tmp_path: Path) -> None:
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    first.write_text("extends: second.yaml\n")
    second.write_text("extends: first.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        read_yaml(first)
    first.write_text("extends: [false]\n")
    with pytest.raises(ValueError, match="configuration paths"):
        read_yaml(first)


def test_new_run_changes_epochs_seeds_and_budget_without_changing_frozen_data(
    tmp_path: Path,
    frozen_cohort: Path,
) -> None:
    root = tmp_path / "experiment"
    root.mkdir()
    condition = root / "condition.yaml"
    condition.write_text("budget:\n  max_steps: 7\n  max_generated_tokens: 2222\n")
    selection = frozen_cohort / "training-selection.json"
    original = selection.read_bytes()
    source = root / "run.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "train_manifest": str(frozen_cohort / "manifest.json"),
                "training_selection": str(selection),
                "condition": ["condition.yaml"],
                "dev_plan": str(frozen_cohort / "harness-dev.json"),
                "epochs": 2,
                "shuffle_seed": 912,
                "evaluation_seeds": [11, 23],
                "backend": {"seed": 57},
                "group": "new-experiment",
                "spend_config": "spend.yaml",
            }
        )
    )
    config = RunConfig.read(source)
    resolved, backend = resolve_condition(config)
    manifest = json.loads((frozen_cohort / "manifest.json").read_text())
    assert validate_training_selection(config, manifest)
    assert config.epochs == 2 and config.evaluation_seeds == [11, 23]
    assert resolved.budget.max_steps == 7
    assert resolved.model.sampling.seed == backend.seed == 57
    assert config.backend["group"] == config.training["group"] == "new-experiment"
    assert config.backend["spend_config"] == str(root / "spend.yaml")
    assert selection.read_bytes() == original
