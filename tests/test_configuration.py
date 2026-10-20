"""Configuration composition and run/data separation at public loading seams."""

# ruff: noqa: F811
import hashlib
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from task_fixtures import role_sources  # noqa: F401
from test_cli import cli
from test_cohort_freeze import frozen_cohort  # noqa: F401
from test_opsd_watchdog import _cli, _events, _fixture, _tmux_calls, _write_json

from bashlet.config import config_record, load_config, read_yaml
from bashlet.distillation.config import (
    RunConfig,
    WatchConfig,
    resolve_condition,
    validate_training_selection,
)
from bashlet.spend import SpendPolicy, read_spend_config


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


def test_inline_sections_share_paths_and_keep_preparation_separate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "run.yaml"
    settings: dict[str, Any] = {
        "train_manifest": "manifest.json",
        "train_task_ids": ["task/1"],
        "dev_plan": "dev.json",
        "condition": {
            "budget": {"max_steps": 7},
            "environment": {"python": {"source": "python"}},
            "model": {"thinking": True},
        },
        "preparation": {"budget": {"max_steps": 2}},
        "spend": {"limits": asdict(SpendPolicy(groups={"train": 1}))},
        "group": "train",
    }
    source.write_text(yaml.safe_dump(settings))
    run = RunConfig.read(source)
    condition, _ = resolve_condition(run)
    assert condition.budget.max_steps == 7
    assert condition.environment.python.source == str(tmp_path / "python")
    assert load_config([source], section="preparation").budget.max_steps == 2
    assert run.backend["spend_config"] == str(source)
    assert SpendPolicy.read(source).groups == {"train": 1}
    assert read_spend_config(source) == settings["spend"]
    settings["backend"] = {"thinking": False}
    source.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="thinking settings must match"):
        resolve_condition(RunConfig.read(source))
    settings["condition"]["budget"]["typo"] = 3
    source.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config([source])


def test_service_config_cli_needs_no_training_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "run.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "training_selection": "missing.json",
                "evaluation": {
                    "base_path": "./base",
                    "model": {"base_url": "http://127.0.0.1:8004"},
                    "server": {
                        "session": "test-service",
                        "cuda_module": "cuda/12.8",
                        "options": {"port": 8004, "max-model-len": 32768},
                    },
                },
            }
        )
    )
    output = tmp_path / "service/serve.yaml"
    result = cli(
        "write-vllm-config", str(source), "--output", str(output), cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert yaml.safe_load(output.read_text()) == {
        "model": str(tmp_path / "base"),
        "tokenizer": str(tmp_path / "base"),
        "port": 8004,
        "max-model-len": 32768,
    }


@pytest.mark.parametrize("change", [None, "condition", "data", "pending", "update"])
def test_direct_run_monitor_preserves_recovery_gates(
    tmp_path: Path, change: str | None
) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["run_root"]
    source = fixture["root"] / "configs/current.yaml"
    for name in ("manifest.json", "dev.json"):
        (root / name).write_text("{}")
    settings = {
        "train_manifest": str(root / "manifest.json"),
        "dev_plan": str(root / "dev.json"),
        "train_task_ids": ["task/1"],
        "condition": {},
        "group": "train",
        "ledger_path": str(fixture["ledger"]),
        "output_dir": str(root),
        "monitor": {
            "repository_root": str(fixture["root"]),
            "worker_session": "bashlet-worker-test",
            "monitor_session": "bashlet-monitor-test",
            "poll_seconds": 1,
            "backoff_min_seconds": 60,
            "backoff_max_seconds": 600,
        },
    }
    source.write_text(yaml.safe_dump(settings))
    fixture["config_path"] = source
    run = RunConfig.read(source)
    condition, _ = resolve_condition(run)
    frozen = {
        "pilot": asdict(run),
        "condition": config_record(condition),
        "dev_plan_sha256": hashlib.sha256((root / "dev.json").read_bytes()).hexdigest(),
        "train_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    pilot = {
        "configuration": frozen,
        "status": "interrupted",
        "error_type": "KeyboardInterrupt",
        "optimizer_step": 0,
        "training_tasks": [],
    }
    _write_json(root / "pilot.json", pilot)
    _write_json(root / "training/opsd.json", {"pending_update": None, "batches": []})
    if change == "condition":
        settings["condition"] = {"budget": {"max_steps": 7}}
        source.write_text(yaml.safe_dump(settings))
    elif change == "data":
        (root / "manifest.json").write_text('{"changed": true}')
    elif change == "pending":
        with sqlite3.connect(fixture["ledger"]) as connection:
            connection.execute(
                "INSERT INTO requests VALUES ('in-flight', 'train', 'pending')"
            )
    elif change == "update":
        _write_json(
            root / "training/opsd.json", {"pending_update": {"step": 1}, "batches": []}
        )
    monitor = WatchConfig.read(source)
    assert monitor.preflight_required is False
    assert monitor.formal_pilot == root / "pilot.json"
    assert monitor.ledger_groups == ("train",)
    result = _cli(fixture)
    assert result.returncode == 0, result.stdout + result.stderr
    decision = _events(fixture)[-1]["decision"]
    assert decision == ("waiting_backoff" if change is None else "blocked")
    assert not any(
        call[0] in {"new-session", "respawn-pane"} for call in _tmux_calls(fixture)
    )
