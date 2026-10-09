"""Epoch visits and checkpoint evaluation through the paid-workflow CLI seam."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from test_cli import cli
from test_mbpp import double_row, prepare
from test_monitoring import external_sdks

from bashlet import SpendLedger, SpendPolicy


@pytest.fixture
def epoch_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    prepared = prepare(
        tmp_path, [double_row(602), double_row(603), double_row(511), double_row(512)]
    )
    dev = tmp_path / "dev.json"
    dev.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "epoch-fixture",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511", "mbpp/512"],
            }
        )
    )
    condition = tmp_path / "condition.yaml"
    condition.write_text(yaml.safe_dump({"budget": {"max_steps": 3}}))
    tokenizer = Path(
        "runs/tokenizers/models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ).resolve()
    policy = yaml.safe_load(Path("configs/spend.yaml").read_text())
    policy["limits"]["groups"].update(
        {
            "epoch-train": 4,
            "epoch-baseline": 1,
            "epoch-middle": 1,
            "epoch-final": 1,
        }
    )
    policy_path = tmp_path / "spend.yaml"
    policy_path.write_text(yaml.safe_dump(policy))
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(
        yaml.safe_dump(
            {
                "train_manifest": str(prepared / "manifest.json"),
                "train_task_ids": ["mbpp/602", "mbpp/603"],
                "epochs": 2,
                "shuffle_seed": 4,
                "evaluation_seeds": [4, 5],
                "evaluate_each_epoch": True,
                "evaluation_groups": {
                    "baseline": "epoch-baseline",
                    "intermediate": "epoch-middle",
                    "final": "epoch-final",
                },
                "dev_plan": str(dev),
                "condition": [str(condition)],
                "ledger_path": str(tmp_path / "spend.sqlite3"),
                "backend": {
                    "tokenizer_cache_dir": str(tokenizer.parents[2]),
                    "spend_config": str(policy_path),
                    "group": "epoch-train",
                },
                "training": {"spend_config": str(policy_path), "group": "epoch-train"},
                "monitoring": {"group": "epoch-fixture"},
                "stop_on_training_budget": True,
                "defer_decision": True,
            }
        )
    )
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import opsd_sdk\nopsd_sdk.install()\n")
    monkeypatch.setenv("pytrio_api_key", "test-trio-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_OPSD_TOKENIZER", str(tokenizer))
    return pilot, tmp_path / "pilot-run"


def test_each_epoch_resamples_current_student_and_evaluates_paired_seeds(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    config, root = epoch_pilot
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["status"] == "complete"
    assert record["optimizer_step"] == 4
    assert record["epochs_completed"] == 2
    visits = record["training_tasks"]
    assert [row["epoch"] for row in visits] == [1, 1, 2, 2]
    assert len({row["visit_id"] for row in visits}) == 4
    assert len({row["episode"] for row in visits}) == 4
    training = json.loads((root / "training/opsd.json").read_text())
    teacher = record["initial_snapshot"]["identity"]
    for index, batch in enumerate(training["batches"]):
        assert batch["teacher_policy"] == teacher
        assert batch["student_policy"] == (
            teacher
            if index == 0
            else training["batches"][index - 1]["update"]["student_policy"]
        )
    assert {
        phase: stage["optimizer_step"] for phase, stage in record["evaluations"].items()
    } == {
        "baseline": 0,
        "epoch-1": 2,
        "final": 4,
    }
    for stage in record["evaluations"].values():
        assert set(stage["runs"]) == {"4", "5"}
        assert stage["status"] == "complete"
    for seed in ("4", "5"):
        before = json.loads(
            Path(record["evaluations"]["baseline"]["runs"][seed]["path"]).read_text()
        )
        after = json.loads(
            Path(record["evaluations"]["final"]["runs"][seed]["path"]).read_text()
        )
        assert before["comparison_sha256"] == after["comparison_sha256"]
        assert (
            before["summary"]["passed_tasks"] == after["summary"]["passed_tasks"] == 2
        )
    cost = record["cost"]
    resumed = cli(
        "train-opsd", str(config), "--output-dir", str(root), "--resume", cwd=tmp_path
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((root / "pilot.json").read_text())["cost"] == cost


@pytest.mark.parametrize("interrupt_step", [3, 4])
def test_resume_recovers_second_epoch_update_without_reusing_first_epoch_episode(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_step: int,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP", str(interrupt_step))
    interrupted = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert interrupted.returncode == 2
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == interrupt_step
    assert record["training_tasks"][interrupt_step - 1]["status"] == "running"
    training = json.loads((root / "training/opsd.json").read_text())
    completed_episode = training["batches"][interrupt_step - 1]["episode"]
    frozen = {
        ref["path"]: ref["sha256"]
        for stage in record["evaluations"].values()
        for ref in stage["runs"].values()
    }
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "pilot.json").read_text())
    assert after["optimizer_step"] == 4
    recovered = after["training_tasks"][interrupt_step - 1]
    assert recovered["episode"] == completed_episode
    assert recovered["optimizer_step"] == interrupt_step
    assert recovered["recovered_completed_update"]
    batches = json.loads((root / "training/opsd.json").read_text())["batches"]
    assert len(batches) == 4
    assert after["training_progress"]["optimizer_steps"] == 4
    assert after["training_progress"]["trajectories"] == 4
    assert after["training_progress"]["supervised_tokens"] == sum(
        batch["supervised_tokens"] for batch in batches
    )
    assert after["training_progress"]["nonzero_signal_tokens"] == sum(
        batch["nonzero_signal_tokens"] for batch in batches
    )
    for path, digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_training_budget_stop_preserves_final_evaluation_and_reuses_same_checkpoint(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    config, root = epoch_pilot
    configuration = yaml.safe_load(config.read_text())
    policy_path = Path(configuration["training"]["spend_config"])
    policy = yaml.safe_load(policy_path.read_text())
    policy["limits"]["groups"]["epoch-train"] = 0.7
    policy_path.write_text(yaml.safe_dump(policy))
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["training_stop"] == "budget_refused"
    assert record["optimizer_step"] == 2
    assert record["epochs_completed"] == 1
    assert record["evaluations"]["final"]["status"] == "complete"
    assert record["evaluations"]["final"]["reused_from"] == "epoch-1"
    assert (
        record["evaluations"]["final"]["runs"]
        == record["evaluations"]["epoch-1"]["runs"]
    )
    assert record["comparison"]["matched_complete"]
    assert not record["cost"]["pending_upper_bound_cny"]


def test_incomplete_final_seed_does_not_claim_a_complete_paired_comparison(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    config, root = epoch_pilot
    configuration = yaml.safe_load(config.read_text())
    policy_path = Path(configuration["training"]["spend_config"])
    policy = yaml.safe_load(policy_path.read_text())
    policy["limits"]["groups"]["epoch-final"] = 0.31
    policy_path.write_text(yaml.safe_dump(policy))
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 2
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == 4
    assert record["evaluations"]["final"]["status"] == "incomplete"
    assert record["evaluations"]["final"]["runs"]["4"]["status"] == "complete"
    assert not record["comparison"]["matched_complete"]
    assert record["decision"]["action"] == "insufficient_evidence"


def test_pending_second_epoch_update_is_retained_without_new_paid_requests(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_OPTIM_FAIL_STEP", "3")
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 2
    training = json.loads((root / "training/opsd.json").read_text())
    assert training["optimizer_step"] == 2
    assert training["pending_update"]["optimizer_step"] == 3
    configuration = yaml.safe_load(config.read_text())
    ledger = SpendLedger(
        Path(configuration["ledger_path"]),
        policy=SpendPolicy.read(Path(configuration["training"]["spend_config"])),
    )
    before = ledger.snapshot()
    assert before["pending_cny"] > 0
    monkeypatch.delenv("FAKE_OPTIM_FAIL_STEP")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 2
    assert ledger.snapshot()["entries"] == before["entries"]
    after = json.loads((root / "training/opsd.json").read_text())
    assert after["pending_update"] == training["pending_update"]
    assert after["optimizer_step"] == 2


def test_monitor_start_failure_before_sampling_can_resume_the_same_visit(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_WANDB_TRAIN_START_FAIL", "1")
    failed = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert failed.returncode == 2
    before = json.loads((root / "pilot.json").read_text())
    assert before["optimizer_step"] == 1
    assert "W&B startup failed" in before["reason"]
    assert before["training_tasks"][1]["status"] == "not_started"
    assert before["training_tasks"][1]["recovered_unstarted_visit"]
    episode = before["training_tasks"][0]["episode"]
    baseline = before["evaluations"]["baseline"]["runs"]
    monkeypatch.delenv("FAKE_WANDB_TRAIN_START_FAIL")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "pilot.json").read_text())
    assert after["optimizer_step"] == 4
    assert after["training_tasks"][0]["episode"] == episode
    assert after["evaluations"]["baseline"]["runs"] == baseline
    assert len(json.loads((root / "training/opsd.json").read_text())["batches"]) == 4
