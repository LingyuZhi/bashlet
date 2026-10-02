"""SFT training, recovery and selection via the CLI experiment boundary."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_cli import cli
from test_demonstrations import TrainingAdapter, training_task
from test_monitoring import external_sdks

from bashlet import QwenChatMLEncoding, run_experiment
from bashlet.config import load_config
from bashlet.demonstrations import reference_demonstration


def training_input(tmp_path: Path) -> Path:
    task = training_task()
    encoding = QwenChatMLEncoding(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cache_dir="runs/tokenizers",
    )
    adapter = TrainingAdapter(encoding.identity)
    config = load_config()
    result = run_experiment(
        task=task,
        model=reference_demonstration(task=task, encoding=encoding, config=config),
        config=config,
        output_dir=tmp_path / "episodes",
        sft_adapter=adapter,
    )
    assert result.passed
    datum, metadata = adapter.requests[0]
    samples = tmp_path / "examples.jsonl"
    samples.write_text(
        json.dumps({"metadata": metadata, "datum": datum.model_dump()}) + "\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "source": metadata["source"],
                "encoding": encoding.identity,
                "encoding_details": encoding.describe(),
                "fingerprints": metadata["fingerprints"],
                "source_manifest_sha256": "a" * 64,
                "examples": "examples.jsonl",
                "examples_sha256": hashlib.sha256(samples.read_bytes()).hexdigest(),
                "counts": {"accepted": 1, "rejected": 0},
                "task_ids": [task.task_id],
                "tasks": [
                    {
                        "task_id": task.task_id,
                        "status": "accepted",
                        "run_dir": str(result.run_dir),
                        "run_sha256": hashlib.sha256(
                            (result.run_dir / "run.json").read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        )
    )
    return manifest


def training_config(tmp_path: Path, manifest: Path) -> Path:
    config = tmp_path / "training.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": str(manifest),
                "base_model": "Qwen/Qwen3.5-4B",
                "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "steps": 2,
                "batch_size": 1,
                "checkpoint_every": 1,
                "rank": 8,
                "learning_rate": 0.0001,
                "ledger_path": str(tmp_path / "spend.sqlite3"),
                "group": "pytrio-smoke",
                "monitor": {"group": "test-training"},
            }
        )
    )
    return config


def test_training_waits_for_updates_and_resumes_optimizer_without_relogging_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = training_input(tmp_path)
    config = training_config(tmp_path, manifest)
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import training_sdk\ntraining_sdk.install()\n")
    monkeypatch.setenv("PYTRIO_API_KEY", "test-trio-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    output = tmp_path / "training"
    args = ["train-sft", str(config), "--output-dir", str(output)]
    first = cli(*args, "--stop-after", "1", cwd=tmp_path, timeout=45)
    assert first.returncode == 0, first.stdout + first.stderr
    state: dict[str, Any] = json.loads((output / "training.json").read_text())
    assert state["optimizer_step"] == 1
    assert state["checkpoints"][0]["sampler"]["kind"] == "sampler_weights"
    assert state["checkpoints"][0]["training_state"]["kind"] == "training_state"
    second = cli(*args, "--resume", cwd=tmp_path, timeout=45)
    assert second.returncode == 0, second.stdout + second.stderr
    final = json.loads((output / "training.json").read_text())
    assert final["optimizer_step"] == 2
    assert final["run_id"] == state["run_id"]
    assert final["status"] == "complete"
    remote = json.loads((tmp_path / "training-service.json").read_text())
    assert remote["optimizer_step"] == 2
    assert remote["restored_optimizer"] is True
    assert remote["supervised_tokens"] > 0
    assert remote["observations_masked"] is True
    events = [
        json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()
    ]
    updates = [e["metrics"] for e in events if "train/loss" in e["metrics"]]
    assert [e["train/optimizer_step"] for e in updates] == [1, 2]
    assert all(e["train/loss"] == 2.0 for e in updates)
    assert final["cost"]["actual_cny"] is None


@pytest.mark.parametrize("failure", ["tampered", "encoding", "auth", "pending"])
def test_invalid_or_uncertain_training_never_claims_an_optimizer_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    manifest = training_input(tmp_path)
    config = training_config(tmp_path, manifest)
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import training_sdk\ntraining_sdk.install()\n")
    monkeypatch.setenv("PYTRIO_API_KEY", "test-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    if failure == "tampered":
        with (tmp_path / "examples.jsonl").open("a") as stream:
            stream.write("{}\n")
    elif failure == "encoding":
        settings = yaml.safe_load(config.read_text())
        settings["model_revision"] = "different-tokenizer-revision"
        config.write_text(yaml.safe_dump(settings))
    elif failure == "auth":
        monkeypatch.setenv("FAKE_WANDB_AUTH_FAIL", "1")
    else:
        monkeypatch.setenv("FAKE_OPTIM_FAIL", "1")
    output = tmp_path / "training"
    process = cli(
        "train-sft", str(config), "--output-dir", str(output), cwd=tmp_path, timeout=45
    )
    assert process.returncode != 0
    assert not (tmp_path / "training-service.json").exists()
    events = output / "metrics.jsonl"
    if events.exists():
        assert "train/loss" not in events.read_text()
    if failure == "pending":
        record = json.loads((output / "training.json").read_text())
        assert record["optimizer_step"] == 0
        assert record["pending_update"] == 1
        assert record["cost"]["pending_upper_bound_cny"] > 0
        again = cli(
            "train-sft",
            str(config),
            "--output-dir",
            str(output),
            "--resume",
            cwd=tmp_path,
            timeout=45,
        )
        assert again.returncode != 0 and "reconciled" in again.stderr


def test_selection_requires_comparable_dev_results_and_preserves_snapshot_roles(
    tmp_path: Path,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    checkpoints = [
        {
            "optimizer_step": step,
            "sampler": {
                "kind": "sampler_weights",
                "path": f"trio://run/sampler-{step}",
            },
            "training_state": {
                "kind": "training_state",
                "path": f"trio://run/state-{step}",
            },
        }
        for step in (12, 24)
    ]
    (training / "training.json").write_text(
        json.dumps(
            {"checkpoints": checkpoints, "metadata": {"base_model": "Qwen/Qwen3.5-4B"}}
        )
    )
    base: dict[str, Any] = {
        "status": "complete",
        "evidence_kind": "experimental",
        "plan": {"split": "dev", "task_ids": ["mbpp/511", "mbpp/512"]},
        "comparison": {"fixed": "conditions"},
        "comparison_sha256": hashlib.sha256(
            json.dumps({"fixed": "conditions"}, sort_keys=True).encode()
        ).hexdigest(),
        "summary": {"planned_tasks": 2, "passed_tasks": 1},
    }
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(base))
    evaluations = []
    for step, passed in ((12, 2), (24, 2)):
        path = tmp_path / f"eval-{step}.json"
        record = base | {
            "snapshot": f"pytrio/Qwen/Qwen3.5-4B@trio://run/sampler-{step}",
            "summary": {"planned_tasks": 2, "passed_tasks": passed},
        }
        path.write_text(json.dumps(record))
        evaluations.append(path)
    args = ["select-sft", str(training), "--baseline", str(baseline)]
    for path in evaluations:
        args.extend(["--evaluation", str(path)])
    selected = cli(*args, cwd=tmp_path)
    assert selected.returncode == 0, selected.stdout + selected.stderr
    selection = json.loads((training / "selection.json").read_text())
    assert (
        selection["optimizer_step"] == 12
    )  # Equal pass rates prefer the earlier update.
    assert selection["student_snapshot"] == selection["frozen_teacher_snapshot"]
    assert selection["student_snapshot"] == "trio://run/sampler-12"
    assert selection["optimizer_transition"] == "reset-for-opsd"
    assert selection["sft_training_state"]["path"] == "trio://run/state-12"
    original = (training / "selection.json").read_bytes()
    bad = json.loads(evaluations[1].read_text())
    bad["plan"]["split"] = "test"
    evaluations[1].write_text(json.dumps(bad))
    refused = cli(*args, cwd=tmp_path)
    assert refused.returncode == 2 and "development" in refused.stderr
    assert (training / "selection.json").read_bytes() == original
