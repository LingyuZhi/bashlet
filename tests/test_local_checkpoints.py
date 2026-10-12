"""Local checkpoint evaluation through the CLI boundary."""

# ruff: noqa: F811

import json
from pathlib import Path

import pytest
import yaml
from test_cli import cli
from test_local_snapshots import base_fixture, local_service_config, tensor_file
from test_opsd_epochs import epoch_pilot  # noqa: F401
from test_vllm import FakeVLLM, completion, service  # noqa: F401

from bashlet.config import config_record, load_config
from bashlet.encoding import QwenChatMLEncoding


def test_checkpoint_cli_exports_weights_and_continues_only_the_missing_task(
    epoch_pilot: tuple[Path, Path],
    service: FakeVLLM,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_config, formal = epoch_pilot
    settings = yaml.safe_load(pilot_config.read_text())
    settings.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    pilot_config.write_text(yaml.safe_dump(settings))
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    failed = cli(
        "train-opsd",
        str(pilot_config),
        "--output-dir",
        str(formal),
        cwd=tmp_path,
        timeout=90,
    )
    assert failed.returncode == 2, failed.stdout + failed.stderr
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_EVAL")
    source = formal / "evaluation/baseline/seed-4/evaluation.json"
    source_bytes = source.read_bytes()
    base = base_fixture(tmp_path)
    tensor_file(
        base / "model.safetensors",
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": [2, 2],
        },
    )
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    output = tmp_path / "local-eval"
    merged = output / "checkpoints/step-0/local/merged-model"
    encoding = QwenChatMLEncoding(str(base), revision=revision, thinking=True)
    condition = config_record(load_config([Path(settings["condition"][0])]))
    condition["model"] = {
        "class": "bashlet.vllm.VLLMModel",
        "base_url": service.base_url,
        "model": "test-qwen",
        "model_revision": revision,
        "tokenizer": str(base),
        "tokenizer_revision": revision,
        "thinking": True,
        "api_key_env": None,
        "max_retries": 0,
        "sampling": condition["model"]["sampling"],
    }
    condition_path = tmp_path / "local-condition.yaml"
    condition_path.write_text(yaml.safe_dump(condition))
    service.version = "0.17.1"
    service.tokenizer = encoding
    service.models = [
        {"id": "test-qwen", "root": str(merged), "parent": None, "max_model_len": 8192}
    ]
    service.server_config = local_service_config(base)
    service.server_config["model_config"]["model"] = str(merged)
    reply = (
        "Done.\n</think>\n```bash\n"
        "printf 'def double(n): return n * 2\\n' > solution.py\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    service.responses.extend([completion([65]), completion(encoding.encode(reply))])
    local = tmp_path / "local-eval.yaml"
    local.write_text(
        yaml.safe_dump(
            {
                "pilot_config": str(pilot_config),
                "pilot_run": str(formal),
                "output_dir": str(output),
                "base_path": str(base),
                "model_revision": revision,
                "condition": [str(condition_path)],
                "historical_baseline": str(formal / "evaluation/baseline"),
                "retry_infrastructure": True,
            }
        )
    )
    evaluated = cli(
        "evaluate-checkpoint",
        str(local),
        "--phase",
        "baseline",
        cwd=tmp_path,
        timeout=90,
    )
    assert evaluated.returncode == 0, evaluated.stdout + evaluated.stderr
    report = json.loads((output / "baseline/report.json").read_text())
    assert report["status"] == "complete"
    assert report["mean_accuracy"] == 1.0
    assert report["backend_counts"] == {"pytrio": 1, "vllm": 1}
    assert source.read_bytes() == source_bytes
    assert len(list((output / "baseline/seed-4/episodes").iterdir())) == 1
    assert not (tmp_path / "sdk-updates.json").exists()

    # The training controller imports the local stage before its next resume.
    pilot_path = formal / "pilot.json"
    pilot = json.loads(pilot_path.read_text())
    pilot["evaluations"]["baseline"] = json.loads(
        (output / "baseline/stage.json").read_text()
    )
    pilot_path.write_text(json.dumps(pilot))
    resumed = cli(
        "evaluate-checkpoint", str(local), "--phase", "baseline", cwd=tmp_path
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert len(list((output / "baseline/seed-4/episodes").iterdir())) == 1
    assert not (tmp_path / "sdk-updates.json").exists()

    previous = json.loads(source_bytes)
    previous["tasks"][0]["outcome"] = "model_failure"
    source.write_text(json.dumps(previous))
    rejected = cli(
        "evaluate-checkpoint", str(local), "--phase", "baseline", cwd=tmp_path
    )
    assert rejected.returncode == 2
    assert "differs from its saved receipt" in rejected.stderr
    source.write_bytes(source_bytes)

    dev = Path(settings["dev_plan"])
    dev_bytes = dev.read_bytes()
    dev.write_bytes(dev_bytes + b"\n")
    changed_data = cli(
        "evaluate-checkpoint", str(local), "--phase", "baseline", cwd=tmp_path
    )
    assert changed_data.returncode == 2
    assert "frozen conditions or data changed" in changed_data.stderr
    dev.write_bytes(dev_bytes)

    # Resume the real training controller against fake external SDK/HTTP services.
    # Baseline stays cached; only the saved final checkpoint gets new dev episodes.
    updated = output / "checkpoints/step-1/local/merged-model"
    service.models[0]["root"] = str(updated)
    service.server_config["model_config"]["model"] = str(updated)
    service.responses.extend(
        [
            completion([65]),
            completion(encoding.encode(reply)),
            completion(encoding.encode(reply)),
        ]
    )
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP", "1")
    interrupted = cli(
        "train-opsd",
        str(pilot_config),
        "--output-dir",
        str(formal),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    assert not (output / "final/report.json").exists()
    assert len(json.loads((tmp_path / "sdk-updates.json").read_text())) == 1
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP")
    trained = cli(
        "train-opsd",
        str(pilot_config),
        "--output-dir",
        str(formal),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    final = json.loads((output / "final/report.json").read_text())
    assert final["backend_counts"] == {"pytrio": 0, "vllm": 2}
    assert final["mean_accuracy"] == 1.0
    pilot = json.loads(pilot_path.read_text())
    assert pilot["training_complete"] is True
    assert pilot["main_result"]["optimizer_step"] == 1
    assert len(json.loads((tmp_path / "sdk-updates.json").read_text())) == 1
    assert len(list((output / "baseline/seed-4/episodes").iterdir())) == 1
    assert source.read_bytes() == source_bytes


def test_local_final_test_rejects_an_incomplete_training_endpoint(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    (formal / "pilot.json").write_text(
        json.dumps(
            {
                "training_complete": False,
                "epochs_completed": 3,
                "optimizer_step": 804,
                "configuration": {"pilot": {"epochs": 4}},
                "main_result": None,
            }
        )
    )
    config = tmp_path / "local.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "pilot_run": str(formal),
                "output_dir": str(tmp_path / "evaluations"),
            }
        )
    )
    result = cli("evaluate-checkpoint", str(config), "--phase", "test", cwd=tmp_path)
    assert result.returncode == 2
    assert "complete final epoch" in result.stderr
    assert not (tmp_path / "evaluations").exists()
