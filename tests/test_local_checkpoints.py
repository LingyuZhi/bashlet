"""One configuration drives training and verified local checkpoint evaluation."""

# ruff: noqa: F811
import json
from pathlib import Path

import pytest
import yaml
from test_cli import cli
from test_local_snapshots import base_fixture, local_service_config, tensor_file
from test_opsd_epochs import epoch_pilot  # noqa: F401
from test_vllm import FakeVLLM, completion, service  # noqa: F401

from bashlet.backends.encoding import QwenChatMLEncoding


def test_checkpoint_cli_resumes_local_evaluation_and_training_from_one_config(
    epoch_pilot: tuple[Path, Path],
    service: FakeVLLM,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, formal = epoch_pilot
    base = base_fixture(tmp_path)
    tensor_file(
        base / "model.safetensors",
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": [2, 2],
        },
    )
    settings = yaml.safe_load(config.read_text())
    settings.update(
        epochs=1,
        train_task_ids=["mbpp/602"],
        evaluation_seeds=[4],
        output_dir=str(formal),
        evaluation={
            "base_path": str(base),
            "model": {
                "base_url": service.base_url,
                "model": "test-qwen",
                "api_key_env": None,
                "max_retries": 0,
            },
        },
    )
    config.write_text(yaml.safe_dump(settings))
    # Create step-zero state; the SDK is in memory and never makes paid calls.
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    failed = cli("train-opsd", str(config), cwd=tmp_path, timeout=90)
    assert failed.returncode == 2, failed.stdout + failed.stderr
    (formal / "evaluation").rename(formal / "test-double-evaluation")
    (tmp_path / "eval-interrupted").unlink()
    monkeypatch.setenv("FAKE_USE_LOCAL_EVALUATION", "1")
    output = formal / "evaluation"
    merged = output / "checkpoints/step-0/local/merged-model"
    encoding = QwenChatMLEncoding(
        str(base), revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", thinking=True
    )
    service.version, service.tokenizer = "0.17.1", encoding
    service.models = [
        {
            "id": "test-qwen",
            "root": str(merged),
            "parent": None,
            "max_model_len": 8192,
        }
    ]
    service.server_config = local_service_config(base)
    service.server_config["model_config"]["model"] = str(merged)
    reply = (
        "Done.\n</think>\n```bash\n"
        "printf 'def double(n): return n * 2\\n' > solution.py\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    service.responses.extend([completion([65]), completion(encoding.encode(reply))])
    interrupted = cli("evaluate-checkpoint", str(config), cwd=tmp_path, timeout=90)
    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    evaluation_path = output / "baseline/seed-4/evaluation.json"
    before = json.loads(evaluation_path.read_text())
    assert [row["outcome"] for row in before["tasks"]] == ["passed", "not_started"]
    first = output / "baseline/seed-4" / before["tasks"][0]["run"] / "run.json"
    first_bytes = first.read_bytes()
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_EVAL")
    service.responses.extend([completion([65]), completion(encoding.encode(reply))])
    resumed = cli("evaluate-checkpoint", str(config), cwd=tmp_path, timeout=90)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert first.read_bytes() == first_bytes
    assert len(list((output / "baseline/seed-4/episodes").iterdir())) == 2
    assert not (tmp_path / "sdk-updates.json").exists()
    report_path = output / "baseline/report.json"
    report = json.loads(report_path.read_text())
    assert report["status"] == "complete" and report["mean_accuracy"] == 1.0
    assert report["backend_counts"] == {"vllm": 2}
    report_path.write_text(json.dumps(report, indent=4, sort_keys=True) + "\n")
    report_bytes = report_path.read_bytes()
    cached = cli("evaluate-checkpoint", str(config), cwd=tmp_path, timeout=60)
    assert cached.returncode == 0, cached.stdout + cached.stderr
    assert report_path.read_bytes() == report_bytes
    original = evaluation_path.read_bytes()
    evaluation_path.write_bytes(original + b"\n")
    rejected = cli("evaluate-checkpoint", str(config), cwd=tmp_path)
    assert rejected.returncode == 2 and "evidence changed" in rejected.stderr
    evaluation_path.write_bytes(original)
    dev = Path(settings["dev_plan"])
    original = dev.read_bytes()
    dev.write_bytes(original + b"\n")
    rejected = cli("evaluate-checkpoint", str(config), cwd=tmp_path)
    assert rejected.returncode == 2 and "conditions or data changed" in rejected.stderr
    dev.write_bytes(original)
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
    trained = cli("train-opsd", str(config), "--resume", cwd=tmp_path, timeout=90)
    assert trained.returncode == 0, trained.stdout + trained.stderr
    pilot = json.loads((formal / "pilot.json").read_text())
    assert pilot["optimizer_step"] == 1 and pilot["training_complete"]
    assert pilot["development_curve"]["final"]["status"] == "complete"
    assert len(list((output / "baseline/seed-4/episodes").iterdir())) == 2
    assert len(list((output / "final/seed-4/episodes").iterdir())) == 2
    assert not (formal / "evaluation-routing.json").exists()


def test_local_final_test_rejects_an_incomplete_training_endpoint(
    tmp_path: Path,
) -> None:
    formal = tmp_path / "formal"
    formal.mkdir()
    (formal / "pilot.json").write_text(
        json.dumps(
            {
                "training_complete": False,
                "epochs_completed": 1,
                "optimizer_step": 2,
                "configuration": {"pilot": {"epochs": 2}},
                "main_result": None,
            }
        )
    )
    config = tmp_path / "experiment.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "train_manifest": "manifest.json",
                "train_task_ids": ["task"],
                "dev_plan": "dev.json",
                "condition": [],
                "output_dir": str(formal),
                "evaluation": {
                    "base_path": "base",
                    "model": {"base_url": "http://unused"},
                },
            }
        )
    )
    result = cli("evaluate-checkpoint", str(config), "--phase", "test", cwd=tmp_path)
    assert result.returncode == 2 and "complete final epoch" in result.stderr
    assert not (formal / "evaluation").exists()
