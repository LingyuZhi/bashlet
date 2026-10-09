"""Local snapshot preparation and sampling through the CLI and experiment seams."""

import hashlib
import io
import json
import shutil
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import ml_dtypes
import numpy as np
import pytest
import yaml
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file, save_file
from test_cli import cli
from test_mbpp import double_row, prepare
from test_vllm import FakeVLLM, completion, double_task
from test_vllm import service as service

from bashlet import QwenChatMLEncoding, run_experiment
from bashlet.vllm import VLLMModel

BASE = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SAMPLER = "trio://run-test/sampler_weights/sampler-0012"


def export_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_format: str = "zip"
) -> Path:
    archive = tmp_path / "remote.zip"
    config = json.dumps(
        {
            "base_model_name_or_path": BASE,
            "peft_type": "LORA",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj"],
        }
    ).encode()
    files = {
        "checkpoint/adapter_config.json": config,
        "checkpoint/adapter_model.safetensors": b"external weights",
    }
    if archive_format == "zip":
        with zipfile.ZipFile(archive, "w") as zipped:
            for name, content in files.items():
                zipped.writestr(name, content)
    else:
        with tarfile.open(archive, "w") as tarred:
            for name, content in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tarred.addfile(member, io.BytesIO(content))
    training = tmp_path / "training"
    training.mkdir()
    (training / "training.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "base_model": BASE,
                    "server_model_revision": None,
                    "encoding": "test-training-encoding",
                },
                "checkpoints": [
                    {
                        "optimizer_step": 12,
                        "sampler": {"kind": "sampler_weights", "path": SAMPLER},
                        "training_state": {
                            "kind": "training_state",
                            "path": "trio://run-test/weights/state-0012",
                        },
                        "training_run_cuid": "run-test",
                    }
                ],
            }
        )
    )
    site = tmp_path / "sitecustomize.py"
    site.write_text(
        "import shutil\n"
        "from concurrent.futures import Future\n"
        "from types import SimpleNamespace\n"
        "import pytrio\n"
        "def response(data):\n"
        "    return SimpleNamespace(**data, model_dump=lambda **kw: data)\n"
        "def done(value):\n"
        "    future = Future(); future.set_result(value); return future\n"
        "class Rest:\n"
        "    def get_training_run(self, run):\n"
        f"        return done(response(dict(training_run_id=run, base_model={BASE!r}, "
        "is_lora=True, corrupted=False, lora_rank=16)))\n"
        "    def list_checkpoints(self, run):\n"
        "        checkpoint = response(dict(checkpoint_id='ckpt-test', "
        f"checkpoint_type='sampler', path={SAMPLER!r}))\n"
        "        return done(SimpleNamespace(checkpoints=[checkpoint]))\n"
        "    def download_checkpoint(self, checkpoint, destination, **kwargs):\n"
        f"        shutil.copyfile({str(archive)!r}, destination)\n"
        "        return done(response(dict(checkpoint_id=checkpoint, "
        "status_code=200, resumed=False)))\n"
        "pytrio.ServiceClient = lambda **kwargs: SimpleNamespace("
        "workspace_cuid='workspace-test', create_rest_client=lambda: Rest())\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("PYTRIO_API_KEY", "test-key-not-for-records")
    return training


@pytest.mark.parametrize("archive_format", ["zip", "tar"])
def test_export_preserves_the_existing_sampler_identity_and_download_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive_format: str
) -> None:
    training = export_fixture(tmp_path, monkeypatch, archive_format)
    output = tmp_path / "export"
    process = cli(
        "export-snapshot",
        str(training),
        "--step",
        "12",
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((output / "export.json").read_text())
    assert record["checkpoint"]["path"] == SAMPLER
    assert record["checkpoint"]["checkpoint_id"] == "ckpt-test"
    assert record["checkpoint"]["checkpoint_type"] == "sampler"
    assert (
        record["archive_sha256"]
        == hashlib.sha256((tmp_path / "remote.zip").read_bytes()).hexdigest()
    )
    assert (
        record["files_sha256"]["adapter_model.safetensors"]
        == hashlib.sha256(b"external weights").hexdigest()
    )
    assert record["adapter_config"]["r"] == 16
    assert record["server_model_revision"] is None
    assert record["training_encoding"] == "test-training-encoding"
    archived = Path(record["training_record"])
    assert archived == (
        training / "training-records" / f"{record['training_record_sha256']}.json"
    )
    original = (training / "training.json").read_bytes()
    (training / "training.json").write_text('{"status": "resumed"}')
    assert archived.read_bytes() == original
    assert (
        hashlib.sha256(archived.read_bytes()).hexdigest()
        == (record["training_record_sha256"])
    )
    assert (output / "adapter/adapter_model.safetensors").read_bytes() == (
        b"external weights"
    )
    assert "test-key-not-for-records" not in (
        (output / "export.json").read_text() + process.stdout + process.stderr
    )


def opsd_export_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    training = export_fixture(tmp_path, monkeypatch)
    (training / "training.json").unlink()
    encoding = {"format": "test-encoding", "thinking": False}
    update = {
        "optimizer_step": 12,
        "student_policy": f"pytrio/{BASE}@{SAMPLER}",
        "training_state": "trio://run-test/weights/state-0012",
    }
    (training / "opsd.json").write_text(
        json.dumps(
            {
                "snapshot": {"encoding": "test-training-encoding"},
                "encoding_details": encoding,
                "batches": [{"status": "completed", "update": update}],
            }
        )
    )
    (training / "backend.json").write_text(
        json.dumps(
            {
                "sdk_version": "0.2.9",
                "config": {"base_model": BASE},
                "server_model_revision": None,
                "initial_training_run_cuid": "run-before-resume",
                "encoding": encoding,
                "updates": [
                    {
                        "optimizer_step": 12,
                        "status": "completed",
                        "training_run_cuid": "run-test",
                        "sampler": SAMPLER,
                        "training_state": update["training_state"],
                    }
                ],
            }
        )
    )
    return training


def test_export_opsd_uses_the_resumed_run_and_preserves_both_source_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    training = opsd_export_fixture(tmp_path, monkeypatch)
    output = tmp_path / "export"
    process = cli(
        "export-snapshot",
        str(training),
        "--step",
        "12",
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((output / "export.json").read_text())
    assert record["remote_training_run"]["training_run_id"] == "run-test"
    assert record["checkpoint"]["path"] == SAMPLER
    assert record["training_encoding_details"] == {
        "format": "test-encoding",
        "thinking": False,
    }
    assert set(record["source_records"]) == {"opsd.json", "backend.json"}
    for name, source in record["source_records"].items():
        original = (training / name).read_bytes()
        assert Path(source["path"]).read_bytes() == original
        assert source["sha256"] == hashlib.sha256(original).hexdigest()
    assert record["training_record"] == record["source_records"]["opsd.json"]["path"]


@pytest.mark.parametrize("problem", ["encoding", "checkpoint", "pending"])
def test_export_opsd_rejects_unconfirmed_checkpoint_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    training = opsd_export_fixture(tmp_path, monkeypatch)
    backend_path = training / "backend.json"
    backend = json.loads(backend_path.read_text())
    if problem == "encoding":
        backend["encoding"]["thinking"] = True
    elif problem == "checkpoint":
        backend["updates"][0]["sampler"] = SAMPLER.replace("run-test", "run-other")
    else:
        backend["updates"][0]["status"] = "pending"
    backend_path.write_text(json.dumps(backend))
    output = tmp_path / "export"
    process = cli(
        "export-snapshot",
        str(training),
        "--step",
        "12",
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode != 0
    assert not (output / "export.json").exists()


def tensor_file(path: Path, shapes: dict[str, list[int]]) -> None:
    """A tiny valid external safetensors checkpoint with literal tensor dimensions."""
    header: dict[str, Any] = {}
    offset = 0
    for name, shape in shapes.items():
        size = 4
        for dimension in shape:
            size *= dimension
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    data = json.dumps(header).encode()
    data += b" " * (-len(data) % 8)
    path.write_bytes(struct.pack("<Q", len(data)) + data + b"\0" * offset)


def base_fixture(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    base.mkdir()
    tokenizer = Path(
        snapshot_download(
            BASE, revision=REVISION, cache_dir="runs/tokenizers", local_files_only=True
        )
    )
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
    ):
        shutil.copyfile(tokenizer / name, base / name)
    tensor_file(
        base / "model.safetensors",
        {"model.language_model.layers.0.self_attn.q_proj.weight": [4, 4]},
    )
    return base


def local_service_config(base: Path) -> dict[str, Any]:
    return {
        "model_config": {
            "model": str(base),
            "tokenizer": str(base),
            "dtype": "torch.bfloat16",
            "max_model_len": 8192,
            "generation_config": "vllm",
            "override_generation_config": {},
        },
        "load_config": {
            "load_format": "auto",
            "safetensors_load_strategy": "lazy",
            "model_loader_extra_config": {},
            "ignore_patterns": ["original/**/*"],
        },
        "lora_config": None,
    }


def test_verified_local_base_uses_the_existing_experiment_and_is_traceable(
    tmp_path: Path, service: FakeVLLM
) -> None:
    base = base_fixture(tmp_path)
    output = tmp_path / "snapshot"
    prepared = cli(
        "prepare-local-snapshot",
        str(base),
        "--model-revision",
        REVISION,
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    manifest = json.loads((output / "snapshot.json").read_text())
    assert manifest["base_model"] == BASE
    assert (
        manifest["base_files_sha256"]["model.safetensors"]
        == hashlib.sha256((base / "model.safetensors").read_bytes()).hexdigest()
    )
    encoding = QwenChatMLEncoding(str(base), revision=REVISION)
    service.version = "0.17.1"
    service.models = [
        {"id": "test-qwen", "root": str(base), "parent": None, "max_model_len": 8192}
    ]
    service.tokenizer = encoding
    service.server_config = local_service_config(base)
    reply = (
        "```bash\nprintf 'def solve(n): return 2 * n\\n' > solution.py\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    service.responses.extend([completion([65]), completion(encoding.encode(reply))])
    model = VLLMModel(
        base_url=service.base_url,
        model="test-qwen",
        model_revision=REVISION,
        encoding=encoding,
        snapshot_manifest=str(output / "snapshot.json"),
    )
    result = run_experiment(
        task=double_task(), model=model, output_dir=tmp_path / "episodes"
    )
    assert result.passed, result.reason
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["policy_details"]["development_only"] is False
    assert record["policy_details"]["snapshot_kind"] == "base"
    assert record["policy_details"]["identity_verified"] is True
    assert record["policy"].startswith(f"vllm/{BASE}@{REVISION}")
    assert record["policy_details"]["server_model_revision"] is None


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "tied_head",
        "auxiliary_vocab",
        "structure",
        "encoding",
        "identity",
        "adapter_unloaded",
        "fallback",
        "missing_tokens",
        "missing_logprobs",
        "merged_adapter_unloaded",
        "merged_fallback",
        "merged_missing_tokens",
        "merged_missing_logprobs",
        "merged_dummy_loader",
        "merged_loader_extra_config",
        "merged_loader_ignore_patterns",
        "merged_missing_loader_config",
        "merged_external_generation_config",
        "merged_logits_processor",
    ],
)
def test_exported_split_projections_load_as_the_selected_local_lora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: FakeVLLM, defect: str
) -> None:
    base = base_fixture(tmp_path)
    config = json.loads((base / "config.json").read_text())
    config["text_config"].update(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    (base / "config.json").write_text(json.dumps(config))
    base_shapes = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [6, 4]
    }
    if defect == "tied_head":
        base_shapes["model.language_model.embed_tokens.weight"] = [4, 4]
    tensor_file(base / "model.safetensors", base_shapes)
    if defect == "tied_head":
        save_file(
            {
                name: np.zeros(shape, dtype=ml_dtypes.bfloat16)
                for name, shape in base_shapes.items()
            },
            base / "model.safetensors",
        )
    encoding = QwenChatMLEncoding(str(base), revision=REVISION)
    training = export_fixture(tmp_path, monkeypatch)
    manifest = tmp_path / "data.json"
    manifest.write_text(json.dumps({"encoding_details": encoding.describe()}))
    training_data = json.loads((training / "training.json").read_text())
    training_data["config"] = {"dataset": str(manifest)}
    (training / "training.json").write_text(json.dumps(training_data))
    tensors: dict[str, Any] = {}
    a_values = [
        [[1, 0, 0, 0], [0, 1, 0, 0]],
        [[0, 0, 1, 0], [0, 0, 0, 1]],
        [[1, 1, 0, 0], [0, 0, 1, 1]],
    ]
    for label, values, diagonal in zip(
        "qkv", a_values, ([2, 3], [4, 5], [6, 7]), strict=True
    ):
        prefix = f"model.language_model.layers.0.linear_attn.in_proj_{label}"
        a = np.zeros((16, 4), dtype=np.float32)
        a[:2] = values
        b = np.zeros((2, 16), dtype=np.float32)
        b[0, 0], b[1, 1] = diagonal
        tensors[prefix + ".lora_A.weight"] = a
        tensors[prefix + ".lora_B.weight"] = b
    if defect == "tied_head":
        tensors["lm_head.lora_A.weight"] = np.ones((16, 4), dtype=np.float32)
        tensors["lm_head.lora_B.weight"] = np.ones((4, 16), dtype=np.float32)
        tensors = {
            name: value.astype(ml_dtypes.bfloat16) for name, value in tensors.items()
        }
    original = tmp_path / "adapter_model.safetensors"
    save_file(tensors, original)
    with zipfile.ZipFile(tmp_path / "remote.zip", "w") as stream:
        stream.writestr(
            "adapter_config.json",
            json.dumps(
                {
                    "peft_type": "LORA",
                    "base_model_name_or_path": "/cloud/models/Qwen/Qwen3.5-4B",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "k_proj", "v_proj"],
                }
            ),
        )
        stream.write(original, "adapter_model.safetensors")
    exported = tmp_path / "export"
    downloaded = cli(
        "export-snapshot",
        str(training),
        "--step",
        "12",
        "--output-dir",
        str(exported),
        cwd=tmp_path,
    )
    assert downloaded.returncode == 0, downloaded.stdout + downloaded.stderr
    if defect == "structure":
        tensor_file(
            base / "model.safetensors",
            {"model.language_model.layers.0.linear_attn.in_proj_qkv.weight": [6, 5]},
        )
    elif defect == "auxiliary_vocab":
        # The authoritative fast tokenizer.json is unchanged; training didn't save
        # this redundant slow-tokenizer asset from the same Hub snapshot.
        (base / "vocab.json").write_text("{}")
    elif defect == "encoding":
        data = json.loads((exported / "export.json").read_text())
        data["training_encoding_details"]["files_sha256"]["tokenizer.json"] = "f" * 64
        (exported / "export.json").write_text(json.dumps(data))
    output = tmp_path / "snapshot"
    merge = defect == "tied_head" or defect.startswith("merged_")
    extra = ["--merge"] if merge else []
    prepared = cli(
        "prepare-local-snapshot",
        str(base),
        "--model-revision",
        REVISION,
        "--export",
        str(exported / "export.json"),
        "--output-dir",
        str(output),
        *extra,
        cwd=tmp_path,
    )
    if defect in {"structure", "encoding"}:
        assert prepared.returncode == 2
        assert ("structure" if defect == "structure" else "encoding") in prepared.stderr
        assert not (output / "snapshot.json").exists()
        return
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    snapshot = json.loads((output / "snapshot.json").read_text())
    adapter = snapshot["adapter"]
    assert adapter["checkpoint"]["path"] == SAMPLER
    assert adapter["load_method"] == ("merged" if merge else "lora")
    normalized = Path(adapter["path"])
    loaded = load_file(normalized / "adapter_model.safetensors")
    if defect == "tied_head":
        assert {value.dtype.name for value in loaded.values()} == {"bfloat16"}
    loaded = {name: value.astype(np.float32) for name, value in loaded.items()}
    prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    delta = 2 * loaded[prefix + ".lora_B.weight"] @ loaded[prefix + ".lora_A.weight"]
    assert delta.tolist() == [
        [4, 0, 0, 0],
        [0, 6, 0, 0],
        [0, 0, 8, 0],
        [0, 0, 0, 10],
        [12, 12, 0, 0],
        [0, 0, 14, 14],
    ]
    if merge:
        merged = Path(adapter["merged"]["path"])
    if defect == "tied_head":
        head_delta = (
            2 * loaded["lm_head.lora_B.weight"] @ loaded["lm_head.lora_A.weight"]
        )
        assert head_delta.tolist() == [[32] * 4] * 4
        merged_weights = load_file(merged / "model.safetensors")
        assert {value.dtype.name for value in merged_weights.values()} == {"bfloat16"}
        assert merged_weights["lm_head.weight"].tolist() == [[32] * 4] * 4
        assert (
            merged_weights["model.language_model.embed_tokens.weight"].tolist()
            == [[0] * 4] * 4
        )
        assert merged_weights[prefix + ".weight"].tolist() == delta.tolist()
        merged_config = json.loads((merged / "config.json").read_text())
        assert merged_config["tie_word_embeddings"] is False
        assert merged_config["text_config"]["tie_word_embeddings"] is False
    service.version = "0.17.1"
    service.tokenizer = encoding
    service.models = [
        {"id": "base", "root": str(base), "parent": None, "max_model_len": 8192},
        {
            "id": "test-qwen",
            "root": str(normalized),
            "parent": "base",
            "max_model_len": 8192,
        },
    ]
    service.server_config = local_service_config(base)
    service.server_config["lora_config"] = {"max_lora_rank": 64}
    if merge:
        service.models[-1].update(root=str(merged), parent=None)
        service.server_config["model_config"]["model"] = str(merged)
        service.server_config["lora_config"] = None
    fault = defect.removeprefix("merged_")
    if fault == "adapter_unloaded":
        service.models[-1]["root"] = str(base)
        service.models[-1]["parent"] = None
    elif defect == "identity":
        service.server_config["model_config"]["model"] = "/a/different/model"
    elif fault == "dummy_loader":
        service.server_config["load_config"]["load_format"] = "dummy"
    elif fault == "loader_extra_config":
        service.server_config["load_config"]["model_loader_extra_config"] = {
            "pt_weights_files": ["alternate.pt"]
        }
    elif fault == "loader_ignore_patterns":
        service.server_config["load_config"]["ignore_patterns"] = ["*.safetensors"]
    elif fault == "missing_loader_config":
        del service.server_config["load_config"]
    elif fault == "external_generation_config":
        service.server_config["model_config"]["generation_config"] = "/unhashed/config"
    elif fault == "logits_processor":
        service.server_config["model_config"]["logits_processors"] = ["custom.Logits"]
    reply = (
        "```bash\nprintf 'def solve(n): return 2 * n\\n' > solution.py\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    probe = completion([65])
    if fault == "fallback":
        probe[1]["model"] = "base"
    elif fault == "missing_tokens":
        del probe[1]["choices"][0]["token_ids"]
    elif fault == "missing_logprobs":
        probe[1]["choices"][0]["logprobs"] = None
    service.responses.extend([probe, completion(encoding.encode(reply))])
    model = VLLMModel(
        base_url=service.base_url,
        model="test-qwen",
        model_revision=REVISION,
        encoding=encoding,
        snapshot_manifest=str(output / "snapshot.json"),
    )
    result = run_experiment(
        task=double_task(), model=model, output_dir=tmp_path / "episodes"
    )
    if defect not in {"none", "tied_head", "auxiliary_vocab"}:
        assert result.status in {"model_error", "infrastructure_error"}, result.reason
        assert not result.trajectory and result.grade is None
        assert result.policy_details["identity_verified"] is False
        assert not (result.workspace / "solution.py").exists()
        return
    assert result.passed, result.reason
    assert result.policy_details["identity_verified"] is True
    assert SAMPLER in result.policy


def test_local_evaluations_require_loaded_identity_and_matching_runtime_conditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: FakeVLLM
) -> None:
    base = base_fixture(tmp_path)
    encoding = QwenChatMLEncoding(str(base), revision=REVISION)
    training = export_fixture(tmp_path, monkeypatch)
    data = tmp_path / "training-data.json"
    data.write_text(json.dumps({"encoding_details": encoding.describe()}))
    training_record = json.loads((training / "training.json").read_text())
    training_record["config"] = {"dataset": str(data)}
    (training / "training.json").write_text(json.dumps(training_record))
    weights = tmp_path / "weights.safetensors"
    prefix = "model.language_model.layers.0.self_attn.q_proj"
    save_file(
        {
            prefix + ".lora_A.weight": np.ones((16, 4), dtype=np.float32),
            prefix + ".lora_B.weight": np.ones((4, 16), dtype=np.float32),
        },
        weights,
    )
    with zipfile.ZipFile(tmp_path / "remote.zip", "w") as stream:
        stream.writestr(
            "adapter_config.json",
            json.dumps(
                {
                    "base_model_name_or_path": BASE,
                    "peft_type": "LORA",
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj"],
                }
            ),
        )
        stream.write(weights, "adapter_model.safetensors")
    exported = tmp_path / "export"
    process = cli(
        "export-snapshot",
        str(training),
        "--step",
        "12",
        "--output-dir",
        str(exported),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifests = []
    for name, extra in [
        ("base-snapshot", []),
        ("sft-snapshot", ["--export", str(exported / "export.json")]),
    ]:
        output = tmp_path / name
        process = cli(
            "prepare-local-snapshot",
            str(base),
            "--model-revision",
            REVISION,
            "--output-dir",
            str(output),
            *extra,
            cwd=tmp_path,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        manifests.append(output / "snapshot.json")
    snapshots = [json.loads(path.read_text()) for path in manifests]
    prepared = prepare(tmp_path, [double_row(511)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "local-smoke",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511"],
            }
        )
    )
    service.version = "0.17.1"
    service.tokenizer = encoding
    service.models = [
        {"id": "base", "root": str(base), "parent": None, "max_model_len": 8192},
        {
            "id": "test-qwen",
            "root": snapshots[1]["adapter"]["path"],
            "parent": "base",
            "max_model_len": 8192,
        },
    ]
    service.server_config = local_service_config(base)
    service.server_config["lora_config"] = {"max_lora_rank": 64}
    service.server_config["cache_config"] = {"enable_prefix_caching": False}
    config = tmp_path / "model.yaml"
    records = []
    reply = (
        "```bash\nprintf 'def double(n): return 2 * n\\n' > solution.py\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    for index, cache, overrides in [
        (0, False, {}),
        (1, False, {}),
        (1, True, {}),
        (1, False, {"repetition_penalty": 1.2, "top_k": 8, "min_p": 0.1}),
    ]:
        service.server_config["cache_config"]["enable_prefix_caching"] = cache
        service.server_config["model_config"]["override_generation_config"] = overrides
        alias = "base" if index == 0 else "test-qwen"
        for tokens in ([65], encoding.encode(reply)):
            response = completion(tokens)
            response[1]["model"] = alias
            service.responses.append(response)
        config.write_text(
            yaml.safe_dump(
                {
                    "model": {
                        "class": "bashlet.vllm.VLLMModel",
                        "base_url": service.base_url,
                        "model": alias,
                        "model_revision": REVISION,
                        "tokenizer": str(base),
                        "tokenizer_revision": REVISION,
                        "snapshot_manifest": str(manifests[index]),
                        "max_retries": 0,
                    }
                }
            )
        )
        process = cli(
            "evaluate",
            str(plan),
            "--snapshot",
            snapshots[index]["identity"],
            "-c",
            str(config),
            "--output-dir",
            str(tmp_path / "evaluations"),
            cwd=tmp_path,
            timeout=60,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        path = Path(process.stdout.strip().splitlines()[-1])
        record = json.loads(path.read_text())
        assert record["status"] == "complete" and record["summary"]["passed_tasks"] == 1
        assert record["snapshot_details"]["identity_verified"] is True
        records.append(path)
    loaded = [json.loads(path.read_text()) for path in records]
    assert loaded[0]["comparison_sha256"] == loaded[1]["comparison_sha256"]
    assert loaded[1]["comparison_sha256"] != loaded[2]["comparison_sha256"]
    assert loaded[1]["comparison_sha256"] != loaded[3]["comparison_sha256"]
    for changed in records[2:]:
        mismatch = cli(
            "select-sft",
            str(training),
            "--baseline",
            str(records[0]),
            "--evaluation",
            str(changed),
            cwd=tmp_path,
        )
        assert mismatch.returncode == 2 and "comparison conditions" in mismatch.stderr
    valid = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(records[0]),
        "--evaluation",
        str(records[1]),
        cwd=tmp_path,
    )
    assert valid.returncode == 2 and "dual-track" in valid.stderr
    assert not (training / "selection.json").exists()
    relabeled = loaded[1]
    relabeled["snapshot_details"]["identity_verified"] = False
    forged = tmp_path / "relabeled.json"
    forged.write_text(json.dumps(relabeled))
    refused = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(records[0]),
        "--evaluation",
        str(forged),
        cwd=tmp_path,
    )
    assert refused.returncode == 2 and "verified" in refused.stderr
    legacy = yaml.safe_load(config.read_text())
    del legacy["model"]["snapshot_manifest"]
    config.write_text(yaml.safe_dump(legacy))
    refused = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        f"dev/vllm/test-qwen@{REVISION}",
        "-c",
        str(config),
        "--output-dir",
        str(tmp_path / "legacy"),
        cwd=tmp_path,
    )
    assert (
        refused.returncode == 2 and "Experimental evaluation requires" in refused.stderr
    )
    assert not (tmp_path / "legacy").exists()
