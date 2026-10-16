"""Identify local inference artifacts and verify the service loading them."""

import hashlib
import json
from pathlib import Path
from typing import Any

from bashlet.backends.pytrio import MODEL
from bashlet.harness.models import Encoding


def _snapshot_identity(record: dict[str, Any]) -> str:
    adapter = record["adapter"]
    if adapter is None:
        return f"vllm/{MODEL}@{record['model_revision']}"
    digest = (
        hashlib.sha256(
            json.dumps(adapter["merged"]["files_sha256"], sort_keys=True).encode()
        ).hexdigest()
        if adapter["load_method"] == "merged"
        else adapter["files_sha256"]["adapter_model.safetensors"]
    )
    return f"vllm/{MODEL}@{adapter['checkpoint']['path']}:{digest}"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoding_condition(details: dict[str, Any]) -> dict[str, Any]:
    # Fast tokenizer.json is authoritative; vocab/merges are redundant slow assets.
    # Keep added-token and special-token configs, which can change the fast codec.
    return {
        key: value
        for key, value in details.items()
        if key not in {"tokenizer", "revision", "files_sha256"}
    } | {
        "files_sha256": {
            key: value
            for key, value in details["files_sha256"].items()
            if key
            not in {"config.json", "generation_config.json", "vocab.json", "merges.txt"}
        }
    }


class LocalSnapshot:
    """Verify local artifact hashes and the actual service before admitting evidence."""

    def __init__(self, path: Path, model_revision: str, encoding: Encoding) -> None:
        self.record = json.loads(path.read_text())
        record = self.record
        if record.get("schema_version") != 1 or record.get("base_model") != MODEL:
            raise ValueError("Unknown local snapshot manifest")
        if record["model_revision"] != model_revision:
            raise ValueError("Local snapshot model revision mismatch")
        self.base_path = Path(record["base_path"])
        files = record["base_files_sha256"]
        required = {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        }
        actual = {
            file.name
            for file in self.base_path.iterdir()
            if file.is_file() and file.suffix in {".json", ".jinja", ".safetensors"}
        }
        if (
            not self.base_path.is_absolute()
            or not required <= files.keys()
            or set(files) != actual
            or not any(name.endswith(".safetensors") for name in files)
        ):
            raise ValueError(
                "Local snapshot must identify every base and tokenizer artifact"
            )
        if record["service_version"] != "0.17.1" or record["dtype"] != "bfloat16":
            raise ValueError("Unknown verified local inference configuration")
        base_config = json.loads((self.base_path / "config.json").read_text())
        if base_config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
            raise ValueError("Local snapshot must identify Qwen3.5 model structure")
        for name, digest in record["base_files_sha256"].items():
            if file_sha256(self.base_path / name) != digest:
                raise ValueError(f"Local snapshot file hash mismatch: {name}")
        if encoding_condition(record["encoding_details"]) != encoding_condition(
            encoding.describe()
        ):
            raise ValueError("Local snapshot encoding mismatch")
        adapter = record["adapter"]
        kind = (
            "base"
            if adapter is None
            else ("merged" if adapter["load_method"] == "merged" else "lora")
        )
        if record["snapshot_kind"] != kind or record["identity"] != _snapshot_identity(
            record
        ):
            raise ValueError("Local snapshot identity does not match its artifacts")
        if adapter is not None:
            for name, digest in adapter["files_sha256"].items():
                if file_sha256(Path(adapter["path"]) / name) != digest:
                    raise ValueError(f"Local adapter file hash mismatch: {name}")
            if adapter["load_method"] == "merged":
                for name, digest in adapter["merged"]["files_sha256"].items():
                    if file_sha256(Path(adapter["merged"]["path"]) / name) != digest:
                        raise ValueError(f"Local merged file hash mismatch: {name}")
        self.identity: str = record["identity"]
        self.service_version: str = record["service_version"]
        self.details = record | {"snapshot_manifest_sha256": file_sha256(path)}

    def verify_service(
        self, served: dict[str, Any], config: dict[str, Any], models: list[Any]
    ) -> dict[str, Any]:
        model = config.get("model_config", {})
        adapter = self.record["adapter"]
        merged = adapter and adapter["load_method"] == "merged"
        expected_root = (
            adapter["merged"]["path"]
            if merged
            else (adapter["path"] if adapter else str(self.base_path))
        )
        expected_model = expected_root if merged else str(self.base_path)
        if adapter and not merged:
            parent: dict[str, Any] = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("id") == served.get("parent")
                ),
                {},
            )
            if parent.get("root") != str(self.base_path) or not config.get(
                "lora_config"
            ):
                raise ValueError("Service has not loaded the requested LoRA adapter")
        if (
            served.get("root") != expected_root
            or ((adapter is None or merged) and served.get("parent") is not None)
            or model.get("model") != expected_model
            or model.get("tokenizer") != str(self.base_path)
        ):
            raise ValueError("Service model identity does not match the local snapshot")
        dtype = str(model.get("dtype")).removeprefix("torch.")
        if dtype != self.record["dtype"]:
            raise ValueError("Service precision does not match the local snapshot")
        if (
            model.get("quantization") is not None
            or model.get("hf_overrides")
            or model.get("hf_config_path") is not None
        ):
            raise ValueError(
                "Service must use the recorded unquantized model configuration"
            )
        load = config.get("load_config", {})
        if (
            load.get("load_format") not in {"auto", "safetensors"}
            or load.get("safetensors_load_strategy") not in {"lazy", "eager"}
            or load.get("model_loader_extra_config") != {}
            or load.get("ignore_patterns") not in ([], ["original/**/*"])
        ):
            raise ValueError("Service must load the recorded safetensors weights")
        if (
            model.get("generation_config") != "vllm"
            or not isinstance(model.get("override_generation_config"), dict)
            or model.get("logits_processors")
        ):
            raise ValueError("Service sampling defaults must be explicitly traceable")
        return {
            "dtype": dtype,
            "max_model_len": model.get("max_model_len"),
            "quantization": model.get("quantization"),
            "backend": "vllm",
            "service_version": self.service_version,
            "model_revision": self.record["model_revision"],
            "load": {
                key: load[key]
                for key in (
                    "load_format",
                    "safetensors_load_strategy",
                    "model_loader_extra_config",
                    "ignore_patterns",
                )
            },
            "base_weights_sha256": {
                name: digest
                for name, digest in self.record["base_files_sha256"].items()
                if name.endswith(".safetensors")
            },
            "model": {
                key: model.get(key)
                for key in (
                    "enforce_eager",
                    "generation_config",
                    "override_generation_config",
                    "seed",
                    "model_impl",
                    "override_attention_dtype",
                    "disable_sliding_window",
                    "disable_cascade_attn",
                    "logprobs_mode",
                    "max_logprobs",
                    "enable_prompt_embeds",
                )
            },
            "cache": {
                key: config.get("cache_config", {}).get(key)
                for key in (
                    "block_size",
                    "cache_dtype",
                    "mamba_cache_dtype",
                    "mamba_ssm_cache_dtype",
                    "mamba_cache_mode",
                    "enable_prefix_caching",
                    "gpu_memory_utilization",
                )
            },
            "parallel": {
                key: config.get("parallel_config", {}).get(key)
                for key in (
                    "tensor_parallel_size",
                    "pipeline_parallel_size",
                    "data_parallel_size",
                    "distributed_executor_backend",
                )
            },
            "scheduler": {
                key: config.get("scheduler_config", {}).get(key)
                for key in (
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "enable_chunked_prefill",
                )
            },
            "compilation": {
                key: config.get("compilation_config", {}).get(key)
                for key in ("mode", "cudagraph_mode", "backend")
            },
            "lora": {
                key: config.get("lora_config", {}).get(key)
                for key in ("max_lora_rank", "lora_dtype")
            }
            if config.get("lora_config")
            else None,
        }
