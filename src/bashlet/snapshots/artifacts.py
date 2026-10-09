"""Export existing sampler checkpoints and prepare validated local weights."""

import hashlib
import json
import math
import shutil
import stat
import tarfile
import zipfile
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path, PurePosixPath
from typing import Any, cast

import ml_dtypes  # noqa: F401 -- registers the BF16 NumPy dtype
import numpy as np
from safetensors import TensorSpec, deserialize, safe_open, serialize_file

from ..pytrio import MODEL, SDK_VERSION, _credential
from .manifest import _snapshot_identity, encoding_condition, file_sha256

_DTYPE_NAMES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
}


def _raw_tensors(path: Path) -> dict[str, dict[str, Any]]:
    read = cast(Callable[[bytes], list[tuple[str, dict[str, Any]]]], deserialize)
    return dict(read(path.read_bytes()))


def _write(path: Path, record: dict[str, Any]) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=True, allow_nan=False))


def _extract_adapter(archive: Path, destination: Path) -> dict[str, str]:
    container: zipfile.ZipFile | tarfile.TarFile
    if zipfile.is_zipfile(archive):
        container = zipfile.ZipFile(archive)
    elif tarfile.is_tarfile(archive):
        container = tarfile.open(archive)
    else:
        raise ValueError("Checkpoint download is neither ZIP nor TAR")
    with container:
        names = (
            container.namelist()
            if isinstance(container, zipfile.ZipFile)
            else container.getnames()
        )
        configs = [
            name for name in names if PurePosixPath(name).name == "adapter_config.json"
        ]
        if len(configs) != 1:
            raise ValueError("Checkpoint archive must contain exactly one adapter")
        parent = PurePosixPath(configs[0]).parent
        files = {}
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe checkpoint archive path")
            if isinstance(container, zipfile.ZipFile):
                zip_info = container.getinfo(name)
                regular = not stat.S_ISLNK(zip_info.external_attr >> 16)
                directory = zip_info.is_dir()
            else:
                tar_info = container.getmember(name)
                regular, directory = tar_info.isfile(), tar_info.isdir()
            if directory or not path.is_relative_to(parent):
                continue
            if not regular:
                raise ValueError("Checkpoint archive must contain regular files")
            relative = str(path.relative_to(parent))
            if relative in files:
                raise ValueError("Duplicate checkpoint archive file")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            source = (
                container.open(name)
                if isinstance(container, zipfile.ZipFile)
                else container.extractfile(name)
            )
            if source is None:
                raise ValueError("Checkpoint archive file is unreadable")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            files[relative] = file_sha256(target)
    if not {"adapter_config.json", "adapter_model.safetensors"} <= files.keys():
        raise ValueError("Checkpoint must supply a PEFT safetensors adapter")
    return files


def _opsd_export_record(
    training: dict[str, Any], backend: dict[str, Any], step: int
) -> dict[str, Any]:
    """Resolve an update against both the algorithm and backend records."""
    if backend["sdk_version"] != SDK_VERSION:
        raise ValueError(f"Expected OPSD backend PyTRIO {SDK_VERSION}")
    if training["encoding_details"] != backend["encoding"]:
        raise ValueError("OPSD algorithm and backend encoding records disagree")
    updates = [
        item
        for item in backend["updates"]
        if item["optimizer_step"] == step and item["status"] == "completed"
    ]
    if len(updates) != 1:
        raise ValueError("Select exactly one existing sampler checkpoint")
    update = updates[0]
    base_model = backend["config"]["base_model"]
    policy = f"pytrio/{base_model}@{update['sampler']}"
    algorithm_updates = [
        batch["update"]
        for batch in training["batches"]
        if batch["status"] == "completed"
        and batch.get("update", {}).get("optimizer_step") == step
    ]
    if (
        len(algorithm_updates) != 1
        or algorithm_updates[0]["student_policy"] != policy
        or algorithm_updates[0]["training_state"] != update["training_state"]
        or not update["sampler"].startswith(
            f"trio://{update['training_run_cuid']}/sampler_weights/"
        )
    ):
        raise ValueError("OPSD algorithm and backend checkpoint records disagree")
    return {
        "metadata": {
            "base_model": base_model,
            "server_model_revision": backend["server_model_revision"],
            "encoding": training["snapshot"]["encoding"],
        },
        "checkpoints": [
            {
                "optimizer_step": step,
                "training_run_cuid": update["training_run_cuid"],
                "sampler": {"kind": "sampler_weights", "path": update["sampler"]},
            }
        ],
    }


def export_snapshot(
    training_run: Path,
    step: int,
    output_dir: Path,
    *,
    api_key_env: str = "PYTRIO_API_KEY",
    dotenv_path: Path = Path(".env"),
) -> Path:
    """Download a saved sampler, without creating a trainer or a sampler client."""
    training_path = training_run / "training.json"
    if not training_path.exists():
        training_path = training_run / "opsd.json"
    training_bytes = training_path.read_bytes()
    training = json.loads(training_bytes)
    training_digest = hashlib.sha256(training_bytes).hexdigest()
    source_bytes = {training_path.name: training_bytes}
    if training_path.name == "opsd.json":
        backend_bytes = (training_run / "backend.json").read_bytes()
        backend = json.loads(backend_bytes)
        source_bytes["backend.json"] = backend_bytes
        training = _opsd_export_record(training, backend, step)
        encoding_details = backend["encoding"]
    else:
        encoding_details = (
            json.loads(Path(training["config"]["dataset"]).read_text())[
                "encoding_details"
            ]
            if "config" in training
            else None
        )
    if version("pytrio") != SDK_VERSION:
        raise ValueError(f"Expected PyTRIO {SDK_VERSION}")
    if training["metadata"]["base_model"] != MODEL:
        raise ValueError(f"Local inference requires {MODEL}")
    candidates = [
        item
        for item in training["checkpoints"]
        if item["optimizer_step"] == step
        and item["sampler"]["kind"] == "sampler_weights"
    ]
    if len(candidates) != 1:
        raise ValueError("Select exactly one existing sampler checkpoint")
    saved = candidates[0]
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = output_dir / "export.json"
    if destination.exists():
        raise ValueError("Export already exists; preserve it or use a new directory")
    source_records = {}
    for name, content in source_bytes.items():
        digest = hashlib.sha256(content).hexdigest()
        archived = training_run.resolve() / "training-records" / f"{digest}.json"
        archived.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            with archived.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if archived.read_bytes() != content:
                raise ValueError("Archived training record hash mismatch") from None
        source_records[name] = {"path": str(archived), "sha256": digest}
    import pytrio as trio

    trio.configure(
        home=str(output_dir / "pytrio-sdk"),
        retry_max_attempts=1,
        control_retry_timeout_seconds=0,
        request_timeout_seconds=60,
        log_level="WARNING",
    )
    client = trio.ServiceClient(api_key=_credential(api_key_env, dotenv_path))
    rest = client.create_rest_client()
    remote_run = rest.get_training_run(saved["training_run_cuid"]).result(timeout=60)
    if remote_run.base_model != MODEL or not remote_run.is_lora or remote_run.corrupted:
        raise ValueError("Checkpoint training run is incompatible or corrupted")
    checkpoints = rest.list_checkpoints(saved["training_run_cuid"]).result(timeout=60)
    matches = [
        item
        for item in checkpoints.checkpoints
        if item.path == saved["sampler"]["path"] and item.checkpoint_type == "sampler"
    ]
    if len(matches) != 1:
        raise ValueError("Remote sampler identity does not match the saved checkpoint")
    checkpoint = matches[0]
    archive = output_dir / "checkpoint.archive"
    download = rest.download_checkpoint(
        checkpoint.checkpoint_id, archive, resume=False
    ).result(timeout=300)
    files = _extract_adapter(archive, output_dir / "adapter")
    adapter_config = json.loads(
        (output_dir / "adapter/adapter_config.json").read_text()
    )
    if adapter_config.get("peft_type") != "LORA" or (
        adapter_config.get("r") != remote_run.lora_rank
    ):
        raise ValueError(
            "Exported adapter does not match the remote LoRA configuration"
        )
    _write(
        destination,
        {
            "schema_version": 1,
            "sdk_version": SDK_VERSION,
            "training_record": source_records[training_path.name]["path"],
            "training_record_sha256": training_digest,
            "source_records": source_records,
            "optimizer_step": step,
            "base_model": MODEL,
            "server_model_revision": training["metadata"]["server_model_revision"],
            "training_encoding": training["metadata"]["encoding"],
            "training_encoding_details": encoding_details,
            "remote_training_run": remote_run.model_dump(mode="json"),
            "checkpoint": checkpoint.model_dump(mode="json"),
            "download": download.model_dump(mode="json"),
            "archive_sha256": file_sha256(archive),
            "files_sha256": files,
            "adapter_config": adapter_config,
        },
    )
    return destination


def prepare_local_snapshot(
    base_path: Path,
    model_revision: str,
    output_dir: Path,
    *,
    export_path: Path | None = None,
    merge: bool = False,
) -> Path:
    """Bind local weights to their tokenizer and an immutable inference manifest."""
    if len(model_revision) != 40 or any(
        char not in "0123456789abcdef" for char in model_revision
    ):
        raise ValueError("Record a full Hugging Face model revision")
    base_path = base_path.resolve()
    config = json.loads((base_path / "config.json").read_text())
    if config.get("model_type") not in {"qwen3_5", "qwen3_5_text"}:
        raise ValueError("Local inference requires Qwen3.5 base model structure")
    weights = sorted(base_path.glob("*.safetensors"))
    if not weights:
        raise ValueError("Local base model weights are missing")
    from ..encoding import QwenChatMLEncoding

    encoding = QwenChatMLEncoding(str(base_path), revision=model_revision)
    files = {
        path.name: file_sha256(path)
        for path in sorted(base_path.iterdir())
        if path.is_file() and path.suffix in {".json", ".jinja", ".safetensors"}
    }
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination = output_dir / "snapshot.json"
    if destination.exists():
        raise ValueError("Snapshot manifest already exists; preserve the original")
    adapter = None
    if merge and export_path is None:
        raise ValueError("Merging requires an exported adapter")
    if export_path is not None:
        exported = json.loads(export_path.read_text())
        if exported["base_model"] != MODEL:
            raise ValueError("Exported adapter base model mismatch")
        training_encoding = exported.get("training_encoding_details")
        if not isinstance(training_encoding, dict) or (
            encoding_condition(training_encoding)
            != encoding_condition(encoding.describe())
        ):
            raise ValueError("Exported adapter training encoding mismatch")
        source = export_path.parent / "adapter"
        for name, digest in exported["files_sha256"].items():
            if file_sha256(source / name) != digest:
                raise ValueError(f"Exported adapter file hash mismatch: {name}")
        adapter = _normalize_adapter(source, output_dir / "adapter", config, weights)
        adapter.update(
            {
                "checkpoint": exported["checkpoint"],
                "optimizer_step": exported["optimizer_step"],
                "export_sha256": file_sha256(export_path),
                "source_files_sha256": exported["files_sha256"],
                "source_adapter_config": exported["adapter_config"],
                "archive_sha256": exported["archive_sha256"],
                "server_model_revision": exported["server_model_revision"],
                "training_record_sha256": exported["training_record_sha256"],
                "load_method": "merged" if merge else "lora",
            }
        )
        if merge:
            adapter["merged"] = _merge_adapter(
                base_path, weights, config, adapter, output_dir / "merged-model"
            )
        elif (
            config.get("text_config", config).get("tie_word_embeddings")
            and "lm_head" in adapter["config"]["target_modules"]
        ):
            raise ValueError(
                "vLLM cannot load this tied lm_head LoRA; prepare with --merge"
            )
    record = {
        "schema_version": 1,
        "base_model": MODEL,
        "model_revision": model_revision,
        "base_path": str(base_path),
        "base_files_sha256": files,
        "encoding_details": encoding.describe(),
        "snapshot_kind": ("merged" if merge else "lora") if adapter else "base",
        "service_version": "0.17.1",
        "dtype": "bfloat16",
        "server_model_revision": None,
        "revision_limitation": (
            "Cloud base weight revision is not published; "
            "local/cloud bitwise equality is unverified"
        ),
        "adapter": adapter,
    }
    record["identity"] = _snapshot_identity(record)
    _write(destination, record)
    return destination


def _normalize_adapter(
    source: Path,
    destination: Path,
    base_config: dict[str, Any],
    weights: list[Path],
) -> dict[str, Any]:
    """Repack TRIO's split GDN Q/K/V into the HF fused projection for vLLM.

    Stack A and block-diagonalize B, then zero-pad to one supported rank while
    preserving alpha/r. Operate on floating-point storage bits, including BF16;
    no dtype conversion or rounding is introduced by this repacking.
    """
    config = json.loads((source / "adapter_config.json").read_text())
    name = config.get("base_model_name_or_path", "")
    if not isinstance(name, str) or not (name == MODEL or name.endswith("/" + MODEL)):
        raise ValueError("Adapter names a different base model")
    rank = config.get("r")
    if config.get("peft_type") != "LORA" or type(rank) is not int or rank <= 0:
        raise ValueError("Adapter must use a positive, fixed LoRA rank")
    alpha = config.get("lora_alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Adapter requires a finite positive LoRA scaling factor")
    if (
        any(
            config.get(key)
            for key in (
                "use_dora",
                "use_rslora",
                "use_qalora",
                "rank_pattern",
                "alpha_pattern",
                "modules_to_save",
                "fan_in_fan_out",
                "alora_invocation_tokens",
                "layer_replication",
                "lora_bias",
            )
        )
        or config.get("bias", "none") != "none"
    ):
        raise ValueError("Unsupported adapter transformation or variable LoRA scaling")
    base_shapes = {}
    for file in weights:
        with safe_open(file, framework="np") as tensors:
            for key in tensors.keys():
                base_shapes[key] = tensors.get_slice(key).get_shape()
    text = base_config.get("text_config", base_config)
    if text.get("tie_word_embeddings") and "lm_head.weight" not in base_shapes:
        embedding = base_shapes.get("model.language_model.embed_tokens.weight")
        if embedding is not None:
            base_shapes["lm_head.weight"] = embedding
    raw = _raw_tensors(source / "adapter_model.safetensors")
    pairs: dict[str, dict[str, Any]] = {}
    for key, tensor in raw.items():
        key = key.removeprefix("base_model.model.")
        module, separator, suffix = key.rpartition(".lora_")
        if not separator or suffix not in {"A.weight", "B.weight"}:
            raise ValueError(f"Unsupported adapter tensor: {key}")
        pairs.setdefault(module, {})[suffix[0]] = tensor
    if not pairs:
        raise ValueError("Empty LoRA adapter")
    dtypes = {tensor["dtype"] for pair in pairs.values() for tensor in pair.values()}
    if len(dtypes) != 1 or not dtypes <= {"BF16", "F16", "F32", "F64"}:
        raise ValueError("Adapter requires one supported floating-point dtype")
    dtype = dtypes.pop()
    width = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8}[dtype]
    matrices = {}
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"} or any(
            len(tensor["shape"]) != 2 for tensor in pair.values()
        ):
            raise ValueError(f"Incomplete LoRA tensor pair: {module}")
        a, b = pair["A"], pair["B"]
        if a["shape"][0] != rank or b["shape"][1] != rank:
            raise ValueError(f"LoRA tensor rank mismatch: {module}")
        matrices[module] = (
            np.frombuffer(a["data"], dtype=f"u{width}").reshape(a["shape"]),
            np.frombuffer(b["data"], dtype=f"u{width}").reshape(b["shape"]),
        )
    fused_groups = {
        module.rsplit("_", 1)[0]
        for module in matrices
        if module.endswith(
            (
                ".linear_attn.in_proj_q",
                ".linear_attn.in_proj_k",
                ".linear_attn.in_proj_v",
            )
        )
    }
    for group in sorted(fused_groups):
        segments = [group + "_" + label for label in "qkv"]
        if any(segment not in matrices for segment in segments):
            raise ValueError(f"Incomplete split Q/K/V adapter: {group}")
        sizes = [
            text["linear_num_key_heads"] * text["linear_key_head_dim"],
            text["linear_num_key_heads"] * text["linear_key_head_dim"],
            text["linear_num_value_heads"] * text["linear_value_head_dim"],
        ]
        original = [matrices.pop(segment) for segment in segments]
        if [b.shape[0] for _, b in original] != sizes:
            raise ValueError(f"Split Q/K/V dimensions do not match base: {group}")
        a = np.concatenate([a for a, _ in original], axis=0)
        b = np.zeros((sum(sizes), rank * 3), dtype=f"u{width}")
        offset = 0
        for index, (_, part) in enumerate(original):
            b[offset : offset + part.shape[0], index * rank : (index + 1) * rank] = part
            offset += part.shape[0]
        matrices[group + "_qkv"] = (a, b)
    required_rank = max(a.shape[0] for a, _ in matrices.values())
    target_rank = next(
        (value for value in (8, 16, 32, 64) if value >= required_rank), None
    )
    if target_rank is None:
        raise ValueError("Normalized adapter exceeds the supported local LoRA rank")
    normalized = {}
    for module, (a, b) in matrices.items():
        if base_shapes.get(module + ".weight") != [b.shape[0], a.shape[1]]:
            raise ValueError(f"Adapter structure does not match base tensor: {module}")
        a = np.pad(a, ((0, target_rank - a.shape[0]), (0, 0)))
        b = np.pad(b, ((0, 0), (0, target_rank - b.shape[1])))
        for label, matrix in (("A", a), ("B", b)):
            normalized[module + f".lora_{label}.weight"] = matrix
    destination.mkdir(parents=True, exist_ok=True)
    # TensorSpec borrows pointers. Keep every array in normalized alive through save.
    specs = {
        key: TensorSpec(
            dtype=_DTYPE_NAMES[dtype],
            shape=matrix.shape,
            data_ptr=matrix.ctypes.data,
            data_len=matrix.nbytes,
        )
        for key, matrix in normalized.items()
    }
    serialize_file(specs, str(destination / "adapter_model.safetensors"))
    prepared_config = config | {
        "r": target_rank,
        "lora_alpha": config["lora_alpha"] * target_rank / rank,
        "target_modules": sorted({module.rsplit(".", 1)[-1] for module in matrices}),
    }
    _write(destination / "adapter_config.json", prepared_config)
    return {
        "path": str(destination),
        "config": prepared_config,
        "files_sha256": {
            file: file_sha256(destination / file)
            for file in (
                "adapter_config.json",
                "adapter_model.safetensors",
            )
        },
        "normalization": {
            "version": "qwen35-split-qkv-v1",
            "source_rank": rank,
            "rank": target_rank,
            "fused_qkv_groups": len(fused_groups),
            "validated_target_modules": len(matrices),
            "dtype": dtype,
        },
    }


def _merge_adapter(
    base_path: Path,
    weights: list[Path],
    base_config: dict[str, Any],
    adapter: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Apply validated plain LoRA in FP32, then round to each base tensor dtype.

    Head-only LoRA needs an independent output weight when the base ties it to
    input embeddings. Clone the original head before applying its delta, keeping
    the input embeddings unchanged. Record this configuration change.
    """
    destination.mkdir(parents=True, exist_ok=True)
    raw_adapter = _raw_tensors(Path(adapter["path"]) / "adapter_model.safetensors")

    def array(tensor: dict[str, Any]) -> Any:
        if tensor["dtype"] not in _DTYPE_NAMES:
            raise ValueError("Merge supports unquantized floating-point weights only")
        return np.frombuffer(
            tensor["data"], dtype=_DTYPE_NAMES[tensor["dtype"]]
        ).reshape(tensor["shape"])

    updates = {
        key.removesuffix(".lora_A.weight")
        for key in raw_adapter
        if key.endswith(".lora_A.weight")
    }
    scale = adapter["config"]["lora_alpha"] / adapter["config"]["r"]
    text = base_config.get("text_config", base_config)
    untied = bool(text.get("tie_word_embeddings") and "lm_head" in updates)
    applied = set()
    weight_map = {}
    total_size = 0
    for file in weights:
        raw = _raw_tensors(file)
        if untied and "model.language_model.embed_tokens.weight" in raw:
            raw["lm_head.weight"] = raw["model.language_model.embed_tokens.weight"]
        arrays = {}
        for key, tensor in raw.items():
            module = key.removesuffix(".weight") if key.endswith(".weight") else None
            original = array(tensor)
            if module in updates:
                a = array(raw_adapter[module + ".lora_A.weight"]).astype(np.float32)
                b = array(raw_adapter[module + ".lora_B.weight"]).astype(np.float32)
                merged = original.astype(np.float32) + (b @ a) * scale
                if not np.isfinite(merged).all():
                    raise ValueError(f"Non-finite merged weights: {module}")
                original = merged.astype(original.dtype)
                if not np.isfinite(original).all():
                    raise ValueError(
                        f"Merged weights overflow base precision: {module}"
                    )
                applied.add(module)
            arrays[key] = original
            total_size += original.nbytes
            weight_map[key] = file.name
        # Arrays and the raw buffers stay alive while TensorSpec borrows pointers.
        specs = {
            key: TensorSpec(
                dtype=_DTYPE_NAMES[raw[key]["dtype"]],
                shape=value.shape,
                data_ptr=value.ctypes.data,
                data_len=value.nbytes,
            )
            for key, value in arrays.items()
        }
        serialize_file(specs, str(destination / file.name), metadata={"format": "pt"})
    if applied != updates:
        raise ValueError(
            f"Adapter targets were not merged: {sorted(updates - applied)}"
        )
    for file in base_path.iterdir():
        if (
            file.is_file()
            and file.suffix in {".json", ".jinja"}
            and file.name not in {"config.json", "model.safetensors.index.json"}
        ):
            shutil.copyfile(file, destination / file.name)
    merged_config = json.loads(json.dumps(base_config))
    if untied:
        merged_config["tie_word_embeddings"] = False
        merged_config.get("text_config", merged_config)["tie_word_embeddings"] = False
    _write(destination / "config.json", merged_config)
    _write(
        destination / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
    )
    return {
        "path": str(destination),
        "files_sha256": {
            file.name: file_sha256(file)
            for file in sorted(destination.iterdir())
            if file.is_file()
        },
        "version": "fp32-lora-merge-v1",
        "compute_dtype": "float32",
        "output_dtype": "each original base tensor dtype",
        "untied_output_head": untied,
        "applied_target_modules": len(applied),
        "dependencies": {
            name: version(name) for name in ("numpy", "safetensors", "ml-dtypes")
        },
    }
