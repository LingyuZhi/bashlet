"""Verified SFT inputs, guarded PyTRIO updates and distinct checkpoint roles."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .demonstrations import EXECUTED_SOURCE, SOURCE, validate_executed_example
from .encoding import QwenChatMLEncoding
from .models import TaskOrigin
from .monitoring import (
    MonitorConfig,
    RunMonitor,
    code_revision,
    spend_metrics,
    write_document,
)
from .pytrio import MODEL, SDK_VERSION, _credential
from .snapshots import encoding_condition
from .spend import (
    DEFAULT_POLICY,
    SpendLedger,
    SpendPolicy,
    TokenPricing,
    summarize_cost,
)
from .tasks import validate_origin

if TYPE_CHECKING:
    import pytrio as trio


@dataclass(frozen=True)
class SFTConfig:
    dataset: str
    base_model: str = MODEL
    model_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    rank: int = 16
    seed: int = 4
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    batch_size: int = 8
    steps: int = 24
    max_epochs: int | None = None
    checkpoint_every: int = 12
    learning_rate: float = 0.0001
    ledger_path: str = "runs/spend.sqlite3"
    spend_config: str = str(DEFAULT_POLICY)
    group: str = "sft"
    timeout_seconds: int = 180
    api_key_env: str = "PYTRIO_API_KEY"
    dotenv_path: str = ".env"
    monitor: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> SFTConfig:
        config = cls(**yaml.safe_load(path.read_text()))
        if config.base_model != MODEL or not config.model_revision:
            raise ValueError("SFT must start from the confirmed Qwen3.5-4B model")
        for name in (
            "rank",
            "batch_size",
            "steps",
            "checkpoint_every",
            "timeout_seconds",
        ):
            value = getattr(config, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if config.max_epochs is not None and (
            type(config.max_epochs) is not int or config.max_epochs < 1
        ):
            raise ValueError("max_epochs must be a positive integer")
        if type(config.seed) is not int or not 4 <= config.rank <= 64:
            raise ValueError("LoRA requires integer seed and rank between 4 and 64")
        if any(
            type(getattr(config, name)) is not bool
            for name in ("train_mlp", "train_attn", "train_unembed")
        ) or not any((config.train_mlp, config.train_attn, config.train_unembed)):
            raise ValueError(
                "LoRA requires boolean module flags and a trainable module"
            )
        return config


def load_demonstrations(path: Path) -> tuple[dict[str, Any], list[trio.Datum]]:
    import pytrio as trio

    manifest = json.loads(path.read_text())
    data = (path.parent / manifest["examples"]).read_bytes()
    if (
        manifest["status"] != "complete"
        or hashlib.sha256(data).hexdigest() != manifest["examples_sha256"]
    ):
        raise ValueError("SFT dataset must be complete with its recorded content hash")
    rows = [json.loads(line) for line in data.splitlines()]
    identified = manifest["schema_version"] == 3
    identity_field = "demonstration_id" if identified else "task_id"
    accepted = {
        entry[identity_field]: entry
        for entry in manifest["tasks"]
        if entry["status"] == "accepted"
    }
    if (
        not rows
        or len(rows) != len(accepted)
        or len(rows) != manifest["counts"]["accepted"]
    ):
        raise ValueError("SFT dataset accepted count does not match its examples")
    datums = []
    seen = set()
    family_counts: dict[str, int] = {}
    for row in rows:
        meta = row["metadata"]
        origin = TaskOrigin(**meta["origin"])
        validate_origin(origin)
        if (
            origin.split != "train"
            or meta["source"] not in (SOURCE, EXECUTED_SOURCE)
            or meta["source"] != manifest["source"]
            or meta["on_policy_eligible"] is not False
            or not meta["grade"]["passed"]
            or meta["encoding"] != manifest["encoding"]
            or meta["encoding_details"] != manifest["encoding_details"]
            or meta["fingerprints"] != manifest["fingerprints"]
            or meta[identity_field] in seen
            or meta[identity_field] not in accepted
        ):
            raise ValueError(
                "SFT requires distinct, verified training-split demonstrations"
            )
        seen.add(meta[identity_field])
        entry = accepted[meta[identity_field]]
        if identified:
            if (
                not isinstance(meta[identity_field], str)
                or not meta[identity_field].strip()
                or entry["task_id"] != meta["task_id"]
                or entry["task_sha256"] != meta["task_sha256"]
                or entry["origin"] != meta["origin"]
                or family_counts.get(origin.family_id, 0) >= 2
                or (
                    family_counts.get(origin.family_id, 0)
                    and not meta.get("complementary_reason", "").strip()
                )
            ):
                raise ValueError("Invalid demonstration identity or family admission")
            family_counts[origin.family_id] = family_counts.get(origin.family_id, 0) + 1
        run_dir = path.parent / entry["run_dir"]
        if (
            hashlib.sha256((run_dir / "run.json").read_bytes()).hexdigest()
            != entry["run_sha256"]
            or json.loads((run_dir / "private/sft.json").read_text()) != row
        ):
            raise ValueError(
                "SFT example differs from its executed demonstration evidence"
            )
        if meta["source"] == EXECUTED_SOURCE:
            validate_executed_example(
                meta,
                json.loads((run_dir / "run.json").read_text()),
                run_dir,
            )
        datum = trio.Datum.model_validate(row["datum"])
        inputs = datum.model_input.to_ints()
        targets = datum.loss_fn_inputs["target_tokens"].tolist()
        weights = datum.loss_fn_inputs["weights"].tolist()
        expected = [0.0] * len(inputs)
        previous = 0
        for span in meta["supervised_spans"]:
            start, end = span["start"], span["end"]
            if not 0 < start < end <= len(inputs) + 1 or start < previous:
                raise ValueError("Invalid generated supervision span")
            expected[start - 1 : end - 1] = [1.0] * (end - start)
            previous = end
        if (
            not inputs
            or len(inputs) != len(targets)
            or inputs[1:] != targets[:-1]
            or (
                meta["source"] == EXECUTED_SOURCE
                and inputs + targets[-1:]
                != json.loads((run_dir / "run.json").read_text())["context_tokens"]
            )
            or weights != expected
            or not sum(weights)
            or sum(weights) != meta["supervised_tokens"]
            or meta["auto_shift"] is not False
            or meta["loss_fn"] != "cross_entropy"
        ):
            raise ValueError("Invalid shifted assistant-only SFT supervision")
        datums.append(datum)
    return manifest, datums


class PyTRIOTraining:
    """One reservation spans forward/backward and the completed optimizer update."""

    def __init__(
        self,
        config: SFTConfig,
        ledger: SpendLedger,
        run_id: str,
        restore: dict[str, Any] | None = None,
    ) -> None:
        from importlib.metadata import version

        import pytrio as trio

        if version("pytrio") != SDK_VERSION:
            raise ValueError(f"Expected PyTRIO {SDK_VERSION}")
        self.config, self.ledger, self.run_id = config, ledger, run_id
        self.pricing = TokenPricing(
            yaml.safe_load(Path(config.spend_config).read_text())
        )
        self.catalog = self.pricing.verify(config.base_model, config.timeout_seconds)
        trio.configure(
            home=str(ledger.path.parent / "pytrio-sdk"),
            retry_max_attempts=1,
            control_retry_timeout_seconds=0,
            training_retry_timeout_seconds=0,
            log_level="WARNING",
            request_timeout_seconds=config.timeout_seconds,
        )
        self.service = trio.ServiceClient(
            api_key=_credential(config.api_key_env, Path(config.dotenv_path))
        )
        ledger.bind_account(self.service.workspace_cuid)
        supported = next(
            (
                m
                for m in self.service.get_server_capabilities().supported_models
                if m.model_name == config.base_model
            ),
            None,
        )
        if supported is None or not supported.training.available:
            raise ValueError("Workspace does not support training the selected model")
        self.limits = supported.training
        if restore:
            if restore["kind"] != "training_state":
                raise ValueError("Sampler weights cannot restore optimizer progress")
            self.client = self.service.create_training_client_from_state_with_optimizer(
                restore["path"]
            )
        else:
            self.client = self.service.create_lora_training_client(
                base_model=config.base_model,
                rank=config.rank,
                seed=config.seed,
                train_mlp=config.train_mlp,
                train_attn=config.train_attn,
                train_unembed=config.train_unembed,
            )
        if self.client.model_id != config.base_model:
            raise ValueError("Remote training model identity differs")

    def update(self, batch: list[trio.Datum], step: int) -> dict[str, Any]:
        import pytrio as trio

        lengths = [len(d.model_input.to_ints()) for d in batch]
        for value, limit in (
            (len(batch), self.limits.max_batch_size),
            (max(lengths), self.limits.max_seq_len),
            (sum(lengths), self.limits.max_tokens_per_request),
        ):
            if limit is not None and value > limit:
                raise ValueError("Training batch exceeds the remote capacity")
        payload = {
            "run_id": self.run_id,
            "optimizer_step": step,
            "training_run_cuid": self.client.training_run_cuid,
            "batch_sha256": hashlib.sha256(
                json.dumps([d.model_dump() for d in batch], sort_keys=True).encode()
            ).hexdigest(),
            "base_model": self.config.base_model,
            "input_tokens": sum(lengths),
        }
        entry = self.ledger.reserve(
            operation="sft_update",
            group=self.config.group,
            payload=payload,
            upper_cny=self.pricing.training_bound(sum(lengths)),
        )
        start = time.monotonic()
        try:
            self.ledger.annotate(
                entry, {"stage": "forward_backward", "run_id": self.run_id}
            )
            forward = self.client.forward_backward(
                batch, loss_fn="cross_entropy", auto_shift=False
            ).result(timeout=self.config.timeout_seconds)
            self.ledger.annotate(
                entry, {"stage": "optim_step", "forward_metrics": forward.metrics}
            )
            optimizer = self.client.optim_step(
                trio.AdamParams(learning_rate=self.config.learning_rate)
            ).result(timeout=self.config.timeout_seconds)
            self.ledger.annotate(
                entry,
                {
                    "stage": "completed",
                    "optimizer_metrics": optimizer.metrics,
                    "optimizer_step": step,
                },
            )
            self.ledger.complete(entry)
        except BaseException as error:
            self.ledger.annotate(entry, {"error_type": type(error).__name__})
            if getattr(error, "status_code", 0) in (400, 401, 403, 404, 409, 422, 429):
                self.ledger.complete(entry, failed=True)
            raise
        supervised = int(sum(sum(d.loss_fn_inputs["weights"].tolist()) for d in batch))
        loss = float(forward.metrics["loss_sum"]) / supervised
        if not math.isfinite(loss):
            raise ValueError("Remote training loss is not finite")
        return {
            "optimizer_step": step,
            "loss": loss,
            "samples": len(batch),
            "supervised_tokens": supervised,
            "input_tokens": sum(lengths),
            "learning_rate": self.config.learning_rate,
            "seconds": time.monotonic() - start,
            "ledger_request": entry,
            "remote_optimizer_metrics": optimizer.metrics,
        }

    def save(self, step: int) -> dict[str, Any]:
        sampler = self.client.save_weights_for_sampler(f"sampler-{step:04d}").result(
            timeout=self.config.timeout_seconds
        )
        state = self.client.save_state(f"state-{step:04d}").result(
            timeout=self.config.timeout_seconds
        )
        if (
            sampler.model != self.config.base_model
            or state.model != self.config.base_model
            or sampler.path == state.path
        ):
            raise ValueError("Checkpoint identities or model do not match")
        return {
            "optimizer_step": step,
            "sampler": {"kind": "sampler_weights", "path": sampler.path},
            "training_state": {"kind": "training_state", "path": state.path},
            "training_run_cuid": self.client.training_run_cuid,
        }


def training_cost(ledger: SpendLedger, run_id: str) -> dict[str, Any]:
    snapshot = ledger.snapshot()
    entries = [
        e
        for e in snapshot["entries"]
        if e["evidence"]["payload"].get("run_id") == run_id
    ]
    return summarize_cost(snapshot, entries)


def _data_progress(
    datums: list[trio.Datum], updates: list[dict[str, Any]]
) -> dict[str, int | float]:
    samples = sum(update["samples"] for update in updates)
    return {
        "dataset_samples": len(datums),
        "dataset_supervised_tokens": int(
            sum(sum(d.loss_fn_inputs["weights"].tolist()) for d in datums)
        ),
        "samples_processed": samples,
        "epochs_processed": samples / len(datums),
        "input_tokens_processed": sum(update["input_tokens"] for update in updates),
        "supervised_tokens_processed": sum(
            update["supervised_tokens"] for update in updates
        ),
    }


def _train_sft(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> Path:
    if stop_after is not None and (type(stop_after) is not int or stop_after < 1):
        raise ValueError("stop_after must be a positive optimizer step")
    config = SFTConfig.read(config_path)
    manifest, datums = load_demonstrations(Path(config.dataset))
    batches_per_epoch = math.ceil(len(datums) / config.batch_size)
    if config.max_epochs is not None and (
        config.steps > config.max_epochs * batches_per_epoch
    ):
        raise ValueError("Planned SFT training exceeds max_epochs")
    details = manifest["encoding_details"]
    if details.get("revision") != config.model_revision:
        raise ValueError("Training model and demonstration tokenizer identity differ")
    if details.get("tokenizer") != config.base_model:
        if manifest["source"] != EXECUTED_SOURCE:
            raise ValueError(
                "Training model and demonstration tokenizer identity differ"
            )
        canonical = QwenChatMLEncoding(
            config.base_model,
            revision=config.model_revision,
            thinking=details["thinking"],
            cache_dir=manifest.get("tokenizer_cache_dir"),
        )
        if encoding_condition(canonical.describe()) != encoding_condition(details):
            raise ValueError("Training model and demonstration encoding assets differ")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = output_dir / "training.json"
    settings = asdict(config)
    settings.pop("api_key_env")
    settings.pop("dotenv_path")
    if config.max_epochs is None:
        settings.pop("max_epochs")
    # Tokenizer identity comes from executed input evidence, never re-encoding.
    metadata = {
        "base_model": config.base_model,
        "server_model_revision": None,
        "encoding": manifest["encoding"],
        "data_sha256": manifest["examples_sha256"],
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "data_source": manifest["source"],
        "fingerprints": manifest["fingerprints"],
        "training": {
            k: settings[k]
            for k in (
                "rank",
                "seed",
                "train_mlp",
                "train_attn",
                "train_unembed",
                "steps",
                "batch_size",
                "learning_rate",
                "checkpoint_every",
            )
        },
        "loss_aggregation": "sum masked cross-entropy / supervised token count",
    }
    if config.max_epochs is not None:
        metadata["training"]["max_epochs"] = config.max_epochs
    if resume:
        record = json.loads(path.read_text())
        if record["config"] != settings or record["metadata"] != metadata:
            raise ValueError("Resume must preserve training configuration and data")
        if record["pending_update"] is not None:
            raise ValueError("Uncertain remote update must be reconciled before resume")
        restore = next(
            (
                c["training_state"]
                for c in record["checkpoints"]
                if c["optimizer_step"] == record["optimizer_step"]
            ),
            None,
        )
        if restore is None and record["optimizer_step"] != 0:
            raise ValueError(
                "Resume requires full training state at the recorded optimizer step"
            )
    else:
        if path.exists():
            raise ValueError("Training run already exists; use --resume")
        record = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "config": settings,
            "metadata": metadata,
            "optimizer_step": 0,
            "checkpoints": [],
            "updates": [],
            "pending_update": None,
            "code_revision": code_revision(),
            "status": "starting",
        }
        restore = None
    final_step = min(config.steps, stop_after or config.steps)
    if final_step <= record["optimizer_step"]:
        raise ValueError("Requested stopping step must be after recorded progress")
    ledger = SpendLedger(
        Path(config.ledger_path), policy=SpendPolicy.read(Path(config.spend_config))
    )
    monitor = RunMonitor(
        output_dir,
        MonitorConfig(**config.monitor),
        phase="sft",
        metadata=metadata | {"code_revision": record["code_revision"]},
        require_online=True,
    )
    record["data_progress"] = _data_progress(datums, record["updates"])
    write_document(path, record)
    try:
        backend = PyTRIOTraining(config, ledger, record["run_id"], restore)
        record["live_prices"] = backend.catalog
        if restore:
            record.setdefault("restores", []).append(
                {
                    "optimizer_step": record["optimizer_step"],
                    "training_state": restore,
                    "optimizer_restored": True,
                    "training_run_cuid": backend.client.training_run_cuid,
                }
            )
        record["status"] = "training"
        for step in range(record["optimizer_step"] + 1, final_step + 1):
            epoch, batch_index = divmod(step - 1, batches_per_epoch)
            indices = list(range(len(datums)))
            random.Random(config.seed + epoch).shuffle(indices)
            batch = [
                datums[i]
                for i in indices[
                    batch_index * config.batch_size : (batch_index + 1)
                    * config.batch_size
                ]
            ]
            record["pending_update"] = step
            write_document(path, record)
            update = backend.update(batch, step)
            record["updates"].append(update)
            record["optimizer_step"], record["pending_update"] = step, None
            record["data_progress"] = _data_progress(datums, record["updates"])
            record["cost"] = training_cost(ledger, record["run_id"])
            write_document(path, record)
            if step % monitor.config.log_every == 0 or step == final_step:
                metrics = {
                    f"train/{k}": update[k]
                    for k in (
                        "optimizer_step",
                        "loss",
                        "samples",
                        "supervised_tokens",
                        "input_tokens",
                        "learning_rate",
                        "seconds",
                    )
                }
                metrics.update(spend_metrics(record["cost"]))
                metrics.update(
                    {f"train/{k}": v for k, v in record["data_progress"].items()}
                )
                monitor.log(f"update-{step}", metrics)
            if step % config.checkpoint_every == 0 or step == final_step:
                checkpoint = backend.save(step)
                checkpoint["data_progress"] = dict(record["data_progress"])
                record["checkpoints"].append(checkpoint)
                write_document(path, record)
                monitor.log(
                    f"checkpoint-{step}",
                    {
                        "train/optimizer_step": step,
                        "checkpoint/sampler": checkpoint["sampler"]["path"],
                        "checkpoint/training_state": checkpoint["training_state"][
                            "path"
                        ],
                    },
                )
            print(
                f"SFT step {step}/{config.steps}: loss={update['loss']:.6f}", flush=True
            )
        record["status"] = "complete" if final_step == config.steps else "checkpointed"
    except BaseException as error:
        record["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        )
        record["error_type"] = type(error).__name__
        raise
    finally:
        record["cost"] = training_cost(ledger, record["run_id"])
        write_document(path, record)
        monitor.log(
            f"status-{len(record['updates'])}-{record['status']}",
            {
                "run/status": record["status"],
                "train/optimizer_step": record["optimizer_step"],
                **spend_metrics(record["cost"]),
            },
        )
        monitor.finish()
    return path


def train_sft(
    config_path: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (output_dir / "training.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Training run is already active") from None
        return _train_sft(config_path, output_dir, resume=resume, stop_after=stop_after)
