"""Frozen initial weights, raw-token scoring and completed PyTRIO OPSD updates."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from bashlet.backends.encoding import QwenChatMLEncoding
from bashlet.backends.pytrio import (
    MODEL,
    SDK_VERSION,
    PyTRIOModel,
    _credential,
    _RequestIdentity,
    save_training_checkpoints,
    validate_training_capacity,
)
from bashlet.distillation import (
    InitialSnapshot,
    OPSDRun,
    TeacherScores,
    UpdateRejected,
    UpdateResult,
)
from bashlet.harness.models import Encoding, Model, ModelResponseError
from bashlet.monitoring import write_document
from bashlet.spend import DEFAULT_POLICY, SpendLedger, TokenPricing

if TYPE_CHECKING:
    import pytrio as trio


@dataclass(frozen=True)
class PyTRIOOPSDConfig:
    base_model: str = MODEL
    model_revision: str = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    tokenizer_cache_dir: str = "runs/tokenizers"
    rank: int = 16
    seed: int = 4
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    thinking: bool = True
    source: Literal["base"] = "base"
    timeout_seconds: int = 300
    api_key_env: str = "pytrio_api_key"
    dotenv_path: str = ".env"
    group: str = "opsd-pilot"
    spend_config: str = str(DEFAULT_POLICY)

    def __post_init__(self) -> None:
        if type(self.thinking) is not bool:
            raise ValueError("OPSD thinking must be a boolean")
        if self.base_model != MODEL or not self.model_revision:
            raise ValueError("OPSD requires the selected Qwen3.5-4B model")
        if self.source != "base":
            raise ValueError("OPSD starts from the original base model")
        for value in (self.rank, self.timeout_seconds):
            if type(value) is not int or value < 1:
                raise ValueError("Rank and timeout must be positive integers")


class _Future[T]:
    """Defer remote work until the caller's durable reservation exists."""

    def __init__(self, complete: Callable[[float], T], timeout: float) -> None:
        self.complete, self.timeout, self.used = complete, timeout, False

    def result(self, timeout: float | None = None) -> T:
        if self.used:
            raise ValueError("An OPSD request future cannot submit twice")
        self.used = True
        return self.complete(time.monotonic() + (timeout or self.timeout))


def _wait(future: Any, deadline: float) -> Any:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("OPSD operation deadline exhausted")
    return future.result(timeout=remaining)


class PyTRIOOPSD:
    """One trainer, a frozen sampler, and the existing guarded student backend.

    Step 0 is saved before either role samples, so both load the same immutable
    URI. Initialization has fresh LoRA/optimizer state; resume alone loads the full
    optimizer state.
    """

    def __init__(
        self,
        config: PyTRIOOPSDConfig,
        *,
        ledger: SpendLedger,
        output_dir: Path,
        service_client: Any = None,
        encoding: Encoding | None = None,
    ) -> None:
        import json

        import pytrio as trio

        from bashlet.spend import read_spend_config

        if version("pytrio") != SDK_VERSION:
            raise ValueError(f"Expected PyTRIO {SDK_VERSION}")
        self.config, self.ledger = config, ledger
        self.path = output_dir.resolve() / "backend.json"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run_id: str | None = None
        self.pricing = TokenPricing(read_spend_config(Path(config.spend_config)))
        if service_client is None:
            catalog = self.pricing.verify(config.base_model, config.timeout_seconds)
            trio.configure(
                home=str(ledger.path.parent / "pytrio-sdk"),
                retry_max_attempts=1,
                control_retry_timeout_seconds=0,
                training_retry_timeout_seconds=0,
                sampling_retry_timeout_seconds=0,
                log_level="WARNING",
                request_timeout_seconds=config.timeout_seconds,
                sampling_timeout_seconds=config.timeout_seconds,
                sampling_first_chunk_timeout_seconds=config.timeout_seconds,
                sampling_chunk_idle_timeout_seconds=config.timeout_seconds,
                actor_capacity_wait_timeout_seconds=config.timeout_seconds,
            )
            service_client = trio.ServiceClient(
                api_key=_credential(config.api_key_env, Path(config.dotenv_path))
            )
        else:
            catalog = None
        self.service = service_client
        ledger.bind_account(self.service.workspace_cuid)
        supported = next(
            (
                item
                for item in self.service.get_server_capabilities().supported_models
                if item.model_name == config.base_model
            ),
            None,
        )
        if (
            supported is None
            or not supported.training.available
            or not supported.sampling.available
        ):
            raise ValueError("Selected model must support sampling and training")
        self.limits = supported.training
        self.teacher_encoding = encoding or QwenChatMLEncoding(
            config.base_model,
            revision=config.model_revision,
            thinking=config.thinking,
            cache_dir=config.tokenizer_cache_dir,
        )
        self.client: Any = None
        if self.path.exists():
            self.record = json.loads(self.path.read_text())
            if (
                self.record["config"] != asdict(config)
                or self.record["encoding"] != self.teacher_encoding.describe()
                or self.record["workspace_cuid"] != self.service.workspace_cuid
            ):
                raise ValueError("OPSD backend resume must preserve loading identities")
        else:
            self.client = self.service.create_lora_training_client(
                base_model=config.base_model,
                rank=config.rank,
                seed=config.seed,
                train_attn=config.train_attn,
                train_mlp=config.train_mlp,
                train_unembed=config.train_unembed,
            )
            self._validate_trainer()
            sampler, state = self._save(0)
            self.record = {
                "config": asdict(config),
                "encoding": self.teacher_encoding.describe(),
                "source": config.source,
                "initial_sampler": sampler,
                "initial_state": state,
                "server_model_revision": None,
                "reference_model_revision": config.model_revision,
                "revision_limitation": "Control exposes an alias, not a weight commit",
                "optimizer_initialization": "fresh",
                "sdk_version": SDK_VERSION,
                "workspace_cuid": self.service.workspace_cuid,
                "initial_training_run_cuid": self.client.training_run_cuid,
                "price_catalog": catalog,
                "updates": [],
                "loads": [],
            }
        initial_path = self.record["initial_sampler"]
        self._student = self._model(initial_path)
        student_proof = self._student.verify_loaded_sampler()
        # Independent sampler for scoring; it never receives an optimizer.
        self.teacher = self.service.create_sampling_client(
            base_model=config.base_model, model_path=initial_path
        )
        teacher_proof = _wait(
            self.service.create_rest_client().get_sampler(
                self.teacher.sampling_run_cuid
            ),
            time.monotonic() + config.timeout_seconds,
        )
        if (
            self.teacher.model_id != config.base_model
            or teacher_proof.base_model != config.base_model
            or teacher_proof.model_path != initial_path
        ):
            raise ModelResponseError("Frozen teacher did not load the selected weights")
        self.snapshot = InitialSnapshot(
            self._student.identity,
            self.teacher_encoding.identity,
            "experimental",
            stage=config.source,
            kind="sampler_weights",
        )
        self.teacher_identity = self.snapshot.identity
        self.record["loads"].append(
            {
                "student": student_proof,
                "teacher": {
                    "base_model": teacher_proof.base_model,
                    "model_path": teacher_proof.model_path,
                    "sampling_run_cuid": self.teacher.sampling_run_cuid,
                },
                "service_session_cuid": self.service.service_session_cuid,
            }
        )
        write_document(self.path, self.record)

    def _validate_trainer(self) -> None:
        if self.client.model_id != self.config.base_model:
            raise ValueError("Training model differs from the selected initial model")

    def _model(self, path: str, *, group: str | None = None) -> PyTRIOModel:
        return PyTRIOModel(
            model=self.config.base_model,
            model_revision=self.config.model_revision,
            model_path=path,
            encoding=self.teacher_encoding,
            service_client=self.service,
            ledger=self.ledger,
            spend_config=self.config.spend_config,
            group=group if group is not None else self.config.group,
            timeout_seconds=self.config.timeout_seconds,
            max_retries=0,
            run_id=self.run_id,
        )

    @property
    def student(self) -> Model:
        return self._student

    def bind_run(self, run_id: str) -> None:
        self.run_id = self._student.run_id = run_id

    def _save(self, step: int, deadline: float | None = None) -> tuple[str, str]:
        deadline = deadline or time.monotonic() + self.config.timeout_seconds
        return save_training_checkpoints(
            self.client, self.config.base_model, step, deadline
        )

    def score(self, tokens: Sequence[int]) -> _Future[TeacherScores]:
        original = tuple(tokens)

        def complete(deadline: float) -> TeacherScores:
            import pytrio as trio

            capture = _RequestIdentity(self.teacher.sampling_run_cuid)
            logger = logging.getLogger("pytrio.transport.actor.sampling_run.sample")
            logger.setLevel(logging.DEBUG)
            logger.addHandler(capture)
            try:
                values = _wait(
                    self.teacher.compute_logprobs(
                        trio.ModelInput.from_ints(list(original))
                    ),
                    deadline,
                )
                if not capture.sent or capture.sent != capture.received:
                    raise ModelResponseError(
                        "Teacher request identity was not preserved"
                    )
                return TeacherScores(
                    original,
                    tuple(values),
                    self.teacher_identity,
                    self.teacher_encoding.identity,
                    capture.received,
                )
            finally:
                logger.removeHandler(capture)

        return _Future(complete, self.config.timeout_seconds)

    def update(
        self,
        batch: list[trio.Datum],
        *,
        loss_fn: str,
        auto_shift: bool,
        optimizer_step: int,
        learning_rate: float,
    ) -> _Future[UpdateResult]:
        def complete(deadline: float) -> UpdateResult:
            import pytrio as trio

            if not batch or loss_fn != "importance_sampling" or auto_shift:
                raise UpdateRejected("invalid_batch")
            lengths = [len(d.model_input.to_ints()) for d in batch]
            try:
                validate_training_capacity(lengths, self.limits)
            except ValueError:
                raise UpdateRejected("capacity") from None
            supervised = 0
            for datum, length in zip(batch, lengths, strict=True):
                inputs = {
                    k: np.asarray(v.tolist()) for k, v in datum.loss_fn_inputs.items()
                }
                if set(inputs) != {"target_tokens", "logprobs", "advantages"} or any(
                    values.shape != (length,) or not np.isfinite(values).all()
                    for values in inputs.values()
                ):
                    raise UpdateRejected("invalid_batch")
                supervised += int(
                    np.count_nonzero(
                        (inputs["target_tokens"] != 0)
                        | (inputs["logprobs"] != 0)
                        | (inputs["advantages"] != 0)
                    )
                )
            if not supervised or not any(
                np.any(d.loss_fn_inputs["advantages"].tolist()) for d in batch
            ):
                raise UpdateRejected("invalid_batch")
            if self.client is None:
                self.client = self.service.create_training_client_from_state(
                    self.record["initial_state"]
                )
                self._validate_trainer()
            receipt: dict[str, Any] = {
                "optimizer_step": optimizer_step,
                "status": "forward_backward",
                "training_run_cuid": self.client.training_run_cuid,
            }
            self.record["updates"].append(receipt)
            write_document(self.path, self.record)
            forward = _wait(
                self.client.forward_backward(
                    batch,
                    loss_fn=loss_fn,
                    auto_shift=False,
                ),
                deadline,
            )
            loss = float(forward.metrics["loss_sum"]) / supervised
            if not math.isfinite(loss):
                raise ValueError("Non-finite forward loss; optimizer was not submitted")
            receipt.update(status="optim_step", forward_metrics=forward.metrics)
            write_document(self.path, self.record)
            optimizer = _wait(
                self.client.optim_step(trio.AdamParams(learning_rate=learning_rate)),
                deadline,
            )
            receipt.update(
                status="optimizer_completed", optimizer_metrics=optimizer.metrics
            )
            write_document(self.path, self.record)
            sampler, state = self._save(optimizer_step, deadline)
            receipt.update(status="completed", sampler=sampler, training_state=state)
            write_document(self.path, self.record)
            return UpdateResult(
                optimizer_step,
                loss,
                f"pytrio/{self.config.base_model}@{sampler}",
                state,
            )

        return _Future(complete, self.config.timeout_seconds)

    def refresh_student(self, update: UpdateResult) -> Model:
        prefix = f"pytrio/{self.config.base_model}@"
        if not update.student_policy.startswith(prefix):
            raise ValueError("Updated sampler belongs to another model")
        self._student = self._model(update.student_policy.removeprefix(prefix))
        self._student.verify_loaded_sampler()
        return self._student

    def restore_student(self, update: UpdateResult) -> Model:
        self.client = self.service.create_training_client_from_state_with_optimizer(
            update.training_state
        )
        self._validate_trainer()
        # Client creation registers a lazy load. Saving forces initialization
        # and waits for the remote weights + optimizer restore to finish.
        restored = self.client.save_state(
            f"restored-state-{update.optimizer_step:04d}"
        ).result(timeout=self.config.timeout_seconds)
        if restored.model != self.config.base_model or not restored.path:
            raise ValueError("Restored training state belongs to another model")
        self.record["optimizer_restoration"] = {
            "training_state": update.training_state,
            "optimizer_step": update.optimizer_step,
            "training_run_cuid": self.client.training_run_cuid,
            "completed_state": restored.path,
        }
        write_document(self.path, self.record)
        return self.refresh_student(update)

    def verify_progress(self, run: OPSDRun) -> dict[str, Any]:
        """Verify Base parameters, sampler identities and completed restoration."""
        import json

        if self.config.source != "base":
            raise ValueError(
                "Checkpoint parameters: only Base initialization is supported"
            )
        run._roles()
        restoration = self.record.get("optimizer_restoration")
        update = run.record.get("last_update")
        if (
            run.record["pending_update"] is not None
            or run.record["optimizer_step"] < 1
            or not isinstance(update, dict)
            or not isinstance(restoration, dict)
            or update.get("optimizer_step") != run.record["optimizer_step"]
            or update.get("student_policy") != self._student.identity
            or not update.get("training_state")
            or restoration.get("optimizer_step") != run.record["optimizer_step"]
            or restoration.get("training_state") != update["training_state"]
            or not restoration.get("completed_state")
        ):
            raise ValueError(
                "Checkpoint parameters: a completed update and optimizer "
                "restoration are required"
            )
        first = next(b for b in run.record["batches"] if "update" in b)
        episode = Path(first["episode"])
        original = json.loads((episode / "private/teacher-0000.json").read_text())
        if original["policy"] != self.teacher_identity or (
            original["encoding"] != self.teacher_encoding.identity
        ):
            raise ValueError(
                "Checkpoint parameters: original teacher scoring identity changed"
            )
        parameter_evidence = self._verify_original_parameters(run)
        return {
            "status": "complete",
            "verification_basis": "restored_optimizer_and_adapter_parameters",
            "frozen_teacher_unchanged": True,
            "student_parameters_changed": True,
            "parameter_evidence": parameter_evidence,
            "training_state": restoration,
            "student_policy": self._student.identity,
            "teacher_policy": self.teacher_identity,
            "limitation": (
                "Downloaded LoRA parameters and Control loading identity; "
                "remote base weight hash unknown"
            ),
        }

    def _verify_original_parameters(self, run: OPSDRun) -> dict[str, Any]:
        import json

        from bashlet.snapshots.artifacts import (
            export_snapshot,
            verify_original_adapter_update,
        )

        teacher = _wait(
            self.service.create_rest_client().get_sampler(
                self.teacher.sampling_run_cuid
            ),
            time.monotonic() + self.config.timeout_seconds,
        )
        if teacher.base_model != self.config.base_model or (
            teacher.model_path != self.record["initial_sampler"]
        ):
            raise ValueError(
                "Checkpoint parameters: frozen teacher loading identity changed"
            )
        student_load = self._student.verify_loaded_sampler()
        exports = []
        for step, sampler in (
            (0, self.record["initial_sampler"]),
            (run.record["optimizer_step"], self._student.model_path),
        ):
            directory = run.root / "checkpoint-weights" / f"step-{step}"
            path = directory / "export.json"
            if not path.exists():
                path = export_snapshot(
                    run.root,
                    step,
                    directory,
                    api_key_env=self.config.api_key_env,
                    dotenv_path=Path(self.config.dotenv_path),
                )
            exported = json.loads(path.read_text())
            if exported["optimizer_step"] != step or (
                exported["checkpoint"]["path"] != sampler
                or exported["base_model"] != self.config.base_model
                or exported["training_encoding"] != self.teacher_encoding.identity
            ):
                raise ValueError(
                    "Checkpoint parameters: cached export identity mismatch"
                )
            exports.append(path)
        evidence = verify_original_adapter_update(*exports)
        evidence["teacher_load"] = {
            "base_model": teacher.base_model,
            "model_path": teacher.model_path,
            "sampling_run_cuid": self.teacher.sampling_run_cuid,
        }
        evidence["student_load"] = student_load
        return evidence
