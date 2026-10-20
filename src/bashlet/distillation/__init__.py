"""Auditable OPSD orchestration; model calls are an external Adapter boundary."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, TextIO

from bashlet.config import ExperimentConfig, read_yaml
from bashlet.distillation.supervision import validate_history
from bashlet.harness.models import (
    ContextExhausted,
    Encoding,
    ExperimentResult,
    Generation,
    JSONValue,
    Model,
    ModelBackend,
    ModelResponseError,
    SamplingParams,
    Task,
)
from bashlet.harness.protocol import fingerprints
from bashlet.monitoring import (
    MonitorConfig,
    RunMonitor,
    code_revision,
    spend_metrics,
    write_document,
)
from bashlet.spend import (
    DEFAULT_POLICY,
    BudgetRefused,
    RemoteFuture,
    SpendLedger,
    TokenPricing,
    read_spend_config,
    summarize_cost,
)
from bashlet.tasks import validate_training_task

if TYPE_CHECKING:
    import pytrio as trio


@dataclass(frozen=True)
class InitialSnapshot:
    """The selected original model weights, never an optimizer state."""

    identity: str
    encoding: str
    evidence_kind: Literal["test-double", "experimental"]
    stage: str = "base"
    kind: str = "sampler_weights"

    def __post_init__(self) -> None:
        if not self.identity or not self.encoding:
            raise ValueError("Initial snapshot and encoding identities are required")
        if (self.stage, self.kind) not in (
            ("base", "base_model"),
            ("base", "sampler_weights"),
        ):
            raise ValueError(
                "OPSD requires an original model weights, not training state"
            )
        if self.evidence_kind not in ("test-double", "experimental"):
            raise ValueError("Unknown initial snapshot evidence kind")


@dataclass(frozen=True)
class OPSDConfig:
    distillation_coefficient: float = 1.0
    learning_rate: float = 0.000005
    timeout_seconds: float = 180
    max_context_tokens: int = 32768
    group: str = "opsd-pilot"
    spend_config: str = str(DEFAULT_POLICY)

    @classmethod
    def read(cls, path: Path) -> OPSDConfig:
        return cls(**read_yaml(path))

    def __post_init__(self) -> None:
        for name in ("distillation_coefficient", "learning_rate", "timeout_seconds"):
            value = getattr(self, name)
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if type(self.max_context_tokens) is not int or self.max_context_tokens < 2:
            raise ValueError("max_context_tokens must be an integer of at least two")


@dataclass(frozen=True)
class TeacherScores:
    """Unshifted probabilities at the exact requested token positions."""

    tokens: tuple[int, ...]
    logprobs: tuple[float | None, ...]
    policy: str
    encoding: str
    request_id: str


@dataclass(frozen=True)
class UpdateResult:
    """Returned only after optimizer completion and distinct checkpoint saves."""

    optimizer_step: int
    loss: float
    student_policy: str
    training_state: str


class UpdateRejected(Exception):
    """A confirmed refusal: no parameter update or retained gradients remain.

    Only this explicit Adapter guarantee permits another batch after a failed
    update. Transport failures and interruption retain an uncertain update.
    """

    def __init__(
        self, reason: Literal["transient", "capacity", "invalid_batch"]
    ) -> None:
        if reason not in ("transient", "capacity", "invalid_batch"):
            raise ValueError("Unknown remote update rejection category")
        self.reason = reason
        super().__init__(reason)


class _RemoteFailure(Exception):
    """External exception contents are private; telemetry records only its type."""

    def __init__(self, error: Exception) -> None:
        self.error_type = type(error).__name__
        super().__init__(self.error_type)


class OPSDAdapter(Protocol):
    """External sampling, frozen scoring, updating and training-state restoration.

    ``update`` must wait for both forward/backward and optimizer completion.
    Its loss is the summed importance-sampling loss / supervised token count.
    ``restore_student`` must load the named full state including its optimizer.
    """

    @property
    def student(self) -> Model: ...

    @property
    def teacher_identity(self) -> str: ...

    @property
    def teacher_encoding(self) -> Encoding: ...

    def score(self, tokens: Sequence[int]) -> RemoteFuture[TeacherScores]: ...

    def update(
        self,
        batch: list[trio.Datum],
        *,
        loss_fn: str,
        auto_shift: bool,
        optimizer_step: int,
        learning_rate: float,
    ) -> RemoteFuture[UpdateResult]: ...

    def refresh_student(self, update: UpdateResult) -> Model: ...

    def restore_student(self, update: UpdateResult) -> Model: ...


class _SampleFuture:
    def __init__(self, generate: Callable[[], Generation]) -> None:
        self.generate = generate

    def result(self, timeout: float | None = None) -> Generation:
        return self.generate()


class _Student:
    def __init__(self, run: OPSDRun, model: Model) -> None:
        self.run, self.model = run, model
        self.identity, self.encoding = model.identity, model.encoding

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        if self.model.identity != self.identity:
            raise ValueError("Student policy changed inside a rollout batch")
        from bashlet.backends.pytrio import PyTRIOModel

        if isinstance(self.model, PyTRIOModel):
            self.model.validate_spend_owner(
                self.run.ledger, self.run.config.group, self.run.record["run_id"]
            )
            generation = self.model.sample(prompt, sampling)
            self.run.calls.append((tuple(prompt), sampling, generation))
            return generation
        generation = self.run._call(
            "sampling",
            {
                "policy": self.identity,
                "prompt_tokens": list(prompt),
                "sampling": asdict(sampling),
            },
            self.run.pricing.upper_bound(len(prompt), sampling.max_tokens),
            lambda: _SampleFuture(lambda: self.model.sample(prompt, sampling)),
        )
        self.run.calls.append((tuple(prompt), sampling, generation))
        return generation

    def probe(self) -> None:
        if isinstance(self.model, ModelBackend):
            self.model.probe()

    @property
    def max_context_tokens(self) -> int:
        return (
            self.model.max_context_tokens
            if isinstance(self.model, ModelBackend)
            else self.run.config.max_context_tokens
        )

    def describe(self) -> dict[str, JSONValue]:
        return self.model.describe() if isinstance(self.model, ModelBackend) else {}


class OPSDRun:
    """One episode per rollout batch, sharing policy progress, spend and monitoring.

    Use ``run_experiment(model=run.student, opsd_run=run, ...)``. Each call keeps
    its real episode and independent grade. Only sampled-token probabilities
    produce advantages; the grade never enters the update.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        snapshot: InitialSnapshot,
        adapter: OPSDAdapter,
        ledger: SpendLedger,
        monitoring: MonitorConfig,
        config: OPSDConfig | None = None,
        resume: bool = False,
    ) -> None:
        config = config or OPSDConfig()
        self.root = output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "opsd.json"
        self._lockfile: TextIO | None = None
        self.snapshot, self.adapter, self.ledger = snapshot, adapter, ledger
        self.config, self.monitoring = config, monitoring
        self.pricing = TokenPricing(read_spend_config(Path(config.spend_config)))
        self.calls: list[tuple[tuple[int, ...], SamplingParams, Generation]] = []
        self.monitor: RunMonitor | None = None
        self.record: dict[str, Any] = {
            "schema_version": 1,
            "run_id": uuid.uuid4().hex,
            "snapshot": asdict(snapshot),
            "config": asdict(config),
            "code_revision": code_revision(),
            "optimizer_step": 0,
            "student_policy": snapshot.identity,
            "teacher_policy": snapshot.identity,
            "pending_update": None,
            "batches": [],
            "status": "ready",
            "ledger": str(ledger.path),
            "last_update": None,
            "encoding_details": self.student.encoding.describe(),
        }
        self._acquire_lock()
        try:
            if self.path.exists() and not resume:
                raise ValueError("OPSD run already exists")
            if resume:
                previous = json.loads(self.path.read_text())
                if any(
                    previous.get(key) != self.record[key]
                    for key in ("schema_version", "snapshot", "config", "ledger")
                ):
                    raise ValueError(
                        "OPSD resume must preserve configuration, snapshot and ledger"
                    )
                if previous["pending_update"] is not None:
                    raise ValueError(
                        "An uncertain remote update must be reconciled before resume"
                    )
                self.record = previous
                if adapter.teacher_identity != snapshot.identity:
                    raise ValueError("Resume must preserve the frozen initial teacher")
                if previous["optimizer_step"]:
                    update = UpdateResult(**previous["last_update"])
                    if (
                        update.optimizer_step != previous["optimizer_step"]
                        or update.student_policy != previous["student_policy"]
                    ):
                        raise ValueError(
                            "Training state differs from recorded optimizer progress"
                        )
                    model = adapter.restore_student(update)
                    if model.identity != update.student_policy:
                        raise ValueError(
                            "Restored sampler differs from the full training state"
                        )
            self._roles()
            write_document(self.path, self.record)
        finally:
            self._release_lock()

    @property
    def student(self) -> Model:
        return self.adapter.student

    def _roles(self) -> None:
        if self.adapter.teacher_identity != self.snapshot.identity:
            raise ValueError(
                "Frozen teacher differs from the selected initial snapshot"
            )
        if self.student.identity != self.record["student_policy"]:
            raise ValueError("Student differs from the recorded policy")
        if (
            self.student.encoding.identity != self.snapshot.encoding
            or self.adapter.teacher_encoding.identity != self.snapshot.encoding
            or self.student.encoding.describe()
            != self.adapter.teacher_encoding.describe()
            or self.student.encoding.describe() != self.record["encoding_details"]
        ):
            raise ValueError("Student and frozen teacher encodings are incompatible")

    def validate_request(self, task: Task, model: Model) -> None:
        if self._lockfile is not None:
            raise ValueError("OPSD run is already active")
        self._roles()
        if model.identity != self.record["student_policy"]:
            raise ValueError("Rollout must use the current student policy")
        if model.encoding.describe() != self.record["encoding_details"]:
            raise ValueError("Rollout encoding differs from the initial snapshot")
        validate_training_task(task)
        if not task.reference_implementation.strip():
            raise ValueError("OPSD teacher requires a reference implementation")

    def begin(self, task: Task, model: Model, condition: ExperimentConfig) -> Model:
        self.validate_request(task, model)
        assert task.origin is not None
        if self.record["pending_update"] is not None:
            raise ValueError("An uncertain remote update must be reconciled first")
        self._acquire_lock()
        metadata = {
            "snapshot": asdict(self.snapshot),
            "encoding": self.snapshot.encoding,
            "evidence_kind": self.snapshot.evidence_kind,
            "code_revision": self.record["code_revision"],
            "fingerprints": fingerprints(condition),
            "sampling": asdict(condition.model.sampling),
            "training": {
                "distillation_coefficient": self.config.distillation_coefficient,
                "learning_rate": self.config.learning_rate,
            },
            "loss_aggregation": "sum importance-sampling loss / supervised token count",
            "data_source": {
                key: asdict(task.origin)[key]
                for key in ("dataset", "revision", "source_sha256", "split")
            },
        }
        if task.assignment is not None:
            metadata["data_source"].update(
                data_version=task.assignment.version,
                cohort_sha256=task.assignment.cohort_sha256,
                role=task.assignment.role,
            )
        try:
            latest = json.loads(self.path.read_text())
            if latest != self.record:
                raise ValueError("OPSD progress changed; reopen the run with resume")
            source = metadata["data_source"]
            monitor_path = self.root / "monitor.json"
            if monitor_path.exists():
                # A run can visit several validated training sources. Keep all
                # other metadata subject to the normal immutable-config check.
                metadata["data_source"] = json.loads(monitor_path.read_text())[
                    "config"
                ]["data_source"]
            self.monitor = RunMonitor(
                self.root, self.monitoring, phase=self.config.group, metadata=metadata
            )
            known = metadata["data_source"]
            sources = known.get("sources", [known])
            if source not in sources:
                self.monitor.update_metadata(
                    {"data_source": {"sources": [*sources, source]}}
                )
        except BaseException:
            self._release_lock()
            raise
        self.calls = []
        return _Student(self, model)

    def _acquire_lock(self) -> None:
        lock = (self.root / "opsd.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise ValueError("OPSD run is already active") from None
        self._lockfile = lock

    def _release_lock(self) -> None:
        if self._lockfile is not None:
            self._lockfile.close()
            self._lockfile = None

    def _call[T](
        self,
        operation: str,
        payload: dict[str, Any],
        upper: object,
        submit: Callable[[], RemoteFuture[T]],
    ) -> T:
        started = time.monotonic()
        submitted: str | None = None

        def guarded_submit(entry: str) -> RemoteFuture[T]:
            nonlocal submitted
            submitted = entry
            input_tokens = (
                len(payload["prompt_tokens"])
                if operation == "sampling"
                else len(payload["tokens"])
                if operation == "teacher_scoring"
                else payload["input_tokens"]
            )
            self.ledger.annotate(
                entry,
                {
                    "run_id": self.record["run_id"],
                    "submitted_input_tokens": input_tokens,
                },
            )
            if operation == "opsd_update":
                self.record["pending_update"] = {
                    "optimizer_step": payload["optimizer_step"],
                    "ledger_request": entry,
                }
                write_document(self.path, self.record)
            return submit()

        try:
            result, entry = self.ledger.execute(
                operation=operation,
                group=self.config.group,
                payload=payload | {"run_id": self.record["run_id"]},
                upper_cny=upper,
                submit=guarded_submit,
                timeout=self.config.timeout_seconds,
                terminal_failure=lambda error: isinstance(
                    error, (UpdateRejected, ModelResponseError, ContextExhausted)
                ),
            )
        except ModelResponseError as error:
            if submitted is not None and error.generation is not None:
                self.ledger.annotate(
                    submitted,
                    {
                        "usage": {"output_tokens": len(error.generation.tokens)},
                    },
                )
            raise ModelResponseError(
                "Sampling response rejected", error.generation
            ) from None
        except ContextExhausted:
            raise ContextExhausted("Sampling context exhausted") from None
        except (UpdateRejected, BudgetRefused):
            raise
        except Exception as error:
            raise _RemoteFailure(error) from None
        finally:
            if submitted is not None:
                self.ledger.annotate(submitted, {"seconds": time.monotonic() - started})
        usage: dict[str, int] = {}
        if isinstance(result, Generation):
            usage = {
                "input_tokens": len(payload["prompt_tokens"]),
                "output_tokens": len(result.tokens),
            }
        elif isinstance(result, TeacherScores):
            usage = {"input_tokens": len(payload["tokens"])}
        elif isinstance(result, UpdateResult):
            usage = {"input_tokens": payload["input_tokens"]}
        self.ledger.annotate(
            entry,
            {
                "run_id": self.record["run_id"],
                "usage": usage,
            },
        )
        return result

    def _score_step(
        self,
        result: ExperimentResult,
        tokens: Sequence[int],
        index: int,
        target_start: int,
    ) -> TeacherScores:
        self._roles()
        scores = self._call(
            "teacher_scoring",
            {
                "teacher_policy": self.snapshot.identity,
                "tokens": list(tokens),
                "target_start": target_start,
            },
            # SDK 0.2.9 implements scoring with an extra one-token decode.
            self.pricing.upper_bound(len(tokens), 1, scoring=True),
            partial(self.adapter.score, tokens),
        )
        # Preserve rejected responses locally before any identity/alignment check.
        (result.run_dir / f"private/teacher-{index:04d}.json").write_text(
            json.dumps(asdict(scores), indent=2)
        )
        self._roles()
        if (
            scores.policy != self.snapshot.identity
            or scores.encoding != result.encoding
        ):
            raise ValueError(
                "Teacher scoring identity or encoding differs from frozen initial model"
            )
        if list(scores.tokens) != list(tokens) or len(scores.logprobs) != len(tokens):
            raise ValueError(
                "Teacher token or probability lengths differ from the request"
            )
        return scores

    def _cost(self) -> dict[str, Any]:
        snapshot = self.ledger.snapshot()
        entries = [
            e
            for e in snapshot["entries"]
            if e["evidence"]["payload"].get("run_id") == self.record["run_id"]
        ]
        return summarize_cost(snapshot, entries)

    def _usage_metrics(self) -> dict[str, Any]:
        entries = [
            e
            for e in self.ledger.snapshot()["entries"]
            if e["evidence"]["payload"].get("run_id") == self.record["run_id"]
        ]
        metrics: dict[str, Any] = {}
        for operation, label in (
            ("sampling", "rollout"),
            ("teacher_scoring", "teacher"),
            ("opsd_update", "training"),
        ):
            rows = [e["evidence"] for e in entries if e["operation"] == operation]
            known = all("input_tokens" in row.get("usage", {}) for row in rows)
            token_count = sum(
                row.get("usage", {}).get("input_tokens", 0) for row in rows
            )
            metrics[f"train/{label}_usage_known"] = known
            metrics[f"train/{label}_submitted_input_tokens"] = sum(
                row["submitted_input_tokens"] for row in rows
            )
            metrics[
                f"train/{label}_input_tokens"
                if known
                else f"train/{label}_observed_input_tokens"
            ] = token_count
            metrics[f"train/{label}_seconds"] = sum(
                row.get("seconds", 0) for row in rows
            )
        sampling_entries = [e for e in entries if e["operation"] == "sampling"]
        output_known = all(
            "output_tokens" in e["evidence"].get("usage", {}) for e in sampling_entries
        )
        metrics["train/generated_tokens_known"] = output_known
        metrics[
            "train/generated_tokens"
            if output_known
            else "train/observed_generated_tokens"
        ] = sum(
            e["evidence"].get("usage", {}).get("output_tokens", 0)
            for e in sampling_entries
        )
        return metrics

    def finish(
        self,
        result: ExperimentResult,
        task: Task,
        model: Model,
        condition: ExperimentConfig,
    ) -> None:
        from bashlet.distillation.supervision import build_supervision

        batch: dict[str, Any] = {
            "index": len(self.record["batches"]),
            "episode": str(result.run_dir),
            "student_policy": result.policy,
            "teacher_policy": self.snapshot.identity,
            "status": "rejected",
            "reason": "",
            "supervised_tokens": 0,
            "rollout_seconds": result.usage.elapsed_seconds,
            "interaction_steps": result.usage.steps,
        }
        if task.assignment is not None:
            batch["task_assignment"] = asdict(task.assignment)
        result.opsd = batch
        self.record["batches"].append(batch)
        try:
            self._roles()
            if result.status == "interrupted":
                raise KeyboardInterrupt
            if not result.isolation.passed or result.status in (
                "model_error",
                "infrastructure_error",
                "interrupted",
                "invalid_submission",
            ):
                raise ValueError(
                    f"Episode cannot supply valid supervision: {result.status}"
                )
            if result.budget_kind == "spend" and not result.trajectory:
                raise BudgetRefused(
                    "No rollout fits within the remaining cash allocation"
                )
            validate_history(asdict(result), task, model.encoding, condition.protocol)
            if result.policy != self.record["student_policy"] or not result.trajectory:
                raise ValueError("Trajectory has no current-student sampling evidence")
            if len(result.trajectory) != len(self.calls):
                raise ValueError("Trajectory differs from the sampling calls")
            for step, (prompt, sampling, generation) in zip(
                result.trajectory,
                self.calls,
                strict=True,
            ):
                call = step.model_call
                if (
                    call is None
                    or call.policy != result.policy
                    or (
                        call.generation != generation
                        or call.sampling != sampling
                        or list(prompt) != result.context_tokens[: call.context_offset]
                    )
                ):
                    raise ValueError(
                        "Trajectory differs from original sampling evidence"
                    )
                if generation.source != "sampled":
                    raise ValueError("OPSD rejects non-sampled generation sources")
            datum, audit = build_supervision(
                result,
                task,
                model.encoding,
                condition,
                config=self.config,
                teacher_encoding=self.adapter.teacher_encoding,
                score=partial(self._score_step, result),
            )
            batch.update(audit)
            self._roles()
            (result.run_dir / "private/opsd.json").write_text(
                json.dumps({"datum": datum.model_dump(), "metadata": batch}, indent=2)
            )
            step = self.record["optimizer_step"] + 1
            update = self._call(
                "opsd_update",
                {
                    "optimizer_step": step,
                    "student_policy": result.policy,
                    "batch_sha256": hashlib.sha256(
                        json.dumps(datum.model_dump(), sort_keys=True).encode()
                    ).hexdigest(),
                    "input_tokens": len(datum.model_input.to_ints()),
                },
                self.pricing.training_bound(len(datum.model_input.to_ints())),
                lambda: self.adapter.update(
                    [datum],
                    loss_fn="importance_sampling",
                    auto_shift=False,
                    optimizer_step=step,
                    learning_rate=self.config.learning_rate,
                ),
            )
            if (
                update.optimizer_step != step
                or not math.isfinite(update.loss)
                or not update.student_policy
                or update.student_policy == result.policy
                or not update.training_state
                or update.training_state == update.student_policy
            ):
                raise ValueError("Invalid completed optimizer update receipt")
            batch.update(status="completed", update=asdict(update))
            self.record.update(
                optimizer_step=step,
                pending_update=None,
                student_policy=update.student_policy,
                last_update=asdict(update),
            )
            write_document(self.path, self.record)
            try:
                refreshed = self.adapter.refresh_student(update)
            except Exception as error:
                raise _RemoteFailure(error) from None
            if refreshed.identity != update.student_policy:
                raise ValueError("Refreshed sampler differs from the completed update")
            self._roles()
        except BudgetRefused:
            batch.update(status="rejected", reason="budget_refused")
        except UpdateRejected as error:
            self.record["pending_update"] = None
            batch.update(
                status="rejected", reason="remote_update_rejected:" + error.reason
            )
        except _RemoteFailure as error:
            batch.update(status="failed", reason=error.error_type)
        except ValueError as error:
            batch.update(
                status="failed"
                if self.record["pending_update"] is not None or "update" in batch
                else "rejected",
                reason=str(error),
            )
        except BaseException as error:
            batch.update(
                status="interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                reason=type(error).__name__,
            )
        finally:
            try:
                self._publish_batch(batch)
            except KeyboardInterrupt:
                batch.update(status="interrupted", reason="monitor_interrupted")
                self.record["status"] = "interrupted"
                write_document(self.path, self.record)
                raise
            finally:
                self._release_lock()

    def _publish_batch(self, batch: dict[str, Any]) -> None:
        batch["cost"] = self._cost()
        self.record["status"] = batch["status"]
        write_document(self.path, self.record)
        if self.monitor is not None:
            metrics: dict[str, Any] = {
                "train/optimizer_step": self.record["optimizer_step"],
                "train/distillation_coefficient": (
                    self.config.distillation_coefficient
                ),
                "train/supervised_tokens": batch["supervised_tokens"],
                "train/student_policy": batch["student_policy"],
                "train/teacher_policy": batch["teacher_policy"],
                "train/valid_batches": sum(
                    "update" in b for b in self.record["batches"]
                ),
                "train/rejected_batches": sum(
                    b["status"] == "rejected" for b in self.record["batches"]
                ),
                "run/status": batch["status"],
                "run/reason": batch["reason"],
                **spend_metrics(batch["cost"]),
                **self._usage_metrics(),
            }
            metrics["train/sampling_seconds"] = metrics["train/rollout_seconds"]
            metrics["train/rollout_seconds"] = sum(
                b["rollout_seconds"] for b in self.record["batches"]
            )
            if "sampled_token_reverse_kl" in batch:
                metrics["train/sampled_token_reverse_kl"] = batch[
                    "sampled_token_reverse_kl"
                ]
            if "update" in batch:
                metrics["train/loss"] = batch["update"]["loss"]
                metrics["checkpoint/sampler"] = batch["update"]["student_policy"]
                metrics["checkpoint/training_state"] = batch["update"]["training_state"]
            try:
                self.monitor.log(f"batch-{batch['index']}", metrics)
            finally:
                self.monitor.finish()
                self.monitor = None
