"""Qwen3.5-4B sampling and raw-token scoring with durable request records."""

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import dotenv_values

from bashlet.backends.journal import (
    RequestConflict,
    RequestJournal,
)
from bashlet.harness.models import (
    ContextExhausted,
    Encoding,
    Generation,
    JSONValue,
    ModelResponseError,
    SamplingParams,
)

MODEL = "Qwen/Qwen3.5-4B"
SDK_VERSION = "0.2.9"
_SDK_LOG = logging.getLogger("pytrio.transport.actor.sampling_run.sample")


def validate_training_capacity(lengths: list[int], limits: Any) -> None:
    if not lengths or min(lengths) < 1:
        raise ValueError("Training requires nonempty inputs")
    for value, limit in (
        (len(lengths), limits.max_batch_size),
        (max(lengths), limits.max_seq_len),
        (sum(lengths), limits.max_tokens_per_request),
    ):
        if limit is not None and value > limit:
            raise ValueError("Training batch exceeds the remote capacity")


def save_training_checkpoints(
    client: Any,
    model: str,
    step: int,
    deadline: float,
) -> tuple[str, str]:
    paths = []
    for save, name in (
        (client.save_weights_for_sampler, f"sampler-{step:04d}"),
        (client.save_state, f"state-{step:04d}"),
    ):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Checkpoint deadline exhausted")
        receipt = save(name).result(timeout=remaining)
        if receipt.model != model or not receipt.path:
            raise ValueError("Checkpoint identity or model differs")
        paths.append(receipt.path)
    if paths[0] == paths[1]:
        raise ValueError("Sampler and training state must be distinct checkpoints")
    return paths[0], paths[1]


def _credential(env: str, dotenv: Path) -> str:
    key = os.environ.get(env, "").strip()
    if not key and dotenv.is_file():
        key = (dotenv_values(dotenv, interpolate=False).get(env) or "").strip()
        if not key:
            raw = dotenv.read_text().strip()
            if re.fullmatch(r"trio_[A-Za-z0-9]+", raw):
                key = raw
    if not key:
        raise ValueError(f"Missing PyTRIO credential: set {env} or put it in {dotenv}")
    return key


class _RequestIdentity(logging.Handler):
    """SDK 0.2.9 drops raw response IDs; retain its transport identity events.

    Only the two fixed, credential-free messages are consumed. The submitted
    request ID is persisted before the SDK sends the HTTP request. The returned
    ID must match it. Each sampling client has at most one active call here.
    """

    def __init__(self, run: str, journal: RequestJournal | None = None) -> None:
        super().__init__(logging.DEBUG)
        self.run, self.journal = run, journal
        self.entry = ""
        self.sent = ""
        self.received = ""

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3 or args[0] != self.run:
            return
        if record.msg == "Sending Actor sample request task=%s request=%s actor=%s":
            self.sent = str(args[1])
            if self.journal is None:
                return
            self.journal.annotate(
                self.entry,
                {
                    "request_id": self.sent,
                    "sampling_run_cuid": self.run,
                    "actor_cuid": str(args[2]),
                },
            )
        elif (
            record.msg
            == "Received Actor sample response task=%s request=%s sequences=%s actor=%s"
        ):
            self.received = str(args[1])
            if self.journal is not None:
                self.journal.annotate(
                    self.entry, {"response_request_id": self.received}
                )


class PyTRIOModel:
    """Use the official SDK; never silently replace tokens or probability evidence."""

    def __init__(
        self,
        *,
        model: str,
        model_revision: str,
        model_path: str | None = None,
        encoding: Encoding | None = None,
        tokenizer: str | None = None,
        tokenizer_revision: str | None = None,
        tokenizer_cache_dir: str | None = None,
        thinking: bool = False,
        journal_path: str = "runs/requests.sqlite3",
        journal: RequestJournal | None = None,
        service_client: Any = None,
        api_key_env: str = "PYTRIO_API_KEY",
        dotenv_path: str = ".env",
        timeout_seconds: int = 60,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        group: str = "pytrio-smoke",
        operation: str = "sampling",
        run_id: str | None = None,
    ) -> None:
        if model != MODEL:
            raise ValueError(f"This experiment requires {MODEL}")
        if not model_revision:
            raise ValueError("Record the reference model revision")
        if model_path is not None and (
            not isinstance(model_path, str) or not model_path.strip()
        ):
            raise ValueError("model_path must be a nonempty sampler checkpoint URI")
        if type(timeout_seconds) is not int or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive integer")
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if not math.isfinite(retry_delay_seconds) or retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        if encoding is not None and tokenizer is not None:
            raise ValueError("Supply either encoding or tokenizer")
        if encoding is None:
            if tokenizer != MODEL or not tokenizer_revision:
                raise ValueError("Use the pinned Qwen3.5-4B tokenizer")
            from bashlet.backends.encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(
                tokenizer,
                revision=tokenizer_revision,
                thinking=thinking,
                cache_dir=tokenizer_cache_dir,
            )
        if version("pytrio") != SDK_VERSION:
            raise ValueError(f"Expected PyTRIO {SDK_VERSION}")
        self.journal = journal or RequestJournal(Path(journal_path))
        self.model, self.encoding = model, encoding
        self.model_path = model_path
        # The SDK accepts a base alias, not a Hub commit.
        self.identity = f"pytrio/{model}@{model_path or 'unversioned-base'}"
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.group, self.operation = group, operation
        self.run_id = run_id
        self._service = service_client
        self._sampler: Any = None
        self._key_env, self._dotenv = api_key_env, Path(dotenv_path)
        self._lock = threading.RLock()
        self._probed = False
        self._retries = 0
        self.max_context_tokens = 0
        self._details: dict[str, Any] = {
            "backend": "pytrio",
            "development_only": False,
            "sdk_version": SDK_VERSION,
            "model": model,
            "model_path": model_path,
            "snapshot_kind": "sampler_weights" if model_path else "unversioned_base",
            "reference_model_revision": model_revision,
            "server_model_revision": None,
            "revision_limitation": "SDK exposes an alias, not a weight commit",
            "capability_probe": {"passed": False},
        }

    def describe(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            json.loads(
                json.dumps(
                    self._details
                    | {
                        "retry_count": self._retries,
                        "requests": self.journal.snapshot(),
                    }
                )
            ),
        )

    def validate_request_owner(
        self, journal: RequestJournal, group: str, run_id: str
    ) -> None:
        if (
            self.journal.path != journal.path
            or self.group != group
            or self.run_id != run_id
        ):
            raise ValueError(
                "Student must share the OPSD request journal, group and run"
            )

    def _connect(self) -> None:
        if self._sampler is not None:
            return
        if self._service is None:
            key = _credential(self._key_env, self._dotenv)
            import pytrio as trio

            # Persist each explicit retry as a separate request attempt.
            trio.configure(
                home=str(self.journal.path.parent / "pytrio-sdk"),
                request_timeout_seconds=self.timeout_seconds,
                retry_max_attempts=1,
                control_retry_timeout_seconds=0,
                sampling_retry_timeout_seconds=0,
                training_retry_timeout_seconds=0,
                sampling_timeout_seconds=self.timeout_seconds,
                sampling_first_chunk_timeout_seconds=self.timeout_seconds,
                sampling_chunk_idle_timeout_seconds=self.timeout_seconds,
                actor_capacity_wait_timeout_seconds=self.timeout_seconds,
                log_level="WARNING",
            )
            self._service = trio.ServiceClient(api_key=key)
        capabilities = self._service.get_server_capabilities()
        supported = next(
            (
                item
                for item in capabilities.supported_models
                if item.model_name == self.model
            ),
            None,
        )
        if supported is None or not supported.sampling.available:
            raise RuntimeError(f"PyTRIO workspace cannot sample {self.model}")
        limits = [
            supported.max_context_length,
            supported.sampling.max_seq_len,
            supported.sampling.max_tokens_per_request,
        ]
        known = [value for value in limits if type(value) is int and value >= 2]
        if not known:
            raise ModelResponseError("PyTRIO must report a finite context limit")
        self.max_context_tokens = min(known)
        self.journal.bind_account(self._service.workspace_cuid)
        arguments = {"base_model": self.model}
        if self.model_path is not None:
            arguments["model_path"] = self.model_path
        sampler = self._service.create_sampling_client(**arguments)
        if sampler.model_id != self.model:
            raise ModelResponseError("PyTRIO created a different model")
        self._sampler = sampler
        self._details.update(
            {
                "max_context_tokens": self.max_context_tokens,
                "workspace_cuid": self._service.workspace_cuid,
                "service_session_cuid": self._service.service_session_cuid,
                "sampling_run_cuid": self._sampler.sampling_run_cuid,
            }
        )

    @staticmethod
    def _terminal(error: BaseException) -> bool:
        # A timeout/gateway failure may conceal continuing remote work.
        status = getattr(error, "status_code", 0)
        return status in (400, 401, 403, 404, 409, 422, 429)

    def _call(
        self,
        prompt: Sequence[int],
        sampling: SamplingParams,
        *,
        operation: str,
        group: str,
        scoring: bool = False,
    ) -> tuple[Any, str, str]:
        import pytrio as trio

        payload = {
            "model": self.model,
            "prompt_tokens": list(prompt),
            "sampling": asdict(sampling),
            "scoring": scoring,
        }
        if self.run_id is not None:
            payload["run_id"] = self.run_id
        if self.model_path is not None:
            payload["model_path"] = self.model_path
        with self._lock:
            for attempt in range(self.max_retries + 1):
                capture = _RequestIdentity(
                    self._sampler.sampling_run_cuid, self.journal
                )
                _SDK_LOG.setLevel(logging.DEBUG)
                _SDK_LOG.addHandler(capture)

                def submit(
                    entry: str,
                    capture: _RequestIdentity = capture,
                ) -> Any:
                    capture.entry = entry
                    self.journal.annotate(
                        entry,
                        {
                            "service_session_cuid": self._service.service_session_cuid,
                            "sampling_run_cuid": self._sampler.sampling_run_cuid,
                            "submitted_input_tokens": len(prompt),
                        },
                    )
                    token_input = trio.ModelInput.from_ints(list(prompt))
                    if scoring:
                        return self._sampler.compute_logprobs(prompt=token_input)
                    return self._sampler.sample(
                        prompt=token_input,
                        num_samples=1,
                        sampling_params=trio.SamplingParams(
                            **asdict(sampling),
                            stop=list(self.encoding.stop_tokens),
                        ),
                        return_text=False,
                    )

                started = time.monotonic()
                try:
                    value, entry = self.journal.execute(
                        operation=operation,
                        group=group,
                        payload=payload,
                        submit=submit,
                        timeout=self.timeout_seconds,
                        terminal_failure=self._terminal,
                    )
                    if not capture.sent or capture.received != capture.sent:
                        raise ModelResponseError(
                            "PyTRIO did not preserve its server request identity"
                        )
                    return value, capture.received, entry
                except (RequestConflict, ModelResponseError, KeyboardInterrupt):
                    raise
                except Exception as error:
                    code = getattr(error, "code", type(error).__name__)
                    status = getattr(error, "status_code", 0)
                    reason = f"PyTRIO {code} (HTTP {status}, attempt {attempt + 1})"
                    if code == "actor.invalid_response":
                        raise ModelResponseError(reason) from None
                    if status == 400 and code in (
                        "context_length_exceeded",
                        "max_context_length_exceeded",
                    ):
                        raise ContextExhausted(reason) from None
                    if (
                        getattr(error, "retryable", False)
                        and self._terminal(error)
                        and attempt < self.max_retries
                    ):
                        self._retries += 1
                        time.sleep(self.retry_delay_seconds)
                        continue
                    raise RuntimeError(reason) from None
                finally:
                    _SDK_LOG.removeHandler(capture)
                    if capture.entry:
                        self.journal.annotate(
                            capture.entry, {"seconds": time.monotonic() - started}
                        )
        raise RuntimeError("Unreachable retry state")

    def verify_loaded_sampler(self) -> dict[str, JSONValue]:
        """Check Control metadata against the exact requested sampler weights."""
        self._connect()
        metadata = (
            self._service.create_rest_client()
            .get_sampler(self._sampler.sampling_run_cuid)
            .result(timeout=self.timeout_seconds)
        )
        if metadata.base_model != self.model or metadata.model_path != self.model_path:
            raise ModelResponseError(
                "Server sampler metadata differs from requested weights"
            )
        proof: dict[str, JSONValue] = {
            "sampling_run_cuid": self._sampler.sampling_run_cuid,
            "base_model": metadata.base_model,
            "model_path": metadata.model_path,
        }
        self._details["load_proof"] = proof
        return proof

    def _generate(
        self, prompt: Sequence[int], sampling: SamplingParams, *, probe: bool = False
    ) -> Generation:
        remaining = self.max_context_tokens - len(prompt)
        if remaining <= 0:
            raise ContextExhausted(
                f"Reached PyTRIO context limit={self.max_context_tokens}"
            )
        sampling = replace(sampling, max_tokens=min(sampling.max_tokens, remaining))
        response, request_id, entry = self._call(
            prompt,
            sampling,
            operation="capability_probe" if probe else self.operation,
            group=self.group,
        )
        sequences = getattr(response, "sequences", None)
        if not isinstance(sequences, list) or len(sequences) != 1:
            raise ModelResponseError("PyTRIO must return exactly one completion")
        seq = sequences[0]
        tokens, logprobs = getattr(seq, "tokens", None), getattr(seq, "logprobs", None)
        if not isinstance(tokens, list):
            raise ModelResponseError("PyTRIO requires raw token IDs")
        if logprobs is not None and not isinstance(logprobs, list):
            raise ModelResponseError("PyTRIO logprobs must be a list")
        stop = getattr(seq, "stop_reason", None)
        if stop not in ("stop", "length"):
            raise ModelResponseError(f"Unknown PyTRIO stop reason {stop!r}")
        reason: Literal["stop", "length", "context"] = (
            "context" if stop == "length" and len(tokens) == remaining else stop
        )
        generation = Generation(
            tuple(tokens),
            reason,
            tuple(logprobs) if logprobs is not None else None,
            request_id,
        )
        generation.validate(sampling)
        if response.input_tokens != len(prompt) or response.output_tokens != len(
            tokens
        ):
            raise ModelResponseError(
                "PyTRIO usage does not match raw token lengths", generation
            )
        self.journal.annotate(
            entry,
            {
                "usage": {
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                },
                "generation": asdict(generation),
            },
        )
        return generation

    def probe(self) -> None:
        """Validate sampled token evidence without a probability comparison."""
        with self._lock:
            if self._probed:
                return
            try:
                self._connect()
                prompt = self.encoding.prompt("", "Hi")
                sampling = SamplingParams(max_tokens=1, temperature=1, top_p=1, seed=4)
                generated = self._generate(prompt, sampling, probe=True)
                if len(generated.tokens) != 1:
                    raise ModelResponseError(
                        "PyTRIO capability probe must return one token"
                    )
                self._details["capability_probe"] = {
                    "passed": True,
                    "generation": asdict(generated),
                    "prompt_tokens": prompt,
                    "frozen_sft_teacher_verified": False,
                }
                self._probed = True
            except Exception as error:
                if hasattr(error, "code") and hasattr(error, "status_code"):
                    reason = f"PyTRIO {error.code} (HTTP {error.status_code})"
                    self._details["capability_probe"] = {
                        "passed": False,
                        "reason": reason,
                    }
                    raise RuntimeError(reason) from None
                self._details["capability_probe"] = {
                    "passed": False,
                    "reason": str(error),
                }
                raise

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        self.probe()
        return self._generate(prompt, sampling)
