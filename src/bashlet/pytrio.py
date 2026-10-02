"""Qwen3.5-4B sampling and raw-token scoring under the shared spend guard."""

import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import yaml
from dotenv import dotenv_values

from .models import (
    ContextExhausted,
    Encoding,
    Generation,
    JSONValue,
    ModelResponseError,
    SamplingParams,
)
from .spend import DEFAULT_POLICY, BudgetRefused, SpendLedger, SpendPolicy

MODEL = "Qwen/Qwen3.5-4B"
SDK_VERSION = "0.2.9"
_SDK_LOG = logging.getLogger("pytrio.transport.actor.sampling_run.sample")


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

    def __init__(self, run: str, ledger: SpendLedger) -> None:
        super().__init__(logging.DEBUG)
        self.run, self.ledger = run, ledger
        self.entry = ""
        self.sent = ""
        self.received = ""

    def emit(self, record: logging.LogRecord) -> None:
        args = record.args
        if not isinstance(args, tuple) or len(args) < 3 or args[0] != self.run:
            return
        if record.msg == "Sending Actor sample request task=%s request=%s actor=%s":
            self.sent = str(args[1])
            self.ledger.annotate(
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
            self.ledger.annotate(self.entry, {"response_request_id": self.received})


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
        ledger_path: str = "runs/spend.sqlite3",
        spend_config: str | None = None,
        ledger: SpendLedger | None = None,
        service_client: Any = None,
        api_key_env: str = "PYTRIO_API_KEY",
        dotenv_path: str = ".env",
        timeout_seconds: int = 60,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        group: str = "pytrio-smoke",
        operation: str = "sampling",
        probe_budget_cny: float = 0.20,
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
        if not math.isfinite(probe_budget_cny) or probe_budget_cny <= 0:
            raise ValueError("probe_budget_cny must be finite and positive")
        if encoding is not None and tokenizer is not None:
            raise ValueError("Supply either encoding or tokenizer")
        if encoding is None:
            if tokenizer != MODEL or not tokenizer_revision:
                raise ValueError("Use the pinned Qwen3.5-4B tokenizer")
            from .encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(
                tokenizer,
                revision=tokenizer_revision,
                thinking=thinking,
                cache_dir=tokenizer_cache_dir,
            )
        if version("pytrio") != SDK_VERSION:
            raise ValueError(f"Expected PyTRIO {SDK_VERSION}")
        policy_path = Path(spend_config) if spend_config else DEFAULT_POLICY
        self._pricing = yaml.safe_load(policy_path.read_text())
        self.ledger = ledger or SpendLedger(
            Path(ledger_path), policy=SpendPolicy.read(policy_path)
        )
        self.model, self.encoding = model, encoding
        self.model_path = model_path
        # The SDK accepts a base alias, not a Hub commit.
        self.identity = f"pytrio/{model}@{model_path or 'unversioned-base'}"
        self.timeout_seconds, self.max_retries = timeout_seconds, max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.group, self.operation = group, operation
        self._probe_cap = Decimal(str(probe_budget_cny))
        self._probe_reserved = Decimal(0)
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
            "probe_budget_cny": probe_budget_cny,
            "capability_probe": {"passed": False},
            "pricing": self._pricing,
        }

    def describe(self) -> dict[str, JSONValue]:
        return cast(
            dict[str, JSONValue],
            json.loads(
                json.dumps(
                    self._details
                    | {"retry_count": self._retries, "spend": self.ledger.snapshot()}
                )
            ),
        )

    def _connect(self) -> None:
        if self._sampler is not None:
            return
        if self._service is None:
            key = _credential(self._key_env, self._dotenv)
            self._verify_prices()
            import pytrio as trio

            # Each explicit retry needs its own reservation.
            trio.configure(
                home=str(self.ledger.path.parent / "pytrio-sdk"),
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
        self.ledger.bind_account(self._service.workspace_cuid)
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

    def _verify_prices(self) -> None:
        """Read the same public catalog as the official price page before spending."""
        response = httpx.get(
            self._pricing["price_catalog_url"], timeout=self.timeout_seconds
        )
        response.raise_for_status()
        catalog = response.json()
        if (
            catalog.get("currency") != "CNY"
            or catalog.get("token_unit") != "per_1m_tokens"
            or catalog.get("rounding") != "ceil"
        ):
            raise BudgetRefused("Unknown live PyTRIO billing units")
        item = next(
            (item for item in catalog["items"] if item["display_name"] == MODEL), None
        )
        if item is None:
            raise BudgetRefused("Qwen3.5-4B pricing is not available")
        configured = self._pricing["prices_cny_per_million"]
        for remote, local in (
            ("prefill", "prefill"),
            ("sample", "sample"),
            ("train", "train"),
            ("cached_prefill", "cache"),
        ):
            price = item["prices"][remote]
            if price["billing_unit"] != "per_1m_tokens" or Decimal(
                str(price["unit_price"])
            ) / 100 > Decimal(str(configured[local])):
                raise BudgetRefused(
                    "Live prices exceed the configured ceilings; review spend.yaml"
                )
        self._details["live_price_catalog"] = {
            "version": catalog["version"],
            "currency": "CNY",
            "unit_price_currency": "fen",
            "rounding": catalog["rounding"],
            "model": item,
        }

    def _upper_bound(
        self, prompt_tokens: int, max_tokens: int, *, scoring: bool = False
    ) -> Decimal:
        prices = self._pricing.get("prices_cny_per_million")
        if not isinstance(prices, dict) or any(
            type(prices.get(name)) not in (int, float)
            or not math.isfinite(prices[name])
            or prices[name] <= 0
            for name in ("prefill", "sample", "train")
        ):
            raise BudgetRefused("Unknown token pricing; cannot bound paid work")
        multiplier = self._pricing.get("ceiling_multiplier")
        if (
            type(multiplier) not in (int, float)
            or not math.isfinite(multiplier)
            or multiplier < 1
        ):
            raise BudgetRefused("A finite cost ceiling multiplier >= 1 is required")
        try:
            quantum = Decimal(str(self._pricing.get("billing_quantum_cny")))
        except InvalidOperation:
            raise BudgetRefused("Unknown billing rounding quantum") from None
        if not quantum.is_finite() or quantum < Decimal("0.01"):
            raise BudgetRefused(
                "A billing rounding quantum of at least one fen is required"
            )
        prefill = (
            max(prices[name] for name in ("prefill", "sample", "train"))
            if scoring
            else prices["prefill"]
        )

        def rounded(rate: float, tokens: int) -> Decimal:
            cost = Decimal(str(rate)) * tokens / 1_000_000
            return (cost / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum

        return Decimal(str(multiplier)) * (
            rounded(prefill, prompt_tokens) + rounded(prices["sample"], max_tokens)
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

        upper = self._upper_bound(len(prompt), sampling.max_tokens, scoring=scoring)
        payload = {
            "model": self.model,
            "prompt_tokens": list(prompt),
            "sampling": asdict(sampling),
            "scoring": scoring,
        }
        if self.model_path is not None:
            payload["model_path"] = self.model_path
        with self._lock:
            for attempt in range(self.max_retries + 1):
                probing = operation in ("capability_probe", "logprob_probe")
                if probing and self._probe_reserved + upper > self._probe_cap:
                    raise BudgetRefused("Capability probe cash ceiling is exhausted")
                capture = _RequestIdentity(self._sampler.sampling_run_cuid, self.ledger)
                _SDK_LOG.setLevel(logging.DEBUG)
                _SDK_LOG.addHandler(capture)

                def submit(
                    entry: str,
                    capture: _RequestIdentity = capture,
                    probing: bool = probing,
                ) -> Any:
                    if probing:
                        self._probe_reserved += upper
                    capture.entry = entry
                    self.ledger.annotate(
                        entry,
                        {
                            "service_session_cuid": self._service.service_session_cuid,
                            "sampling_run_cuid": self._sampler.sampling_run_cuid,
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

                try:
                    value, entry = self.ledger.execute(
                        operation=operation,
                        group=group,
                        payload=payload,
                        upper_cny=upper,
                        submit=submit,
                        timeout=self.timeout_seconds,
                        terminal_failure=self._terminal,
                    )
                    if not capture.sent or capture.received != capture.sent:
                        raise ModelResponseError(
                            "PyTRIO did not preserve its server request identity"
                        )
                    return value, capture.received, entry
                except (BudgetRefused, ModelResponseError, KeyboardInterrupt):
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
        raise RuntimeError("Unreachable retry state")

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
        self.ledger.annotate(
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
                raw = [*prompt, *generated.tokens]
                scores, score_id, entry = self._call(
                    raw,
                    sampling,
                    operation="logprob_probe",
                    group=self.group,
                    scoring=True,
                )
                if (
                    not isinstance(scores, list)
                    or len(scores) != len(raw)
                    or any(
                        type(value) not in (int, float)
                        or not math.isfinite(value)
                        or value > 0
                        for value in scores[1:]
                    )
                ):
                    raise ModelResponseError(
                        "PyTRIO scoring must align to each original token"
                    )
                assert generated.logprobs is not None
                difference = abs(scores[-1] - generated.logprobs[-1])
                self.ledger.annotate(entry, {"scored_tokens": raw, "logprobs": scores})
                if difference > 0.05:
                    raise ModelResponseError(
                        "Sample logprobs disagree with raw-token model scoring"
                    )
                self._details["capability_probe"] = {
                    "passed": True,
                    "generation": asdict(generated),
                    "prompt_tokens": prompt,
                    "scoring_request_id": score_id,
                    "scored_tokens": raw,
                    "scored_logprobs": scores,
                    "sample_score_max_difference": difference,
                    "probability_check": "model logprobs at temperature=1, top_p=1",
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
