"""Token-in/token-out sampling with optional verified local snapshot provenance."""

import inspect
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import httpx

from bashlet.harness.models import (
    ContextExhausted,
    Encoding,
    Generation,
    JSONValue,
    ModelResponseError,
    SamplingParams,
)

VLLM_VERSION = "0.11.0"
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
_REQUEST_FIELDS = ("timeout_seconds", "max_retries", "retry_delay_seconds")


def configured_request_policy(model: dict[str, Any]) -> dict[str, Any]:
    """Use the constructor's defaults when a VLLM config omits HTTP settings."""
    vllm = model.get("class", "").replace(":", ".").split(".")[-1] == "VLLMModel"
    signature = inspect.signature(VLLMModel)
    policy = {
        name: model.get(name, signature.parameters[name].default)
        for name in _REQUEST_FIELDS
        if vllm or name in model
    }
    return {
        name: float(value) if name != "max_retries" else value
        for name, value in policy.items()
    }


class VLLMModel:
    """Keep raw server token IDs and sampled-token logprobs as development evidence."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        model_revision: str,
        encoding: Encoding | None = None,
        tokenizer: str | None = None,
        tokenizer_revision: str | None = None,
        tokenizer_cache_dir: str | None = None,
        thinking: bool = False,
        timeout_seconds: float = 60,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.5,
        api_key_env: str | None = "BASHLET_VLLM_API_KEY",
        snapshot_manifest: str | None = None,
    ) -> None:
        if type(max_retries) is not int or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        for name, value in (
            ("timeout_seconds", timeout_seconds),
            ("retry_delay_seconds", retry_delay_seconds),
        ):
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name == "timeout_seconds" and value == 0)
            ):
                raise ValueError(f"{name} must be finite and within its valid range")
        if not model or not model_revision:
            raise ValueError("model and model_revision must identify fixed weights")
        if encoding is not None and tokenizer is not None:
            raise ValueError("Supply either encoding or tokenizer")
        if encoding is None:
            if tokenizer is None or tokenizer_revision is None:
                raise ValueError("tokenizer and tokenizer_revision are required")
            from bashlet.backends.encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(
                tokenizer,
                revision=tokenizer_revision,
                thinking=thinking,
                cache_dir=tokenizer_cache_dir,
            )
        url = httpx.URL(base_url)
        if (
            url.scheme not in ("http", "https")
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
            or url.path not in ("", "/")
        ):
            raise ValueError("base_url must be the service origin without credentials")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.encoding = encoding
        self.identity = f"dev/vllm/{model}@{model_revision}"
        self._snapshot = None
        if snapshot_manifest is not None:
            from bashlet.snapshots import LocalSnapshot

            self._snapshot = LocalSnapshot(
                Path(snapshot_manifest), model_revision, encoding
            )
            self.identity = self._snapshot.identity
        self.service_version = (
            self._snapshot.service_version if self._snapshot else VLLM_VERSION
        )
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.api_key_env = api_key_env
        self._retries = 0
        self._probed = False
        self.max_context_tokens = 0
        self._details: dict[str, JSONValue] = {
            "backend": "vllm",
            "development_only": self._snapshot is None,
            "base_url": self.base_url,
            "model": model,
            "model_revision": model_revision,
            "service_version": None,
            "required_service_version": self.service_version,
            "identity_verified": False,
            "capability_probe": {"passed": False},
            "request_policy": {
                name: float(getattr(self, name))
                if name != "max_retries"
                else getattr(self, name)
                for name in _REQUEST_FIELDS
            },
        }
        if self._snapshot is not None:
            self._details.update(self._snapshot.details)

    def describe(self) -> dict[str, JSONValue]:
        return self._details | {"retry_count": self._retries}

    def _request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.request(
                    method,
                    self.base_url + path,
                    json=body,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    trust_env=False,
                )
                if response.status_code == 400:
                    try:
                        error_body = response.json()
                    except ValueError:
                        error_body = None
                    info = (
                        error_body.get("error")
                        if isinstance(error_body, dict)
                        else None
                    )
                    message = info.get("message") if isinstance(info, dict) else None
                    if isinstance(message, str) and (
                        message.startswith("This model's maximum context length is ")
                        or (
                            message.startswith(
                                "'max_tokens' or 'max_completion_tokens' is too large: "
                            )
                            and "maximum context length" in message
                        )
                    ):
                        raise ContextExhausted(message)
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                retryable = isinstance(error, httpx.TransportError) or (
                    error.response.status_code in _TRANSIENT_STATUS
                )
                if not retryable or attempt == self.max_retries:
                    raise RuntimeError(
                        f"vLLM request failed after {attempt + 1} attempt(s): {error}"
                    ) from error
                self._retries += 1
                time.sleep(self.retry_delay_seconds)
        try:
            data = response.json()
        except ValueError as error:
            raise ModelResponseError("vLLM returned invalid JSON") from error
        if not isinstance(data, dict):
            raise ModelResponseError("vLLM response must be a JSON object")
        return data

    def probe(self) -> None:
        """Before any episode, prove that the pinned service supplies token evidence."""
        if self._probed:
            return
        try:
            version = self._request("GET", "/version").get("version")
            self._details["service_version"] = version
            if version != self.service_version:
                raise RuntimeError(
                    f"Expected vLLM {self.service_version}, got {version!r}"
                )
            models = self._request("GET", "/v1/models").get("data")
            if not isinstance(models, list):
                raise ModelResponseError("vLLM did not return a model list")
            served = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict) and item.get("id") == self.model
                ),
                None,
            )
            if served is None:
                raise RuntimeError(f"vLLM is not serving {self.model!r}")
            if self._snapshot is not None:
                config = self._request("GET", "/server_info?config_format=json").get(
                    "vllm_config"
                )
                if not isinstance(config, dict):
                    raise ModelResponseError(
                        "vLLM must supply its actual model configuration"
                    )
                self._details["inference_conditions"] = self._snapshot.verify_service(
                    served, config, models
                )
                probe_text = "Hi <|im_start|>assistant\n<think>\n</think><|im_end|>"
                remote_tokens = self._request(
                    "POST",
                    "/tokenize",
                    {
                        "model": self.model,
                        "prompt": probe_text,
                        "add_special_tokens": False,
                    },
                ).get("tokens")
                if remote_tokens != self.encoding.encode(probe_text):
                    raise ModelResponseError(
                        "Service tokenizer does not match the local encoding"
                    )
                self._details["load_proof"] = {
                    "served_model": served,
                    "parent_model": next(
                        (
                            item
                            for item in models
                            if isinstance(item, dict)
                            and item.get("id") == served.get("parent")
                        ),
                        None,
                    ),
                    "model_config": config.get("model_config"),
                    "load_config": config.get("load_config"),
                    "lora_config": config.get("lora_config"),
                    "tokenizer_probe": {"prompt": probe_text, "tokens": remote_tokens},
                }
            self._details["served_model_root"] = served.get("root")
            context_limit = served.get("max_model_len")
            if type(context_limit) is not int or context_limit < 2:
                raise ModelResponseError("vLLM must advertise a valid max_model_len")
            self.max_context_tokens = context_limit
            self._details["max_model_len"] = context_limit
            prompt = self.encoding.encode("Hi")[:1]
            sampling = SamplingParams(max_tokens=1, temperature=0, seed=0)
            generation = self._generate(prompt, sampling)
            if len(generation.tokens) != 1:
                raise ModelResponseError("Capability probe must return one token")
            self._details["capability_probe"] = {
                "passed": True,
                "prompt_tokens": [*prompt],
                "generation": asdict(generation),
            }
            self._probed = True
            self._details["identity_verified"] = self._snapshot is not None
        except Exception as error:
            self._details["capability_probe"] = {"passed": False, "reason": str(error)}
            raise

    def _generate(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        remaining = self.max_context_tokens - len(prompt)
        if remaining <= 0:
            raise ContextExhausted(
                f"Reached vLLM max_model_len={self.max_context_tokens}"
            )
        sampling = replace(sampling, max_tokens=min(sampling.max_tokens, remaining))
        data = self._request(
            "POST",
            "/v1/completions",
            {
                "model": self.model,
                "prompt": list(prompt),
                "max_tokens": sampling.max_tokens,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "seed": sampling.seed,
                "n": 1,
                "stream": False,
                "echo": False,
                "add_special_tokens": False,
                "return_token_ids": True,
                "return_tokens_as_token_ids": True,
                "logprobs": 0,
                "stop_token_ids": list(self.encoding.stop_tokens),
                "skip_special_tokens": False,
            },
        )
        choices = data.get("choices")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
        ):
            raise ModelResponseError("vLLM must return exactly one completion")
        choice = choices[0]
        tokens = choice.get("token_ids")
        if not isinstance(tokens, list):
            raise ModelResponseError("vLLM response requires token_ids")
        logprobs = choice.get("logprobs")
        values = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
        if values is not None and not isinstance(values, list):
            raise ModelResponseError("vLLM token_logprobs must be a list")
        reason = choice.get("finish_reason")
        if reason not in ("stop", "length"):
            raise ModelResponseError(f"Unknown vLLM finish_reason: {reason!r}")
        stop_reason: Literal["stop", "length", "context"] = (
            "context"
            if reason == "length" and len(tokens) == remaining
            else "length"
            if reason == "length"
            else "stop"
        )
        request_id = data.get("id")
        if not isinstance(request_id, str) or not request_id:
            raise ModelResponseError("vLLM response requires a request id")
        generation = Generation(
            tokens=tuple(tokens),
            logprobs=tuple(values) if values is not None else None,
            stop_reason=stop_reason,
            request_id=request_id,
        )
        generation.validate(sampling)
        if data.get("model") != self.model:
            raise ModelResponseError("vLLM returned a different model", generation)
        if choice.get("prompt_token_ids") != list(prompt):
            raise ModelResponseError("vLLM changed the prompt token IDs", generation)
        assert isinstance(logprobs, dict)
        if logprobs.get("tokens") != [f"token_id:{token}" for token in tokens]:
            raise ModelResponseError(
                "vLLM logprobs do not identify the sampled token IDs", generation
            )
        return generation

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        self.probe()
        return self._generate(prompt, sampling)
