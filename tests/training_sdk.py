"""SDK-only training double: parameter changes happen when futures complete."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class Future:
    def __init__(self, action: Any) -> None:
        self.action = action

    def result(self, timeout: float | None = None) -> Any:
        return self.action()


class Training:
    model_id = "Qwen/Qwen3.5-4B"
    training_run_cuid = "train-test"

    def __init__(self, path: str | None = None) -> None:
        self.root = Path(os.environ["FAKE_TRAINING_ROOT"])
        self.path = self.root / "training-service.json"
        self.state = (
            json.loads(self.path.read_text()) if path else {"optimizer_step": 0}
        )
        if path:
            assert path.startswith("trio://train-test/state-")
            self.state["restored_optimizer"] = True

    def forward_backward(self, data: list[Any], **kwargs: Any) -> Future:
        assert kwargs["loss_fn"] == "cross_entropy" and kwargs["auto_shift"] is False
        with (self.root / "training-requests.jsonl").open("a") as stream:
            stream.write(
                json.dumps(kwargs | {"data": [datum.model_dump() for datum in data]})
                + "\n"
            )
        weights = [w for d in data for w in d.loss_fn_inputs["weights"].tolist()]
        self.state["supervised_tokens"] = int(sum(weights))
        self.state["observations_masked"] = 0 in weights and 1 in weights
        return Future(lambda: SimpleNamespace(metrics={"loss_sum": 2.0 * sum(weights)}))

    def optim_step(self, params: Any) -> Future:
        def complete() -> Any:
            if os.environ.get("FAKE_OPTIM_FAIL") or os.environ.get(
                "FAKE_OPTIM_FAIL_STEP"
            ) == str(self.state["optimizer_step"] + 1):
                raise TimeoutError("update still pending")
            self.state["optimizer_step"] += 1
            self.path.write_text(json.dumps(self.state))
            return SimpleNamespace(
                metrics={"optimizer_step": self.state["optimizer_step"]}
            )

        return Future(complete)

    def save_weights_for_sampler(self, name: str) -> Future:
        return Future(
            lambda: SimpleNamespace(
                path=f"trio://train-test/{name}", model=self.model_id, size=10
            )
        )

    def save_state(self, name: str) -> Future:
        return Future(
            lambda: SimpleNamespace(
                path=f"trio://train-test/{name}", model=self.model_id, size=20
            )
        )


class Service:
    workspace_cuid = "test-workspace"
    service_session_cuid = "test-session"

    def __init__(self, **kwargs: Any) -> None:
        pass

    def get_server_capabilities(self) -> Any:
        return SimpleNamespace(
            supported_models=[
                SimpleNamespace(
                    model_name=Training.model_id,
                    max_context_length=32768,
                    training=SimpleNamespace(
                        available=True,
                        max_seq_len=32768,
                        max_batch_size=64,
                        max_tokens_per_request=131072,
                    ),
                )
            ]
        )

    def create_lora_training_client(self, **kwargs: Any) -> Training:
        return Training()

    def create_training_client_from_state_with_optimizer(self, path: str) -> Training:
        return Training(path)


def install() -> None:
    import httpx
    import pytrio

    pytrio.ServiceClient = Service
    original = httpx.get

    def prices(url: str, **kwargs: Any) -> Any:
        if not url.endswith("/api/model-prices"):
            return original(url, **kwargs)
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "currency": "CNY",
                "token_unit": "per_1m_tokens",
                "rounding": "ceil",
                "version": "test",
                "items": [
                    {
                        "display_name": Training.model_id,
                        "prices": {
                            name: {"billing_unit": "per_1m_tokens", "unit_price": price}
                            for name, price in [
                                ("prefill", 149),
                                ("cached_prefill", 30),
                                ("sample", 454),
                                ("train", 454),
                            ]
                        },
                    }
                ],
            },
        )

    import pytest

    pytest.MonkeyPatch().setattr(httpx, "get", prices)
