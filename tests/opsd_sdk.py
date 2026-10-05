"""SDK boundary for OPSD; weights change only on completed optimizer futures."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from adapters import ByteEncoding, submit, write_file
from test_pytrio import FakeSampling, response
from training_sdk import Future, Training


class Trainer(Training):
    def __init__(self, path: str | None = None) -> None:
        super().__init__(path)
        self.requests: list[dict[str, Any]] = []

    def forward_backward(self, data: list[Any], **kwargs: Any) -> Future:
        self.requests.append(kwargs | {"data": data})
        count = sum(
            sum(value != 0 for value in d.loss_fn_inputs["logprobs"].tolist())
            for d in data
        )
        return Future(lambda: SimpleNamespace(metrics={"loss_sum": count * 0.125}))

    def save_state(self, name: str) -> Future:
        def saved() -> Any:
            self.path.write_text(json.dumps(self.state))
            return SimpleNamespace(
                path=f"trio://train-test/{name}", model=self.model_id
            )

        return Future(saved)


class Service:
    workspace_cuid = "test-workspace"
    service_session_cuid = "test-session"

    def __init__(self, **kwargs: Any) -> None:
        self.clients: dict[str, tuple[FakeSampling, str | None]] = {}
        self.trainer: Trainer | None = None
        self.wrong_loaded_path = False

    def get_server_capabilities(self) -> Any:
        return SimpleNamespace(
            supported_models=[
                SimpleNamespace(
                    model_name=Training.model_id,
                    max_context_length=32768,
                    sampling=SimpleNamespace(
                        available=True, max_seq_len=32768, max_tokens_per_request=32768
                    ),
                    training=SimpleNamespace(
                        available=True,
                        max_seq_len=32768,
                        max_batch_size=64,
                        max_tokens_per_request=32768,
                    ),
                )
            ]
        )

    def create_lora_training_client(self, **kwargs: Any) -> Trainer:
        self.trainer = Trainer()
        return self.trainer

    def create_training_client_from_state_with_optimizer(self, path: str) -> Trainer:
        self.trainer = Trainer(path)
        return self.trainer

    def create_training_client_from_state(self, path: str) -> Trainer:
        self.trainer = Trainer(path)
        self.trainer.state["optimizer_step"] = 0
        self.trainer.state["restored_optimizer"] = False
        return self.trainer

    def create_sampling_client(
        self, base_model: str, model_path: str | None = None
    ) -> Any:
        encoding: Any = ByteEncoding()
        if os.environ.get("FAKE_OPSD_TOKENIZER"):
            from bashlet.encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(
                os.environ["FAKE_OPSD_TOKENIZER"],
                revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            )
        sampler = FakeSampling(
            [
                response([65]),
                *[
                    reply
                    for _ in range(10)
                    for reply in (
                        response(
                            encoding.encode(write_file("def double(n): return n * 2\n"))
                        ),
                        response(encoding.encode(submit())),
                    )
                ],
            ]
        )
        if model_path and "sampler-0001" in model_path:
            for reply in sampler.replies:
                reply.sequences[0].logprobs = [-0.26] * len(reply.sequences[0].tokens)
            old_score = sampler.compute_logprobs

            def updated(prompt: Any) -> Any:
                old_score(prompt)
                from pytrio.types.future.api_future import APIFuture

                return APIFuture.completed([None, *([-0.26] * (prompt.length - 1))])

            sampler.compute_logprobs = updated  # type: ignore[method-assign]
        if len(self.clients) == 1:
            original = sampler.compute_logprobs

            def privileged(prompt: Any) -> Any:
                original(prompt)
                from pytrio.types.future.api_future import APIFuture

                context = (
                    encoding.render(prompt.to_ints())
                    if isinstance(encoding, ByteEncoding)
                    else encoding.decode(prompt.to_ints())
                )
                probability = (
                    -0.1
                    if "Reference implementation (teacher only)" in context
                    else -0.25
                )
                return APIFuture.completed(
                    [None, *([probability] * (prompt.length - 1))]
                )

            sampler.compute_logprobs = privileged  # type: ignore[method-assign]
        self.clients[sampler.sampling_run_cuid] = sampler, model_path
        return sampler

    def create_rest_client(self) -> Any:
        def metadata(identity: str) -> Future:
            _, path = self.clients[identity]
            return Future(
                lambda: SimpleNamespace(
                    base_model=Training.model_id,
                    model_path="trio://wrong/sampler"
                    if self.wrong_loaded_path
                    else path,
                )
            )

        return SimpleNamespace(get_sampler=metadata)


def install() -> None:
    import pytrio
    import training_sdk

    training_sdk.install()
    pytrio.ServiceClient = Service
    import monitoring_sdk

    old_log = monitoring_sdk.Run.log

    def log(self: Any, data: dict[str, Any], *, step: int) -> None:
        old_log(self, data, step=step)
        marker = Path(os.environ["FAKE_TRAINING_ROOT"]) / "eval-interrupted"
        if (
            os.environ.get("FAKE_OPSD_INTERRUPT_EVAL")
            and data.get("eval/completed_tasks") == 1
            and not marker.exists()
        ):
            marker.touch()
            raise KeyboardInterrupt

    monitoring_sdk.Run.log = log  # type: ignore[method-assign]
