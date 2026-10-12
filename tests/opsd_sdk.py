"""SDK boundary for OPSD; weights change only on completed optimizer futures."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from adapters import ByteEncoding, submit, write_file
from test_pytrio import FakeSampling, response
from training_sdk import Future, Training


class NativeSampling(FakeSampling):
    """Return native thinking only when the actual prompt opens it."""

    def __init__(self, replies: list[Any], encoding: Any) -> None:
        super().__init__(replies)
        self.encoding = encoding

    def sample(self, **request: Any) -> Any:
        if os.environ.get("FAKE_OPSD_TOKENIZER") and self.replies:
            encoding = self.encoding
            prompt = encoding.decode(request["prompt"].to_ints())
            if prompt.endswith("<think>\n"):
                sequence = self.replies[0].sequences[0]
                body = encoding.decode(sequence.tokens)
                text = "Consider only: ```bash\ntouch MUST_NOT_RUN\n```\n"
                if os.environ.get("FAKE_OPSD_UNFINISHED"):
                    sequence.stop_reason = "length"
                else:
                    text += "</think>\n" + body + "<|im_end|>"
                sequence.tokens = encoding.encode(text)
                probability = sequence.logprobs[0]
                sequence.logprobs = [probability] * len(sequence.tokens)
                self.replies[0].output_tokens = len(sequence.tokens)
        sequence = self.replies[0].sequences[0]
        limit = request["sampling_params"].max_tokens
        if len(sequence.tokens) > limit:
            sequence.tokens = sequence.tokens[:limit]
            sequence.logprobs = sequence.logprobs[:limit]
            sequence.stop_reason = "length"
            self.replies[0].output_tokens = len(sequence.tokens)
        return super().sample(**request)


class Trainer(Training):
    def __init__(self, path: str | None = None) -> None:
        super().__init__(path)
        self.requests: list[dict[str, Any]] = []

    def forward_backward(self, data: list[Any], **kwargs: Any) -> Future:
        self.requests.append(kwargs | {"data": data})
        destination = Path(os.environ["FAKE_TRAINING_ROOT"]) / "sdk-updates.json"
        saved = json.loads(destination.read_text()) if destination.exists() else []
        saved.append(
            kwargs
            | {
                "data": [
                    {
                        "model_input": datum.model_input.to_ints(),
                        "loss_fn_inputs": {
                            key: value.tolist()
                            for key, value in datum.loss_fn_inputs.items()
                        },
                    }
                    for datum in data
                ]
            }
        )
        destination.write_text(json.dumps(saved))
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

    def save_weights_for_sampler(self, name: str) -> Future:
        return Future(
            lambda: SimpleNamespace(
                path=f"trio://train-test/sampler_weights/{name}",
                model=self.model_id,
                size=10,
            )
        )


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
            ],
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
        sampler = NativeSampling(
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
            ],
            encoding,
        )
        if (
            model_path
            and "/sampler-" in model_path
            and not model_path.endswith("sampler-0000")
        ):
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
                if os.environ.get("FAKE_FROZEN_SCORE_VARIATION"):
                    probability -= 0.2
                if os.environ.get("FAKE_FROZEN_SCORE_INVALID"):
                    probability = 0.1
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

        def response(data: dict[str, Any]) -> Any:
            return SimpleNamespace(**data, model_dump=lambda **kwargs: data)

        def checkpoints(run: str) -> Future:
            return Future(
                lambda: SimpleNamespace(
                    checkpoints=[
                        response(
                            {
                                "checkpoint_id": f"ckpt-{step}",
                                "checkpoint_type": "sampler",
                                "path": f"trio://{run}/sampler_weights/sampler-{step:04d}",
                            }
                        )
                        for step in range(5)
                    ]
                )
            )

        def download(checkpoint: str, destination: Path, **kwargs: Any) -> Future:
            def saved() -> Any:
                import tempfile
                import zipfile

                import numpy as np
                from safetensors.numpy import save_file

                step = int(checkpoint.removeprefix("ckpt-"))
                value = float(step > 0)
                if os.environ.get("FAKE_TEACHER_WEIGHTS_CHANGED") and step == 0:
                    value = 1.0
                if os.environ.get("FAKE_STUDENT_WEIGHTS_UNCHANGED"):
                    value = 0.0
                with tempfile.TemporaryDirectory() as temporary:
                    weights = Path(temporary) / "adapter_model.safetensors"
                    save_file(
                        {
                            (
                                "model.language_model.layers.0.self_attn.q_proj."
                                "lora_A.weight"
                            ): np.ones((16, 2), dtype=np.float32),
                            (
                                "model.language_model.layers.0.self_attn.q_proj."
                                "lora_B.weight"
                            ): np.full((2, 16), value, dtype=np.float32),
                        },
                        weights,
                    )
                    with zipfile.ZipFile(destination, "w") as zipped:
                        zipped.write(weights, "checkpoint/adapter_model.safetensors")
                        zipped.writestr(
                            "checkpoint/adapter_config.json",
                            json.dumps(
                                {
                                    "peft_type": "LORA",
                                    "r": 16,
                                    "lora_alpha": 32,
                                    "base_model_name_or_path": Training.model_id,
                                    "target_modules": ["q_proj"],
                                }
                            ),
                        )
                return response(
                    {"checkpoint_id": checkpoint, "status_code": 200, "resumed": False}
                )

            return Future(saved)

        return SimpleNamespace(
            get_sampler=metadata,
            get_training_run=lambda run: Future(
                lambda: response(
                    {
                        "training_run_id": run,
                        "base_model": Training.model_id,
                        "is_lora": True,
                        "corrupted": False,
                        "lora_rank": 16,
                    }
                )
            ),
            list_checkpoints=checkpoints,
            download_checkpoint=download,
        )


def install() -> None:
    import pytrio
    import training_sdk

    training_sdk.install()
    pytrio.ServiceClient = Service
    import monitoring_sdk

    old_log = monitoring_sdk.Run.log

    def log(self: Any, data: dict[str, Any], *, step: int) -> None:
        old_log(self, data, step=step)
        train_marker = Path(os.environ["FAKE_TRAINING_ROOT"]) / "train-interrupted"
        interrupt_step = os.environ.get("FAKE_OPSD_INTERRUPT_TRAIN_STEP")
        if (
            interrupt_step
            and data.get("train/optimizer_step") == int(interrupt_step)
            and not train_marker.exists()
        ):
            train_marker.touch()
            raise KeyboardInterrupt
        marker = Path(os.environ["FAKE_TRAINING_ROOT"]) / "eval-interrupted"
        if (
            os.environ.get("FAKE_OPSD_INTERRUPT_EVAL")
            and data.get("eval/completed_tasks") == 1
            and not marker.exists()
        ):
            marker.touch()
            raise KeyboardInterrupt

    monitoring_sdk.Run.log = log  # type: ignore[method-assign]
