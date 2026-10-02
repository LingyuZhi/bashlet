"""Development sampling through a local HTTP service and the experiment seam."""

import json
import os
import subprocess
import sys
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
import yaml
from adapters import ByteEncoding
from transformers import AutoTokenizer

from bashlet import FunctionGrader, GradingCase, Task, run_experiment
from bashlet.protocol import SYSTEM_PROMPT
from bashlet.vllm import VLLMModel

type HTTPReply = tuple[int, dict[str, Any]]


def completion(tokens: list[int]) -> HTTPReply:
    return 200, {
        "id": "cmpl-test",
        "model": "test-qwen",
        "choices": [
            {
                "token_ids": tokens,
                # The Adapter must use token IDs, never this unrelated text.
                "text": "```bash\ntouch MUST_NOT_RUN\n```",
                "logprobs": {
                    "tokens": [f"token_id:{token}" for token in tokens],
                    "token_logprobs": [-0.25] * len(tokens),
                },
                "finish_reason": "stop",
            }
        ],
    }


@dataclass
class FakeVLLM:
    base_url: str = ""
    responses: deque[HTTPReply] = field(default_factory=deque)
    requests: list[dict[str, Any]] = field(default_factory=list)
    max_model_len: int = 8192
    version: str = "0.11.0"
    authorizations: list[str | None] = field(default_factory=list)


@pytest.fixture
def service() -> Iterator[FakeVLLM]:
    fake = FakeVLLM()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def respond(self, reply: HTTPReply) -> None:
            status, data = reply
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            fake.authorizations.append(self.headers.get("Authorization"))
            if self.path == "/version":
                self.respond((200, {"version": fake.version}))
            elif self.path == "/v1/models":
                self.respond(
                    (
                        200,
                        {
                            "data": [
                                {
                                    "id": "test-qwen",
                                    "root": "/models/test-qwen/fixed-revision",
                                    "max_model_len": fake.max_model_len,
                                }
                            ]
                        },
                    )
                )
            else:
                self.respond((404, {"error": {"message": "Unknown path"}}))

        def do_POST(self) -> None:
            fake.authorizations.append(self.headers.get("Authorization"))
            assert self.path == "/v1/completions"
            fake.requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            reply: HTTPReply = (
                fake.responses.popleft()
                if fake.responses
                else (500, {"error": {"message": "Unexpected request"}})
            )
            if reply[0] == 200:
                reply[1]["choices"][0].setdefault(
                    "prompt_token_ids", fake.requests[-1]["prompt"]
                )
            self.respond(reply)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.base_url = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def double_task() -> Task:
    return Task(
        task_id="dev-vllm/double",
        prompt="Write solution.py with solve(n) returning twice n, then submit.",
        initial_files={},
        grader=FunctionGrader(
            module_path="solution.py",
            function_name="solve",
            cases=(GradingCase(args=(7,), expected=14),),
        ),
    )


def model_for(service: FakeVLLM) -> VLLMModel:
    return VLLMModel(
        base_url=service.base_url,
        model="test-qwen",
        model_revision="fixed-revision",
        encoding=ByteEncoding(),
        retry_delay_seconds=0,
    )


def test_token_sampling_submits_a_development_episode_without_reencoding(
    tmp_path: Path, service: FakeVLLM
) -> None:
    # Whole-word tokens 300/301 cannot be recreated by ByteEncoding.encode.
    tokens = [
        *b"```bash\nprintf 'def ",
        300,
        *b"(n): ",
        301,
        *b" n * 2\\n' > solution.py\n",
        *b"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
        259,
    ]
    service.responses.extend([completion([65]), completion(tokens)])
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.passed, result.reason
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    call = result.trajectory[0].model_call
    assert call is not None
    assert call.generation.tokens == tuple(tokens)
    assert call.generation.logprobs == (-0.25,) * len(tokens)
    assert call.generation.request_id == "cmpl-test"
    assert call.generation.source == "sampled"
    assert result.context_tokens[call.context_offset :] == tokens
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["policy_details"]["development_only"] is True
    assert record["policy_details"]["service_version"] == "0.11.0"
    assert record["policy_details"]["model_revision"] == "fixed-revision"
    assert record["policy_details"]["capability_probe"]["passed"] is True
    assert len(service.requests) == 2
    request = service.requests[-1]
    assert request["prompt"] == result.context_tokens[: call.context_offset]
    assert request["return_token_ids"] is True
    assert request["logprobs"] == 0
    assert request["stop_token_ids"] == [259]
    assert request["stream"] is False


def test_transient_failures_in_probe_and_sampling_are_retried(
    tmp_path: Path, service: FakeVLLM
) -> None:
    service.responses.extend(
        [
            (503, {"error": {"message": "Temporarily unavailable"}}),
            completion([65]),
            (429, {"error": {"message": "Try again"}}),
            completion(
                list(
                    b"```bash\nprintf 'def solve(n): return n * 2\\n' > solution.py\n"
                    b"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
                )
            ),
        ]
    )
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.passed, result.reason
    assert result.usage.steps == 1
    assert service.requests[0] == service.requests[1]
    assert service.requests[2] == service.requests[3]


@pytest.mark.parametrize("stage", ["probe", "sampling"])
@pytest.mark.parametrize(
    "defect",
    [
        "missing_tokens",
        "missing_logprobs",
        "length_mismatch",
        "null_logprob",
        "nan_logprob",
        "positive_logprob",
        "boolean_token",
        "wrong_logprob_tokens",
        "wrong_prompt_tokens",
        "wrong_model",
    ],
)
def test_invalid_token_evidence_never_enters_the_episode(
    tmp_path: Path, service: FakeVLLM, stage: str, defect: str
) -> None:
    status, response = completion([65])
    choice = response["choices"][0]
    if defect == "missing_tokens":
        del choice["token_ids"]
    elif defect == "missing_logprobs":
        choice["logprobs"] = None
    elif defect == "length_mismatch":
        choice["token_ids"] = [65, 66]
    elif defect == "null_logprob":
        choice["logprobs"]["token_logprobs"] = [None]
    elif defect == "nan_logprob":
        choice["logprobs"]["token_logprobs"] = [float("nan")]
    elif defect == "positive_logprob":
        choice["logprobs"]["token_logprobs"] = [0.25]
    elif defect == "boolean_token":
        choice["token_ids"] = [True]
    elif defect == "wrong_logprob_tokens":
        choice["logprobs"]["tokens"] = ["token_id:66"]
    elif defect == "wrong_prompt_tokens":
        choice["prompt_token_ids"] = [17]
    elif defect == "wrong_model":
        response["model"] = "different-weights"
    if stage == "sampling":
        service.responses.append(completion([65]))
    service.responses.append((status, response))
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == result.episode_status == "model_error", result.reason
    assert result.trajectory == []
    assert result.usage.steps == 0
    assert result.grade is None
    assert list(result.workspace.iterdir()) == []
    assert len(service.requests) == (1 if stage == "probe" else 2)
    if stage == "probe":
        assert result.context_tokens == []
        assert result.usage.generated_tokens == 0
    else:
        assert result.context_tokens[-1] == 258
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["policy_details"]["development_only"] is True
    assert record["policy_details"]["capability_probe"]["passed"] == (
        stage == "sampling"
    )


@pytest.mark.parametrize("stage", ["probe", "sampling"])
def test_retry_exhaustion_is_an_infrastructure_error(
    tmp_path: Path, service: FakeVLLM, stage: str
) -> None:
    if stage == "sampling":
        service.responses.append(completion([65]))
    service.responses.extend([(503, {"error": {"message": "Unavailable"}})] * 3)
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == result.episode_status == "infrastructure_error"
    assert "after 3 attempt(s)" in result.reason
    assert result.trajectory == [] and result.grade is None
    assert len(service.requests) == (3 if stage == "probe" else 4)


@pytest.mark.parametrize("status", [400, 401])
def test_permanent_http_errors_are_not_retried_or_called_context_exhaustion(
    tmp_path: Path, service: FakeVLLM, status: int
) -> None:
    service.responses.extend(
        [
            completion([65]),
            (status, {"error": {"message": "Invalid request; maximum context length"}}),
        ]
    )
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == "infrastructure_error", result.reason
    assert result.trajectory == [] and result.grade is None
    assert len(service.requests) == 2


def test_a_different_service_version_cannot_start_an_episode(
    tmp_path: Path, service: FakeVLLM
) -> None:
    service.version = "0.10.2"
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == "infrastructure_error"
    assert "Expected vLLM 0.11.0" in result.reason
    assert result.context_tokens == [] and result.trajectory == []
    assert not service.requests


@pytest.mark.parametrize(
    "message",
    [
        "This model's maximum context length is 8192 tokens. However, your request "
        "has 8192 input tokens. Please reduce the length of the input messages.",
        "'max_tokens' or 'max_completion_tokens' is too large: 1024. This model's "
        "maximum context length is 8192 tokens and your request has 8000 input "
        "tokens (1024 > 8192 - 8000).",
    ],
)
def test_service_context_errors_end_normally_without_retry_or_grading(
    tmp_path: Path, service: FakeVLLM, message: str
) -> None:
    service.responses.extend([completion([65]), (400, {"error": {"message": message}})])
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == result.episode_status == "context_exhausted", result.reason
    assert result.reason == message
    assert result.grade is None and result.trajectory == []
    assert result.rejected_generation is None
    assert len(service.requests) == 2


def test_a_full_context_ends_before_requesting_another_generation(
    tmp_path: Path, service: FakeVLLM
) -> None:
    service.max_model_len = 2
    service.responses.append(completion([65]))
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == "context_exhausted", result.reason
    assert result.trajectory == [] and result.grade is None
    assert len(service.requests) == 1  # Only the tiny capability probe.


def test_generation_at_the_context_limit_keeps_its_tokens_and_logprobs(
    tmp_path: Path, service: FakeVLLM
) -> None:
    prompt = ByteEncoding().prompt(SYSTEM_PROMPT, double_task().prompt)
    service.max_model_len = len(prompt) + 4
    _, response = completion([65, 66, 67, 68])
    response["choices"][0]["finish_reason"] = "length"
    service.responses.extend([completion([65]), (200, response)])
    result = run_experiment(
        task=double_task(), model=model_for(service), output_dir=tmp_path
    )

    assert result.status == "context_exhausted", result.reason
    assert result.usage.generated_tokens == 4
    assert result.usage.invalid_actions == 0
    assert result.context_tokens[-4:] == [65, 66, 67, 68]
    call = result.trajectory[0].model_call
    assert call is not None and call.generation.logprobs == (-0.25,) * 4
    assert result.trajectory[0].observation is None
    assert result.grade is None
    assert service.requests[-1]["max_tokens"] == 4


def test_cli_selects_vllm_and_displays_only_development_evidence(
    tmp_path: Path, service: FakeVLLM
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        local_files_only=True,
    )
    reply = (
        "```bash\nprintf 'visible vLLM observation\\n'\n"
        "printf 'hello\\n' > result.txt\n```"
    )
    service.responses.extend(
        [
            completion([65]),
            completion(tokenizer.encode(reply, add_special_tokens=False)),
            completion(
                tokenizer.encode(
                    "```bash\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
                    add_special_tokens=False,
                )
            ),
        ]
    )
    config = tmp_path / "vllm.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "VLLMModel",
                    "base_url": service.base_url,
                    "model": "test-qwen",
                    "model_revision": "fixed-revision",
                    "tokenizer": "Qwen/Qwen3-1.7B",
                    "tokenizer_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
                    "api_key_env": "TEST_VLLM_API_KEY",
                }
            }
        )
    )
    workspace = tmp_path / "source"
    workspace.mkdir()
    output = tmp_path / "runs"
    key = "test-vllm-secret-never-record"
    environment = os.environ | {"NO_COLOR": "1", "TEST_VLLM_API_KEY": key}
    live = subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "agent",
            "Create result.txt and submit.",
            "--workdir",
            str(workspace),
            "--output-dir",
            str(output),
            "-c",
            str(config),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert live.returncode == 0, live.stdout + live.stderr
    record_path = next(output.glob("*/run.json"))
    record_text = record_path.read_text()
    record = json.loads(record_text)
    assert (record_path.parent / "workspace/result.txt").read_text() == "hello\n"
    assert not (workspace / "result.txt").exists()
    assert "Evidence: development only" in live.stdout
    assert record["policy_details"]["development_only"] is True
    assert record["encoding_details"]["revision"] == (
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    )
    assert key not in record_text + live.stdout + live.stderr
    assert service.authorizations == [f"Bearer {key}"] * 5
    replay = subprocess.run(
        [sys.executable, "-m", "bashlet", "show", str(record_path.parent)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert replay.returncode == 0, replay.stderr
    assert replay.stdout == live.stdout
