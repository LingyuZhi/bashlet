"""Sampling evidence and durable request records through the experiment seam."""

import json
import logging
import subprocess
import sys
from collections import deque
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest
from adapters import ByteEncoding
from pytrio.errors.actor import ActorAPIError
from pytrio.types.future.api_future import APIFuture

from bashlet import FunctionGrader, GradingCase, Task, run_experiment
from bashlet.backends.journal import RequestJournal
from bashlet.backends.pytrio import PyTRIOModel

MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SDK_LOG = logging.getLogger("pytrio.transport.actor.sampling_run.sample")


def response(tokens: Sequence[int], **changes: Any) -> Any:
    return SimpleNamespace(
        sequences=[
            SimpleNamespace(
                tokens=list(tokens),
                logprobs=[-0.25] * len(tokens),
                stop_reason="stop",
                text="touch MUST_NOT_RUN",
            )
        ],
        input_tokens=None,
        output_tokens=len(tokens),
        **changes,
    )


class FakeSampling:
    """Replace only the SDK boundary; bash and grading still run for real."""

    model_id = MODEL

    def __init__(self, replies: Sequence[Any]) -> None:
        self.sampling_run_cuid = f"sampling-{id(self)}"
        self.replies = deque(replies)
        self.requests: list[dict[str, Any]] = []
        self.score_requests: list[list[int]] = []
        self.probe_barrier: Barrier | None = None
        self.episode_started: Event | None = None
        self.fail_before_identity = False

    def sample(self, **request: Any) -> APIFuture[Any]:
        if self.fail_before_identity:
            raise TimeoutError("No service request identity was received")
        self.requests.append(request)
        request_id = f"remote-{self.sampling_run_cuid}-{len(self.requests)}"
        SDK_LOG.debug(
            "Sending Actor sample request task=%s request=%s actor=%s",
            self.sampling_run_cuid,
            request_id,
            "actor-test",
        )
        value = self.replies.popleft()
        if isinstance(value, Future):

            def received(completed: Future[Any]) -> None:
                result = completed.result()
                if result.input_tokens is None:
                    result.input_tokens = request["prompt"].length
                SDK_LOG.debug(
                    "Received Actor sample response task=%s request=%s "
                    "sequences=%s actor=%s",
                    self.sampling_run_cuid,
                    request_id,
                    len(result.sequences),
                    "actor-test",
                )

            value.add_done_callback(received)
            if self.episode_started is not None:
                self.episode_started.set()
            return APIFuture(value)
        future: Future[Any] = Future()
        if isinstance(value, BaseException):
            future.set_exception(value)
        else:
            if value.input_tokens is None:
                value.input_tokens = request["prompt"].length
            future.set_result(value)
            SDK_LOG.debug(
                "Received Actor sample response task=%s request=%s "
                "sequences=%s actor=%s",
                self.sampling_run_cuid,
                request_id,
                len(value.sequences),
                "actor-test",
            )
            if self.probe_barrier is not None and len(self.requests) == 1:
                self.probe_barrier.wait(timeout=20)
        return APIFuture(future)

    def compute_logprobs(self, prompt: Any) -> APIFuture[list[float | None]]:
        # SDK 0.2.9 scores via a one-token sample. Preserve its request identity.
        self.score_requests.append(prompt.to_ints())
        request_id = f"scoring-{self.sampling_run_cuid}"
        SDK_LOG.debug(
            "Sending Actor sample request task=%s request=%s actor=%s",
            self.sampling_run_cuid,
            request_id,
            "actor-test",
        )
        SDK_LOG.debug(
            "Received Actor sample response task=%s request=%s sequences=%s actor=%s",
            self.sampling_run_cuid,
            request_id,
            1,
            "actor-test",
        )
        return APIFuture.completed([None, *([-0.25] * (prompt.length - 1))])


class FakeService:
    service_session_cuid = "session-test"
    workspace_cuid = "workspace-test"

    def __init__(self, sampling: FakeSampling) -> None:
        self.sampling = sampling
        self.loaded_path: str | None = None

    def get_server_capabilities(self) -> Any:
        return SimpleNamespace(
            supported_models=[
                SimpleNamespace(
                    model_name=MODEL,
                    max_context_length=8192,
                    sampling=SimpleNamespace(
                        available=True, max_seq_len=8192, max_tokens_per_request=8192
                    ),
                )
            ]
        )

    def create_sampling_client(
        self, base_model: str, model_path: str | None = None
    ) -> FakeSampling:
        assert base_model == MODEL
        self.loaded_path = model_path
        return self.sampling


def double_task() -> Task:
    return Task(
        "pytrio/double",
        "Write solution.py with solve(n) returning twice n, then submit.",
        {},
        FunctionGrader("solution.py", "solve", (GradingCase((7,), 14),)),
    )


def model_for(
    sdk: FakeSampling, journal: RequestJournal, **options: Any
) -> PyTRIOModel:
    return PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=options.pop("encoding", ByteEncoding()),
        service_client=FakeService(sdk),
        journal=journal,
        retry_delay_seconds=0,
        **options,
    )


def test_explicit_sampler_snapshot_is_loaded_and_traced_through_the_experiment(
    tmp_path: Path,
) -> None:
    snapshot = "trio://training-test/weights/sft-step-10"
    tokens = list(b"```bash\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```")
    service = FakeService(FakeSampling([response([65]), response(tokens)]))
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    model = PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        model_path=snapshot,
        encoding=ByteEncoding(),
        service_client=service,
        journal=journal,
    )
    result = run_experiment(task=double_task(), model=model, output_dir=tmp_path)
    assert service.loaded_path == snapshot
    assert result.policy == f"pytrio/{MODEL}@{snapshot}"
    assert result.policy_details["model_path"] == snapshot
    assert result.policy_details["snapshot_kind"] == "sampler_weights"
    assert all(
        entry["evidence"]["payload"]["model_path"] == snapshot
        for entry in journal.snapshot()["entries"]
    )


def test_episode_keeps_raw_tokens_service_identity_and_probe_requests(
    tmp_path: Path,
) -> None:
    tokens = [
        *b"```bash\nprintf 'def ",
        300,
        *b"(n): ",
        301,
        *b" n * 2\\n' > solution.py\n",
        *b"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
        259,
    ]
    sdk = FakeSampling([response([65]), response(tokens)])
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    model = PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=ByteEncoding(),
        service_client=FakeService(sdk),
        journal=journal,
    )
    result = run_experiment(task=double_task(), model=model, output_dir=tmp_path)

    assert result.passed, result.reason
    assert sdk.requests[0]["prompt"].to_ints() == ByteEncoding().prompt("", "Hi")
    call = result.trajectory[0].model_call
    assert call is not None
    assert call.generation.tokens == tuple(tokens)
    assert call.generation.logprobs == (-0.25,) * len(tokens)
    assert call.generation.request_id == f"remote-{sdk.sampling_run_cuid}-2"
    assert result.context_tokens[call.context_offset :] == tokens
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    record = json.loads((result.run_dir / "run.json").read_text())
    details = record["policy_details"]
    assert details["backend"] == "pytrio"
    assert details["sdk_version"] == "0.2.9"
    assert details["capability_probe"]["passed"] is True
    entries = details["requests"]["entries"]
    assert [item["operation"] for item in entries] == [
        "capability_probe",
        "sampling",
    ]
    assert all(item["state"] == "completed" for item in entries)
    assert details["requests"]["pending_count"] == 0


@pytest.mark.parametrize(
    ("sampled_logprob", "rescored_logprob"),
    [
        (-0.3061845600605011, -0.3722420334815979),
        (-0.30621740221977234, -0.3722420334815979),
    ],
)
def test_probe_accepts_recorded_sampling_logprobs_without_student_rescoring(
    tmp_path: Path, sampled_logprob: float, rescored_logprob: float
) -> None:
    class MismatchingScores(FakeSampling):
        def compute_logprobs(self, prompt: Any) -> APIFuture[list[float | None]]:
            super().compute_logprobs(prompt)
            return APIFuture.completed(
                [None, *([rescored_logprob] * (prompt.length - 1))]
            )

    probe = response([90700])
    probe.sequences[0].logprobs = [sampled_logprob]
    tokens = list(
        b"```bash\nprintf 'def solve(n): return n * 2\\n' > solution.py\n"
        b"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    completion = response(tokens)
    old_logprobs = [-0.2 - 0.01 * (index % 7) for index in range(len(tokens))]
    completion.sequences[0].logprobs = old_logprobs
    sdk = MismatchingScores([probe, completion])
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )

    assert result.passed, result.reason
    assert sdk.score_requests == []
    assert len(sdk.requests) == 2
    capability = result.policy_details["capability_probe"]
    assert isinstance(capability, dict)
    evidence = capability["generation"]
    assert isinstance(evidence, dict)
    assert evidence["tokens"] == [90700]
    assert evidence["logprobs"] == [sampled_logprob]
    call = result.trajectory[0].model_call
    assert call is not None
    assert call.generation.tokens == tuple(tokens)
    assert call.generation.logprobs == tuple(old_logprobs)


def test_terminal_failure_and_retry_both_keep_their_request_records(
    tmp_path: Path,
) -> None:
    sdk = FakeSampling(
        [
            response([65]),
            ActorAPIError("busy", code="too_many_requests", status_code=429),
            response(list(b"```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```")),
        ]
    )
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )

    assert result.status == "completed", result.reason
    assert len(sdk.requests) == 3
    entries = journal.snapshot()["entries"]
    assert [item["state"] for item in entries] == ["completed", "failed", "completed"]
    assert len(result.trajectory) == 1


@pytest.mark.parametrize("stage", ["probe", "episode"])
@pytest.mark.parametrize(
    "defect", ["tokens", "logprobs", "length", "nan", "positive", "boolean"]
)
def test_invalid_evidence_is_recorded_but_never_executed(
    tmp_path: Path,
    stage: str,
    defect: str,
) -> None:
    invalid = response([65])
    sequence = invalid.sequences[0]
    if defect in ("tokens", "logprobs"):
        setattr(sequence, defect, None)
    elif defect == "length":
        sequence.logprobs = []
    elif defect == "nan":
        sequence.logprobs = [float("nan")]
    elif defect == "positive":
        sequence.logprobs = [0.25]
    elif defect == "boolean":
        sequence.tokens = [True]
    sdk = FakeSampling([invalid] if stage == "probe" else [response([65]), invalid])
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )

    assert result.status == "model_error", result.reason
    assert result.grade is None and not result.trajectory
    assert list(result.workspace.iterdir()) == []
    assert len(journal.snapshot()["entries"]) == (1 if stage == "probe" else 2)
    assert journal.snapshot()["pending_count"] == 0


def test_cli_inspects_and_resolves_a_timed_out_request_without_resubmission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.sqlite3"
    sdk = FakeSampling([response([65]), TimeoutError("remote outcome unknown")])
    journal = RequestJournal(path)
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )
    assert result.status == "infrastructure_error"

    status = subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "requests",
            "status",
            "--journal",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert status.returncode == 0, status.stderr
    data = json.loads(status.stdout)
    assert data["pending_count"] == 1
    pending = data["entries"][-1]
    assert pending["evidence"]["request_id"] == f"remote-{sdk.sampling_run_cuid}-2"
    evidence_path = tmp_path / "outcome.json"
    evidence_path.write_text(
        json.dumps(
            {
                "request_id": pending["evidence"]["request_id"],
                "terminal_status": "failed",
                "source": "controlled remote completion record",
            }
        )
    )
    reconciled = subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "requests",
            "resolve",
            pending["id"],
            "--journal",
            str(path),
            "--outcome",
            "failed",
            "--evidence",
            str(evidence_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert reconciled.returncode == 0, reconciled.stderr
    confirmed = json.loads(reconciled.stdout)
    assert confirmed["pending_count"] == 0
    assert confirmed["entries"][-1]["state"] == "failed"
    assert (
        confirmed["entries"][-1]["evidence"]["resolution"]["terminal_status"]
        == "failed"
    )
    assert len(sdk.requests) == 2


def test_restart_preserves_pending_requests_and_refuses_duplicate_work(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.sqlite3"
    original = FakeSampling([response([65]), TimeoutError("unknown outcome")])
    first = run_experiment(
        task=double_task(),
        model=model_for(original, RequestJournal(path)),
        output_dir=tmp_path,
    )
    assert first.status == "infrastructure_error"
    resumed = FakeSampling([response([65])])
    journal = RequestJournal(path)
    second = run_experiment(
        task=double_task(), model=model_for(resumed, journal), output_dir=tmp_path
    )

    assert second.status == "infrastructure_error", second.reason
    assert "still pending" in second.reason
    assert len(resumed.requests) == 1
    assert journal.snapshot()["pending_count"] == 1


def test_concurrent_episodes_cannot_duplicate_an_in_flight_request(
    tmp_path: Path,
) -> None:
    path = tmp_path / "requests.sqlite3"
    suspended: Future[Any] = Future()
    started = Event()
    barrier = Barrier(2)
    services = [FakeSampling([response([65]), suspended]) for _ in range(2)]
    for sdk in services:
        sdk.probe_barrier, sdk.episode_started = barrier, started

    def episode(index: int) -> Any:
        class ProbeEncoding(ByteEncoding):
            def prompt(self, system: str, user: str) -> list[int]:
                return super().prompt(
                    system, user + str(index) if user == "Hi" else user
                )

        return run_experiment(
            task=double_task(),
            model=model_for(
                services[index],
                RequestJournal(path),
                timeout_seconds=30,
                encoding=ProbeEncoding(),
            ),
            output_dir=tmp_path,
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(episode, index) for index in range(2)]
        try:
            assert started.wait(timeout=20)
            done, pending = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
            assert len(done) == len(pending) == 1
            denied = next(iter(done)).result()
            assert denied.status == "infrastructure_error", denied.reason
            assert "still pending" in denied.reason
            journal = RequestJournal(path)
            assert journal.snapshot()["pending_count"] == 1
        finally:
            suspended.set_result(
                response(
                    list(
                        b"```bash\nprintf 'def solve(n): return n * 2\\n' "
                        b"> solution.py\n"
                        b"printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
                    )
                )
            )
        results = [future.result(timeout=20) for future in futures]
    assert sorted(result.passed for result in results) == [False, True]
    assert sum(len(sdk.requests) for sdk in services) == 3
    assert RequestJournal(path).snapshot()["pending_count"] == 0


def test_account_permission_error_does_not_expose_sdk_details(tmp_path: Path) -> None:
    secret = "trio_FAKESENSITIVEKEY"

    class DeniedService(FakeService):
        def get_server_capabilities(self) -> Any:
            raise ActorAPIError(secret, code="unauthorized", status_code=401)

    sdk = FakeSampling([])
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    model = PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=ByteEncoding(),
        service_client=DeniedService(sdk),
        journal=journal,
    )
    result = run_experiment(task=double_task(), model=model, output_dir=tmp_path)
    assert result.status == "infrastructure_error"
    assert "unauthorized" in result.reason and "401" in result.reason
    assert secret not in (result.run_dir / "run.json").read_text()
    assert not sdk.requests and journal.snapshot()["entries"] == []


def test_another_workspace_cannot_reuse_a_request_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    first = FakeSampling([response([65])])
    model_for(first, journal).probe()
    before = journal.snapshot()["entries"]
    monkeypatch.setattr(FakeService, "workspace_cuid", "another-workspace")
    second = FakeSampling([])
    result = run_experiment(
        task=double_task(), model=model_for(second, journal), output_dir=tmp_path
    )
    assert result.status == "infrastructure_error"
    assert "another PyTRIO workspace" in result.reason
    assert not second.requests
    assert journal.snapshot()["entries"] == before


def test_gateway_failure_keeps_its_pending_request_and_is_not_retried(
    tmp_path: Path,
) -> None:
    sdk = FakeSampling(
        [
            response([65]),
            ActorAPIError("unknown outcome", code="unavailable", status_code=503),
        ]
    )
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )
    assert result.status == "infrastructure_error", result.reason
    assert len(sdk.requests) == 2
    assert journal.snapshot()["pending_count"] == 1


@pytest.mark.parametrize("stage", ["probe", "episode"])
def test_sdk_invalid_response_maps_to_model_error_and_retains_requests(
    tmp_path: Path, stage: str
) -> None:
    invalid = ActorAPIError(
        "missing probability evidence", code="actor.invalid_response"
    )
    sdk = FakeSampling([invalid] if stage == "probe" else [response([65]), invalid])
    journal = RequestJournal(tmp_path / "requests.sqlite3")
    result = run_experiment(
        task=double_task(), model=model_for(sdk, journal), output_dir=tmp_path
    )
    assert result.status == "model_error", result.reason
    assert not result.trajectory and result.grade is None
    assert len(sdk.requests) == (1 if stage == "probe" else 2)
    assert journal.snapshot()["pending_count"] == 1
