"""Paid sampling evidence and cash limits through the experiment seam."""

import json
import logging
import subprocess
import sys
from collections import deque
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest
from adapters import ByteEncoding, ScriptedModel, bash
from pytrio.errors.actor import ActorAPIError
from pytrio.types.future.api_future import APIFuture

from bashlet import FunctionGrader, GradingCase, Task, run_experiment
from bashlet.pytrio import PyTRIOModel
from bashlet.spend import SpendLedger, SpendPolicy

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
        self.score_barrier: Barrier | None = None
        self.episode_started: Event | None = None

    def sample(self, **request: Any) -> APIFuture[Any]:
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
        return APIFuture(future)

    def compute_logprobs(self, prompt: Any) -> APIFuture[list[float | None]]:
        # SDK 0.2.9 scores via a one-token sample. Preserve its request identity.
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
        if self.score_barrier is not None:
            self.score_barrier.wait(timeout=20)
        return APIFuture.completed([None, *([-0.25] * (prompt.length - 1))])


class FakeService:
    service_session_cuid = "session-test"
    workspace_cuid = "workspace-test"

    def __init__(self, sampling: FakeSampling) -> None:
        self.sampling = sampling

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

    def create_sampling_client(self, base_model: str) -> FakeSampling:
        assert base_model == MODEL
        return self.sampling


def double_task() -> Task:
    return Task(
        "pytrio/double",
        "Write solution.py with solve(n) returning twice n, then submit.",
        {},
        FunctionGrader("solution.py", "solve", (GradingCase((7,), 14),)),
    )


def model_for(sdk: FakeSampling, ledger: SpendLedger, **options: Any) -> PyTRIOModel:
    return PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=options.pop("encoding", ByteEncoding()),
        service_client=FakeService(sdk),
        ledger=ledger,
        retry_delay_seconds=0,
        **options,
    )


def test_paid_episode_keeps_raw_tokens_service_identity_and_probe_spend(
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
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=10))
    model = PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=ByteEncoding(),
        service_client=FakeService(sdk),
        ledger=ledger,
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
    entries = details["spend"]["entries"]
    assert [item["operation"] for item in entries] == [
        "capability_probe",
        "logprob_probe",
        "sampling",
    ]
    assert all(item["charged_cny"] > 0 for item in entries)
    assert all(item["billing_basis"] == "upper_bound" for item in entries)
    assert details["spend"]["pending_cny"] == 0


def test_exhausted_cash_budget_stops_before_an_episode_request(tmp_path: Path) -> None:
    sdk = FakeSampling([response([65])])
    ledger = SpendLedger(
        tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=0.08)
    )
    result = run_experiment(
        task=double_task(), model=model_for(sdk, ledger), output_dir=tmp_path
    )

    assert result.status == result.episode_status == "budget_exhausted", result.reason
    assert "funded" in result.reason
    assert not result.trajectory and result.grade is None
    assert len(sdk.requests) == 1  # Only the one-token sampling probe.
    # The live price catalog stores fen and rounds upward. Reserve each phase
    # to one fen, then apply the configured 2x ceiling: two probes cost <= ¥0.08.
    assert ledger.snapshot()["charged_cny"] == 0.08
    assert ledger.snapshot()["pending_cny"] == 0


def test_billed_failure_and_retry_both_consume_the_available_cash(
    tmp_path: Path,
) -> None:
    sdk = FakeSampling(
        [
            response([65]),
            ActorAPIError("busy", code="too_many_requests", status_code=429),
            response([65]),
        ]
    )
    ledger = SpendLedger(
        tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=0.12)
    )
    result = run_experiment(
        task=double_task(), model=model_for(sdk, ledger), output_dir=tmp_path
    )

    assert result.status == "budget_exhausted", result.reason
    assert len(sdk.requests) == 2  # The retry was refused before reaching the SDK.
    entries = ledger.snapshot()["entries"]
    assert [item["state"] for item in entries] == ["completed", "completed", "failed"]
    assert ledger.snapshot()["charged_cny"] == 0.12
    assert not result.trajectory and result.grade is None


@pytest.mark.parametrize("stage", ["probe", "episode"])
@pytest.mark.parametrize(
    "defect", ["tokens", "logprobs", "length", "nan", "positive", "boolean"]
)
def test_invalid_paid_evidence_is_charged_but_never_executed(
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
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=10))
    result = run_experiment(
        task=double_task(), model=model_for(sdk, ledger), output_dir=tmp_path
    )

    assert result.status == "model_error", result.reason
    assert result.grade is None and not result.trajectory
    assert list(result.workspace.iterdir()) == []
    assert ledger.snapshot()["charged_cny"] == (0.04 if stage == "probe" else 0.12)
    assert ledger.snapshot()["pending_cny"] == 0


@pytest.mark.parametrize("missing", ["prefill", "billing_quantum_cny"])
def test_unknown_prices_refuse_even_the_first_paid_probe(
    tmp_path: Path, missing: str
) -> None:
    import yaml

    from bashlet.spend import DEFAULT_POLICY

    data = yaml.safe_load(DEFAULT_POLICY.read_text())
    if missing == "prefill":
        data["prices_cny_per_million"][missing] = None
    else:
        data[missing] = None
    policy_file = tmp_path / "unknown-prices.yaml"
    policy_file.write_text(yaml.safe_dump(data))
    sdk = FakeSampling([response([65])])
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=10))
    result = run_experiment(
        task=double_task(),
        model=model_for(sdk, ledger, spend_config=str(policy_file)),
        output_dir=tmp_path,
    )

    assert result.status == "budget_exhausted", result.reason
    assert "pricing" in result.reason or "rounding" in result.reason
    assert not sdk.requests and not result.trajectory
    assert ledger.snapshot()["entries"] == []


def test_all_paid_categories_share_the_cumulative_opsd_pilot_cap(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=20))

    class PaidWorkflow(ScriptedModel):
        def sample(self, prompt: Sequence[int], sampling: Any) -> Any:
            for category in (
                "sampling",
                "teacher_scoring",
                "training",
                "demonstration",
                "evaluation",
                "training",
            ):
                ledger.execute(
                    operation=category,
                    group="opsd-pilot",
                    payload={"tokens": list(prompt)},
                    upper_cny=2,
                    timeout=1,
                    submit=lambda request_id: APIFuture.completed(None),
                )
            return super().sample(prompt, sampling)

    result = run_experiment(
        task=double_task(),
        model=PaidWorkflow(lambda context: bash("touch MUST_NOT_RUN")),
        output_dir=tmp_path,
    )
    assert result.status == "budget_exhausted", result.reason
    assert "opsd-pilot" in result.reason
    assert ledger.snapshot()["charged_cny"] == 10
    assert [item["operation"] for item in ledger.snapshot()["entries"]] == [
        "sampling",
        "teacher_scoring",
        "training",
        "demonstration",
        "evaluation",
    ]
    assert not (result.workspace / "MUST_NOT_RUN").exists()


def test_cli_inspects_and_reconciles_a_timed_out_request_without_resubmission(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.sqlite3"
    sdk = FakeSampling([response([65]), TimeoutError("remote outcome unknown")])
    ledger = SpendLedger(path, policy=SpendPolicy(funded_cny=10))
    result = run_experiment(
        task=double_task(), model=model_for(sdk, ledger), output_dir=tmp_path
    )
    assert result.status == "infrastructure_error"

    status = subprocess.run(
        [sys.executable, "-m", "bashlet", "spend", "status", "--ledger", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert status.returncode == 0, status.stderr
    data = json.loads(status.stdout)
    assert data["pending_cny"] == 0.04 and data["charged_cny"] == 0.08
    pending = data["entries"][-1]
    assert pending["evidence"]["request_id"] == f"remote-{sdk.sampling_run_cuid}-2"
    bill = tmp_path / "bill.json"
    bill.write_text(
        json.dumps(
            {
                "request_id": pending["evidence"]["request_id"],
                "terminal_status": "failed",
                "source": "controlled billing record",
            }
        )
    )
    reconciled = subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "spend",
            "reconcile",
            pending["id"],
            "--ledger",
            str(path),
            "--actual-cny",
            "0.02",
            "--outcome",
            "failed",
            "--evidence",
            str(bill),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert reconciled.returncode == 0, reconciled.stderr
    confirmed = json.loads(reconciled.stdout)
    assert confirmed["pending_cny"] == 0 and confirmed["charged_cny"] == 0.10
    assert confirmed["entries"][-1]["billing_basis"] == "confirmed"
    assert len(sdk.requests) == 2


def test_restart_preserves_pending_cost_and_refuses_duplicate_work(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.sqlite3"
    policy = SpendPolicy(funded_cny=10)
    original = FakeSampling([response([65]), TimeoutError("unknown outcome")])
    first = run_experiment(
        task=double_task(),
        model=model_for(original, SpendLedger(path, policy=policy)),
        output_dir=tmp_path,
    )
    assert first.status == "infrastructure_error"
    resumed = FakeSampling([response([65])])
    ledger = SpendLedger(path, policy=policy)
    second = run_experiment(
        task=double_task(), model=model_for(resumed, ledger), output_dir=tmp_path
    )

    assert second.status == "budget_exhausted", second.reason
    assert "still pending" in second.reason
    assert len(resumed.requests) == 1
    assert ledger.snapshot()["pending_cny"] == 0.04
    assert ledger.snapshot()["charged_cny"] == 0.16
    assert ledger.snapshot()["funded_remaining_cny"] == 9.80


def test_concurrent_episodes_cannot_spend_an_in_flight_reservation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "spend.sqlite3"
    policy = SpendPolicy(funded_cny=0.20)
    suspended: Future[Any] = Future()
    started = Event()
    barrier = Barrier(2)
    services = [FakeSampling([response([65]), suspended]) for _ in range(2)]
    for sdk in services:
        sdk.score_barrier, sdk.episode_started = barrier, started

    def episode(index: int) -> Any:
        class ProbeEncoding(ByteEncoding):
            def prompt(self, system: str, user: str) -> list[int]:
                return super().prompt(
                    system, user + str(index) if user == "Hi" else user
                )

        return run_experiment(
            task=replace(
                double_task(), prompt=double_task().prompt + f" Attempt {index}."
            ),
            model=model_for(
                services[index],
                SpendLedger(path, policy=policy),
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
            assert denied.status == "budget_exhausted", denied.reason
            ledger = SpendLedger(path, policy=policy)
            assert ledger.snapshot()["charged_cny"] == 0.16
            assert ledger.snapshot()["pending_cny"] == 0.04
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
    assert SpendLedger(path, policy=policy).snapshot()["charged_cny"] == 0.20


def test_capability_probes_also_consume_the_pilot_sub_budget(tmp_path: Path) -> None:
    ledger = SpendLedger(
        tmp_path / "spend.sqlite3",
        policy=SpendPolicy(
            funded_cny=20, groups={"opsd-pilot": 0.04, "pytrio-probe": 0.20}
        ),
    )
    sdk = FakeSampling([response([65])])
    result = run_experiment(
        task=double_task(),
        model=model_for(sdk, ledger, group="opsd-pilot"),
        output_dir=tmp_path,
    )
    assert result.status == "budget_exhausted", result.reason
    assert "opsd-pilot" in result.reason
    assert ledger.snapshot()["charged_cny"] == 0.04
    assert all(entry["group"] == "opsd-pilot" for entry in ledger.snapshot()["entries"])


def test_unknown_spend_group_is_refused_before_any_paid_request(tmp_path: Path) -> None:
    sdk = FakeSampling([response([65])])
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy())
    result = run_experiment(
        task=double_task(),
        model=model_for(sdk, ledger, group="opsd-pilto"),
        output_dir=tmp_path,
    )
    assert result.status == "budget_exhausted", result.reason
    assert "Unknown sub-budget" in result.reason
    assert not sdk.requests and ledger.snapshot()["entries"] == []


def test_account_permission_error_does_not_expose_sdk_details(tmp_path: Path) -> None:
    secret = "trio_FAKESENSITIVEKEY"

    class DeniedService(FakeService):
        def get_server_capabilities(self) -> Any:
            raise ActorAPIError(secret, code="unauthorized", status_code=401)

    sdk = FakeSampling([])
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy())
    model = PyTRIOModel(
        model=MODEL,
        model_revision=REVISION,
        encoding=ByteEncoding(),
        service_client=DeniedService(sdk),
        ledger=ledger,
    )
    result = run_experiment(task=double_task(), model=model, output_dir=tmp_path)
    assert result.status == "infrastructure_error"
    assert "unauthorized" in result.reason and "401" in result.reason
    assert secret not in (result.run_dir / "run.json").read_text()
    assert not sdk.requests and ledger.snapshot()["entries"] == []


def test_gateway_failure_keeps_its_reservation_and_is_not_retried(
    tmp_path: Path,
) -> None:
    sdk = FakeSampling(
        [
            response([65]),
            ActorAPIError("unknown outcome", code="unavailable", status_code=503),
        ]
    )
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy())
    result = run_experiment(
        task=double_task(), model=model_for(sdk, ledger), output_dir=tmp_path
    )
    assert result.status == "infrastructure_error", result.reason
    assert len(sdk.requests) == 2
    assert ledger.snapshot()["pending_cny"] == 0.04
    assert ledger.snapshot()["charged_cny"] == 0.08
