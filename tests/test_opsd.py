"""OPSD behavior through run_experiment, with only model services replaced."""

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any

import monitoring_sdk
import pytest
from adapters import ByteEncoding, ScriptedModel, bash, submit, turn_number, write_file

from bashlet import (
    FunctionGrader,
    GradingCase,
    SpendLedger,
    SpendPolicy,
    Task,
    run_experiment,
)
from bashlet.distillation import (
    OPSDConfig,
    OPSDRun,
    SFTSnapshot,
    TeacherScores,
    UpdateResult,
)
from bashlet.models import EpisodeResult, Generation, Model, TaskOrigin
from bashlet.monitoring import MonitorConfig


class Future[T]:
    def __init__(self, complete: Callable[[], T]) -> None:
        self.complete = complete

    def result(self, timeout: float | None = None) -> T:
        return self.complete()


class ControlledOPSD:
    """Declared test SFT, token scorer, and completed optimizer receipt."""

    teacher_identity = "test/sft@selected"
    teacher_encoding = ByteEncoding()

    def __init__(self) -> None:
        self.student = ScriptedModel(self.respond, identity=self.teacher_identity)
        self.scoring: list[tuple[int, ...]] = []
        self.updates: list[dict[str, Any]] = []
        self.refreshed: list[int] = []
        self.completed = 0
        self.restored: list[UpdateResult] = []
        self.update_completion: Callable[[], None] = lambda: None
        self.score_response: Callable[[TeacherScores], TeacherScores] = lambda value: (
            value
        )

    def respond(self, context: str) -> Generation:
        assert "TEACHER_ONLY_REFERENCE" not in context
        turn = turn_number(context)
        reply = {
            1: write_file("def double(n): return n + 2\n"),
            2: bash("python3 feedback.py"),
            3: write_file("def double(n): return n * 2\n"),
            4: bash("python3 feedback.py"),
            5: bash("true"),
            6: submit(),
        }[turn]
        if turn == 3:
            assert "AssertionError" in context
        from adapters import sample_text

        raw = sample_text(reply, 1024)
        return replace(raw, logprobs=(-1.0,) * len(raw.tokens))

    def score(self, tokens: Sequence[int]) -> Future[TeacherScores]:
        self.scoring.append(tuple(tokens))
        assert "TEACHER_ONLY_REFERENCE" in self.teacher_encoding.render(tokens)
        values: list[float | None] = [-0.25] * len(tokens)
        values[0], values[-1] = None, -1.75
        return Future(
            lambda: self.score_response(
                TeacherScores(
                    tuple(tokens),
                    tuple(values),
                    self.teacher_identity,
                    self.teacher_encoding.identity,
                    f"score-{len(self.scoring)}",
                )
            )
        )

    def update(self, batch: list[Any], **kwargs: Any) -> Future[UpdateResult]:
        self.updates.append({"batch": batch, **kwargs})

        def complete() -> UpdateResult:
            self.update_completion()
            self.completed += 1
            return UpdateResult(
                kwargs["optimizer_step"],
                0.125,
                f"test/student@opsd-{self.completed}",
                f"test/training-state@opsd-{self.completed}",
            )

        return Future(complete)

    def refresh_student(self, update: UpdateResult) -> Model:
        assert self.completed == update.optimizer_step
        self.refreshed.append(update.optimizer_step)
        self.student = ScriptedModel(self.respond, identity=update.student_policy)
        return self.student

    def restore_student(self, update: UpdateResult) -> Model:
        self.restored.append(update)
        self.completed = update.optimizer_step
        return self.refresh_student(update)


def repair_task() -> Task:
    return Task(
        task_id="opsd/double",
        prompt="Implement double(n) in solution.py.",
        initial_files={
            "feedback.py": "from solution import double\nassert double(3) == 6\n",
        },
        grader=FunctionGrader(
            "solution.py",
            "double",
            (GradingCase((7,), 14), GradingCase((-2,), -4)),
        ),
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback.py",
        reference_implementation=(
            "# TEACHER_ONLY_REFERENCE\n"
            + "# Longer privileged context\n" * 40
            + "def double(n): return 2 * n\n"
        ),
        origin=TaskOrigin(
            "google-research/mbpp",
            "test-revision",
            "a" * 64,
            "601",
            "mbpp/601",
            "train",
        ),
    )


def controlled_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    adapter: ControlledOPSD | None = None,
    *,
    funded_cny: float = 10,
) -> tuple[OPSDRun, ControlledOPSD]:
    import sys

    monkeypatch.setitem(sys.modules, "wandb", monitoring_sdk)
    monkeypatch.setenv("WANDB_API_KEY", "secret-opsd-key")
    adapter = adapter or ControlledOPSD()
    ledger = SpendLedger(
        tmp_path / "spend.sqlite3", policy=SpendPolicy(funded_cny=funded_cny)
    )
    run = OPSDRun(
        tmp_path / "training",
        snapshot=SFTSnapshot(
            "test/sft@selected", ByteEncoding().identity, "test-double"
        ),
        adapter=adapter,
        ledger=ledger,
        config=OPSDConfig.read(
            Path(__file__).parents[1] / "configs/opsd-contract-v1.yaml"
        ),
        monitoring=MonitorConfig(group="controlled-opsd"),
    )
    return run, adapter


def test_real_repair_scores_original_tokens_and_submits_shifted_sampled_token_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)
    student = run.student
    result = run_experiment(
        task=repair_task(),
        model=student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed and result.repair_attempts[0].passed
    assert result.trajectory[4].observation is not None
    assert result.trajectory[4].observation.output == ""
    assert len(adapter.updates) == 1
    request = adapter.updates[0]
    assert request["loss_fn"] == "importance_sampling"
    assert request["auto_shift"] is False
    datum = request["batch"][0]
    assert datum.model_input.to_ints() == result.context_tokens[:-1]
    targets = datum.loss_fn_inputs["target_tokens"].tolist()
    old = datum.loss_fn_inputs["logprobs"].tolist()
    advantages = datum.loss_fn_inputs["advantages"].tolist()
    supervised: set[int] = set()
    for step, teacher_input in zip(result.trajectory, adapter.scoring, strict=True):
        call = step.model_call
        assert call is not None
        raw = list(call.generation.tokens)
        start = call.context_offset - 1
        end = start + len(raw)
        supervised.update(range(start, end))
        assert targets[start:end] == raw
        assert old[start:end] == [-1.0] * len(raw)
        assert advantages[start:end] == [0.375] * (len(raw) - 1) + [-0.375]
        assert list(teacher_input[-len(raw) :]) == raw
        assert len(teacher_input) > call.context_offset + len(raw)
    assert 301 in targets  # Whole-word token is lost by decode/re-encode.
    assert all(
        targets[i] == old[i] == advantages[i] == 0
        for i in range(len(targets))
        if i not in supervised
    )
    assert run.student.identity == "test/student@opsd-1"
    assert adapter.teacher_identity == "test/sft@selected"
    assert adapter.refreshed == [1]
    assert result.opsd is not None and result.opsd["status"] == "completed"


def test_frozen_teacher_drift_after_scoring_is_detected_before_updating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)

    def drift(scores: TeacherScores) -> TeacherScores:
        if len(adapter.scoring) == 6:
            adapter.teacher_identity = "test/student@opsd-1"
        return scores

    adapter.score_response = drift
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None and result.opsd["status"] == "rejected"
    assert "teacher" in str(result.opsd["reason"]).lower()
    assert not adapter.updates and not adapter.refreshed


def test_update_metrics_use_supervised_denominator_and_keep_unknown_bills_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)
    original_student = adapter.student
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None
    count = result.usage.generated_tokens
    expected = -0.75 + 9.0 / count  # Six terminal tokens each add 1.5.
    assert result.opsd["supervised_tokens"] == count
    assert result.opsd["sampled_token_reverse_kl"] == pytest.approx(expected)
    remote = json.loads((run.root / "remote.json").read_text())
    row = remote["rows"][-1]
    assert row["train/optimizer_step"] == 1
    assert row["train/loss"] == 0.125
    assert row["train/distillation_coefficient"] == 0.5
    assert row["train/supervised_tokens"] == count
    assert row["train/sampled_token_reverse_kl"] == pytest.approx(expected)
    assert row["train/sampled_token_reverse_kl"] < 0
    assert row["train/rollout_input_tokens"] == sum(
        len(prompt) for prompt in original_student.prompts
    )
    assert row["train/generated_tokens"] == count
    assert row["train/teacher_input_tokens"] == sum(map(len, adapter.scoring))
    assert row["train/training_input_tokens"] == len(result.context_tokens) - 1
    assert row["spend/run_actual_known"] is False
    assert "spend/run_actual_cny" not in row
    assert row["spend/run_upper_cny"] > 0
    assert remote["config"]["evidence_kind"] == "test-double"
    assert not any(
        value in json.dumps(remote)
        for value in (
            "secret-opsd-key",
            "TEACHER_ONLY_REFERENCE",
            "feedback.py",
            "context_tokens",
        )
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "nan",
        "infinite",
        "positive",
        "overflow",
        "length",
        "tokens",
        "encoding",
        "base-teacher",
        "latest-student",
    ],
)
def test_invalid_teacher_supervision_keeps_paid_calls_and_never_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    adapter = ControlledOPSD()

    def corrupt(scores: TeacherScores) -> TeacherScores:
        bad_values: dict[str, float | None] = {
            "missing": None,
            "nan": float("nan"),
            "infinite": float("inf"),
            "positive": 0.1,
            "overflow": -1e100,
        }
        if problem in bad_values:
            return replace(
                scores, logprobs=(*scores.logprobs[:-1], bad_values[problem])
            )
        if problem == "length":
            return replace(scores, logprobs=scores.logprobs[:-1])
        if problem == "tokens":
            return replace(scores, tokens=(*scores.tokens[:-1], 7))
        if problem == "encoding":
            return replace(scores, encoding="incompatible/tokenizer")
        return replace(
            scores,
            policy=(
                "test/base-model"
                if problem == "base-teacher"
                else "test/student@opsd-1"
            ),
        )

    adapter.score_response = corrupt
    run, _ = controlled_run(tmp_path, monkeypatch, adapter)
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed  # The independent grade cannot repair bad probabilities.
    assert result.opsd is not None and result.opsd["status"] == "rejected"
    assert result.opsd["reason"]
    assert not adapter.updates and not adapter.refreshed
    assert (result.run_dir / "private/teacher-0000.json").is_file()
    saved = json.loads((result.run_dir / "run.json").read_text())
    assert len(saved["trajectory"]) == 6
    ledger = run.ledger.snapshot()
    assert any(entry["operation"] == "teacher_scoring" for entry in ledger["entries"])
    assert ledger["charged_cny"] > 0
    metrics = json.loads((run.root / "remote.json").read_text())["rows"][-1]
    assert metrics["train/optimizer_step"] == 0
    assert metrics["train/rejected_batches"] == 1
    assert metrics["spend/run_actual_known"] is False
    assert "train/loss" not in metrics


def test_monitor_disconnect_and_training_state_resume_preserve_progress_and_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from test_cli import cli
    from test_monitoring import external_sdks

    external_sdks(tmp_path, monkeypatch)
    run, _ = controlled_run(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_WANDB_LOG_FAIL", "1")
    first = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert first.opsd is not None and first.opsd["status"] == "completed"
    before = json.loads((run.root / "monitor.json").read_text())
    original_journal = (run.root / "metrics.jsonl").read_bytes()
    assert before["sync_status"] == "pending"
    assert not json.loads((run.root / "remote.json").read_text())["rows"]
    monkeypatch.delenv("FAKE_WANDB_LOG_FAIL")
    adapter = ControlledOPSD()
    resumed = OPSDRun(
        run.root,
        snapshot=run.snapshot,
        adapter=adapter,
        ledger=run.ledger,
        config=run.config,
        monitoring=run.monitoring,
        resume=True,
    )
    assert adapter.restored == [
        UpdateResult(
            1,
            0.125,
            "test/student@opsd-1",
            "test/training-state@opsd-1",
        )
    ]
    second = run_experiment(
        task=repair_task(),
        model=resumed.student,
        opsd_run=resumed,
        output_dir=tmp_path / "episodes",
    )
    assert second.policy == "test/student@opsd-1"
    assert second.opsd is not None and second.opsd["status"] == "completed"
    after = json.loads((run.root / "monitor.json").read_text())
    assert after["id"] == before["id"]
    journal = (run.root / "metrics.jsonl").read_bytes()
    assert journal.startswith(original_journal)
    for _ in range(2):
        process = cli("monitor", "sync", str(run.root), cwd=tmp_path)
        assert process.returncode == 0, process.stdout + process.stderr
    assert (run.root / "metrics.jsonl").read_bytes() == journal
    remote = json.loads((run.root / "remote.json").read_text())
    assert [row["train/optimizer_step"] for row in remote["rows"]] == [1, 2]
    record = json.loads((run.root / "opsd.json").read_text())
    assert record["optimizer_step"] == 2
    assert record["student_policy"] == "test/student@opsd-2"
    assert record["teacher_policy"] == "test/sft@selected"
    assert record["batches"][-1]["cost"]["charged_upper_bound_cny"] > 1


def test_rejected_remote_update_keeps_its_charge_and_allows_a_safe_new_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bashlet.distillation import UpdateRejected

    run, adapter = controlled_run(tmp_path, monkeypatch)

    def reject() -> None:
        raise UpdateRejected("transient")

    adapter.update_completion = reject
    rejected = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert rejected.opsd is not None and rejected.opsd["status"] == "rejected"
    assert adapter.completed == 0 and not adapter.refreshed
    first_entries = run.ledger.snapshot()["entries"]
    assert first_entries[-1]["operation"] == "opsd_update"
    assert first_entries[-1]["state"] == "failed"
    first_journal = (run.root / "metrics.jsonl").read_bytes()
    adapter.update_completion = lambda: None
    retried = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert retried.opsd is not None and retried.opsd["status"] == "completed"
    assert (run.root / "metrics.jsonl").read_bytes().startswith(first_journal)
    metrics = json.loads((run.root / "remote.json").read_text())["rows"]
    assert [row["train/optimizer_step"] for row in metrics] == [0, 1]
    assert metrics[-1]["train/rejected_batches"] == 1
    assert metrics[-1]["train/valid_batches"] == 1
    assert metrics[-1]["spend/run_upper_cny"] > metrics[0]["spend/run_upper_cny"]
    assert metrics[0]["run/reason"] == "remote_update_rejected:transient"


@pytest.mark.parametrize(
    "problem",
    [
        "prefix",
        "rollout-probability",
        "mixed-policy",
        "replayed",
        "missing-logprobs",
        "nan-logprobs",
        "overflow-logprobs",
    ],
)
def test_invalid_student_evidence_cannot_become_an_on_policy_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)

    def respond(context: str) -> Generation:
        raw = adapter.respond(context)
        if problem == "replayed":
            return replace(raw, source="replayed", logprobs=None)
        if problem == "missing-logprobs":
            return replace(raw, logprobs=None)
        if problem == "nan-logprobs":
            return replace(raw, logprobs=(float("nan"),) * len(raw.tokens))
        if problem == "overflow-logprobs":
            return replace(raw, logprobs=(-1e100,) * len(raw.tokens))
        return raw

    adapter.student.respond = respond

    def corrupt(result: EpisodeResult) -> None:
        if result.status != "running" or len(result.trajectory) != 1:
            return
        if problem == "prefix":
            result.context_tokens[0] = 17
        elif problem in ("rollout-probability", "mixed-policy"):
            call = result.trajectory[0].model_call
            assert call is not None
            changed = (
                replace(call, policy="test/other-policy")
                if problem == "mixed-policy"
                else replace(
                    call,
                    generation=replace(
                        call.generation, logprobs=(-2.0,) * len(call.generation.tokens)
                    ),
                )
            )
            result.trajectory[0] = replace(result.trajectory[0], model_call=changed)

    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
        on_update=corrupt,
    )
    assert result.opsd is not None and result.opsd["status"] == "rejected"
    assert not adapter.updates and not adapter.refreshed
    assert run.ledger.snapshot()["charged_cny"] > 0
    assert json.loads((run.root / "opsd.json").read_text())["optimizer_step"] == 0
    metrics = json.loads((run.root / "remote.json").read_text())["rows"][-1]
    assert metrics["train/generated_tokens"] > 0
    assert metrics["train/optimizer_step"] == 0


@pytest.mark.parametrize("stage", ["score", "update", "refresh"])
def test_remote_failure_preserves_pending_cost_and_never_uploads_exception_contents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)

    def fail() -> None:
        raise ValueError("secret-opsd-key TEACHER_ONLY_REFERENCE HIDDEN_TEST")

    if stage == "score":

        def score(scores: TeacherScores) -> TeacherScores:
            fail()
            return scores

        adapter.score_response = score
    elif stage == "update":
        adapter.update_completion = fail
    else:

        def refresh(update: UpdateResult) -> Model:
            fail()
            return adapter.student

        monkeypatch.setattr(adapter, "refresh_student", refresh)
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed
    assert result.opsd is not None and result.opsd["status"] == "failed"
    assert result.opsd["reason"] == "ValueError"
    metrics = json.loads((run.root / "remote.json").read_text())["rows"][-1]
    assert metrics["train/optimizer_step"] == (1 if stage == "refresh" else 0)
    assert "secret-opsd-key" not in json.dumps(metrics)
    assert "TEACHER_ONLY_REFERENCE" not in json.dumps(metrics)
    assert "HIDDEN_TEST" not in json.dumps(metrics)
    if stage != "refresh":
        assert metrics["spend/run_pending_cny"] > 0
    if stage == "update":
        entries = run.ledger.snapshot()["entries"]
        with pytest.raises(ValueError, match="uncertain remote update"):
            run_experiment(
                task=repair_task(),
                model=run.student,
                opsd_run=run,
                output_dir=tmp_path / "episodes",
            )
        assert run.ledger.snapshot()["entries"] == entries


@pytest.mark.parametrize("stage", ["sampling", "update"])
def test_in_flight_batch_prevents_new_rollout_and_refresh_until_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)
    entered, release = Event(), Event()
    initial_student = run.student

    def wait_for_completion() -> None:
        entered.set()
        assert release.wait(10)

    if stage == "sampling":

        def respond(context: str) -> Generation:
            if not entered.is_set():
                wait_for_completion()
            return adapter.respond(context)

        adapter.student.respond = respond
    else:
        adapter.update_completion = wait_for_completion
    with ThreadPoolExecutor(max_workers=1) as workers:
        running = workers.submit(
            run_experiment,
            task=repair_task(),
            model=initial_student,
            opsd_run=run,
            output_dir=tmp_path / "episodes",
        )
        try:
            assert entered.wait(10)
            before = run.ledger.snapshot()["entries"]
            record = json.loads((run.root / "opsd.json").read_text())
            assert record["optimizer_step"] == 0
            assert run.student.identity == "test/sft@selected"
            assert not adapter.refreshed
            other_adapter = ControlledOPSD()
            with pytest.raises(ValueError, match="already active"):
                run_experiment(
                    task=repair_task(),
                    model=other_adapter.student,
                    opsd_run=OPSDRun(
                        run.root,
                        snapshot=run.snapshot,
                        adapter=other_adapter,
                        ledger=run.ledger,
                        config=run.config,
                        monitoring=run.monitoring,
                        resume=True,
                    ),
                    output_dir=tmp_path / "episodes",
                )
            assert not other_adapter.restored
            with pytest.raises(ValueError, match="already active"):
                run_experiment(
                    task=repair_task(),
                    model=initial_student,
                    opsd_run=run,
                    output_dir=tmp_path / "episodes",
                )
            assert run.ledger.snapshot()["entries"] == before
        finally:
            release.set()
        first = running.result(timeout=10)
    assert first.opsd is not None and first.opsd["status"] == "completed"
    second = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert second.policy == "test/student@opsd-1"
    assert run.student.identity == "test/student@opsd-2"
    assert adapter.teacher_identity == "test/sft@selected"


def test_monitor_outage_and_pending_spend_cannot_bypass_the_update_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch, funded_cny=1.0)
    pending = run.ledger.reserve(
        operation="evaluation",
        group="opsd-pilot",
        payload={"other_run": "pending"},
        upper_cny=0.48,
    )
    monkeypatch.setenv("FAKE_WANDB_LOG_FAIL", "1")
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed
    assert result.opsd is not None and result.opsd["status"] == "rejected"
    assert result.opsd["reason"] == "budget_refused"
    assert not adapter.updates and not adapter.refreshed
    entries = run.ledger.snapshot()["entries"]
    assert entries[0]["id"] == pending and entries[0]["state"] == "pending"
    assert not any(entry["operation"] == "opsd_update" for entry in entries)
    record = json.loads((run.root / "opsd.json").read_text())
    assert record["optimizer_step"] == 0 and record["pending_update"] is None
    metrics = [
        json.loads(line)
        for line in (run.root / "metrics.jsonl").read_text().splitlines()
    ]
    assert metrics[-1]["metrics"]["spend/project_pending_cny"] == 0.48
    assert metrics[-1]["metrics"]["train/optimizer_step"] == 0
    assert (
        json.loads((run.root / "monitor.json").read_text())["sync_status"] == "pending"
    )


@pytest.mark.parametrize("stage", ["sampling", "update"])
def test_interruption_keeps_progress_and_blocks_uncertain_optimizer_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)

    def interrupt() -> None:
        raise KeyboardInterrupt

    if stage == "sampling":

        def respond(context: str) -> Generation:
            interrupt()
            return adapter.respond(context)

        adapter.student.respond = respond
    else:
        adapter.update_completion = interrupt
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None and result.opsd["status"] == "interrupted"
    record = json.loads((run.root / "opsd.json").read_text())
    assert record["optimizer_step"] == 0
    assert not adapter.refreshed
    assert record["batches"][-1]["cost"]["actual_cny"] is None
    assert record["batches"][-1]["cost"]["pending_upper_bound_cny"] > 0
    original_journal = (run.root / "metrics.jsonl").read_bytes()
    if stage == "update":
        with pytest.raises(ValueError, match="uncertain remote update"):
            OPSDRun(
                run.root,
                snapshot=run.snapshot,
                adapter=ControlledOPSD(),
                ledger=run.ledger,
                config=run.config,
                monitoring=run.monitoring,
                resume=True,
            )
    else:
        pending = run.ledger.snapshot()["entries"][-1]
        run.ledger.reconcile(
            pending["id"],
            actual_cny=0,
            outcome="failed",
            evidence={"controlled_bill": "interrupted-read-only-sample"},
        )
        resumed = OPSDRun(
            run.root,
            snapshot=run.snapshot,
            adapter=ControlledOPSD(),
            ledger=run.ledger,
            config=run.config,
            monitoring=run.monitoring,
            resume=True,
        )
        completed = run_experiment(
            task=repair_task(),
            model=resumed.student,
            opsd_run=resumed,
            output_dir=tmp_path / "episodes",
        )
        assert completed.opsd is not None and completed.opsd["status"] == "completed"
        assert (run.root / "metrics.jsonl").read_bytes().startswith(original_journal)
        rows = json.loads((run.root / "remote.json").read_text())["rows"]
        assert [row["train/optimizer_step"] for row in rows] == [0, 1]


def test_failed_independent_grade_does_not_change_the_distillation_objective(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)
    task = replace(
        repair_task(),
        grader=FunctionGrader(
            "solution.py",
            "double",
            (GradingCase((7,), 99),),
        ),
    )
    result = run_experiment(
        task=task,
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert not result.passed
    assert result.opsd is not None and result.opsd["status"] == "completed"
    assert adapter.updates[0]["loss_fn"] == "importance_sampling"
    datum = adapter.updates[0]["batch"][0]
    for step in result.trajectory:
        call = step.model_call
        assert call is not None
        start = call.context_offset - 1
        assert datum.loss_fn_inputs["advantages"].tolist()[start] == 0.375


def test_confirmed_zero_bill_is_reported_as_known_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)

    def reconcile_controlled_bills() -> None:
        for entry in run.ledger.snapshot()["entries"]:
            run.ledger.reconcile(
                entry["id"],
                actual_cny=0,
                outcome="completed",
                evidence={"controlled_bill": "zero-charge"},
            )

    adapter.update_completion = reconcile_controlled_bills
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None and result.opsd["status"] == "completed"
    metrics = json.loads((run.root / "remote.json").read_text())["rows"][-1]
    assert metrics["spend/run_actual_known"] is True
    assert metrics["spend/run_actual_cny"] == 0


@pytest.mark.parametrize(
    "role", ["base-teacher", "latest-teacher", "base-student", "encoding"]
)
def test_incorrect_sft_roles_are_refused_before_any_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
) -> None:
    run, adapter = controlled_run(tmp_path, monkeypatch)
    if role == "base-teacher":
        adapter.teacher_identity = "test/base-model"
    elif role == "latest-teacher":
        adapter.teacher_identity = "test/student@opsd-1"
    elif role == "base-student":
        adapter.student.identity = "test/base-model"
    else:
        adapter.teacher_encoding = ByteEncoding()
        adapter.teacher_encoding.identity = "incompatible/tokenizer"
    with pytest.raises(ValueError, match="teacher|Student"):
        run_experiment(
            task=repair_task(),
            model=run.student,
            opsd_run=run,
            output_dir=tmp_path / "episodes",
        )
    assert not run.ledger.snapshot()["entries"]
    assert not adapter.student.prompts and not adapter.scoring and not adapter.updates


def test_monitor_interrupt_after_completed_update_can_resume_without_duplicate_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, _ = controlled_run(tmp_path, monkeypatch)
    original_log = monitoring_sdk.Run.log

    def interrupt_log(
        self: monitoring_sdk.Run, data: dict[str, Any], *, step: int
    ) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(monitoring_sdk.Run, "log", interrupt_log)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(
            task=repair_task(),
            model=run.student,
            opsd_run=run,
            output_dir=tmp_path / "episodes",
        )
    before = json.loads((run.root / "monitor.json").read_text())
    record = json.loads((run.root / "opsd.json").read_text())
    assert record["status"] == "interrupted"
    assert record["optimizer_step"] == 1 and record["pending_update"] is None
    assert not monitoring_sdk.current_run.rows
    monkeypatch.setattr(monitoring_sdk.Run, "log", original_log)
    resumed = OPSDRun(
        run.root,
        snapshot=run.snapshot,
        adapter=ControlledOPSD(),
        ledger=run.ledger,
        config=run.config,
        monitoring=run.monitoring,
        resume=True,
    )
    result = run_experiment(
        task=repair_task(),
        model=resumed.student,
        opsd_run=resumed,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None and result.opsd["status"] == "completed"
    assert json.loads((run.root / "monitor.json").read_text())["id"] == before["id"]
    rows = json.loads((run.root / "remote.json").read_text())["rows"]
    assert [row["train/optimizer_step"] for row in rows] == [1, 2]
