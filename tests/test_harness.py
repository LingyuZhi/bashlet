"""Model-driven episodes through the same experiment entry point."""

import json
import shlex
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from adapters import (
    WORDS,
    ScriptedModel,
    bash,
    latest_feedback,
    turn_number,
    write_file,
)

from bashlet import (
    PROTOCOL_SHA256,
    Bash,
    Generation,
    GradingCase,
    InteractionBudget,
    InvalidAction,
    SamplingParams,
    Submit,
    Task,
    run_experiment,
)

FROZEN_PROTOCOL_SHA256 = (
    "e3212936cc2175b313e7f2036e13d3dc5b61be5dee0ea68236a7ccae8f865077"
)
WRONG = "def solve(numbers):\n    return 3\n"
FIXED = "def solve(numbers):\n    return sum(numbers)\n"


def sum_task() -> Task:
    return Task(
        task_id="manual/sum-repair-v1",
        prompt="Write solution.py with solve(numbers), returning the sum of integers.",
        feedback_tests=(
            "from solution import solve\n"
            "assert solve([1, 2]) == 3\n"
            "assert solve([4]) == 4, 'solve([4]) should be 4'\n"
            "print('feedback passed')\n"
        ),
        grading_cases=(
            GradingCase(args=([1, 2],), expected=3),
            GradingCase(args=([],), expected=0),
            GradingCase(args=([-5, 2],), expected=-3),
        ),
        reference_implementation=FIXED,
    )


def repairing_policy(context: str) -> str:
    """Writes a wrong first implementation and repairs only after seeing why."""
    feedback = latest_feedback(context)
    if turn_number(context) == 1:
        return write_file(WRONG)
    if "solve([4]) should be 4" in feedback:
        return write_file(FIXED)
    if "feedback passed" in feedback:
        return "Feedback passed.\n```bash\nsubmit\n```"
    return bash("python3 feedback_tests.py")


def test_policy_repairs_after_real_feedback_and_submission_is_graded(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(repairing_policy)
    sampling = SamplingParams(max_tokens=256, temperature=0.7, top_p=0.95, seed=11)
    result = run_experiment(
        task=sum_task(),
        model=model,
        sampling=sampling,
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert [type(step.action) for step in result.trajectory] == [
        Bash,
        Bash,
        Bash,
        Bash,
        Submit,
    ]
    failed, passed = result.trajectory[1].observation, result.trajectory[3].observation
    assert failed is not None and failed.status == "command_failed"
    assert "solve([4]) should be 4" in failed.stderr
    assert passed is not None and passed.stdout == "feedback passed\n"
    assert result.grade is not None and result.grade.passed
    assert result.submission is not None
    assert (result.run_dir / result.submission.path).read_text() == FIXED

    # Each generation keeps its sampled tokens, logprobs, conditions and policy.
    for step, generation in zip(result.trajectory, model.generations, strict=True):
        call = step.model_call
        assert call is not None
        assert call.generation == generation
        assert call.policy == "test/scripted-student@step-0"
        assert call.sampling.temperature == 0.7 and call.sampling.seed == 11
    assert result.policy == "test/scripted-student@step-0"
    assert result.encoding == "test/byte-chat-v1"

    # Later prompts extend earlier ones with the raw sampled token IDs, not a
    # decode → re-encode approximation; WORDS tokens can only come from sampling.
    for earlier, later, step in zip(
        model.prompts, model.prompts[1:], result.trajectory, strict=False
    ):
        call = step.model_call
        assert call is not None
        assert later[: len(earlier)] == earlier
        assert call.context_offset == len(earlier)
        end = call.context_offset + len(call.generation.tokens)
        assert later[call.context_offset : end] == call.generation.tokens
    assert any(token in WORDS for token in model.prompts[-1])

    # A reply is conditioned only on observations that had already happened.
    renders = [model.encoding.render(prompt) for prompt in model.prompts]
    assert "solve([4]) should be 4" not in renders[1]
    assert "solve([4]) should be 4" in renders[2]
    assert "feedback passed" not in "".join(renders[:4])

    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["mode"] == "model"
    assert record["protocol"]["name"] == "bash-block-v1"
    # Frozen before demonstrations: changing any agent-visible protocol text
    # must come with a new protocol version and an updated fingerprint here.
    assert record["protocol"]["sha256"] == PROTOCOL_SHA256 == FROZEN_PROTOCOL_SHA256
    assert record["trajectory"][0]["model_call"]["generation"]["tokens"] == list(
        model.generations[0].tokens
    )


def test_single_invalid_action_is_fed_back_without_execution(tmp_path: Path) -> None:
    def policy(context: str) -> str:
        if turn_number(context) == 1:
            return "I will write the file now."  # no block
        if "<invalid_action>" in latest_feedback(context):
            return write_file(FIXED)
        return "```bash\nsubmit\n```"

    result = run_experiment(
        task=sum_task(), model=ScriptedModel(policy), output_dir=tmp_path
    )

    assert result.status == "completed", result.reason
    invalid = result.trajectory[0]
    assert invalid.action == InvalidAction(
        "Expected exactly one ```bash block, found 0"
    )
    assert invalid.observation is None
    assert invalid.feedback is not None and "Nothing was executed" in invalid.feedback
    assert result.usage.invalid_actions == 1 and result.usage.steps == 3
    assert result.grade is not None and result.grade.passed


def raising(context: str) -> str:
    raise TimeoutError("sampler did not answer")


def mismatched(context: str) -> Generation:
    return Generation(tokens=(65, 66), logprobs=(-0.1,), stop_reason="stop")


@pytest.mark.parametrize(
    ("respond", "budget", "sampling", "status", "reason"),
    [
        (
            lambda context: "no block",
            InteractionBudget(),
            SamplingParams(),
            "invalid_action",
            "3 consecutive invalid actions",
        ),
        (
            raising,
            InteractionBudget(),
            SamplingParams(),
            "infrastructure_error",
            "TimeoutError: sampler did not answer",
        ),
        (
            mismatched,
            InteractionBudget(),
            SamplingParams(),
            "model_error",
            "2 tokens but 1 logprobs",
        ),
        (
            lambda context: bash("true"),
            InteractionBudget(max_steps=4),
            SamplingParams(),
            "budget_exhausted",
            "max_steps=4",
        ),
        (
            lambda context: bash("true"),
            InteractionBudget(max_generated_tokens=60),
            SamplingParams(),
            "budget_exhausted",
            "max_generated_tokens=60",
        ),
        (
            lambda context: bash("sleep 0.3"),
            InteractionBudget(max_seconds=0.5),
            SamplingParams(),
            "budget_exhausted",
            "max_seconds=0.5",
        ),
        (
            lambda context: write_file(FIXED),
            InteractionBudget(max_steps=1),
            SamplingParams(max_tokens=8),
            "budget_exhausted",
            "max_steps=1",
        ),
    ],
    ids=["invalid", "raised", "misaligned", "steps", "tokens", "seconds", "truncated"],
)
def test_terminations_are_distinct_and_budgets_are_audited(
    tmp_path: Path,
    respond: Callable[[str], str | Generation],
    budget: InteractionBudget,
    sampling: SamplingParams,
    status: str,
    reason: str,
) -> None:
    model = ScriptedModel(respond)
    result = run_experiment(
        task=sum_task(),
        model=model,
        budget=budget,
        sampling=sampling,
        output_dir=tmp_path,
    )

    assert result.status == status
    assert reason in result.reason
    assert result.submission is None and result.grade is None
    assert result.usage.steps <= budget.max_steps
    if status not in ("model_error", "infrastructure_error"):
        assert result.usage.generated_tokens <= budget.max_generated_tokens
    assert result.usage.generated_tokens == sum(
        len(generation.tokens) for generation in model.generations
    )
    assert all(request.max_tokens <= sampling.max_tokens for request in model.requests)
    if status == "budget_exhausted" and "tokens" in reason:
        # The last request is clipped to what remains of the generation budget.
        assert model.requests[-1].max_tokens < sampling.max_tokens
    if sampling.max_tokens == 8:
        assert result.trajectory[0].action == InvalidAction(
            "The reply hit the generation length limit"
        )
        assert not (result.run_dir / "workspace" / "solution.py").exists()
    if respond is mismatched:
        # Contract violations keep the raw output out of the student's context.
        assert result.rejected_generation == model.generations[0]
        assert result.trajectory == []
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["status"] == status
    assert record["budget"] == asdict(budget)
    assert record["usage"]["steps"] == result.usage.steps


def test_unavailable_isolation_stops_before_any_model_call(tmp_path: Path) -> None:
    model = ScriptedModel(repairing_policy)
    result = run_experiment(
        task=sum_task(), model=model, output_dir=tmp_path, bwrap="/does-not-exist/bwrap"
    )
    assert result.status == "infrastructure_error"
    assert model.prompts == [] and result.trajectory == []


def test_repair_attempts_are_counted_apart_from_interaction_steps(
    tmp_path: Path,
) -> None:
    first_only = "def solve(numbers):\n    return numbers[0]\n"
    script = [
        write_file(WRONG),  # 0: candidate 0, the first implementation
        bash("python3 feedback_tests.py"),  # 1: fails, opens repair 0
        bash("python3 feedback_tests.py"),  # 2: same candidate, same attempt
        bash("cat feedback_tests.py"),  # 3: inspection, no verification
        write_file(first_only),  # 4: candidate 1
        bash("python3 feedback_tests.py"),  # 5: fails, closes 0 and opens 1
        write_file(FIXED),  # 6: candidate 2
        bash("python3 feedback_tests.py"),  # 7: passes, closes 1
        "```bash\nsubmit\n```",
    ]
    result = run_experiment(
        task=sum_task(),
        model=ScriptedModel(lambda context: script[turn_number(context) - 1]),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert result.usage.steps == 9
    assert [(c.index, c.formed_at_step) for c in result.candidates] == [
        (0, 0),
        (1, 4),
        (2, 6),
    ]
    assert [
        (r.failed_step, r.from_candidate, r.to_candidate, r.verified_at_step, r.passed)
        for r in result.repair_attempts
    ] == [(1, 0, 1, 5, False), (5, 1, 2, 7, True)]
    assert [step.verification for step in result.trajectory] == [
        False, True, True, False, False, True, False, True, False
    ]  # fmt: skip
    assert [step.candidate for step in result.trajectory] == [0, 0, 0, 0, 1, 1, 2, 2, 2]
    assert (result.run_dir / result.candidates[0].path).read_text() == WRONG
    assert result.submission is not None
    assert result.submission.sha256 == result.candidates[2].sha256
    record = json.loads((result.run_dir / "run.json").read_text())
    assert len(record["repair_attempts"]) == 2
    assert record["candidates"][1]["formed_at_step"] == 4


def test_student_never_sees_reference_grading_material_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = "svc-credential-5d2e91"
    monkeypatch.setenv("PYTRIO_API_KEY", credential)
    task = replace(
        sum_task(),
        reference_implementation="# reference-marker-8c41\n" + FIXED,
        grading_cases=(GradingCase(args=([424242, 7],), expected=424249),),
    )
    probes = [
        bash("env; cat task.md feedback_tests.py; ls -la . .."),
        bash("cat ../private/task.json ../run.json ../submission/solution.py"),
        # The searched value is computed so the command itself cannot leak it.
        bash("grep -rs $((424240 + 2)) / | head; cat /proc/1/environ"),
        write_file(FIXED),
        "```bash\nsubmit\n```",
    ]
    model = ScriptedModel(lambda context: probes[turn_number(context) - 1])
    result = run_experiment(task=task, model=model, output_dir=tmp_path)

    assert result.status == "completed", result.reason
    assert result.grade is not None and result.grade.passed
    student_visible = [model.encoding.render(prompt) for prompt in model.prompts]
    student_visible += [step.feedback or "" for step in result.trajectory]
    student_visible += [
        path.read_text(errors="replace")
        for path in (result.run_dir / "workspace").rglob("*")
        if path.is_file()
    ]
    for secret in (credential, "reference-marker-8c41", "424242", "424249"):
        assert all(secret not in text for text in student_visible), secret
    # The first observation is real: the task files the student may read.
    assert "feedback passed" in student_visible[1]


def test_executor_failure_mid_episode_is_an_infrastructure_error(
    tmp_path: Path,
) -> None:
    # Real bwrap for the isolation probe and the first action, then unavailable.
    launcher, counter = tmp_path / "bwrap-twice", tmp_path / "count"
    launcher.write_text(
        "#!/bin/bash\n"
        f"count_file={shlex.quote(str(counter))}\n"
        'n=0; if [ -f "$count_file" ]; then read -r n < "$count_file"; fi\n'
        'n=$((n+1)); printf "%s\\n" "$n" > "$count_file"\n'
        'if [ "$n" -gt 2 ]; then echo "executor unavailable" >&2; exit 70; fi\n'
        'exec /usr/bin/bwrap "$@"\n'
    )
    launcher.chmod(0o700)
    model = ScriptedModel(repairing_policy)
    result = run_experiment(
        task=sum_task(),
        model=model,
        output_dir=tmp_path / "runs",
        bwrap=str(launcher),
    )
    assert result.status == "infrastructure_error"
    assert "executor unavailable" in result.reason
    assert len(model.prompts) == 2 and len(result.trajectory) == 2
    assert result.grade is None


def test_only_unmodified_feedback_runs_verify_and_reverts_reuse_candidates(
    tmp_path: Path,
) -> None:
    script = [
        write_file(WRONG),  # 0: candidate 0
        bash("python3 feedback_tests.py; true"),  # 1: masked exit, not verification
        bash("python3 feedback_tests.py"),  # 2: fails, opens repair 0
        write_file(FIXED),  # 3: candidate 1
        bash("printf 'print(1)' > feedback_tests.py"),  # 4: tampers with the tests
        bash("python3 feedback_tests.py"),  # 5: tests modified, not verification
        write_file(WRONG),  # 6: revert to candidate 0
        "```bash\nsubmit\n```",
    ]
    result = run_experiment(
        task=sum_task(),
        model=ScriptedModel(lambda context: script[turn_number(context) - 1]),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert [step.verification for step in result.trajectory] == [
        False, False, True, False, False, False, False, False
    ]  # fmt: skip
    assert [step.candidate for step in result.trajectory] == [0, 0, 0, 1, 1, 1, 0, 0]
    assert len(result.candidates) == 2
    [attempt] = result.repair_attempts
    assert (attempt.failed_step, attempt.passed) == (2, None)
    assert result.grade is not None and not result.grade.passed
