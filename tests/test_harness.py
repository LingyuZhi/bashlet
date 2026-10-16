"""Model-driven episodes through the same experiment entry point."""

import json
import shlex
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from adapters import (
    ASSISTANT,
    END,
    SYSTEM,
    USER,
    WORDS,
    ByteEncoding,
    ScriptedModel,
    bash,
    condition_overrides,
    latest_feedback,
    replay,
    submit,
    turn_number,
    write_file,
)

from bashlet import (
    PROTOCOL_SHA256,
    Bash,
    FunctionGrader,
    Generation,
    GradingCase,
    InteractionBudget,
    InvalidAction,
    ReplayModel,
    SamplingParams,
    Task,
    run_experiment,
)
from bashlet.harness.protocol import SYSTEM_PROMPT

WRONG = "def solve(numbers):\n    return 3\n"
FIXED = "def solve(numbers):\n    return sum(numbers)\n"


def sum_task() -> Task:
    prompt = "Write solution.py with solve(numbers), returning the sum of integers."
    return Task(
        task_id="manual/sum-repair-v1",
        prompt=prompt,
        initial_files={
            "task.md": prompt,
            "feedback_tests.py": (
                "from solution import solve\n"
                "assert solve([1, 2]) == 3\n"
                "assert solve([4]) == 4, 'solve([4]) should be 4'\n"
                "print('feedback passed')\n"
            ),
        },
        grader=FunctionGrader(
            module_path="solution.py",
            function_name="solve",
            cases=(
                GradingCase(args=([1, 2],), expected=3),
                GradingCase(args=([],), expected=0),
                GradingCase(args=([-5, 2],), expected=-3),
            ),
        ),
        reference_implementation=FIXED,
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback_tests.py",
    )


def repairing_policy(context: str) -> str:
    """Writes a wrong first implementation and repairs only after seeing why."""
    feedback = latest_feedback(context)
    if turn_number(context) == 1:
        return write_file(WRONG)
    if "solve([4]) should be 4" in feedback:
        return write_file(FIXED)
    if "feedback passed" in feedback:
        return submit()
    return bash("python3 feedback_tests.py")


def test_successful_first_line_marker_submits_workspace_and_remaining_output(
    tmp_path: Path,
) -> None:
    task = sum_task()
    result = run_experiment(
        task=replace(task, initial_files=task.initial_files | {"solution.py": FIXED}),
        model=replay(
            [
                "The workspace is ready to submit.\n"
                "```bash\n"
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nReady.\\n雪\\n'\n"
                "```",
                bash("touch MUST_NOT_RUN"),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.episode_status == "submitted", result.reason
    assert result.passed
    assert result.submission_text == "Ready.\n雪\n"
    assert result.usage.steps == 1
    assert isinstance(result.trajectory[0].action, Bash)
    assert result.trajectory[0].observation is not None
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["submission_text"] == "Ready.\n雪\n"


def test_observation_preserves_stdout_and_stderr_write_order(tmp_path: Path) -> None:
    command = (
        "import os; "
        "os.write(1, b'first\\n'); os.write(2, b'second\\n'); "
        "os.write(1, b'third\\n'); os.write(2, b'fourth\\n')"
    )
    result = run_experiment(
        task=sum_task(),
        model=replay([bash("python3 -c " + shlex.quote(command))]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
    )

    step = result.trajectory[0]
    assert step.observation is not None
    assert step.observation.output == "first\nsecond\nthird\nfourth\n"
    assert step.feedback == (
        "<returncode>0</returncode>\n<output>\nfirst\nsecond\nthird\nfourth\n</output>"
    )


def test_long_observation_keeps_head_and_tail_without_stopping_the_command(
    tmp_path: Path,
) -> None:
    source = (
        "import os; from pathlib import Path; "
        "os.write(1, ('H' * 2000 + '雪' * 1001 + 'T' * 2000).encode()); "
        f"Path('solution.py').write_text({FIXED!r})"
    )
    result = run_experiment(
        task=sum_task(),
        model=replay([bash("python3 -c " + shlex.quote(source)), submit()]),
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    step = result.trajectory[0]
    assert step.observation is not None
    assert step.observation.output == "H" * 2000 + "雪" * 1001 + "T" * 2000
    assert step.feedback == (
        "<returncode>0</returncode>\n<output>\n"
        + "H" * 2000
        + "\n[... 1001 characters omitted ...]\n"
        "Narrow the output with head, tail, sed, or a more selective search.\n"
        + "T" * 2000
        + "\n</output>"
    )
    assert (result.workspace / "solution.py").read_text() == FIXED


def test_hard_output_limit_discards_bytes_and_allows_execution_and_next_step(
    tmp_path: Path,
) -> None:
    source = (
        "import os; from pathlib import Path; "
        "os.write(1, b'X' * 40000); "
        f"Path('solution.py').write_text({FIXED!r})"
    )
    result = run_experiment(
        task=sum_task(),
        model=replay([bash("python3 -c " + shlex.quote(source)), submit()]),
        output_dir=tmp_path,
        overrides=["environment.output_bytes=128"],
    )

    assert result.passed, result.reason
    step = result.trajectory[0]
    assert step.observation is not None
    assert step.observation.status == "ok"
    assert step.observation.exit_code == 0
    assert step.observation.output == "X" * 128
    assert step.observation.output_truncated
    assert step.feedback is not None
    assert "further output was discarded" in step.feedback
    assert result.usage.steps == 2


def test_timeout_has_its_own_feedback_and_kills_detached_writers(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash(
                    "setsid bash -c 'sleep 0.4; touch delayed-write' "
                    ">/dev/null 2>&1 < /dev/null & "
                    "printf 'before timeout\\n'; sleep 5"
                ),
                bash("printf 'next step\\n'"),
            ]
        ),
        output_dir=tmp_path,
        overrides=["environment.timeout_seconds=0.2", "budget.max_steps=2"],
    )

    first, second = result.trajectory
    assert first.observation is not None and first.observation.status == "timeout"
    assert first.feedback is not None
    assert first.feedback.startswith(
        "The command timed out. All processes from this step were terminated.\n"
    )
    assert "before timeout\n" in first.feedback
    assert second.observation is not None and second.observation.output == "next step\n"
    time.sleep(0.5)
    assert not (result.workspace / "delayed-write").exists()


@pytest.mark.parametrize("explanation", ["", "Run the command.\n"])
def test_reply_executes_a_bash_block_with_an_optional_explanation(
    tmp_path: Path,
    explanation: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([explanation + "```bash\nprintf 'action ran\\n'\n```"]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
    )

    step = result.trajectory[0]
    assert isinstance(step.action, Bash)
    assert step.observation is not None
    assert step.observation.output == "action ran\n"


def test_output_cutoff_cannot_turn_a_longer_first_line_into_a_submission(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([bash("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT_EXTRA\\n'")]),
        output_dir=tmp_path,
        overrides=["environment.output_bytes=37", "budget.max_steps=1"],
    )

    assert result.episode_status == "budget_exhausted"
    assert result.submission_text is None and result.grade is None
    observation = result.trajectory[0].observation
    assert observation is not None and observation.output_truncated
    assert observation.output == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def test_an_extra_unclosed_bash_block_prevents_execution(tmp_path: Path) -> None:
    reply = bash("touch MUST_NOT_RUN") + "\n```bash\ntrue"
    result = run_experiment(
        task=sum_task(),
        model=replay([reply]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
    )

    assert isinstance(result.trajectory[0].action, InvalidAction)
    assert result.trajectory[0].observation is None
    assert not (result.workspace / "MUST_NOT_RUN").exists()


@pytest.mark.parametrize(
    "reply",
    [
        "There is no command.",
        bash("touch MUST_NOT_RUN") + "\n" + bash("true"),
        bash(" \t"),
        "The block is incomplete.\n```bash\ntouch MUST_NOT_RUN",
    ],
    ids=["missing", "multiple", "empty", "unclosed"],
)
def test_format_errors_count_steps_and_stop_at_the_configured_limit(
    tmp_path: Path,
    reply: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([reply, reply, bash("touch MUST_NOT_RUN")]),
        output_dir=tmp_path,
        overrides=["budget.max_consecutive_invalid=2"],
    )

    assert result.episode_status == "format_error"
    assert result.usage.steps == result.usage.invalid_actions == 2
    assert result.grade is None and result.submission_text is None
    assert all(step.observation is None for step in result.trajectory)
    assert all(
        step.feedback is not None and "Nothing was executed" in step.feedback
        for step in result.trajectory
    )
    assert not (result.workspace / "MUST_NOT_RUN").exists()


def test_a_valid_command_resets_consecutive_format_errors(tmp_path: Path) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            ["no block", bash("true"), "no block", write_file(FIXED), submit()]
        ),
        output_dir=tmp_path,
        overrides=["budget.max_consecutive_invalid=2"],
    )

    assert result.passed, result.reason
    assert result.usage.steps == 5 and result.usage.invalid_actions == 2


@pytest.mark.parametrize(
    "reply",
    [
        bash("printf 'prefix\\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'"),
        bash("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; exit 7"),
        "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```bash\ntrue\n```",
        bash("printf '\\nCOMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'"),
        bash("printf ' COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'"),
        bash("submit"),
    ],
    ids=["later-line", "nonzero", "thought-only", "blank-line", "space", "v1-command"],
)
def test_marker_elsewhere_or_in_a_failed_command_does_not_submit(
    tmp_path: Path,
    reply: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([reply]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
    )

    assert result.episode_status == "budget_exhausted"
    assert result.submission_text is None and result.grade is None


def test_marker_on_stderr_submits_when_it_is_the_first_combined_line(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=replace(sum_task(), initial_files={"solution.py": FIXED}),
        model=replay(
            [bash("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n' >&2; printf Done")]
        ),
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    assert result.submission_text == "Done"


def test_cd_and_export_do_not_persist_into_the_next_step(tmp_path: Path) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash(
                    "mkdir sub; cd sub; export BASHLET_STEP_VALUE=present; touch kept"
                ),
                bash(
                    "pwd; printf '%s\\n' \"${BASHLET_STEP_VALUE-unset}\"; cat sub/kept"
                ),
            ]
        ),
        output_dir=tmp_path,
        overrides=["budget.max_steps=2"],
    )

    observation = result.trajectory[1].observation
    assert observation is not None and observation.status == "ok"
    assert observation.output == "/workspace\nunset\n"
    assert (result.workspace / "sub/kept").is_file()
    assert "fresh shell" in SYSTEM_PROMPT and "shell state" in SYSTEM_PROMPT


def test_observation_head_and_tail_lengths_follow_protocol_config(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([bash("printf 'abc雪雪XYZ'")]),
        output_dir=tmp_path,
        overrides=[
            "budget.max_steps=1",
            "protocol.observation_head_chars=3",
            "protocol.observation_tail_chars=2",
        ],
    )

    assert result.trajectory[0].feedback == (
        "<returncode>0</returncode>\n<output>\nabc\n"
        "[... 3 characters omitted ...]\n"
        "Narrow the output with head, tail, sed, or a more selective search.\n"
        "YZ\n</output>"
    )


def test_replay_keeps_reply_tokens_and_append_only_context(tmp_path: Path) -> None:
    task = sum_task()
    replies = [bash("printf '雪\\n'"), write_file(FIXED), submit()]
    result = run_experiment(
        task=task,
        model=ReplayModel(replies, encoding=ByteEncoding()),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert result.grade is not None and result.grade.passed
    assert result.trajectory[0].observation is not None
    assert result.trajectory[0].observation.output == "雪\n"
    expected_context = [
        SYSTEM, *SYSTEM_PROMPT.encode(), END, USER, *task.prompt.encode(), END,
        ASSISTANT,
    ]  # fmt: skip
    for reply, step in zip(replies, result.trajectory, strict=True):
        call = step.model_call
        assert call is not None
        assert call.context_offset == len(expected_context)
        assert call.text == reply
        assert call.generation.tokens == tuple(reply.encode())
        assert call.generation.source == "replayed"
        assert call.generation.logprobs is None
        expected_context.extend(reply.encode())
        if step.feedback is not None:
            expected_context.extend(
                [END, USER, *step.feedback.encode(), END, ASSISTANT]
            )
        assert result.context_tokens[: len(expected_context)] == expected_context
    assert result.context_tokens == expected_context
    assert result.usage.generated_tokens == sum(
        len(reply.encode()) for reply in replies
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["context_tokens"] == expected_context
    assert all(
        step["model_call"]["generation"]["source"] == "replayed"
        and step["model_call"]["generation"]["logprobs"] is None
        for step in record["trajectory"]
    )


@pytest.mark.parametrize("replies", [[], [bash("true")]])
def test_exhausted_replies_end_through_normal_invalid_action_rules(
    tmp_path: Path, replies: list[str]
) -> None:
    result = run_experiment(task=sum_task(), model=replay(replies), output_dir=tmp_path)

    assert result.status == "format_error"
    assert result.usage.steps == len(replies) + 3
    assert result.usage.invalid_actions == 3
    assert result.rejected_generation is None
    assert result.submission is None and result.grade is None
    for step in result.trajectory[len(replies) :]:
        assert isinstance(step.action, InvalidAction)
        assert step.observation is None
        assert step.model_call is not None
        assert step.model_call.generation.tokens == ()
        assert step.model_call.generation.source == "replayed"
        assert step.model_call.generation.logprobs is None


@pytest.mark.parametrize(
    ("budget", "sampling", "reason"),
    [
        (
            InteractionBudget(max_generated_tokens=8),
            SamplingParams(),
            "max_generated_tokens=8",
        ),
        (
            InteractionBudget(max_steps=1),
            SamplingParams(max_tokens=8),
            "max_steps=1",
        ),
    ],
)
def test_replay_obeys_generation_and_interaction_limits(
    tmp_path: Path, budget: InteractionBudget, sampling: SamplingParams, reason: str
) -> None:
    reply = write_file(FIXED)
    result = run_experiment(
        task=sum_task(),
        model=replay([reply]),
        overrides=condition_overrides(budget=budget, sampling=sampling),
        output_dir=tmp_path,
    )

    assert result.status == "budget_exhausted"
    assert reason in result.reason
    assert result.usage.generated_tokens == 8
    step = result.trajectory[0]
    assert step.action == InvalidAction("The reply hit the generation length limit")
    assert step.model_call is not None
    assert step.model_call.sampling.max_tokens == 8
    assert step.model_call.generation.tokens == tuple(reply.encode()[:8])
    assert step.model_call.generation.stop_reason == "length"
    assert not (result.run_dir / "workspace" / "solution.py").exists()


def test_policy_repairs_after_real_feedback_and_submission_is_graded(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(repairing_policy)
    sampling = SamplingParams(max_tokens=256, temperature=0.7, top_p=0.95, seed=11)
    result = run_experiment(
        task=sum_task(),
        model=model,
        overrides=condition_overrides(sampling=sampling),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert [type(step.action) for step in result.trajectory] == [
        Bash,
        Bash,
        Bash,
        Bash,
        Bash,
    ]
    failed, passed = result.trajectory[1].observation, result.trajectory[3].observation
    assert failed is not None and failed.status == "command_failed"
    assert "solve([4]) should be 4" in failed.output
    assert passed is not None and passed.output == "feedback passed\n"
    assert result.grade is not None and result.grade.passed
    assert result.submission is not None
    assert (
        result.run_dir / result.submission.path / "solution.py"
    ).read_text() == FIXED

    # Each generation keeps its sampled tokens, logprobs, conditions and policy.
    for step, generation in zip(result.trajectory, model.generations, strict=True):
        call = step.model_call
        assert call is not None
        assert call.generation == generation
        assert call.generation.source == "sampled"
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
    assert record["schema_version"] == 11
    assert record["protocol"]["name"] == "bash-block-v6"
    assert (
        record["protocol"]["sha256"]
        == PROTOCOL_SHA256
        == ("511a961cbe0398f6c58c23a802f1499155c6b681a8b149ceed74da71c96d4dc3")
    )
    assert record["trajectory"][0]["model_call"]["generation"]["tokens"] == list(
        model.generations[0].tokens
    )


def test_single_invalid_action_is_fed_back_without_execution(tmp_path: Path) -> None:
    def policy(context: str) -> str:
        if turn_number(context) == 1:
            return "I will write the file now."  # no block
        if "<format_error>" in latest_feedback(context):
            return write_file(FIXED)
        return submit()

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
    ("logprobs", "reason"),
    [
        (None, "require per-token logprobs"),
        ((), "2 tokens but 0 logprobs"),
        ((-0.1, float("nan")), "finite and non-positive"),
        ((-0.1, float("inf")), "finite and non-positive"),
        ((-0.1, float("-inf")), "finite and non-positive"),
        ((-0.1, 0.1), "finite and non-positive"),
    ],
    ids=["missing", "empty", "nan", "inf", "negative-inf", "positive"],
)
def test_sampled_generations_require_valid_per_token_logprobs(
    tmp_path: Path, logprobs: tuple[float, ...] | None, reason: str
) -> None:
    generation = Generation(tokens=(65, 66), logprobs=logprobs, stop_reason="stop")
    result = run_experiment(
        task=sum_task(),
        model=ScriptedModel(lambda context: generation),
        output_dir=tmp_path,
    )

    assert result.status == "model_error"
    assert reason in result.reason
    assert result.rejected_generation == generation
    assert result.trajectory == []
    assert result.usage.generated_tokens == 2
    assert result.context_tokens[-1] == ASSISTANT
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["rejected_generation"]["source"] == "sampled"


@pytest.mark.parametrize(
    ("source", "logprobs", "reason"),
    [
        ("unknown", None, "Unknown generation source"),
        ("replayed", (-0.1,), "Replayed generations cannot carry sampling logprobs"),
    ],
)
def test_generation_source_cannot_bypass_the_probability_contract(
    tmp_path: Path, source: str, logprobs: tuple[float, ...] | None, reason: str
) -> None:
    generation = Generation(
        tokens=(65,),
        stop_reason="stop",
        logprobs=logprobs,
        source=source,  # type: ignore[arg-type]
    )
    result = run_experiment(
        task=sum_task(),
        model=ScriptedModel(lambda context: generation),
        output_dir=tmp_path,
    )

    assert result.status == "model_error"
    assert reason in result.reason
    assert result.rejected_generation == generation
    assert result.trajectory == []


@pytest.mark.parametrize(
    ("respond", "budget", "sampling", "status", "reason"),
    [
        (
            lambda context: "no block",
            InteractionBudget(),
            SamplingParams(),
            "format_error",
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
        overrides=condition_overrides(budget=budget, sampling=sampling),
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
    assert all(step.workspace_manifest for step in result.trajectory)


def test_unavailable_isolation_stops_before_any_model_call(tmp_path: Path) -> None:
    model = ScriptedModel(repairing_policy)
    result = run_experiment(
        task=sum_task(),
        model=model,
        output_dir=tmp_path,
        overrides=condition_overrides(bwrap="/does-not-exist/bwrap"),
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
        submit(),
    ]
    result = run_experiment(
        task=sum_task(),
        model=replay(script),
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
    assert (
        result.submission.manifest["solution.py"].sha256 == result.candidates[2].sha256
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert len(record["repair_attempts"]) == 2
    assert record["candidates"][1]["formed_at_step"] == 4


def test_student_never_sees_reference_grading_material_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential = "svc-credential-5d2e91"
    monkeypatch.setenv("PYTRIO_API_KEY", credential)
    base = sum_task()
    assert isinstance(base.grader, FunctionGrader)
    task = replace(
        base,
        reference_implementation="# reference-marker-8c41\n" + FIXED,
        grader=replace(
            base.grader,
            cases=(GradingCase(args=([424242, 7],), expected=424249),),
        ),
    )
    probes = [
        bash("env; cat task.md feedback_tests.py; ls -la . .."),
        bash("cat ../private/task.json ../run.json ../submission/solution.py"),
        # The searched value is computed so the command itself cannot leak it.
        bash("grep -rs $((424240 + 2)) / | head; cat /proc/1/environ"),
        write_file(FIXED),
        submit(),
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
        overrides=condition_overrides(bwrap=str(launcher)),
    )
    assert result.status == "infrastructure_error"
    assert "executor unavailable" in result.reason
    assert len(model.prompts) == 2 and len(result.trajectory) == 2
    assert result.grade is None
    assert result.trajectory[0].workspace_manifest == (
        result.trajectory[1].workspace_manifest
    )
    assert [step.candidate for step in result.trajectory] == [0, 0]


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
        submit(),
    ]
    result = run_experiment(
        task=sum_task(),
        model=replay(script),
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


def test_candidate_analysis_uses_task_declarations_after_the_episode(
    tmp_path: Path,
) -> None:
    base = sum_task()
    assert isinstance(base.grader, FunctionGrader)
    task = replace(
        base,
        implementation_paths=("answer.py",),
        feedback_command="python3 -B feedback_tests.py",
        initial_files={
            "feedback_tests.py": "from answer import solve\nassert solve([4]) == 4\n",
        },
        grader=replace(base.grader, module_path="answer.py"),
    )
    replies = [
        bash(f"cat > answer.py <<'PY'\n{WRONG}PY"),
        bash("python3 feedback_tests.py"),  # Not the task's feedback command.
        bash("python3 -B feedback_tests.py"),
        bash(f"cat > answer.py <<'PY'\n{FIXED}PY"),
        bash("python3 -B feedback_tests.py"),
        write_file(FIXED),  # An extra file does not change the declared candidate.
        submit(),
    ]

    def policy(context: str) -> str:
        record = json.loads(next(tmp_path.glob("*/run.json")).read_text())
        # Operator-visible records contain evidence during the episode; analysis
        # only appears after termination and never influences the next action.
        assert record["candidates"] == record["repair_attempts"] == []
        assert all(
            step["candidate"] is None and not step["verification"]
            for step in record["trajectory"]
        )
        return replies[turn_number(context) - 1]

    result = run_experiment(task=task, model=ScriptedModel(policy), output_dir=tmp_path)

    assert result.status == "completed", result.reason
    assert result.grade is not None and result.grade.passed
    assert [(c.index, c.formed_at_step) for c in result.candidates] == [(0, 0), (1, 3)]
    assert [step.candidate for step in result.trajectory] == [0, 0, 0, 1, 1, 1, 1]
    assert [step.verification for step in result.trajectory] == [
        False, False, True, False, True, False, False
    ]  # fmt: skip
    [attempt] = result.repair_attempts
    assert (
        attempt.failed_step, attempt.from_candidate, attempt.to_candidate,
        attempt.verified_at_step, attempt.passed,
    ) == (2, 0, 1, 4, True)  # fmt: skip
    assert (result.run_dir / result.candidates[0].path).read_text() == WRONG
    record = json.loads((result.run_dir / "run.json").read_text())
    assert len(record["candidates"]) == 2 and len(record["repair_attempts"]) == 1


def test_candidates_deduplicate_the_complete_declared_implementation(
    tmp_path: Path,
) -> None:
    source = "from helpers import total\ndef solve(numbers): return total(numbers)\n"
    wrong = "def total(numbers): return 3\n"
    fixed = "def total(numbers): return sum(numbers)\n"
    result = run_experiment(
        task=replace(sum_task(), implementation_paths=("solution.py", "helpers.py")),
        model=replay(
            [
                bash(
                    f"cat > solution.py <<'PY'\n{source}PY\n"
                    f"cat > helpers.py <<'PY'\n{wrong}PY"
                ),
                bash("python3 feedback_tests.py"),
                bash(f"cat > helpers.py <<'PY'\n{fixed}PY"),
                bash("python3 feedback_tests.py"),
                bash(f"cat > helpers.py <<'PY'\n{wrong}PY"),
                bash(f"cat > helpers.py <<'PY'\n{fixed}PY"),
            ]
        ),
        overrides=condition_overrides(budget=InteractionBudget(max_steps=6)),
        output_dir=tmp_path,
    )

    # Analysis remains available when the episode ends without submitting.
    assert result.status == "budget_exhausted", result.reason
    assert [step.candidate for step in result.trajectory] == [0, 0, 1, 1, 0, 1]
    first, second = result.candidates
    assert first.sha256 != second.sha256
    assert first.files["solution.py"] == second.files["solution.py"]
    blobs = result.run_dir / "blobs"
    assert (blobs / first.files["helpers.py"]).read_text() == wrong
    assert (blobs / second.files["helpers.py"]).read_text() == fixed
    assert json.loads((result.run_dir / first.path).read_text()) == first.files
    [attempt] = result.repair_attempts
    assert (attempt.from_candidate, attempt.to_candidate, attempt.passed) == (
        0,
        1,
        True,
    )


@pytest.mark.parametrize(
    ("paths", "expected_candidates"),
    [((), 0), (("solution.py",), 2)],
)
def test_optional_analysis_declarations_do_not_prevent_an_episode(
    tmp_path: Path, paths: tuple[str, ...], expected_candidates: int
) -> None:
    result = run_experiment(
        task=replace(sum_task(), implementation_paths=paths, feedback_command=None),
        model=replay(
            [
                write_file(WRONG),
                bash("python3 feedback_tests.py"),
                write_file(FIXED),
                bash("python3 feedback_tests.py"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert result.grade is not None and result.grade.passed
    assert len(result.candidates) == expected_candidates
    assert result.repair_attempts == []
    assert not any(step.verification for step in result.trajectory)
    if not paths:
        assert all(step.candidate is None for step in result.trajectory)


def test_feedback_symlinks_do_not_verify_and_restoring_materials_enables_verification(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                write_file(WRONG),
                bash(
                    "cp feedback_tests.py copy.py; rm feedback_tests.py; "
                    "ln -s copy.py feedback_tests.py"
                ),
                bash("python3 feedback_tests.py"),
                bash("rm feedback_tests.py; cp copy.py feedback_tests.py"),
                bash("python3 feedback_tests.py"),
                write_file(FIXED),
                bash("python3 feedback_tests.py"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    assert [step.verification for step in result.trajectory] == [
        False, False, False, False, True, False, True, False
    ]  # fmt: skip
    [attempt] = result.repair_attempts
    assert (attempt.failed_step, attempt.verified_at_step, attempt.passed) == (
        4,
        6,
        True,
    )


@pytest.mark.parametrize(
    "command",
    [
        "rm solution.py",
        "rm solution.py; mkfifo solution.py",
        "head -c 65537 /dev/zero > solution.py",
    ],
)
def test_unavailable_or_oversized_implementation_keeps_the_last_candidate(
    tmp_path: Path, command: str
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([write_file(WRONG), bash(command), submit()]),
        output_dir=tmp_path,
    )

    assert result.status == (
        "completed" if command == "rm solution.py" else "invalid_submission"
    ), result.reason
    assert not result.passed
    assert [step.candidate for step in result.trajectory] == [0, 0, 0]
    [candidate] = result.candidates
    assert (result.run_dir / candidate.path).read_text() == WRONG
