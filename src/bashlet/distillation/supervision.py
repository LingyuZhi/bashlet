"""Align frozen-teacher probabilities to the student's original causal history."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np

from bashlet.config import ExperimentConfig, ProtocolConfig
from bashlet.harness import protocol
from bashlet.harness.models import (
    Encoding,
    ExperimentResult,
    Generation,
    InvalidAction,
    ModelResponseError,
    Observation,
    SamplingParams,
    Task,
)

if TYPE_CHECKING:
    import pytrio as trio

    from bashlet.distillation import OPSDConfig, TeacherScores


def build_supervision(
    result: ExperimentResult,
    task: Task,
    encoding: Encoding,
    condition: ExperimentConfig,
    *,
    config: OPSDConfig,
    teacher_encoding: Encoding,
    score: Callable[[Sequence[int], int, int], TeacherScores],
) -> tuple[trio.Datum, dict[str, Any]]:
    import pytrio as trio

    initial = encoding.prompt(condition.protocol.system_prompt, task.prompt)
    privileged = teacher_encoding.prompt(
        condition.protocol.system_prompt,
        task.prompt
        + "\n\nReference implementation (teacher only):\n"
        + task.reference_implementation,
    )
    tokens = result.context_tokens
    if len(tokens) > config.max_context_tokens:
        raise ValueError("Student history exceeds the training context capacity")
    targets = [0] * (len(tokens) - 1)
    old = [0.0] * len(targets)
    advantages = [0.0] * len(targets)
    differences: list[float] = []
    spans = []
    for index, step in enumerate(result.trajectory):
        call = step.model_call
        assert call is not None and call.generation.logprobs is not None
        raw, rollout = call.generation.tokens, call.generation.logprobs
        if not raw:
            raise ValueError("Empty sampled generation cannot supply supervision")
        start, end = call.context_offset, call.context_offset + len(raw)
        teacher_tokens = [*privileged, *tokens[len(initial) : end]]
        teacher_start = len(privileged) + start - len(initial)
        if len(teacher_tokens) > config.max_context_tokens:
            raise ValueError(
                "Privileged teacher history exceeds the scoring context capacity"
            )
        scores = score(teacher_tokens, index, teacher_start)
        teacher = scores.logprobs[teacher_start:]
        if len(teacher) != len(raw) or any(
            value is None
            or type(value) not in (int, float)
            or not math.isfinite(value)
            or value > 0
            for value in teacher
        ):
            raise ValueError(
                "Teacher sampled-token logprobs must be finite and non-positive"
            )
        signal = []
        for student_lp, teacher_lp in zip(rollout, teacher, strict=True):
            assert teacher_lp is not None
            difference = student_lp - teacher_lp
            advantage = config.distillation_coefficient * (teacher_lp - student_lp)
            if not math.isfinite(advantage):
                raise ValueError("Distillation advantage is not finite")
            differences.append(difference)
            signal.append(advantage)
        targets[start - 1 : end - 1] = raw
        old[start - 1 : end - 1] = rollout
        advantages[start - 1 : end - 1] = signal
        spans.append(
            {
                "step": index,
                "student_start": start,
                "student_end": end,
                "teacher_start": teacher_start,
                "teacher_end": len(teacher_tokens),
                "target_start": start - 1,
                "request_id": scores.request_id,
            }
        )
    if not differences:
        raise ValueError("No sampled tokens are available for supervision")
    with np.errstate(over="ignore", invalid="ignore"):
        old_values = np.asarray(old, dtype=np.float32)
        advantage_values = np.asarray(advantages, dtype=np.float32)
    if not np.isfinite(old_values).all() or not np.isfinite(advantage_values).all():
        raise ValueError("Distillation values overflow the training tensor encoding")
    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": np.asarray(targets, dtype=np.int64),
            "logprobs": old_values,
            "advantages": advantage_values,
        },
    )
    return datum, {
        "loss_fn": "importance_sampling",
        "auto_shift": False,
        "distillation_coefficient": config.distillation_coefficient,
        "supervised_tokens": len(differences),
        "nonzero_signal_tokens": int(np.count_nonzero(advantage_values)),
        "spans": spans,
        "sampled_token_reverse_kl": math.fsum(differences) / len(differences),
        "kl_denominator": "supervised sampled student tokens",
    }


def validate_history(
    record: dict[str, Any],
    task: Task,
    encoding: Encoding,
    config: ProtocolConfig,
) -> None:
    """Reconstruct only framing; sampled generations remain their raw token IDs."""
    if (
        record["encoding"] != encoding.identity
        or record["encoding_details"] != encoding.describe()
    ):
        raise ValueError("Trajectory encoding identity differs")
    context = encoding.prompt(config.system_prompt, task.prompt)
    for index, step in enumerate(record["trajectory"]):
        call = step["model_call"]
        if call is None or call["context_offset"] != len(context):
            raise ValueError("Trajectory has missing calls or invalid context offsets")
        raw = call["generation"]
        generation = Generation(
            tokens=tuple(raw["tokens"]),
            stop_reason=raw["stop_reason"],
            logprobs=tuple(raw["logprobs"]) if raw["logprobs"] is not None else None,
            request_id=raw["request_id"],
            source=raw["source"],
        )
        try:
            generation.validate(SamplingParams(**call["sampling"]))
        except ModelResponseError as error:
            raise ValueError(
                f"Invalid trajectory sampling evidence: {error}"
            ) from error
        reply = encoding.reply(generation.tokens)
        if call["text"] != encoding.decode(generation.tokens) or call[
            "reply"
        ] != asdict(reply):
            raise ValueError("Trajectory text differs from its raw generation")
        action = (
            protocol.parse_action(reply.action_text, generation.stop_reason == "length")
            if reply.thinking_complete
            else InvalidAction("Native thinking did not reach its end marker")
        )
        if step["action"] != asdict(action):
            raise ValueError("Trajectory action differs from the protocol")
        context.extend(generation.tokens)
        feedback: str | None
        if isinstance(action, InvalidAction):
            if step["observation"] is not None:
                raise ValueError(
                    "An invalid action cannot have an executed observation"
                )
            feedback = protocol.render_invalid(action, config.invalid_action_format)
        else:
            if step["observation"] is None:
                raise ValueError("Executable action has no observation")
            observation = Observation(**step["observation"])
            submitted = protocol.submission_text(observation, config.submission_marker)
            if submitted is not None and index != len(record["trajectory"]) - 1:
                raise ValueError("Trajectory continued after an explicit submission")
            feedback = (
                None
                if submitted is not None
                else protocol.render_observation(observation, config)
            )
        if step["feedback"] != feedback:
            raise ValueError("Trajectory feedback differs from its observation")
        if feedback is not None:
            if not (
                index == len(record["trajectory"]) - 1
                and record["status"] not in ("completed", "submitted")
                and context == record["context_tokens"]
            ):
                context.extend(encoding.continuation(generation.tokens, feedback))
    if context != record["context_tokens"]:
        raise ValueError(
            "Trajectory context does not preserve the complete causal history"
        )
