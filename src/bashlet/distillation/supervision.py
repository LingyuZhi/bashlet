"""Align frozen-teacher probabilities to the student's original causal history."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import ExperimentConfig
from ..models import Encoding, ExperimentResult, Task

if TYPE_CHECKING:
    import pytrio as trio

    from . import OPSDConfig, TeacherScores


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
        "spans": spans,
        "sampled_token_reverse_kl": math.fsum(differences) / len(differences),
        "kl_denominator": "supervised sampled student tokens",
    }
