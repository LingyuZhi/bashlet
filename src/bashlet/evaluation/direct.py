"""One direct code generation using the experiment's workspace and grader."""

import re
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from ..config import DirectCodeConfig, ExperimentConfig
from ..episode import _sample, _SamplerError, _verify_isolation
from ..models import (
    Code,
    ContextExhausted,
    EpisodeResult,
    FunctionGrader,
    InvalidAction,
    Isolation,
    Model,
    ModelBackend,
    ModelResponseError,
    Step,
    Task,
)
from ..sandbox import Sandbox
from ..spend import BudgetRefused
from ..workspace import capture_workspace


def run_direct_code(
    task: Task,
    model: Model,
    workspace: Path,
    run_dir: Path,
    condition: ExperimentConfig,
    coding: DirectCodeConfig,
    save: Callable[[EpisodeResult], None],
) -> EpisodeResult:
    """Sample once; retain raw tokens and grade only the extracted Python file."""
    if not isinstance(task.grader, FunctionGrader):
        raise ValueError("Direct coding requires a Python function grader")
    sandbox = Sandbox(condition.environment)
    backend = model if isinstance(model, ModelBackend) else None
    result = EpisodeResult(
        run_dir=run_dir,
        workspace=workspace,
        limits=sandbox.limits,
        isolation=Isolation(False, reason="Not yet checked"),
        policy=model.identity,
        encoding=model.encoding.identity,
        encoding_details=model.encoding.describe(),
        budget=condition.budget,
    )
    started = time.monotonic()
    try:
        result.initial_workspace_manifest = capture_workspace(workspace, run_dir)
        save(result)
        result.isolation = _verify_isolation(sandbox, run_dir)
        if not result.isolation.passed:
            result.status, result.reason = (
                "infrastructure_error",
                result.isolation.reason,
            )
            return result
        if backend is not None:
            try:
                backend.probe()
            except (ModelResponseError, ContextExhausted, BudgetRefused):
                raise
            except Exception as error:
                raise _SamplerError(f"Capability probe failed: {error}") from error
        started = time.monotonic()
        instruction = task.prompt
        if task.origin is not None and task.origin.dataset == "google-research/mbpp":
            instruction = instruction.split("\nRun `", maxsplit=1)[0]
        user = (
            Environment(undefined=StrictUndefined)
            .from_string(coding.user_prompt)
            .render(instruction=instruction, function_name=task.grader.function_name)
        )
        context = result.context_tokens
        context.extend(model.encoding.prompt(coding.system_prompt, user))
        remaining = condition.budget.max_generated_tokens
        if backend is not None:
            remaining = min(remaining, backend.max_context_tokens - len(context))
        if remaining <= 0:
            raise ContextExhausted("Direct coding context exhausted")
        sampling = replace(
            condition.model.sampling,
            max_tokens=min(remaining, condition.model.sampling.max_tokens),
        )
        call = _sample(model, context, sampling)
        context.extend(call.generation.tokens)
        result.usage.steps = 1
        result.usage.generated_tokens = len(call.generation.tokens)
        result.usage.elapsed_seconds = time.monotonic() - started
        assert call.reply is not None
        match = re.fullmatch(
            r"\s*```python[ \t]*\r?\n(.*?)\r?\n```\s*",
            call.reply.action_text,
            flags=re.DOTALL,
        )
        reason = ""
        if call.generation.stop_reason != "stop":
            reason = "Direct coding generation was truncated"
        elif not call.reply.thinking_complete:
            reason = "Direct coding thinking was incomplete"
        elif match is None or "```" in match[1] or not match[1].strip():
            reason = (
                "Expected exactly one nonempty python fence and no surrounding text"
            )
        elif result.usage.elapsed_seconds > condition.budget.max_seconds:
            reason = "Direct coding time budget exhausted"
        action: Code | InvalidAction
        if reason:
            action = InvalidAction(reason)
            result.status, result.reason = "model_failure", reason
        else:
            assert match is not None
            action = Code(task.grader.module_path, match[1] + "\n")
        # Seal the raw answer before interruptible file writes.
        result.trajectory.append(Step(action, model_call=call))
        save(result)
        if isinstance(action, Code):
            target = workspace / action.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(action.text)
            result.status = "submitted"
        result.trajectory[0] = replace(
            result.trajectory[0],
            workspace_manifest=capture_workspace(workspace, run_dir),
            usage=replace(result.usage),
        )
        return result
    except BudgetRefused as error:
        result.status, result.reason = "budget_exhausted", str(error)
        result.budget_kind = "spend"
    except ContextExhausted as error:
        result.status, result.reason = "context_exhausted", str(error)
    except ModelResponseError as error:
        result.status, result.reason = "model_error", str(error)
        result.rejected_generation = error.generation
    except (OSError, _SamplerError) as error:
        result.status, result.reason = "infrastructure_error", str(error)
    except KeyboardInterrupt:
        result.status, result.reason = "interrupted", "Direct coding interrupted"
    finally:
        result.usage.elapsed_seconds = time.monotonic() - started
        if backend is not None:
            result.policy_details = backend.describe()
        save(result)
    return result
