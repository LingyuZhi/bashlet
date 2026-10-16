"""Resume committed interaction checkpoints without replaying sampled actions."""

import json
from pathlib import Path
from typing import Any

import pytest
from adapters import ScriptedModel, bash, submit, turn_number, write_file
from test_mbpp import double_row, prepare

from bashlet import CommandGrader, Task, run_experiment
from bashlet.config import load_config
from bashlet.evaluation import evaluate
from bashlet.harness.models import EpisodeResult


class PowerLoss(BaseException):
    pass


def checkpoint(
    tmp_path: Path, *, max_steps: int = 3, before_action: bool = False
) -> tuple[Task, Path, dict[str, Any]]:
    task = Task(
        "resume/once",
        "Write DONE once, then submit.",
        {"input.txt": "original"},
        CommandGrader("test -f DONE", {}),
        implementation_paths=("DONE",),
    )
    model = ScriptedModel(lambda _: bash("printf once >> DONE"))

    def interrupt(result: EpisodeResult) -> None:
        if len(result.trajectory) == (0 if before_action else 1):
            raise PowerLoss

    with pytest.raises(PowerLoss):
        run_experiment(
            task=task,
            model=model,
            output_dir=tmp_path,
            overrides=[f"budget.max_steps={max_steps}"],
            on_update=interrupt,
        )
    run_dir = next(tmp_path.iterdir())
    record = json.loads((run_dir / "run.json").read_text())
    assert record["status"] == "running"
    return task, run_dir, record


def test_resume_keeps_raw_history_and_does_not_execute_prior_action(
    tmp_path: Path,
) -> None:
    task, run_dir, old = checkpoint(tmp_path)
    model = ScriptedModel(lambda _: submit())
    result = run_experiment(
        task=task,
        model=model,
        output_dir=tmp_path,
        overrides=["budget.max_steps=3"],
        resume_run_dir=run_dir,
        assess_first_implementation=True,
    )
    assert result.passed, result.reason
    assert result.run_dir == run_dir
    assert (result.workspace / "DONE").read_text() == "once"
    assert model.prompts[0] == tuple(old["context_tokens"])
    assert result.usage.steps == 2
    call = result.trajectory[0].model_call
    assert call is not None
    assert call.generation.tokens == tuple(
        old["trajectory"][0]["model_call"]["generation"]["tokens"]
    )
    assert result.usage.generated_tokens == (
        old["usage"]["generated_tokens"] + len(model.generations[0].tokens)
    )
    assert result.first_implementation is not None
    assert result.first_implementation.grade is not None
    assert result.first_implementation.grade.passed


def test_resume_preserves_consumed_step_budget(tmp_path: Path) -> None:
    task, run_dir, _ = checkpoint(tmp_path, max_steps=1)
    model = ScriptedModel(lambda _: submit())
    result = run_experiment(
        task=task,
        model=model,
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
        resume_run_dir=run_dir,
    )
    assert result.status == "budget_exhausted"
    assert result.usage.steps == 1
    assert not model.prompts


def test_resume_rejects_workspace_changes_before_sampling(tmp_path: Path) -> None:
    task, run_dir, _ = checkpoint(tmp_path)
    (run_dir / "workspace/DONE").write_text("unrecorded")
    model = ScriptedModel(lambda _: submit())
    with pytest.raises(ValueError, match="workspace"):
        run_experiment(
            task=task,
            model=model,
            output_dir=tmp_path,
            overrides=["budget.max_steps=3"],
            resume_run_dir=run_dir,
        )
    assert not model.prompts


def test_resume_initial_checkpoint_verifies_original_task_files(tmp_path: Path) -> None:
    task, run_dir, old = checkpoint(tmp_path, before_action=True)
    model = ScriptedModel(
        lambda context: (
            submit() if turn_number(context) > 1 else bash("printf once > DONE")
        )
    )
    result = run_experiment(
        task=task,
        model=model,
        output_dir=tmp_path,
        overrides=["budget.max_steps=3"],
        resume_run_dir=run_dir,
    )
    assert result.passed
    assert result.started_at == old["started_at"]
    assert (result.workspace / "input.txt").read_text() == "original"


def test_resume_can_charge_elapsed_time_without_resetting_budget(
    tmp_path: Path,
) -> None:
    task, run_dir, _ = checkpoint(tmp_path)
    model = ScriptedModel(lambda _: submit())
    result = run_experiment(
        task=task,
        model=model,
        output_dir=tmp_path,
        overrides=["budget.max_steps=3"],
        resume_run_dir=run_dir,
        resume_elapsed_seconds=1000.0,
    )
    assert result.status == "budget_exhausted"
    assert "max_seconds" in result.reason
    assert not model.prompts


def test_evaluator_continues_the_same_running_episode(tmp_path: Path) -> None:
    prepared = prepare(tmp_path, [double_row(511)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "resume-one",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511"],
            }
        )
    )
    condition = load_config(overrides=["budget.max_steps=3"])

    def stop_after_checkpoint(context: str) -> str:
        if turn_number(context) == 1:
            return write_file("def double(n): return n * 2\n")
        raise PowerLoss

    model = ScriptedModel(stop_after_checkpoint)
    folder = tmp_path / "evaluation"
    with pytest.raises(PowerLoss):
        evaluate(
            plan_path=plan,
            snapshot=model.identity,
            evidence_kind="test-double",
            condition=condition,
            output_dir=tmp_path,
            run_dir=folder,
            model=model,
        )
    previous = json.loads((folder / "evaluation.json").read_text())
    episode_path = folder / previous["tasks"][0]["run"] / "run.json"
    saved = json.loads(episode_path.read_text())
    resumed = ScriptedModel(lambda _: submit())
    path = evaluate(
        plan_path=plan,
        snapshot=resumed.identity,
        evidence_kind="test-double",
        condition=condition,
        output_dir=tmp_path,
        run_dir=folder,
        model=resumed,
        resume=True,
        resume_running_episodes=True,
    )
    result = json.loads(path.read_text())
    assert result["status"] == "complete"
    assert result["summary"]["passed_tasks"] == 1
    assert result["tasks"][0]["run"] == previous["tasks"][0]["run"]
    assert len(list((folder / "episodes").iterdir())) == 1
    assert resumed.prompts[0] == tuple(saved["context_tokens"])
