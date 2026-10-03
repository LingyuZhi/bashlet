"""Dual evaluation behavior at the experiment and CLI boundaries."""

import json
import os
from pathlib import Path

import pytest
import yaml
from adapters import ScriptedModel
from test_cli import cli
from test_experiment import sum_task
from test_mbpp import double_row, prepare
from test_monitoring import external_sdks

from bashlet import run_experiment
from bashlet.config import DirectCodeConfig


def test_direct_code_is_written_once_and_independently_graded_without_feedback(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(lambda _: "```python\ndef solve(numbers): return 3\n```")
    result = run_experiment(
        task=sum_task(),
        model=model,
        output_dir=tmp_path,
        direct_code=DirectCodeConfig(
            version="direct-code-v1",
            system_prompt="Write Python. Return exactly one python code fence.",
            user_prompt="{{ instruction }}\nDefine {{ function_name }}.",
            extraction="single-python-fence-v1",
        ),
    )

    assert not result.passed
    assert result.grade is not None and result.grade.passed_cases == 1
    assert len(model.prompts) == 1
    assert "feedback_tests.py" not in model.encoding.render(model.prompts[0])
    assert (result.run_dir / "submission/solution.py").read_text() == (
        "def solve(numbers): return 3\n"
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["evaluation_track"] == "coding"
    assert record["trajectory"][0]["model_call"]["generation"]["tokens"] == list(
        model.generations[0].tokens
    )
    assert record["trajectory"][0]["observation"] is None


@pytest.mark.parametrize(
    "reply",
    [
        "```bash\ntouch escaped\n```",
        "```python\n\n```",
        "```python\ndef solve(numbers): return sum(numbers)\n```\nExtra explanation",
        "```python\ndef solve(numbers): return sum(numbers)\n```\n```python\npass\n```",
    ],
)
def test_direct_code_rejects_ambiguous_answers_without_execution_or_repair(
    tmp_path: Path,
    reply: str,
) -> None:
    model = ScriptedModel(lambda _: reply)
    result = run_experiment(
        task=sum_task(),
        model=model,
        output_dir=tmp_path,
        direct_code=DirectCodeConfig(
            "direct-code-v1",
            "Return Python.",
            "{{ instruction }}",
            "single-python-fence-v1",
        ),
    )
    assert result.status == "model_failure" and result.grade is None
    assert len(model.prompts) == 1
    assert not (result.workspace / "escaped").exists()
    assert not (result.workspace / "solution.py").exists()


def test_direct_code_never_accepts_a_truncated_answer(tmp_path: Path) -> None:
    model = ScriptedModel(
        lambda _: "```python\ndef solve(numbers): return sum(numbers)\n```"
    )
    result = run_experiment(
        task=sum_task(),
        model=model,
        output_dir=tmp_path,
        overrides=["model.sampling.max_tokens=5"],
        direct_code=DirectCodeConfig(
            "direct-code-v1",
            "Return Python.",
            "{{ instruction }}",
            "single-python-fence-v1",
        ),
    )
    assert not result.passed and result.grade is None
    assert "truncated" in result.reason
    assert len(model.requests) == 1


def test_dual_evaluation_reuses_completed_seed_results_and_keeps_two_denominators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_sdks(tmp_path, monkeypatch)
    prepared = prepare(tmp_path, [double_row(511)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "one-family",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511"],
            }
        )
    )
    policy = json.loads(
        (Path(__file__).parents[1] / "configs/sft-acceptance-v1.json").read_text()
    )
    policy.update(
        harness_plan=str(plan),
        coding_plan=str(plan),
        variant_plan=None,
        harness_counts={"short_program": 1},
        coding_count=1,
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy))
    config = tmp_path / "model.yaml"
    config.write_text(
        yaml.safe_dump({"model": {"class": "adapters.DirectEvaluationModel"}})
    )
    monitor = tmp_path / "monitor.yaml"
    monitor.write_text("group: dual-evaluation\nmode: offline\n")
    previous = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        str(Path(__file__).parent.resolve()) + os.pathsep + previous
    )
    try:
        args = [
            "evaluate-acceptance",
            str(policy_path),
            "--snapshot",
            "test/direct-v1",
            "--evidence-kind",
            "test-double",
            "-c",
            str(config),
            "--output-dir",
            str(tmp_path / "dual"),
            "--monitor-config",
            str(monitor),
        ]
        first = cli(*args, cwd=tmp_path, timeout=60)
        assert first.returncode == 0, first.stdout + first.stderr
        path = Path(first.stdout.strip().splitlines()[-1])
        original = json.loads(path.read_text())
        # A crash after the track finished, before the suite reference was sealed.
        pending = original["runs"]["coding"].pop("4")
        original["active_attempt"] = {
            "track": "coding",
            "seed": 4,
            "path": pending["path"],
        }
        path.write_text(json.dumps(original))
        repeated = cli(*args, "--resume", str(path), cwd=tmp_path, timeout=60)
        assert repeated.returncode == 0, repeated.stdout + repeated.stderr
        confirmed = cli(
            *args,
            "--resume",
            str(path),
            "--stage",
            "confirmation",
            cwd=tmp_path,
            timeout=60,
        )
        assert confirmed.returncode == 0, confirmed.stdout + confirmed.stderr
        result = json.loads(path.read_text())
        assert result["runs"]["harness"]["4"] == original["runs"]["harness"]["4"]
        assert result["runs"]["coding"]["4"] == pending
        assert result["summary"]["harness"]["task_families"] == 1
        assert result["summary"]["coding"]["task_families"] == 1
        assert result["summary"]["harness"]["attempts"] == 3
        assert result["summary"]["coding"]["passed_attempts"] == 3
        assert result["summary"]["overlap_family_ids"] == ["mbpp/511"]
        assert len(list(path.parent.glob("coding/*/evaluation.json"))) == 3
        coding = json.loads(Path(result["runs"]["coding"]["4"]["path"]).read_text())
        assert coding["track"] == "coding"
        assert (
            "feedback_tests.py"
            not in coding["comparison"]["direct_code"]["system_prompt"]
        )
        remote = json.loads(
            (
                Path(result["runs"]["coding"]["4"]["path"]).parent / "remote.json"
            ).read_text()
        )
        assert remote["rows"][-1]["eval/passed_tasks"] == 1
        assert "eval/first_implementation_pass_rate" not in remote["rows"][-1]
        for key, value in (
            ("timeout_seconds", "125"),
            ("max_retries", "1"),
            ("retry_delay_seconds", "125"),
        ):
            changed = cli(
                *args,
                "--resume",
                str(path),
                "--set",
                f"model.{key}={value}",
                cwd=tmp_path,
                timeout=30,
            )
            assert changed.returncode == 2 and "comparison conditions" in changed.stderr
        assert len(list(path.parent.glob("coding/*/evaluation.json"))) == 3
    finally:
        os.environ["PYTHONPATH"] = previous
