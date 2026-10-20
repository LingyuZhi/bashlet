"""Evaluation behavior through the experiment and CLI seams."""

import json
from pathlib import Path

import pytest
import yaml
from adapters import bash, replay, submit, write_file
from test_cli import cli
from test_experiment import sum_task
from test_mbpp import double_row, prepare

from bashlet import run_experiment


def test_first_implementation_is_graded_from_its_snapshot_after_the_episode(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash("ls"),
                write_file("def solve(numbers): return 3\n"),
                bash("python3 feedback_tests.py"),
                write_file("def solve(numbers): return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
        assess_first_implementation=True,
    )

    assert result.passed
    first = result.first_implementation
    assert first is not None
    assert first.status == "graded"
    assert first.formed_at_step == 1
    assert first.grade is not None and not first.grade.passed
    assert first.grade.passed_cases == 1
    assert first.snapshot is not None
    assert (result.run_dir / first.snapshot.path / "solution.py").read_text() == (
        "def solve(numbers): return 3\n"
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["first_implementation"]["grade"]["passed"] is False
    assert record["grade"]["passed"] is True
    assert all(
        "wrong_answer" not in (step.feedback or "") for step in result.trajectory
    )


def test_evaluation_preserves_the_planned_denominator_and_failure_categories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    modes = ["success", "wrong", "model", "protocol", "budget", "service", "success"]
    rows = [
        double_row(511 + index) | {"prompt": mode} for index, mode in enumerate(modes)
    ]
    prepared = prepare(tmp_path, rows)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "fixed-seven",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": [f"mbpp/{511 + index}" for index in range(7)],
            }
        )
    )
    config = tmp_path / "evaluation.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {"class": "adapters.EvaluationModel"},
                "budget": {"max_steps": 3},
            }
        )
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.resolve()))
    process = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        "test/evaluation-v1",
        "--evidence-kind",
        "test-double",
        "-c",
        str(config),
        "--output-dir",
        str(tmp_path / "evaluation"),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    records = list((tmp_path / "evaluation").glob("*/evaluation.json"))
    assert len(records) == 1, process.stdout + process.stderr
    record = json.loads(records[0].read_text())
    assert record["status"] == "incomplete"
    assert record["summary"]["planned_tasks"] == 7
    assert record["summary"]["passed_tasks"] == 1
    assert record["summary"]["final_pass_rate"] == 1 / 7
    assert [row["outcome"] for row in record["tasks"]] == [
        "passed",
        "model_failure",
        "model_error",
        "protocol_error",
        "budget_exhausted",
        "infrastructure_error",
        "not_started",
    ]
    assert record["summary"]["missing_task_ids"] == ["mbpp/516", "mbpp/517"]
    assert record["summary"]["first_implementation_passed"] == 1
    assert record["summary"]["first_implementation_pass_rate"] == 1 / 7
    assert record["summary"]["recovery"]["task_ids"] == ["mbpp/512"]
    assert record["evidence_kind"] == "test-double"
    assert "1/7" in process.stdout and "incomplete" in process.stdout
    for row in record["tasks"][:-1]:
        episode = json.loads((records[0].parent / row["run"] / "run.json").read_text())
        assert episode["config"]["budget"]["max_steps"] == 3
        assert episode["task_sha256"] == row["task_sha256"]


@pytest.mark.parametrize("fail_before_identity", [False, True])
def test_evaluation_remote_failures_keep_missing_tasks_and_all_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_before_identity: bool,
) -> None:
    prepared = prepare(tmp_path, [double_row(511), double_row(512), double_row(513)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "remote-failure",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511", "mbpp/512", "mbpp/513"],
            }
        )
    )
    journal = tmp_path / "requests.sqlite3"
    config = tmp_path / "remote.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "adapters.RemoteEvaluationModel",
                    "journal_path": str(journal),
                    "fail_before_identity": fail_before_identity,
                }
            }
        )
    )
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.resolve()))
    process = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        "pytrio/Qwen/Qwen3.5-4B@unversioned-base",
        "--evidence-kind",
        "test-double",
        "-c",
        str(config),
        "--output-dir",
        str(tmp_path / "evaluation"),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    records = list((tmp_path / "evaluation").glob("*/evaluation.json"))
    assert len(records) == 1, process.stdout + process.stderr
    record = json.loads(records[0].read_text())
    assert record["status"] == "incomplete"
    if fail_before_identity:
        assert record["summary"]["missing_task_ids"] == [
            "mbpp/511",
            "mbpp/512",
            "mbpp/513",
        ]
        assert record["tasks"][0]["outcome"] == "infrastructure_error"
        assert record["requests"]["pending_count"] == 1
        assert len(record["requests"]["request_ids"]) == 1
        return
    assert record["summary"]["final_pass_rate"] == 1 / 3
    assert record["summary"]["missing_task_ids"] == ["mbpp/512", "mbpp/513"]
    assert record["tasks"][1]["outcome"] == "infrastructure_error"
    assert record["tasks"][2]["outcome"] == "not_started"
    assert record["requests"]["journal"] == str(journal)
    assert record["requests"]["pending_count"] == 1
    assert len(record["requests"]["request_ids"]) == 4


@pytest.mark.parametrize("source", ["", "def solve(:\n"])
def test_empty_or_invalid_first_implementation_counts_even_without_submission(
    tmp_path: Path, source: str
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([write_file(source)]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
        assess_first_implementation=True,
    )
    assert result.status == "budget_exhausted" and not result.passed
    assert result.budget_kind == "interaction"
    assert result.first_implementation is not None
    assert result.first_implementation.formed_at_step == 0
    grade = result.first_implementation.grade
    assert grade is not None and not grade.passed


def test_comparison_conditions_ignore_snapshot_but_track_sampling_and_preserve_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare(tmp_path, [double_row(511)])
    plan = tmp_path / "plan.json"
    plan_data = {
        "version": "development-evaluation-v1",
        "name": "compare",
        "manifest": str(prepared / "manifest.json"),
        "split": "dev",
        "task_ids": ["mbpp/511"],
    }
    plan.write_text(json.dumps(plan_data))
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model": {"class": "adapters.EvaluationModel"}}))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.resolve()))
    output = tmp_path / "evaluation"
    base = [
        "evaluate",
        str(plan),
        "--evidence-kind",
        "test-double",
        "-c",
        str(config),
        "--output-dir",
        str(output),
    ]
    records = []
    for identity, temperature in [("base", 1.0), ("sft", 1.0), ("sft", 0.5)]:
        process = cli(
            *base,
            "--snapshot",
            identity,
            "--set",
            f"model.identity={identity}",
            "--set",
            f"model.sampling.temperature={temperature}",
            cwd=tmp_path,
            timeout=30,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        record_path = Path(process.stdout.strip().splitlines()[-1])
        records.append(json.loads(record_path.read_text()))
    assert records[0]["comparison_sha256"] == records[1]["comparison_sha256"]
    assert records[1]["comparison_sha256"] != records[2]["comparison_sha256"]
    assert len(list(output.glob("*/evaluation.json"))) == 3
    wrong = cli(*base, "--snapshot", "not-loaded", cwd=tmp_path)
    assert wrong.returncode == 2 and "Snapshot mismatch" in wrong.stderr
    plan.write_text(json.dumps(plan_data | {"split": "test"}))
    heldout = cli(*base, "--snapshot", "test/evaluation-v1", cwd=tmp_path)
    assert heldout.returncode == 2 and "split=dev" in heldout.stderr
    assert len(list(output.glob("*/evaluation.json"))) == 3


def test_interrupted_evaluation_retains_unattempted_tasks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = prepare(
        tmp_path, [double_row(511) | {"prompt": "interrupt"}, double_row(512)]
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "interrupted",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511", "mbpp/512"],
            }
        )
    )
    config = tmp_path / "model.yaml"
    config.write_text(yaml.safe_dump({"model": {"class": "adapters.EvaluationModel"}}))
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent.resolve()))
    process = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        "test/evaluation-v1",
        "--evidence-kind",
        "test-double",
        "-c",
        str(config),
        "--output-dir",
        str(tmp_path / "evaluation"),
        cwd=tmp_path,
        timeout=30,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    record = json.loads(Path(process.stdout.strip().splitlines()[-1]).read_text())
    assert record["summary"]["planned_tasks"] == 2
    assert record["summary"]["missing_task_ids"] == ["mbpp/511", "mbpp/512"]
    assert [row["outcome"] for row in record["tasks"]] == ["interrupted", "not_started"]
