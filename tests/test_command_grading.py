"""Hidden test-command grading through the experiment seam."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from adapters import bash, replay, submit

from bashlet import CommandGrader, EpisodeResult, ExperimentResult, Task, run_experiment


def package_task() -> Task:
    return Task(
        task_id="manual/package-repair",
        prompt="Fix doubled(n) in the package. Check tests/test_public.py and submit.",
        initial_files={
            "package/__init__.py": "",
            "package/math.py": "def doubled(n):\n    return n\n",
            "package/api.py": "from .math import doubled\n",
            "tests/test_public.py": (
                "from package.api import doubled\n"
                "def test_doubled(): assert doubled(2) == 4\n"
            ),
        },
        grader=CommandGrader(
            command="python3 -m pytest -q tests/test_hidden.py",
            hidden_files={
                "tests/test_hidden.py": (
                    "from package.api import doubled\n"
                    "def test_hidden():\n"
                    "    assert doubled(-7) == -14, 'hidden-marker-6c19'\n"
                    "    assert doubled(0) == 0\n"
                ),
            },
        ),
        implementation_paths=("package/math.py", "package/api.py"),
        feedback_command="pytest -q tests/test_public.py",
    )


def test_multifile_bug_is_repaired_verified_and_graded_with_hidden_tests(
    tmp_path: Path,
) -> None:
    task = package_task()
    assert isinstance(task.grader, CommandGrader)
    assert task.feedback_command is not None

    def check_progress(episode: EpisodeResult) -> None:
        if not isinstance(episode, ExperimentResult):
            assert not (episode.run_dir / "private/grading-workspace").exists()

    result = run_experiment(
        task=task,
        model=replay(
            [
                bash(task.feedback_command),
                bash("sed -i 's/return n$/return 2 * n/' package/math.py"),
                bash(task.feedback_command),
                submit(),
            ]
        ),
        output_dir=tmp_path,
        on_update=check_progress,
    )

    assert result.status == "completed" and result.passed, result.reason
    failed, _, verified, _ = result.trajectory
    assert failed.observation is not None
    assert failed.observation.status == "command_failed"
    assert "assert 2 == 4" in failed.observation.output
    assert verified.observation is not None and verified.observation.status == "ok"
    assert failed.verification and verified.verification
    assert len(result.repair_attempts) == 1 and result.repair_attempts[0].passed
    assert result.grade is not None and result.grade.passed_cases == 1
    assert result.submission is not None
    assert "tests/test_hidden.py" not in result.submission.manifest
    assert not (result.workspace / "tests/test_hidden.py").exists()
    assert (
        result.run_dir / "private/grading-workspace/tests/test_hidden.py"
    ).read_text() == task.grader.hidden_files["tests/test_hidden.py"]


def test_passing_only_public_tests_fails_grading_without_leaking_hidden_material(
    tmp_path: Path,
) -> None:
    task = package_task()
    assert task.feedback_command is not None
    model = replay(
        [
            bash("sed -i 's/return n$/return 4/' package/math.py"),
            bash(task.feedback_command),
            bash("cat ../private/task.json ../run.json ../private/grading-workspace"),
            submit(),
        ]
    )
    result = run_experiment(task=task, model=model, output_dir=tmp_path)

    assert result.status == "completed" and not result.passed
    public = result.trajectory[1].observation
    assert public is not None and public.status == "ok" and "1 passed" in public.output
    denied = result.trajectory[2].observation
    assert denied is not None and denied.status == "command_failed"
    assert result.grade is not None and result.grade.cases[0].status == "command_failed"
    assert all(
        "tests/test_hidden.py" not in step.workspace_manifest
        for step in result.trajectory
    )
    assert not (result.workspace / "tests/test_hidden.py").exists()
    assert result.submission is not None
    assert "tests/test_hidden.py" not in result.submission.manifest
    student_context = model.encoding.decode(result.context_tokens)
    record = (result.run_dir / "run.json").read_text()
    assert "hidden-marker-6c19" not in student_context + record
    assert all(
        b"hidden-marker-6c19" not in blob.read_bytes()
        for blob in (result.run_dir / "blobs").iterdir()
    )
    assert json.loads(record)["grading_protocol"] == "test-command-v1"


@pytest.mark.parametrize(
    ("command", "passed", "status"),
    [
        ("printf 'failed'; exit 0", True, "passed"),
        ("printf 'passed'; exit 9", False, "command_failed"),
        ("head -c 20000 /dev/zero", True, "passed"),
        ("sleep 5", False, "timeout"),
    ],
)
def test_test_command_grade_uses_exit_status_even_when_output_is_truncated(
    tmp_path: Path, command: str, passed: bool, status: str
) -> None:
    task = replace(package_task(), grader=CommandGrader(command, {}))
    result = run_experiment(
        task=task,
        model=replay([submit()]),
        output_dir=tmp_path,
        overrides=["environment.timeout_seconds=0.3", "environment.output_bytes=128"],
    )
    assert result.status == "completed" and result.passed is passed, result.reason
    assert result.grade is not None
    assert result.grade.total_cases == 1 and result.grade.cases[0].status == status


@pytest.mark.parametrize("path", ["../outside", "/tmp/outside", "a/../outside"])
def test_hidden_grading_files_are_validated_before_creating_a_run(
    tmp_path: Path, path: str
) -> None:
    task = replace(package_task(), grader=CommandGrader("true", {path: "hidden"}))
    with pytest.raises(ValueError, match="workspace path"):
        run_experiment(
            task=task, model=replay([submit()]), output_dir=tmp_path / "runs"
        )
    assert not (tmp_path / "runs").exists()


def test_grading_injects_authoritative_hidden_files_into_only_the_snapshot_copy(
    tmp_path: Path,
) -> None:
    task = package_task()
    result = run_experiment(
        task=task,
        model=replay(
            [
                bash("sed -i 's/return n$/return 2 * n/' package/math.py"),
                bash(
                    "printf 'raise RuntimeError(\"student decoy\")' "
                    "> tests/test_hidden.py"
                ),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )
    assert result.passed, result.reason
    assert result.submission is not None
    for workspace in (result.workspace, result.run_dir / result.submission.path):
        assert (workspace / "tests/test_hidden.py").read_text() == (
            'raise RuntimeError("student decoy")'
        )
