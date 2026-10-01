"""Sandbox capabilities through the experiment seam and real execution."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from adapters import bash, replay, submit

from bashlet import FunctionGrader, GradingCase, Task, run_experiment


def workspace_task() -> Task:
    return Task(
        task_id="manual/sandbox-tools",
        prompt="Verify the existing program and submit the workspace.",
        initial_files={
            "main.py": "def answer(): return 42\n",
            "tests/test_public.py": (
                "from main import answer\ndef test_answer(): assert answer() == 42\n"
            ),
        },
        grader=FunctionGrader("main.py", "answer", (GradingCase((), 42),)),
    )


def test_common_tools_and_pytest_work_without_polluting_the_submission(
    tmp_path: Path,
) -> None:
    task = workspace_task()
    result = run_experiment(
        task=task,
        model=replay(
            [
                bash(
                    "set -e\n"
                    'test -n "$HOME"\n'
                    'printf writable > "$HOME/check"\n'
                    "whoami\n"
                    "printf '3 4\\n' | awk '{print $1 + $2}'\n"
                    "printf 'old\\n' | sed 's/old/new/' | grep '^new$'\n"
                    'git init -q "$HOME/repo"\n'
                    'git -C "$HOME/repo" config user.name\n'
                    "pytest --version\n"
                    "pytest -q\n"
                    "python3 -m pytest -q"
                ),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    observation = result.trajectory[0].observation
    assert observation is not None and observation.status == "ok", observation
    assert "agent\n7\nnew\n" in observation.output
    assert "pytest 9.1.1" in observation.output
    assert observation.output.count("1 passed") == 2
    assert result.submission is not None
    assert set(result.submission.manifest) == {
        "main.py",
        "tests",
        "tests/test_public.py",
    }
    assert result.limits.timeout_seconds == 30.0
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["config"]["environment"]["timeout_seconds"] == 30.0
    for mount in ("/etc", "/etc/alternatives", "/opt/bin", "/opt/python"):
        assert result.isolation.checks["readonly_" + mount]
    for tool in ("awk", "sed", "grep", "git", "pytest"):
        assert result.isolation.checks["tool_" + tool]
    assert result.isolation.runtime["task_pytest"] == "9.1.1"


def test_home_and_temporary_files_reset_between_steps_and_before_grading(
    tmp_path: Path,
) -> None:
    task = workspace_task()
    task = replace(
        task,
        initial_files=task.initial_files
        | {
            "main.py": (
                "import os\nfrom pathlib import Path\n"
                "def answer():\n"
                "    assert not (Path(os.environ['HOME']) / 'step-marker').exists()\n"
                "    assert not Path('/tmp/step-marker').exists()\n"
                "    return 42\n"
            )
        },
    )
    result = run_experiment(
        task=task,
        model=replay(
            [
                bash(
                    "set -e\n"
                    'test ! -e "$HOME/writable"\n'
                    'printf stale > "$HOME/step-marker"\n'
                    "printf stale > /tmp/step-marker\n"
                    "printf persisted > saved"
                ),
                bash(
                    "set -e\n"
                    'test ! -e "$HOME/step-marker"\n'
                    "test ! -e /tmp/step-marker\n"
                    "cat saved\n"
                    'printf second-step > "$HOME/step-marker"\n'
                    "printf second-step > /tmp/step-marker"
                ),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    assert all(
        step.observation is not None and step.observation.status == "ok"
        for step in result.trajectory
    )
    observation = result.trajectory[1].observation
    assert observation is not None and observation.output == "persisted"
    assert (result.workspace / "saved").read_text() == "persisted"


@pytest.mark.parametrize(
    "override",
    [
        "environment.python.source=/does-not-exist/python",
        "environment.python_version=0.0.0",
        "environment.pytest_version=0.0.0",
    ],
)
def test_unavailable_or_mismatched_python_environment_stops_before_actions(
    tmp_path: Path, override: str
) -> None:
    result = run_experiment(
        task=workspace_task(),
        model=replay([bash("touch MUST_NOT_RUN"), submit()]),
        output_dir=tmp_path,
        overrides=[override],
    )
    assert result.status == "infrastructure_error" and not result.isolation.passed
    assert result.reason and result.trajectory == [] and result.grade is None
    assert not (result.workspace / "MUST_NOT_RUN").exists()
