import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from bashlet import (
    Bash,
    ExperimentResult,
    GradingCase,
    Limits,
    Observation,
    Submit,
    Task,
    run_experiment,
)


def sum_task() -> Task:
    return Task(
        task_id="manual/sum-v1",
        prompt="Write solution.py with solve(numbers), returning the sum of integers.",
        feedback_tests="from solution import solve\nassert solve([1, 2]) == 3\n",
        grading_cases=(
            GradingCase(args=([1, 2],), expected=3),
            GradingCase(args=([],), expected=0),
            GradingCase(args=([-5, 2],), expected=-3),
        ),
        reference_implementation="def solve(numbers):\n    return sum(numbers)\n",
    )


def observed(result: ExperimentResult, index: int) -> Observation:
    observation = result.trajectory[index].observation
    assert observation is not None, result.trajectory[index]
    return observation


def write_solution(source: str) -> Bash:
    return Bash("cat > solution.py <<'PY'\n" + source + "PY\n")


def test_experiment_executes_feedback_and_grades_a_distinct_correct_submission(
    tmp_path: Path,
) -> None:
    implementation = (
        "def solve(numbers):\n"
        "    result = 0\n"
        "    for number in numbers:\n"
        "        result += number\n"
        "    return result\n"
    )
    result = run_experiment(
        task=sum_task(),
        actions=[
            Bash("cat > solution.py <<'PY'\n" + implementation + "PY\n"),
            Bash("python3 feedback_tests.py && printf 'feedback passed'"),
            Submit(),
        ],
        output_dir=tmp_path,
    )

    assert result.status == "completed", result
    assert result.isolation.passed, result.isolation
    assert observed(result, 1).stdout == "feedback passed"
    assert result.grade is not None and result.grade.passed
    assert result.grade.passed_cases == 3
    assert result.submission is not None
    assert (result.run_dir / result.submission.path).read_text() == implementation
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["task_id"] == "manual/sum-v1"
    assert record["grade"]["passed"] is True
    assert record["submission"]["sha256"] == result.submission.sha256


@pytest.mark.parametrize(
    ("source", "feedback_command"),
    [
        ("def solve(numbers):\n    return 3\n", "python3 feedback_tests.py"),
        (
            "def solve(numbers):\n    return 999\n",
            "printf 'print(\"all passed\")' > feedback_tests.py; "
            "python3 feedback_tests.py",
        ),
        (
            "import os\nos._exit(0)\n",
            "python3 feedback_tests.py; printf 'all passed'",
        ),
    ],
)
def test_feedback_or_self_report_cannot_determine_the_grade(
    tmp_path: Path,
    source: str,
    feedback_command: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        actions=[write_solution(source), Bash(feedback_command), Submit()],
        output_dir=tmp_path,
    )
    assert result.status == "completed", result.reason
    assert observed(result, 1).status == "ok"
    assert result.grade is not None and not result.grade.passed


def test_task_and_grading_processes_cannot_read_host_secrets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "host-secret-7b3969"
    host_private = tmp_path / "credentials"
    host_private.write_text(secret)
    monkeypatch.setenv("SERVICE_API_TOKEN", secret)
    # If a host FD accidentally survives spawning, a task could read it via /proc.
    with host_private.open() as descriptor:
        os.set_inheritable(descriptor.fileno(), True)
        result = run_experiment(
            task=replace(sum_task(), reference_implementation=secret),
            actions=[
                Bash(f"cat {shlex.quote(str(host_private))}"),
                Bash("printenv SERVICE_API_TOKEN"),
                Bash("cat /proc/1/root" + shlex.quote(str(host_private))),
                Bash("cat /workspace/../private/task.json"),
                Bash("cat /workspace/../run.json"),
                Bash(f"printf 'overwrite' > {shlex.quote(str(host_private))}"),
                Bash(
                    "python3 -c "
                    + shlex.quote(
                        "from pathlib import Path; "
                        f"assert {str(host_private)!r} not in "
                        "[str(p.resolve()) for p in Path('/proc/self/fd').iterdir()]"
                    )
                ),
                write_solution(
                    "import os\nfrom pathlib import Path\n"
                    "def solve(numbers):\n"
                    "    assert 'SERVICE_API_TOKEN' not in os.environ\n"
                    f"    assert not Path({str(host_private)!r}).exists()\n"
                    "    assert not Path('/workspace/feedback_tests.py').exists()\n"
                    "    assert not Path('/workspace/../private/task.json').exists()\n"
                    "    return sum(numbers)\n"
                ),
                Submit(),
            ],
            output_dir=tmp_path / "runs",
        )
    assert result.isolation.passed, result.isolation
    for index in range(6):
        assert observed(result, index).status == "command_failed"
        assert observed(result, index).exit_code != 0
    assert secret not in json.dumps(
        [(step.observation, step.feedback) for step in result.trajectory],
        default=str,
    )
    assert result.grade is not None and result.grade.passed
    assert observed(result, 6).status == "ok"
    assert host_private.read_text() == secret


def test_unavailable_isolation_preserves_reason_without_executing_actions(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        actions=[Bash("touch SHOULD_NOT_RUN"), Submit()],
        output_dir=tmp_path,
        bwrap="/does-not-exist/bwrap",
    )
    assert result.status == "infrastructure_error"
    assert not result.isolation.passed
    assert result.reason
    assert result.trajectory == []
    assert result.submission is None and result.grade is None
    assert not (result.run_dir / "workspace" / "SHOULD_NOT_RUN").exists()
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["status"] == "infrastructure_error"
    assert record["reason"] == result.reason


@pytest.mark.parametrize(
    ("command", "expected_status"),
    [
        ("sleep 5", "timeout"),
        ("yes excessive-output", "output_limit"),
        ("printf 'diagnostic' >&2; exit 17", "command_failed"),
        ("echo invalid\x00command", "command_failed"),
    ],
)
def test_execution_limits_produce_recorded_real_observations(
    tmp_path: Path,
    command: str,
    expected_status: str,
) -> None:
    limits = Limits(timeout_seconds=0.3, output_bytes=128)
    result = run_experiment(
        task=sum_task(),
        actions=[Bash(command)],
        output_dir=tmp_path,
        limits=limits,
    )
    assert result.status == "not_submitted", result.reason
    observation = observed(result, 0)
    assert observation.status == expected_status
    assert len(observation.stdout.encode()) + len(observation.stderr.encode()) <= 128
    if "exit 17" in command:
        assert observation.exit_code == 17
        assert observation.stderr == "diagnostic"
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["limits"]["timeout_seconds"] == 0.3
    assert record["limits"]["output_bytes"] == 128


def test_infrastructure_failure_during_grading_retains_its_diagnostic(
    tmp_path: Path,
) -> None:
    # A real executable boundary double removes bwrap only after probe + action.
    launcher = tmp_path / "bwrap-once"
    counter = tmp_path / "count"
    launcher.write_text(
        "#!/bin/bash\n"
        f"count_file={shlex.quote(str(counter))}\n"
        'n=0; if [ -f "$count_file" ]; then read -r n < "$count_file"; fi\n'
        'n=$((n+1)); printf "%s\\n" "$n" > "$count_file"\n'
        'if [ "$n" -gt 2 ]; then echo "executor unavailable" >&2; exit 70; fi\n'
        'exec /usr/bin/bwrap "$@"\n'
    )
    launcher.chmod(0o700)
    result = run_experiment(
        task=sum_task(),
        actions=[
            write_solution("def solve(numbers):\n    return sum(numbers)\n"),
            Submit(),
        ],
        output_dir=tmp_path / "runs",
        bwrap=str(launcher),
    )
    assert result.isolation.passed
    assert result.status == "infrastructure_error"
    assert "executor unavailable" in result.reason


def test_malformed_grading_output_is_a_failed_submission_not_a_crashed_run(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        actions=[
            write_solution(
                "import os\n"
                "os.write(1, b'[' * 2000 + b'0' + b']' * 2000)\nos._exit(0)\n"
            ),
            Submit(),
        ],
        output_dir=tmp_path,
    )
    assert result.status == "completed"
    assert result.grade is not None and not result.grade.passed
    assert all(
        case.status in {"invalid_output", "wrong_answer"} for case in result.grade.cases
    )


@pytest.mark.parametrize(
    "command",
    [
        "ln -s /etc/passwd solution.py",
        "mkfifo solution.py",
        "head -c 65537 /dev/zero > solution.py",
    ],
)
def test_submission_must_be_a_bounded_regular_file(
    tmp_path: Path, command: str
) -> None:
    result = run_experiment(
        task=sum_task(),
        actions=[Bash(command), Submit()],
        output_dir=tmp_path,
    )
    assert result.status == "invalid_submission"
    assert result.submission is None and result.grade is None
    assert result.reason


def test_submission_ends_the_episode_and_runs_keep_independent_snapshots(
    tmp_path: Path,
) -> None:
    actions: list[Bash | Submit] = [
        write_solution("def solve(numbers):\n    return sum(numbers)\n"),
        Submit(),
        Bash("echo 'MUST NOT EXECUTE' > solution.py"),
    ]
    first = run_experiment(task=sum_task(), actions=actions, output_dir=tmp_path)
    first_record = (first.run_dir / "run.json").read_bytes()
    assert first.submission is not None
    snapshot = (first.run_dir / first.submission.path).read_bytes()
    (first.run_dir / "workspace" / "solution.py").write_text("later mutation")
    second = run_experiment(task=sum_task(), actions=actions, output_dir=tmp_path)
    assert first.run_dir != second.run_dir
    assert first.grade is not None and first.grade.passed
    assert second.grade is not None and second.grade.passed
    assert [type(step.action) for step in first.trajectory] == [Bash, Submit]
    assert (first.run_dir / "run.json").read_bytes() == first_record
    assert (first.run_dir / first.submission.path).read_bytes() == snapshot
    assert not (first.run_dir / "workspace" / "grading_tests.py").exists()


def test_cli_runs_the_same_small_experiment(tmp_path: Path) -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "examples/sum.json",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["passed"] is True
    record = json.loads((Path(report["run_dir"]) / "run.json").read_text())
    assert record["task_id"] == "manual/sum-v1"
    assert record["grade"]["passed_cases"] == 3


def test_task_cannot_connect_to_a_host_service(tmp_path: Path) -> None:
    with socket.socket() as host_service:
        host_service.bind(("127.0.0.1", 0))
        host_service.listen()
        port = host_service.getsockname()[1]
        command = (
            "import socket; "
            f"socket.create_connection(('127.0.0.1', {port}), timeout=0.2)"
        )
        result = run_experiment(
            task=sum_task(),
            actions=[Bash("python3 -c " + shlex.quote(command))],
            output_dir=tmp_path,
        )
    assert result.isolation.passed
    assert observed(result, 0).status == "command_failed"
    assert "ConnectionRefusedError" in observed(result, 0).stderr


@pytest.mark.parametrize("wait_command", ["sleep 10", ":"])
def test_detached_processes_cannot_mutate_the_workspace_after_an_action(
    tmp_path: Path,
    wait_command: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        actions=[
            Bash(
                "setsid bash -c 'sleep 0.6; echo late > delayed-write' "
                ">/dev/null 2>&1 < /dev/null & " + wait_command
            )
        ],
        output_dir=tmp_path,
        limits=Limits(timeout_seconds=0.2),
    )
    assert result.status == "not_submitted", result.reason
    if wait_command == "sleep 10":
        assert observed(result, 0).status == "timeout"
    time.sleep(0.7)
    assert not (result.run_dir / "workspace" / "delayed-write").exists()
