import json
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from adapters import bash, condition_overrides, replay, submit, write_file

from bashlet import (
    Bash,
    ExperimentResult,
    FunctionGrader,
    GradingCase,
    InteractionBudget,
    Limits,
    Observation,
    Task,
    run_experiment,
)


def sum_task() -> Task:
    prompt = "Write solution.py with solve(numbers), returning the sum of integers."
    return Task(
        task_id="manual/sum-v1",
        prompt=prompt,
        initial_files={
            "task.md": prompt,
            "feedback_tests.py": (
                "from solution import solve\nassert solve([1, 2]) == 3\n"
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
        reference_implementation="def solve(numbers):\n    return sum(numbers)\n",
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback_tests.py",
    )


def observed(result: ExperimentResult, index: int) -> Observation:
    observation = result.trajectory[index].observation
    assert observation is not None, result.trajectory[index]
    return observation


def test_task_files_and_named_grader_support_a_complete_multifile_submission(
    tmp_path: Path,
) -> None:
    task = Task(
        task_id="manual/package-sum-v2",
        prompt="Submit the existing package and its data.",
        initial_files={
            "package/__init__.py": "",
            "package/helpers.py": "def total(numbers): return sum(numbers)\n",
            "package/answer.py": (
                "from pathlib import Path\n"
                "from .helpers import total\n"
                "def calculate(numbers):\n"
                "    assert Path('/workspace/data/offset').read_text() == '2'\n"
                "    assert not Path('/tmp/case-marker').exists()\n"
                "    Path('/tmp/case-marker').write_text('fresh process')\n"
                "    try:\n"
                "        Path('/workspace/data/offset').write_text('999')\n"
                "    except OSError:\n"
                "        return total(numbers) + 2\n"
                "    raise AssertionError('grading workspace must be read-only')\n"
            ),
            "data/offset": "2",
        },
        environment_id="linux-bwrap-v1",
        grader=FunctionGrader(
            module_path="package/answer.py",
            function_name="calculate",
            cases=(
                GradingCase(args=([],), expected=2),
                GradingCase(args=([-5, 2],), expected=-1),
            ),
        ),
    )
    result = run_experiment(task=task, model=replay([submit()]), output_dir=tmp_path)

    assert result.status == "completed", result.reason
    assert result.grade is not None and result.grade.passed_cases == 2
    assert [case.status for case in result.grade.cases] == ["passed", "passed"]
    assert result.submission is not None
    snapshot = result.run_dir / result.submission.path
    assert (snapshot / "data/offset").read_text() == "2"
    assert (snapshot / "package/helpers.py").read_text() == task.initial_files[
        "package/helpers.py"
    ]
    assert not (snapshot / "solution.py").exists()
    assert set(result.initial_workspace_manifest) == {
        "package",
        "package/__init__.py",
        "package/helpers.py",
        "package/answer.py",
        "data",
        "data/offset",
    }
    assert result.candidates == [] and result.repair_attempts == []


def test_grading_a_package_initializer_preserves_relative_imports(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=Task(
            task_id="manual/package-initializer",
            prompt="Submit the nested package.",
            initial_files={
                "outer/__init__.py": "",
                "outer/helpers.py": "def total(numbers): return sum(numbers)\n",
                "outer/inner/__init__.py": (
                    "from ..helpers import total\n"
                    "def calculate(numbers): return total(numbers)\n"
                ),
            },
            grader=FunctionGrader(
                "outer/inner/__init__.py", "calculate", (GradingCase(([2, 5],), 7),)
            ),
        ),
        model=replay([submit()]),
        output_dir=tmp_path,
    )
    assert result.passed, result.grade


def test_submission_preserves_executable_helpers_used_by_the_graded_function(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=Task(
            task_id="manual/executable-helper",
            prompt="Make helper.sh executable and submit the program.",
            initial_files={
                "helper.sh": "#!/bin/sh\nprintf 42\n",
                "main.py": (
                    "import subprocess\n"
                    "def answer():\n"
                    "    return int(subprocess.check_output("
                    "['/workspace/helper.sh']))\n"
                ),
            },
            grader=FunctionGrader("main.py", "answer", (GradingCase((), 42),)),
        ),
        model=replay(
            [
                bash(
                    "chmod +x helper.sh; "
                    "python3 -B -c 'from main import answer; print(answer())'"
                ),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )
    assert observed(result, 0).status == "ok"
    assert observed(result, 0).output == "42\n"
    assert result.passed, result.grade


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ("environment.submission_bytes=40", "size limit"),
        ("environment.submission_files=2", "file count limit"),
    ],
)
def test_submission_limits_cover_all_workspace_files(
    tmp_path: Path, override: str, reason: str
) -> None:
    result = run_experiment(
        task=Task(
            task_id="manual/whole-workspace-limit",
            prompt="Submit the existing files.",
            initial_files={
                "main.py": "def run(): return 7\n",
                "a": "abcdefghijklmnop",
                "b": "abcdefghijklmnop",
            },
            grader=FunctionGrader("main.py", "run", (GradingCase((), 7),)),
        ),
        model=replay([submit()]),
        output_dir=tmp_path,
        overrides=[override],
    )
    assert result.status == "invalid_submission"
    assert reason in result.reason
    assert result.episode_status == "submitted"
    assert result.submission is None and result.grade is None
    assert not result.passed
    assert (result.workspace / "b").read_text() == "abcdefghijklmnop"


def test_submission_preserves_binary_files_empty_directories_and_independent_contents(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                bash("mkdir empty; printf '\\000\\377' > binary; touch blank"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )
    assert result.passed
    assert result.submission is not None
    snapshot = result.run_dir / result.submission.path
    assert (snapshot / "empty").is_dir()
    assert list((snapshot / "empty").iterdir()) == []
    assert (snapshot / "binary").read_bytes() == b"\x00\xff"
    assert (snapshot / "blank").read_bytes() == b""
    (result.workspace / "binary").write_bytes(b"later mutation")
    (result.workspace / "empty").rmdir()
    assert (snapshot / "binary").read_bytes() == b"\x00\xff"
    assert (snapshot / "empty").is_dir()


def test_unsubmitted_workspace_is_retained_and_counted_as_a_failure(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay([write_file("def solve(numbers):\n    return sum(numbers)\n")]),
        output_dir=tmp_path,
        overrides=["budget.max_steps=1"],
    )
    assert result.status == result.episode_status == "budget_exhausted"
    assert result.submission is None and result.grade is None
    assert not result.passed
    assert (
        (result.workspace / "solution.py").read_text().endswith("return sum(numbers)\n")
    )
    assert not (result.run_dir / "private/grading-workspace").exists()
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["passed"] is False


@pytest.mark.parametrize(
    "path", ["../outside", "/tmp/outside", "a/../outside", "a//b", "a/./b", ""]
)
def test_initial_task_files_cannot_escape_the_workspace(
    tmp_path: Path, path: str
) -> None:
    task = replace(sum_task(), initial_files={path: "must not be written"})
    with pytest.raises(ValueError, match="workspace path"):
        run_experiment(
            task=task, model=replay([submit()]), output_dir=tmp_path / "runs"
        )
    assert not (tmp_path / "runs").exists()


def test_task_environment_must_match_the_selected_execution_environment(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="Task environment"):
        run_experiment(
            task=replace(sum_task(), environment_id="other-environment"),
            model=replay([submit()]),
            output_dir=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def run_cli(
    request: Path,
    output_dir: Path,
    *,
    overrides: list[str] | None = None,
    configs: list[Path] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "bashlet",
        str(request),
        "--output-dir",
        str(output_dir),
    ]
    for path in configs or []:
        command.extend(["--config", str(path)])
    for override in overrides or []:
        command.extend(["--set", override])
    return subprocess.run(command, capture_output=True, text=True, timeout=20)


def test_experiment_rejects_scripted_actions(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="actions"):
        run_experiment(
            task=sum_task(),
            actions=[],  # type: ignore[call-arg]
            output_dir=tmp_path,
        )


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
        model=replay(
            [
                write_file(implementation),
                bash("python3 feedback_tests.py && printf 'feedback passed'"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result
    assert result.isolation.passed, result.isolation
    assert observed(result, 1).output == "feedback passed"
    assert result.grade is not None and result.grade.passed
    assert result.grade.passed_cases == 3
    assert result.submission is not None
    assert (
        result.run_dir / result.submission.path / "solution.py"
    ).read_text() == implementation
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["task_id"] == "manual/sum-v1"
    assert record["grade"]["passed"] is True
    assert record["submission"]["sha256"] == result.submission.sha256


def test_workspace_manifests_preserve_each_step_and_deduplicate_file_contents(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash("mkdir nested; printf hello > nested/a; printf hello > copy"),
                "No command block.",
                bash("printf world > nested/a; rm copy; touch empty"),
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.status == "completed", result.reason
    hello = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    world = "486ea46224d1bb4fb680f34f7c9ad96a8f24ec88be73ea8e5a6c65260e9cb8a7"
    empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    first = result.trajectory[0].workspace_manifest
    assert first["nested"].kind == "directory"
    assert first["nested/a"].kind == "file"
    assert first["nested/a"].sha256 == first["copy"].sha256 == hello
    assert first["nested/a"].size_bytes == 5
    assert result.trajectory[1].workspace_manifest == first
    changed = result.trajectory[2].workspace_manifest
    assert "copy" not in changed
    assert changed["nested/a"].sha256 == world
    assert changed["empty"].sha256 == empty
    assert result.trajectory[-1].workspace_manifest == (
        result.trajectory[-2].workspace_manifest
    )
    blobs = result.run_dir / "blobs"
    assert (blobs / hello).read_bytes() == b"hello"
    assert (blobs / world).read_bytes() == b"world"
    assert (blobs / empty).read_bytes() == b""
    assert sum(path.read_bytes() == b"hello" for path in blobs.iterdir()) == 1
    record = json.loads((result.run_dir / "run.json").read_text())
    assert set(record["initial_workspace_manifest"]) == {"task.md", "feedback_tests.py"}
    assert [step["workspace_manifest"] for step in record["trajectory"]] == [
        {path: asdict(entry) for path, entry in step.workspace_manifest.items()}
        for step in result.trajectory
    ]


def test_workspace_manifests_record_special_files_without_following_them(
    tmp_path: Path,
) -> None:
    secret = b"host-only-content-48a1"
    private = tmp_path / "private.txt"
    private.write_bytes(secret)
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash(
                    f"ln -s {shlex.quote(str(private))} external; "
                    "mkdir nested; printf hello > nested/a; "
                    "ln -s nested linked_dir; ln -s . loop; "
                    "ln -s missing dangling; mkfifo pipe; "
                    'python3 -c "import socket; '
                    "s = socket.socket(socket.AF_UNIX); s.bind('/workspace/socket')\""
                ),
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path / "runs",
    )

    assert result.status == "invalid_submission", result.reason
    assert result.grade is None
    manifest = result.trajectory[0].workspace_manifest
    assert manifest["external"].kind == "symlink"
    assert manifest["external"].target == str(private)
    assert manifest["linked_dir"].kind == manifest["loop"].kind == "symlink"
    assert manifest["dangling"].kind == "symlink"
    assert manifest["pipe"].kind == "fifo"
    assert manifest["socket"].kind == "socket"
    assert all(
        manifest[path].sha256 is None
        for path in ("external", "linked_dir", "loop", "dangling", "pipe", "socket")
    )
    assert "linked_dir/a" not in manifest and "loop/nested/a" not in manifest
    assert all(
        path.read_bytes() != secret for path in (result.run_dir / "blobs").iterdir()
    )


def test_unreadable_workspace_entries_are_recorded_and_can_be_repaired(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash(
                    "printf hello > locked; mkdir locked_dir; "
                    "printf hello > locked_dir/a; chmod 000 locked locked_dir"
                ),
                bash("chmod 600 locked; chmod 700 locked_dir"),
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )
    # Restore permissions even when the episode fails, so pytest can clean up.
    for path in (result.run_dir / "workspace").glob("locked*"):
        path.chmod(0o700)

    assert result.status == "completed", result.reason
    first = result.trajectory[0].workspace_manifest
    assert first["locked"].kind == "file" and first["locked"].error
    assert first["locked"].sha256 is None
    assert first["locked_dir"].kind == "directory" and first["locked_dir"].error
    repaired = result.trajectory[1].workspace_manifest
    assert (
        repaired["locked"].sha256
        == repaired["locked_dir/a"].sha256
        == ("2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824")
    )
    assert not repaired["locked"].error and not repaired["locked_dir"].error


@pytest.mark.parametrize("paths", [("solution.py",), ("solution.py", "\udcff")])
def test_workspace_manifests_preserve_non_utf8_paths_and_symlink_targets(
    tmp_path: Path,
    paths: tuple[str, ...],
) -> None:
    result = run_experiment(
        task=replace(sum_task(), implementation_paths=paths),
        model=replay(
            [
                bash(
                    "python3 -c "
                    + shlex.quote(
                        "import os; open(bytes([255]), 'wb').write(b'hello'); "
                        "os.symlink(bytes([254]), b'link')"
                    )
                ),
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )

    assert result.status == "invalid_submission", result.reason
    manifest = result.trajectory[0].workspace_manifest
    assert manifest["\udcff"].sha256 == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )
    assert manifest["link"].target == "\udcfe"
    record = json.loads((result.run_dir / "run.json").read_text())
    saved = record["trajectory"][0]["workspace_manifest"]
    assert saved["\udcff"] == asdict(manifest["\udcff"])
    assert saved["link"]["target"] == "\udcfe"
    task_record = json.loads((result.run_dir / "private" / "task.json").read_text())
    assert task_record["implementation_paths"] == list(paths)
    if len(paths) > 1:
        [candidate] = result.candidates
        snapshot = json.loads((result.run_dir / candidate.path).read_text())
        assert snapshot["\udcff"] == manifest["\udcff"].sha256


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
        model=replay([write_file(source), bash(feedback_command), submit()]),
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
            model=replay(
                [
                    bash(f"cat {shlex.quote(str(host_private))}"),
                    bash("printenv SERVICE_API_TOKEN"),
                    bash("cat /proc/1/root" + shlex.quote(str(host_private))),
                    bash("cat /workspace/../private/task.json"),
                    bash("cat /workspace/../run.json"),
                    bash(f"printf 'overwrite' > {shlex.quote(str(host_private))}"),
                    bash(
                        "python3 -c "
                        + shlex.quote(
                            "from pathlib import Path; "
                            f"assert {str(host_private)!r} not in "
                            "[str(p.resolve()) for p in "
                            "Path('/proc/self/fd').iterdir()]"
                        )
                    ),
                    write_file(
                        "import os\nfrom pathlib import Path\n"
                        "def solve(numbers):\n"
                        "    assert 'SERVICE_API_TOKEN' not in os.environ\n"
                        f"    assert not Path({str(host_private)!r}).exists()\n"
                        "    assert Path('/workspace/feedback_tests.py').exists()\n"
                        "    assert not "
                        "Path('/workspace/../private/task.json').exists()\n"
                        "    return sum(numbers)\n"
                    ),
                    submit(),
                ]
            ),
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
        model=replay([bash("touch SHOULD_NOT_RUN"), submit()]),
        output_dir=tmp_path,
        overrides=condition_overrides(bwrap="/does-not-exist/bwrap"),
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
        ("yes excessive-output", "timeout"),
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
        model=replay([bash(command)]),
        output_dir=tmp_path,
        overrides=condition_overrides(
            budget=InteractionBudget(max_steps=1), limits=limits
        ),
    )
    assert result.status == "budget_exhausted", result.reason
    observation = observed(result, 0)
    assert observation.status == expected_status
    assert len(observation.output.encode()) <= 128
    if "exit 17" in command:
        assert observation.exit_code == 17
        assert observation.output == "diagnostic"
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["limits"]["timeout_seconds"] == 0.3
    assert record["limits"]["output_bytes"] == 128


def test_infrastructure_failure_during_grading_retains_its_diagnostic(
    tmp_path: Path,
) -> None:
    # The executable boundary double fails after probe + write + submission.
    launcher = tmp_path / "bwrap-once"
    counter = tmp_path / "count"
    launcher.write_text(
        "#!/bin/bash\n"
        f"count_file={shlex.quote(str(counter))}\n"
        'n=0; if [ -f "$count_file" ]; then read -r n < "$count_file"; fi\n'
        'n=$((n+1)); printf "%s\\n" "$n" > "$count_file"\n'
        'if [ "$n" -gt 3 ]; then echo "executor unavailable" >&2; exit 70; fi\n'
        'exec /usr/bin/bwrap "$@"\n'
    )
    launcher.chmod(0o700)
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                write_file("def solve(numbers):\n    return sum(numbers)\n"),
                submit(),
            ]
        ),
        output_dir=tmp_path / "runs",
        overrides=condition_overrides(bwrap=str(launcher)),
    )
    assert result.isolation.passed
    assert result.status == "infrastructure_error"
    assert "executor unavailable" in result.reason


def test_malformed_grading_output_is_a_failed_submission_not_a_crashed_run(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                write_file(
                    "import os\n"
                    "os.write(1, b'[' * 2000 + b'0' + b']' * 2000)\nos._exit(0)\n"
                ),
                submit(),
            ]
        ),
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
        model=replay([bash(command), submit()]),
        output_dir=tmp_path,
    )
    assert result.status == "invalid_submission"
    assert result.submission is None and result.grade is None
    assert result.reason


def test_submission_ends_the_episode_and_runs_keep_independent_snapshots(
    tmp_path: Path,
) -> None:
    replies = [
        write_file("def solve(numbers):\n    return sum(numbers)\n"),
        submit(),
        bash("echo 'MUST NOT EXECUTE' > solution.py"),
    ]
    first = run_experiment(task=sum_task(), model=replay(replies), output_dir=tmp_path)
    first_record = (first.run_dir / "run.json").read_bytes()
    assert first.submission is not None
    snapshot = (first.run_dir / first.submission.path / "solution.py").read_bytes()
    (first.run_dir / "workspace" / "solution.py").write_text("later mutation")
    second = run_experiment(task=sum_task(), model=replay(replies), output_dir=tmp_path)
    assert first.run_dir != second.run_dir
    assert first.grade is not None and first.grade.passed
    assert second.grade is not None and second.grade.passed
    assert [type(step.action) for step in first.trajectory] == [Bash, Bash]
    assert (first.run_dir / "run.json").read_bytes() == first_record
    assert (
        first.run_dir / first.submission.path / "solution.py"
    ).read_bytes() == snapshot
    assert not (first.run_dir / "workspace" / "grading_tests.py").exists()


def test_run_records_one_merged_config_and_three_stable_fingerprints(
    tmp_path: Path,
) -> None:
    first = run_experiment(
        task=sum_task(), model=replay([submit()]), output_dir=tmp_path
    )
    second = run_experiment(
        task=sum_task(), model=replay([submit()]), output_dir=tmp_path
    )

    record = json.loads((first.run_dir / "run.json").read_text())
    again = json.loads((second.run_dir / "run.json").read_text())
    assert set(record["config"]) == {"protocol", "environment", "budget", "model"}
    assert record["fingerprints"] == again["fingerprints"]
    assert set(record["fingerprints"]) == {"protocol", "environment", "budget"}
    assert record["fingerprints"]["protocol"] == record["protocol"]["sha256"]
    assert record["config"]["protocol"]["name"] == "bash-block-v2"
    assert record["config"]["budget"] == record["budget"]
    assert record["config"]["model"]["sampling"]["max_tokens"] == 1024


def test_protocol_text_and_fingerprint_are_independent_of_task_shape(
    tmp_path: Path,
) -> None:
    mbpp = run_experiment(
        task=sum_task(),
        model=replay(
            [write_file("def solve(numbers): return sum(numbers)\n"), submit()]
        ),
        output_dir=tmp_path,
    )
    package = run_experiment(
        task=Task(
            task_id="manual/other-shape",
            prompt="Fix the multiplier in the existing package and submit it.",
            initial_files={
                "helpers.py": "def twice(n): return n\n",
                "main.py": (
                    "from helpers import twice\ndef answer(n): return twice(n)\n"
                ),
            },
            grader=FunctionGrader("main.py", "answer", (GradingCase((6,), 12),)),
        ),
        model=replay(
            [bash("printf 'def twice(n): return n * 2\\n' > helpers.py"), submit()]
        ),
        output_dir=tmp_path,
    )

    assert mbpp.passed and package.passed
    first = json.loads((mbpp.run_dir / "run.json").read_text())
    second = json.loads((package.run_dir / "run.json").read_text())
    assert first["protocol"] == second["protocol"]
    assert first["fingerprints"]["protocol"] == second["fingerprints"]["protocol"]
    text = json.dumps(first["protocol"])
    for filename in (
        "solution.py",
        "feedback_tests.py",
        "task.md",
        "helpers.py",
        "main.py",
    ):
        assert filename not in text


def test_truncated_grading_json_cannot_pass_as_a_valid_output_prefix(
    tmp_path: Path,
) -> None:
    result = run_experiment(
        task=Task(
            task_id="manual/truncated-grading",
            prompt="Submit the existing program.",
            initial_files={
                "main.py": (
                    "import os\ndef answer():\n"
                    "    os.write(1, b'7' + b' ' * 256 + b'invalid suffix')\n"
                    "    os._exit(0)\n"
                ),
            },
            grader=FunctionGrader("main.py", "answer", (GradingCase((), 7),)),
        ),
        model=replay([submit()]),
        output_dir=tmp_path,
        overrides=["environment.output_bytes=128"],
    )

    assert result.status == "completed"
    assert result.grade is not None and not result.grade.passed
    assert result.grade.cases[0].status == "output_limit"


def _fingerprints(tmp_path: Path, config: list[Path] | None = None) -> dict[str, str]:
    result = run_experiment(
        task=sum_task(),
        model=replay([submit()]),
        output_dir=tmp_path,
        config=config or [],
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    fingerprints = record["fingerprints"]
    assert isinstance(fingerprints, dict)
    return {str(name): str(digest) for name, digest in fingerprints.items()}


def test_editing_one_block_changes_only_its_fingerprint(tmp_path: Path) -> None:
    baseline = _fingerprints(tmp_path / "baseline")
    overlays = {
        "protocol": "protocol:\n  user_prompt: other\n",
        "environment": "environment:\n  timeout_seconds: 1.5\n",
        "budget": "budget:\n  max_steps: 4\n",
    }
    for name, text in overlays.items():
        path = tmp_path / f"{name}.yaml"
        path.write_text(text)
        changed = _fingerprints(tmp_path / name, [path])
        assert changed[name] != baseline[name]
        for other in baseline:
            if other != name:
                assert changed[other] == baseline[other]


def test_model_block_does_not_change_condition_fingerprints(tmp_path: Path) -> None:
    baseline = _fingerprints(tmp_path / "baseline")
    overlay = tmp_path / "model.yaml"
    overlay.write_text(
        "model:\n"
        "  backend: other\n"
        "  thinking: true\n"
        "  sampling:\n"
        "    temperature: 0.2\n"
    )
    result = run_experiment(
        task=sum_task(),
        model=replay([submit()]),
        output_dir=tmp_path / "runs",
        config=[overlay],
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["fingerprints"] == baseline
    assert record["config"]["model"]["backend"] == "other"
    assert record["config"]["model"]["thinking"] is True
    assert record["config"]["model"]["sampling"]["temperature"] == 0.2
    assert record["config"]["model"]["sampling"]["max_tokens"] == 1024


def test_later_config_file_and_key_override_replace_earlier_values(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text("budget:\n  max_steps: 4\n")
    second.write_text("budget:\n  max_steps: 6\n")
    result = run_experiment(
        task=sum_task(),
        model=replay([submit()]),
        output_dir=tmp_path / "runs",
        config=[first, second],
        overrides=["budget.max_steps=7"],
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["config"]["budget"]["max_steps"] == 7
    assert record["budget"]["max_steps"] == 7


@pytest.mark.parametrize(
    ("text", "overrides", "match"),
    [
        ("budget:\n  max_turns: 1\n", [], "max_turns"),
        ("budget:\n  max_steps: 1.5\n", [], "integer"),
        (
            "protocol:\n  system_prompt: 'Hello {{ missing_name }}'\n",
            [],
            "missing_name",
        ),
        ("", ["budget.nope=1"], "budget.nope"),
        ("", ["budget.max_steps=0"], "positive"),
    ],
)
def test_invalid_config_is_rejected_before_a_run(
    tmp_path: Path, text: str, overrides: list[str], match: str
) -> None:
    configs: list[Path] = []
    if text:
        overlay = tmp_path / "overlay.yaml"
        overlay.write_text(text)
        configs.append(overlay)
    with pytest.raises(ValueError, match=match):
        run_experiment(
            task=sum_task(),
            model=replay([submit()]),
            output_dir=tmp_path / "runs",
            config=configs,
            overrides=overrides,
        )
    assert not (tmp_path / "runs").exists()


def test_cli_runs_the_same_small_experiment(tmp_path: Path) -> None:
    process = run_cli(Path("examples/sum.json"), tmp_path)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["passed"] is True
    record = json.loads((Path(report["run_dir"]) / "run.json").read_text())
    assert record["task_id"] == "manual/sum-v1"
    assert record["grade"]["passed_cases"] == 3
    assert record["encoding"] == "replay/utf8-v1"
    assert record["context_tokens"]
    assert len(record["candidates"]) == 1
    assert [step["verification"] for step in record["trajectory"]] == [
        False, True, False
    ]  # fmt: skip
    assert all(
        step["model_call"]["generation"]["source"] == "replayed"
        and step["model_call"]["generation"]["logprobs"] is None
        for step in record["trajectory"]
    )


@pytest.mark.parametrize(
    ("replies", "bwrap", "exit_code", "status", "passed"),
    [
        (
            [write_file("def solve(numbers):\n    return 3\n"), submit()],
            "/usr/bin/bwrap",
            1,
            "completed",
            False,
        ),
        ([bash("true")], "/usr/bin/bwrap", 1, "budget_exhausted", None),
        ([submit()], "/does-not-exist/bwrap", 2, "infrastructure_error", None),
    ],
    ids=["failed-grade", "unsubmitted", "unavailable-isolation"],
)
def test_cli_preserves_failure_and_infrastructure_exit_codes(
    tmp_path: Path,
    replies: list[str],
    bwrap: str,
    exit_code: int,
    status: str,
    passed: bool | None,
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"task": asdict(sum_task()), "replies": replies}))
    process = run_cli(
        request,
        tmp_path / "runs",
        overrides=["budget.max_steps=2", f"environment.bwrap={bwrap}"],
    )

    assert process.returncode == exit_code, process.stderr
    report = json.loads(process.stdout)
    assert report["status"] == status
    assert report["passed"] is passed
    record = json.loads((Path(report["run_dir"]) / "run.json").read_text())
    assert record["status"] == status


@pytest.mark.parametrize(
    "invalid_request",
    [
        {"replies": "a single string"},
        {"replies": [1]},
        {"replies": None},
        {"actions": [{"kind": "submit"}]},
        {"actions": [], "replies": [submit()]},
        {"replies": [submit()], "budget": {"max_steps": 1}},
        {"replies": [submit()], "limits": {"timeout_seconds": 1}},
    ],
    ids=[
        "string",
        "non-text",
        "null",
        "old-actions",
        "mixed-inputs",
        "request-budget",
        "request-limits",
    ],
)
def test_cli_rejects_invalid_or_scripted_requests_as_configuration_errors(
    tmp_path: Path, invalid_request: dict[str, object]
) -> None:
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"task": asdict(sum_task())} | invalid_request))
    process = run_cli(request, tmp_path / "runs")

    assert process.returncode == 2
    assert "error:" in process.stderr
    assert not (tmp_path / "runs").exists()


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
            model=replay([bash("python3 -c " + shlex.quote(command))]),
            output_dir=tmp_path,
            overrides=condition_overrides(budget=InteractionBudget(max_steps=1)),
        )
    assert result.isolation.passed
    assert observed(result, 0).status == "command_failed"
    assert "ConnectionRefusedError" in observed(result, 0).output


@pytest.mark.parametrize("wait_command", ["sleep 10", ":"])
def test_detached_processes_cannot_mutate_the_workspace_after_an_action(
    tmp_path: Path,
    wait_command: str,
) -> None:
    result = run_experiment(
        task=sum_task(),
        model=replay(
            [
                bash(
                    "setsid bash -c 'sleep 0.6; echo late > delayed-write' "
                    ">/dev/null 2>&1 < /dev/null & " + wait_command
                )
            ]
        ),
        output_dir=tmp_path,
        overrides=condition_overrides(
            budget=InteractionBudget(max_steps=1),
            limits=Limits(timeout_seconds=0.2),
        ),
    )
    assert result.status == "budget_exhausted", result.reason
    if wait_command == "sleep 10":
        assert observed(result, 0).status == "timeout"
    time.sleep(0.7)
    assert not (result.run_dir / "workspace" / "delayed-write").exists()
