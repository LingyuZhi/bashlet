"""CLI behavior through subprocesses, saved evidence and real sandbox execution."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from adapters import bash, submit, write_file


def replay_config(tmp_path: Path, replies: list[str]) -> Path:
    path = tmp_path / "replay.yaml"
    path.write_text(
        yaml.safe_dump({"model": {"class": "ReplayModel", "replies": replies}})
    )
    return path


def cli(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "bashlet", *arguments],
        cwd=cwd,
        env=os.environ | {"NO_COLOR": "1"},
        capture_output=True,
        text=True,
        timeout=20,
    )


def saved_run(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    records = list(output_dir.glob("*/run.json"))
    assert len(records) == 1
    return records[0].parent, json.loads(records[0].read_text())


def task_file(tmp_path: Path) -> Path:
    task = tmp_path / "task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "cli/double",
                "prompt": "Double the argument and submit.",
                "initial_files": {},
                "grader": {
                    "module_path": "solution.py",
                    "function_name": "solve",
                    "cases": [
                        {"args": [7], "expected": 14},
                        {"args": [-2], "expected": -4},
                    ],
                },
            }
        )
    )
    return task


def test_agent_runs_without_a_task_and_changes_only_the_copied_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / "note.txt").write_text("original\n")
    (workspace / "binary").write_bytes(b"\x00\xff")
    (workspace / "relative-link").symlink_to("note.txt")
    config = replay_config(
        tmp_path,
        [bash("cat note.txt; printf 'updated\\n' > relative-link"), submit()],
    )
    output_dir = tmp_path / "runs"
    process = cli(
        "agent",
        "Update note.txt",
        "--workdir",
        str(workspace),
        "-c",
        str(config),
        "--output-dir",
        str(output_dir),
        cwd=tmp_path,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "submitted"
    assert "task_id" not in record and "grade" not in record
    assert (workspace / "note.txt").read_text() == "original\n"
    assert (run_dir / "workspace/note.txt").read_text() == "updated\n"
    assert (run_dir / "workspace/binary").read_bytes() == b"\x00\xff"
    assert (run_dir / "workspace/relative-link").is_symlink()
    assert record["config"]["model"]["class"] == "ReplayModel"
    assert record["trajectory"][0]["model_call"]["generation"]["source"] == "replayed"
    for visible in ("Step 1", "Budget:", "cat note.txt", "original", "submitted"):
        assert visible in process.stdout


def test_task_displays_grading_and_show_replays_the_same_steps(tmp_path: Path) -> None:
    task = task_file(tmp_path)
    config = replay_config(
        tmp_path,
        [
            "A malformed reply without a command.",
            write_file("def solve(n): return n * 2\n"),
            bash("printf '[bold]literal observation[/bold]\\n'"),
            submit(),
        ],
    )
    output_dir = tmp_path / "runs"
    live = cli(
        "task",
        str(task),
        "-c",
        str(config),
        "--output-dir",
        str(output_dir),
        cwd=tmp_path,
    )

    assert live.returncode == 0, live.stdout + live.stderr
    run_dir, record = saved_run(output_dir)
    assert record["episode_status"] == "submitted"
    assert record["grade"]["passed_cases"] == 2
    assert "Grade: passed (2/2)" in live.stdout
    for visible in ("Step 1", "Format error:", "Run the next command.", "def solve(n)"):
        assert visible in live.stdout
    assert record["trajectory"][2]["feedback"] in live.stdout
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == live.stdout


def test_task_accepts_a_hidden_test_command_and_show_preserves_its_grade(
    tmp_path: Path,
) -> None:
    task = tmp_path / "command-task.json"
    task.write_text(
        json.dumps(
            {
                "task_id": "cli/hidden-command",
                "prompt": "Submit the existing program.",
                "initial_files": {"solution.py": "def solve(n): return n * 2\n"},
                "grader": {
                    "command": "pytest -q tests/test_hidden.py",
                    "hidden_files": {
                        "tests/test_hidden.py": (
                            "from solution import solve\n"
                            "def test_solve(): assert solve(-2) == -4\n"
                        )
                    },
                },
            }
        )
    )
    config = replay_config(tmp_path, [submit()])
    live = cli("task", str(task), "-c", str(config), cwd=tmp_path)

    assert live.returncode == 0, live.stdout + live.stderr
    run_dir, record = saved_run(tmp_path / "runs")
    assert record["grading_protocol"] == "test-command-v1"
    assert record["grade"]["passed"] is True
    assert "Grade: passed (1/1)" in live.stdout
    assert not (run_dir / "workspace/tests/test_hidden.py").exists()
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0 and replayed.stdout == live.stdout


@pytest.mark.parametrize("command", ["agent", "task"])
def test_ctrl_c_keeps_the_inflight_action_output_and_workspace(
    tmp_path: Path,
    command: str,
) -> None:
    workspace = tmp_path / "source"
    workspace.mkdir()
    config = replay_config(
        tmp_path,
        [
            bash("printf 'finished first step\\n'"),
            bash("printf 'partial output\\n'; touch ready; sleep 30; touch TOO_LATE"),
            submit(),
        ],
    )
    output_dir = tmp_path / "runs"
    args = (
        ["agent", "Do some work", "--workdir", str(workspace)]
        if command == "agent"
        else ["task", str(task_file(tmp_path))]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bashlet",
            *args,
            "-c",
            str(config),
            "--output-dir",
            str(output_dir),
            "--set",
            "environment.timeout_seconds=40",
        ],
        cwd=tmp_path,
        env=os.environ | {"NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not list(output_dir.glob("*/workspace/ready")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "The action never started"
            time.sleep(0.02)
        time.sleep(0.1)  # Allow the parent to drain the command's flushed output.
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert process.returncode == 130, stdout + stderr
    assert "interrupted" in stdout and "Traceback" not in stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "interrupted"
    assert len(record["trajectory"]) == record["usage"]["steps"] == 2
    step = record["trajectory"][1]
    assert step["observation"]["status"] == "interrupted"
    assert step["observation"]["output"] == "partial output\n"
    assert "sleep 30" in step["action"]["command"]
    assert step["model_call"]["generation"]["tokens"]
    assert "ready" in step["workspace_manifest"]
    assert not (run_dir / "workspace/TOO_LATE").exists()
    assert not (run_dir / "run.json.tmp").exists()
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == stdout


def test_show_accepts_steps_with_optional_untagged_explanations(tmp_path: Path) -> None:
    config = replay_config(tmp_path, [bash("printf 'visible\\n'"), submit()])
    live = cli("agent", "Submit", "-c", str(config), cwd=tmp_path)
    assert live.returncode == 0, live.stdout + live.stderr
    run_dir, record = saved_run(tmp_path / "runs")
    # v2 requires THOUGHT; recorded fixtures exercise the renderer's v3 input
    # without changing the action protocol before task 19.
    record["trajectory"][0]["model_call"]["text"] = "```bash\nprintf 'visible\\n'\n```"
    record["trajectory"][1]["model_call"]["text"] = (
        "The files are ready.\n```bash\n"
        "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```"
    )
    (run_dir / "run.json").write_text(json.dumps(record))
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    first, second = replayed.stdout.split("Step 2", 1)
    assert "Step 1" in first and "visible" in first and "Explanation:" not in first
    assert "The files are ready." in second
    assert "Outcome: submitted" in second


@pytest.mark.parametrize(
    "class_name", ["bashlet.replay.ReplayModel", "bashlet.replay:ReplayModel"]
)
def test_imported_model_receives_its_constructor_parameters_and_layered_sampling(
    tmp_path: Path,
    class_name: str,
) -> None:
    config = replay_config(tmp_path, [submit()])
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text(
        yaml.safe_dump({"model": {"class": class_name, "identity": "cli/configured"}})
    )
    process = cli(
        "agent",
        "Submit",
        "-c",
        str(config),
        "--config",
        str(overlay),
        "--set",
        "model.sampling.max_tokens=200",
        "--set",
        "budget.max_steps=3",
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["policy"] == "cli/configured"
    assert record["config"]["model"]["class"] == class_name
    assert record["trajectory"][0]["model_call"]["sampling"]["max_tokens"] == 200
    assert record["budget"]["max_steps"] == 3
    assert "runs" not in record["initial_workspace_manifest"]


def test_live_view_uses_the_stored_elided_observation_and_step_budget(
    tmp_path: Path,
) -> None:
    config = replay_config(
        tmp_path, [bash("python3 -c 'print(\"A\" * 200)'"), submit()]
    )
    process = cli(
        "agent",
        "Read output",
        "-c",
        str(config),
        "--set",
        "protocol.observation_head_chars=5",
        "--set",
        "protocol.observation_tail_chars=5",
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    run_dir, record = saved_run(tmp_path / "runs")
    step = record["trajectory"][0]
    assert step["feedback"] in process.stdout
    assert step["observation"]["output"] not in process.stdout
    assert "191 characters omitted" in process.stdout
    assert step["usage"]["steps"] == 1
    assert step["usage"]["generated_tokens"] == len(
        step["model_call"]["generation"]["tokens"]
    )
    assert f"tokens {step['usage']['generated_tokens']}/8192" in process.stdout
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == process.stdout


@pytest.mark.parametrize(
    "class_name", ["NotAClass", "missing_backend.Model", "pathlib.Path"]
)
def test_invalid_model_configuration_fails_before_creating_a_run(
    tmp_path: Path,
    class_name: str,
) -> None:
    config = tmp_path / "bad-model.yaml"
    config.write_text(yaml.safe_dump({"model": {"class": class_name}}))
    process = cli("agent", "Submit", "-c", str(config), cwd=tmp_path)
    assert process.returncode == 2
    assert (
        "Cannot construct model" in process.stderr and "Traceback" not in process.stderr
    )
    assert not (tmp_path / "runs").exists()


def test_ctrl_c_during_grading_retains_the_submission_and_a_complete_record(
    tmp_path: Path,
) -> None:
    config = replay_config(
        tmp_path,
        [
            write_file(
                "import time\ndef solve(n):\n    time.sleep(30)\n    return n * 2\n"
            ),
            submit(),
        ],
    )
    output_dir = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bashlet",
            "task",
            str(task_file(tmp_path)),
            "-c",
            str(config),
            "--output-dir",
            str(output_dir),
            "--set",
            "environment.timeout_seconds=40",
        ],
        cwd=tmp_path,
        env=os.environ | {"NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not list(output_dir.glob("*/private/grading-workspace")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "Grading never started"
            time.sleep(0.02)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 130, stdout + stderr
    assert "Grading interrupted" in stdout and "Traceback" not in stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "interrupted"
    assert record["episode_status"] == "submitted"
    assert record["submission"] is not None and record["grade"] is None
    assert record["passed"] is False
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == stdout


def test_task_accepts_the_original_replay_request_and_retains_experiment_evidence(
    tmp_path: Path,
) -> None:
    example = Path(__file__).resolve().parents[1] / "examples/sum.json"
    output_dir = tmp_path / "runs"
    process = cli("task", str(example), "--output-dir", str(output_dir), cwd=tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    _, record = saved_run(output_dir)
    assert record["task_id"] == "manual/sum-v1"
    assert record["grade"]["passed_cases"] == 3
    assert record["encoding"] == "replay/utf8-v1"
    assert record["context_tokens"]
    assert len(record["candidates"]) == 1
    assert [step["verification"] for step in record["trajectory"]] == [
        False,
        True,
        False,
    ]
    assert all(
        step["model_call"]["generation"]["source"] == "replayed"
        and step["model_call"]["generation"]["logprobs"] is None
        for step in record["trajectory"]
    )
    assert "Grade: passed (3/3)" in process.stdout
    old_command = cli(str(example), cwd=tmp_path)
    assert old_command.returncode == 2


@pytest.mark.parametrize(
    ("replies", "bwrap", "exit_code", "status", "passed"),
    [
        (
            [write_file("def solve(n): return 3\n"), submit()],
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
def test_task_preserves_failure_and_infrastructure_exit_codes(
    tmp_path: Path,
    replies: list[str],
    bwrap: str,
    exit_code: int,
    status: str,
    passed: bool | None,
) -> None:
    config = replay_config(tmp_path, replies)
    process = cli(
        "task",
        str(task_file(tmp_path)),
        "-c",
        str(config),
        "--set",
        "budget.max_steps=2",
        "--set",
        f"environment.bwrap={bwrap}",
        cwd=tmp_path,
    )
    assert process.returncode == exit_code, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["status"] == status
    assert (
        record["grade"]["passed"] if record["grade"] is not None else None
    ) is passed
    assert f"Outcome: {status}" in process.stdout


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
def test_task_rejects_invalid_or_scripted_requests_before_creating_a_run(
    tmp_path: Path,
    invalid_request: dict[str, object],
) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {"task": json.loads(task_file(tmp_path).read_text())} | invalid_request
        )
    )
    process = cli("task", str(request), cwd=tmp_path)
    assert process.returncode == 2
    assert "error:" in process.stderr and "Traceback" not in process.stderr
    assert not (tmp_path / "runs").exists()


def test_ctrl_c_while_recording_workspace_keeps_the_completed_action(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = replay_config(
        tmp_path,
        [
            bash(
                'python3 -c \'from pathlib import Path; d=Path("data"); d.mkdir(); '
                "[(d/str(i)).write_text(str(i)) for i in range(1000)]'; "
                "printf 'built files\\n'"
            ),
            submit(),
        ],
    )
    output_dir = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bashlet",
            "agent",
            "Create files",
            "--workdir",
            str(source),
            "-c",
            str(config),
            "--output-dir",
            str(output_dir),
            "--set",
            "environment.timeout_seconds=15",
        ],
        cwd=tmp_path,
        env=os.environ | {"NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        # An initially empty workspace has no blobs. Its first new blob means
        # the completed command's workspace is now being recorded.
        while not list(output_dir.glob("*/blobs/*")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "Workspace recording never started"
            time.sleep(0.001)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 130, stdout + stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "interrupted"
    assert len(record["trajectory"]) == record["usage"]["steps"] == 1
    step = record["trajectory"][0]
    assert step["observation"]["status"] == "ok"
    assert step["observation"]["output"] == "built files\n"
    assert step["model_call"]["generation"]["tokens"]
    entry = step["workspace_manifest"]["data/999"]
    assert (run_dir / "blobs" / entry["sha256"]).read_text() == "999"
    assert "Step 1" in stdout and "built files" in stdout
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == stdout


def test_ctrl_c_while_copying_source_saves_the_aborted_preparation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for index in range(1000):
        (source / str(index)).write_text(str(index))
    config = replay_config(tmp_path, [submit()])
    output_dir = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bashlet",
            "agent",
            "Submit",
            "--workdir",
            str(source),
            "-c",
            str(config),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        env=os.environ | {"NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not list(output_dir.glob("*/workspace/*")):
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "Source copying never started"
            time.sleep(0.001)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 130, stdout + stderr
    assert "Traceback" not in stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "interrupted"
    assert record["reason"] == "Workspace copy interrupted"
    assert record["trajectory"] == [] and record["usage"]["steps"] == 0
    assert record["initial_workspace_manifest"]
    assert "Workspace copy interrupted" in stdout
    assert (source / "999").read_text() == "999"
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == stdout


def test_ctrl_c_while_printing_a_step_finishes_its_visible_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    config = replay_config(
        tmp_path, [bash("python3 -c 'print(\"x\" * 200000)'"), submit()]
    )
    output_dir = tmp_path / "runs"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "bashlet",
            "agent",
            "Read output",
            "--workdir",
            str(source),
            "-c",
            str(config),
            "--output-dir",
            str(output_dir),
            "--set",
            "protocol.observation_head_chars=200000",
            "--set",
            "protocol.observation_tail_chars=1",
            "--set",
            "environment.output_bytes=262144",
        ],
        cwd=tmp_path,
        env=os.environ | {"NO_COLOR": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while True:
            paths = list(output_dir.glob("*/run.json"))
            if paths:
                try:
                    if json.loads(paths[0].read_text())["trajectory"]:
                        break
                except FileNotFoundError:
                    pass  # The run directory can precede its first record.
            assert process.poll() is None, process.communicate()
            assert time.monotonic() < deadline, "The step was never recorded"
            time.sleep(0.01)
        # Leave stdout unread so the long observation fills the pipe and the
        # subprocess is still printing this step when it receives Ctrl-C.
        time.sleep(0.1)
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 130, stderr
    run_dir, record = saved_run(output_dir)
    assert record["status"] == "interrupted" and len(record["trajectory"]) == 1
    assert record["trajectory"][0]["feedback"] in stdout
    replayed = cli("show", str(run_dir), cwd=tmp_path)
    assert replayed.returncode == 0, replayed.stderr
    assert replayed.stdout == stdout
