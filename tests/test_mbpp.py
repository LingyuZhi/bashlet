"""Task preparation and consumption through the CLI and real isolated execution."""

import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from adapters import bash, submit, write_file
from test_cli import cli, replay_config, saved_run


def source_files(tmp_path: Path, rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    source = tmp_path / "source.json"
    source.write_text(json.dumps(rows))
    lock = tmp_path / "source-lock.json"
    lock.write_text(
        json.dumps(
            {
                "dataset": "google-research/mbpp",
                "variant": "sanitized",
                "revision": "a" * 40,
                "url": "https://example.invalid/sanitized-mbpp.json",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "conversion": "mbpp-json-functions-v1",
            }
        )
    )
    return source, lock


def double_row(task_id: int = 602) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "prompt": "Write a function to double an integer.",
        "code": "def double(n):\n    return n * 2\n",
        "test_imports": [],
        "test_list": [
            "assert double(2) == 4",
            "assert double(-37) == -74",
            "assert double(5) == 10",
        ],
    }


def prepare(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    source, lock = source_files(tmp_path, rows)
    output = tmp_path / "prepared"
    process = cli(
        "prepare-mbpp",
        "--source-lock",
        str(lock),
        "--source-file",
        str(source),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return output


def test_prepared_task_runs_with_v3_and_keeps_independent_grading_private(
    tmp_path: Path,
) -> None:
    output = prepare(tmp_path, [double_row()])
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["tasks"][0]
    assert entry["status"] == "verified"
    assert entry["family_id"] == "mbpp/602" and entry["split"] == "train"
    task_path = output / entry["task_file"]
    task = json.loads(task_path.read_text())
    assert task["feedback_command"] == "python3 feedback_tests.py"
    assert task["implementation_paths"] == ["solution.py"]
    assert "double(n)" in task["prompt"]
    visible = json.dumps(task["initial_files"]) + task["prompt"]
    assert "-37" not in visible and "-74" not in visible
    assert "return n * 2" not in visible

    model = replay_config(
        tmp_path,
        [
            write_file("def double(n): return 4\n"),
            bash("python3 feedback_tests.py"),
            submit(),
        ],
    )
    process = cli("task", str(task_path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == 1, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["protocol"]["name"] == "bash-block-v3"
    assert record["trajectory"][1]["observation"]["exit_code"] == 0
    assert record["grade"]["passed"] is False
    assert record["grading_protocol"] == "json-function-v1"
    assert "-37" not in json.dumps(record["trajectory"])


def test_augmented_source_labels_cannot_put_a_training_family_in_final_test(
    tmp_path: Path,
) -> None:
    row = double_row()
    row.update(task_id="Mbpp/602", split="test")
    output = prepare(tmp_path, [row, double_row(511), double_row(11), double_row(2)])
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["counts"] == {"train": 1, "dev": 1, "test": 1}
    prompt = next(
        entry for entry in manifest["tasks"] if entry["original_task_id"] == 2
    )
    assert prompt["reason"] == "reserved_prompt_example"
    model = replay_config(
        tmp_path,
        [
            write_file("def double(n): return n + n\n"),
            submit(),
        ],
    )
    wrong = cli(
        "task",
        str(manifest_path),
        "--task-id",
        "mbpp/602",
        "--split",
        "test",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert wrong.returncode == 2 and "split" in wrong.stderr
    assert not (tmp_path / "runs").exists()
    right = cli(
        "task",
        str(manifest_path),
        "--task-id",
        "mbpp/602",
        "--split",
        "train",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert right.returncode == 0, right.stdout + right.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["task_origin"]["family_id"] == "mbpp/602"
    assert record["task_origin"]["split"] == "train"
    assert record["task_origin"]["source_sha256"] == manifest["source"]["sha256"]


@pytest.mark.parametrize("change", ["family_conflict", "training_in_test"])
def test_task_set_rejects_cross_split_family_assignments(
    tmp_path: Path,
    change: str,
) -> None:
    output = prepare(tmp_path, [double_row()])
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["tasks"][0]
    if change == "family_conflict":
        manifest["tasks"].append(
            entry | {"task_id": "mbpp/602/rewrite", "split": "test"}
        )
    else:
        entry["split"] = "test"
    path.write_text(json.dumps(manifest))
    process = cli(
        "task",
        str(path),
        "--task-id",
        "mbpp/602",
        "--split",
        "test",
        cwd=tmp_path,
    )
    assert process.returncode == 2 and "MBPP family split" in process.stderr
    assert not (tmp_path / "runs").exists()


def test_derived_task_and_its_trajectory_keep_the_original_family(
    tmp_path: Path,
) -> None:
    output = prepare(tmp_path, [double_row()])
    manifest = json.loads((output / "manifest.json").read_text())
    task = json.loads((output / manifest["tasks"][0]["task_file"]).read_text())
    origin = task["origin"]
    task["task_id"] = "mbpp/602/repair-1"
    task["prompt"] = "Repair double(n) in solution.py, run feedback tests and submit."
    task["initial_files"]["solution.py"] = "def double(n): return n\n"
    derived = tmp_path / "derived.json"
    derived.write_text(json.dumps(task))
    model = replay_config(
        tmp_path,
        [
            bash("python3 feedback_tests.py"),
            write_file("def double(n): return n + n\n"),
            bash("python3 feedback_tests.py"),
            submit(),
        ],
    )
    process = cli("task", str(derived), "-c", str(model), cwd=tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["task_id"] == "mbpp/602/repair-1"
    assert record["task_origin"] == origin
    assert record["repair_attempts"][0]["passed"] is True


@pytest.mark.parametrize("change", ["task_bytes", "incomplete"])
def test_task_selection_rejects_changed_materials_and_incomplete_preparation(
    tmp_path: Path,
    change: str,
) -> None:
    output = prepare(tmp_path, [double_row()])
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    if change == "task_bytes":
        task_path = output / manifest["tasks"][0]["task_file"]
        task_path.write_text(task_path.read_text() + "\n")
    else:
        manifest["status"] = "preparing"
        path.write_text(json.dumps(manifest))
    process = cli(
        "task",
        str(path),
        "--task-id",
        "mbpp/602",
        "--split",
        "train",
        cwd=tmp_path,
    )
    assert process.returncode == 2
    assert ("SHA-256" if change == "task_bytes" else "incomplete") in process.stderr
    assert not (tmp_path / "runs").exists()


def test_wrong_references_are_excluded_after_actual_feedback_and_grading(
    tmp_path: Path,
) -> None:
    wrong_hidden = double_row(602) | {"code": "def double(n): return 4\n"}
    wrong_feedback = double_row(603) | {
        "code": "def double(n): return 999 if n == 2 else n * 2\n"
    }
    output = prepare(tmp_path, [wrong_hidden, wrong_feedback])
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"train": 0, "dev": 0, "test": 0}
    for entry in manifest["tasks"]:
        assert entry["status"] == "excluded" and entry["reason"] == "reference_failed"
        record = json.loads(
            (output / entry["verification_run"] / "run.json").read_text()
        )
        assert record["isolation"]["passed"]
        assert record["status"] == "completed"
        assert "task_file" not in entry
        if entry["original_task_id"] == 603:
            assert record["grade"]["passed"]
            assert record["trajectory"][0]["observation"]["exit_code"] != 0
        else:
            assert not record["grade"]["passed"]


def test_preparation_fails_closed_on_unavailable_isolation(tmp_path: Path) -> None:
    source, lock = source_files(tmp_path, [double_row()])
    output = tmp_path / "prepared"
    process = cli(
        "prepare-mbpp",
        "--source-lock",
        str(lock),
        "--source-file",
        str(source),
        "--output-dir",
        str(output),
        "--set",
        "environment.bwrap=/missing/bwrap",
        cwd=tmp_path,
    )
    assert process.returncode == 2 and "did not complete" in process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] != "complete"
    assert not list((output / "private/tasks").glob("*/*.json"))


@pytest.mark.parametrize("change", ["hash", "duplicate_id"])
def test_source_integrity_errors_do_not_produce_a_task_set(
    tmp_path: Path,
    change: str,
) -> None:
    rows = [double_row()]
    if change == "duplicate_id":
        rows.append(double_row() | {"task_id": "Mbpp/602", "split": "test"})
    source, lock = source_files(tmp_path, rows)
    if change == "hash":
        source.write_text(source.read_text() + "\n")
    output = tmp_path / "prepared"
    process = cli(
        "prepare-mbpp",
        "--source-lock",
        str(lock),
        "--source-file",
        str(source),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 2
    assert (
        "SHA-256" if change == "hash" else "Duplicate MBPP family"
    ) in process.stderr
    assert not output.exists()


def test_unsupported_and_duplicate_inputs_have_explicit_exclusion_reasons(
    tmp_path: Path,
) -> None:
    tuples = double_row(602) | {
        "test_list": [
            "assert double((1, 2)) == (1, 2, 1, 2)",
            "assert double((3,)) == (3, 3)",
        ]
    }
    duplicates = double_row(603) | {
        "test_list": [
            "assert double(2) == 4",
            "assert double(2) == 4",
        ]
    }
    output = prepare(tmp_path, [tuples, duplicates])
    manifest = json.loads((output / "manifest.json").read_text())
    assert [entry["reason"] for entry in manifest["tasks"]] == [
        "unsupported_json_type: tuple",
        "no_independent_grading_inputs",
    ]
    assert not (output / "private/reference-validation").exists()


def test_download_uses_environment_proxy_and_recovers_from_a_transient_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, lock = source_files(tmp_path, [double_row()])
    data = json.loads(lock.read_text())
    data["url"] = "http://mbpp-source.invalid/sanitized-mbpp.json"
    lock.write_text(json.dumps(data))
    requests: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            requests.append(self.path)
            body = b"retry" if len(requests) == 1 else source.read_bytes()
            self.send_response(503 if len(requests) == 1 else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("NO_PROXY", "")
    monkeypatch.setenv("no_proxy", "")
    try:
        process = cli(
            "prepare-mbpp",
            "--source-lock",
            str(lock),
            "--output-dir",
            str(tmp_path / "prepared"),
            cwd=tmp_path,
            timeout=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert process.returncode == 0, process.stdout + process.stderr
    assert requests == [data["url"], data["url"]]
