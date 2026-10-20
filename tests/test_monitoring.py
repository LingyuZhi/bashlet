"""Monitoring through the approved CLI seam, without network or credentials."""

import json
from pathlib import Path

import pytest
import yaml
from test_cli import cli
from test_mbpp import double_row, prepare


def external_sdks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\nimport monitoring_sdk\nsys.modules['wandb'] = monitoring_sdk\n"
    )
    monkeypatch.setenv("PYTHONPATH", f"{tmp_path}:{Path(__file__).parent.resolve()}")
    monkeypatch.setenv("WANDB_API_KEY", "secret-test-key")


def baseline_record(tmp_path: Path) -> Path:
    path = tmp_path / "original/evaluation.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "evidence_kind": "experimental",
                "snapshot": "pytrio/Qwen/Qwen3.5-4B@unversioned-base",
                "comparison_sha256": "fixed-conditions",
                "config": {"api_key": "secret"},
                "summary": {
                    "planned_tasks": 8,
                    "passed_tasks": 5,
                    "final_pass_rate": 0.625,
                    "first_implementation_passed": 6,
                    "first_implementation_pass_rate": 0.75,
                    "missing_task_ids": [],
                    "first_grading_missing_task_ids": [],
                    "outcomes": {
                        "passed": 5,
                        "protocol_error": 2,
                        "budget_exhausted": 1,
                    },
                    "recovery": {
                        "denominator": 1,
                        "recovered_task_ids": [],
                        "rate": 0.0,
                    },
                },
                "requests": {
                    "count": 41,
                    "pending_count": 0,
                },
                "tasks": [{"private": "HIDDEN_TEST_AND_TOKENS"}],
            }
        )
    )
    return path


def test_historical_baseline_import_retains_identity_and_uploads_only_allowlisted_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_sdks(tmp_path, monkeypatch)
    original = baseline_record(tmp_path)
    before = original.read_bytes()
    config = tmp_path / "monitor.yaml"
    config.write_text(yaml.safe_dump({"project": "bashlet", "group": "comparison"}))
    output = tmp_path / "import"
    process = cli(
        "monitor",
        "import-evaluation",
        str(original),
        "--monitor-config",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    state = json.loads((output / "monitor.json").read_text())
    remote = json.loads((output / "remote.json").read_text())
    assert original.read_bytes() == before
    assert state["entity"] == "test-account"
    assert remote["config"]["historical_import"] is True
    assert remote["config"]["source_run"] == str(original.parent)
    assert remote["config"]["comparison_sha256"] == "fixed-conditions"
    row = remote["rows"][-1]
    assert row["eval/planned_tasks"] == 8
    assert row["eval/passed_tasks"] == 5
    assert row["eval/final_pass_rate"] == 0.625
    assert row["requests/count"] == 41
    assert row["requests/pending_count"] == 0
    assert not any(s in json.dumps(remote) for s in ("secret", "HIDDEN_TEST"))
    assert remote["settings"]["console"] == "off"
    assert remote["settings"]["save_code"] is False


def test_network_failure_retains_metrics_and_sync_replays_without_duplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_sdks(tmp_path, monkeypatch)
    original = baseline_record(tmp_path)
    config = tmp_path / "monitor.yaml"
    config.write_text("group: recovered\n")
    output = tmp_path / "import"
    monkeypatch.setenv("FAKE_WANDB_LOG_FAIL", "1")
    process = cli(
        "monitor",
        "import-evaluation",
        str(original),
        "--monitor-config",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stderr
    before = json.loads((output / "monitor.json").read_text())
    assert before["sync_status"] == "pending"
    journal = (output / "metrics.jsonl").read_bytes()
    assert len(journal.splitlines()) == 1
    monkeypatch.delenv("FAKE_WANDB_LOG_FAIL")
    for _ in range(2):
        synced = cli("monitor", "sync", str(output), cwd=tmp_path)
        assert synced.returncode == 0, synced.stdout + synced.stderr
    state = json.loads((output / "monitor.json").read_text())
    assert state["sync_status"] == "synced"
    assert state["id"] == before["id"]
    assert (output / "metrics.jsonl").read_bytes() == journal
    remote = json.loads((output / "remote.json").read_text())
    assert len(remote["rows"]) == 1
    assert remote["rows"][0]["eval/planned_tasks"] == 8


def test_development_monitoring_preserves_comparison_and_fixed_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    external_sdks(tmp_path, monkeypatch)
    prepared = prepare(tmp_path, [double_row(511), double_row(512)])
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "dev",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511", "mbpp/512"],
            }
        )
    )
    model_config = tmp_path / "model.yaml"
    model_config.write_text("model:\n  class: adapters.EvaluationModel\n")
    config = tmp_path / "monitor.yaml"
    config.write_text("group: evaluated\n")
    records = []
    for monitor_options in ([], ["--monitor-config", str(config)]):
        process = cli(
            "evaluate",
            str(plan),
            "--snapshot",
            "test/evaluation-v1",
            "--evidence-kind",
            "test-double",
            "-c",
            str(model_config),
            "--output-dir",
            str(tmp_path / "evaluations"),
            *monitor_options,
            cwd=tmp_path,
            timeout=45,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        path = Path(process.stdout.strip().splitlines()[-1])
        records.append(json.loads(path.read_text()))
    assert records[0]["comparison_sha256"] == records[1]["comparison_sha256"]
    remote = json.loads((path.parent / "remote.json").read_text())
    assert remote["rows"][-1]["eval/planned_tasks"] == 2
    assert remote["rows"][-1]["eval/passed_tasks"] == 2
    assert remote["rows"][-1]["eval/outcome/not_started"] == 0
    assert remote["rows"][-1]["eval/outcome/running"] == 0
    assert remote["config"]["comparison_sha256"] == records[1]["comparison_sha256"]
    assert all(row["eval/planned_tasks"] == 2 for row in remote["rows"])
    assert remote["rows"][-1]["eval/generated_tokens"] > 0
    assert remote["rows"][-1]["eval/wall_seconds"] > 0

    monkeypatch.setenv("FAKE_WANDB_AUTH_FAIL", "1")
    refused = tmp_path / "refused"
    process = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        "test/evaluation-v1",
        "--evidence-kind",
        "test-double",
        "-c",
        str(model_config),
        "--output-dir",
        str(refused),
        "--monitor-config",
        str(config),
        cwd=tmp_path,
    )
    assert process.returncode != 0
    failed_path = next(refused.glob("*/evaluation.json"))
    failed = json.loads(failed_path.read_text())
    assert failed["status"] == "incomplete"
    assert failed["summary"]["planned_tasks"] == 2
    assert len(failed["summary"]["missing_task_ids"]) == 2
    assert not (failed_path.parent / "episodes").exists()
