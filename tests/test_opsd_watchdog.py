"""CLI-level safety checks for the issue 32 run watchdog."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_cli import cli as project_cli

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, proof_status: str = "complete") -> dict[str, Any]:
    root = tmp_path / "repo"
    configs = root / "configs"
    configs.mkdir(parents=True)
    frozen_configs = {
        configs / "opsd-expanded-v1.yaml": "formal configuration\n",
        configs / "opsd-expanded-preflight-v1.yaml": "preflight configuration\n",
    }
    for path, content in frozen_configs.items():
        path.write_text(content)

    run_root = root / "runs" / "issue32"
    monitor_root = run_root / "monitor"
    journal = root / "runs" / "requests.sqlite3"
    journal.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(journal) as connection:
        connection.execute(
            "CREATE TABLE requests (id TEXT, request_group TEXT, state TEXT)"
        )

    configuration_hashes = {
        str(path.resolve()): _digest(path) for path in frozen_configs
    }
    paths = {
        "monitor_root": "runs/issue32/monitor",
        "workflow": "runs/issue32/workflow.json",
        "preflight_pilot": "runs/issue32/preflight/pilot.json",
        "preflight_training": "runs/issue32/preflight/training/opsd.json",
        "preflight_validation": "runs/issue32/preflight/checkpoint-validation.json",
        "formal_pilot": "runs/issue32/formal/pilot.json",
        "formal_training": "runs/issue32/formal/training/opsd.json",
        "journal": "runs/requests.sqlite3",
    }
    monitor_config = {
        "schema_version": 1,
        "repository_root": "..",
        "monitor_session": "bashlet-monitor-test",
        "worker": {
            "session": "bashlet-worker-test",
            "log": "runs/issue32/workflow.log",
            "command": ["bash", "scripts/train-expanded-opsd.sh", "--resume"],
        },
        "paths": paths,
        "frozen_configurations": [
            "configs/opsd-expanded-v1.yaml",
            "configs/opsd-expanded-preflight-v1.yaml",
        ],
        "journal_groups": [
            "opsd-expanded-preflight-v1",
            "opsd-expanded-train-v1",
            "opsd-expanded-dev-v1",
        ],
        "timing": {
            "poll_seconds": 1,
            "backoff_min_seconds": 60,
            "backoff_max_seconds": 600,
        },
    }
    config_path = configs / "monitor.yaml"
    config_path.write_text(yaml.safe_dump(monitor_config, sort_keys=False))

    workflow = {
        "status": "incomplete",
        "stage": "preflight-verification",
        "configuration_sha256": configuration_hashes,
    }
    _write_json(run_root / "workflow.json", workflow)
    preflight = {
        "status": "complete",
        "training_complete": True,
        "epochs_completed": 1,
        "optimizer_step": 3,
        "training_tasks": [{"status": "completed"}],
    }
    _write_json(run_root / "preflight" / "pilot.json", preflight)
    _write_json(
        run_root / "preflight" / "training" / "opsd.json",
        {
            "pending_update": None,
            "batches": [],
        },
    )
    _write_json(
        run_root / "preflight" / "checkpoint-validation.json",
        {"status": proof_status, "native": {"status": proof_status}},
    )
    worker_log = run_root / "workflow.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    worker_log.write_text("previous worker output\n")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    service_state = tmp_path / "process-services.json"
    service_state.write_text(
        json.dumps({"session": False, "pane": "1\t4242\tbash", "ps": ""})
    )
    tmux_log = tmp_path / "tmux-calls.jsonl"
    tmux_log.touch()
    tmux = fake_bin / "tmux"
    tmux.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "state_path = os.environ['WATCHDOG_SERVICE_STATE']\n"
        "state = json.loads(pathlib.Path(state_path).read_text())\n"
        "args = sys.argv[1:]\n"
        "with pathlib.Path(os.environ['WATCHDOG_TMUX_LOG']).open('a') as out:\n"
        "    out.write(json.dumps(args) + chr(10))\n"
        "if args[0] == 'has-session':\n"
        "    sys.exit(0 if state['session'] else 1)\n"
        "if args[0] == 'list-panes':\n"
        "    print(state['pane'])\n"
        "    sys.exit(0)\n"
        "if args[0] in {'respawn-pane', 'new-session'}:\n"
        "    sys.exit(state.get('restart_code', 0))\n"
        "sys.exit(2)\n"
    )
    tmux.chmod(0o700)
    ps = fake_bin / "ps"
    ps.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import sys\n"
        "state_path = os.environ['WATCHDOG_SERVICE_STATE']\n"
        "state = json.loads(pathlib.Path(state_path).read_text())\n"
        "sys.stdout.write(state['ps'])\n"
    )
    ps.chmod(0o700)
    return {
        "root": root,
        "run_root": run_root,
        "monitor_root": monitor_root,
        "config_path": config_path,
        "service_state": service_state,
        "tmux_log": tmux_log,
        "fake_bin": fake_bin,
        "journal": journal,
        "worker_log": worker_log,
        "preflight": preflight,
        "configuration_hashes": configuration_hashes,
    }


def _cli(fixture: dict[str, Any]) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "NO_COLOR": "1",
        "PYTHONPATH": str(ROOT / "src"),
        "PATH": str(fixture["fake_bin"]) + os.pathsep + os.environ["PATH"],
        "WATCHDOG_SERVICE_STATE": str(fixture["service_state"]),
        "WATCHDOG_TMUX_LOG": str(fixture["tmux_log"]),
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "bashlet",
            "watch-opsd",
            "--config",
            str(fixture["config_path"]),
            "--once",
        ],
        cwd=fixture["root"],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _events(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    path = fixture["monitor_root"] / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _tmux_calls(fixture: dict[str, Any]) -> list[list[str]]:
    return [json.loads(line) for line in fixture["tmux_log"].read_text().splitlines()]


def _formal_interruption(
    fixture: dict[str, Any], pilot: dict[str, Any], training: dict[str, Any]
) -> None:
    run_root: Path = fixture["run_root"]
    proof = run_root / "preflight" / "checkpoint-validation.json"
    _write_json(
        run_root / "workflow.json",
        {
            "status": "incomplete",
            "stage": "formal",
            "configuration_sha256": fixture["configuration_hashes"],
            "preflight_sha256": _digest(proof),
        },
    )
    _write_json(run_root / "formal" / "pilot.json", pilot)
    _write_json(run_root / "formal" / "training" / "opsd.json", training)
    events = fixture["monitor_root"] / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "event": "heartbeat",
                "worker_state": "stopped",
                "restart_not_before": "2020-01-01T00:00:00+00:00",
                "restart_attempts": 0,
            }
        )
        + "\n"
    )


def test_failed_checkpoint_gate_is_recorded_without_restart(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, proof_status="failed")
    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"decision": "blocked"}
    assert (
        _events(fixture)[-1]["reason"]
        == "checkpoint validation failed and needs review"
    )
    assert all(
        call[0] not in {"respawn-pane", "new-session"} for call in _tmux_calls(fixture)
    )


def test_frozen_configuration_drift_blocks_before_resume(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["root"] / "configs" / "opsd-expanded-v1.yaml").write_text(
        "changed configuration\n"
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert _events(fixture)[-1]["reason"] == "frozen configuration changed"
    assert all(
        call[0] not in {"respawn-pane", "new-session"} for call in _tmux_calls(fixture)
    )


def test_pending_journal_request_blocks_a_dead_worker_restart(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with sqlite3.connect(fixture["journal"]) as connection:
        connection.execute(
            "INSERT INTO requests VALUES (?, ?, ?)",
            ("request-in-flight", "opsd-expanded-preflight-v1", "pending"),
        )
    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert _events(fixture)[-1]["pending_request_count"] == 1
    assert all(
        call[0] not in {"respawn-pane", "new-session"} for call in _tmux_calls(fixture)
    )


def test_unresolved_update_blocks_a_dead_worker_restart(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture["run_root"] / "preflight" / "training" / "opsd.json",
        {"pending_update": {"optimizer_step": 4}, "batches": []},
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert "unresolved remote update" in _events(fixture)[-1]["reason"]
    assert all(
        call[0] not in {"respawn-pane", "new-session"} for call in _tmux_calls(fixture)
    )


def test_rejected_supervision_blocks_a_dead_worker_restart(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _write_json(
        fixture["run_root"] / "preflight" / "pilot.json",
        {"status": "incomplete", "training_tasks": [{"status": "rejected"}]},
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert "rejected supervision" in _events(fixture)[-1]["reason"]
    assert all(
        call[0] not in {"respawn-pane", "new-session"} for call in _tmux_calls(fixture)
    )


def test_interrupted_update_with_durable_receipt_can_resume(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _formal_interruption(
        fixture,
        {
            "status": "interrupted",
            "error_type": "KeyboardInterrupt",
            "training_tasks": [{"status": "running", "batch_index": 0}],
        },
        {
            "pending_update": None,
            "optimizer_step": 1,
            "student_policy": "student-1",
            "last_update": {
                "optimizer_step": 1,
                "student_policy": "student-1",
            },
            "batches": [
                {
                    "index": 0,
                    "status": "interrupted",
                    "reason": "monitor_interrupted",
                    "update": {
                        "optimizer_step": 1,
                        "student_policy": "student-1",
                    },
                }
            ],
        },
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"
    assert any(call[0] == "new-session" for call in _tmux_calls(fixture))


def _reconciled_probe_history(
    fixture: dict[str, Any],
    *,
    isolation_failure: bool = False,
) -> None:
    root: Path = fixture["run_root"]
    formal = root / "formal"
    updates: list[dict[str, Any]] = [
        {"optimizer_step": step, "student_policy": f"student-{step}"} for step in (1, 2)
    ]
    batches: list[dict[str, Any]] = [
        {"index": index, "status": "completed", "update": update}
        for index, update in enumerate(updates)
    ]
    row: dict[str, Any] = {
        "visit_id": "epoch-1/task-3",
        "task_id": "task-3",
        "status": "not_started",
        "attempts": [],
    }
    training = {
        "status": "rejected",
        "pending_update": None,
        "optimizer_step": 2,
        "student_policy": "student-2",
        "last_update": updates[-1],
        "batches": batches,
    }
    pilot = {"status": "incomplete", "training_tasks": [row]}
    for index in (2, 3):
        episode = formal / "rollouts" / f"failed-{index}"
        reason = "Sample logprobs disagree with raw-token model scoring"
        if isolation_failure:
            reason = (
                "infrastructure_error: bwrap: Can't mount proc on /newroot/proc: "
                "Operation not permitted\n"
            )
        _write_json(
            episode / "run.json",
            {
                "task_id": "task-3",
                "status": "infrastructure_error"
                if isolation_failure
                else "model_error",
                "reason": reason,
                "trajectory": [],
                "usage": {"steps": 0, "generated_tokens": 0},
                **(
                    {
                        "context_tokens": [],
                        "isolation": {"passed": False, "reason": reason},
                    }
                    if isolation_failure
                    else {}
                ),
            },
        )
        batch = {
            "index": index,
            "status": "rejected",
            "episode": str(episode),
            "student_policy": "student-2",
            "interaction_steps": 0,
            "supervised_tokens": 0,
        }
        batches.append(batch)
        attempt = {
            "visit_id": row["visit_id"],
            "task_id": "task-3",
            "status": "rejected",
            "batch_index": index,
            "episode": str(episode),
        }
        before_pilot = {"training_tasks": [attempt]}
        recovery = root / "recoveries" / f"probe-{index}"
        originals = {
            "formal/pilot.json": before_pilot,
            "formal/training/opsd.json": training,
            "formal/training/backend.json": {"optimizer_step": 2},
            "workflow.json": {"status": "failed", "stage": "formal"},
            f"formal/rollouts/failed-{index}/run.json": json.loads(
                (episode / "run.json").read_text()
            ),
        }
        archived = {}
        for relative, value in originals.items():
            copy = recovery / "before" / relative
            _write_json(copy, value)
            archived[relative] = {"copy": str(copy), "sha256": _digest(copy)}
        receipt = recovery / "recovery.json"
        _write_json(
            receipt,
            {
                "action": (
                    "explicit user-authorized retry of a visit "
                    "rejected before task rollout"
                ),
                "status": "reconciled; waiting for existing resume entrypoint",
                "visit_id": row["visit_id"],
                "optimizer_step": 2,
                "pending_requests": 0,
                "pending_update": None,
                "rejected_batch_index": index,
                "rejected_episode": str(episode),
                "rejected_episode_sha256": _digest(episode / "run.json"),
                "reason": reason,
                "archived": archived,
            },
        )
        attempt["retry_receipt"] = str(receipt)
        row["attempts"].append(attempt)
        row["retry_receipt"] = str(receipt)
    _formal_interruption(fixture, pilot, training)


@pytest.mark.parametrize(
    "damage",
    [None, "context", "trajectory", "tokens", "isolation", "reason"],
)
def test_isolation_failure_retry_requires_evidence_before_model_calls(
    tmp_path: Path,
    damage: str | None,
) -> None:
    fixture = _fixture(tmp_path)
    _reconciled_probe_history(fixture, isolation_failure=True)
    root: Path = fixture["run_root"]
    episode_path = root / "formal/rollouts/failed-3/run.json"
    episode = json.loads(episode_path.read_text())
    if damage == "context":
        episode["context_tokens"] = [1]
    elif damage == "trajectory":
        episode["trajectory"] = [{}]
    elif damage == "tokens":
        episode["usage"]["generated_tokens"] = 1
    elif damage == "isolation":
        episode["isolation"]["passed"] = True
    elif damage == "reason":
        episode["isolation"]["reason"] = "another failure"
    _write_json(episode_path, episode)
    receipt_path = root / "recoveries/probe-3/recovery.json"
    receipt = json.loads(receipt_path.read_text())
    relative = "formal/rollouts/failed-3/run.json"
    archive = Path(receipt["archived"][relative]["copy"])
    _write_json(archive, episode)
    receipt["archived"][relative]["sha256"] = _digest(archive)
    receipt["rejected_episode_sha256"] = _digest(episode_path)
    _write_json(receipt_path, receipt)
    training_path = root / "formal/training/opsd.json"
    frozen = training_path.read_bytes()

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == (
        "restart_started" if damage is None else "blocked"
    )
    assert training_path.read_bytes() == frozen
    if damage is not None:
        assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


@pytest.mark.parametrize(
    "damage",
    [None, "pending", "update", "fingerprint_mismatch", "payload_mismatch"],
)
@pytest.mark.parametrize("operation", ["sampling", "capability_probe"])
def test_sampling_transport_recovery_preserves_request_identity(
    tmp_path: Path,
    damage: str | None,
    operation: str,
) -> None:
    fixture = _fixture(tmp_path)
    _reconciled_probe_history(fixture)
    root: Path = fixture["run_root"]
    episode_path = root / "formal/rollouts/failed-3/run.json"
    episode = json.loads(episode_path.read_text())
    episode.update(
        status="infrastructure_error",
        policy="student-2",
        context_tokens=[1, 2],
        reason="RuntimeError: PyTRIO actor.request_failed (HTTP 0, attempt 1)",
        trajectory=[{} for _ in range(5)],
        usage={"steps": 5, "generated_tokens": 1250},
    )
    probe = operation == "capability_probe"
    if probe:
        episode.update(
            context_tokens=[],
            trajectory=[],
            reason=(
                "Capability probe failed: PyTRIO actor.request_failed "
                "(HTTP 0, attempt 1)"
            ),
            usage={"steps": 0, "generated_tokens": 0},
        )
    _write_json(episode_path, episode)
    training_path = root / "formal/training/opsd.json"
    training = json.loads(training_path.read_text())
    training.update(journal=str(fixture["journal"]), run_id="training-run")
    training["batches"][3].update(
        interaction_steps=0 if probe else 5,
        reason="Episode cannot supply valid supervision: infrastructure_error",
    )
    _write_json(training_path, training)
    receipt_path = root / "recoveries/probe-3/recovery.json"
    receipt = json.loads(receipt_path.read_text())
    receipt.update(
        action=(
            "explicit user-authorized retry after sampling transport failure "
            "without update"
        ),
        reason=episode["reason"],
        rejected_episode_sha256=_digest(episode_path),
        sampling_request={
            "id": "discarded-sample",
            "request_id": "remote-sample",
            "response_abandoned": True,
            "remote_outcome": "unknown",
            "update_submitted": False,
        },
    )
    pending: dict[str, Any] = {
        "id": "discarded-sample",
        "operation": operation,
        "state": "pending",
        "fingerprint": "original-sampling",
        "evidence": json.dumps(
            {
                "request_id": "remote-sample",
                "payload": {
                    "run_id": "training-run",
                    "model_path": "student-2",
                    "prompt_tokens": [1, 2],
                    **(
                        {"sampling": {"max_tokens": 1}, "scoring": False}
                        if probe
                        else {}
                    ),
                },
            }
        ),
    }
    if damage == "payload_mismatch":
        evidence = json.loads(pending["evidence"])
        if probe:
            evidence["payload"]["sampling"]["max_tokens"] = 2
        else:
            evidence["payload"]["prompt_tokens"] = [9, 9]
        pending["evidence"] = json.dumps(evidence)
    for relative, value in {
        "formal/training/opsd.json": training,
        "formal/rollouts/failed-3/run.json": episode,
        "pending-sampling-request.json": pending,
    }.items():
        path = receipt_path.parent / "before" / relative
        _write_json(path, value)
        receipt["archived"][relative] = {"copy": str(path), "sha256": _digest(path)}
    _write_json(receipt_path, receipt)
    evidence = json.loads(pending["evidence"])
    evidence["sampling_abandoned"] = {
        "recovery_receipt": str(receipt_path),
        "remote_outcome": "unknown",
        "response_will_not_be_used": True,
    }
    with sqlite3.connect(fixture["journal"]) as connection:
        for column in (
            "operation TEXT",
            "fingerprint TEXT",
            "evidence TEXT",
        ):
            connection.execute("ALTER TABLE requests ADD COLUMN " + column)
        connection.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
            (
                "discarded-sample",
                "opsd-expanded-train-v1",
                "pending" if damage == "pending" else "failed",
                "opsd_update" if damage == "update" else operation,
                "changed-sampling"
                if damage == "fingerprint_mismatch"
                else "original-sampling",
                json.dumps(evidence),
            ),
        )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    expected = "restart_started" if damage is None else "blocked"
    assert _events(fixture)[-1]["decision"] == expected
    assert json.loads(training_path.read_text())["optimizer_step"] == 2
    assert json.loads(episode_path.read_text())["trajectory"] == (
        [] if probe else [{} for _ in range(5)]
    )
    if damage is not None:
        assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


@pytest.mark.parametrize("advance", [False, True])
def test_reconciled_zero_action_failures_do_not_count_as_updates(
    tmp_path: Path,
    advance: bool,
) -> None:
    fixture = _fixture(tmp_path)
    _reconciled_probe_history(fixture)
    training_path = fixture["run_root"] / "formal/training/opsd.json"
    if advance:
        training = json.loads(training_path.read_text())
        update = {"optimizer_step": 3, "student_policy": "student-3"}
        training["batches"].append(
            {
                "index": 4,
                "status": "completed",
                "update": update,
            }
        )
        training.update(
            status="ready",
            optimizer_step=3,
            student_policy="student-3",
            last_update=update,
        )
        _write_json(training_path, training)
    frozen = training_path.read_bytes()

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"
    assert training_path.read_bytes() == frozen
    assert any(call[0] == "new-session" for call in _tmux_calls(fixture))


@pytest.mark.parametrize("damage", ["receipt", "episode", "archive", "new_failure"])
def test_unreconciled_or_changed_failed_attempt_blocks_restart(
    tmp_path: Path,
    damage: str,
) -> None:
    fixture = _fixture(tmp_path)
    _reconciled_probe_history(fixture)
    root: Path = fixture["run_root"]
    if damage == "receipt":
        (root / "recoveries/probe-3/recovery.json").unlink()
    elif damage == "episode":
        path = root / "formal/rollouts/failed-3/run.json"
        episode = json.loads(path.read_text())
        episode["trajectory"] = [{"action": "unexpected"}]
        _write_json(path, episode)
    elif damage == "archive":
        path = root / "recoveries/probe-3/before/formal/training/opsd.json"
        path.write_text("{}\n")
    else:
        path = root / "formal/training/opsd.json"
        training = json.loads(path.read_text())
        training["batches"].append(
            {
                "index": 4,
                "status": "rejected",
                "interaction_steps": 0,
                "supervised_tokens": 0,
            }
        )
        _write_json(path, training)

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


def test_real_monitor_interrupt_receipt_stays_resumable_after_new_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test_opsd_epochs import epoch_pilot

    fixture = _fixture(tmp_path)
    pilot_fixture = epoch_pilot.__wrapped__  # type: ignore[attr-defined]
    config, _ = pilot_fixture(tmp_path, monkeypatch)
    formal_root: Path = fixture["run_root"] / "formal"
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP", "3")

    interrupted = project_cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(formal_root),
        cwd=tmp_path,
        timeout=90,
    )
    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    first_pilot = json.loads((formal_root / "pilot.json").read_text())
    first_training = json.loads((formal_root / "training/opsd.json").read_text())
    assert first_training["batches"][2]["status"] == "interrupted"
    assert first_training["batches"][2]["reason"] == "monitor_interrupted"
    assert first_training["batches"][2]["update"] == first_training["last_update"]
    _formal_interruption(fixture, first_pilot, first_training)

    watched = _cli(fixture)

    assert watched.returncode == 0, watched.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"

    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP", "4")
    (tmp_path / "train-interrupted").unlink()
    interrupted_again = project_cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(formal_root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert interrupted_again.returncode == 2, (
        interrupted_again.stdout + interrupted_again.stderr
    )
    resumed_pilot = json.loads((formal_root / "pilot.json").read_text())
    resumed_training = json.loads((formal_root / "training/opsd.json").read_text())
    assert resumed_pilot["training_tasks"][2]["status"] == "completed"
    assert resumed_training["batches"][2]["status"] == "interrupted"
    assert resumed_training["batches"][3]["status"] == "interrupted"
    assert resumed_training["optimizer_step"] == 4

    watched_again = _cli(fixture)

    assert watched_again.returncode == 0, watched_again.stderr
    assert _events(fixture)[-1]["decision"] == "waiting_backoff"


def test_interrupted_update_with_pending_remote_receipt_stays_blocked(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _formal_interruption(
        fixture,
        {
            "status": "interrupted",
            "error_type": "KeyboardInterrupt",
            "training_tasks": [{"status": "running", "batch_index": 0}],
        },
        {
            "pending_update": {"optimizer_step": 1},
            "batches": [],
        },
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert "unresolved remote update" in _events(fixture)[-1]["reason"]
    assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


def test_known_wandb_startup_failure_can_resume_before_sampling(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _formal_interruption(
        fixture,
        {
            "status": "failed",
            "error_type": "ValueError",
            "reason": "W&B startup failed; resume is safe",
            "training_tasks": [{"status": "not_started"}],
        },
        {"pending_update": None, "batches": []},
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"


@pytest.mark.parametrize("failure", ["known", "unknown", "missing_formal", "preflight"])
def test_failed_workflow_requires_a_known_stage_failure(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = _fixture(tmp_path)
    _formal_interruption(
        fixture,
        {
            "status": "failed",
            "error_type": "ValueError",
            "reason": "W&B startup failed; resume is safe",
            "training_tasks": [{"status": "not_started"}],
        }
        if failure == "known"
        else {
            "status": "incomplete",
            "training_tasks": [{"status": "not_started"}],
        },
        {"pending_update": None, "batches": []},
    )
    path = fixture["run_root"] / "workflow.json"
    workflow = json.loads(path.read_text())
    workflow["status"] = "failed"
    if failure == "missing_formal":
        (fixture["run_root"] / "formal/pilot.json").unlink()
    if failure == "preflight":
        workflow["stage"] = "preflight-verification"
    _write_json(path, workflow)

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    decision = "restart_started" if failure == "known" else "blocked"
    assert _events(fixture)[-1]["decision"] == decision
    if failure != "known":
        assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


def test_unrecognized_validation_failure_stays_blocked(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _formal_interruption(
        fixture,
        {
            "status": "failed",
            "error_type": "ValueError",
            "reason": "Checkpoint probabilities do not align",
            "training_tasks": [{"status": "not_started"}],
        },
        {"pending_update": None, "batches": []},
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "blocked"
    assert "explicit failure" in _events(fixture)[-1]["reason"]
    assert all(call[0] != "new-session" for call in _tmux_calls(fixture))


def test_dead_pane_resumes_only_through_existing_entrypoint(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["service_state"].write_text(
        json.dumps({"session": True, "pane": "1\t4242\tbash", "ps": ""})
    )
    events = fixture["monitor_root"] / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "event": "heartbeat",
                "worker_state": "stopped",
                "restart_not_before": "2020-01-01T00:00:00+00:00",
                "restart_attempts": 0,
            }
        )
        + "\n"
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"
    restart = next(call for call in _tmux_calls(fixture) if call[0] == "respawn-pane")
    command = restart[-1]
    assert "scripts/train-expanded-opsd.sh" in command
    assert "--resume" in command
    assert ">> " in command
    assert "-k" not in restart

    second = _cli(fixture)
    assert second.returncode == 0, second.stderr
    assert _events(fixture)[-1]["decision"] == "waiting_backoff"
    assert sum(call[0] == "respawn-pane" for call in _tmux_calls(fixture)) == 1


def test_live_worker_with_stale_log_is_left_alone(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["service_state"].write_text(
        json.dumps(
            {
                "session": True,
                "pane": "0\t4242\tbash",
                "ps": (
                    "S 4242 bash scripts/train-expanded-opsd.sh --resume >> "
                    f"{fixture['worker_log']} 2>&1\n"
                ),
            }
        )
    )
    old_time = time.time() - 7200
    os.utime(fixture["worker_log"], (old_time, old_time))

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "worker_active"
    assert _events(fixture)[-1]["log_age_seconds"] >= 7200
    assert all(
        call[0] not in {"respawn-pane", "new-session", "kill-pane"}
        for call in _tmux_calls(fixture)
    )


def test_unrelated_train_opsd_process_does_not_block_resume(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    events = fixture["monitor_root"] / "events.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text(
        json.dumps(
            {
                "event": "heartbeat",
                "worker_state": "stopped",
                "restart_not_before": "2020-01-01T00:00:00+00:00",
                "restart_attempts": 0,
            }
        )
        + "\n"
    )
    fixture["service_state"].write_text(
        json.dumps(
            {
                "session": False,
                "pane": "1\t4242\tbash",
                "ps": (
                    "S 7777 /data/home/other/env/bin/bashlet train-opsd "
                    "configs/other-run.yaml --output-dir runs/issue99 --resume\n"
                ),
            }
        )
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert _events(fixture)[-1]["decision"] == "restart_started"
    assert any(call[0] == "new-session" for call in _tmux_calls(fixture))


def test_formal_endpoint_stops_monitor_only_after_four_epochs(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    run_root: Path = fixture["run_root"]
    proof_path = run_root / "preflight" / "checkpoint-validation.json"
    workflow_path = run_root / "workflow.json"
    proof_hash = _digest(proof_path)
    workflow = {
        "status": "complete",
        "stage": "formal",
        "configuration_sha256": fixture["configuration_hashes"],
        "preflight_sha256": proof_hash,
    }
    _write_json(workflow_path, workflow)
    formal_pilot = {
        "status": "complete",
        "training_complete": True,
        "epochs_completed": 4,
        "configuration": {"pilot": {"epochs": 4}},
        "training_tasks": [],
    }
    _write_json(run_root / "formal" / "pilot.json", formal_pilot)
    _write_json(
        run_root / "formal" / "training" / "opsd.json",
        {"pending_update": None, "batches": []},
    )

    result = _cli(fixture)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"decision": "complete"}
    assert _events(fixture)[-1]["reason"] == (
        "configured formal run and development are complete"
    )
    assert all(call[0] == "has-session" for call in _tmux_calls(fixture))
