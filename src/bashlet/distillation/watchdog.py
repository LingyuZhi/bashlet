"""Conservatively resume the frozen expanded OPSD workflow."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from .recovery import reconciled_rejection


class WatchdogError(ValueError):
    """The watchdog cannot safely inspect or control the configured run."""


@dataclass(frozen=True)
class WatchConfig:
    root: Path
    monitor_root: Path
    workflow: Path
    preflight_pilot: Path
    preflight_training: Path
    preflight_validation: Path
    formal_pilot: Path
    formal_training: Path
    ledger: Path
    ledger_groups: tuple[str, ...]
    config_files: tuple[Path, ...]
    worker_session: str
    monitor_session: str
    worker_log: Path
    worker_command: tuple[str, ...]
    poll_seconds: int
    backoff_min_seconds: int
    backoff_max_seconds: int

    @classmethod
    def read(cls, path: Path) -> WatchConfig:
        try:
            raw = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError):
            raise WatchdogError("monitor configuration is unreadable") from None
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise WatchdogError("unsupported monitor configuration")

        config_dir = path.resolve().parent
        root_text = raw.get("repository_root")
        if not isinstance(root_text, str) or not root_text:
            raise WatchdogError("repository_root must be a path")
        root = _resolve(config_dir, root_text)

        paths = raw.get("paths")
        worker = raw.get("worker")
        timing = raw.get("timing")
        if not isinstance(paths, dict) or not isinstance(worker, dict):
            raise WatchdogError("monitor paths and worker settings are required")
        if not isinstance(timing, dict):
            raise WatchdogError("monitor timing settings are required")

        def path_value(key: str) -> Path:
            value = paths.get(key)
            if not isinstance(value, str) or not value:
                raise WatchdogError(f"paths.{key} must be a path")
            return _resolve(root, value)

        session = worker.get("session")
        command = worker.get("command")
        if not isinstance(session, str) or not session:
            raise WatchdogError("worker.session is required")
        monitor_session = raw.get("monitor_session")
        if not isinstance(monitor_session, str) or not monitor_session:
            raise WatchdogError("monitor_session is required")
        if command != ["bash", "scripts/train-expanded-opsd.sh", "--resume"]:
            raise WatchdogError("worker command must use the fixed --resume entrypoint")

        config_files_raw = raw.get("frozen_configurations")
        if not isinstance(config_files_raw, list) or not config_files_raw:
            raise WatchdogError("frozen_configurations must be a non-empty list")
        config_files = tuple(_resolve(root, item) for item in config_files_raw)
        groups = raw.get("ledger_groups")
        expected_groups = (
            "opsd-expanded-preflight-v1",
            "opsd-expanded-train-v1",
            "opsd-expanded-dev-v1",
        )
        if groups != list(expected_groups):
            raise WatchdogError("ledger_groups must match the frozen OPSD run")
        poll_seconds = _required_int(timing, "poll_seconds")
        backoff_min = _required_int(timing, "backoff_min_seconds")
        backoff_max = _required_int(timing, "backoff_max_seconds")
        if not 1 <= poll_seconds <= 60:
            raise WatchdogError("poll_seconds must be between 1 and 60")
        if not 60 <= backoff_min <= backoff_max <= 600:
            raise WatchdogError("restart backoff must stay between 60 and 600 seconds")

        return cls(
            root=root,
            monitor_root=path_value("monitor_root"),
            workflow=path_value("workflow"),
            preflight_pilot=path_value("preflight_pilot"),
            preflight_training=path_value("preflight_training"),
            preflight_validation=path_value("preflight_validation"),
            formal_pilot=path_value("formal_pilot"),
            formal_training=path_value("formal_training"),
            ledger=path_value("ledger"),
            ledger_groups=expected_groups,
            config_files=config_files,
            worker_session=session,
            monitor_session=monitor_session,
            worker_log=_resolve(root, _required_string(worker, "log")),
            worker_command=tuple(command),
            poll_seconds=poll_seconds,
            backoff_min_seconds=backoff_min,
            backoff_max_seconds=backoff_max,
        )


@dataclass(frozen=True)
class WorkerState:
    state: str
    reason: str
    pane_dead: bool | None = None


@dataclass(frozen=True)
class EvidenceState:
    state: str
    reason: str
    workflow_status: str | None
    workflow_stage: str | None
    pending_requests: int
    ledger_requests: dict[str, int]


def _resolve(base: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise WatchdogError("configured paths must be non-empty strings")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _required_string(raw: dict[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise WatchdogError(f"worker.{key} is required")
    return value


def _required_int(raw: dict[str, Any], key: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise WatchdogError(f"{key} must be an integer")
    return value


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        raise WatchdogError("structured run evidence is unreadable") from None
    if not isinstance(value, dict):
        raise WatchdogError("structured run evidence must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        raise WatchdogError("a frozen input is missing or unreadable") from None


def _running_worker_process(config: WatchConfig) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-u", str(os.getuid()), "-o", "stat=,pid=,args="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise WatchdogError("process state is unavailable") from None
    if result.returncode != 0:
        raise WatchdogError("process state is unavailable")
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) != 3 or fields[0].startswith("Z"):
            continue
        try:
            pid = int(fields[1])
        except ValueError:
            continue
        if _is_this_worker(fields[2], pid, config):
            return True
    return False


def _looks_like_worker(args: str) -> bool:
    return "scripts/train-expanded-opsd.sh" in args or (
        "bashlet" in args and "train-opsd" in args and "watch-opsd" not in args
    )


def _is_this_worker(args: str, pid: int, config: WatchConfig) -> bool:
    if not _looks_like_worker(args):
        return False
    run_root = config.formal_pilot.parent.parent
    run_relative = run_root.relative_to(config.root).as_posix()
    if str(run_root) in args or run_relative in args or str(config.worker_log) in args:
        return True
    if "scripts/train-expanded-opsd.sh" not in args:
        return False
    try:
        process_cwd = Path(f"/proc/{pid}/cwd").resolve(strict=True)
    except OSError:
        raise WatchdogError("matching worker process cwd is unavailable") from None
    return process_cwd == config.root


def _tmux(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["tmux", *args],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise WatchdogError("tmux state is unavailable") from None


def _worker_state(config: WatchConfig) -> WorkerState:
    if _running_worker_process(config):
        return WorkerState("active", "matching OPSD process is alive")

    session = _tmux(["has-session", "-t", config.worker_session])
    if session.returncode == 0:
        panes = _tmux(
            [
                "list-panes",
                "-t",
                config.worker_session,
                "-F",
                "#{pane_dead}\t#{pane_pid}\t#{pane_current_command}",
            ]
        )
        if panes.returncode != 0:
            return WorkerState("unknown", "worker pane state is unavailable")
        lines = [line for line in panes.stdout.splitlines() if line]
        if len(lines) != 1:
            return WorkerState(
                "unknown", "worker session does not have exactly one pane"
            )
        fields = lines[0].split("\t", maxsplit=2)
        if len(fields) != 3 or fields[0] not in {"0", "1"}:
            return WorkerState("unknown", "worker pane state is malformed")
        dead = fields[0] == "1"
        if dead:
            return WorkerState("stopped", "worker pane is confirmed dead", True)
        return WorkerState(
            "unknown", "worker pane is live without a confirmed worker", False
        )
    if session.returncode == 1:
        message = session.stderr.lower()
        if (
            not message
            or "can't find session" in message
            or "no server running" in message
        ):
            return WorkerState(
                "stopped", "worker session and matching process are absent"
            )
        return WorkerState("unknown", "tmux could not confirm the worker session state")
    return WorkerState("unknown", "worker session state is unavailable")


def _ledger_state(config: WatchConfig) -> tuple[dict[str, int], dict[str, int]]:
    groups = config.ledger_groups
    if not config.ledger.is_file():
        raise WatchdogError("the shared spend ledger is unavailable")
    try:
        connection = sqlite3.connect(
            config.ledger.as_uri() + "?mode=ro", uri=True, timeout=2
        )
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            "SELECT budget_group, state, COUNT(*) FROM requests "
            "WHERE budget_group IN (?, ?, ?) GROUP BY budget_group, state",
            groups,
        ).fetchall()
    except sqlite3.Error:
        raise WatchdogError("the shared spend ledger cannot be checked") from None
    finally:
        if "connection" in locals():
            connection.close()
    counts = {group: 0 for group in groups}
    pending = {group: 0 for group in groups}
    for group, state, count in rows:
        if group in counts:
            counts[group] += int(count)
            if state not in {"completed", "failed"}:
                pending[group] += int(count)
    return counts, pending


def _training_gate(
    pilot: dict[str, Any], training: dict[str, Any] | None, root: Path,
) -> str | None:
    if training is None:
        return "training state record is missing"
    if "pending_update" not in training:
        return "training record does not declare pending-update state"
    if training.get("pending_update") is not None:
        return "training record has an unresolved remote update"
    if training.get("status") == "failed":
        return "training record has an explicit failure"
    if pilot.get("training_stop"):
        return "training stopped at a budget or policy gate"
    batches = training.get("batches", [])
    if not isinstance(batches, list):
        return "training batch record is malformed"
    updates: list[dict[str, Any]] = []
    for position, batch in enumerate(batches):
        if not isinstance(batch, dict):
            return "training history contains an unresolved batch"
        if type(batch.get("index")) is not int or batch.get("index") != position:
            return "training history contains an unresolved batch"
        update = batch.get("update")
        if not isinstance(update, dict):
            if reconciled_rejection(batch, pilot, root, len(updates)):
                continue
            return "training history contains an unreconciled rejected batch"
        if type(update.get("optimizer_step")) is not int or update.get(
            "optimizer_step"
        ) != len(updates) + 1:
            return "training history contains an unresolved batch"
        if not isinstance(update.get("student_policy"), str) or not update.get(
            "student_policy"
        ):
            return "training history contains an unresolved batch"
        updates.append(update)
        if batch.get("status") == "completed":
            continue
        if (
            batch.get("status") != "interrupted"
            or batch.get("reason") != "monitor_interrupted"
        ):
            return "training history contains an unresolved batch"
    if training.get("status") == "rejected" and (
        not batches or isinstance(batches[-1].get("update"), dict)
    ):
        return "training record has an explicit failure"
    if batches and not updates and (
        training.get("optimizer_step") != 0 or training.get("last_update") is not None
    ):
        return "training state does not match its final durable optimizer receipt"
    if updates:
        last_update = training.get("last_update")
        optimizer_step = training.get("optimizer_step")
        final_update = updates[-1]
        if (
            not isinstance(last_update, dict)
            or type(optimizer_step) is not int
            or optimizer_step != len(updates)
            or final_update != last_update
            or final_update.get("optimizer_step") != optimizer_step
            or training.get("student_policy") != final_update.get("student_policy")
        ):
            return "training state does not match its final durable optimizer receipt"
    rows = pilot.get("training_tasks", [])
    if not isinstance(rows, list):
        return "training visit record is malformed"
    completed_updates = {
        batch.get("index")
        for batch in batches
        if isinstance(batch, dict)
        and batch.get("status") in {"completed", "interrupted"}
        and isinstance(batch.get("update"), dict)
    }
    for row in rows:
        if not isinstance(row, dict):
            return "training visit record is malformed"
        status = row.get("status")
        if status in {"failed", "rejected", "interrupted"}:
            return "training visit has a failed or rejected supervision gate"
        if status not in {"completed", "not_started", "running"}:
            return "training visit has an unknown status"
        if status == "running" and row.get("batch_index") not in completed_updates:
            return "training visit has no durable update receipt"
    return None


def _recoverable_record_failure(record: dict[str, Any]) -> bool:
    if record.get("status") == "interrupted":
        return record.get("error_type") == "KeyboardInterrupt"
    reason = record.get("reason")
    return (
        record.get("status") == "failed"
        and record.get("error_type") == "ValueError"
        and isinstance(reason, str)
        and reason.startswith(
            (
                "W&B startup failed;",
                "W&B network unavailable;",
                "W&B online verification required",
            )
        )
    )


def _has_blocking_failure(record: dict[str, Any]) -> bool:
    status = record.get("status")
    error_type = record.get("error_type")
    if status in {"failed", "rejected", "interrupted"} or error_type:
        return not _recoverable_record_failure(record)
    return False


def _proof_is_complete(
    config: WatchConfig, workflow: dict[str, Any]
) -> tuple[bool, str]:
    proof = _read_object(config.preflight_validation)
    if proof is None:
        return False, "checkpoint validation evidence is missing"
    if proof.get("status") == "failed":
        return False, "checkpoint validation failed and needs review"
    if proof.get("status") != "complete" or not isinstance(proof.get("native"), dict):
        return False, "checkpoint validation is incomplete"
    if proof["native"].get("status") != "complete":
        return False, "native checkpoint validation is incomplete"
    proof_hash = _sha256(config.preflight_validation)
    expected_hash = workflow.get("preflight_sha256")
    if workflow.get("stage") == "formal" and expected_hash != proof_hash:
        return False, "checkpoint validation evidence changed after approval"
    return True, "checkpoint validation is complete"


def _formal_complete(pilot: dict[str, Any], training: dict[str, Any] | None) -> bool:
    return (
        training is not None
        and pilot.get("status") == "complete"
        and pilot.get("training_complete") is True
        and pilot.get("epochs_completed") == 4
        and training.get("pending_update") is None
    )


def _inspect_evidence(config: WatchConfig) -> EvidenceState:
    workflow = _read_object(config.workflow)
    if workflow is None:
        return EvidenceState("blocked", "workflow record is missing", None, None, 0, {})
    workflow_status = workflow.get("status")
    workflow_stage = workflow.get("stage")
    if workflow_status == "preflight" and workflow_stage is None:
        workflow_stage = "preflight-training"
    if not isinstance(workflow_status, str) or not isinstance(workflow_stage, str):
        return EvidenceState(
            "blocked", "workflow status is malformed", None, None, 0, {}
        )

    counts: dict[str, int] = {}
    pending_count = 0

    def result(kind: str, reason: str) -> EvidenceState:
        return EvidenceState(
            kind, reason, workflow_status, workflow_stage, pending_count, counts
        )

    if workflow_status not in {
        "preflight",
        "incomplete",
        "interrupted",
        "formal",
        "complete",
        "failed",
    }:
        return result("blocked", "workflow status is not recognized")
    if workflow_status in {"formal", "complete"} and workflow_stage != "formal":
        return result("blocked", "workflow status and stage do not agree")

    declared_hashes = workflow.get("configuration_sha256")
    actual_hashes = {str(path): _sha256(path) for path in config.config_files}
    if declared_hashes != actual_hashes:
        return result("blocked", "frozen configuration changed")

    counts, pending = _ledger_state(config)
    pending_count = sum(pending.values())
    if pending_count:
        return result("blocked", "a related spend-ledger request is pending")

    if workflow_stage not in {"preflight-training", "preflight-verification", "formal"}:
        return result("blocked", "workflow stage is not a known resumable stage")

    preflight = _read_object(config.preflight_pilot)
    preflight_training = _read_object(config.preflight_training)
    if preflight is None:
        if workflow_stage != "preflight-training" or counts.get(
            "opsd-expanded-preflight-v1", 0
        ):
            return result(
                "blocked", "preflight record is missing after related work started"
            )
    else:
        if _has_blocking_failure(preflight):
            return result("blocked", "preflight has an explicit failure")
        if preflight_training is not None:
            problem = _training_gate(
                preflight, preflight_training, config.preflight_pilot.parent,
            )
            if problem is not None:
                return result("blocked", problem)
        elif preflight.get("status") == "complete" or counts.get(
            "opsd-expanded-preflight-v1", 0
        ):
            return result("blocked", "preflight has work but no training-state record")

    if preflight is not None and preflight.get("status") == "complete":
        proof_complete, proof_reason = _proof_is_complete(config, workflow)
        if not proof_complete:
            return result("blocked", proof_reason)
    elif workflow_stage in {"preflight-verification", "formal"}:
        return result("blocked", "workflow advanced without a complete preflight")

    if workflow_stage in {"preflight-training", "preflight-verification"}:
        if workflow_status == "failed" and (
            preflight is None or not _recoverable_record_failure(preflight)
        ):
            return result(
                "blocked", "workflow failure requires explicit reconciliation"
            )
        if workflow_status == "complete":
            return result("blocked", "workflow ended before formal training")
        return result(
            "resume", "preflight evidence permits the existing --resume workflow"
        )

    formal = _read_object(config.formal_pilot)
    formal_training = _read_object(config.formal_training)
    if formal is None:
        if workflow_status == "failed":
            return result(
                "blocked", "workflow failure requires explicit reconciliation"
            )
        if counts.get("opsd-expanded-train-v1", 0) or counts.get(
            "opsd-expanded-dev-v1", 0
        ):
            return result(
                "blocked", "formal pilot record is missing after related work started"
            )
        return result(
            "resume", "preflight gates pass; resume can start formal training"
        )
    if _has_blocking_failure(formal):
        return result("blocked", "formal run has an explicit failure")
    development = formal.get("development_execution")
    if development is not None and _sha256(Path(development["config"])) != development[
        "sha256"
    ]:
        return result("blocked", "background development configuration changed")
    if workflow_status == "failed" and not _recoverable_record_failure(formal):
        return result("blocked", "workflow failure requires explicit reconciliation")
    execution = formal.get("execution_limit")
    if execution is not None:
        if _sha256(Path(execution["config"])) != execution["sha256"]:
            return result("blocked", "frozen execution limit changed")
    if formal_training is not None:
        problem = _training_gate(formal, formal_training, config.formal_pilot.parent)
        if problem is not None:
            return result("blocked", problem)
    elif formal.get("status") == "complete":
        return result("blocked", "completed formal run has no training-state record")
    elif counts.get("opsd-expanded-train-v1", 0) or counts.get(
        "opsd-expanded-dev-v1", 0
    ):
        return result(
            "blocked", "formal run has ledger activity but no training-state record"
        )
    if formal.get("status") == "complete":
        if not _formal_complete(formal, formal_training):
            return result(
                "blocked", "formal run says complete without the four-epoch endpoint"
            )
        if workflow_status == "complete":
            return result(
                "complete", "four-epoch formal run and development are complete"
            )
        return result(
            "resume", "formal endpoint is complete; resume can finalize workflow state"
        )
    if formal.get("execution_status") == "complete":
        if (
            not isinstance(execution, dict)
            or formal_training is None
            or formal.get("optimizer_step") != execution["max_updates"]
            or formal_training.get("optimizer_step") != execution["max_updates"]
            or formal.get("training_complete") is not False
            or formal.get("main_result") is not None
            or formal.get("evaluations", {}).get(
                f"epoch-{formal.get('epochs_completed')}", {}
            ).get("status") != "complete"
        ):
            return result(
                "blocked", "execution limit endpoint evidence is inconsistent"
            )
        return result(
            "complete", "execution limit and dev complete; full plan incomplete"
        )
    if workflow_status == "complete":
        return result(
            "blocked", "workflow is complete but the formal endpoint is incomplete"
        )
    return result("resume", "formal run is incomplete and saved state permits --resume")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds")


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with path.open() as source:
            for line in source:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise WatchdogError("monitor history is malformed")
                records.append(value)
    except (OSError, json.JSONDecodeError):
        raise WatchdogError("monitor history is unreadable") from None
    return records


def _append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = (json.dumps(event, sort_keys=True, ensure_ascii=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restart_count(history: list[dict[str, Any]]) -> int:
    return sum(event.get("event") == "restart_attempt" for event in history)


def _current_schedule(
    history: list[dict[str, Any]], worker_state: str
) -> datetime | None:
    if worker_state != "stopped" or not history:
        return None
    latest = history[-1]
    value = latest.get("restart_not_before")
    if latest.get("worker_state") != "stopped" or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise WatchdogError("monitor backoff history is malformed") from None


def _backoff(config: WatchConfig, attempts: int) -> int:
    delay: int = config.backoff_min_seconds * (2 ** min(attempts, 10))
    return min(delay, config.backoff_max_seconds)


def _worker_command(config: WatchConfig) -> str:
    command = " ".join(shlex.quote(part) for part in config.worker_command)
    log = shlex.quote(str(config.worker_log))
    return f"bash -lc {shlex.quote(f'exec {command} >> {log} 2>&1')}"


def _launch_worker(config: WatchConfig, state: WorkerState) -> int:
    command = _worker_command(config)
    if state.pane_dead is True:
        args = [
            "tmux",
            "respawn-pane",
            "-t",
            f"{config.worker_session}:0.0",
            "-c",
            str(config.root),
            command,
        ]
    else:
        args = [
            "tmux",
            "new-session",
            "-d",
            "-s",
            config.worker_session,
            "-c",
            str(config.root),
            command,
        ]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127
    return result.returncode


def _log_cycle(
    config: WatchConfig,
    history: list[dict[str, Any]],
    worker: WorkerState,
    evidence: EvidenceState,
    decision: str,
    reason: str,
    attempts: int,
    restart_not_before: datetime | None,
    *,
    event_type: str = "heartbeat",
    restart_exit_code: int | None = None,
) -> None:
    try:
        log_age = max(0.0, time.time() - config.worker_log.stat().st_mtime)
    except OSError:
        log_age = None
    event: dict[str, Any] = {
        "timestamp": _timestamp(_utc_now()),
        "monitor_session": config.monitor_session,
        "event": event_type,
        "decision": decision,
        "reason": reason,
        "workflow_status": evidence.workflow_status,
        "workflow_stage": evidence.workflow_stage,
        "worker_state": worker.state,
        "worker_reason": worker.reason,
        "log_age_seconds": round(log_age, 1) if log_age is not None else None,
        "pending_request_count": evidence.pending_requests,
        "related_request_counts": evidence.ledger_requests,
        "restart_attempts": attempts,
        "restart_not_before": (
            _timestamp(restart_not_before) if restart_not_before is not None else None
        ),
    }
    if restart_exit_code is not None:
        event["restart_exit_code"] = restart_exit_code
    _append_event(config.monitor_root / "events.jsonl", event)
    history.append(event)


def _cycle(config: WatchConfig, history: list[dict[str, Any]]) -> str:
    try:
        worker = _worker_state(config)
    except WatchdogError as error:
        worker = WorkerState("unknown", str(error))
    try:
        evidence = _inspect_evidence(config)
    except WatchdogError as error:
        evidence = EvidenceState("blocked", str(error), None, None, 0, {})
    attempts = _restart_count(history)
    if worker.state == "active":
        _log_cycle(
            config,
            history,
            worker,
            evidence,
            "worker_active",
            "worker is alive; leave all in-flight work untouched",
            attempts,
            None,
        )
        return "worker_active"
    if worker.state == "unknown":
        _log_cycle(
            config,
            history,
            worker,
            evidence,
            "worker_state_uncertain",
            "worker state is uncertain; automatic recovery is withheld",
            attempts,
            None,
        )
        return "worker_state_uncertain"
    if evidence.state == "blocked":
        _log_cycle(
            config,
            history,
            worker,
            evidence,
            "blocked",
            evidence.reason,
            attempts,
            None,
        )
        return "blocked"
    if evidence.state == "complete":
        _log_cycle(
            config,
            history,
            worker,
            evidence,
            "complete",
            evidence.reason,
            attempts,
            None,
        )
        return "complete"

    schedule = _current_schedule(history, worker.state)
    now = _utc_now()
    if schedule is None:
        schedule = now + timedelta(seconds=_backoff(config, attempts))
    if now < schedule:
        _log_cycle(
            config,
            history,
            worker,
            evidence,
            "waiting_backoff",
            "recoverable exit confirmed; waiting before --resume",
            attempts,
            schedule,
        )
        return "waiting_backoff"

    # Recheck process and pane state immediately before any restart.
    try:
        current = _worker_state(config)
    except WatchdogError as error:
        current = WorkerState("unknown", str(error))
    if current.state != "stopped":
        _log_cycle(
            config,
            history,
            current,
            evidence,
            "worker_state_changed",
            "worker state changed during the restart check",
            attempts,
            None,
        )
        return "worker_state_changed"
    try:
        current_evidence = _inspect_evidence(config)
    except WatchdogError as error:
        current_evidence = EvidenceState("blocked", str(error), None, None, 0, {})
    if current_evidence.state != "resume":
        _log_cycle(
            config,
            history,
            current,
            current_evidence,
            "recovery_gate_changed",
            current_evidence.reason,
            attempts,
            None,
        )
        return "recovery_gate_changed"
    exit_code = _launch_worker(config, current)
    attempts += 1
    next_schedule = _utc_now() + timedelta(seconds=_backoff(config, attempts))
    decision = "restart_started" if exit_code == 0 else "restart_failed"
    reason = (
        "launched the existing --resume entrypoint"
        if exit_code == 0
        else "tmux did not accept the restart command; retry after bounded backoff"
    )
    _log_cycle(
        config,
        history,
        current,
        evidence,
        decision,
        reason,
        attempts,
        next_schedule,
        event_type="restart_attempt",
        restart_exit_code=exit_code,
    )
    return decision


def run_watch(config_path: Path, *, once: bool = False) -> str:
    """Watch until the formal endpoint is complete or the process is stopped."""
    config = WatchConfig.read(config_path)
    config.monitor_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = config.monitor_root / "watchdog.lock"
    events_path = config.monitor_root / "events.jsonl"
    try:
        lock = lock_path.open("a")
    except OSError:
        raise WatchdogError("monitor lock is unavailable") from None
    with lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WatchdogError("another monitor already owns this run") from None
        history = _load_history(events_path)
        stop_signals = (signal.SIGTERM, signal.SIGHUP)
        previous_signals = {item: signal.getsignal(item) for item in stop_signals}

        def stop_monitor(signum: int, frame: Any) -> None:
            raise KeyboardInterrupt

        for item in stop_signals:
            signal.signal(item, stop_monitor)
        try:
            while True:
                decision = _cycle(config, history)
                print(json.dumps({"decision": decision}, ensure_ascii=True), flush=True)
                if once or decision == "complete":
                    return decision
                time.sleep(config.poll_seconds)
        except KeyboardInterrupt:
            _append_event(
                events_path,
                {
                    "timestamp": _timestamp(_utc_now()),
                    "event": "monitor_stopped",
                    "decision": "stopped_by_user",
                    "reason": "monitor process received a stop signal",
                    "restart_attempts": _restart_count(history),
                },
            )
            return "stopped_by_user"
        finally:
            for item, handler in previous_signals.items():
                signal.signal(item, handler)
