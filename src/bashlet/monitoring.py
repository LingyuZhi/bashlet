"""Allowlisted experiment telemetry with a durable, resumable local journal."""

import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import dotenv_values

from bashlet.config import read_yaml


def write_document(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def code_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return result.stdout.strip() or None


@dataclass(frozen=True)
class MonitorConfig:
    project: str = "bashlet"
    entity: str | None = None
    group: str = "bashlet"
    mode: Literal["online", "offline"] = "online"
    log_every: int = 1
    timeout_seconds: int = 20
    dotenv_path: str = ".env"

    @classmethod
    def read(cls, path: Path) -> "MonitorConfig":
        return cls(**read_yaml(path))

    def __post_init__(self) -> None:
        if self.mode not in {"online", "offline"}:
            raise ValueError("Monitoring mode must be online or offline")
        if not self.project or not self.group:
            raise ValueError("Monitoring requires project and group")
        if type(self.log_every) is not int or self.log_every < 1:
            raise ValueError("log_every must be a positive integer")
        if type(self.timeout_seconds) is not int or self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be a positive integer")

    def credential(self) -> str:
        values = dotenv_values(self.dotenv_path, interpolate=False)
        key = (
            os.environ.get("WANDB_API_KEY")
            or values.get("WANDB_API_KEY")
            or values.get("wandb_api_key")
            or ""
        ).strip()
        if not key:
            raise ValueError("Missing WANDB_API_KEY (environment or .env)")
        return key


_CONFIG_FIELDS = {
    "local_run",
    "phase",
    "code_revision",
    "evidence_kind",
    "historical_import",
    "source_run",
    "source_sha256",
    "comparison_sha256",
    "snapshot",
    "training_run",
    "base_model",
    "server_model_revision",
    "encoding",
    "data_sha256",
    "source_manifest_sha256",
    "data_source",
    "training",
    "fingerprints",
    "sampling",
    "loss_aggregation",
    "optimizer_transition",
}


def _network_failure(error: Exception) -> bool:
    """Only known transport/transient failures may bypass online startup."""
    pending: list[Any] = [error]
    visited: set[int] = set()
    transient = False
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        response = getattr(current, "response", None)
        status = getattr(response, "status_code", None) or getattr(
            response, "code", None
        )
        status = status or getattr(current, "status_code", None)
        if status is not None:
            try:
                code = int(status)
            except (TypeError, ValueError):
                return False
            if code not in (408, 429) and not 500 <= code <= 599:
                return False
            transient = True
        if isinstance(
            current,
            (TimeoutError, ConnectionError, httpx.NetworkError, httpx.TimeoutException),
        ):
            transient = True
        # W&B wraps requests transport errors inside CommError.exc.
        if type(current).__module__ == "requests.exceptions" and type(
            current
        ).__name__ in {"ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout"}:
            transient = True
        pending.extend(
            child
            for attr in ("exc", "__cause__", "__context__")
            if (child := getattr(current, attr, None)) is not None
        )
    return transient


class RunMonitor:
    """W&B is a projection; the local journal remains the source of progress.

    Explicit log steps and event IDs let a resumed online run replay only events
    the server has not seen. Offline segments retain the same W&B identity.
    """

    @classmethod
    def resume(cls, run_dir: Path, *, require_online: bool = False) -> "RunMonitor":
        state = json.loads((run_dir / "monitor.json").read_text())
        config = MonitorConfig(**state["config"]["monitoring"])
        return cls(
            run_dir,
            replace(config, mode="online") if require_online else config,
            phase=state["phase"],
            metadata={k: v for k, v in state["config"].items() if k in _CONFIG_FIELDS},
            require_online=require_online,
        )

    def __init__(
        self,
        run_dir: Path,
        config: MonitorConfig,
        *,
        phase: str,
        metadata: dict[str, Any],
        require_online: bool = False,
    ) -> None:
        self.root = run_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config = config
        self.path = self.root / "monitor.json"
        self.journal = self.root / "metrics.jsonl"
        self.events = (
            [json.loads(line) for line in self.journal.read_text().splitlines()]
            if self.journal.exists()
            else []
        )
        self.ids = {event["event_id"] for event in self.events}
        if set(metadata) - _CONFIG_FIELDS:
            raise ValueError("Monitoring metadata contains non-allowlisted fields")
        safe = metadata | {
            "local_run": str(self.root),
            "phase": phase,
            "monitoring": {
                k: v for k, v in asdict(config).items() if k != "dotenv_path"
            },
        }
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            previous = json.loads(json.dumps(self.state["config"]))
            previous["monitoring"]["mode"] = safe["monitoring"]["mode"]
            if previous != safe:
                raise ValueError(
                    "Monitoring resume must preserve the run configuration"
                )
        else:
            self.state = {
                "id": uuid.uuid4().hex,
                "project": config.project,
                "entity": config.entity,
                "group": config.group,
                "phase": phase,
                "config": safe,
                "url": None,
                "online_verified": False,
                "caches": [],
                "sync_status": "pending",
                "cached_events": 0,
            }
        self.run: Any = None
        self.delivery_failed = False
        self.state["sync_status"] = "pending"
        write_document(self.path, self.state)
        try:
            self._start()
        except Exception as error:
            # init may have succeeded before history lookup failed. Until replay
            # is complete, sending newer events could skip the missing prefix.
            self.delivery_failed = True
            self.state["error_type"] = type(error).__name__
            self.state["sync_status"] = "pending"
            write_document(self.path, self.state)
            self.finish()
            # Authentication/configuration errors must never become offline success.
            if not _network_failure(error):
                raise ValueError(
                    "W&B startup failed; authentication/configuration must be checked: "
                    f"{type(error).__name__}"
                ) from None
            if require_online and not self.state["online_verified"]:
                raise ValueError(
                    "W&B network unavailable; online verification required "
                    "before paid training"
                ) from None
        if require_online and not self.state["online_verified"]:
            self.finish()
            raise ValueError("W&B online verification required before paid training")

    def _start(self) -> None:
        import wandb

        key = self.config.credential() if self.config.mode == "online" else None
        if key is not None:
            # The SDK's core subprocess also resolves credentials from the
            # standard environment field. The agent sandbox clears its env.
            os.environ["WANDB_API_KEY"] = key
        for name, folder in (
            ("WANDB_CACHE_DIR", "cache"),
            ("WANDB_CONFIG_DIR", "settings"),
            ("WANDB_DATA_DIR", "staging"),
        ):
            os.environ[name] = str(self.root / folder)
        settings = wandb.Settings(
            api_key=key,
            console="off",
            disable_git=True,
            save_code=False,
            disable_code=True,
            disable_job_creation=True,
            x_save_requirements=False,
            x_disable_stats=True,
            x_disable_meta=True,
            x_disable_machine_info=True,
            init_timeout=self.config.timeout_seconds,
            finish_timeout=self.config.timeout_seconds,
            finish_timeout_raises=True,
        )
        entity = self.state["entity"]
        if self.config.mode == "online" and entity is None:
            entity = wandb.Api(
                api_key=key, timeout=self.config.timeout_seconds
            ).default_entity
        self.run = wandb.init(
            project=self.config.project,
            entity=entity,
            group=self.config.group,
            job_type=self.state["phase"],
            id=self.state["id"],
            resume="allow" if self.config.mode == "online" else None,
            mode=self.config.mode,
            dir=str(self.root),
            config=self.state["config"],
            settings=settings,
        )
        self.state.update(
            entity=self.run.entity,
            url=self.run.url or self.state["url"],
            online_verified=self.state["online_verified"]
            or self.config.mode == "online",
        )
        cache = self.run.settings.sync_dir
        if cache not in self.state["caches"]:
            self.state["caches"].append(cache)
        self.run.define_metric("train/*", step_metric="train/optimizer_step")
        self.run.define_metric("eval/*", step_metric="eval/completed_tasks")
        start = (
            self.run.step
            if self.config.mode == "online"
            else self.state["cached_events"]
        )
        if self.config.mode == "online" and self.events:
            remote = wandb.Api(api_key=key, timeout=self.config.timeout_seconds).run(
                f"{self.state['entity']}/{self.state['project']}/{self.state['id']}"
            )
            start = max(start, remote.lastHistoryStep + 1)
            if start > len(self.events):
                self.finish()
                raise ValueError("Cloud history is ahead of the local metric journal")
        self.state["cached_events"] = max(self.state["cached_events"], start)
        for event in self.events[start:]:
            self._send(event)
        write_document(self.path, self.state)

    def _send(self, event: dict[str, Any]) -> None:
        if self.run is None or self.delivery_failed:
            return
        try:
            self.run.log(event["metrics"], step=event["sequence"])
            self.state["cached_events"] = event["sequence"] + 1
        except Exception as error:
            self.delivery_failed = True
            self.state["error_type"] = type(error).__name__
            self.state["sync_status"] = "pending"

    def log(self, event_id: str, metrics: dict[str, Any]) -> None:
        if event_id in self.ids:
            return
        for key, value in metrics.items():
            if key.split("/", 1)[0] not in {
                "train",
                "eval",
                "requests",
                "run",
                "monitor",
                "checkpoint",
            }:
                raise ValueError("Unknown metric namespace")
            if not isinstance(value, (int, float, bool, str)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError("Metrics must be finite scalars")
        event = {"event_id": event_id, "sequence": len(self.events), "metrics": metrics}
        with self.journal.open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        self.ids.add(event_id)
        self.state["sync_status"] = "pending"
        self._send(event)
        write_document(self.path, self.state)

    def update_metadata(self, values: dict[str, Any]) -> None:
        if set(values) - _CONFIG_FIELDS:
            raise ValueError("Monitoring metadata contains non-allowlisted fields")
        self.state["config"].update(values)
        if self.run is not None:
            try:
                self.run.config.update(values, allow_val_change=True)
            except Exception as error:
                self.state["error_type"] = type(error).__name__
                self.delivery_failed = True
                self.state["sync_status"] = "pending"
        write_document(self.path, self.state)

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
                if (
                    self.config.mode == "online"
                    and not self.delivery_failed
                    and self.state["cached_events"] == len(self.events)
                ):
                    self.state["sync_status"] = "synced"
            except Exception as error:
                self.state["error_type"] = type(error).__name__
            self.run = None
        self.state["sync_command"] = (
            "uv run --env-file .env bashlet monitor sync " + shlex.quote(str(self.root))
        )
        self.state["cache_sync_commands"] = [
            "uv run --env-file .env " + shlex.join(_sync_arguments(self.state, [path]))
            for path in self.state["caches"]
        ]
        write_document(self.path, self.state)


def evaluation_metrics(record: dict[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    metrics = {
        f"eval/{key}": summary[key]
        for key in (
            "planned_tasks",
            "passed_tasks",
            "final_pass_rate",
            "first_implementation_passed",
            "first_implementation_pass_rate",
        )
    }
    metrics.update(
        {
            "eval/completed_tasks": summary["planned_tasks"]
            - len(summary["missing_task_ids"]),
            "eval/missing_tasks": len(summary["missing_task_ids"]),
            "eval/first_grading_missing": len(
                summary["first_grading_missing_task_ids"]
            ),
            "eval/recovery_denominator": summary["recovery"]["denominator"],
            "eval/recovered_tasks": len(summary["recovery"]["recovered_task_ids"]),
            "run/status": record["status"],
        }
    )
    usages = [task.get("usage", {}) for task in record.get("tasks", [])]
    for key in ("generated_tokens", "steps", "elapsed_seconds"):
        if any(key in usage for usage in usages):
            metrics[f"eval/{key}"] = sum(usage.get(key, 0) for usage in usages)
    if record.get("finished_at") and record.get("started_at"):
        metrics["eval/wall_seconds"] = (
            datetime.fromisoformat(record["finished_at"])
            - datetime.fromisoformat(record["started_at"])
        ).total_seconds()
    entries = record.get("snapshot_details", {}).get("requests", {}).get("entries", [])
    request_ids = set(record["requests"].get("request_ids", []))
    usage_rows = [
        (e["evidence"].get("usage") or {}) for e in entries if e["id"] in request_ids
    ]
    for key in ("input_tokens", "output_tokens"):
        if usage_rows and all(key in usage for usage in usage_rows):
            metrics[f"eval/request_{key}"] = sum(usage[key] for usage in usage_rows)
    if summary["recovery"]["rate"] is not None:
        metrics["eval/recovery_rate"] = summary["recovery"]["rate"]
    outcomes = {
        "passed",
        "model_failure",
        "model_error",
        "protocol_error",
        "budget_exhausted",
        "context_exhausted",
        "infrastructure_error",
        "interrupted",
        "not_started",
        "running",
    }
    metrics.update(
        {
            f"eval/outcome/{k}": summary["outcomes"].get(k, 0)
            for k in sorted(outcomes | set(summary["outcomes"]))
        }
    )
    metrics.update(request_metrics(record["requests"]))
    if record.get("track") == "coding":
        for key in (
            "first_implementation_passed",
            "first_implementation_pass_rate",
            "first_grading_missing",
            "recovery_denominator",
            "recovered_tasks",
            "recovery_rate",
        ):
            metrics.pop(f"eval/{key}", None)
    for kind, group in summary.get("by_task_type", {}).items():
        for key in (
            "planned_tasks",
            "passed_tasks",
            "final_pass_rate",
            "protocol_errors",
            "submitted_tasks",
        ):
            metrics[f"eval/type/{kind}/{key}"] = group[key]
        for status, count in group["command_problems"].items():
            metrics[f"eval/type/{kind}/command/{status}"] = count
    return metrics


def request_metrics(requests: dict[str, Any]) -> dict[str, Any]:
    return {
        "requests/count": requests["count"],
        "requests/pending_count": requests["pending_count"],
    }


def import_evaluation(path: Path, output_dir: Path, config: MonitorConfig) -> None:
    record = json.loads(path.read_text())
    monitor = RunMonitor(
        output_dir,
        config,
        phase="coding-eval" if record.get("track") == "coding" else "baseline-eval",
        metadata={
            "historical_import": True,
            "source_run": str(path.resolve().parent),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "snapshot": record["snapshot"],
            "evidence_kind": record["evidence_kind"],
            "comparison_sha256": record["comparison_sha256"],
            "code_revision": record.get("code_revision"),
            "data_source": {
                k: record.get("plan", {}).get(k)
                for k in ("source", "conversion", "split", "task_ids")
            },
        },
    )
    try:
        monitor.log("import", evaluation_metrics(record))
    finally:
        monitor.finish()


def _sync_arguments(state: dict[str, Any], caches: list[str]) -> list[str]:
    command = [
        "wandb",
        "sync",
        "--include-online",
        "--id",
        state["id"],
        "--project",
        state["project"],
    ]
    if state["entity"]:
        command.extend(["--entity", state["entity"]])
    return command + caches


def sync_monitor(run_dir: Path) -> None:
    state = json.loads((run_dir / "monitor.json").read_text())
    config = MonitorConfig(**state["config"]["monitoring"])
    key = config.credential()
    results = state.get("sync_results", [])
    caches = [cache for cache in state["caches"] if Path(cache).exists()]
    if caches:
        # W&B 0.30 sync orders segments of one run by start time itself.
        command = [sys.executable, "-m", *_sync_arguments(state, caches)]
        try:
            result = subprocess.run(
                command,
                cwd=run_dir.resolve(),
                env=os.environ | {"WANDB_API_KEY": key},
                capture_output=True,
                text=True,
                timeout=max(60, config.timeout_seconds),
            )
            results.append(
                {
                    "caches": caches,
                    "returncode": result.returncode,
                    "command": shlex.join(command),
                    "output": (result.stdout + result.stderr).replace(
                        key, "[REDACTED]"
                    ),
                }
            )
        except subprocess.TimeoutExpired:
            results.append(
                {"caches": caches, "returncode": None, "error_type": "TimeoutExpired"}
            )
    state["sync_results"] = results
    write_document(run_dir / "monitor.json", state)
    if caches and results[-1]["returncode"] != 0:
        raise ValueError("W&B cache sync did not finish; retry the recorded command")
    # Include events whose SDK log failed before reaching any W&B cache. Online
    # resume supplies the server's next step, so replays cannot double-count.
    monitor = RunMonitor.resume(run_dir, require_online=True)
    monitor.finish()
    if monitor.state["sync_status"] != "synced":
        raise ValueError("W&B metrics remain pending; retry the recorded sync command")
