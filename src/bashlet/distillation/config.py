"""Resolve training encoding and enforce the published cohort and visit plan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from bashlet.config import (
    ExperimentConfig,
    ModelConfig,
    bind_config,
    load_config,
    read_yaml,
)
from bashlet.distillation.pytrio import PyTRIOOPSDConfig
from bashlet.tasks.freeze import validate_selection


def resolve_condition(config: RunConfig) -> tuple[ExperimentConfig, PyTRIOOPSDConfig]:
    backend = PyTRIOOPSDConfig(**config.backend)
    explicit = any(
        "thinking" in read_yaml(Path(p)).get("model", {}) for p in config.condition
    )
    condition = load_config(
        [Path(p) for p in config.condition],
        [] if explicit else [f"model.thinking={backend.thinking}"],
    )
    thinking = condition.model.parameters["thinking"]
    if "thinking" in config.backend and backend.thinking != thinking:
        raise ValueError("Student and Teacher thinking settings must match")
    backend = replace(backend, thinking=thinking)
    condition = replace(
        condition,
        model=ModelConfig(
            "PyTRIOModel",
            replace(condition.model.sampling, seed=backend.seed),
            {
                "model": backend.base_model,
                "model_revision": backend.model_revision,
                "tokenizer": backend.base_model,
                "tokenizer_revision": backend.model_revision,
                "tokenizer_cache_dir": backend.tokenizer_cache_dir,
                "thinking": backend.thinking,
                "timeout_seconds": backend.timeout_seconds,
                "max_retries": 0,
                "api_key_env": backend.api_key_env,
                "dotenv_path": backend.dotenv_path,
            },
        ),
    )
    return condition, backend


def validate_training_selection(
    config: RunConfig,
    manifest: dict[str, Any],
) -> str | None:
    if config.training_selection is None:
        return None
    root = Path(config.train_manifest).resolve().parent
    validate_selection(root, manifest)
    path = Path(config.training_selection).resolve()
    artifact = manifest.get("training_selection", {})
    if path != root / artifact.get("file", ""):
        raise ValueError("Frozen training selection must belong to the manifest")
    raw = path.read_bytes()
    selected = json.loads(raw)
    # The dataset fixes membership and provenance. Each new run freezes its own
    # epochs, sampling seeds and visit order in the run configuration/journal.
    if (
        hashlib.sha256(raw).hexdigest() != artifact["sha256"]
        or config.train_task_ids != selected["task_ids"]
    ):
        raise ValueError("Frozen training subset or provenance differs")
    from bashlet.evaluation.harness import _tasks

    plan, _ = _tasks(Path(config.dev_plan))
    if (
        (Path(config.dev_plan).parent / plan["manifest"]).resolve()
        != Path(config.train_manifest).resolve()
        or plan["split"] != "dev"
        or plan["task_ids"] != manifest["cohort"]["development"]["task_ids"]
    ):
        raise ValueError("Frozen training requires the complete new development split")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    train_manifest: str
    train_task_ids: list[str]
    dev_plan: str
    condition: list[str]
    ledger_path: str = "runs/spend.sqlite3"
    backend: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    evaluation_group: str | None = None
    stop_on_training_budget: bool = False
    epochs: int = 1
    shuffle_seed: int | None = None
    evaluation_seeds: list[int] = field(default_factory=list)
    evaluate_each_epoch: bool = False
    evaluation_groups: dict[str, str] = field(default_factory=dict)
    require_complete_training: bool = False
    training_selection: str | None = None
    preflight_evidence: str | None = None
    output_dir: str = "runs/opsd"
    preflight: str | None = None
    evaluation: dict[str, Any] | None = None
    execution_limit: str | None = None
    monitor: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, path: Path) -> RunConfig:
        settings = read_yaml(path)
        for key in ("group", "spend_config"):
            if key not in settings:
                continue
            shared = settings.pop(key)
            for section in ("backend", "training"):
                target = settings.setdefault(section, {})
                if key in target and target[key] != shared:
                    raise ValueError(f"{section}.{key} conflicts with the shared {key}")
                target[key] = shared
        if "train_task_ids" not in settings and settings.get("training_selection"):
            selected = json.loads(Path(settings["training_selection"]).read_text())
            settings["train_task_ids"] = selected["task_ids"]
        if "evaluation_seeds" not in settings:
            settings["evaluation_seeds"] = [
                PyTRIOOPSDConfig(**settings.get("backend", {})).seed
            ]
        value = bind_config(cls, settings)
        if not value.evaluation_seeds:
            raise ValueError("Evaluation needs at least one explicit seed")
        if value.evaluation is not None:
            _validate_evaluation(value.evaluation)
        if not value.train_task_ids or len(set(value.train_task_ids)) != len(
            value.train_task_ids
        ):
            raise ValueError("Pilot needs unique fixed training task IDs")
        if type(value.epochs) is not int or value.epochs < 1:
            raise ValueError("Pilot epochs must be a positive integer")
        if value.shuffle_seed is not None and (
            type(value.shuffle_seed) is not int or value.shuffle_seed < 0
        ):
            raise ValueError("Pilot shuffle seed must be a nonnegative integer")
        if len(set(value.evaluation_seeds)) != len(value.evaluation_seeds) or any(
            type(seed) is not int or seed < 0 for seed in value.evaluation_seeds
        ):
            raise ValueError(
                "Pilot evaluation seeds must be distinct nonnegative integers"
            )
        if type(value.evaluate_each_epoch) is not bool:
            raise ValueError("Pilot epoch evaluation must be a boolean")
        if type(value.require_complete_training) is not bool:
            raise ValueError("Complete training requirement must be a boolean")
        if value.evaluation_groups and (
            set(value.evaluation_groups) != {"baseline", "intermediate", "final"}
            or any(
                not isinstance(group, str) or not group
                for group in value.evaluation_groups.values()
            )
        ):
            raise ValueError(
                "Pilot evaluation groups require baseline, intermediate and final"
            )
        return value


def _validate_evaluation(settings: dict[str, Any]) -> None:
    allowed = {"base_path", "test_plan", "executor", "model", "server", "monitoring"}
    if unknown := set(settings) - allowed:
        raise ValueError(f"Unknown evaluation setting: {sorted(unknown)[0]}")
    if not isinstance(settings.get("base_path"), str) or not settings["base_path"]:
        raise ValueError("Local evaluation needs a base_path")
    model = settings.get("model")
    allowed_model = {
        "base_url",
        "model",
        "api_key_env",
        "timeout_seconds",
        "max_retries",
        "retry_delay_seconds",
    }
    if not isinstance(model, dict) or set(model) - allowed_model:
        raise ValueError("Local evaluation model must contain only service settings")
    if not isinstance(model.get("base_url"), str) or not model["base_url"].startswith(
        ("http://", "https://")
    ):
        raise ValueError("Local evaluation needs a service base_url")
    executor = settings.get("executor", [])
    if not isinstance(executor, list) or any(
        not isinstance(part, str) or not part for part in executor
    ):
        raise ValueError(
            "Local evaluation executor must be a list of command arguments"
        )
    server = settings.get("server")
    if server is not None:
        allowed_server = {"session", "template", "cuda_module", "timeout_seconds"}
        if (
            not isinstance(server, dict)
            or set(server) - allowed_server
            or any(
                not isinstance(server.get(key), str) or not server[key]
                for key in ("session", "template", "cuda_module")
            )
        ):
            raise ValueError(
                "Local evaluation server needs session, template and cuda_module"
            )


class WatchdogError(ValueError):
    """The watchdog cannot safely inspect or control the configured run."""


def _monitor_settings(path: Path) -> dict[str, Any]:

    config = RunConfig.read(path)
    if config.preflight is None:
        raise WatchdogError("The workflow monitor requires a configured preflight")
    preflight = RunConfig.read(Path(config.preflight))
    monitor = config.monitor
    root = Path(config.output_dir)
    paths = {
        "monitor_root": root / "monitor",
        "workflow": root / "workflow.json",
        "preflight_pilot": root / "preflight/pilot.json",
        "preflight_training": root / "preflight/training/opsd.json",
        "preflight_validation": root / "preflight/checkpoint-validation.json",
        "formal_pilot": root / "formal/pilot.json",
        "formal_training": root / "formal/training/opsd.json",
        "ledger": Path(config.ledger_path),
    }
    return {
        "schema_version": 1,
        "repository_root": monitor["repository_root"],
        "monitor_session": monitor["monitor_session"],
        "worker": {
            "session": monitor["worker_session"],
            "log": str(root / "workflow.log"),
            "command": [
                "uv",
                "run",
                "--locked",
                "bashlet",
                "train-opsd",
                str(path.resolve()),
                "--resume",
            ],
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "frozen_configurations": [str(path.resolve()), config.preflight],
        "ledger_groups": [preflight.training["group"], config.training["group"]],
        "timing": {
            key: monitor[key]
            for key in ("poll_seconds", "backoff_min_seconds", "backoff_max_seconds")
        },
    }


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
            raw = read_yaml(path)
        except (OSError, yaml.YAMLError):
            raise WatchdogError("monitor configuration is unreadable") from None
        if "monitor" in raw:
            raw = _monitor_settings(path)
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
        if (
            not isinstance(command, list)
            or not all(isinstance(part, str) and part for part in command)
            or "--resume" not in command
        ):
            raise WatchdogError(
                "worker command must use an explicit --resume entrypoint"
            )

        config_files_raw = raw.get("frozen_configurations")
        if not isinstance(config_files_raw, list) or not config_files_raw:
            raise WatchdogError("frozen_configurations must be a non-empty list")
        config_files = tuple(_resolve(root, item) for item in config_files_raw)
        groups = raw.get("ledger_groups")
        if (
            not isinstance(groups, list)
            or len(groups) < 2
            or any(not isinstance(group, str) or not group for group in groups)
            or len(groups) != len(set(groups))
        ):
            raise WatchdogError(
                "ledger_groups require distinct preflight and training allocations"
            )
        expected_groups = tuple(groups)
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
