"""Versioned experimental conditions: protocol, environment, budget, and model."""

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, asdict, dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, cast, get_args, get_origin, get_type_hints

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError

from bashlet.backends.pytrio import PyTRIOModel
from bashlet.backends.replay import ReplayModel
from bashlet.backends.vllm import VLLMModel
from bashlet.harness.models import Encoding, InteractionBudget, Model, SamplingParams

_DEFAULT_PATH = Path(__file__).with_name("configs") / "defaults.yaml"
_SOURCE_DEFAULT = Path(__file__).resolve().parents[2] / "configs" / _DEFAULT_PATH.name
if _SOURCE_DEFAULT.is_file():
    _DEFAULT_PATH = _SOURCE_DEFAULT
_JINJA = Environment(undefined=StrictUndefined, keep_trailing_newline=True)


@dataclass(frozen=True)
class Mount:
    source: str
    target: str


@dataclass(frozen=True)
class ProtocolConfig:
    """Agent-visible text and the rules that define the protocol fingerprint."""

    name: str
    system_prompt: str
    user_prompt: str
    observation_template: str
    timeout_template: str
    observation_head_chars: int
    observation_tail_chars: int
    elision_template: str
    output_limit_message: str
    invalid_action_format: str
    history: str
    submission_marker: str


@dataclass(frozen=True)
class EnvironmentConfig:
    """Executor, mounts, environment, and per-command resource limits."""

    id: str
    bwrap: str
    timeout_seconds: float
    probe_timeout_seconds: float
    output_bytes: int
    submission_bytes: int
    submission_files: int
    memory_bytes: int
    file_bytes: int
    nofile: int
    path: str
    lang: str
    home: str
    python: Mount
    python_version: str
    pytest_version: str
    variables: dict[str, str]
    etc_files: dict[str, str]
    ro_bind: tuple[Mount, ...]
    symlink: tuple[Mount, ...]


@dataclass(frozen=True)
class ModelConfig:
    """Model class and constructor arguments, separate from Harness sampling."""

    class_name: str
    sampling: SamplingParams
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    protocol: ProtocolConfig
    environment: EnvironmentConfig
    budget: InteractionBudget
    model: ModelConfig


def config_record(config: ExperimentConfig) -> dict[str, Any]:
    """Record the resolved condition using the same keys as its YAML input."""
    return asdict(config) | {
        "model": {
            "class": config.model.class_name,
            "sampling": asdict(config.model.sampling),
            **config.model.parameters,
        }
    }


def load_model(config: ModelConfig) -> Model:
    """Instantiate a configured class without coupling the Harness to backends."""
    name = config.class_name
    factory: type[Any]
    try:
        if name == "ReplayModel":
            factory = ReplayModel
        elif name == "VLLMModel":
            factory = VLLMModel
        elif name == "PyTRIOModel":
            factory = PyTRIOModel
        else:
            module_name, separator, class_name = name.replace(":", ".").rpartition(".")
            if not separator:
                raise ValueError(
                    "Use ReplayModel, VLLMModel, PyTRIOModel "
                    "or a fully qualified model class"
                )
            factory = getattr(importlib.import_module(module_name), class_name)
        if not isinstance(factory, type):
            raise ValueError(f"{name!r} is not a model class")
        model = factory(**config.parameters)
        if not isinstance(model, Model) or not isinstance(model.encoding, Encoding):
            raise ValueError("The model must provide identity, encoding and sample")
        return model
    except Exception as error:
        raise ValueError(f"Cannot construct model {name!r}: {error}") from error


def load_config(
    paths: Sequence[Path] = (),
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Layer the built-in default, then ``paths``, then ``key=value`` overrides.

    Unknown keys and type errors are rejected before a run starts. Protocol
    strings are Jinja templates; an undefined variable is an error.
    """
    merged = read_yaml(_DEFAULT_PATH)
    for path in paths:
        merged = _merge(merged, read_yaml(path))
    for override in overrides:
        _apply_override(merged, override)
    config = bind_config(ExperimentConfig, merged, "config")
    protocol = _render_protocol(config.protocol)
    _require_positive_numbers(config)
    return ExperimentConfig(protocol, config.environment, config.budget, config.model)


def read_yaml(path: Path, *, _parents: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve YAML inheritance and paths relative to their declaring file.

    Parents are merged in order, then the child. Lists replace earlier lists.
    Cycles and malformed references fail before any experiment work starts.
    """
    path = path.resolve()
    if path in _parents:
        raise ValueError(f"Configuration reference cycle: {path}")
    try:
        text = path.read_text()
    except OSError:
        raise
    try:
        loaded: object = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping")
    parents = loaded.pop("extends", [])
    if isinstance(parents, str):
        parents = [parents]
    if not isinstance(parents, list) or any(
        not isinstance(item, str) or not item for item in parents
    ):
        raise ValueError(f"{path}: extends must contain configuration paths")
    base: dict[str, Any] = {}
    for parent in parents:
        base = _merge(base, read_yaml(path.parent / parent, _parents=(*_parents, path)))
    _resolve_paths(loaded, path.parent)
    environment = loaded.get("environment")
    if isinstance(environment, dict):
        python = environment.get("python")
        if isinstance(python, dict) and isinstance(python.get("source"), str):
            python["source"] = str((path.parent / python["source"]).resolve())
    return _merge(base, cast(dict[str, Any], loaded))


def _resolve_paths(data: dict[str, Any], directory: Path) -> None:
    scalar = {
        "train_manifest",
        "dev_plan",
        "training_selection",
        "preflight_evidence",
        "ledger_path",
        "spend_config",
        "base_path",
        "test_plan",
        "template",
        "output_dir",
        "preflight",
        "execution_limit",
        "dotenv_path",
        "tokenizer_cache_dir",
        "snapshot_manifest",
        "repository_root",
    }
    for key, value in data.items():
        if isinstance(value, dict):
            _resolve_paths(value, directory)
        elif key in scalar and isinstance(value, str):
            data[key] = str((directory / value).resolve())
        elif key == "tokenizer" and isinstance(value, str) and value.startswith(
            ("./", "../", "/")
        ):
            # Explicit local paths are distinct from Hub IDs such as Qwen/model.
            data[key] = str((directory / value).resolve())
        elif key == "condition" and isinstance(value, list):
            if any(not isinstance(item, str) for item in value):
                raise ValueError("condition must contain configuration paths")
            data[key] = [str((directory / item).resolve()) for item in value]


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _merge(cast(dict[str, Any], current), value)
        else:
            merged[key] = value
    return merged


def _apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be key=value, got {expression!r}")
    raw_key, raw_value = expression.split("=", 1)
    key = raw_key.strip()
    if not key:
        raise ValueError(f"Override must be key=value, got {expression!r}")
    try:
        value: object = yaml.safe_load(raw_value)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid value for {key}: {error}") from error
    node: object = config
    parts = key.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or not isinstance(node.get(part), dict):
            raise ValueError(f"Unknown configuration key {key!r}")
        node = node[part]
    constructor_key = len(parts) >= 2 and parts[0] == "model" and parts[1] != "sampling"
    if not isinstance(node, dict) or (parts[-1] not in node and not constructor_key):
        raise ValueError(f"Unknown configuration key {key!r}")
    node[parts[-1]] = value


def bind_config[T](cls: type[T], data: object, path: str = "config") -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    if cls is ModelConfig:
        parameters = dict(data)
        name = _convert(str, parameters.pop("class", None), "model.class")
        if not name.strip():
            raise ValueError("model.class must not be empty")
        sampling = bind_config(
            SamplingParams, parameters.pop("sampling", None), "model.sampling"
        )
        if "thinking" in parameters:
            _convert(bool, parameters["thinking"], "model.thinking")
        return cast(T, ModelConfig(name, sampling, parameters))
    if not is_dataclass(cls):
        raise ValueError("Configuration schema must be a dataclass")
    hints = get_type_hints(cls)
    known = set(hints)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown configuration key {_join(path, unknown[0])!r}")
    required = {
        field.name
        for field in fields(cls)
        if field.default is MISSING and field.default_factory is MISSING
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Missing configuration key {_join(path, missing[0])!r}")
    return cls(
        **{name: _convert(hints[name], data[name], _join(path, name)) for name in data}
    )


def _convert(hint: Any, value: object, path: str) -> Any:
    if hint is Any:
        return value
    origin = get_origin(hint)
    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be a mapping")
        key_type, value_type = get_args(hint)
        return {
            _convert(key_type, key, path): _convert(value_type, item, f"{path}.{key}")
            for key, item in value.items()
        }
    if origin in (tuple, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        inner = get_args(hint)[0]
        result = [
            _convert(inner, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return tuple(result) if origin is tuple else result
    if origin in (UnionType,):
        if value is None and type(None) in get_args(hint):
            return None
        inner = next(arg for arg in get_args(hint) if arg is not type(None))
        return _convert(inner, value, path)
    if isinstance(hint, type) and is_dataclass(hint):
        return bind_config(hint, value, path)
    if hint is bool:
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean")
        return value
    if hint is int:
        if type(value) is not int:
            raise ValueError(f"{path} must be an integer")
        return value
    if hint is float:
        if type(value) is int:
            return float(value)
        if type(value) is float:
            return value
        raise ValueError(f"{path} must be a number")
    if hint is str:
        if type(value) is not str:
            raise ValueError(f"{path} must be a string")
        return value
    raise ValueError(f"Unsupported configuration type for {path}")


def _join(path: str, name: str) -> str:
    return name if path == "config" else f"{path}.{name}"


def _render_protocol(protocol: ProtocolConfig) -> ProtocolConfig:
    rendered: dict[str, Any] = {}
    for field in fields(protocol):
        value = getattr(protocol, field.name)
        if not isinstance(value, str):
            rendered[field.name] = value
            continue
        try:
            rendered[field.name] = _JINJA.from_string(value).render()
        except TemplateError as error:
            raise ValueError(
                f"Protocol template {field.name} failed: {error}"
            ) from error
    return ProtocolConfig(**rendered)


def _require_positive_numbers(config: ExperimentConfig) -> None:
    numbers: dict[str, float] = {
        "protocol.observation_head_chars": config.protocol.observation_head_chars,
        "protocol.observation_tail_chars": config.protocol.observation_tail_chars,
        "environment.timeout_seconds": config.environment.timeout_seconds,
        "environment.probe_timeout_seconds": config.environment.probe_timeout_seconds,
        "environment.output_bytes": config.environment.output_bytes,
        "environment.submission_bytes": config.environment.submission_bytes,
        "environment.submission_files": config.environment.submission_files,
        "environment.memory_bytes": config.environment.memory_bytes,
        "environment.file_bytes": config.environment.file_bytes,
        "environment.nofile": config.environment.nofile,
        "budget.max_steps": config.budget.max_steps,
        "budget.max_generated_tokens": config.budget.max_generated_tokens,
        "budget.max_seconds": config.budget.max_seconds,
        "budget.max_consecutive_invalid": config.budget.max_consecutive_invalid,
        "model.sampling.max_tokens": config.model.sampling.max_tokens,
    }
    for path, value in numbers.items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path} must be finite and positive")
