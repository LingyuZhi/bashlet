"""Versioned experimental conditions: protocol, environment, budget, and model."""

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, cast, get_args, get_origin, get_type_hints

import yaml
from jinja2 import Environment, StrictUndefined, TemplateError

from .models import InteractionBudget, SamplingParams

_DEFAULT_PATH = Path(__file__).with_name("configs") / "bash-block-v1.yaml"
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
    invalid_action_format: str
    history: str
    verification_pattern: str


@dataclass(frozen=True)
class EnvironmentConfig:
    """Executor, mounts, environment, and per-command resource limits."""

    bwrap: str
    timeout_seconds: float
    output_bytes: int
    submission_bytes: int
    memory_bytes: int
    file_bytes: int
    nofile: int
    path: str
    lang: str
    ro_bind: tuple[Mount, ...]
    symlink: tuple[Mount, ...]


@dataclass(frozen=True)
class ModelConfig:
    """Sampling condition recorded with a run. The caller still supplies the model."""

    backend: str
    sampling: SamplingParams
    thinking: bool


@dataclass(frozen=True)
class ExperimentConfig:
    protocol: ProtocolConfig
    environment: EnvironmentConfig
    budget: InteractionBudget
    model: ModelConfig


def load_config(
    paths: Sequence[Path] = (),
    overrides: Sequence[str] = (),
) -> ExperimentConfig:
    """Layer the built-in default, then ``paths``, then ``key=value`` overrides.

    Unknown keys and type errors are rejected before a run starts. Protocol
    strings are Jinja templates; an undefined variable is an error.
    """
    merged = _read_yaml(_DEFAULT_PATH)
    for path in paths:
        merged = _merge(merged, _read_yaml(path))
    for override in overrides:
        _apply_override(merged, override)
    config = _bind(ExperimentConfig, merged, "config")
    protocol = _render_protocol(config.protocol)
    try:
        re.compile(protocol.verification_pattern)
    except re.error as error:
        raise ValueError(
            f"protocol.verification_pattern is invalid: {error}"
        ) from error
    _require_positive_numbers(config)
    return ExperimentConfig(protocol, config.environment, config.budget, config.model)


def _read_yaml(path: Path) -> dict[str, Any]:
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
    return cast(dict[str, Any], loaded)


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
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"Unknown configuration key {key!r}")
    node[parts[-1]] = value


def _bind(cls: type[Any], data: object, path: str) -> Any:
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping")
    hints = get_type_hints(cls)
    known = set(hints)
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown configuration key {_join(path, unknown[0])!r}")
    missing = sorted(known - set(data))
    if missing:
        raise ValueError(f"Missing configuration key {_join(path, missing[0])!r}")
    return cls(
        **{name: _convert(hints[name], data[name], _join(path, name)) for name in hints}
    )


def _convert(hint: Any, value: object, path: str) -> Any:
    origin = get_origin(hint)
    if origin is tuple:
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        inner = get_args(hint)[0]
        return tuple(
            _convert(inner, item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if origin in (UnionType,):
        if value is None and type(None) in get_args(hint):
            return None
        inner = next(arg for arg in get_args(hint) if arg is not type(None))
        return _convert(inner, value, path)
    if isinstance(hint, type) and is_dataclass(hint):
        return _bind(hint, value, path)
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
    rendered: dict[str, str] = {}
    for field in fields(protocol):
        text = getattr(protocol, field.name)
        try:
            rendered[field.name] = _JINJA.from_string(text).render()
        except TemplateError as error:
            raise ValueError(
                f"Protocol template {field.name} failed: {error}"
            ) from error
    return ProtocolConfig(**rendered)


def _require_positive_numbers(config: ExperimentConfig) -> None:
    numbers: dict[str, float] = {
        "environment.timeout_seconds": config.environment.timeout_seconds,
        "environment.output_bytes": config.environment.output_bytes,
        "environment.submission_bytes": config.environment.submission_bytes,
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
