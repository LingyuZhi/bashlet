"""Inputs and recorded outcomes of one small experiment."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


@dataclass(frozen=True)
class GradingCase:
    args: tuple[JSONValue, ...]
    expected: JSONValue


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    feedback_tests: str
    grading_cases: tuple[GradingCase, ...]
    reference_implementation: str = ""
    function_name: str = "solve"


@dataclass(frozen=True)
class Bash:
    command: str


@dataclass(frozen=True)
class Submit:
    """Submit the current solution.py and end the experiment's episode."""


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 2.0
    output_bytes: int = 16_384
    submission_bytes: int = 65_536
    memory_bytes: int = 536_870_912
    file_bytes: int = 1_048_576


@dataclass(frozen=True)
class Observation:
    status: Literal[
        "ok", "command_failed", "timeout", "output_limit", "infrastructure_error"
    ]
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class Isolation:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    runtime: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Step:
    action: Bash
    observation: Observation


@dataclass(frozen=True)
class Submission:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class CaseResult:
    passed: bool
    status: str
    reason: str = ""


@dataclass(frozen=True)
class Grade:
    passed: bool
    passed_cases: int
    total_cases: int
    cases: tuple[CaseResult, ...]


@dataclass
class ExperimentResult:
    run_dir: Path
    task_id: str
    task_sha256: str
    limits: Limits
    isolation: Isolation
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"
    trajectory: list[Step] = field(default_factory=list)
    submission: Submission | None = None
    grade: Grade | None = None
    reason: str = ""
