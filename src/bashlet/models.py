"""Inputs and recorded outcomes of one small experiment."""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


@dataclass(frozen=True)
class GradingCase:
    args: tuple[JSONValue, ...]
    expected: JSONValue


@dataclass(frozen=True)
class FunctionGrader:
    """Host-held function entry point and grading cases."""

    module_path: str
    function_name: str
    cases: tuple[GradingCase, ...]


@dataclass(frozen=True)
class CommandGrader:
    """Host-held test command and files injected only into the grading copy."""

    command: str
    hidden_files: dict[str, str]


@dataclass(frozen=True)
class TaskOrigin:
    """Immutable task-family identity inherited by variants and their trajectories."""

    dataset: str
    revision: str
    source_sha256: str
    original_task_id: str
    family_id: str
    split: Literal["train", "dev", "test"]


@dataclass(frozen=True)
class Task:
    task_id: str
    prompt: str
    initial_files: dict[str, str]
    grader: FunctionGrader | CommandGrader
    environment_id: str = "linux-bwrap-v1"
    reference_implementation: str = ""
    implementation_paths: tuple[str, ...] = ()
    feedback_command: str | None = None
    origin: TaskOrigin | None = None


@dataclass(frozen=True)
class Bash:
    command: str
    kind: Literal["bash"] = field(default="bash", init=False)


@dataclass(frozen=True)
class InvalidAction:
    """A generation the action protocol rejected; nothing was executed."""

    reason: str
    kind: Literal["invalid"] = field(default="invalid", init=False)


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None


@dataclass(frozen=True)
class Generation:
    """Raw model output; only sampled tokens carry sampling logprobs."""

    tokens: tuple[int, ...]
    stop_reason: Literal["stop", "length", "context"]
    logprobs: tuple[float, ...] | None = None
    request_id: str = ""
    source: Literal["sampled", "replayed"] = "sampled"

    def validate(self, sampling: SamplingParams) -> None:
        """Reject incomplete sampling evidence without repairing its values."""
        tokens, logprobs = self.tokens, self.logprobs
        problem = ""
        if self.source not in ("sampled", "replayed"):
            problem = f"Unknown generation source {self.source!r}"
        elif self.source == "replayed" and logprobs is not None:
            problem = "Replayed generations cannot carry sampling logprobs"
        elif self.source == "sampled" and logprobs is None:
            problem = "Sampled generations require per-token logprobs"
        elif logprobs is not None and len(tokens) != len(logprobs):
            problem = f"{len(tokens)} tokens but {len(logprobs)} logprobs"
        elif len(tokens) > sampling.max_tokens:
            problem = f"{len(tokens)} tokens exceed max_tokens={sampling.max_tokens}"
        elif not all(type(token) is int and token >= 0 for token in tokens):
            problem = "Token IDs must be non-negative integers"
        elif logprobs is not None and not all(
            type(value) in (int, float) and math.isfinite(value) and value <= 0
            for value in logprobs
        ):
            problem = "Logprobs must be finite and non-positive"
        elif self.stop_reason not in ("stop", "length", "context"):
            problem = f"Unknown stop reason {self.stop_reason!r}"
        if problem:
            raise ModelResponseError(problem, self)


class ModelResponseError(Exception):
    """A backend returned invalid model output, rather than a service failure."""

    def __init__(self, reason: str, generation: Generation | None = None) -> None:
        super().__init__(reason)
        self.generation = generation


class ContextExhausted(Exception):
    """The model cannot generate another token in the remaining context."""


@runtime_checkable
class Encoding(Protocol):
    """Tokenizer and chat template that frame the student's context."""

    @property
    def identity(self) -> str: ...

    @property
    def stop_tokens(self) -> tuple[int, ...]: ...

    def describe(self) -> dict[str, JSONValue]:
        """Recorded tokenizer identity and chat framing conditions."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode a pre-written reply using this tokenizer."""
        ...

    def prompt(self, system: str, user: str) -> list[int]:
        """Tokens up to and including the first assistant header."""
        ...

    def continuation(self, generated: Sequence[int], observation: str) -> list[int]:
        """Tokens that follow a raw generation: turn close, observation turn and
        the next assistant header. Earlier tokens are never re-encoded."""
        ...

    def decode(self, tokens: Sequence[int]) -> str: ...

    def reply(self, tokens: Sequence[int]) -> "DecodedReply":
        """Separate the action body from native thinking and terminal tokens."""
        ...


@dataclass(frozen=True)
class DecodedReply:
    action_text: str
    thinking: str | None = None
    thinking_complete: bool = True


@runtime_checkable
class Model(Protocol):
    """A sampling or replay model behind a token-based Adapter."""

    @property
    def identity(self) -> str: ...

    @property
    def encoding(self) -> Encoding: ...

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation: ...


@runtime_checkable
class ModelBackend(Model, Protocol):
    """An external sampling service with a cached capability probe and evidence."""

    def probe(self) -> None: ...

    @property
    def max_context_tokens(self) -> int: ...

    def describe(self) -> dict[str, JSONValue]: ...


@dataclass(frozen=True)
class InteractionBudget:
    max_steps: int = 10
    max_generated_tokens: int = 8192
    max_seconds: float = 120.0
    max_consecutive_invalid: int = 3


@dataclass(frozen=True)
class Limits:
    timeout_seconds: float = 30.0
    output_bytes: int = 16_384
    submission_bytes: int = 65_536
    submission_files: int = 1024
    memory_bytes: int = 536_870_912
    file_bytes: int = 1_048_576


@dataclass(frozen=True)
class Observation:
    status: Literal[
        "ok", "command_failed", "timeout", "interrupted", "infrastructure_error"
    ]
    stdout: str = ""
    stderr: str = ""
    output: str = ""
    output_truncated: bool = False
    exit_code: int | None = None
    duration_seconds: float = 0.0


@dataclass(frozen=True)
class Isolation:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    reason: str = ""
    runtime: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCall:
    """One generation request; ``context_offset`` is where its tokens start."""

    policy: str
    sampling: SamplingParams
    context_offset: int
    generation: Generation
    text: str
    duration_seconds: float
    reply: DecodedReply | None = None


type WorkspaceKind = Literal[
    "file",
    "directory",
    "symlink",
    "fifo",
    "socket",
    "block_device",
    "character_device",
    "unknown",
]


@dataclass(frozen=True)
class WorkspaceEntry:
    """One path in a recorded Task Workspace."""

    kind: WorkspaceKind
    sha256: str | None = None
    size_bytes: int | None = None
    target: str | None = None
    error: str = ""
    mode: int | None = None


@dataclass(frozen=True)
class Step:
    """One Interaction Step. ``feedback`` is the exact text the policy sees next;
    ``candidate`` is derived after the episode from the workspace manifest."""

    action: Bash | InvalidAction
    observation: Observation | None = None
    model_call: ModelCall | None = None
    feedback: str | None = None
    candidate: int | None = None
    verification: bool = False
    workspace_manifest: dict[str, WorkspaceEntry] = field(default_factory=dict)
    usage: "Usage | None" = None


@dataclass(frozen=True)
class Candidate:
    """A distinct state of the task's declared implementation paths."""

    index: int
    sha256: str
    path: str
    formed_at_step: int
    files: dict[str, str] = field(default_factory=dict)


@dataclass
class RepairAttempt:
    """Opens when a verification fails; closes when a different candidate is
    verified. It may span several Interaction Steps."""

    index: int
    failed_step: int
    from_candidate: int | None
    to_candidate: int | None = None
    verified_at_step: int | None = None
    passed: bool | None = None


@dataclass
class Usage:
    steps: int = 0
    generated_tokens: int = 0
    invalid_actions: int = 0
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class Submission:
    path: str
    sha256: str
    size_bytes: int
    file_count: int
    manifest: dict[str, WorkspaceEntry]


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


@dataclass(kw_only=True)
class EpisodeResult:
    """Task-independent interaction evidence and the final workspace."""

    run_dir: Path
    workspace: Path
    limits: Limits
    isolation: Isolation
    policy: str = ""
    policy_details: dict[str, JSONValue] = field(default_factory=dict)
    encoding: str = ""
    encoding_details: dict[str, JSONValue] = field(default_factory=dict)
    budget: InteractionBudget = field(default_factory=InteractionBudget)
    budget_kind: Literal["interaction", "spend"] | None = None
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"
    trajectory: list[Step] = field(default_factory=list)
    initial_workspace_manifest: dict[str, WorkspaceEntry] = field(default_factory=dict)
    context_tokens: list[int] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    rejected_generation: Generation | None = None
    submission_text: str | None = None
    reason: str = ""


@dataclass(kw_only=True)
class ExperimentResult(EpisodeResult):
    task_id: str
    task_sha256: str
    environment_id: str
    episode_status: str
    candidates: list[Candidate] = field(default_factory=list)
    repair_attempts: list[RepairAttempt] = field(default_factory=list)
    submission: Submission | None = None
    grade: Grade | None = None
    first_implementation: "FirstImplementation | None" = None

    @property
    def passed(self) -> bool:
        """Unsubmitted episodes are failures, even if feedback passed."""
        return self.grade is not None and self.grade.passed


@dataclass
class FirstImplementation:
    """Host-only, post-episode grading of the earliest recorded candidate."""

    status: str = "no_candidate"
    candidate_index: int | None = None
    formed_at_step: int | None = None
    snapshot: Submission | None = None
    grade: Grade | None = None
    reason: str = ""
