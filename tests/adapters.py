"""Controllable model-call Adapter for tests. It is a test tool, not a backend."""

from collections.abc import Callable, Sequence
from typing import Literal

from bashlet import (
    DecodedReply,
    Generation,
    InteractionBudget,
    Limits,
    ModelResponseError,
    PyTRIOModel,
    ReplayModel,
    SamplingParams,
)
from bashlet.models import JSONValue

SYSTEM, USER, ASSISTANT, END = 256, 257, 258, 259
# Whole-word tokens a model may sample. ByteEncoding never produces them when
# encoding text, so decode → re-encode would silently change such generations.
WORDS = {300: "solve", 301: "return"}
_MARKERS = {SYSTEM: "<|system|>", USER: "<|user|>", ASSISTANT: "<|assistant|>"}


class ByteEncoding:
    """A reversible toy chat template: bytes, four special tokens, two words."""

    identity = "test/byte-chat-v1"
    stop_tokens = (END,)

    def describe(self) -> dict[str, JSONValue]:
        return {"format": self.identity, "thinking": False, "stop_tokens": [END]}

    def reply(self, tokens: Sequence[int]) -> DecodedReply:
        return DecodedReply(self.decode(tokens))

    def encode(self, text: str) -> list[int]:
        return list(text.encode())

    def prompt(self, system: str, user: str) -> list[int]:
        return [SYSTEM, *system.encode(), END, USER, *user.encode(), END, ASSISTANT]

    def continuation(self, generated: Sequence[int], observation: str) -> list[int]:
        close = [] if generated and generated[-1] == END else [END]
        return [*close, USER, *observation.encode(), END, ASSISTANT]

    def decode(self, tokens: Sequence[int]) -> str:
        return b"".join(
            bytes([token]) if token < 256 else WORDS.get(token, "").encode()
            for token in tokens
        ).decode("utf-8", errors="replace")

    def render(self, tokens: Sequence[int]) -> str:
        """The whole context as the model sees it, special tokens included."""
        return "".join(
            _MARKERS.get(token, "<|end|>" if token == END else self.decode([token]))
            for token in tokens
        )


def sample_text(reply: str, max_tokens: int) -> Generation:
    """Tokenize like a sampler would: prefer whole-word tokens, then stop."""
    tokens: list[int] = []
    position = 0
    while position < len(reply):
        word = next(
            (
                token
                for token, text in WORDS.items()
                if reply.startswith(text, position)
            ),
            None,
        )
        if word is not None:
            tokens.append(word)
            position += len(WORDS[word])
        else:
            tokens.extend(reply[position].encode())
            position += 1
    tokens.append(END)
    stop_reason: Literal["stop", "length"] = (
        "length" if len(tokens) > max_tokens else "stop"
    )
    tokens = tokens[:max_tokens]
    return Generation(
        tokens=tuple(tokens),
        logprobs=tuple(-0.05 * (1 + index % 7) for index in range(len(tokens))),
        stop_reason=stop_reason,
        request_id=f"scripted-{len(reply)}",
    )


class ScriptedModel:
    """Chooses each reply only from the context it has actually been sent."""

    def __init__(
        self,
        respond: Callable[[str], str | Generation],
        identity: str = "test/scripted-student@step-0",
    ) -> None:
        self.identity = identity
        self.encoding = ByteEncoding()
        self.respond = respond
        self.prompts: list[tuple[int, ...]] = []
        self.requests: list[SamplingParams] = []
        self.generations: list[Generation] = []

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        self.prompts.append(tuple(prompt))
        self.requests.append(sampling)
        reply = self.respond(self.encoding.render(prompt))
        generation = (
            reply
            if isinstance(reply, Generation)
            else sample_text(reply, sampling.max_tokens)
        )
        self.generations.append(generation)
        return generation


class EvaluationModel(ScriptedModel):
    """CLI-loadable external model double with task-controlled outcomes."""

    def __init__(
        self, thinking: bool = False, identity: str = "test/evaluation-v1"
    ) -> None:
        assert not thinking
        super().__init__(self._respond, identity=identity)

    def _respond(self, context: str) -> str:
        task = context.split("<|user|>")[1]
        if "interrupt\n" in task:
            raise KeyboardInterrupt
        if "model\n" in task:
            raise ModelResponseError("Invalid model evidence")
        if "service\n" in task:
            raise RuntimeError("Service unavailable")
        if "protocol\n" in task:
            return "Missing bash block"
        if "budget\n" in task:
            return bash("ls")
        if turn_number(context) == 1:
            return write_file(
                "def double(n): return 4\n"
                if "wrong\n" in task
                else "def double(n): return n * 2\n"
            )
        return submit()


class PaidEvaluationModel(PyTRIOModel):
    """Construct the production adapter against the controllable SDK boundary."""

    def __init__(
        self,
        ledger_path: str,
        thinking: bool = False,
        fail_before_identity: bool = False,
    ) -> None:
        from pathlib import Path

        from test_pytrio import MODEL, REVISION, FakeSampling, FakeService, response

        from bashlet import SpendLedger, SpendPolicy

        ledger = SpendLedger(Path(ledger_path), policy=SpendPolicy(funded_cny=0.16))
        sdk = FakeSampling(
            [
                response([65]),
                response(list(write_file("def double(n): return n * 2\n").encode())),
                response(list(submit().encode())),
            ]
        )
        sdk.fail_before_identity = fail_before_identity
        super().__init__(
            model=MODEL,
            model_revision=REVISION,
            encoding=ByteEncoding(),
            service_client=FakeService(sdk),
            ledger=ledger,
            group="development-baseline",
            operation="evaluation",
            retry_delay_seconds=0,
        )


def turn_number(context: str) -> int:
    """1 for the first reply; each assistant header opens one reply."""
    return context.count("<|assistant|>")


def latest_feedback(context: str) -> str:
    """The most recent user turn after the task, or '' before any action."""
    turns = context.split("<|user|>")
    return turns[-1] if len(turns) > 2 else ""


def bash(command: str) -> str:
    return f"Run the next command.\n```bash\n{command}\n```"


def write_file(source: str) -> str:
    return bash(f"cat > solution.py <<'PY'\n{source}PY")


def submit() -> str:
    return bash("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'")


def replay(replies: Sequence[str]) -> ReplayModel:
    return ReplayModel(replies, encoding=ByteEncoding())


def condition_overrides(
    *,
    budget: InteractionBudget | None = None,
    sampling: SamplingParams | None = None,
    limits: Limits | None = None,
    bwrap: str | None = None,
) -> list[str]:
    """key=value overlays that reproduce a budget, sampling condition, or limits."""
    overrides: list[str] = []
    if budget is not None:
        overrides.extend(
            [
                f"budget.max_steps={budget.max_steps}",
                f"budget.max_generated_tokens={budget.max_generated_tokens}",
                f"budget.max_seconds={budget.max_seconds}",
                f"budget.max_consecutive_invalid={budget.max_consecutive_invalid}",
            ]
        )
    if sampling is not None:
        seed = "null" if sampling.seed is None else str(sampling.seed)
        overrides.extend(
            [
                f"model.sampling.max_tokens={sampling.max_tokens}",
                f"model.sampling.temperature={sampling.temperature}",
                f"model.sampling.top_p={sampling.top_p}",
                f"model.sampling.seed={seed}",
            ]
        )
    if limits is not None:
        overrides.extend(
            [
                f"environment.timeout_seconds={limits.timeout_seconds}",
                f"environment.output_bytes={limits.output_bytes}",
                f"environment.submission_bytes={limits.submission_bytes}",
                f"environment.submission_files={limits.submission_files}",
                f"environment.memory_bytes={limits.memory_bytes}",
                f"environment.file_bytes={limits.file_bytes}",
            ]
        )
    if bwrap is not None:
        overrides.append(f"environment.bwrap={bwrap}")
    return overrides
