"""Pre-written replies behind the same token interface as a sampling model."""

from collections.abc import Sequence

from .models import Encoding, Generation, SamplingParams


class Utf8Encoding:
    """A byte encoding for replay-only CLI demonstrations, without a model."""

    identity = "replay/utf8-v1"

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def prompt(self, system: str, user: str) -> list[int]:
        return self.encode(f"[system]\n{system}\n[user]\n{user}\n[assistant]\n")

    def continuation(self, generated: Sequence[int], observation: str) -> list[int]:
        return self.encode(f"\n[user]\n{observation}\n[assistant]\n")

    def decode(self, tokens: Sequence[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


class ReplayModel:
    """Replay replies without inventing a policy's sampling probabilities.

    Exhausted replies produce empty generations, which the Harness handles
    through its ordinary invalid-action and interaction-budget rules.
    """

    def __init__(
        self,
        replies: Sequence[str] = (),
        *,
        encoding: Encoding | None = None,
        identity: str = "replay/prewritten-v1",
        thinking: bool = False,
    ) -> None:
        if isinstance(replies, str) or not all(
            isinstance(reply, str) for reply in replies
        ):
            raise ValueError("Replay replies must be a list of strings")
        self.identity = identity
        self.encoding = encoding if encoding is not None else Utf8Encoding()
        self.thinking = thinking
        self._replies = iter(tuple(replies))

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        tokens = self.encoding.encode(next(self._replies, ""))
        return Generation(
            tokens=tuple(tokens[: sampling.max_tokens]),
            stop_reason="length" if len(tokens) > sampling.max_tokens else "stop",
            source="replayed",
        )
