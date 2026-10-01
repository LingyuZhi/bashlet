"""Qwen ChatML framing bound to a real, identifiable tokenizer snapshot."""

import hashlib
import json
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import cast

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from .models import DecodedReply, JSONValue

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "config.json",
    "generation_config.json",
)


def _snapshot(name: str, revision: str, cache_dir: str | None) -> Path:
    local = Path(name)
    if local.is_dir():
        return local
    try:
        cached = Path(
            snapshot_download(
                name,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        )
        if all((cached / file).is_file() for file in _TOKENIZER_FILES[:2]):
            config = json.loads((cached / "tokenizer_config.json").read_text())
            if (
                config.get("chat_template")
                or (cached / "chat_template.jinja").is_file()
            ):
                return cached
    except LocalEntryNotFoundError:
        pass
    # Fetch tokenizer assets only, never model weights. Prefer the local cache.
    return Path(
        snapshot_download(
            name,
            revision=revision,
            cache_dir=cache_dir,
            allow_patterns=list(_TOKENIZER_FILES),
            endpoint="https://hf-mirror.com",
        )
    )


class QwenChatMLEncoding:
    """The official first prompt and append-only subsequent ChatML turns.

    ``tokenizer`` is a Hub name or a local tokenizer directory. Hub revisions
    resolve to a snapshot commit; file hashes also identify local snapshots.
    No model weights, remote code, or accelerator are needed.
    """

    def __init__(
        self,
        tokenizer: str,
        *,
        revision: str,
        thinking: bool = False,
        cache_dir: str | None = None,
    ) -> None:
        if not tokenizer.strip() or not revision.strip():
            raise ValueError("A tokenizer name and revision are required")
        if type(thinking) is not bool:
            raise ValueError("thinking must be a boolean")
        snapshot = _snapshot(tokenizer, revision, cache_dir)
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
        )
        if not isinstance(self._tokenizer.chat_template, str):
            raise ValueError("The Qwen tokenizer must provide one chat template")
        self.thinking = thinking
        self._im_end = self._token_id("<|im_end|>")
        self.stop_tokens = (self._im_end, self._token_id("<|endoftext|>"))
        self.thinking_start_token = self._token_id("<think>")
        self.thinking_end_token = self._token_id("</think>")
        probe = [{"role": "user", "content": ""}]
        closed = self._tokenizer.apply_chat_template(
            probe,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=thinking,
        )
        opened = self._tokenizer.apply_chat_template(
            probe,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        if closed != "<|im_start|>user\n<|im_end|>\n" or not opened.startswith(closed):
            raise ValueError("The tokenizer template is not supported Qwen ChatML")
        self._assistant_header: str = opened[len(closed) :]
        resolved_revision = (
            snapshot.name if snapshot.parent.name == "snapshots" else revision
        )
        file_hashes: dict[str, JSONValue] = {
            file: hashlib.sha256((snapshot / file).read_bytes()).hexdigest()
            for file in _TOKENIZER_FILES
            if (snapshot / file).is_file()
        }
        self._details: dict[str, JSONValue] = {
            "format": "qwen/chatml-v1",
            "tokenizer": tokenizer,
            "revision": resolved_revision,
            "files_sha256": file_hashes,
            "chat_template_sha256": hashlib.sha256(
                self._tokenizer.chat_template.encode()
            ).hexdigest(),
            "transformers_version": version("transformers"),
            "tokenizers_version": version("tokenizers"),
            "thinking": thinking,
            "stop_tokens": list(self.stop_tokens),
            "thinking_start_token": self.thinking_start_token,
            "thinking_end_token": self.thinking_end_token,
            "assistant_header": self._assistant_header,
        }
        digest = hashlib.sha256(
            json.dumps(self._details, sort_keys=True).encode()
        ).hexdigest()
        self.identity = f"qwen/chatml-v1:{tokenizer}@{resolved_revision}:{digest}"

    def _token_id(self, marker: str) -> int:
        token = self._tokenizer.get_vocab().get(marker)
        if token is None or self.encode(marker) != [token]:
            raise ValueError(f"The Qwen tokenizer must define {marker!r} as one token")
        return cast(int, token)

    def describe(self) -> dict[str, JSONValue]:
        """Serializable tokenizer identity, including the actual asset hashes."""
        return cast(dict[str, JSONValue], json.loads(json.dumps(self._details)))

    def encode(self, text: str) -> list[int]:
        return cast(list[int], self._tokenizer.encode(text, add_special_tokens=False))

    def prompt(self, system: str, user: str) -> list[int]:
        return cast(
            list[int],
            self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            ),
        )

    def decode(self, tokens: Sequence[int]) -> str:
        return cast(
            str,
            self._tokenizer.decode(
                list(tokens),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        )

    def continuation(self, generated: Sequence[int], observation: str) -> list[int]:
        close = "" if generated and generated[-1] == self._im_end else "<|im_end|>"
        return self.encode(
            close
            + "\n<|im_start|>user\n"
            + observation
            + "<|im_end|>\n"
            + self._assistant_header
        )

    def reply(self, tokens: Sequence[int]) -> DecodedReply:
        body = list(tokens)
        if body and body[-1] in self.stop_tokens:
            body.pop()
        if not self.thinking:
            return DecodedReply(self.decode(body))
        # The opening marker may already be in the official generation prompt.
        ends = [
            index
            for index, token in enumerate(body)
            if token == self.thinking_end_token
        ]
        boundary = ends[-1] if ends else len(body)
        reasoning = body[:boundary]
        if reasoning and reasoning[0] == self.thinking_start_token:
            reasoning.pop(0)
        return DecodedReply(
            action_text=self.decode(body[boundary + 1 :]) if ends else "",
            thinking=self.decode(reasoning),
            thinking_complete=bool(ends),
        )
