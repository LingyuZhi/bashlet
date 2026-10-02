"""Real Qwen tokenization, checked against the official chat template."""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from bashlet import (
    CommandGrader,
    Generation,
    InvalidAction,
    QwenChatMLEncoding,
    ReplayModel,
    SamplingParams,
    Task,
    run_experiment,
)

QWEN = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


@pytest.fixture(scope="session")
def official_tokenizer() -> Any:
    snapshot = snapshot_download(QWEN, revision=REVISION, local_files_only=True)
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)


@pytest.mark.parametrize("thinking", [False, True])
def test_first_prompt_matches_the_official_template_token_for_token(
    official_tokenizer: Any,
    thinking: bool,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=thinking)
    system, user = "Work in the task workspace.", "检查雪和空白。\n"
    expected = official_tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    assert encoding.prompt(system, user) == expected


def test_missing_cached_template_is_fetched_from_the_mirror(
    tmp_path: Path,
    official_tokenizer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "partial-tokenizer"
    official_tokenizer.save_pretrained(partial)
    (partial / "chat_template.jinja").unlink(missing_ok=True)
    config_file = partial / "tokenizer_config.json"
    config = json.loads(config_file.read_text())
    config.pop("chat_template", None)
    config_file.write_text(json.dumps(config))

    def hub_download(
        repo_id: str,
        *,
        local_files_only: bool = False,
        **options: Any,
    ) -> str:
        if local_files_only:
            return str(partial)
        assert options["endpoint"] == "https://hf-mirror.com"
        assert options["revision"] == REVISION
        assert "*.safetensors" not in options["allow_patterns"]
        return str(official_tokenizer.name_or_path)

    # Replace the external Hub download boundary; tokenization remains real.
    monkeypatch.setattr("bashlet.encoding.snapshot_download", hub_download)
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    expected = official_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Work."},
            {"role": "user", "content": "Submit."},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert encoding.prompt("Work.", "Submit.") == expected


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("closed", [False, True])
def test_continuation_matches_a_new_official_turn_without_reencoding_history(
    official_tokenizer: Any,
    thinking: bool,
    closed: bool,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=thinking)
    generated = official_tokenizer.encode(
        "```bash\ntrue\n```" + ("<|im_end|>" if closed else ""),
        add_special_tokens=False,
    )
    original = list(generated)
    observation = "<returncode>0</returncode>\n<output>\n雪\n</output>"
    expected = official_tokenizer.encode(
        "\n" if closed else "<|im_end|>\n",
        add_special_tokens=False,
    ) + official_tokenizer.apply_chat_template(
        [{"role": "user", "content": observation}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    assert encoding.continuation(generated, observation) == expected
    assert generated == original


@pytest.mark.parametrize("opening", ["<think>\n", ""])
@pytest.mark.parametrize("explanation", ["", "Run the action.\n"])
def test_harness_executes_only_the_body_after_thinking_and_keeps_raw_tokens(
    tmp_path: Path,
    opening: str,
    explanation: str,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=True)
    considered = (
        opening + "Considering commands, including a quoted </think>.\n```bash\n"
        "touch MUST_NOT_RUN; printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n"
        "```\n</think>\n"
    )
    action = explanation + "```bash\nprintf 'action ran\\n'; touch DONE\n```"
    result = run_experiment(
        task=Task(
            "encoding/thinking",
            "Create DONE and submit.",
            {},
            CommandGrader("test -f DONE", {}),
        ),
        model=ReplayModel(
            [
                considered + action + "<|im_end|>",
                "<think>Submit now.</think>\n```bash\n"
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```<|im_end|>",
            ],
            encoding=encoding,
        ),
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    assert len(result.trajectory) == 2
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    step = result.trajectory[0]
    assert step.observation is not None and step.observation.output == "action ran\n"
    call = step.model_call
    assert call is not None and call.generation.source == "replayed"
    assert result.context_tokens[
        call.context_offset : call.context_offset + len(call.generation.tokens)
    ] == list(call.generation.tokens)
    assert "Considering commands," in encoding.decode(result.context_tokens)


@pytest.mark.parametrize("max_tokens", [8, 1024])
def test_unfinished_thinking_never_executes_a_complete_considered_bash_block(
    tmp_path: Path,
    max_tokens: int,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=True)
    result = run_experiment(
        task=Task("encoding/unfinished", "Submit.", {}, CommandGrader("true", {})),
        model=ReplayModel(["```bash\ntouch MUST_NOT_RUN\n```"], encoding=encoding),
        output_dir=tmp_path,
        overrides=[
            "budget.max_consecutive_invalid=1",
            f"model.sampling.max_tokens={max_tokens}",
        ],
    )

    assert result.episode_status == "format_error"
    step = result.trajectory[0]
    assert isinstance(step.action, InvalidAction)
    assert step.observation is None
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    assert step.model_call is not None
    assert step.model_call.generation.stop_reason == (
        "length" if max_tokens == 8 else "stop"
    )


def test_template_prefilled_thinking_start_is_not_required_in_the_generation(
    tmp_path: Path,
    official_tokenizer: Any,
) -> None:
    # A real tokenizer with the prefill used by thinking-only Qwen templates.
    # Qwen3.5's own assets will be checked separately by task 04 (H-11).
    prefilled = AutoTokenizer.from_pretrained(
        official_tokenizer.name_or_path,
        local_files_only=True,
    )
    prefilled.chat_template += (
        "{%- if add_generation_prompt and enable_thinking %}"
        "{{- '<think>\\n' }}{%- endif %}"
    )
    location = tmp_path / "tokenizer"
    prefilled.save_pretrained(location)
    encoding = QwenChatMLEncoding(
        str(location), revision="prefill-fixture", thinking=True
    )
    instruction = "Submit the workspace."
    result = run_experiment(
        task=Task("encoding/prefilled", instruction, {}, CommandGrader("true", {})),
        model=ReplayModel(
            [
                "I considered ```bash but will submit.</think>\n```bash\n"
                "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
            ],
            encoding=encoding,
        ),
        output_dir=tmp_path / "runs",
    )

    assert result.passed, result.reason
    call = result.trajectory[0].model_call
    assert call is not None and not call.text.startswith("<think>")
    record = json.loads((result.run_dir / "run.json").read_text())
    expected = prefilled.apply_chat_template(
        [
            {"role": "system", "content": record["protocol"]["system_prompt"]},
            {"role": "user", "content": instruction},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    assert encoding.decode(expected).endswith("<think>\n")
    assert result.context_tokens[: call.context_offset] == expected


def test_run_records_the_tokenizer_revision_asset_hashes_and_stop_markers(
    tmp_path: Path,
    official_tokenizer: Any,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    result = run_experiment(
        task=Task("encoding/identity", "Submit.", {}, CommandGrader("true", {})),
        model=ReplayModel(
            [
                "```bash\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
            ],
            encoding=encoding,
        ),
        output_dir=tmp_path,
    )
    record = json.loads((result.run_dir / "run.json").read_text())
    details = record["encoding_details"]
    assert details["tokenizer"] == QWEN and details["revision"] == REVISION
    tokenizer_file = Path(official_tokenizer.name_or_path) / "tokenizer.json"
    assert (
        details["files_sha256"]["tokenizer.json"]
        == hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    )
    assert details["stop_tokens"] == [151645, 151643]
    assert details["thinking_start_token"] == 151667
    assert details["thinking_end_token"] == 151668
    assert details["transformers_version"] == "4.57.3"
    assert details["tokenizers_version"] == "0.22.1"
    assert record["encoding"] == encoding.identity


class TokenModel:
    """A sampling boundary that supplies a noncanonical token segmentation."""

    identity = "test/qwen-sampled-tokens"

    def __init__(
        self, encoding: QwenChatMLEncoding, generations: list[Generation]
    ) -> None:
        self.encoding = encoding
        self._generations = iter(generations)
        self.prompts: list[tuple[int, ...]] = []

    def sample(self, prompt: Sequence[int], sampling: SamplingParams) -> Generation:
        self.prompts.append(tuple(prompt))
        return next(self._generations)


def test_subsequent_qwen_prompts_preserve_sampled_tokens_without_retokenizing(
    tmp_path: Path,
    official_tokenizer: Any,
) -> None:
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=True)
    first = "Reasoning stays.</think>\n```bash\nprintf '雪\\n'\n```"
    # Encode characters independently so decode -> encode changes the IDs.
    reasoning, _, body = first.partition("</think>")
    raw = (
        tuple(
            token
            for char in reasoning
            for token in official_tokenizer.encode(
                char,
                add_special_tokens=False,
            )
        )
        + (151668,)
        + tuple(
            token
            for char in body
            for token in official_tokenizer.encode(
                char,
                add_special_tokens=False,
            )
        )
        + (151645,)
    )
    assert list(raw) != official_tokenizer.encode(
        first + "<|im_end|>", add_special_tokens=False
    )
    final = tuple(
        official_tokenizer.encode(
            "Done.</think>\n```bash\n"
            "printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n```",
            add_special_tokens=False,
        )
    )
    model = TokenModel(
        encoding,
        [
            Generation(raw, "stop", (-0.5,) * len(raw)),
            Generation(final, "stop", (-0.5,) * len(final)),
        ],
    )
    result = run_experiment(
        task=Task(
            "encoding/sampled", "Print snow and submit.", {}, CommandGrader("true", {})
        ),
        model=model,
        output_dir=tmp_path,
    )

    assert result.passed, result.reason
    assert (
        model.prompts[1][: len(model.prompts[0]) + len(raw)] == model.prompts[0] + raw
    )
    assert "Reasoning stays." in encoding.decode(model.prompts[1])
    assert result.trajectory[0].observation is not None
    assert result.trajectory[0].observation.output == "雪\n"


@pytest.mark.parametrize("thinking", [False, True])
def test_qwen35_official_template_and_native_thinking_in_an_episode(
    tmp_path: Path,
    thinking: bool,
) -> None:
    model = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    cache = "runs/tokenizers"
    snapshot = snapshot_download(
        model, revision=revision, cache_dir=cache, local_files_only=True
    )
    official = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    encoding = QwenChatMLEncoding(
        model, revision=revision, cache_dir=cache, thinking=thinking
    )
    reasoning = (
        "Do not execute ```bash\ntouch MUST_NOT_RUN\n```.\n</think>\n"
        if thinking
        else ""
    )
    instruction = "创建 DONE，打印雪，再提交。\n"
    result = run_experiment(
        task=Task(
            "qwen35/template", instruction, {}, CommandGrader("test -f DONE", {})
        ),
        model=ReplayModel(
            [
                reasoning + "```bash\ntouch DONE; printf '雪\\n'\n```<|im_end|>",
                ("Submit now.</think>\n" if thinking else "")
                + "```bash\nprintf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'\n"
                "```<|im_end|>",
            ],
            encoding=encoding,
        ),
        output_dir=tmp_path,
    )
    assert result.passed, result.reason
    record = json.loads((result.run_dir / "run.json").read_text())
    expected = official.apply_chat_template(
        [
            {"role": "system", "content": record["protocol"]["system_prompt"]},
            {"role": "user", "content": instruction},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    first = result.trajectory[0].model_call
    assert first is not None
    assert result.context_tokens[: first.context_offset] == expected
    assert official.decode(expected).endswith(
        "<think>\n" if thinking else "<think>\n\n</think>\n\n"
    )
    assert not (result.workspace / "MUST_NOT_RUN").exists()
    assert result.trajectory[0].observation is not None
    assert result.trajectory[0].observation.output == "雪\n"
    assert record["encoding_details"]["revision"] == revision
    if thinking:
        assert reasoning.strip() in encoding.decode(result.context_tokens)
