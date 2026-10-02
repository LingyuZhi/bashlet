"""Executed demonstrations at the experiment and CLI seams."""

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
import pytrio as trio
import yaml
from test_cli import cli, saved_run
from test_encoding import QWEN, REVISION, official_tokenizer
from test_mbpp import double_row, prepare

from bashlet import (
    FunctionGrader,
    GradingCase,
    QwenChatMLEncoding,
    ReplayModel,
    Task,
    TaskOrigin,
    run_experiment,
)
from bashlet.config import load_config
from bashlet.demonstrations import reference_demonstration

# Register the existing official-tokenizer fixture in this test module.
__all__ = ["official_tokenizer"]


class TrainingAdapter:
    """Record the actual training inputs without making a paid request."""

    def __init__(self, encoding: str) -> None:
        self.encoding = encoding
        self.requests: list[tuple[trio.Datum, dict[str, Any]]] = []

    def submit(self, datum: trio.Datum, *, metadata: dict[str, Any]) -> None:
        self.requests.append((datum, metadata))


def training_task() -> Task:
    return Task(
        task_id="mbpp/602",
        prompt="Implement double(n) in solution.py; run python3 feedback_tests.py.",
        initial_files={
            "feedback_tests.py": (
                "from solution import double\nassert double(2) == 4\n"
                "print('observed: 4')\n"
            )
        },
        grader=FunctionGrader("solution.py", "double", (GradingCase((-37,), -74),)),
        reference_implementation="def double(n):\n    return n * 2\n",
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback_tests.py",
        origin=TaskOrigin(
            "google-research/mbpp", "a" * 40, "b" * 64, "602", "mbpp/602", "train"
        ),
    )


@pytest.mark.parametrize("thinking", [False, True])
def test_executed_reference_reaches_training_with_only_generated_targets(
    tmp_path: Path, official_tokenizer: Any, thinking: bool
) -> None:
    task = training_task()
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION, thinking=thinking)
    adapter = TrainingAdapter(encoding.identity)
    config = load_config(overrides=[f"model.thinking={str(thinking).lower()}"])
    result = run_experiment(
        task=task,
        model=reference_demonstration(task=task, encoding=encoding, config=config),
        config=config,
        output_dir=tmp_path,
        sft_adapter=adapter,
    )

    assert result.passed, result.reason
    assert len(adapter.requests) == 1
    datum, metadata = adapter.requests[0]
    inputs = datum.model_input.to_ints()
    targets = datum.loss_fn_inputs["target_tokens"].tolist()
    weights = datum.loss_fn_inputs["weights"].tolist()
    assert inputs == result.context_tokens[:-1]
    assert targets == result.context_tokens[1:]
    assert len(weights) == len(inputs) == len(targets)
    first = result.trajectory[0].model_call
    assert first is not None
    prompt = official_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": config.protocol.system_prompt},
            {"role": "user", "content": task.prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    assert inputs[: first.context_offset] == prompt
    assert weights[: first.context_offset - 1] == [0.0] * (len(prompt) - 1)
    for index, step in enumerate(result.trajectory):
        call = step.model_call
        assert call is not None
        assert call.generation.source == "replayed"
        assert call.generation.logprobs is None
        start = call.context_offset - 1
        end = start + len(call.generation.tokens)
        assert weights[start:end] == [1.0] * len(call.generation.tokens)
        assert targets[start:end] == list(call.generation.tokens)
        assert targets[end - 1] == 151645  # Qwen im_end is a supervised stop.
        if index < 2:
            next_call = result.trajectory[index + 1].model_call
            assert next_call is not None
            stop = next_call.context_offset - 1
            assert weights[end:stop] == [0.0] * (stop - end)
    assert targets[-1] == 151645 and weights[-1] == 1.0
    assert result.trajectory[0].observation is not None
    assert result.trajectory[0].observation.output == ""
    observed = result.trajectory[1].observation
    assert observed is not None and observed.output == "observed: 4\n"
    assert "<output>\nobserved: 4\n</output>" in official_tokenizer.decode(inputs)
    assert "-37" not in official_tokenizer.decode(inputs)
    origin = task.origin
    assert origin is not None
    assert metadata["origin"] == asdict(origin)
    assert metadata["source"] == "reference-implementation-v1"
    assert metadata["on_policy_eligible"] is False
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["protocol"]["name"] == "bash-block-v3"
    assert record["demonstration"]["status"] == "accepted"
    assert (result.run_dir / "submission/solution.py").read_text() == (
        task.reference_implementation
    )


def test_cli_exports_only_verified_training_families_and_replays_in_a_fresh_workspace(
    tmp_path: Path,
) -> None:
    prepared = prepare(tmp_path, [double_row(602), double_row(511), double_row(11)])
    config = tmp_path / "sft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "ReplayModel",
                    "tokenizer": QWEN,
                    "tokenizer_revision": REVISION,
                    "thinking": False,
                }
            }
        )
    )
    output = tmp_path / "demonstrations"
    process = cli(
        "prepare-sft",
        str(prepared / "manifest.json"),
        "-c",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {"accepted": 1, "rejected": 0}
    assert [entry["task_id"] for entry in manifest["tasks"]] == ["mbpp/602"]
    samples = [
        json.loads(line)
        for line in (output / "examples.jsonl").read_text().splitlines()
    ]
    assert len(samples) == 1
    sample = samples[0]
    assert sample["metadata"]["origin"]["split"] == "train"
    assert sample["metadata"]["origin"]["family_id"] == "mbpp/602"
    assert sample["metadata"]["encoding"] == manifest["encoding"]
    datum = trio.Datum.model_validate(sample["datum"])
    assert datum.model_input.to_ints()
    run_dir, record = saved_run(output / "episodes")
    (run_dir / "workspace/not_initial.txt").write_text("must not leak into replay")
    replayed = cli(
        "task",
        str(run_dir / "private/replay.json"),
        "-c",
        str(config),
        "--output-dir",
        str(tmp_path / "replayed"),
        cwd=tmp_path,
        timeout=60,
    )
    assert replayed.returncode == 0, replayed.stdout + replayed.stderr
    replay_dir, replay_record = saved_run(tmp_path / "replayed")
    assert replay_record["grade"] == record["grade"]
    assert replay_record["context_tokens"] == record["context_tokens"]
    assert replay_record["submission"]["sha256"] == record["submission"]["sha256"]
    assert not (replay_dir / "workspace/not_initial.txt").exists()


def test_failed_adapter_is_recorded_without_claiming_the_export_succeeded(
    tmp_path: Path,
) -> None:
    class UnavailableTraining(TrainingAdapter):
        def submit(self, datum: trio.Datum, *, metadata: dict[str, Any]) -> None:
            raise OSError("training consumer unavailable")

    task = training_task()
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    config = load_config()
    with pytest.raises(OSError, match="consumer unavailable"):
        run_experiment(
            task=task,
            model=reference_demonstration(task=task, encoding=encoding, config=config),
            config=config,
            output_dir=tmp_path,
            sft_adapter=UnavailableTraining(encoding.identity),
        )
    run_dir, record = saved_run(tmp_path)
    assert record["passed"] is True
    assert record["demonstration"]["status"] == "export_error"
    assert "consumer unavailable" in record["demonstration"]["reason"]
    assert (run_dir / "private/sft.json").is_file()


@pytest.mark.parametrize("failure", ["feedback", "grading", "budget", "truncated"])
def test_failed_demonstrations_never_reach_training(
    tmp_path: Path, failure: str
) -> None:
    task = training_task()
    overrides: list[str] = []
    if failure == "feedback":
        task = replace(
            task, initial_files={"feedback_tests.py": "raise AssertionError\n"}
        )
    elif failure == "grading":
        task = replace(task, reference_implementation="def double(n): return 4\n")
    elif failure == "budget":
        overrides = ["budget.max_steps=2"]
    else:
        overrides = ["model.sampling.max_tokens=3", "budget.max_consecutive_invalid=1"]
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    config = load_config(overrides=overrides)
    adapter = TrainingAdapter(encoding.identity)
    result = run_experiment(
        task=task,
        model=reference_demonstration(task=task, encoding=encoding, config=config),
        config=config,
        output_dir=tmp_path,
        sft_adapter=adapter,
    )
    assert adapter.requests == []
    assert result.demonstration is not None
    assert result.demonstration["status"] == "rejected"
    assert result.demonstration["reason"]
    assert not (result.run_dir / "private/sft.json").exists()
    if failure == "feedback":
        assert result.passed  # Grading success cannot conceal failed feedback.
    elif failure == "grading":
        observation = result.trajectory[1].observation
        assert observation is not None and observation.exit_code == 0
        assert not result.passed


@pytest.mark.parametrize(
    "invalid", ["missing_origin", "dev", "test", "forged_train", "utf8", "encoding"]
)
def test_invalid_origin_or_encoding_is_refused_before_execution(
    tmp_path: Path, invalid: str
) -> None:
    task = training_task()
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    model = reference_demonstration(task=task, encoding=encoding, config=load_config())
    adapter = TrainingAdapter(encoding.identity)
    origin = task.origin
    assert origin is not None
    if invalid == "missing_origin":
        task = replace(task, origin=None)
    elif invalid in ("dev", "test", "forged_train"):
        task = replace(
            task,
            origin=replace(
                origin,
                original_task_id="511" if invalid == "dev" else "11",
                family_id="mbpp/511" if invalid == "dev" else "mbpp/11",
                split="dev"
                if invalid == "dev"
                else "test"
                if invalid == "test"
                else "train",
            ),
        )
    elif invalid == "utf8":
        model = ReplayModel([])
    else:
        adapter.encoding = "a different tokenizer or template"
    with pytest.raises(ValueError):
        run_experiment(task=task, model=model, output_dir=tmp_path, sft_adapter=adapter)
    assert adapter.requests == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["budget", "infrastructure", "test_split"])
def test_cli_preserves_rejections_and_incomplete_runs(
    tmp_path: Path, failure: str
) -> None:
    prepared = prepare(tmp_path, [double_row(602), double_row(11)])
    config = tmp_path / "sft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "ReplayModel",
                    "tokenizer": QWEN,
                    "tokenizer_revision": REVISION,
                    "thinking": False,
                }
            }
        )
    )
    output = tmp_path / "demonstrations"
    options = {
        "budget": ["--set", "budget.max_steps=2"],
        "infrastructure": ["--set", "environment.bwrap=/missing-bwrap"],
        "test_split": ["--task-id", "mbpp/11"],
    }[failure]
    process = cli(
        "prepare-sft",
        str(prepared / "manifest.json"),
        "-c",
        str(config),
        "--output-dir",
        str(output),
        *options,
        cwd=tmp_path,
        timeout=60,
    )
    if failure == "test_split":
        assert process.returncode == 2
        assert "belongs to split test, not train" in process.stderr
        assert not output.exists()
    else:
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["counts"] == {"accepted": 0, "rejected": 1}
        assert (output / "examples.jsonl").read_text() == ""
        assert manifest["tasks"][0]["reason"]
        if failure == "budget":
            assert process.returncode == 0, process.stderr
            assert manifest["status"] == "complete"
        else:
            assert process.returncode == 2
            assert manifest["status"] == "incomplete"


def test_qwen35_prefilled_thinking_is_context_and_generated_end_is_supervised(
    tmp_path: Path,
) -> None:
    encoding = QwenChatMLEncoding(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cache_dir="runs/tokenizers",
        thinking=True,
    )
    task = training_task()
    config = load_config(overrides=["model.thinking=true"])
    adapter = TrainingAdapter(encoding.identity)
    result = run_experiment(
        task=task,
        model=reference_demonstration(task=task, encoding=encoding, config=config),
        config=config,
        output_dir=tmp_path,
        sft_adapter=adapter,
    )
    assert result.passed, result.reason
    datum, _ = adapter.requests[0]
    targets = datum.loss_fn_inputs["target_tokens"].tolist()
    weights = datum.loss_fn_inputs["weights"].tolist()
    for step in result.trajectory:
        call = step.model_call
        assert call is not None and call.reply is not None
        assert call.reply.thinking == ""
        assert call.generation.tokens[0] == encoding.thinking_end_token
        assert encoding.thinking_start_token not in call.generation.tokens
        assert weights[call.context_offset - 1] == 1.0
        assert targets[call.context_offset - 1] == encoding.thinking_end_token
    assert all(
        weight == 0.0
        for token, weight in zip(targets, weights, strict=True)
        if token == encoding.thinking_start_token
    )
