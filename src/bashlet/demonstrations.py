"""Reference-constructed, executed SFT data; never on-policy sampling evidence."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .config import ExperimentConfig, config_record, load_model
from .encoding import QwenChatMLEncoding
from .models import ExperimentResult, Model, Task
from .protocol import fingerprints
from .replay import ReplayModel
from .tasks import task_path, validate_origin
from .workspace import validate_workspace_path

if TYPE_CHECKING:
    import pytrio as trio

SOURCE = "reference-implementation-v1"


class SFTAdapter(Protocol):
    """Consumer of validated, manually shifted cross-entropy training inputs.

    A production training Adapter must apply the shared spend guard. This slice
    supplies a file consumer and tests the contract without remote training.
    """

    @property
    def encoding(self) -> str: ...

    def submit(self, datum: trio.Datum, *, metadata: dict[str, Any]) -> None: ...


def _validate_reference(task: Task) -> None:
    if task.origin is None or task.origin.split != "train":
        raise ValueError("SFT demonstrations require a training-split origin")
    validate_origin(task.origin)
    if not task.reference_implementation.strip():
        raise ValueError("A reference implementation is required")
    if len(task.implementation_paths) != 1:
        raise ValueError("Reference demonstrations require one implementation path")
    validate_workspace_path(task.implementation_paths[0])
    if not task.feedback_command or not task.feedback_command.strip():
        raise ValueError("Reference demonstrations require a feedback command")


def _reference_replies(
    task: Task, encoding: QwenChatMLEncoding, config: ExperimentConfig
) -> list[str]:
    _validate_reference(task)
    # Python string literals keep newlines, quotes and fenced text out of the
    # surrounding bash syntax while writing the exact reference bytes.
    script = (
        f"from pathlib import Path; p = Path({task.implementation_paths[0]!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text({task.reference_implementation!r}, encoding='utf-8')"
    )
    commands = [
        "python3 -c " + shlex.quote(script),
        str(task.feedback_command),
        "printf '%s\\n' " + shlex.quote(config.protocol.submission_marker),
    ]
    return [encoding.replay_reply(f"```bash\n{command}\n```") for command in commands]


def reference_demonstration(
    *, task: Task, encoding: QwenChatMLEncoding, config: ExperimentConfig
) -> ReplayModel:
    """Construct three replies; all observations still come from the Harness."""
    return ReplayModel(
        _reference_replies(task, encoding, config),
        encoding=encoding,
        identity=f"replay/{SOURCE}",
    )


def validate_sft_request(task: Task, model: Model, adapter: SFTAdapter) -> None:
    """Reject invalid provenance or encoding before executing any task code."""
    _validate_reference(task)
    if not isinstance(model.encoding, QwenChatMLEncoding):
        raise ValueError("SFT requires the student's Qwen ChatML encoding")
    if adapter.encoding != model.encoding.identity:
        raise ValueError("SFT Adapter encoding does not match the demonstration")


def export_demonstration(
    result: ExperimentResult,
    task: Task,
    model: Model,
    config: ExperimentConfig,
    adapter: SFTAdapter,
) -> None:
    """Accept only executed reference demonstrations and preserve raw token IDs."""
    result.demonstration = {"source": SOURCE, "status": "rejected", "reason": ""}
    if result.status != "completed" or not result.passed:
        result.demonstration["reason"] = "submission_did_not_pass"
        return
    assert isinstance(model.encoding, QwenChatMLEncoding)
    replies = _reference_replies(task, model.encoding, config)
    steps = result.trajectory
    if len(steps) != 3 or any(
        step.model_call is None
        or step.model_call.text != reply
        or step.model_call.generation.source != "replayed"
        or step.model_call.generation.stop_reason != "stop"
        or step.observation is None
        or step.observation.status != "ok"
        for step, reply in zip(steps, replies, strict=False)
    ):
        result.demonstration["reason"] = "reference_sequence_or_execution_failed"
        return
    if not steps[1].verification:
        result.demonstration["reason"] = "feedback_verification_failed"
        return
    tokens = result.context_tokens
    weights = [0.0] * len(tokens)
    spans: list[dict[str, int]] = []
    previous_end = 0
    for step in steps:
        call = step.model_call
        assert call is not None
        start, generated = call.context_offset, call.generation.tokens
        end = start + len(generated)
        if start <= previous_end or tokens[start:end] != list(generated):
            raise ValueError("Demonstration token offsets do not match raw generations")
        weights[start:end] = [1.0] * len(generated)
        spans.append({"start": start, "end": end})
        previous_end = end
    import pytrio as trio

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": weights[1:],
        },
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source": SOURCE,
        "on_policy_eligible": False,
        "task_id": task.task_id,
        "task_sha256": result.task_sha256,
        "origin": asdict(task.origin) if task.origin else None,
        "task_type": task.task_type,
        "variant_id": task.variant_id,
        "feedback_start_id": task.feedback_start_id,
        "environment_variant_id": task.environment_variant_id,
        "reference_sha256": hashlib.sha256(
            task.reference_implementation.encode()
        ).hexdigest(),
        "run_dir": str(result.run_dir),
        "encoding": result.encoding,
        "encoding_details": result.encoding_details,
        "fingerprints": fingerprints(config),
        "protocol": config.protocol.name,
        "grade": asdict(result.grade) if result.grade else None,
        "submission": asdict(result.submission) if result.submission else None,
        "supervised_spans": spans,
        "context_tokens": len(tokens),
        "supervised_tokens": int(sum(weights)),
        "loss_fn": "cross_entropy",
        "auto_shift": False,
    }
    try:
        artifact = result.run_dir / "private/sft.json"
        artifact.write_text(
            json.dumps({"metadata": metadata, "datum": datum.model_dump()})
        )
        (result.run_dir / "private/replay.json").write_text(
            json.dumps({"task": asdict(task), "replies": replies}, ensure_ascii=True)
        )
        result.demonstration["artifact"] = "private/sft.json"
        adapter.submit(datum, metadata=metadata)
    except (Exception, KeyboardInterrupt) as error:
        result.demonstration.update(status="export_error", reason=str(error))
        raise
    result.demonstration["status"] = "accepted"


class _DatasetWriter:
    """The offline Adapter: persist exactly the Datum a trainer would consume."""

    def __init__(self, path: Path, encoding: str) -> None:
        self.path, self.encoding = path, encoding
        path.touch(exist_ok=False)

    def submit(self, datum: trio.Datum, *, metadata: dict[str, Any]) -> None:
        with self.path.open("a") as stream:
            stream.write(
                json.dumps({"metadata": metadata, "datum": datum.model_dump()})
            )
            stream.write("\n")


def prepare_sft(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    task_ids: Sequence[str] = (),
) -> Path:
    """Execute verified training references into a new, auditable SFT dataset.

    Expected demonstration failures remain in the manifest but never in the
    training file. Infrastructure errors or interrupts leave it incomplete.
    """
    from .experiment import run_experiment, task_from_record

    if config.model.class_name != "ReplayModel" or "replies" in config.model.parameters:
        raise ValueError("prepare-sft requires ReplayModel without supplied replies")
    source_bytes = manifest_path.read_bytes()
    source = json.loads(source_bytes)
    selected = (
        list(task_ids)
        if task_ids
        else [
            entry["task_id"]
            for entry in source["tasks"]
            if entry["split"] == "train" and entry["status"] == "verified"
        ]
    )
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("Select a non-empty list of distinct training task IDs")
    tasks = [
        task_from_record(
            json.loads(task_path(manifest_path, task_id, "train").read_text())
        )
        for task_id in selected
    ]
    for task in tasks:
        _validate_reference(task)
    encoding = load_model(config.model).encoding
    if not isinstance(encoding, QwenChatMLEncoding):
        raise ValueError("prepare-sft requires the student's Qwen ChatML tokenizer")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    writer = _DatasetWriter(output_dir / "examples.jsonl", encoding.identity)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "source": SOURCE,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "task_ids": selected,
        "encoding": encoding.identity,
        "encoding_details": encoding.describe(),
        "config": config_record(config),
        "fingerprints": fingerprints(config),
        "acceptance_rule": (
            "three successful reference actions, feedback and independent grading"
        ),
        "on_policy_eligible": False,
        "loss_fn": "cross_entropy",
        "auto_shift": False,
        "examples": "examples.jsonl",
        "tasks": [],
    }
    destination = output_dir / "manifest.json"

    def save() -> None:
        manifest["counts"] = {
            status: sum(entry["status"] == status for entry in manifest["tasks"])
            for status in ("accepted", "rejected")
        }
        manifest["examples_sha256"] = hashlib.sha256(
            writer.path.read_bytes()
        ).hexdigest()
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        temporary.replace(destination)

    save()
    try:
        for task in tasks:
            result = run_experiment(
                task=task,
                model=reference_demonstration(
                    task=task, encoding=encoding, config=config
                ),
                config=config,
                output_dir=output_dir / "episodes",
                sft_adapter=writer,
            )
            assert result.demonstration is not None
            manifest["tasks"].append(
                {
                    "task_id": task.task_id,
                    "task_sha256": result.task_sha256,
                    "origin": asdict(task.origin) if task.origin else None,
                    "run_dir": str(result.run_dir.relative_to(output_dir)),
                    "run_sha256": hashlib.sha256(
                        (result.run_dir / "run.json").read_bytes()
                    ).hexdigest(),
                    **result.demonstration,
                }
            )
            save()
            if result.status in ("infrastructure_error", "interrupted"):
                raise ValueError(f"Demonstration execution incomplete: {result.reason}")
        manifest["status"] = "complete"
    finally:
        if manifest["status"] != "complete":
            manifest["status"] = "incomplete"
        save()
    return destination
