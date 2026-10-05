"""Executed, auditable SFT data; never current-student on-policy evidence."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from . import protocol
from .config import ExperimentConfig, ProtocolConfig, config_record, load_model
from .encoding import QwenChatMLEncoding
from .models import (
    Encoding,
    ExperimentResult,
    Generation,
    InvalidAction,
    Model,
    ModelResponseError,
    Observation,
    SamplingParams,
    Submission,
    Task,
    WorkspaceEntry,
)
from .protocol import fingerprints
from .replay import RecordedTrajectoryModel, ReplayModel
from .tasks import task_path, validate_harness_origin, validate_origin
from .workspace import validate_workspace_path, verify_submission

if TYPE_CHECKING:
    import pytrio as trio

SOURCE = "reference-implementation-v1"
EXECUTED_SOURCE = "executed-trajectory-v1"


@dataclass(frozen=True)
class ActionSupervision:
    """An explicit quality decision, independent of exit code and final grade."""

    supervise: bool
    reason: str
    author: Literal["model", "codex"] = "model"

    def __post_init__(self) -> None:
        if (
            type(self.supervise) is not bool
            or not isinstance(self.reason, str)
            or not self.reason.strip()
            or self.author not in ("model", "codex")
        ):
            raise ValueError("Supervision requires a boolean, reason and known author")


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


def reference_replies(
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
        reference_replies(task, encoding, config),
        encoding=encoding,
        identity=f"replay/{SOURCE}",
    )


def validate_sft_request(
    task: Task, model: Model, adapter: SFTAdapter, *, executed: bool = False
) -> None:
    """Reject invalid provenance or encoding before executing any task code."""
    if executed:
        if task.origin is None or task.origin.split != "train":
            raise ValueError(
                "Executed SFT trajectories require a training-split origin"
            )
        validate_harness_origin(task.origin)
    else:
        _validate_reference(task)
    if not isinstance(model.encoding, QwenChatMLEncoding):
        raise ValueError("SFT requires the student's Qwen ChatML encoding")
    if adapter.encoding != model.encoding.identity:
        raise ValueError("SFT Adapter encoding does not match the demonstration")


def validate_history(
    record: dict[str, Any],
    task: Task,
    encoding: Encoding,
    config: ProtocolConfig,
) -> None:
    """Reconstruct only framing; sampled generations remain their raw token IDs."""
    if (
        record["encoding"] != encoding.identity
        or record["encoding_details"] != encoding.describe()
    ):
        raise ValueError("Trajectory encoding identity differs")
    context = encoding.prompt(config.system_prompt, task.prompt)
    for index, step in enumerate(record["trajectory"]):
        call = step["model_call"]
        if call is None or call["context_offset"] != len(context):
            raise ValueError("Trajectory has missing calls or invalid context offsets")
        raw = call["generation"]
        generation = Generation(
            tokens=tuple(raw["tokens"]),
            stop_reason=raw["stop_reason"],
            logprobs=tuple(raw["logprobs"]) if raw["logprobs"] is not None else None,
            request_id=raw["request_id"],
            source=raw["source"],
        )
        try:
            generation.validate(SamplingParams(**call["sampling"]))
        except ModelResponseError as error:
            raise ValueError(
                f"Invalid trajectory sampling evidence: {error}"
            ) from error
        reply = encoding.reply(generation.tokens)
        if call["text"] != encoding.decode(generation.tokens) or call[
            "reply"
        ] != asdict(reply):
            raise ValueError("Trajectory text differs from its raw generation")
        action = (
            protocol.parse_action(reply.action_text, generation.stop_reason == "length")
            if reply.thinking_complete
            else InvalidAction("Native thinking did not reach its end marker")
        )
        if step["action"] != asdict(action):
            raise ValueError("Trajectory action differs from the protocol")
        context.extend(generation.tokens)
        feedback: str | None
        if isinstance(action, InvalidAction):
            if step["observation"] is not None:
                raise ValueError(
                    "An invalid action cannot have an executed observation"
                )
            feedback = protocol.render_invalid(action, config.invalid_action_format)
        else:
            if step["observation"] is None:
                raise ValueError("Executable action has no observation")
            observation = Observation(**step["observation"])
            submitted = protocol.submission_text(observation, config.submission_marker)
            if submitted is not None and index != len(record["trajectory"]) - 1:
                raise ValueError("Trajectory continued after an explicit submission")
            feedback = (
                None
                if submitted is not None
                else protocol.render_observation(observation, config)
            )
        if step["feedback"] != feedback:
            raise ValueError("Trajectory feedback differs from its observation")
        if feedback is not None:
            if not (
                index == len(record["trajectory"]) - 1
                and record["status"] not in ("completed", "submitted")
                and context == record["context_tokens"]
            ):
                context.extend(encoding.continuation(generation.tokens, feedback))
    if context != record["context_tokens"]:
        raise ValueError(
            "Trajectory context does not preserve the complete causal history"
        )


def _action_decisions(
    record: dict[str, Any],
    decisions: Sequence[ActionSupervision],
    original: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(decisions) != len(record["trajectory"]):
        raise ValueError("Every trajectory action needs a supervision decision")
    reviewed = []
    for index, (step, decision) in enumerate(
        zip(record["trajectory"], decisions, strict=True)
    ):
        call = step["model_call"]
        assert call is not None
        execution_source = call["generation"]["source"]
        if original is not None and index < len(original["trajectory"]):
            call = original["trajectory"][index]["model_call"]
        generation = call["generation"]
        if decision.author == "model":
            if generation["source"] != "sampled" or call["policy"].startswith(
                ("replay/", "codex/")
            ):
                raise ValueError(
                    "Model-authored supervision requires actual sampled provenance"
                )
        elif generation["source"] != "replayed" or not (
            call["policy"].startswith("codex/")
            or call["policy"] == RecordedTrajectoryModel.identity
        ):
            raise ValueError("Codex corrections require explicit replay provenance")
        if decision.supervise:
            if step["action"]["kind"] == "invalid":
                raise ValueError("Invalid protocol actions cannot receive supervision")
            if (
                not generation["tokens"]
                or generation["tokens"][-1]
                not in record["encoding_details"]["stop_tokens"]
            ):
                raise ValueError(
                    "Supervised actions require their generated stop token"
                )
            if index:
                previous = record["trajectory"][index - 1]
                if (
                    previous["action"] == step["action"]
                    and previous["workspace_manifest"] == step["workspace_manifest"]
                    and _observation_content(previous["observation"])
                    == _observation_content(step["observation"])
                ):
                    raise ValueError(
                        "Meaningless repeated action cannot receive supervision"
                    )
            syntax = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-n"],
                input=step["action"]["command"],
                text=True,
                capture_output=True,
            )
            if syntax.returncode:
                raise ValueError(
                    "Non-executable bash syntax cannot receive supervision"
                )
        reviewed.append(
            asdict(decision)
            | {
                "policy": call["policy"],
                "generation": generation,
                "execution_source": execution_source,
                "start": call["context_offset"],
                "end": call["context_offset"] + len(generation["tokens"]),
            }
        )
    return reviewed


def _observation_content(value: dict[str, Any] | None) -> dict[str, Any] | None:
    return (
        {key: item for key, item in value.items() if key != "duration_seconds"}
        if value is not None
        else None
    )


def verify_reexecuted_prefix(record: dict[str, Any], original: dict[str, Any]) -> None:
    """Compare actual observations and workspace states, excluding wall-clock time."""
    if (
        original["isolation"]["passed"] is not True
        or record["initial_workspace_manifest"]
        != original["initial_workspace_manifest"]
        or len(record["trajectory"]) < len(original["trajectory"])
        or record["context_tokens"][: len(original["context_tokens"])]
        != original["context_tokens"]
    ):
        raise ValueError("Re-execution did not preserve the complete original prefix")
    for observed, prior in zip(
        record["trajectory"], original["trajectory"], strict=False
    ):
        for key in ("action", "feedback", "workspace_manifest"):
            if observed[key] != prior[key]:
                raise ValueError(
                    f"Re-executed {key} differs from the source trajectory"
                )
        current, earlier = observed["model_call"], prior["model_call"]
        if current["context_offset"] != earlier["context_offset"] or list(
            current["generation"]["tokens"]
        ) != list(earlier["generation"]["tokens"]):
            raise ValueError("Re-execution changed the original sampled tokens")

        if _observation_content(observed["observation"]) != _observation_content(
            prior["observation"]
        ):
            raise ValueError(
                "Re-executed observation differs from the source trajectory"
            )
    if original.get("submission") is not None and (
        record["submission"]["sha256"] != original["submission"]["sha256"]
        or protocol.fingerprint(record["grade"])
        != protocol.fingerprint(original["grade"])
    ):
        raise ValueError("Independent re-execution grading differs")


def export_demonstration(
    result: ExperimentResult,
    task: Task,
    model: Model,
    config: ExperimentConfig,
    adapter: SFTAdapter,
    *,
    supervision: Sequence[ActionSupervision] | None = None,
    max_context_tokens: int = 32768,
    demonstration_id: str | None = None,
    provenance: dict[str, Any] | None = None,
    complementary_reason: str = "",
) -> None:
    """Export executed reference or reviewed trajectories, preserving raw tokens."""
    source = SOURCE if supervision is None else EXECUTED_SOURCE
    result.demonstration = {"source": source, "status": "rejected", "reason": ""}
    try:
        validate_sft_request(task, model, adapter, executed=supervision is not None)
        from .experiment import serialize_task

        task_bytes = serialize_task(task).encode()
        if (
            hashlib.sha256(task_bytes).hexdigest() != result.task_sha256
            or (result.run_dir / "private/task.json").read_bytes() != task_bytes
            or task.task_id != result.task_id
        ):
            raise ValueError("Export task identity differs from its executed evidence")
        if (
            model.identity != result.policy
            or model.encoding.identity != result.encoding
        ):
            raise ValueError("Export model identity differs from its executed evidence")
        recorded = json.loads((result.run_dir / "run.json").read_text())
        if recorded["fingerprints"] != fingerprints(config):
            raise ValueError("Export condition differs from its executed evidence")
        evidence = json.loads(json.dumps(asdict(result), default=str))
        if (
            any(
                recorded.get(key) != value
                for key, value in evidence.items()
                if key != "demonstration"
            )
            or recorded.get("passed") != result.passed
        ):
            raise ValueError(
                "Export trajectory or grading differs from saved execution evidence"
            )
        if result.submission is not None:
            verify_submission(result.submission, result.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.demonstration["reason"] = str(error)
        return
    if (
        result.status != "completed"
        or not result.passed
        or result.episode_status != "submitted"
        or result.submission is None
        or not result.isolation.passed
    ):
        result.demonstration["reason"] = "submission_did_not_pass"
        return
    assert isinstance(model.encoding, QwenChatMLEncoding)
    replies: list[str] = []
    steps = result.trajectory
    decisions: list[dict[str, Any]] = []
    if supervision is not None:
        try:
            validate_history(asdict(result), task, model.encoding, config.protocol)
            original = (
                model.record if isinstance(model, RecordedTrajectoryModel) else None
            )
            if original is not None:
                verify_reexecuted_prefix(asdict(result), original)
            decisions = _action_decisions(asdict(result), supervision, original)
            if len(result.context_tokens) > max_context_tokens:
                raise ValueError("Complete causal history exceeds max_context_tokens")
        except ValueError as error:
            result.demonstration["reason"] = str(error)
            return
    else:
        replies = reference_replies(task, model.encoding, config)
    if supervision is None and (
        len(steps) != 3
        or any(
            step.model_call is None
            or step.model_call.text != reply
            or step.model_call.generation.source != "replayed"
            or step.model_call.generation.stop_reason != "stop"
            or step.observation is None
            or step.observation.status != "ok"
            for step, reply in zip(steps, replies, strict=False)
        )
    ):
        result.demonstration["reason"] = "reference_sequence_or_execution_failed"
        return
    if supervision is None and not steps[1].verification:
        result.demonstration["reason"] = "feedback_verification_failed"
        return
    tokens = result.context_tokens
    weights = [0.0] * len(tokens)
    spans: list[dict[str, int]] = []
    previous_end = 0
    for index, step in enumerate(steps):
        call = step.model_call
        assert call is not None
        start, generated = call.context_offset, call.generation.tokens
        end = start + len(generated)
        if start <= previous_end or tokens[start:end] != list(generated):
            raise ValueError("Demonstration token offsets do not match raw generations")
        if supervision is None or decisions[index]["supervise"]:
            weights[start:end] = [1.0] * len(generated)
            spans.append({"start": start, "end": end})
        previous_end = end
    if not sum(weights):
        result.demonstration["reason"] = "no_supervised_actions"
        return
    import pytrio as trio

    datum = trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            "target_tokens": tokens[1:],
            "weights": weights[1:],
        },
    )
    metadata: dict[str, Any] = {
        "schema_version": 1 if supervision is None else 2,
        "source": source,
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
    if supervision is not None:
        metadata.update(
            action_supervision=decisions,
            tokenizer_cache_dir=model.encoding.cache_dir,
            policy=result.policy,
            policy_details=result.policy_details,
            max_context_tokens=max_context_tokens,
            action_supervision_sha256=protocol.fingerprint(decisions),
        )
        if demonstration_id is not None:
            metadata.update(
                demonstration_id=demonstration_id,
                provenance=provenance or {},
                complementary_reason=complementary_reason,
            )
        if isinstance(model, RecordedTrajectoryModel):
            source_data = (result.run_dir / "private/source-run.json").read_bytes()
            if json.loads(source_data) != model.record:
                raise ValueError("Original source evidence differs from the replay")
            metadata["source_run"] = {
                "artifact": "private/source-run.json",
                "sha256": hashlib.sha256(source_data).hexdigest(),
                "prefix_steps": len(model.record["trajectory"]),
                "policy": model.record["policy"],
            }
    try:
        artifact = result.run_dir / "private/sft.json"
        artifact.write_text(
            json.dumps({"metadata": metadata, "datum": datum.model_dump()})
        )
        if supervision is None:
            (result.run_dir / "private/replay.json").write_text(
                json.dumps(
                    {"task": asdict(task), "replies": replies}, ensure_ascii=True
                )
            )
        result.demonstration["artifact"] = "private/sft.json"
        adapter.submit(datum, metadata=metadata)
    except (Exception, KeyboardInterrupt) as error:
        result.demonstration.update(status="export_error", reason=str(error))
        raise
    result.demonstration["status"] = "accepted"


def validate_executed_example(
    metadata: dict[str, Any],
    record: dict[str, Any],
    run_dir: Path,
) -> None:
    """Bind supervision to the complete executed history at the loading boundary."""
    from .experiment import task_from_record

    task_bytes = (run_dir / "private/task.json").read_bytes()
    task = task_from_record(json.loads(task_bytes))
    if task.origin is None:
        raise ValueError("Executed trajectory has no task origin")
    validate_harness_origin(task.origin)
    if (
        metadata["schema_version"] != 2
        or task.origin.split != "train"
        or hashlib.sha256(task_bytes).hexdigest() != metadata["task_sha256"]
        or record["task_sha256"] != metadata["task_sha256"]
        or metadata["origin"] != asdict(task.origin)
        or record["task_origin"] != asdict(task.origin)
        or record["status"] != "completed"
        or record["episode_status"] != "submitted"
        or record["isolation"]["passed"] is not True
        or record["grade"] != metadata["grade"]
        or record["submission"] != metadata["submission"]
        or record["submission"] is None
        or record["fingerprints"] != metadata["fingerprints"]
        or len(record["context_tokens"]) != metadata["context_tokens"]
        or metadata["context_tokens"] > metadata["max_context_tokens"]
    ):
        raise ValueError(
            "Executed trajectory task, grading or execution evidence differs"
        )
    details = metadata["encoding_details"]
    snapshot = dict(record["submission"])
    snapshot["manifest"] = {
        name: WorkspaceEntry(**entry) for name, entry in snapshot["manifest"].items()
    }
    verify_submission(Submission(**snapshot), run_dir)
    encoding = QwenChatMLEncoding(
        details["tokenizer"],
        revision=details["revision"],
        thinking=details["thinking"],
        cache_dir=metadata["tokenizer_cache_dir"],
    )
    validate_history(
        record, task, encoding, ProtocolConfig(**record["config"]["protocol"])
    )
    decisions = [
        ActionSupervision(entry["supervise"], entry["reason"], entry["author"])
        for entry in metadata["action_supervision"]
    ]
    original = None
    if "source_run" in metadata:
        source_data = (run_dir / "private/source-run.json").read_bytes()
        if hashlib.sha256(source_data).hexdigest() != metadata["source_run"]["sha256"]:
            raise ValueError("Original source trajectory hash differs")
        original = json.loads(source_data)
        validate_history(
            original, task, encoding, ProtocolConfig(**original["config"]["protocol"])
        )
        verify_reexecuted_prefix(record, original)
    expected = _action_decisions(record, decisions, original)
    if (
        expected != metadata["action_supervision"]
        or protocol.fingerprint(expected) != metadata["action_supervision_sha256"]
        or metadata["supervised_spans"]
        != [
            {"start": entry["start"], "end": entry["end"]}
            for entry in expected
            if entry["supervise"]
        ]
    ):
        raise ValueError("Action supervision differs from its executed raw generations")


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

    def save_manifest(
        self,
        manifest: dict[str, Any],
        destination: Path,
        *,
        final: bool = False,
    ) -> None:
        """Keep counts, content hashes and interrupted status in one lifecycle."""
        if final and manifest["status"] != "complete":
            manifest["status"] = "incomplete"
        manifest["counts"] = {
            status: sum(entry["status"] == status for entry in manifest["tasks"])
            for status in ("accepted", "rejected")
        }
        manifest["examples_sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n")
        temporary.replace(destination)


def prepare_sft(
    *,
    manifest_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    task_ids: Sequence[str] = (),
    trajectories_path: Path | None = None,
    max_context_tokens: int = 32768,
) -> Path:
    """Execute verified training references into a new, auditable SFT dataset.

    Expected demonstration failures remain in the manifest but never in the
    training file. Infrastructure errors or interrupts leave it incomplete.
    """
    from .experiment import run_experiment, task_from_record

    if trajectories_path is not None:
        if task_ids:
            raise ValueError(
                "Select trajectories in their review file, not with --task-id"
            )
        return _prepare_trajectories(
            manifest_path,
            trajectories_path,
            output_dir,
            config,
            max_context_tokens,
        )

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

    writer.save_manifest(manifest, destination)
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
            writer.save_manifest(manifest, destination)
            if result.status in ("infrastructure_error", "interrupted"):
                raise ValueError(f"Demonstration execution incomplete: {result.reason}")
        manifest["status"] = "complete"
    finally:
        writer.save_manifest(manifest, destination, final=True)
    return destination


def _prepare_trajectories(
    task_manifest: Path,
    review_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    max_context_tokens: int,
) -> Path:
    from .episode import save_record
    from .experiment import run_experiment, task_from_record

    if config.model.class_name != "ReplayModel" or "replies" in config.model.parameters:
        raise ValueError(
            "Trajectory export requires ReplayModel without supplied replies"
        )
    if type(max_context_tokens) is not int or max_context_tokens < 2:
        raise ValueError("max_context_tokens must be an integer of at least two")
    review_data = review_path.read_bytes()
    review = json.loads(review_data)
    if review["schema_version"] not in (1, 2) or not review["trajectories"]:
        raise ValueError("Select a non-empty versioned trajectory review")
    identified = review["schema_version"] == 2
    if identified:
        identifiers = [item.get("demonstration_id") for item in review["trajectories"]]
        if any(
            not isinstance(value, str) or not value.strip() for value in identifiers
        ):
            raise ValueError("Every demonstration needs a non-empty identity")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate demonstration identity")
    encoding = load_model(config.model).encoding
    if not isinstance(encoding, QwenChatMLEncoding):
        raise ValueError("Trajectory export requires Qwen ChatML encoding")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    writer = _DatasetWriter(output_dir / "examples.jsonl", encoding.identity)
    manifest: dict[str, Any] = {
        "schema_version": 3 if identified else 2,
        "status": "preparing",
        "source": EXECUTED_SOURCE,
        "source_manifest": str(task_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            task_manifest.read_bytes()
        ).hexdigest(),
        "review_sha256": hashlib.sha256(review_data).hexdigest(),
        "tokenizer_cache_dir": encoding.cache_dir,
        "encoding": encoding.identity,
        "encoding_details": encoding.describe(),
        "config": config_record(config),
        "fingerprints": fingerprints(config),
        "max_context_tokens": max_context_tokens,
        "on_policy_eligible": False,
        "loss_fn": "cross_entropy",
        "auto_shift": False,
        "examples": "examples.jsonl",
        "tasks": [],
        "task_ids": [],
        "acceptance_rule": (
            "complete history, action review, actual replay and independent grading"
        ),
    }
    (output_dir / "review.json").write_bytes(review_data)
    destination = output_dir / "manifest.json"

    writer.save_manifest(manifest, destination)
    family_counts: dict[str, int] = {}
    behaviors: dict[str, set[str]] = {}
    try:
        for index, item in enumerate(review["trajectories"]):
            entry: dict[str, Any] = {
                "review_index": index,
                "task_id": item.get("task_id", "unknown"),
                "source_run_dir": item.get("run_dir"),
                "status": "rejected",
                "reason": "",
            }
            if identified:
                entry["demonstration_id"] = item["demonstration_id"]
            manifest["tasks"].append(entry)
            try:
                source_dir = (review_path.parent / item["run_dir"]).resolve()
                source_data = (source_dir / "run.json").read_bytes()
                if hashlib.sha256(source_data).hexdigest() != item["run_sha256"]:
                    raise ValueError("Source trajectory hash mismatch")
                original = json.loads(source_data)
                task_id = original["task_id"]
                entry["task_id"] = task_id
                if not identified and task_id in manifest["task_ids"]:
                    raise ValueError("Duplicate task in trajectory selection")
                if task_id not in manifest["task_ids"]:
                    manifest["task_ids"].append(task_id)
                task_data = task_path(task_manifest, task_id, "train").read_bytes()
                if hashlib.sha256(task_data).hexdigest() != original["task_sha256"]:
                    raise ValueError(
                        "Source trajectory differs from the frozen training task"
                    )
                task = task_from_record(json.loads(task_data))
                if task.origin is None or task.origin.split != "train":
                    raise ValueError("Source must identify a training task")
                family = task.origin.family_id
                if identified:
                    if family_counts.get(family, 0) >= 2:
                        raise ValueError(
                            "At most two demonstrations per training family"
                        )
                    if (
                        family_counts.get(family, 0)
                        and not item.get("complementary_reason", "").strip()
                    ):
                        raise ValueError(
                            "Second demonstration needs a complementary reason"
                        )
                    behavior = protocol.fingerprint(
                        [
                            step["action"]
                            for step, label in zip(
                                original["trajectory"], item["actions"], strict=True
                            )
                            if label["supervise"]
                        ]
                        + [
                            c["reply"]
                            for c in item.get("corrections", [])
                            if c["supervision"]["supervise"]
                        ]
                    )
                    if behavior in behaviors.get(family, set()):
                        raise ValueError("Meaningless repeated demonstration")
                validate_history(original, task, encoding, config.protocol)
                if any(
                    original["fingerprints"][key] != fingerprints(config)[key]
                    for key in ("protocol", "environment")
                ):
                    raise ValueError("Source protocol or environment differs")
                decisions = [
                    ActionSupervision(**decision) for decision in item["actions"]
                ]
                _action_decisions(original, decisions)
                corrections = item.get("corrections", [])
                if original.get("submission") is None and not corrections:
                    raise ValueError(
                        "Incomplete source trajectory has no corrections or submission"
                    )
                if corrections and original.get("submission") is not None:
                    raise ValueError("Corrections must start before submission")
                correction_replies = []
                for correction in corrections:
                    decision = ActionSupervision(**correction["supervision"])
                    if decision.author != "codex":
                        raise ValueError("Authored corrections must identify Codex")
                    decisions.append(decision)
                    correction_replies.append(
                        encoding.replay_reply(correction["reply"])
                    )
                model = RecordedTrajectoryModel(original, encoding, correction_replies)
                result = run_experiment(
                    task=task,
                    model=model,
                    config=config,
                    output_dir=output_dir / "episodes",
                )
                (result.run_dir / "private/source-run.json").write_bytes(source_data)
                entry.update(
                    run_dir=str(result.run_dir.relative_to(output_dir)),
                    run_sha256=hashlib.sha256(
                        (result.run_dir / "run.json").read_bytes()
                    ).hexdigest(),
                )
                if result.status in ("infrastructure_error", "interrupted"):
                    entry.update(
                        reason=result.reason,
                        run_dir=str(result.run_dir.relative_to(output_dir)),
                    )
                    raise OSError(f"Trajectory execution incomplete: {result.reason}")
                if result.status == "model_error":
                    raise ValueError(
                        f"Recorded trajectory replay rejected: {result.reason}"
                    )
                verify_reexecuted_prefix(asdict(result), original)
                export_demonstration(
                    result,
                    task,
                    model,
                    config,
                    writer,
                    supervision=decisions,
                    max_context_tokens=max_context_tokens,
                    demonstration_id=item["demonstration_id"] if identified else None,
                    provenance=item.get("provenance"),
                    complementary_reason=item.get("complementary_reason", ""),
                )
                save_record(
                    result,
                    config,
                    {"task_origin": asdict(task.origin), "passed": result.passed},
                )
                assert result.demonstration is not None
                entry.update(
                    result.demonstration,
                    task_sha256=result.task_sha256,
                    origin=asdict(task.origin),
                    run_dir=str(result.run_dir.relative_to(output_dir)),
                    run_sha256=hashlib.sha256(
                        (result.run_dir / "run.json").read_bytes()
                    ).hexdigest(),
                )
                if identified and entry["status"] == "accepted":
                    family_counts[family] = family_counts.get(family, 0) + 1
                    behaviors.setdefault(family, set()).add(behavior)
            except (ValueError, KeyError, TypeError) as error:
                entry["reason"] = str(error)
            except OSError as error:
                entry["reason"] = str(error)
                raise
            finally:
                writer.save_manifest(manifest, destination)
        manifest["status"] = "complete"
    finally:
        writer.save_manifest(manifest, destination, final=True)
    return destination
