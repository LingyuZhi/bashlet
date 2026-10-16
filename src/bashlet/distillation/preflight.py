"""Audit real native-thinking updates before the independent formal run."""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import config_record
from ..experiment import run_experiment, task_from_record
from ..monitoring import write_document
from ..tasks import task_path
from . import OPSDRun
from .conditions import resolve_condition
from .pytrio import PyTRIOOPSD


def verify_native_preflight(
    config_path: Path, root: Path, adapter: PyTRIOOPSD, run: OPSDRun
) -> dict[str, Any]:
    """Use saved raw requests, plus a real one-token truncation probe."""
    import pytrio as trio
    from transformers import AutoTokenizer

    from ..encoding import _snapshot
    from .pilot import PilotConfig

    config = PilotConfig.read(config_path)
    condition, backend = resolve_condition(config)
    if not backend.thinking or run.record["pending_update"] is not None:
        raise ValueError("Native preflight requires thinking and settled updates")
    official = AutoTokenizer.from_pretrained(
        str(
            _snapshot(
                backend.base_model, backend.model_revision, backend.tokenizer_cache_dir
            )
        ),
        local_files_only=True,
    )
    batches = run.record["batches"]
    if len(batches) < 2 or any(b["status"] != "completed" for b in batches):
        raise ValueError("Native preflight needs two completed on-policy updates")
    previous = run.snapshot.identity
    reasoning_count = action_count = submitted = 0
    receipts = []
    for batch in batches:
        episode_root = Path(batch["episode"])
        episode = json.loads((episode_root / "run.json").read_text())
        task = task_from_record(
            json.loads(
                task_path(
                    Path(config.train_manifest), episode["task_id"], "train"
                ).read_text()
            )
        )
        if (
            batch["student_policy"] != previous
            or batch["teacher_policy"] != run.snapshot.identity
        ):
            raise ValueError("Native preflight policy chain or frozen Teacher changed")
        datum = trio.Datum.model_validate(
            json.loads((episode_root / "private/opsd.json").read_text())["datum"]
        )
        tokens = episode["context_tokens"]
        if datum.model_input.to_ints() != tokens[:-1]:
            raise ValueError("Native preflight training input differs from raw history")
        inputs = {
            key: np.asarray(value.tolist())
            for key, value in datum.loss_fn_inputs.items()
        }
        supervised: set[int] = set()
        expected_prompt = official.apply_chat_template(
            [
                {"role": "system", "content": condition.protocol.system_prompt},
                {"role": "user", "content": task.prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        first = episode["trajectory"][0]["model_call"]["context_offset"]
        if tokens[:first] != expected_prompt:
            raise ValueError(
                "Native preflight prompt differs from the official template"
            )
        for step, span in zip(episode["trajectory"], batch["spans"], strict=True):
            call = step["model_call"]
            raw = call["generation"]["tokens"]
            start, end = span["target_start"], span["target_start"] + len(raw)
            teacher = json.loads(
                (episode_root / f"private/teacher-{span['step']:04d}.json").read_text()
            )
            scores = teacher["logprobs"][span["teacher_start"] : span["teacher_end"]]
            old = call["generation"]["logprobs"]
            if (
                teacher["tokens"][span["teacher_start"] : span["teacher_end"]] != raw
                or tokens[call["context_offset"] : call["context_offset"] + len(raw)]
                != raw
                or inputs["target_tokens"][start:end].tolist() != raw
                or len(scores) != len(raw)
                or not np.isfinite(scores).all()
                or not np.allclose(inputs["logprobs"][start:end], old, atol=1e-6)
                or not np.allclose(
                    inputs["advantages"][start:end],
                    run.config.distillation_coefficient * (np.asarray(scores) - old),
                    atol=1e-5,
                )
            ):
                raise ValueError(
                    "Native preflight sampled-token supervision is misaligned"
                )
            supervised.update(range(start, end))
            marker = official.convert_tokens_to_ids("</think>")
            if marker in raw:
                boundary = len(raw) - 1 - raw[::-1].index(marker)
                reasoning_count += boundary
                action_count += len(raw) - boundary - 1
        masked = [index for index in range(len(tokens) - 1) if index not in supervised]
        if any(np.any(value[masked]) for value in inputs.values()):
            raise ValueError(
                "Native preflight prompt or observation carries supervision"
            )
        submitted += episode["episode_status"] == "submitted" and episode["passed"]
        previous = batch["update"]["student_policy"]
        receipts.append(
            {
                "episode": str(episode_root),
                "supervised_tokens": len(supervised),
                "masked_tokens": len(masked),
                "optimizer_step": batch["update"]["optimizer_step"],
            }
        )
    if not (reasoning_count and action_count and submitted):
        raise ValueError(
            "Native preflight lacks thinking, action or independently graded submission"
        )
    task = task_from_record(
        json.loads(
            task_path(
                Path(config.train_manifest), config.train_task_ids[0], "train"
            ).read_text()
        )
    )
    stress = replace(
        condition,
        model=replace(
            condition.model, sampling=replace(condition.model.sampling, max_tokens=1)
        ),
        budget=replace(
            condition.budget,
            max_steps=1,
            max_generated_tokens=1,
            max_consecutive_invalid=1,
        ),
    )
    # No optimizer work in the stress probe. Its original tokens and stop reason
    # remain an independent experiment under the restored current Student.
    result = run_experiment(
        task=task, model=run.student, config=stress, output_dir=root / "truncation"
    )
    if (
        not result.trajectory
        or any(step.observation is not None for step in result.trajectory)
        or result.trajectory[0].model_call is None
    ):
        raise ValueError("Truncated thinking unexpectedly executed an action")
    call = result.trajectory[0].model_call
    reply = run.student.encoding.reply(call.generation.tokens)
    if call.generation.stop_reason != "length" or reply.thinking_complete:
        raise ValueError("Real truncation did not cover unfinished native thinking")
    return {
        "status": "complete",
        "condition": config_record(condition),
        "backend": asdict(backend),
        "training": asdict(run.config),
        "thinking_tokens": reasoning_count,
        "action_tokens": action_count,
        "graded_successful_submissions": submitted,
        "updates": receipts,
        "truncation": {
            "episode": str(result.run_dir),
            "stop_reason": "length",
            "raw_tokens": list(call.generation.tokens),
            "actions_executed": 0,
        },
        "initial_snapshot": run.snapshot.identity,
        "restored_student": run.student.identity,
    }


def _run_with_preflight(
    config_path: Path, preflight_config: Path, output_dir: Path,
    execution_limit: Path | None,
) -> Path:
    """Gate the existing training workflow; never mix preflight parameters."""
    from .pilot import run_opsd_pilot, verify_opsd_checkpoints

    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / "workflow.json"
    frozen = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (config_path, preflight_config)
    }
    record = (
        json.loads(path.read_text())
        if path.exists()
        else {
            "configuration_sha256": frozen,
            "status": "preflight",
        }
    )
    if record["configuration_sha256"] != frozen:
        raise ValueError("Workflow resume must preserve both configurations")
    write_document(path, record)
    preflight_root = root / "preflight"
    pilot = run_opsd_pilot(
        preflight_config,
        preflight_root,
        resume=(preflight_root / "pilot.json").exists(),
    )
    if json.loads(pilot.read_text())["status"] != "complete":
        record.update(status="incomplete", stage="preflight-training")
        write_document(path, record)
        return pilot
    proof = preflight_root / "checkpoint-validation.json"
    if not proof.exists():
        verify_opsd_checkpoints(preflight_config, preflight_root, native=True)
    validation = json.loads(proof.read_text())
    if (
        validation.get("status") != "complete"
        or validation.get("native", {}).get("status") != "complete"
    ):
        record.update(status="incomplete", stage="preflight-verification")
        write_document(path, record)
        return proof
    digest = hashlib.sha256(proof.read_bytes()).hexdigest()
    if record.get("preflight_sha256", digest) != digest:
        raise ValueError("Workflow preflight evidence changed")
    record.update(
        status="formal", stage="formal", preflight_sha256=digest, preflight=str(proof)
    )
    write_document(path, record)
    formal_root = root / "formal"
    formal = run_opsd_pilot(
        config_path, formal_root, resume=(formal_root / "pilot.json").exists(),
        execution_limit=execution_limit,
    )
    record.update(status=json.loads(formal.read_text())["status"], formal=str(formal))
    write_document(path, record)
    return formal


def run_with_preflight(
    config_path: Path,
    preflight_config: Path,
    output_dir: Path,
    *,
    resume: bool = False,
    execution_limit: Path | None = None,
) -> Path:
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("OPSD workflow is already active") from None
        if (root / "workflow.json").exists() and not resume:
            raise ValueError("OPSD workflow exists; use --resume")
        try:
            return _run_with_preflight(config_path, preflight_config, root, execution_limit)
        except BaseException as error:
            path = root / "workflow.json"
            if path.exists():
                record = json.loads(path.read_text())
                record.update(
                    status="interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "failed",
                    error_type=type(error).__name__,
                )
                write_document(path, record)
            raise
