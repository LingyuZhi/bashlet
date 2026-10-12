"""Evaluate saved OPSD checkpoints locally using the existing export and Harness."""

from __future__ import annotations

import fcntl
import json
import shlex
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
import yaml

from ..config import config_record, load_config, load_model
from ..monitoring import MonitorConfig, write_document
from ..snapshots import export_snapshot, prepare_local_snapshot
from ..snapshots.manifest import file_sha256
from .harness import _digest, evaluate


def _read(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Checkpoint evidence must be a JSON object")
    return value


def _endpoint(pilot: dict[str, Any]) -> None:
    epochs = pilot["configuration"]["pilot"]["epochs"]
    main = pilot.get("main_result") or {}
    if (
        pilot.get("training_complete") is not True
        or pilot.get("epochs_completed") != epochs
        or pilot.get("status") != "complete"
        or main.get("epoch") != epochs
        or main.get("optimizer_step") != pilot.get("optimizer_step")
        or main.get("snapshot")
        != pilot.get("epoch_checkpoints", {}).get(str(epochs), {}).get("snapshot")
    ):
        raise ValueError("Held-out evaluation requires the complete final epoch")


def _snapshot(pilot: dict[str, Any], phase: str) -> tuple[str, int]:
    if phase in {"baseline", "test-baseline"}:
        return pilot["initial_snapshot"]["identity"], 0
    if phase in {"final", "test-final"}:
        epoch = pilot["configuration"]["pilot"]["epochs"]
    elif phase.startswith("epoch-"):
        epoch = int(phase.removeprefix("epoch-"))
    else:
        raise ValueError("Unknown checkpoint evaluation phase")
    saved = pilot.get("epoch_checkpoints", {}).get(str(epoch))
    if not saved:
        raise ValueError("Checkpoint evaluation requires a completed epoch receipt")
    return saved["snapshot"], saved["optimizer_step"]


def _prepare(
    config: dict[str, Any], source_snapshot: str, step: int, thinking: bool
) -> Path:
    root = Path(config["output_dir"]).resolve() / "checkpoints" / f"step-{step}"
    exported = root / "export/export.json"
    if not exported.exists():
        exported = export_snapshot(
            Path(config["pilot_run"]) / "training",
            step,
            root / "export",
            api_key_env=config.get("api_key_env", "pytrio_api_key"),
        )
    source = _read(exported)
    if (
        source_snapshot
        != f"pytrio/{source['base_model']}@{source['checkpoint']['path']}"
    ):
        raise ValueError("Local export differs from the recorded checkpoint")
    manifest = root / "local/snapshot.json"
    if not manifest.exists():
        manifest = prepare_local_snapshot(
            Path(config["base_path"]),
            config["model_revision"],
            root / "local",
            export_path=exported,
            merge=True,
            thinking=thinking,
        )
    local = _read(manifest)
    adapter = local.get("adapter") or {}
    if adapter.get("checkpoint") != source["checkpoint"] or (
        adapter.get("export_sha256") != file_sha256(exported)
        or local["encoding_details"].get("thinking") is not thinking
    ):
        raise ValueError("Cached local snapshot differs from its export or encoding")
    return manifest


def _server(config: dict[str, Any], manifest: Path) -> None:
    """Own one configured tmux service; never replace an unrelated session."""
    server = config.get("server")
    if server is None:
        return  # An externally managed service is still checked by VLLMModel.probe.
    root = Path(config["output_dir"]).resolve() / "service"
    root.mkdir(parents=True, exist_ok=True)
    identity = file_sha256(manifest)
    receipt = root / "service.json"
    previous = _read(receipt) if receipt.exists() else None
    session = server["session"]
    exists = (
        subprocess.run(
            ["tmux", "has-session", "-t", session], capture_output=True, check=False
        ).returncode
        == 0
    )
    if exists and (previous is None or previous.get("session") != session):
        raise ValueError("Local vLLM session has no matching ownership receipt")
    if exists and previous and previous.get("snapshot_sha256") != identity:
        subprocess.run(["tmux", "send-keys", "-t", session, "C-c"], check=True)
        deadline = time.monotonic() + server.get("timeout_seconds", 600)
        while (
            subprocess.run(
                ["tmux", "has-session", "-t", session], capture_output=True
            ).returncode
            == 0
        ):
            if time.monotonic() > deadline:
                raise ValueError("Previous local vLLM service has not stopped")
            time.sleep(1)
        exists = False
    if not exists:
        local = _read(manifest)
        template = yaml.safe_load(Path(server["template"]).read_text())
        template.update(
            model=local["adapter"]["merged"]["path"], tokenizer=local["base_path"]
        )
        path = root / f"serve-{identity}.yaml"
        path.write_text(yaml.safe_dump(template))
        command = (
            "source /etc/profile\nmodule load "
            + shlex.quote(server["cuda_module"])
            + "\nexec bash scripts/serve-local-vllm.sh "
            + shlex.quote(str(path))
            + " >> "
            + shlex.quote(str(root / "server.log"))
            + " 2>&1"
        )
        write_document(receipt, {"session": session, "snapshot_sha256": identity})
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session,
                "-c",
                str(Path.cwd()),
                "bash -lc " + shlex.quote(command),
            ],
            check=True,
        )
    deadline = time.monotonic() + server.get("timeout_seconds", 600)
    while time.monotonic() < deadline:
        try:
            response = httpx.get(
                server["base_url"] + "/health", timeout=3, trust_env=False
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise ValueError("Local vLLM did not become ready; inspect service/server.log")


def _historical_receipt(
    pilot: dict[str, Any], seed: str, source: Path, snapshot: str
) -> dict[str, Any]:
    reference: dict[str, Any] | None = (
        pilot.get("evaluations", {}).get("baseline", {}).get("runs", {}).get(seed)
    )
    if reference:
        recorded = Path(reference["path"])
        if file_sha256(recorded) != reference["sha256"]:
            raise ValueError(
                "Historical Baseline evidence differs from its saved receipt"
            )
        if recorded.resolve() != source.resolve():
            # Training resume replaces the stage with the local continuation.
            # Follow its pinned provenance back to the original cloud evidence.
            reference = _read(recorded).get("historical_source")
            if not reference or reference.get("snapshot") != snapshot:
                raise ValueError(
                    "Historical Baseline continuation has no source receipt"
                )
    if not reference or (
        Path(reference["path"]).resolve() != source.resolve()
        or file_sha256(source) != reference["sha256"]
    ):
        raise ValueError("Historical Baseline evidence differs from its saved receipt")
    return reference


def _stage(config_path: Path, phase: str, *, heldout: bool = False) -> Path:
    from ..distillation.conditions import resolve_condition
    from ..distillation.development import development_curve
    from ..distillation.pilot import PilotConfig

    settings = yaml.safe_load(config_path.read_text())
    formal = Path(settings["pilot_run"]).resolve()
    pilot = _read(formal / "pilot.json")
    if heldout:
        _endpoint(pilot)
    original = PilotConfig.read(Path(settings["pilot_config"]))
    if _digest(asdict(original)) != _digest(pilot["configuration"]["pilot"]):
        raise ValueError(
            "Local evaluation must retain the recorded training configuration"
        )
    expected, _ = resolve_condition(original)
    condition = load_config([Path(p) for p in settings["condition"]])
    old, new = config_record(expected), config_record(condition)
    frozen = pilot["configuration"]
    if (
        _digest(old) != _digest(frozen["condition"])
        or file_sha256(Path(original.dev_plan)) != frozen["dev_plan_sha256"]
        or file_sha256(Path(original.train_manifest)) != frozen["train_manifest_sha256"]
    ):
        raise ValueError("Local evaluation frozen conditions or data changed")
    if any(
        _digest(old[key]) != _digest(new[key])
        for key in ("protocol", "environment", "budget")
    ):
        raise ValueError("Local evaluation must preserve the frozen Harness conditions")
    if old["model"]["sampling"] != new["model"]["sampling"] or (
        old["model"]["thinking"] != new["model"]["thinking"]
    ):
        raise ValueError("Local evaluation must preserve sampling and native thinking")
    if condition.model.class_name.replace(":", ".").split(".")[-1] != "VLLMModel":
        raise ValueError("Checkpoint evaluation requires the local VLLMModel backend")
    receipt_path = formal / "evaluation-routing.json"
    receipt = {"config": str(config_path.resolve()), "sha256": file_sha256(config_path)}
    if receipt_path.exists() and _read(receipt_path) != receipt:
        raise ValueError("Local evaluation routing is already frozen to another config")
    write_document(receipt_path, receipt)
    source_snapshot, step = _snapshot(pilot, phase)
    manifest = _prepare(settings, source_snapshot, step, new["model"]["thinking"])
    _server(settings, manifest)
    condition = replace(
        condition,
        model=replace(
            condition.model,
            parameters=condition.model.parameters
            | {"snapshot_manifest": str(manifest)},
        ),
    )
    local_snapshot = _read(manifest)["identity"]
    plan_path = Path(settings["test_plan"] if heldout else original.dev_plan)
    plan = _read(plan_path)
    if (plan_path.parent / plan["manifest"]).resolve() != Path(
        original.train_manifest
    ).resolve():
        raise ValueError("Local evaluation plan belongs to another manifest")
    if heldout:
        cohort = _read(Path(original.train_manifest))["cohort"]["test"]["task_ids"]
        if plan["split"] != "test" or plan["task_ids"] != cohort:
            raise ValueError(
                "Held-out evaluation requires the complete frozen test split"
            )
    root = Path(settings["output_dir"]).resolve() / phase
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "stage.json"
    initial = {
        "snapshot": source_snapshot,
        "local_snapshot": local_snapshot,
        "optimizer_step": step,
        "split": "test" if heldout else "dev",
        "routing_sha256": file_sha256(config_path),
        "status": "running",
        "runs": {},
    }
    stage = _read(destination) if destination.exists() else initial
    if any(
        stage[key] != initial[key] for key in initial if key not in {"status", "runs"}
    ):
        raise ValueError(
            "Local evaluation resume must retain its checkpoint and routing"
        )
    monitoring = MonitorConfig(**(original.monitoring | settings.get("monitoring", {})))
    for seed in original.evaluation_seeds:
        seed_key = str(seed)
        source = (
            Path(settings["historical_baseline"]) / f"seed-{seed}/evaluation.json"
            if phase == "baseline" and settings.get("historical_baseline")
            else None
        )
        reference = stage["runs"].get(seed_key)
        source_receipt = (
            _historical_receipt(pilot, seed_key, source, source_snapshot)
            if source and source.exists()
            else None
        )
        if reference:
            path = Path(reference["path"])
            if file_sha256(path) != reference["sha256"]:
                raise ValueError("Local evaluation evidence changed")
            if reference["status"] == "complete":
                continue
        elif source and source.exists() and _read(source)["status"] == "complete":
            previous = _read(source)
            if previous["snapshot"] != source_snapshot or (
                previous["config"]["model"]["sampling"]["seed"] != seed
                or previous["plan"]["task_ids"] != plan["task_ids"]
            ):
                raise ValueError(
                    "Historical Baseline checkpoint, seed or tasks changed"
                )
            if previous["comparison_sha256"] != _digest(previous["comparison"]) or any(
                _digest(previous["config"][key]) != _digest(old[key])
                for key in ("protocol", "environment", "budget")
            ):
                raise ValueError("Historical Baseline conditions or evidence changed")
            sampling = dict(old["model"]["sampling"]) | {"seed": seed}
            if previous["config"]["model"]["sampling"] != sampling:
                raise ValueError("Historical Baseline sampling changed")
            from ..snapshots import encoding_condition

            if _digest(
                encoding_condition(previous["comparison"]["encoding"])
            ) != _digest(
                encoding_condition(_read(manifest)["encoding_details"])
            ) or previous["plan"]["manifest_sha256"] != file_sha256(
                Path(original.train_manifest)
            ):
                raise ValueError("Historical Baseline encoding or data changed")
            stage["runs"][seed_key] = {
                "path": str(source.resolve()),
                "sha256": file_sha256(source),
                "status": "complete",
                "backend": "pytrio",
                "historical": True,
            }
            write_document(destination, stage)
            continue
        resolved = replace(
            condition,
            model=replace(
                condition.model, sampling=replace(condition.model.sampling, seed=seed)
            ),
        )
        directory = root / f"seed-{seed}"
        path = evaluate(
            plan_path=plan_path,
            snapshot=local_snapshot,
            evidence_kind="experimental",
            condition=resolved,
            output_dir=root,
            run_dir=directory,
            model=load_model(resolved.model),
            monitoring=monitoring,
            training_run=formal / "training/opsd.json",
            phase=phase + "-local-eval",
            resume=(directory / "evaluation.json").exists(),
            continue_from=source if source and source.exists() else None,
            source_snapshot=source_snapshot if source and source.exists() else None,
            expected_source_sha256=source_receipt["sha256"] if source_receipt else None,
            retry_infrastructure=settings.get("retry_infrastructure", False)
            if source and source.exists()
            else False,
            plan_split="test" if heldout else "dev",
            require_online=False,
        )
        result = _read(path)
        stage["runs"][seed_key] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "status": result["status"],
            "backend": "vllm",
        }
        write_document(destination, stage)
        if result["status"] != "complete":
            break
    stage["status"] = (
        "complete"
        if len(stage["runs"]) == len(original.evaluation_seeds)
        and all(row["status"] == "complete" for row in stage["runs"].values())
        else "incomplete"
    )
    write_document(destination, stage)
    record = {
        "configuration": {
            "pilot": {"evaluation_seeds": original.evaluation_seeds},
            "condition": config_record(condition),
        },
        "dev_task_ids": plan["task_ids"],
        "evaluations": {phase: stage},
    }
    point = development_curve(record)[phase]
    per_seed, counts = {}, {"pytrio": 0, "vllm": 0}
    for seed, reference in stage["runs"].items():
        result = _read(Path(reference["path"]))
        per_seed[seed] = result["summary"]
        for row in result["tasks"]:
            if row["task_id"] in result["summary"]["missing_task_ids"]:
                continue
            backend = (
                "pytrio"
                if reference.get("historical") or row.get("historical")
                else "vllm"
            )
            counts[backend] += 1
    report = point | {
        "split": stage["split"],
        "per_seed": per_seed,
        "backend_counts": counts,
        "mixed_inference": all(counts.values()),
        "comparison_limitation": (
            "Historical cloud results retained by request; "
            "a mixed Baseline is not a fully local paired control."
        )
        if all(counts.values())
        else None,
    }
    write_document(root / "report.json", report)
    return destination


def evaluate_checkpoint(config_path: Path, *, phase: str = "baseline") -> Path:
    settings = yaml.safe_load(config_path.read_text())
    # Reject premature test access before creating output files or exporting weights.
    if phase == "test":
        _endpoint(_read(Path(settings["pilot_run"]) / "pilot.json"))
    root = Path(settings["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "A local checkpoint evaluation is already active"
            ) from None
        if phase == "test":
            baseline = _stage(config_path, "test-baseline", heldout=True)
            if _read(baseline)["status"] != "complete":
                return baseline
            final = _stage(config_path, "test-final", heldout=True)
            base_stage, final_stage = _read(baseline), _read(final)
            matched = final_stage["status"] == "complete"
            for seed, old in base_stage["runs"].items():
                new = final_stage["runs"].get(seed)
                matched = (
                    matched
                    and new is not None
                    and (
                        _read(Path(old["path"]))["comparison_sha256"]
                        == _read(Path(new["path"]))["comparison_sha256"]
                    )
                )
            before = _read(baseline.parent / "report.json")
            after = _read(final.parent / "report.json")
            comparison = {
                "status": "complete" if matched else "incomplete",
                "matched_complete": matched,
                "baseline": before,
                "opsd": after,
                "mean_accuracy_difference": after["mean_accuracy"]
                - before["mean_accuracy"]
                if matched
                else None,
            }
            destination = root / "test-comparison.json"
            write_document(destination, comparison)
            return destination
        return _stage(config_path, phase)
