"""Evaluate saved OPSD checkpoints locally using the existing export and Harness."""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
import yaml

from bashlet.config import ModelConfig, config_record, load_model
from bashlet.evaluation.harness import _digest, _require_verified_local, evaluate
from bashlet.monitoring import MonitorConfig, write_document
from bashlet.snapshots import export_snapshot, prepare_local_snapshot
from bashlet.snapshots.manifest import file_sha256


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


def _service_config(
    server: dict[str, Any], model: str, tokenizer: str
) -> dict[str, Any]:
    if "options" in server:
        settings = dict(server["options"])
    else:
        settings = yaml.safe_load(Path(server["template"]).read_text())
    settings.update(model=model, tokenizer=tokenizer)
    return settings


def write_vllm_config(config_path: Path, output: Path) -> Path:
    """Materialize the base-model service without requiring prepared training data."""
    from bashlet.config import read_yaml
    from bashlet.distillation.config import _validate_evaluation

    evaluation = read_yaml(config_path)["evaluation"]
    _validate_evaluation(evaluation)
    base = evaluation["base_path"]
    settings = _service_config(evaluation["server"], base, base)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(settings))
    return output


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
        template = _service_config(
            server, local["adapter"]["merged"]["path"], local["base_path"]
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
                config["model"]["base_url"] + "/health", timeout=3, trust_env=False
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise ValueError("Local vLLM did not become ready; inspect service/server.log")


def _settings(config_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    from bashlet.distillation.config import RunConfig
    from bashlet.distillation.pytrio import PyTRIOOPSDConfig

    original = RunConfig.read(config_path)
    if original.evaluation is None:
        raise ValueError("The experiment must configure local checkpoint evaluation")
    formal = run_dir or Path(original.output_dir) / (
        "formal" if original.preflight else ""
    )
    backend = PyTRIOOPSDConfig(**original.backend)
    return original.evaluation | {
        "pilot_run": str(formal.resolve()),
        "output_dir": str(formal.resolve() / "evaluation"),
        "model_revision": backend.model_revision,
        "api_key_env": backend.api_key_env,
    }


def _heldout_plan(manifest_path: Path, directory: Path) -> Path:
    """Derive one immutable evaluation list from the actual frozen task set."""
    manifest = _read(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Held-out evaluation requires a complete frozen manifest")
    identities = manifest["cohort"]["test"]["task_ids"]
    if (
        not isinstance(identities, list)
        or not identities
        or any(not isinstance(identity, str) or not identity for identity in identities)
        or len(identities) != len(set(identities))
    ):
        raise ValueError("Held-out evaluation requires unique frozen test task IDs")
    path = directory / "heldout-plan.json"
    plan = {
        "version": "development-evaluation-v1",
        "name": f"{manifest['source']['revision']}-heldout",
        "manifest": os.path.relpath(manifest_path.resolve(), directory.resolve()),
        "split": "test",
        "task_ids": identities,
    }
    content = (json.dumps(plan, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Frozen held-out evaluation plan changed")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return path


def _stage(
    config_path: Path, phase: str, *, heldout: bool = False, run_dir: Path | None = None
) -> Path:
    from bashlet.distillation.config import RunConfig, resolve_condition
    from bashlet.distillation.development import development_curve

    settings = _settings(config_path, run_dir)
    formal = Path(settings["pilot_run"]).resolve()
    pilot = _read(formal / "pilot.json")
    if heldout:
        _endpoint(pilot)
    original = RunConfig.read(config_path)
    if _digest(asdict(original)) != _digest(pilot["configuration"]["pilot"]):
        raise ValueError(
            "Local evaluation must retain the recorded training configuration"
        )
    expected, _ = resolve_condition(original)
    parameters = dict(settings.get("model", {}))
    parameters.setdefault(
        "model", expected.model.parameters.get("model", "Qwen/Qwen3.5-4B")
    )
    parameters.update(
        model_revision=settings["model_revision"],
        tokenizer=settings["base_path"],
        tokenizer_revision=settings["model_revision"],
        thinking=expected.model.parameters["thinking"],
    )
    condition = replace(
        expected, model=ModelConfig("VLLMModel", expected.model.sampling, parameters)
    )
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
    source_snapshot, step = _snapshot(pilot, phase)
    manifest = _prepare(settings, source_snapshot, step, new["model"]["thinking"])
    condition = replace(
        condition,
        model=replace(
            condition.model,
            parameters=condition.model.parameters
            | {"snapshot_manifest": str(manifest)},
        ),
    )
    local_snapshot = _read(manifest)["identity"]
    plan_path = (
        _heldout_plan(Path(original.train_manifest), Path(settings["output_dir"]))
        if heldout
        else Path(original.dev_plan)
    )
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
        "routing_sha256": _digest(asdict(original)),
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
    server_ready = False
    for seed in original.evaluation_seeds:
        seed_key = str(seed)
        reference = stage["runs"].get(seed_key)
        if reference:
            saved = Path(reference["path"])
            if file_sha256(saved) != reference["sha256"]:
                raise ValueError("Local evaluation evidence changed")
            if reference["status"] == "complete":
                _require_verified_local(_read(saved), saved)
                continue
        if not server_ready:
            _server(settings, manifest)
            server_ready = True
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
            plan_split="test" if heldout else "dev",
            require_online=False,
        )
        result = _read(path)
        if result["status"] == "complete":
            _require_verified_local(result, path)
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
    per_seed = {}
    completed = 0
    for seed, reference in stage["runs"].items():
        result = _read(Path(reference["path"]))
        per_seed[seed] = result["summary"]
        completed += len(result["tasks"]) - len(result["summary"]["missing_task_ids"])
    report = point | {
        "split": stage["split"],
        "per_seed": per_seed,
        "backend_counts": {"vllm": completed},
    }
    report_path = root / "report.json"
    # Dictionary iteration order can differ after a process restart. Reuse the
    # original bytes when the cached report still describes the same results.
    if not report_path.exists() or _read(report_path) != report:
        write_document(report_path, report)
    return destination


def evaluate_checkpoint(
    config_path: Path, *, phase: str = "baseline", run_dir: Path | None = None
) -> Path:
    settings = _settings(config_path, run_dir)
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
            baseline = _stage(
                config_path, "test-baseline", heldout=True, run_dir=run_dir
            )
            if _read(baseline)["status"] != "complete":
                return baseline
            final = _stage(config_path, "test-final", heldout=True, run_dir=run_dir)
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
        return _stage(config_path, phase, run_dir=run_dir)
