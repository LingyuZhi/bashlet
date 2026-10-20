"""Epoch visits and checkpoint evaluation through the training workflow CLI seam."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from test_cli import cli
from test_mbpp import double_row, prepare
from test_monitoring import external_sdks

from bashlet import RequestJournal


@pytest.fixture
def epoch_pilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    prepared = prepare(
        tmp_path, [double_row(602), double_row(603), double_row(511), double_row(512)]
    )
    dev = tmp_path / "dev.json"
    dev.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "epoch-fixture",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511", "mbpp/512"],
            }
        )
    )
    condition = tmp_path / "condition.yaml"
    condition.write_text(yaml.safe_dump({"budget": {"max_steps": 3}}))
    tokenizer = Path(
        "runs/tokenizers/models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    ).resolve()
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(
        yaml.safe_dump(
            {
                "train_manifest": str(prepared / "manifest.json"),
                "train_task_ids": ["mbpp/602", "mbpp/603"],
                "epochs": 2,
                "shuffle_seed": 4,
                "evaluation_seeds": [4, 5],
                "evaluate_each_epoch": True,
                "evaluation_groups": {
                    "baseline": "epoch-baseline",
                    "intermediate": "epoch-middle",
                    "final": "epoch-final",
                },
                "dev_plan": str(dev),
                "condition": [str(condition)],
                "evaluation": {
                    "base_path": str(tokenizer),
                    "model": {"base_url": "http://test-double"},
                },
                "journal_path": str(tmp_path / "requests.sqlite3"),
                "backend": {
                    "tokenizer_cache_dir": str(tokenizer.parents[2]),
                    "group": "epoch-train",
                },
                "training": {"group": "epoch-train"},
                "monitoring": {"group": "epoch-fixture"},
            }
        )
    )
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import opsd_sdk\nopsd_sdk.install()\n")
    monkeypatch.setenv("pytrio_api_key", "test-trio-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_OPSD_TOKENIZER", str(tokenizer))
    return pilot, tmp_path / "pilot-run"


def test_each_epoch_resamples_current_student_and_evaluates_paired_seeds(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    config, root = epoch_pilot
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["status"] == "complete"
    assert record["optimizer_step"] == 4
    assert record["epochs_completed"] == 2
    assert record["configuration"]["condition"]["model"]["thinking"] is True
    backend = json.loads((root / "training/backend.json").read_text())
    assert backend["encoding"]["thinking"] is True
    first_episode = Path(record["training_tasks"][0]["episode"])
    episode = json.loads((first_episode / "run.json").read_text())
    assert not (first_episode / "workspace/MUST_NOT_RUN").exists()
    from transformers import AutoTokenizer

    official = AutoTokenizer.from_pretrained(
        str(
            Path(__file__).parents[1]
            / "runs/tokenizers/models--Qwen--Qwen3.5-4B"
            / "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        ),
        local_files_only=True,
    )
    request = json.loads((tmp_path / "sdk-updates.json").read_text())[0]["data"][0]
    inputs = request["loss_fn_inputs"]
    supervised: set[int] = set()
    for step in episode["trajectory"]:
        call = step["model_call"]
        raw = call["generation"]["tokens"]
        assert official.convert_tokens_to_ids("</think>") in raw
        start = call["context_offset"] - 1
        end = start + len(raw)
        supervised.update(range(start, end))
        assert inputs["target_tokens"][start:end] == raw
        assert inputs["logprobs"][start:end] == [-0.25] * len(raw)
        assert inputs["advantages"][start:end] == pytest.approx([0.15] * len(raw))
    assert all(
        inputs[key][index] == 0
        for index in range(len(inputs["target_tokens"]))
        if index not in supervised
        for key in ("target_tokens", "logprobs", "advantages")
    )
    assert official.decode(episode["context_tokens"]).count("</think>") >= 2
    visits = record["training_tasks"]
    assert [row["epoch"] for row in visits] == [1, 1, 2, 2]
    assert len({row["visit_id"] for row in visits}) == 4
    assert len({row["episode"] for row in visits}) == 4
    training = json.loads((root / "training/opsd.json").read_text())
    teacher = record["initial_snapshot"]["identity"]
    for index, batch in enumerate(training["batches"]):
        assert batch["teacher_policy"] == teacher
        assert batch["student_policy"] == (
            teacher
            if index == 0
            else training["batches"][index - 1]["update"]["student_policy"]
        )
    assert {
        phase: stage["optimizer_step"] for phase, stage in record["evaluations"].items()
    } == {
        "baseline": 0,
        "epoch-1": 2,
        "final": 4,
    }
    for stage in record["evaluations"].values():
        assert set(stage["runs"]) == {"4", "5"}
        assert stage["status"] == "complete"
    assert record["development_curve"]["final"]["mean_accuracy"] == 1.0
    assert record["development_curve"]["final"]["planned_episodes"] == 4
    assert record["epoch_checkpoints"]["2"]["optimizer_step"] == 4
    for seed in ("4", "5"):
        before = json.loads(
            Path(record["evaluations"]["baseline"]["runs"][seed]["path"]).read_text()
        )
        after = json.loads(
            Path(record["evaluations"]["final"]["runs"][seed]["path"]).read_text()
        )
        assert before["comparison_sha256"] == after["comparison_sha256"]
        assert (
            before["summary"]["passed_tasks"] == after["summary"]["passed_tasks"] == 2
        )
    requests = record["requests"]
    resumed = cli(
        "train-opsd", str(config), "--output-dir", str(root), "--resume", cwd=tmp_path
    )
    assert resumed.returncode == 0, resumed.stderr
    assert json.loads((root / "pilot.json").read_text())["requests"] == requests


def execution_limit(config: Path, max_updates: int) -> Path:
    path = config.parent / "execution-limit.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "pilot_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "max_updates": max_updates,
                "reason": "temporary resource limit",
            }
        )
    )
    return path


def test_default_request_journal_uses_the_selected_run_directory(
    epoch_pilot: tuple[Path, Path], tmp_path: Path
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.pop("journal_path")
    settings.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    config.write_text(yaml.safe_dump(settings))
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "pilot.json").read_text())
    journal = RequestJournal(root / "requests.sqlite3")
    assert record["requests"]["journal"] == str(journal.path)
    assert record["training_complete"] and record["optimizer_step"] == 1
    assert record["requests"]["request_ids"]
    assert journal.snapshot()["pending_count"] == 0
    assert not (tmp_path / "runs/requests.sqlite3").exists()


def test_execution_limit_finishes_epoch_development_without_promoting_endpoint(
    epoch_pilot: tuple[Path, Path], tmp_path: Path
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=4, require_complete_training=True)
    config.write_text(yaml.safe_dump(settings))
    limit = execution_limit(config, 4)
    result = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--execution-limit",
        str(limit),
        cwd=tmp_path,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["status"] == "incomplete"
    assert record["execution_status"] == "complete"
    assert record["execution_limit"]["max_updates"] == 4
    assert record["optimizer_step"] == 4
    assert record["epochs_completed"] == 2
    assert not record["training_complete"]
    assert record["main_result"] is None
    assert record["training_progress"]["planned_updates"] == 8
    assert all(row["status"] == "not_started" for row in record["training_tasks"][4:])
    assert set(record["evaluations"]) == {"baseline", "epoch-1", "epoch-2"}
    assert record["development_curve"]["epoch-2"]["status"] == "complete"
    training = json.loads((root / "training/opsd.json").read_text())
    assert training["pending_update"] is None
    assert (
        training["last_update"]["training_state"]
        == (record["epoch_checkpoints"]["2"]["training_state"])
    )


def test_execution_limit_resume_finishes_interrupted_dev_without_another_update(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=4, require_complete_training=True)
    config.write_text(yaml.safe_dump(settings))
    limit = execution_limit(config, 4)
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL_PHASE", "epoch-2-eval")
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL_SEED", "5")
    arguments = (
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--execution-limit",
        str(limit),
    )

    interrupted = cli(*arguments, cwd=tmp_path, timeout=90)

    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    before = json.loads((root / "pilot.json").read_text())
    assert before["optimizer_step"] == 4
    assert before["evaluations"]["epoch-2"]["runs"]["4"]["status"] == "complete"
    completed = {
        ref["path"]: ref["sha256"]
        for stage in before["evaluations"].values()
        for ref in stage["runs"].values()
        if ref["status"] == "complete"
    }
    updates = (tmp_path / "sdk-updates.json").read_bytes()
    assert len(json.loads(updates)) == 4
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_EVAL")

    resumed = cli(*arguments, "--resume", cwd=tmp_path, timeout=90)

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "pilot.json").read_text())
    assert after["optimizer_step"] == 4
    assert after["epochs_completed"] == 2
    assert after["execution_status"] == "complete"
    assert after["training_complete"] is False
    assert after["main_result"] is None
    assert set(after["evaluations"]) == {"baseline", "epoch-1", "epoch-2"}
    assert after["evaluations"]["epoch-2"]["status"] == "complete"
    assert (tmp_path / "sdk-updates.json").read_bytes() == updates
    training = json.loads((root / "training/opsd.json").read_text())
    assert training["pending_update"] is None
    assert len(training["batches"]) == training["optimizer_step"] == 4
    for path, digest in completed.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("interrupt_step", [3, 4])
def test_resume_recovers_second_epoch_update_without_reusing_first_epoch_episode(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_step: int,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP", str(interrupt_step))
    interrupted = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert interrupted.returncode == 2
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == interrupt_step
    assert record["training_tasks"][interrupt_step - 1]["status"] == "running"
    training = json.loads((root / "training/opsd.json").read_text())
    completed_episode = training["batches"][interrupt_step - 1]["episode"]
    frozen = {
        ref["path"]: ref["sha256"]
        for stage in record["evaluations"].values()
        for ref in stage["runs"].values()
    }
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_TRAIN_STEP")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "pilot.json").read_text())
    assert after["optimizer_step"] == 4
    recovered = after["training_tasks"][interrupt_step - 1]
    assert recovered["episode"] == completed_episode
    assert recovered["optimizer_step"] == interrupt_step
    assert recovered["recovered_completed_update"]
    batches = json.loads((root / "training/opsd.json").read_text())["batches"]
    assert len(batches) == 4
    assert after["training_progress"]["optimizer_steps"] == 4
    assert after["training_progress"]["trajectories"] == 4
    assert after["training_progress"]["supervised_tokens"] == sum(
        batch["supervised_tokens"] for batch in batches
    )
    assert after["training_progress"]["nonzero_signal_tokens"] == sum(
        batch["nonzero_signal_tokens"] for batch in batches
    )
    for path, digest in frozen.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def test_incomplete_final_seed_does_not_claim_a_complete_paired_comparison(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL_PHASE", "final-eval")
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL_SEED", "5")
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 2
    assert "Metrics must be finite scalars" not in result.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == 4
    assert record["evaluations"]["final"]["status"] == "incomplete"
    assert record["evaluations"]["final"]["runs"]["4"]["status"] == "complete"
    assert not record["comparison"]["matched_complete"]
    assert record["decision"]["action"] == "insufficient_evidence"
    development = record["development_curve"]["final"]
    assert development["planned_families"] == 2
    assert development["planned_episodes"] == 4
    assert development["mean_accuracy"] is None
    assert development["missing"]
    assert development["observed_success_lower_bound"] < 1


def test_pending_second_epoch_update_is_retained_without_new_remote_requests(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_OPTIM_FAIL_STEP", "3")
    result = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert result.returncode == 2
    training = json.loads((root / "training/opsd.json").read_text())
    assert training["optimizer_step"] == 2
    assert training["pending_update"]["optimizer_step"] == 3
    configuration = yaml.safe_load(config.read_text())
    journal = RequestJournal(
        Path(configuration["journal_path"]),
    )
    before = journal.snapshot()
    assert before["pending_count"] > 0
    monkeypatch.delenv("FAKE_OPTIM_FAIL_STEP")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 2
    assert journal.snapshot()["entries"] == before["entries"]
    after = json.loads((root / "training/opsd.json").read_text())
    assert after["pending_update"] == training["pending_update"]
    assert after["optimizer_step"] == 2


def test_monitor_start_failure_before_sampling_can_resume_the_same_visit(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, root = epoch_pilot
    monkeypatch.setenv("FAKE_WANDB_TRAIN_START_FAIL", "1")
    failed = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert failed.returncode == 2
    before = json.loads((root / "pilot.json").read_text())
    assert before["optimizer_step"] == 1
    assert "W&B startup failed" in before["reason"]
    assert before["training_tasks"][1]["status"] == "not_started"
    assert before["training_tasks"][1]["recovered_unstarted_visit"]
    episode = before["training_tasks"][0]["episode"]
    baseline = before["evaluations"]["baseline"]["runs"]
    monkeypatch.delenv("FAKE_WANDB_TRAIN_START_FAIL")
    resumed = cli(
        "train-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "pilot.json").read_text())
    assert after["optimizer_step"] == 4
    assert after["training_tasks"][0]["episode"] == episode
    assert after["evaluations"]["baseline"]["runs"] == baseline
    assert len(json.loads((root / "training/opsd.json").read_text())["batches"]) == 4


def test_native_preflight_gates_an_independent_formal_run_and_reuses_its_receipts(
    epoch_pilot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = epoch_pilot
    configuration = yaml.safe_load(config.read_text())
    preflight = tmp_path / "preflight.yaml"
    preliminary = dict(configuration) | {
        "epochs": 1,
        "evaluation_seeds": [4],
        "evaluate_each_epoch": False,
        "require_complete_training": True,
    }
    preflight.write_text(yaml.safe_dump(preliminary))
    configuration["require_complete_training"] = True
    configuration["preflight_evidence"] = str(
        root / "preflight/checkpoint-validation.json"
    )
    config.write_text(yaml.safe_dump(configuration))
    arguments = (
        "train-opsd",
        str(config),
        "--preflight-config",
        str(preflight),
        "--output-dir",
        str(root),
    )
    monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    interrupted = cli(*arguments, cwd=tmp_path, timeout=90)
    assert interrupted.returncode == 2, interrupted.stdout + interrupted.stderr
    assert not (root / "formal/pilot.json").exists()
    monkeypatch.delenv("FAKE_OPSD_INTERRUPT_EVAL")
    result = cli(*arguments, "--resume", cwd=tmp_path, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    workflow = json.loads((root / "workflow.json").read_text())
    assert workflow["status"] == "complete"
    proof = json.loads((root / "preflight/checkpoint-validation.json").read_text())
    assert proof["native"]["thinking_tokens"] > 0
    assert proof["native"]["action_tokens"] > 0
    assert proof["native"]["truncation"]["actions_executed"] == 0
    assert proof["native"]["truncation"]["stop_reason"] == "length"
    before = json.loads((root / "preflight/pilot.json").read_text())
    formal = json.loads((root / "formal/pilot.json").read_text())
    assert before["optimizer_step"] == 2
    assert formal["optimizer_step"] == 4
    assert before["run_id"] != formal["run_id"]
    assert formal["main_result"]["epoch"] == 2
    resumed = cli(*arguments, "--resume", cwd=tmp_path, timeout=60)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    after = json.loads((root / "formal/pilot.json").read_text())
    assert after["requests"] == formal["requests"]


@pytest.mark.parametrize("thinking", [False, True])
def test_explicit_mode_and_unfinished_thinking_retain_original_supervision(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    thinking: bool,
) -> None:
    config, root = epoch_pilot
    configuration = yaml.safe_load(config.read_text())
    configuration.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    condition = Path(configuration["condition"][0])
    condition.write_text(
        yaml.safe_dump({"model": {"thinking": thinking}, "budget": {"max_steps": 1}})
    )
    config.write_text(yaml.safe_dump(configuration))
    if thinking:
        monkeypatch.setenv("FAKE_OPSD_UNFINISHED", "1")
    process = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((root / "pilot.json").read_text())
    episode_root = Path(record["training_tasks"][0]["episode"])
    episode = json.loads((episode_root / "run.json").read_text())
    assert episode["encoding_details"]["thinking"] is thinking
    if thinking:
        step = episode["trajectory"][0]
        assert step["observation"] is None
        assert step["model_call"]["generation"]["stop_reason"] == "length"
        assert not (episode_root / "workspace/MUST_NOT_RUN").exists()
        request = json.loads((tmp_path / "sdk-updates.json").read_text())[0]["data"][0]
        raw = step["model_call"]["generation"]["tokens"]
        start = step["model_call"]["context_offset"] - 1
        assert (
            request["loss_fn_inputs"]["target_tokens"][start : start + len(raw)] == raw
        )


@pytest.mark.parametrize("parameter_delta", [1.0, 1e-12])
def test_checkpoint_verification_uses_parameters_without_rescoring(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parameter_delta: float,
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    config.write_text(yaml.safe_dump(settings))
    trained = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    updates = (tmp_path / "sdk-updates.json").read_bytes()
    journal = RequestJournal(
        Path(settings["journal_path"]),
    )
    requests = journal.snapshot()["entries"]
    monkeypatch.setenv("FAKE_CHECKPOINT_SCORING_FORBIDDEN", "1")
    monkeypatch.setenv("FAKE_STUDENT_PARAMETER_DELTA", str(parameter_delta))
    verified = cli(
        "verify-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    proof = json.loads((root / "checkpoint-validation.json").read_text())
    assert proof["frozen_teacher_unchanged"]
    assert proof["student_parameters_changed"]
    assert proof["parameter_evidence"]["initial_b_all_zero"]
    assert proof["parameter_evidence"]["changed_tensors"] > 0
    assert proof["parameter_evidence"]["teacher_load"]["model_path"].endswith(
        "sampler-0000"
    )
    assert proof["parameter_evidence"]["student_load"]["model_path"].endswith(
        "sampler-0001"
    )
    assert proof["training_state"]["optimizer_step"] == 1
    assert proof["training_state"]["completed_state"].endswith("restored-state-0001")
    assert journal.snapshot()["entries"] == requests
    assert (tmp_path / "sdk-updates.json").read_bytes() == updates
    assert not (root / "training/checkpoint-scores.json").exists()
    for obsolete in (
        "frozen_teacher_numeric_consistent",
        "frozen_teacher_max_difference",
        "student_probabilities_changed",
        "student_max_difference",
        "private_scores",
    ):
        assert obsolete not in proof


@pytest.mark.parametrize(
    "damage",
    [
        "FAKE_TEACHER_WEIGHTS_CHANGED",
        "FAKE_STUDENT_WEIGHTS_UNCHANGED",
        "tampered_export",
    ],
)
def test_checkpoint_parameter_evidence_rejects_wrong_weights(
    epoch_pilot: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    config.write_text(yaml.safe_dump(settings))
    trained = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    if damage == "tampered_export":
        exported = root / "training/checkpoint-weights/step-0"
        saved = cli(
            "export-snapshot",
            str(root / "training"),
            "--step",
            "0",
            "--output-dir",
            str(exported),
            "--api-key-env",
            "pytrio_api_key",
            cwd=tmp_path,
            timeout=90,
        )
        assert saved.returncode == 0, saved.stdout + saved.stderr
        with (exported / "adapter/adapter_model.safetensors").open("ab") as stream:
            stream.write(b"corrupt")
    else:
        monkeypatch.setenv(damage, "1")
    verified = cli(
        "verify-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert verified.returncode == 2
    proof = json.loads((root / "checkpoint-validation.json").read_text())
    assert proof["status"] == "failed"
    assert proof["reason"].startswith("Checkpoint parameters:")


def test_checkpoint_retry_preserves_failure_and_does_not_repeat_updates(
    epoch_pilot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, root = epoch_pilot
    settings = yaml.safe_load(config.read_text())
    settings.update(epochs=1, train_task_ids=["mbpp/602"], evaluation_seeds=[4])
    config.write_text(yaml.safe_dump(settings))
    trained = cli(
        "train-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    updates = (tmp_path / "sdk-updates.json").read_bytes()
    scores_path = root / "training/checkpoint-scores.json"
    scores_path.write_text('[{"source": "historical scoring verification"}]\n')
    scores = scores_path.read_bytes()
    monkeypatch.setenv("FAKE_CHECKPOINT_DOWNLOAD_FAILURE", "1")
    failed = cli(
        "verify-opsd", str(config), "--output-dir", str(root), cwd=tmp_path, timeout=90
    )
    assert failed.returncode == 2
    path = root / "checkpoint-validation.json"
    previous = path.read_bytes()
    assert json.loads(previous)["error_type"] == "TimeoutError"
    assert scores_path.read_bytes() == scores
    monkeypatch.delenv("FAKE_CHECKPOINT_DOWNLOAD_FAILURE")
    monkeypatch.setenv("FAKE_CHECKPOINT_SCORING_FORBIDDEN", "1")
    resumed = cli(
        "verify-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--retry",
        cwd=tmp_path,
        timeout=90,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    proof = json.loads(path.read_text())
    assert proof["verification_attempt"] == 2
    archive = proof["previous_attempts"][0]
    assert Path(archive["checkpoint-validation.json"]["path"]).read_bytes() == previous
    assert Path(archive["checkpoint-scores.json"]["path"]).read_bytes() == scores
    assert (tmp_path / "sdk-updates.json").read_bytes() == updates
    # A successful proof cannot be replaced or trigger another remote probe.
    complete = path.read_bytes()
    rejected = cli(
        "verify-opsd",
        str(config),
        "--output-dir",
        str(root),
        "--retry",
        cwd=tmp_path,
        timeout=90,
    )
    assert rejected.returncode == 2
    assert path.read_bytes() == complete
