"""Raw-model OPSD through the established experiment and CLI seams."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from adapters import ByteEncoding, ControlledOPSD
from test_cli import cli
from test_opsd import controlled_run, repair_task

from bashlet import run_experiment
from bashlet.distillation import InitialSnapshot, OPSDRun


def test_real_sdk_adapter_updates_and_refreshes_from_the_same_original_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opsd_sdk import Service

    from bashlet.distillation.pytrio import PyTRIOOPSD, PyTRIOOPSDConfig

    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    old, _ = controlled_run(tmp_path, monkeypatch)
    old.path.unlink()
    service = Service()
    adapter = PyTRIOOPSD(
        PyTRIOOPSDConfig(),
        journal=old.journal,
        output_dir=old.root,
        service_client=service,
        encoding=ByteEncoding(),
    )
    run = OPSDRun(
        old.root,
        snapshot=adapter.snapshot,
        adapter=adapter,
        journal=old.journal,
        monitoring=old.monitoring,
        config=old.config,
    )
    adapter.bind_run(run.record["run_id"])
    before = adapter.teacher_identity
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed and result.opsd is not None
    assert result.opsd["status"] == "completed", result.opsd
    update = result.opsd["update"]
    assert isinstance(update, dict) and update["loss"] == 0.125
    assert result.policy == before and run.student.identity != before
    assert adapter.teacher_identity == before
    assert service.trainer is not None
    assert service.trainer.state["optimizer_step"] == 1
    assert service.trainer.requests[0]["loss_fn"] == "importance_sampling"
    assert service.trainer.requests[0]["auto_shift"] is False
    assert (
        len(
            [
                e
                for e in run.journal.snapshot()["entries"]
                if e["operation"] == "sampling"
            ]
        )
        == 2
    )


def test_original_model_is_an_explicit_start_and_restores_only_the_student(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = ControlledOPSD()
    adapter.teacher_identity = "test/original@frozen"
    adapter.student.identity = adapter.teacher_identity
    old, _ = controlled_run(tmp_path, monkeypatch)
    old.path.unlink()
    snapshot = InitialSnapshot(
        adapter.teacher_identity,
        ByteEncoding().identity,
        "test-double",
        stage="base",
        kind="base_model",
    )
    run = OPSDRun(
        old.root,
        snapshot=snapshot,
        adapter=adapter,
        journal=old.journal,
        config=old.config,
        monitoring=old.monitoring,
    )
    result = run_experiment(
        task=repair_task(),
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.opsd is not None and result.opsd["status"] == "completed"
    record = json.loads(run.path.read_text())
    assert record["snapshot"]["stage"] == "base"
    assert record["teacher_policy"] == "test/original@frozen"
    restored = ControlledOPSD()
    restored.teacher_identity = snapshot.identity
    restored.student.identity = snapshot.identity
    resumed = OPSDRun(
        run.root,
        snapshot=snapshot,
        adapter=restored,
        journal=run.journal,
        config=run.config,
        monitoring=run.monitoring,
        resume=True,
    )
    assert resumed.record["optimizer_step"] == 1
    assert resumed.student.identity == record["student_policy"]
    assert restored.teacher_identity == snapshot.identity
    with pytest.raises(ValueError, match="training state"):
        replace(snapshot, kind="training_state")


@pytest.mark.parametrize("interrupt", [False, True])
def test_cli_pilot_compares_normal_inputs_and_resume_keeps_steps_and_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: bool,
) -> None:
    from test_mbpp import double_row, prepare
    from test_monitoring import external_sdks

    prepared = prepare(
        tmp_path, [double_row(602), double_row(603), double_row(511), double_row(512)]
    )
    dev = tmp_path / "dev.json"
    dev.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "fixed-pilot",
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
                "train_task_ids": ["mbpp/602"],
                "dev_plan": str(dev),
                "condition": [str(condition)],
                "evaluation": {
                    "base_path": str(tokenizer),
                    "model": {"base_url": "http://test-double"},
                },
                "journal_path": str(tmp_path / "requests.sqlite3"),
                "backend": {
                    "tokenizer_cache_dir": str(tokenizer.parents[2]),
                    "group": "opsd-pilot",
                },
                "monitoring": {"group": "opsd-pilot"},
                "training": {"group": "opsd-pilot"},
            }
        )
    )
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import opsd_sdk\nopsd_sdk.install()\n")
    monkeypatch.setenv("pytrio_api_key", "test-trio-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    monkeypatch.setenv("FAKE_OPSD_TOKENIZER", str(tokenizer))
    root = tmp_path / "pilot-run"
    if interrupt:
        monkeypatch.setenv("FAKE_OPSD_INTERRUPT_EVAL", "1")
    process = cli(
        "train-opsd", str(pilot), "--output-dir", str(root), cwd=tmp_path, timeout=60
    )
    if interrupt:
        assert process.returncode == 2, process.stdout + process.stderr
        partial = json.loads(
            (root / "evaluation/baseline/seed-4/evaluation.json").read_text()
        )
        assert [row["outcome"] for row in partial["tasks"]] == ["passed", "not_started"]
        first_hash = partial["tasks"][0]["episode_sha256"]
        first_run = partial["tasks"][0]["run"]
        monkeypatch.delenv("FAKE_OPSD_INTERRUPT_EVAL")
        process = cli(
            "train-opsd",
            str(pilot),
            "--output-dir",
            str(root),
            "--resume",
            cwd=tmp_path,
            timeout=60,
        )
        recovered = json.loads(
            (root / "evaluation/baseline/seed-4/evaluation.json").read_text()
        )
        assert recovered["tasks"][0]["episode_sha256"] == first_hash
        assert recovered["tasks"][0]["run"] == first_run
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == 1
    assert record["comparison"]["matched_complete"] is True
    assert record["comparison"]["baseline"]["passed_tasks"] == 2
    assert record["comparison"]["updated"]["passed_tasks"] == 2
    assert record["decision"]["action"] == "review_results"
    before = json.loads(
        (root / "evaluation/baseline/seed-4/evaluation.json").read_text()
    )
    after = json.loads((root / "evaluation/final/seed-4/evaluation.json").read_text())
    assert before["comparison_sha256"] == after["comparison_sha256"]
    assert before["snapshot"] != after["snapshot"]
    for phase in ("evaluation/baseline/seed-4", "evaluation/final/seed-4"):
        for path in (root / phase / "episodes").glob("*/run.json"):
            episode = json.loads(path.read_text())
            from bashlet.backends.encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(str(tokenizer), revision=tokenizer.name)
            assert "Reference implementation (teacher only)" not in encoding.decode(
                episode["context_tokens"]
            )
    requests = record["requests"]["request_ids"]
    resumed = cli(
        "train-opsd", str(pilot), "--output-dir", str(root), "--resume", cwd=tmp_path
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (
        json.loads((root / "pilot.json").read_text())["requests"]["request_ids"]
        == requests
    )
    verified = cli(
        "verify-opsd", str(pilot), "--output-dir", str(root), cwd=tmp_path, timeout=60
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    proof = json.loads((root / "checkpoint-validation.json").read_text())
    assert proof["training_state"]["optimizer_step"] == 1
    assert proof["training_state"]["completed_state"].endswith("restored-state-0001")
    assert proof["frozen_teacher_unchanged"] and proof["student_parameters_changed"]
    assert proof["teacher_policy"] == before["snapshot"]
    assert proof["student_policy"] == after["snapshot"]
    assert proof["requests"]["request_ids"] == requests
    # Simulate interruption after the evaluation finished, before its parent
    # persisted the returned path. Existing episodes must not run again.
    interrupted = json.loads((root / "pilot.json").read_text())
    interrupted.update(status="incomplete", evaluation=None)
    (root / "pilot.json").write_text(json.dumps(interrupted))
    repaired = cli(
        "train-opsd",
        str(pilot),
        "--output-dir",
        str(root),
        "--resume",
        cwd=tmp_path,
        timeout=60,
    )
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    complete = json.loads((root / "pilot.json").read_text())
    assert complete["optimizer_step"] == 1
    assert complete["comparison"]["matched_complete"] is True
    assert complete["requests"]["request_ids"] == proof["requests"]["request_ids"]
