"""Raw-model OPSD through the established experiment and CLI seams."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from adapters import ByteEncoding, ControlledOPSD
from test_cli import cli
from test_opsd import controlled_run, repair_task

from bashlet import SpendLedger, SpendPolicy, run_experiment
from bashlet.distillation import InitialSnapshot, OPSDRun
from bashlet.spend import BudgetRefused


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
        ledger=old.ledger,
        output_dir=old.root,
        service_client=service,
        encoding=ByteEncoding(),
    )
    run = OPSDRun(
        old.root,
        snapshot=adapter.snapshot,
        adapter=adapter,
        ledger=old.ledger,
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
                for e in run.ledger.snapshot()["entries"]
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
        ledger=old.ledger,
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
        ledger=run.ledger,
        config=run.config,
        monitoring=run.monitoring,
        resume=True,
    )
    assert resumed.record["optimizer_step"] == 1
    assert resumed.student.identity == record["student_policy"]
    assert restored.teacher_identity == snapshot.identity
    with pytest.raises(ValueError, match="training state"):
        replace(snapshot, kind="training_state")


def test_project_bill_releases_total_ceiling_without_inventing_request_bills(
    tmp_path: Path,
) -> None:
    policy = SpendPolicy()
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=policy)
    entry = ledger.reserve(
        operation="sampling",
        group="opsd-pilot",
        payload={"old": True},
        upper_cny=9.4,
    )
    ledger.complete(entry)
    evidence = tmp_path / "bill.json"
    evidence.write_text(json.dumps({"source": "user", "statement": "total paid 1"}))
    policy_path = tmp_path / "spend.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "limits": {
                    "funded_cny": 5,
                    "regular_cny": 150,
                    "reserve_cny": 50,
                    "project_cny": 200,
                    "groups": policy.groups,
                }
            }
        )
    )
    command = (
        "--ledger",
        str(ledger.path),
        "--policy",
        str(policy_path),
    )
    reconciled = cli(
        "spend",
        "reconcile-total",
        "--actual-cny",
        "1",
        "--group-actual-cny",
        "opsd-pilot=1",
        "--evidence",
        str(evidence),
        *command,
        cwd=tmp_path,
    )
    assert reconciled.returncode == 0, reconciled.stderr
    record = json.loads(reconciled.stdout)
    assert record["charged_cny"] == 1
    assert record["funded_remaining_cny"] == 4
    assert record["entries"][0]["billing_basis"] == "upper_bound"
    assert record["project_actual_cny"] == 1
    assert record["project_bill"]["groups"]["opsd-pilot"]["actual_cny"] == 1
    reopened = cli("spend", "status", *command, cwd=tmp_path)
    assert json.loads(reopened.stdout)["charged_cny"] == 1


def test_running_workflow_obeys_a_cash_limit_tightened_by_another_process(
    tmp_path: Path,
) -> None:
    ledger = SpendLedger(tmp_path / "spend.sqlite3", policy=SpendPolicy())
    policy_path = tmp_path / "spend.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "limits": {
                    "funded_cny": 5,
                    "regular_cny": 150,
                    "reserve_cny": 50,
                    "project_cny": 200,
                    "groups": ledger.policy.groups,
                }
            }
        )
    )
    tightened = cli(
        "spend",
        "status",
        "--ledger",
        str(ledger.path),
        "--policy",
        str(policy_path),
        cwd=tmp_path,
    )
    assert tightened.returncode == 0, tightened.stderr
    with pytest.raises(BudgetRefused, match="funded"):
        ledger.reserve(
            operation="sampling", group="opsd-pilot", payload={"new": True}, upper_cny=6
        )


@pytest.mark.parametrize("interrupt", [False, True])
def test_cli_pilot_compares_normal_inputs_and_resuming_keeps_steps_and_cash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: bool,
) -> None:
    from test_mbpp import double_row, prepare
    from test_monitoring import external_sdks

    prepared = prepare(tmp_path, [double_row(602), double_row(511), double_row(512)])
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
                "ledger_path": str(tmp_path / "spend.sqlite3"),
                "backend": {
                    "tokenizer_cache_dir": str(tokenizer.parents[2]),
                    "spend_config": str(Path("configs/spend.yaml").resolve()),
                },
                "training": {"spend_config": str(Path("configs/spend.yaml").resolve())},
                "monitoring": {"group": "opsd-pilot"},
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
        partial = json.loads((root / "baseline/evaluation.json").read_text())
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
        recovered = json.loads((root / "baseline/evaluation.json").read_text())
        assert recovered["tasks"][0]["episode_sha256"] == first_hash
        assert recovered["tasks"][0]["run"] == first_run
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads((root / "pilot.json").read_text())
    assert record["optimizer_step"] == 1
    assert record["comparison"]["matched_complete"] is True
    assert record["comparison"]["baseline"]["passed_tasks"] == 2
    assert record["comparison"]["updated"]["passed_tasks"] == 2
    assert record["decision"]["action"] == "stop_investment"
    before = json.loads((root / "baseline/evaluation.json").read_text())
    after = json.loads((root / "updated/evaluation.json").read_text())
    assert before["comparison_sha256"] == after["comparison_sha256"]
    assert before["snapshot"] != after["snapshot"]
    for phase in ("baseline", "updated"):
        for path in (root / phase / "episodes").glob("*/run.json"):
            episode = json.loads(path.read_text())
            from bashlet.encoding import QwenChatMLEncoding

            encoding = QwenChatMLEncoding(str(tokenizer), revision=tokenizer.name)
            assert "Reference implementation (teacher only)" not in encoding.decode(
                episode["context_tokens"]
            )
    cost = record["cost"]["charged_upper_bound_cny"]
    resumed = cli(
        "train-opsd", str(pilot), "--output-dir", str(root), "--resume", cwd=tmp_path
    )
    assert resumed.returncode == 0, resumed.stderr
    assert (
        json.loads((root / "pilot.json").read_text())["cost"]["charged_upper_bound_cny"]
        == cost
    )
    verified = cli(
        "verify-opsd", str(pilot), "--output-dir", str(root), cwd=tmp_path, timeout=60
    )
    assert verified.returncode == 0, verified.stdout + verified.stderr
    proof = json.loads((root / "checkpoint-validation.json").read_text())
    assert proof["training_state"]["optimizer_step"] == 1
    assert proof["training_state"]["completed_state"].endswith("restored-state-0001")
    assert proof["frozen_teacher_unchanged"] and proof["student_probabilities_changed"]
    assert proof["teacher_policy"] == before["snapshot"]
    assert proof["student_policy"] == after["snapshot"]
    assert proof["cost"]["charged_upper_bound_cny"] > cost
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
    assert (
        complete["cost"]["charged_upper_bound_cny"]
        == proof["cost"]["charged_upper_bound_cny"]
    )

    selection = tmp_path / "ineligible.json"
    selection.write_text(json.dumps({"status": "no_qualified_snapshot"}))
    candidate = yaml.safe_load(pilot.read_text())
    candidate["backend"].update(
        source="sft",
        initial_sampler=after["snapshot"].split("@", 1)[1],
        initial_state=proof["training_state"]["training_state"],
        initial_training_record=str(root / "training/backend.json"),
        initial_selection=str(selection),
    )
    sft = tmp_path / "ineligible-sft.yaml"
    sft.write_text(yaml.safe_dump(candidate))
    refused = cli(
        "train-opsd",
        str(sft),
        "--output-dir",
        str(tmp_path / "refused"),
        cwd=tmp_path,
        timeout=60,
    )
    assert refused.returncode == 2, refused.stdout + refused.stderr
    rejected = json.loads((tmp_path / "refused/pilot.json").read_text())
    assert "qualified real selection" in rejected["reason"]
    assert rejected["optimizer_step"] == 0
