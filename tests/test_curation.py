"""Bounded collection and frozen training inputs at the CLI/loader boundaries."""

import hashlib
import json
from pathlib import Path

import pytest
from test_cli import cli
from test_demonstrations import (
    REPAIR,
    SUBMIT,
    SampledTrajectory,
    executed_manifest,
    executed_task,
)

from bashlet import QwenChatMLEncoding, run_experiment
from bashlet.config import config_record, load_config
from bashlet.training import load_demonstrations


def collection_plan(root: Path) -> Path:
    manifest = executed_manifest(root / "tasks")
    source = json.loads(manifest.read_text())
    entry = source["tasks"][0]
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "max_attempts": 1,
                "max_accepted": 1,
                "max_corrections": 1,
                "max_context_tokens": 8192,
                "target_per_type": 1,
                "task_manifest": str(manifest),
                "task_manifest_sha256": hashlib.sha256(
                    manifest.read_bytes()
                ).hexdigest(),
                "expected_policy": "test/base-model@fixed-weights",
                "quality_rules": ["Keep useful real actions; mask errors."],
                "tasks": [
                    {
                        key: entry[key]
                        for key in ("task_id", "family_id", "task_type", "task_sha256")
                    }
                ],
            }
        )
    )
    return plan


def test_collection_requires_a_finite_positive_attempt_limit(tmp_path: Path) -> None:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema_version": 1, "max_attempts": 0}))
    result = cli(
        "collect-demonstrations",
        str(plan),
        "--output-dir",
        str(tmp_path / "out"),
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode != 0
    assert "max_attempts" in result.stderr + result.stdout
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("fault", ["none", "changed_disposition", "stale_review"])
def test_freeze_loads_reviewed_actions_and_reports_masked_errors(
    tmp_path: Path,
    fault: str,
) -> None:
    plan_path = collection_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    collection = tmp_path / "collection"
    collection.mkdir()
    (collection / "plan.json").write_bytes(plan_path.read_bytes())
    encoding = QwenChatMLEncoding(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cache_dir="runs/tokenizers",
    )
    result = run_experiment(
        task=executed_task(),
        model=SampledTrajectory(encoding, ["no executable action", REPAIR, SUBMIT]),
        output_dir=collection / "episodes",
    )
    assert result.passed
    run_hash = hashlib.sha256((result.run_dir / "run.json").read_bytes()).hexdigest()
    (collection / "collection.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "config": config_record(load_config()),
                "attempts": [
                    dict(
                        plan["tasks"][0],
                        attempt=1,
                        status=result.status,
                        passed=fault != "changed_disposition",
                        run_dir=str(result.run_dir.relative_to(collection)),
                        run_sha256=run_hash,
                    )
                ],
            }
        )
    )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "collection_sha256": (
                    "stale"
                    if fault == "stale_review"
                    else hashlib.sha256(
                        (collection / "collection.json").read_bytes()
                    ).hexdigest()
                ),
                "attempts": [
                    {
                        "attempt": 1,
                        "accept": True,
                        "reason": "Useful repair and submission",
                        "actions": [
                            {
                                "supervise": False,
                                "reason": "Protocol error",
                                "author": "model",
                                "action_type": "invalid",
                            },
                            {
                                "supervise": True,
                                "reason": "Repair normalization",
                                "author": "model",
                                "action_type": "repair",
                            },
                            {
                                "supervise": True,
                                "reason": "Submit correct files",
                                "author": "model",
                                "action_type": "submit",
                            },
                        ],
                    }
                ],
            }
        )
    )
    output = tmp_path / "frozen"
    frozen = cli(
        "freeze-demonstrations",
        str(collection / "collection.json"),
        "--review",
        str(review),
        "-c",
        "configs/executed-trajectories.yaml",
        "--output-dir",
        str(output),
        cwd=Path(__file__).parents[1],
    )
    if fault != "none":
        assert frozen.returncode != 0
        message = (
            "collection and review" if fault == "stale_review" else "sampled provenance"
        )
        assert message in frozen.stdout + frozen.stderr
        assert not output.exists()
        return
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    manifest, datums = load_demonstrations(output / "data/manifest.json")
    assert len(datums) == manifest["counts"]["accepted"] == 1
    report = json.loads((output / "report.json").read_text())
    assert report["counts"]["base_sampled_attempts"] == 1
    assert report["counts"]["accepted_families"] == 1
    assert report["counts"]["replay_attempts"] == 1
    assert report["counts"]["codex_corrected_trajectories"] == 0
    assert report["action_types"]["invalid"]["supervised_tokens"] == 0
    assert report["action_types"]["repair"]["supervised_tokens"] > 0
    assert report["tokens"]["masked_generated"] > 0
    assert report["on_policy_eligible"] is False
    assert (
        hashlib.sha256((result.run_dir / "run.json").read_bytes()).hexdigest()
        == run_hash
    )


def test_collection_rejects_changed_task_identity_before_loading_model(
    tmp_path: Path,
) -> None:
    plan = collection_plan(tmp_path)
    data = json.loads(plan.read_text())
    data["tasks"][0]["family_id"] = "harness/nested-setting"
    plan.write_text(json.dumps(data))
    result = cli(
        "collect-demonstrations",
        str(plan),
        "--output-dir",
        str(tmp_path / "out"),
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode != 0
    assert "frozen training family" in result.stderr + result.stdout
    assert not (tmp_path / "out").exists()


def test_collection_never_admits_a_development_family(tmp_path: Path) -> None:
    plan_path = collection_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    manifest_path = Path(plan["task_manifest"])
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["tasks"][0]
    # A registered development origin must fail even in a caller-supplied plan.
    entry["task_id"] = "harness/nested-setting"
    entry["family_id"] = "harness/nested-setting"
    entry["split"] = "dev"
    entry["origin"].update(
        original_task_id="nested-setting",
        family_id="harness/nested-setting",
        split="dev",
    )
    manifest_path.write_text(json.dumps(manifest))
    plan["task_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    plan["tasks"] = [{key: entry[key] for key in plan["tasks"][0]}]
    plan_path.write_text(json.dumps(plan))
    result = cli(
        "collect-demonstrations",
        str(plan_path),
        "--output-dir",
        str(tmp_path / "out"),
        cwd=Path(__file__).parents[1],
    )
    assert result.returncode != 0
    assert "belongs to split dev, not train" in result.stderr + result.stdout
    assert not (tmp_path / "out").exists()
