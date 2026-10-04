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


def test_two_complementary_demonstrations_keep_one_task_and_family(
    tmp_path: Path,
) -> None:
    manifest = executed_manifest(tmp_path / "tasks")
    encoding = QwenChatMLEncoding(
        "Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        cache_dir="runs/tokenizers",
    )
    trajectories = []
    for index, replies in enumerate(
        (
            [REPAIR, SUBMIT],
            [
                "```bash\npython3 feedback.py\n```",
                REPAIR,
                "```bash\npython3 feedback.py\n```",
                SUBMIT,
            ],
        )
    ):
        result = run_experiment(
            task=executed_task(),
            model=SampledTrajectory(encoding, replies),
            output_dir=tmp_path / "episodes",
        )
        assert result.passed
        trajectories.append(
            {
                "demonstration_id": f"example-{index}",
                "task_id": result.task_id,
                "run_dir": str(result.run_dir),
                "run_sha256": hashlib.sha256(
                    (result.run_dir / "run.json").read_bytes()
                ).hexdigest(),
                "complementary_reason": (
                    "Diagnoses the initial bug with real failing feedback, then "
                    "repairs and verifies it."
                    if index
                    else ""
                ),
                "actions": [
                    {
                        "supervise": True,
                        "reason": "Useful task action",
                        "author": "model",
                    }
                    for _ in replies
                ],
            }
        )
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"schema_version": 2, "trajectories": trajectories}))
    process = cli(
        "prepare-sft",
        str(manifest),
        "--trajectories",
        str(review),
        "-c",
        "configs/executed-trajectories.yaml",
        "--output-dir",
        str(tmp_path / "data"),
        cwd=Path(__file__).parents[1],
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    data, datums = load_demonstrations(tmp_path / "data/manifest.json")
    assert len(datums) == data["counts"]["accepted"] == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "data/examples.jsonl").read_text().splitlines()
    ]
    assert [row["metadata"]["demonstration_id"] for row in rows] == [
        "example-0",
        "example-1",
    ]
    assert {row["metadata"]["task_id"] for row in rows} == {executed_task().task_id}
    assert {row["metadata"]["origin"]["family_id"] for row in rows} == {
        "harness/header-key"
    }


def balanced_plan(root: Path) -> Path:
    path = collection_plan(root)
    plan = json.loads(path.read_text())
    task_id = plan["tasks"][0]["task_id"]
    source = root / "authored.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "attempts": [
                    {"task_id": task_id, "replies": ["invalid"]},
                    {"task_id": task_id, "replies": ["invalid", REPAIR, SUBMIT]},
                    {
                        "task_id": task_id,
                        "replies": [
                            '```bash\npython3 -c "from app.headers '
                            "import normalize_header; "
                            "assert normalize_header(' X_ID ') == 'x-id'\"\n```",
                            REPAIR,
                            "```bash\npython3 feedback.py\n```",
                            SUBMIT,
                        ],
                    },
                ],
            }
        )
    )
    plan.update(
        schema_version=2,
        max_attempts=3,
        max_accepted=2,
        max_attempts_per_family=3,
        max_accepted_per_family=2,
        max_elapsed_seconds=120,
        gpu_seconds=0,
        paid_cny=0,
        sources=[
            {
                "name": "reference-assisted",
                "kind": "authored",
                "author": "codex",
                "policy": "codex/reference-assisted-v1",
                "generation_method": "reference-assisted construction",
                "artifact": str(source),
                "artifact_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ],
    )
    path.write_text(json.dumps(plan))
    return path


def test_bounded_reference_collection_preserves_failed_attempts_and_stops(
    tmp_path: Path,
) -> None:
    plan = balanced_plan(tmp_path)
    output = tmp_path / "collection"
    process = cli(
        "collect-demonstrations",
        str(plan),
        "-c",
        "configs/executed-trajectories.yaml",
        "--output-dir",
        str(output),
        cwd=Path(__file__).parents[1],
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    collection = json.loads((output / "collection.json").read_text())
    attempts = collection["attempts"]
    assert len(attempts) == 3
    assert [item["passed"] for item in attempts] == [False, True, True]
    assert len({item["attempt_id"] for item in attempts}) == 3
    assert {item["task_id"] for item in attempts} == {executed_task().task_id}
    assert {item["family_id"] for item in attempts} == {"harness/header-key"}
    assert collection["stop_reason"] == "attempt_limit"
    for item in attempts:
        run = json.loads((output / item["run_dir"] / "run.json").read_text())
        assert run["policy"] == "codex/reference-assisted-v1"
        assert all(
            step["model_call"]["generation"]["source"] == "replayed"
            and step["model_call"]["generation"]["logprobs"] is None
            for step in run["trajectory"]
        )
        encoding = QwenChatMLEncoding(
            "Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            cache_dir="runs/tokenizers",
        )
        prompt = encoding.decode(
            run["context_tokens"][
                : run["trajectory"][0]["model_call"]["context_offset"]
            ]
        )
        assert "return text.strip().lower().replace('_', '-')" not in prompt


@pytest.mark.parametrize(
    "fault", ["none", "source", "no_complement", "repeat", "repeat_masked", "timestamp"]
)
def test_freeze_balanced_collection_counts_demonstrations_and_families_separately(
    tmp_path: Path,
    fault: str,
) -> None:
    plan = balanced_plan(tmp_path)
    if fault in ("repeat", "repeat_masked", "timestamp"):
        data = json.loads(plan.read_text())
        source_path = Path(data["sources"][0]["artifact"])
        source = json.loads(source_path.read_text())
        if fault == "repeat":
            source["attempts"][2]["replies"] = source["attempts"][1]["replies"]
        elif fault == "repeat_masked":
            source["attempts"][2]["replies"] = source["attempts"][1]["replies"][1:]
        else:
            source["attempts"][2]["replies"].insert(0, "```bash\ndate +%s%N\n```")
        source_path.write_text(json.dumps(source))
        data["sources"][0]["artifact_sha256"] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
        plan.write_text(json.dumps(data))
    output = tmp_path / "collection"
    collected = cli(
        "collect-demonstrations",
        str(plan),
        "-c",
        "configs/executed-trajectories.yaml",
        "--output-dir",
        str(output),
        cwd=Path(__file__).parents[1],
        timeout=90,
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    path = output / "collection.json"
    collection = json.loads(path.read_text())
    if fault == "source":
        collection["attempts"][1]["provenance"]["name"] = "unregistered-author"
        path.write_text(json.dumps(collection))
    decisions = []
    for attempt in collection["attempts"]:
        run = json.loads((output / attempt["run_dir"] / "run.json").read_text())
        decisions.append(
            {
                "attempt_id": attempt["attempt_id"],
                "accept": attempt["passed"],
                "reason": "Useful executed actions"
                if attempt["passed"]
                else "No submission",
                "complementary_reason": "Adds real diagnosis and recovery"
                if attempt["attempt"] == 3 and fault != "no_complement"
                else "",
                "recovery": {
                    "failure_step": 0,
                    "repair_step": 1,
                    "verification_step": 2,
                    "reason": "The real public-requirement check fails on the initial "
                    "implementation; the repair then passes unchanged feedback.",
                }
                if attempt["attempt"] == 3 and fault == "none"
                else None,
                "actions": [
                    {
                        "supervise": attempt["passed"] and "command" in step["action"],
                        "reason": "Actual task action",
                        "author": "codex",
                        "action_type": "task_action",
                    }
                    for step in run["trajectory"]
                ],
            }
        )
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "collection_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "attempts": decisions,
            }
        )
    )
    frozen = tmp_path / "frozen"
    process = cli(
        "freeze-demonstrations",
        str(path),
        "--review",
        str(review),
        "-c",
        "configs/executed-trajectories.yaml",
        "--output-dir",
        str(frozen),
        cwd=Path(__file__).parents[1],
        timeout=90,
    )
    if fault == "source":
        assert process.returncode != 0
        assert "source" in process.stdout + process.stderr
        assert not frozen.exists()
        return
    assert process.returncode == 0, process.stdout + process.stderr
    _, datums = load_demonstrations(frozen / "data/manifest.json")
    assert len(datums) == (2 if fault == "none" else 1)
    report = json.loads((frozen / "report.json").read_text())
    assert report["counts"]["collection_attempts"] == 3
    assert report["counts"]["quality_candidates"] == 2
    assert report["counts"]["accepted_trajectories"] == len(datums)
    assert report["counts"]["accepted_families"] == 1
    assert report["counts"]["base_sampled_attempts"] == 0
    assert report["counts"]["reference_constructed_attempts"] == 3
    assert report["counts"]["recovery_trajectories"] == (1 if fault == "none" else 0)
    assert report["counts"]["replay_attempts"] == (
        1 if fault in ("repeat", "repeat_masked", "no_complement") else 2
    )
    assert report["tokens"]["supervised"] > 0
    assert report["tokens"]["masked_generated"] > 0
    assert report["per_type"]["code_edit"]["accepted_families"] == 1
    assert report["rejected"][0]["reason"] == "No submission"
    if fault != "none":
        assert len(report["rejected"]) == 2
        message = {
            "repeat": "Meaningless",
            "repeat_masked": "Meaningless",
            "no_complement": "complementary",
            "timestamp": "observation",
        }[fault]
        assert message in report["rejected"][1]["reason"]
