"""Versioned roles through preparation, experiment and training entry points."""

import hashlib
import json
import shlex
import shutil
from pathlib import Path

import pytest
import yaml
from adapters import bash, sample_text, submit, turn_number
from test_cli import cli, replay_config, saved_run
from test_monitoring import external_sdks
from test_opsd import controlled_run
from test_task_sets import prepared_tasks, small_cohort
from test_training import training_config

from bashlet import run_experiment
from bashlet.experiment import task_from_record

CONFIGS = Path(__file__).parents[1] / "configs"
__all__ = ["prepared_tasks"]


def test_new_source_requires_a_versioned_role_even_as_a_standalone_task(
    tmp_path: Path,
) -> None:
    lock = json.loads((CONFIGS / "harness-supplement-v2.lock.json").read_text())
    catalog = json.loads((CONFIGS / lock["file"]).read_text())
    family = catalog["families"][0]
    task = family["variants"][0]["task"] | {
        "task_type": family["task_type"],
        "origin": {
            "dataset": lock["dataset"],
            "revision": lock["revision"],
            "source_sha256": lock["sha256"],
            "original_task_id": family["original_task_id"],
            "family_id": family["family_id"],
            "split": family["split"],
        },
    }
    path = tmp_path / "unassigned.json"
    path.write_text(json.dumps(task))
    model = replay_config(tmp_path, [submit()])
    process = cli("task", str(path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == 2, process.stdout + process.stderr
    assert "versioned role" in process.stderr
    assert not (tmp_path / "runs").exists()


def versioned_cohort(tmp_path: Path) -> tuple[Path, Path]:
    mbpp, path = small_cohort(tmp_path)
    path.write_text(
        json.dumps(
            {
                "version": "harness-task-cohort-v2",
                "name": "role-reuse-test-v2",
                "training_candidates": {
                    "task_ids": ["harness/nested-setting", "harness/license-audit"],
                    "per_type": {"short_program": 0, "code_edit": 1, "workspace": 1},
                },
                "development": {"task_ids": [], "per_type": 0},
                "test": {"task_ids": [], "per_type": 0},
                "environment_variants": [],
            }
        )
    )
    return mbpp, path


def test_old_dev_is_trained_without_changing_its_source_or_historical_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mbpp, cohort = versioned_cohort(tmp_path)
    original_bytes = mbpp.read_bytes()
    output = tmp_path / "versioned"
    process = cli(
        "prepare-harness",
        "--mbpp-manifest",
        str(mbpp),
        "--cohort",
        str(cohort),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    entry = manifest["tasks"][0]
    task = task_from_record(json.loads((output / entry["task_file"]).read_text()))
    assert task.origin is not None and task.origin.split == "dev"
    assert entry["assignment"]["role"] == "train_candidate"
    assert entry["origin"]["family_id"] == "harness/nested-setting"
    run, adapter = controlled_run(tmp_path, monkeypatch)
    adapter.reference_marker = task.reference_implementation
    script = (
        "from pathlib import Path; "
        f"p = Path({task.implementation_paths[0]!r}); "
        "p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text({task.reference_implementation!r})"
    )
    replies = [
        bash("python3 -c " + shlex.quote(script)),
        bash(str(task.feedback_command)),
        submit(),
    ]
    adapter.student.respond = lambda context: sample_text(
        replies[turn_number(context) - 1],
        1024,
    )
    result = run_experiment(
        task=task,
        model=run.student,
        opsd_run=run,
        output_dir=tmp_path / "episodes",
    )
    assert result.passed, result.reason
    assert result.opsd is not None and result.opsd["status"] == "completed", result.opsd
    assert len(adapter.updates) == 1
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["task_origin"]["split"] == "dev"
    assert record["task_assignment"] == entry["assignment"]
    assert mbpp.read_bytes() == original_bytes


@pytest.fixture(scope="module")
def prepared_version(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("versioned-roles")
    output = tmp_path / "roles"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(CONFIGS / "harness-cohort-v2.json"),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return output


def test_three_task_types_and_all_roles_have_real_private_quality_evidence(
    tmp_path: Path,
    prepared_version: Path,
) -> None:
    output = prepared_version
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["model_baseline"] is False
    assert {source["sha256"] for source in manifest["sources"]} == {
        source["sha256"] for source in manifest["source_artifacts"]
    }
    for source in manifest["source_artifacts"]:
        assert (
            hashlib.sha256((output / source["file"]).read_bytes()).hexdigest()
            == source["sha256"]
        )
    assert (
        hashlib.sha256(
            (output / "private/source-registry.json").read_bytes()
        ).hexdigest()
        == manifest["source_registry_sha256"]
    )
    assert manifest["counts"] == {
        "train_candidate": {"short_program": 1, "code_edit": 2, "workspace": 2},
        "dev": {"short_program": 1, "code_edit": 1, "workspace": 1},
        "test": {"short_program": 1, "code_edit": 1, "workspace": 1},
    }
    roles: dict[str, set[str]] = {}
    for entry in manifest["tasks"]:
        role = entry["assignment"]["role"]
        roles.setdefault(role, set()).add(entry["family_id"])
        task = json.loads((output / entry["task_file"]).read_text())
        for name, passed in (("reference", True), ("wrong", False)):
            run = json.loads((output / entry[name + "_run"] / "run.json").read_text())
            assert run["status"] == "completed" and run["grade"]["passed"] is passed
            assert run["task_assignment"] == entry["assignment"]
            assert run["trajectory"][-1]["action"]["kind"] == "bash"
            observations = json.dumps(
                [step["observation"] for step in run["trajectory"]]
            )
            assert task["reference_implementation"] not in observations
            assert not any(
                "_grading" in path for path in run["initial_workspace_manifest"]
            )
        if entry["origin"]["dataset"] == "bashlet/harness-supplement":
            assert entry["origin"]["split"] == "unsplit"
    assert roles["train_candidate"].isdisjoint(roles["dev"] | roles["test"])
    assert roles["dev"].isdisjoint(roles["test"])
    entry = next(e for e in manifest["tasks"] if e["task_type"] == "short_program")
    task = json.loads((output / entry["task_file"]).read_text())
    model = replay_config(tmp_path, [submit()])
    process = cli(
        "task",
        str(output / "manifest.json"),
        "--task-id",
        entry["task_id"],
        "--split",
        "train_candidate",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    _, run = saved_run(tmp_path / "runs")
    assert run["task_assignment"] == entry["assignment"]


@pytest.mark.parametrize("change", ["move_candidate", "drop_candidate", "cohort"])
def test_loading_keeps_the_whole_frozen_candidate_pool_reserved(
    tmp_path: Path,
    prepared_version: Path,
    change: str,
) -> None:
    output = tmp_path / "changed"
    shutil.copytree(prepared_version, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    candidate = manifest["tasks"][0]
    if change == "move_candidate":
        candidate["assignment"]["role"] = candidate["split"] = "dev"
        task_path = output / candidate["task_file"]
        task = json.loads(task_path.read_text())
        task["assignment"] = candidate["assignment"]
        task_path.write_text(json.dumps(task))
        candidate["task_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
        selected, split = candidate["task_id"], "dev"
    else:
        if change == "drop_candidate":
            manifest["tasks"].pop(0)
        else:
            manifest["cohort"]["training_candidates"]["task_ids"].pop(0)
        selected, split = "supplement/interval-union", "dev"
    path.write_text(json.dumps(manifest))
    model = replay_config(tmp_path, [submit()])
    process = cli(
        "task",
        str(path),
        "--task-id",
        selected,
        "--split",
        split,
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    assert not (tmp_path / "runs").exists()


def test_training_cli_rejects_old_dev_even_when_its_candidate_is_not_sampled(
    tmp_path: Path,
    prepared_version: Path,
    prepared_tasks: Path,
) -> None:
    pilot = tmp_path / "pilot.yaml"
    policy = str(CONFIGS / "spend.yaml")
    pilot.write_text(
        yaml.safe_dump(
            {
                "train_manifest": str(prepared_version / "manifest.json"),
                "train_task_ids": ["supplement/sign-runs"],
                "dev_plan": str(prepared_tasks / "harness-dev.json"),
                "condition": [],
                "backend": {"spend_config": policy},
                "training": {"spend_config": policy},
            }
        )
    )
    process = cli(
        "train-opsd",
        str(pilot),
        "--output-dir",
        str(tmp_path / "training"),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    assert "disjoint" in process.stderr
    assert not (tmp_path / "spend.sqlite3").exists()


def test_new_development_cli_records_current_roles_and_original_splits(
    tmp_path: Path,
    prepared_version: Path,
) -> None:
    model = replay_config(tmp_path, [submit(), submit(), submit()])
    process = cli(
        "evaluate",
        str(prepared_version / "harness-dev.json"),
        "--snapshot",
        "replay/prewritten-v1",
        "--evidence-kind",
        "test-double",
        "-c",
        str(model),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    record = json.loads(Path(process.stdout.strip().splitlines()[-1]).read_text())
    assert record["summary"]["planned_tasks"] == 3
    assert all(row["task_origin"]["split"] == "unsplit" for row in record["tasks"])
    assert all(row["task_assignment"]["role"] == "dev" for row in record["tasks"])


def test_old_dev_and_new_source_reach_the_existing_sft_training_cli(
    tmp_path: Path,
    prepared_version: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "encoding.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "ReplayModel",
                    "tokenizer": "Qwen/Qwen3.5-4B",
                    "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                    "tokenizer_cache_dir": str(CONFIGS.parent / "runs/tokenizers"),
                }
            }
        )
    )
    output = tmp_path / "demonstrations"
    process = cli(
        "prepare-sft",
        str(prepared_version / "manifest.json"),
        "--task-id",
        "harness/nested-setting",
        "--task-id",
        "supplement/sign-runs",
        "-c",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    examples = [
        json.loads(line)
        for line in (output / "examples.jsonl").read_text().splitlines()
    ]
    assert [e["metadata"]["origin"]["split"] for e in examples] == ["dev", "unsplit"]
    assert all(
        e["metadata"]["assignment"]["role"] == "train_candidate" for e in examples
    )
    external_sdks(tmp_path, monkeypatch)
    with (tmp_path / "sitecustomize.py").open("a") as stream:
        stream.write("import training_sdk\ntraining_sdk.install()\n")
    monkeypatch.setenv("PYTRIO_API_KEY", "test-trio-key")
    monkeypatch.setenv("FAKE_TRAINING_ROOT", str(tmp_path))
    training = training_config(tmp_path, output / "manifest.json")
    process = cli(
        "train-sft",
        str(training),
        "--output-dir",
        str(tmp_path / "training"),
        cwd=tmp_path,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    requests = [
        json.loads(line)
        for line in (tmp_path / "training-requests.jsonl").read_text().splitlines()
    ]
    assert len(requests) == 2
    assert {json.dumps(request["data"][0], sort_keys=True) for request in requests} == {
        json.dumps(example["datum"], sort_keys=True) for example in examples
    }


def test_old_dev_environment_variant_inherits_the_family_training_role(
    tmp_path: Path,
) -> None:
    mbpp, cohort_path = versioned_cohort(tmp_path)
    cohort = json.loads(cohort_path.read_text())
    cohort["environment_variants"] = ["harness/license-audit/policy-green"]
    cohort_path.write_text(json.dumps(cohort))
    output = tmp_path / "with-variant"
    process = cli(
        "prepare-harness",
        "--mbpp-manifest",
        str(mbpp),
        "--cohort",
        str(cohort_path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["tasks"]) == 3
    assert manifest["counts"]["train_candidate"] == {
        "short_program": 0,
        "code_edit": 1,
        "workspace": 1,
    }
    relatives = [
        e for e in manifest["tasks"] if e["family_id"] == "harness/license-audit"
    ]
    assert len(relatives) == 2
    assert all(e["assignment"]["role"] == "train_candidate" for e in relatives)
    assert all(e["origin"]["split"] == "dev" for e in relatives)
    assert (
        relatives[0]["environment_variant_id"] != relatives[1]["environment_variant_id"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "revision",
        "source_sha256",
        "source_record",
        "family",
        "original_split",
        "known_mbpp_alias",
        "task_content",
        "undeclared_source",
        "source_artifact",
        "source_registry",
        "family_guard",
    ],
)
def test_cli_rejects_unregistered_sources_and_conflicting_or_changed_identities(
    tmp_path: Path,
    prepared_version: Path,
    change: str,
) -> None:
    output = tmp_path / "tampered"
    shutil.copytree(prepared_version, output)
    path = output / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = manifest["tasks"][0]
    selected = "supplement/interval-union"
    split = "dev"
    if change == "source":
        entry["origin"]["dataset"] = "model-author/unregistered"
        entry["author"] = "codex"
    elif change == "revision":
        entry["origin"]["revision"] = "unreviewed-v99"
    elif change == "source_sha256":
        entry["origin"]["source_sha256"] = "0" * 64
    elif change == "source_record":
        entry["source_record_sha256"] = "0" * 64
    elif change == "family":
        entry["family_id"] = entry["assignment"]["family_id"] = "renamed/new-family"
    elif change == "original_split":
        entry["origin"]["split"] = "train"
    elif change == "known_mbpp_alias":
        source = json.loads((CONFIGS / "mbpp-sanitized-v1.json").read_text())
        entry["origin"] = {
            "dataset": source["dataset"],
            "revision": source["revision"],
            "source_sha256": source["sha256"],
            "original_task_id": "625",
            "family_id": "mbpp/625",
            "split": "train",
        }
        entry["family_id"] = entry["assignment"]["family_id"] = "mbpp/625"
    elif change == "task_content":
        selected, split = entry["task_id"], "train_candidate"
        task_path = output / entry["task_file"]
        task = json.loads(task_path.read_text())
        task["grader"]["cases"][0]["expected"] = 999
        task_path.write_text(json.dumps(task))
        entry["task_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    elif change in ("source_artifact", "source_registry", "family_guard"):
        target = (
            output / manifest["source_artifacts"][0]["file"]
            if change == "source_artifact"
            else output
            / "private"
            / (
                "source-registry.json"
                if change == "source_registry"
                else "family-guard.json"
            )
        )
        target.write_bytes(target.read_bytes() + b"\n")
    else:
        manifest["sources"] = []
    path.write_text(json.dumps(manifest))
    model = replay_config(tmp_path, [submit()])
    process = cli(
        "task",
        str(path),
        "--task-id",
        selected,
        "--split",
        split,
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    if change == "known_mbpp_alias":
        assert "family mapping" in process.stderr
    assert not (tmp_path / "runs").exists()
