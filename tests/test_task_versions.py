"""Versioned roles through preparation, experiment and training entry points."""

import hashlib
import json
import shlex
import shutil
from pathlib import Path

import pytest
import yaml
from adapters import bash, sample_text, submit, turn_number
from task_fixtures import role_cohort, role_sources
from test_cli import cli, replay_config, saved_run
from test_opsd import controlled_run
from test_task_sets import prepared_tasks, small_cohort

from bashlet import run_experiment
from bashlet.harness.experiment import task_from_record

CONFIGS = Path(__file__).parents[1] / "configs"
__all__ = ["prepared_tasks", "role_sources"]


def test_new_source_requires_a_versioned_role_even_as_a_standalone_task(
    tmp_path: Path,
    role_sources: Path,
) -> None:
    lock = json.loads((role_sources / "sources.json").read_text())["locks"]["roles-v1"]
    catalog = json.loads((role_sources / lock["file"]).read_text())
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


@pytest.mark.parametrize("change", ["role", "version"])
def test_standalone_task_cannot_reassign_a_frozen_test_family(
    tmp_path: Path,
    prepared_version: Path,
    change: str,
) -> None:
    manifest = json.loads((prepared_version / "manifest.json").read_text())
    entry = next(
        e for e in manifest["tasks"] if e["task_id"] == "fixture/circular-windows"
    )
    task = json.loads((prepared_version / entry["task_file"]).read_text())
    task["assignment"]["role"] = "train_candidate"
    if change == "version":
        task["assignment"]["version"] = "unregistered-fake-version"
        task["assignment"]["cohort_sha256"] = "0" * 64
    path = tmp_path / "forged-task.json"
    path.write_text(json.dumps(task))
    model = replay_config(tmp_path, [submit()])
    process = cli("task", str(path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == 2, process.stdout + process.stderr
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
def prepared_version(
    tmp_path_factory: pytest.TempPathFactory, role_sources: Path
) -> Path:
    tmp_path = tmp_path_factory.mktemp("versioned-roles")
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps(role_cohort()))
    output = tmp_path / "roles"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(cohort),
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
    registry = json.loads((output / "private/source-registry.json").read_text())
    for source in registry["sources"]:
        if source["kind"] == "model_assisted":
            assert source["generator"]["model"] == "unknown"
            assert source["generator"]["model_revision"] is None
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
        if entry["origin"]["dataset"] == "test/role-fixture":
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
        selected, split = "fixture/interval-union", "dev"
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


@pytest.mark.parametrize(
    "code",
    [
        "def read_cache(entries, key, now, default=None):\n"
        "    item = entries.pop(key, None)\n"
        "    if item is None: return default\n"
        "    value, expiry = item\n"
        "    return value if now < expiry else default\n",
        "def read_cache(entries, key, now, default=None):\n"
        "    if key not in entries: return default\n"
        "    value, expiry = entries[key]\n"
        "    if isinstance(value, bool): return int(value)\n"
        "    return value if now < expiry else default\n",
        "def read_cache(entries, key, now, default=None):\n"
        "    if key not in entries: return default\n"
        "    value, expiry = entries[key]\n"
        "    return value if now <= expiry else default\n",
    ],
)
def test_corrected_cache_grading_rejects_mutation_false_coercion_and_expiry(
    tmp_path: Path,
    prepared_version: Path,
    code: str,
) -> None:
    script = (
        "from pathlib import Path; Path('app/cache.py').write_text(" + repr(code) + ")"
    )
    model = replay_config(
        tmp_path,
        [
            bash("python3 -c " + shlex.quote(script)),
            bash("python3 feedback.py"),
            submit(),
        ],
    )
    process = cli(
        "task",
        str(prepared_version / "manifest.json"),
        "--task-id",
        "fixture/cache-expiry/v2",
        "--split",
        "dev",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["trajectory"][1]["observation"]["status"] == "ok"
    assert record["grading_protocol"] == "test-command-v1"
    assert record["grade"]["passed"] is False


@pytest.mark.parametrize("same_role", [False, True])
def test_source_revisions_cannot_add_an_independent_family_or_cross_roles(
    tmp_path: Path,
    role_sources: Path,
    same_role: bool,
) -> None:
    cohort = role_cohort()
    if same_role:
        cohort["development"]["task_ids"].append("fixture/cache-expiry")
        cohort["development"]["per_type"] = {
            "short_program": 1,
            "code_edit": 2,
            "workspace": 1,
        }
    else:
        cohort["training_candidates"]["task_ids"].append("fixture/cache-expiry")
        cohort["training_candidates"]["per_type"]["code_edit"] += 1
    path = tmp_path / "conflicting-revisions.json"
    path.write_text(json.dumps(cohort))
    output = tmp_path / "conflicting"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    expected = (
        "independent base task families" if same_role else "family split conflict"
    )
    assert expected in process.stderr
    assert not output.exists()


def test_training_cli_rejects_old_dev_even_when_its_candidate_is_not_sampled(
    tmp_path: Path,
    prepared_version: Path,
    prepared_tasks: Path,
) -> None:
    pilot = tmp_path / "pilot.yaml"
    pilot.write_text(
        yaml.safe_dump(
            {
                "train_manifest": str(prepared_version / "manifest.json"),
                "train_task_ids": ["fixture/sign-runs"],
                "dev_plan": str(prepared_tasks / "harness-dev.json"),
                "condition": [],
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
    assert not (tmp_path / "requests.sqlite3").exists()


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
    selected = "fixture/interval-union"
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
        from bashlet.tasks.sources import mbpp_source_lock

        source = mbpp_source_lock(CONFIGS.parent / "data/sources.json")
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
