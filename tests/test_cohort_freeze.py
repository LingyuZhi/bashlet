"""Fixed stratified cohorts through the existing preparation and task CLI."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from adapters import submit
from task_fixtures import role_cohort, role_sources
from test_cli import cli, replay_config, saved_run

CONFIGS = Path(__file__).parents[1] / "configs"
__all__ = ["role_sources"]


@pytest.fixture(scope="module")
def imported_p3_task(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("p3-grading")
    rows = [
        {
            "name": f"RangeConstruction:{index}",
            "module": "synthetic-fixture.py",
            "sat": (
                f"def sat(answer: List[int], n={n}):\n"
                "    return answer[:n] == list(range(n))\n"
            ),
            "sol_header": f"def sol(n={n}):",
            "sol_bodies": ["    return list(range(n))"],
        }
        for index, n in enumerate((2, 3, 4))
    ]
    raw = json.dumps(rows).encode()
    (root / "puzzles.json").write_bytes(raw)
    (root / "lock.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic/p3-fixture",
                "revision": "fixture-v1",
                "format": "p3",
                "file": "puzzles.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "public-python-plan-v1",
                "dataset": "synthetic/p3-fixture",
                "revision": "fixture-v1",
                "sources": [
                    {
                        "lock": "lock.json",
                        "task_types": {"RangeConstruction": "short_program"},
                        "requirements": {
                            "RangeConstruction": "Return exactly n integers 0..n-1."
                        },
                        "contracts": {
                            "RangeConstruction": (
                                "def _grading_contract(answer, args):\n"
                                "    assert isinstance(answer, list)\n"
                                "    assert len(answer) == args[0]\n"
                            )
                        },
                    }
                ],
            }
        )
    )
    output = root / "imported"
    process = cli(
        "import-public-tasks",
        str(plan),
        "--source-dir",
        str(root),
        "--output-dir",
        str(output),
        cwd=root,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return output / "private/catalog.json"


def test_p3_parameter_instances_and_solutions_count_as_one_family(
    imported_p3_task: Path,
) -> None:
    catalog = json.loads(imported_p3_task.read_text())
    assert len(catalog["families"]) == 1
    family = catalog["families"][0]
    assert family["family_id"] == "p3/rangeconstruction"
    assert len(family["variants"]) == 1
    assert len(family["provenance"]["upstream_instances"]) == 3
    assert family["split"] == "unsplit"
    usage = json.loads((imported_p3_task.parent / "usage.json").read_text())
    assert usage["conversion_attempts"] == usage["converted_families"] == 1


@pytest.mark.parametrize(
    "implementation", ["reference", "memorizes_first", "invalid_length", "sys_exit"]
)
def test_p3_checks_native_parameters_and_strict_output_contracts(
    tmp_path: Path,
    imported_p3_task: Path,
    implementation: str,
) -> None:
    family = json.loads(imported_p3_task.read_text())["families"][0]
    variant = family["variants"][0]
    task = variant["task"]
    source = {
        "reference": task["reference_implementation"],
        "memorizes_first": variant["wrong_files"]["solution.py"],
        "invalid_length": "def solve(n):\n    return list(range(n)) + [99]\n",
        "sys_exit": "import sys\nsys.exit(0)\n",
    }[implementation]
    task["initial_files"]["solution.py"] = source
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    model = replay_config(tmp_path, [submit()])
    process = cli("task", str(path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == (0 if implementation == "reference" else 1), (
        process.stdout + process.stderr
    )
    _, run = saved_run(tmp_path / "runs")
    assert run["grade"]["passed"] is (implementation == "reference")


@pytest.fixture(scope="module")
def imported_public_tasks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("public-grading")
    rows = [
        {
            "task_id": identity,
            "text": "Double the input.",
            "code": "def double(n):\n    return n * 2\n",
            "test_list": [
                "assert double(2) == 4",
                "assert double(0) == 0",
                "assert double(-3) == -6",
            ],
        }
        for identity in (11, 12, 13)
    ]
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    (root / "source.jsonl").write_bytes(raw)
    (root / "lock.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic/grader-fixture",
                "revision": "fixture-v1",
                "format": "mbpp",
                "file": "source.jsonl",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "public-python-plan-v1",
                "dataset": "synthetic/grader-fixture",
                "revision": "fixture-v1",
                "sources": [
                    {
                        "lock": "lock.json",
                        "task_types": {
                            "11": "short_program",
                            "12": "code_edit",
                            "13": "workspace",
                        },
                    }
                ],
            }
        )
    )
    output = root / "imported"
    process = cli(
        "import-public-tasks",
        str(plan),
        "--source-dir",
        str(root),
        "--output-dir",
        str(output),
        cwd=root,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return output / "private/catalog.json"


@pytest.mark.parametrize("kind", ["short_program", "code_edit", "workspace"])
@pytest.mark.parametrize("implementation", ["sys_exit", "os_exit", "reference"])
def test_public_grading_requires_a_result_even_if_the_program_exits_successfully(
    tmp_path: Path,
    imported_public_tasks: Path,
    kind: str,
    implementation: str,
) -> None:
    catalog = json.loads(imported_public_tasks.read_text())
    family = next(f for f in catalog["families"] if f["task_type"] == kind)
    task = family["variants"][0]["task"]
    code = {
        "sys_exit": "import sys\nsys.exit(0)\n",
        "os_exit": "import os\nos._exit(0)\n",
        "reference": task["reference_implementation"],
    }[implementation]
    task["initial_files"][task["implementation_paths"][0]] = code
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    model = replay_config(tmp_path, [submit()])
    process = cli("task", str(path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == (0 if implementation == "reference" else 1), (
        process.stdout + process.stderr
    )
    _, run = saved_run(tmp_path / "runs")
    assert run["grade"]["passed"] is (implementation == "reference")


@pytest.mark.parametrize("kind", ["short_program", "code_edit", "workspace"])
def test_public_grading_preserves_dictionary_values_from_counter_results(
    tmp_path: Path,
    kind: str,
) -> None:
    row = {
        "task_id": 11,
        "text": "Count the occurrences of each item.",
        "code": (
            "from collections import Counter\ndef counts(items):\n"
            "    return Counter(items)\n"
        ),
        "test_list": [
            "assert counts(['x','x','y']) == {'x':2,'y':1}",
            "assert counts([]) == {}",
            "assert counts(['z']) == {'z':1}",
        ],
    }
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "source.jsonl").write_bytes(raw)
    (tmp_path / "lock.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic/counter-fixture",
                "revision": "fixture-v1",
                "format": "mbpp",
                "file": "source.jsonl",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "public-python-plan-v1",
                "dataset": "synthetic/counter-fixture",
                "revision": "fixture-v1",
                "sources": [{"lock": "lock.json", "task_types": {"11": kind}}],
            }
        )
    )
    output = tmp_path / "imported"
    process = cli(
        "import-public-tasks",
        str(plan),
        "--source-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    catalog = json.loads((output / "private/catalog.json").read_text())
    task = catalog["families"][0]["variants"][0]["task"]
    task["initial_files"][task["implementation_paths"][0]] = task[
        "reference_implementation"
    ]
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    model = replay_config(tmp_path, [submit()])
    process = cli("task", str(path), "-c", str(model), cwd=tmp_path)
    assert process.returncode == 0, process.stdout + process.stderr
    _, run = saved_run(tmp_path / "runs")
    assert run["grade"]["passed"] is True


def test_public_import_records_pinned_lineage_and_private_conversion_failures(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "task_id": identity,
            "text": "Synthetic doubling fixture.",
            "code": "def double(n):\n    return n * 2\n",
            "test_setup_code": "",
            "challenge_test_list": [],
            "test_list": [
                "assert double(2) == 4",
                "assert double(0) == 0",
                "assert double(-3) == -6",
            ],
        }
        for identity in (11, 12, 13)
    ]
    rows.append({"task_id": 14, "test_list": []})
    raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    (tmp_path / "source.jsonl").write_bytes(raw)
    lock = {
        "format": "mbpp",
        "dataset": "synthetic/test-fixture",
        "revision": "fixture-v1",
        "file": "source.jsonl",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    (tmp_path / "lock.json").write_text(json.dumps(lock))
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "public-python-plan-v1",
                "dataset": "synthetic/converted",
                "revision": "fixture-v1",
                "sources": [
                    {
                        "lock": "lock.json",
                        "task_types": {
                            "11": "short_program",
                            "12": "code_edit",
                            "13": "workspace",
                            "14": "short_program",
                        },
                    }
                ],
            }
        )
    )
    output = tmp_path / "imported"
    process = cli(
        "import-public-tasks",
        str(plan),
        "--source-dir",
        str(tmp_path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    catalog = json.loads((output / "private/catalog.json").read_text())
    usage = json.loads((output / "private/usage.json").read_text())
    assert usage["conversion_attempts"] == 4
    assert usage["converted_families"] == 3
    assert usage["paid_model_calls"] == 0
    assert usage["rejected"] == [
        {
            "source": "synthetic/test-fixture",
            "task_id": "14",
            "reason": "At least three source assertions are required",
        }
    ]
    families = catalog["families"]
    assert [f["family_id"] for f in families] == ["mbpp/11", "mbpp/12", "mbpp/13"]
    assert all(f["split"] == "test" for f in families)
    edit = families[1]["variants"][0]["task"]
    assert "return None" in edit["initial_files"]["solution.py"]
    workspace = families[2]["variants"][0]["task"]
    assert "data/work_items.txt" in workspace["initial_files"]
    assert "_grading/cases.py" not in workspace["initial_files"]
    assert (output / "private/upstream/source.jsonl").read_bytes() == raw


def test_new_equivalence_guard_rejects_duplicate_families_before_execution(
    tmp_path: Path,
    role_sources: Path,
) -> None:
    path = freeze_cohort(tmp_path / "duplicate-cohort.json")
    cohort = json.loads(path.read_text())
    cohort["independence_guard"] = [
        {
            "members": ["harness/header-key", "harness/nested-setting"],
            "reason": "Synthetic equivalence guard fixture.",
        }
    ]
    path.write_text(json.dumps(cohort))
    output = tmp_path / "rejected"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    assert "equivalent families" in process.stderr
    assert not output.exists()


def test_public_import_has_the_same_content_identity_across_python_hash_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = {
        "task_id": 11,
        "text": "Return the set of input characters.",
        "code": "def chars(items):\n    return set(items)\n",
        "test_list": [
            "assert chars(('x', 'y', 'z')) == {'z', 'y', 'x'}",
            "assert chars(('a', 'b')) == {'b', 'a'}",
            "assert chars(('m',)) == {'m'}",
        ],
    }
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "source.jsonl").write_bytes(raw)
    (tmp_path / "lock.json").write_text(
        json.dumps(
            {
                "dataset": "synthetic/set-fixture",
                "revision": "fixture-v1",
                "format": "mbpp",
                "file": "source.jsonl",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "public-python-plan-v1",
                "dataset": "synthetic/set-fixture",
                "revision": "fixture-v1",
                "sources": [{"lock": "lock.json", "task_types": {"11": "workspace"}}],
            }
        )
    )
    outputs = []
    for seed in (1, 37):
        monkeypatch.setenv("PYTHONHASHSEED", str(seed))
        output = tmp_path / str(seed)
        process = cli(
            "import-public-tasks",
            str(plan),
            "--source-dir",
            str(tmp_path),
            "--output-dir",
            str(output),
            cwd=tmp_path,
        )
        assert process.returncode == 0, process.stdout + process.stderr
        outputs.append((output / "private/catalog.json").read_bytes())
    assert outputs[0] == outputs[1]


def freeze_cohort(path: Path) -> Path:
    cohort = role_cohort()
    cohort["freeze"] = {
        "version": "opsd-cohort-freeze-v1",
        "subset_size": 3,
        "sampling_seed": 31001,
        "training_seed": 31002,
        "epoch_shuffle_seed": 31003,
        "evaluation_seeds": [4, 17, 29, 41, 53],
        "epochs": 4,
        "difficulty_basis": "Reviewed task requirements, before model execution.",
        "difficulty": {
            "fixture/sign-runs": "easy",
            "harness/header-key": "easy",
            "harness/nested-setting": "easy",
            "harness/stock-value": "easy",
            "harness/license-audit": "medium",
        },
    }
    path.write_text(json.dumps(cohort))
    return path


@pytest.fixture(scope="module")
def frozen_cohort(tmp_path_factory: pytest.TempPathFactory, role_sources: Path) -> Path:
    root = tmp_path_factory.mktemp("cohort-freeze")
    cohort = freeze_cohort(root / "cohort.json")
    output = root / "frozen"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(cohort),
        "--output-dir",
        str(output),
        cwd=root,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return output


def test_preparation_freezes_one_stratified_subset_and_four_visit_orders(
    tmp_path: Path,
    frozen_cohort: Path,
) -> None:
    manifest = json.loads((frozen_cohort / "manifest.json").read_text())
    selection = json.loads((frozen_cohort / "training-selection.json").read_text())
    assert selection["per_type"] == {
        "short_program": 1,
        "code_edit": 1,
        "workspace": 1,
    }
    ids = selection["task_ids"]
    assert len(ids) == len(set(ids)) == 3
    assert len(selection["epochs"]) == 4
    assert all(sorted(order) == sorted(ids) for order in selection["epochs"])
    assert set(selection["task_sha256"]) == set(ids)
    assert selection["seeds"] == {
        "sampling": 31001,
        "training": 31002,
        "epoch_shuffle": 31003,
        "evaluation": [4, 17, 29, 41, 53],
        "chosen_by": "execution-config",
    }
    assert len(manifest["checklists"]["training_candidates"]) == 5
    assert manifest["model_baseline"] is False
    model = replay_config(tmp_path, [submit()])
    process = cli(
        "task",
        str(frozen_cohort / "manifest.json"),
        "--task-id",
        ids[0],
        "--split",
        "train_candidate",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    _, run = saved_run(tmp_path / "runs")
    assert run["task_assignment"]["role"] == "train_candidate"


def test_training_cli_rejects_changed_frozen_subset_before_remote_work(
    tmp_path: Path, frozen_cohort: Path
) -> None:
    selection = json.loads((frozen_cohort / "training-selection.json").read_text())
    configuration = {
        "train_manifest": str(frozen_cohort / "manifest.json"),
        "training_selection": str(frozen_cohort / "training-selection.json"),
        "train_task_ids": selection["task_ids"][:-1],
        "dev_plan": str(frozen_cohort / "harness-dev.json"),
        "condition": [],
        "epochs": 4,
        "shuffle_seed": 999,
        "evaluation_seeds": [4, 17, 29, 41, 53],
        "evaluate_each_epoch": True,
        "require_complete_training": True,
        "ledger_path": str(tmp_path / "spend.sqlite3"),
        "backend": {"spend_config": str(CONFIGS / "opsd.yaml")},
        "training": {"spend_config": str(CONFIGS / "opsd.yaml")},
    }
    config = tmp_path / "pilot.yaml"
    config.write_text(yaml.safe_dump(configuration))
    process = cli(
        "train-opsd", str(config), "--output-dir", str(tmp_path / "run"), cwd=tmp_path
    )
    assert process.returncode == 2
    assert "Frozen training" in process.stderr
    assert not (tmp_path / "spend.sqlite3").exists()


def test_subset_and_epoch_orders_repeat_when_pool_order_and_workers_change(
    tmp_path: Path,
    frozen_cohort: Path,
) -> None:
    cohort = json.loads((frozen_cohort / "private/cohort.json").read_text())
    cohort["training_candidates"]["task_ids"].reverse()
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(cohort))
    output = tmp_path / "repeated"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(path),
        "--workers",
        "2",
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    first = json.loads((frozen_cohort / "training-selection.json").read_text())
    repeated = json.loads((output / "training-selection.json").read_text())
    assert repeated["task_ids"] == first["task_ids"]
    assert repeated["family_ids"] == first["family_ids"]
    assert repeated["epochs"] == first["epochs"]
    assert repeated["strata"] == first["strata"]


@pytest.mark.parametrize(
    "change",
    [
        "selection",
        "order",
        "order_and_hash",
        "unselected_task",
        "unselected_task_and_hash",
    ],
)
def test_loading_rejects_changes_to_the_subset_orders_or_unsampled_pool(
    tmp_path: Path,
    frozen_cohort: Path,
    change: str,
) -> None:
    # Restore bytes afterwards: Task assignments deliberately bind this path.
    manifest = json.loads((frozen_cohort / "manifest.json").read_text())
    selection_path = frozen_cohort / "training-selection.json"
    selection = json.loads(selection_path.read_text())
    path = selection_path
    if change.startswith("unselected_task"):
        entry = next(
            e for e in manifest["tasks"] if e["task_id"] not in selection["task_ids"]
        )
        path = frozen_cohort / entry["task_file"]
        changed = json.loads(path.read_text())
        changed["prompt"] += " changed"
    else:
        changed = selection
        if change == "selection":
            changed["task_ids"] = changed["task_ids"][:-1]
        else:
            changed["epochs"][0].reverse()
    original = path.read_bytes()
    manifest_path = frozen_cohort / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    try:
        path.write_text(json.dumps(changed))
        if change == "order_and_hash":
            manifest["training_selection"]["sha256"] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
        if change == "unselected_task_and_hash":
            entry["task_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
        model = replay_config(tmp_path, [submit()])
        process = cli(
            "task",
            str(frozen_cohort / "manifest.json"),
            "--task-id",
            manifest["checklists"]["training_candidates"][0],
            "--split",
            "train",
            "-c",
            str(model),
            cwd=tmp_path,
        )
        assert process.returncode == 2, process.stdout + process.stderr
        assert "changed" in process.stderr
        assert not (tmp_path / "runs").exists()
    finally:
        path.write_bytes(original)
        manifest_path.write_bytes(manifest_bytes)


@pytest.mark.parametrize("change", ["quota", "difficulty", "seed"])
def test_invalid_freeze_configuration_is_rejected_before_any_execution(
    tmp_path: Path,
    role_sources: Path,
    change: str,
) -> None:
    path = freeze_cohort(tmp_path / "invalid.json")
    cohort = json.loads(path.read_text())
    if change == "quota":
        cohort["training_candidates"]["per_type"]["short_program"] = 2
    elif change == "difficulty":
        del cohort["freeze"]["difficulty"]["fixture/sign-runs"]
    else:
        cohort["freeze"]["sampling_seed"] = 4
    path.write_text(json.dumps(cohort))
    output = tmp_path / "invalid"
    process = cli(
        "prepare-harness",
        "--cohort",
        str(path),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 2, process.stdout + process.stderr
    assert not output.exists()
