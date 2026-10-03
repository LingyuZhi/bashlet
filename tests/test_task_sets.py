"""Multi-source task admission through public loading and real experiments."""

import hashlib
import json
import shlex
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest
import yaml
from adapters import bash, replay, submit
from test_cli import cli, replay_config, saved_run
from test_demonstrations import TrainingAdapter
from test_encoding import QWEN, REVISION, official_tokenizer
from test_mbpp import double_row, prepare

from bashlet import (
    FunctionGrader,
    GradingCase,
    QwenChatMLEncoding,
    Task,
    TaskOrigin,
    run_experiment,
)
from bashlet.config import load_config
from bashlet.demonstrations import reference_demonstration

__all__ = ["official_tokenizer"]


def test_registered_authored_family_runs_with_independent_grading(
    tmp_path: Path,
) -> None:
    catalog = Path(__file__).parents[1] / "configs/harness-authored-v1.json"
    task = Task(
        task_id="harness/header-key/renamed",
        prompt="Repair app/headers.py and submit.",
        initial_files={
            "app/headers.py": (
                "def normalize_header(text):\n"
                "    return text.strip().lower().replace('_', '-')\n"
            ),
        },
        grader=FunctionGrader(
            "app/headers.py", "normalize_header", (GradingCase((" X_ID ",), "x-id"),)
        ),
        origin=TaskOrigin(
            "bashlet/harness-authored",
            "harness-authored-v1",
            hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "header-key",
            "harness/header-key",
            "train",
        ),
    )
    result = run_experiment(
        task=task, model=replay([submit()]), output_dir=tmp_path / "runs"
    )
    assert result.passed, result.reason
    record = json.loads((result.run_dir / "run.json").read_text())
    assert task.origin is not None
    assert record["task_origin"] == asdict(task.origin)


def mixed_manifest(tmp_path: Path) -> Path:
    """Two independently registered sources in the private Task format."""
    root = prepare(tmp_path, [double_row()])
    old = json.loads((root / "manifest.json").read_text())
    mbpp = old["tasks"][0]
    mbpp_task = json.loads((root / mbpp["task_file"]).read_text())
    catalog_path = Path(__file__).parents[1] / "configs/harness-authored-v1.json"
    catalog = json.loads(catalog_path.read_text())
    family = catalog["families"][0]
    authored: dict[str, Any] = family["variants"][0]["task"]
    origin = {
        "dataset": catalog["dataset"],
        "revision": catalog["revision"],
        "source_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "original_task_id": family["original_task_id"],
        "family_id": family["family_id"],
        "split": family["split"],
    }
    authored["origin"] = origin
    authored_path = root / "private/tasks/train/header-key.json"
    authored_path.write_text(json.dumps(authored))
    entries = []
    for data, path, kind in (
        (mbpp_task, root / mbpp["task_file"], "short_program"),
        (authored, authored_path, "code_edit"),
    ):
        entries.append(
            {
                "task_id": data["task_id"],
                "origin": data["origin"],
                "family_id": data["origin"]["family_id"],
                "split": "train",
                "task_type": kind,
                "variant_id": "base",
                "status": "verified",
                "task_file": str(path.relative_to(root)),
                "task_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "conversion": "harness-task-families-v1",
        "source": {"dataset": "bashlet/harness-cohort", "revision": "v1"},
        "sources": [
            old["source"],
            {
                "dataset": catalog["dataset"],
                "revision": catalog["revision"],
                "sha256": origin["source_sha256"],
            },
        ],
        "tasks": entries,
    }
    path = root / "mixed.json"
    path.write_text(json.dumps(manifest))
    return path


def test_public_task_selection_reads_mbpp_and_authored_sources(tmp_path: Path) -> None:
    from bashlet.tasks import task_path

    manifest = mixed_manifest(tmp_path)
    selected = [
        json.loads(task_path(manifest, name, "train").read_text())["task_id"]
        for name in ("mbpp/602", "harness/header-key")
    ]
    assert selected == ["mbpp/602", "harness/header-key"]


def small_cohort(tmp_path: Path) -> tuple[Path, Path]:
    mbpp = prepare(tmp_path, [double_row(602), double_row(511)]) / "manifest.json"
    path = tmp_path / "cohort.json"
    path.write_text(
        json.dumps(
            {
                "version": "harness-task-cohort-v1",
                "name": "test-fixture-v1",
                "harness_development": {
                    "task_ids": [
                        "mbpp/511",
                        "harness/nested-setting",
                        "harness/license-audit",
                    ],
                    "per_type": 1,
                },
                "training_candidates": {
                    "task_ids": [
                        "mbpp/602",
                        "harness/header-key",
                        "harness/stock-value",
                    ],
                    "per_type": 1,
                },
                "coding_development": {"task_ids": ["mbpp/511"]},
                "environment_variants": ["harness/license-audit/policy-green"],
                "rules": {"purpose": "test-double fixture; never a model baseline"},
            }
        )
    )
    return mbpp, path


@pytest.fixture(scope="module")
def prepared_tasks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("mixed-source")
    mbpp, cohort = small_cohort(tmp_path)
    output = tmp_path / "harness"
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
    return output


def test_preparation_executes_references_and_rejects_targeted_wrong_submissions(
    prepared_tasks: Path,
) -> None:
    output = prepared_tasks
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["counts"] == {
        "train": {"short_program": 1, "code_edit": 1, "workspace": 1},
        "dev": {"short_program": 1, "code_edit": 1, "workspace": 1},
    }
    assert manifest["checklists"]["overlap_family_ids"] == ["mbpp/511"]
    assert len(manifest["tasks"]) == 7
    assert manifest["evidence_kind"] == "reference-and-mutation-replay"
    for entry in manifest["tasks"]:
        reference = json.loads(
            (output / entry["reference_run"] / "run.json").read_text()
        )
        wrong = json.loads((output / entry["wrong_run"] / "run.json").read_text())
        assert reference["status"] == "completed" and reference["grade"]["passed"]
        assert wrong["status"] == "completed" and not wrong["grade"]["passed"]
        assert entry["feedback_start_id"] and entry["environment_variant_id"]


def test_evaluation_rejects_environment_variants_as_extra_independent_tasks(
    tmp_path: Path,
    prepared_tasks: Path,
) -> None:
    plan = tmp_path / "variants.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "invalid-variants",
                "manifest": str(prepared_tasks / "manifest.json"),
                "split": "dev",
                "task_ids": [
                    "harness/license-audit",
                    "harness/license-audit/policy-green",
                ],
            }
        )
    )
    process = cli(
        "evaluate",
        str(plan),
        "--snapshot",
        "replay/prewritten-v1",
        "--evidence-kind",
        "test-double",
        cwd=tmp_path,
    )
    assert process.returncode == 2
    assert "family" in process.stderr and "independent" in process.stderr
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "change",
    [
        "cross_split",
        "unknown_source",
        "unregistered_family",
        "unregistered_version",
        "source_content",
        "undeclared_source",
        "task_content",
        "outside_path",
        "incomplete",
    ],
)
def test_manifest_loading_rejects_untrusted_lineage_and_changed_content(
    change: str,
    prepared_tasks: Path,
) -> None:
    from bashlet.tasks import task_path

    manifest = json.loads((prepared_tasks / "manifest.json").read_text())
    entry = next(e for e in manifest["tasks"] if e["task_id"] == "harness/header-key")
    origin = entry["origin"]
    if change == "cross_split":
        entry["split"] = origin["split"] = "dev"
    elif change == "unknown_source":
        origin["dataset"] = "unknown/unreviewed"
    elif change == "unregistered_family":
        origin["original_task_id"] = "not-registered"
    elif change == "unregistered_version":
        origin["revision"] = "future-unreviewed-v2"
    elif change == "source_content":
        origin["source_sha256"] = "0" * 64
    elif change == "undeclared_source":
        manifest["sources"] = manifest["sources"][:1]
    elif change == "task_content":
        entry["task_sha256"] = "0" * 64
    elif change == "outside_path":
        entry["task_file"] = "../../operator-answer.json"
    else:
        manifest["status"] = "preparing"
    path = prepared_tasks / ("invalid-" + change + ".json")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        task_path(path, "harness/header-key", "train")


@pytest.mark.parametrize(
    "task_id, kind",
    [
        ("mbpp/602", "short_program"),
        ("harness/header-key", "code_edit"),
        ("harness/stock-value", "workspace"),
    ],
)
def test_each_task_type_runs_via_existing_cli_without_hidden_observations(
    tmp_path: Path,
    prepared_tasks: Path,
    task_id: str,
    kind: str,
) -> None:
    from bashlet.tasks import task_path

    task = json.loads(
        task_path(prepared_tasks / "manifest.json", task_id, "train").read_text()
    )
    implementation = task["implementation_paths"][0]
    script = (
        "from pathlib import Path; "
        f"p = Path({implementation!r}); p.parent.mkdir(parents=True, exist_ok=True); "
        f"p.write_text({task['reference_implementation']!r})"
    )
    replies = [
        bash("find . -type f -print -exec cat {} \\;"),
        bash("python3 -c " + shlex.quote(script)),
        bash(task["feedback_command"]),
        submit(),
    ]
    model = replay_config(tmp_path, replies)
    process = cli(
        "task",
        str(prepared_tasks / "manifest.json"),
        "--task-id",
        task_id,
        "--split",
        "train",
        "-c",
        str(model),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    _, record = saved_run(tmp_path / "runs")
    assert record["passed"] and record["task_type"] == kind
    public_observation = record["trajectory"][0]["observation"]["output"]
    assert "_grading/check.py" not in public_observation
    assert "_grading/cases.py" not in public_observation
    assert task["reference_implementation"] not in public_observation
    assert "_grading/check.py" not in record["initial_workspace_manifest"]
    assert "_grading/cases.py" not in record["initial_workspace_manifest"]
    assert record["task_origin"] == task["origin"]


def test_workspace_policy_variant_needs_actual_local_policy(
    tmp_path: Path,
    prepared_tasks: Path,
) -> None:
    from bashlet.experiment import task_from_record
    from bashlet.tasks import task_path

    task = task_from_record(
        json.loads(
            task_path(
                prepared_tasks / "manifest.json",
                "harness/license-audit/policy-green",
                "dev",
            ).read_text()
        )
    )
    result = run_experiment(
        task=task,
        model=replay(
            [
                bash("cat policy/licenses.json registry/dependencies.json"),
                bash(
                    "mkdir -p tools\ncat > tools/license_audit.py <<'PY'\n"
                    + task.reference_implementation
                    + "PY"
                ),
                bash(str(task.feedback_command)),
                submit(),
            ]
        ),
        output_dir=tmp_path,
    )
    assert result.passed
    observation = result.trajectory[0].observation
    assert observation is not None and "Custom-Green" in observation.output
    assert "Custom-Green" not in task.prompt
    record = json.loads((result.run_dir / "run.json").read_text())
    assert record["variant_id"] == "policy-green"
    assert record["task_origin"]["family_id"] == "harness/license-audit"


def test_unexecuted_mutation_cannot_be_counted_as_successful_grader_validation(
    tmp_path: Path,
) -> None:
    # The reference command fits, but a memorizer of this long public example
    # exceeds the replay generation limit. A missing module is not a mutation.
    row = double_row(602)
    row["test_list"][0] = f"assert double({'x' * 2000!r}) == {'x' * 4000!r}"
    mbpp = prepare(tmp_path, [row, double_row(511)]) / "manifest.json"
    other = tmp_path / "other-source"
    other.mkdir()
    _, cohort = small_cohort(other)
    output = tmp_path / "unexecuted"
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
    assert process.returncode == 2
    assert "mutation was not executed" in process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"


def test_authored_training_trajectory_keeps_family_and_task_type_at_export(
    tmp_path: Path,
    prepared_tasks: Path,
    official_tokenizer: Any,
) -> None:
    from bashlet.experiment import task_from_record
    from bashlet.tasks import task_path

    task = task_from_record(
        json.loads(
            task_path(
                prepared_tasks / "manifest.json", "harness/header-key", "train"
            ).read_text()
        )
    )
    encoding = QwenChatMLEncoding(QWEN, revision=REVISION)
    adapter = TrainingAdapter(encoding.identity)
    config = load_config()
    result = run_experiment(
        task=task,
        model=reference_demonstration(task=task, encoding=encoding, config=config),
        config=config,
        output_dir=tmp_path,
        sft_adapter=adapter,
    )
    assert result.passed and result.demonstration is not None
    assert result.demonstration["status"] == "accepted"
    _, metadata = adapter.requests[0]
    assert metadata["origin"]["family_id"] == "harness/header-key"
    assert metadata["origin"]["split"] == "train"
    assert metadata["task_type"] == "code_edit"
    assert metadata["feedback_start_id"] == task.feedback_start_id
    assert metadata["on_policy_eligible"] is False


def test_existing_sft_cli_consumes_authored_tasks_from_the_mixed_manifest(
    tmp_path: Path,
    prepared_tasks: Path,
) -> None:
    config = tmp_path / "sft.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "ReplayModel",
                    "tokenizer": QWEN,
                    "tokenizer_revision": REVISION,
                }
            }
        )
    )
    output = tmp_path / "demonstrations"
    process = cli(
        "prepare-sft",
        str(prepared_tasks / "manifest.json"),
        "--task-id",
        "harness/header-key",
        "--task-id",
        "harness/stock-value",
        "-c",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["counts"]["accepted"] == 2
    examples = [
        json.loads(line)
        for line in (output / "examples.jsonl").read_text().splitlines()
    ]
    assert [e["metadata"]["task_type"] for e in examples] == ["code_edit", "workspace"]
    assert all(e["metadata"]["origin"]["split"] == "train" for e in examples)


def test_existing_evaluation_cli_keeps_each_type_and_the_fixed_denominator(
    tmp_path: Path,
    prepared_tasks: Path,
) -> None:
    model = replay_config(tmp_path, [submit(), submit(), submit()])
    process = cli(
        "evaluate",
        str(prepared_tasks / "harness-dev.json"),
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
    path = Path(process.stdout.strip().splitlines()[-1])
    record = json.loads(path.read_text())
    assert record["evidence_kind"] == "test-double"
    assert record["summary"]["planned_tasks"] == 3
    assert [r["task_type"] for r in record["tasks"]] == [
        "short_program",
        "code_edit",
        "workspace",
    ]
    assert all(r["task_origin"]["split"] == "dev" for r in record["tasks"])


@pytest.mark.parametrize(
    "task_id, code",
    [
        (
            "harness/last-record",
            "def last_records(records):\n"
            "    records.reverse()\n    seen = set()\n    result = []\n"
            "    for r in records:\n        if r['id'] not in seen:\n"
            "            seen.add(r['id'])\n            result.append(r)\n"
            "    return result[::-1]\n",
        ),
        (
            "harness/secret-fields",
            "def redact(values):\n    for key in values:\n"
            "        if key.lower() in ('token', 'password'):\n"
            "            values[key] = '[redacted]'\n    return values\n",
        ),
        (
            "harness/missing-defaults",
            "def with_defaults(settings, defaults):\n"
            "    defaults.update(settings)\n    return defaults\n",
        ),
    ],
)
def test_correct_return_values_cannot_hide_forbidden_input_mutation(
    tmp_path: Path,
    task_id: str,
    code: str,
) -> None:
    from bashlet.experiment import task_from_record

    source = Path(__file__).parents[1] / "configs/harness-authored-v1.json"
    catalog = json.loads(source.read_text())
    family = next(f for f in catalog["families"] if f["family_id"] == task_id)
    data = family["variants"][0]["task"] | {
        "origin": {
            "dataset": catalog["dataset"],
            "revision": catalog["revision"],
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "original_task_id": family["original_task_id"],
            "family_id": task_id,
            "split": "train",
        }
    }
    task = task_from_record(data)
    script = (
        "from pathlib import Path; "
        f"Path({task.implementation_paths[0]!r}).write_text({code!r})"
    )
    result = run_experiment(
        task=task,
        model=replay([bash("python3 -c " + shlex.quote(script)), submit()]),
        output_dir=tmp_path,
    )
    assert result.status == "completed", result.reason
    assert not result.passed


def test_supplemental_cohort_rejects_known_mbpp_equivalents_across_splits(
    tmp_path: Path,
) -> None:
    row = {
        "task_id": 625,
        "prompt": "Swap the first and last list elements.",
        "code": "def swap(items):\n    result = items[:]\n"
        "    result[0], result[-1] = result[-1], result[0]\n    return result\n",
        "test_imports": [],
        "test_list": [
            "assert swap([1, 2, 3]) == [3, 2, 1]",
            "assert swap([8, 9]) == [9, 8]",
        ],
    }
    mbpp = prepare(tmp_path, [row, row | {"task_id": 591}]) / "manifest.json"
    cohort = tmp_path / "supplement.json"
    cohort.write_text(
        json.dumps(
            {
                "version": "harness-task-cohort-v1",
                "name": "unsafe-supplement",
                "harness_development": {
                    "task_ids": [
                        "mbpp/591",
                        "harness/nested-setting",
                        "harness/license-audit",
                    ],
                    "per_type": 1,
                },
                "training_candidates": {
                    "task_ids": [
                        "mbpp/625",
                        "harness/header-key",
                        "harness/stock-value",
                    ],
                    "per_type": 1,
                },
                "coding_development": {"task_ids": ["mbpp/591"]},
                "environment_variants": [],
                "rules": {},
            }
        )
    )
    output = tmp_path / "unsafe"
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
    assert process.returncode == 2
    assert "equivalent family" in process.stderr
    assert not output.exists()
