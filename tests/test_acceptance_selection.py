"""Acceptance and rejection through saved CLI experiments."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_cli import cli
from test_mbpp import double_row, prepare
from test_monitoring import external_sdks


@pytest.fixture(scope="module")
def paired_acceptance(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path, Path]:
    root = tmp_path_factory.mktemp("paired-acceptance")
    prepared = prepare(root, [double_row(511)])
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "version": "development-evaluation-v1",
                "name": "selection-fixture",
                "manifest": str(prepared / "manifest.json"),
                "split": "dev",
                "task_ids": ["mbpp/511"],
            }
        )
    )
    policy = json.loads(
        (Path(__file__).parents[1] / "configs/sft-acceptance-v1.json").read_text()
    )
    policy.update(
        harness_plan=str(plan),
        coding_plan=str(plan),
        variant_plan=None,
        harness_counts={"short_program": 1},
        coding_count=1,
    )
    policy_path = root / "policy.json"
    policy_path.write_text(json.dumps(policy))
    training = root / "training"
    training.mkdir()
    (training / "training.json").write_text(
        json.dumps(
            {
                "metadata": {"base_model": "Qwen/Qwen3.5-4B"},
                "checkpoints": [
                    {
                        "optimizer_step": 12,
                        "sampler": {
                            "kind": "sampler_weights",
                            "path": "trio://run/sampler-12",
                        },
                        "training_state": {
                            "kind": "training_state",
                            "path": "trio://run/state-12",
                        },
                    }
                ],
            }
        )
    )
    config = root / "model.yaml"
    config.write_text(
        yaml.safe_dump({"model": {"class": "adapters.DirectEvaluationModel"}})
    )
    paths = []
    import os

    old_path = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.resolve())
    try:
        for name, snapshot in [
            ("base", "pytrio/Qwen/Qwen3.5-4B@unversioned-base"),
            ("candidate", "pytrio/Qwen/Qwen3.5-4B@trio://run/sampler-12"),
        ]:
            result = cli(
                "evaluate-acceptance",
                str(policy_path),
                "--snapshot",
                snapshot,
                "--set",
                f"model.identity={snapshot}",
                "--evidence-kind",
                "test-double",
                "--stage",
                "confirmation",
                "-c",
                str(config),
                "--output-dir",
                str(root / name),
                cwd=root,
                timeout=90,
            )
            assert result.returncode == 0, result.stdout + result.stderr
            paths.append(Path(result.stdout.strip().splitlines()[-1]))
    finally:
        os.environ["PYTHONPATH"] = old_path
    return training, paths[0], paths[1]


def test_selection_can_reject_every_candidate_without_student_or_teacher_start(
    paired_acceptance: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    training, baseline, candidate = paired_acceptance
    selected = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(baseline),
        "--evaluation",
        str(candidate),
        "--evidence-kind",
        "test-double",
        "--output-dir",
        str(tmp_path),
        cwd=tmp_path,
    )
    assert selected.returncode == 0, selected.stdout + selected.stderr
    record: dict[str, Any] = json.loads((tmp_path / "selection.json").read_text())
    assert record["status"] == "no_qualified_snapshot"
    assert record["student_snapshot"] is None
    assert record["frozen_teacher_snapshot"] is None
    assert record["sft_training_state"] is None
    assert record["candidates"][0]["decision"] == "ineligible"
    assert "no_task_gain" in record["candidates"][0]["reasons"]


def test_no_qualified_snapshot_is_monitorable_without_a_fake_sampler_value(
    paired_acceptance: tuple[Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training, baseline, candidate = paired_acceptance
    external_sdks(tmp_path, monkeypatch)
    config = tmp_path / "monitor.yaml"
    config.write_text("project: bashlet\ngroup: acceptance\nmode: offline\n")
    output = tmp_path / "selection"
    process = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(baseline),
        "--evaluation",
        str(candidate),
        "--evidence-kind",
        "test-double",
        "--monitor-config",
        str(config),
        "--output-dir",
        str(output),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    remote = json.loads((output / "remote.json").read_text())
    assert remote["rows"][-1]["checkpoint/acceptance_status"] == "no_qualified_snapshot"
    assert "checkpoint/selected_sampler" not in remote["rows"][-1]
    assert "secret-test-key" not in json.dumps(remote)


def test_frozen_task_identity_is_checked_against_the_actual_graded_task(
    paired_acceptance: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    training, baseline, candidate = paired_acceptance
    paths = []
    for index, source in enumerate((baseline, candidate)):
        record = json.loads(source.read_text())
        record["frozen"]["inputs"]["harness"]["tasks"][0]["task_sha256"] = "0" * 64
        record["frozen_sha256"] = hashlib.sha256(
            json.dumps(
                record["frozen"],
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        path = tmp_path / f"evidence-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    process = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(paths[0]),
        "--evaluation",
        str(paths[1]),
        "--evidence-kind",
        "test-double",
        "--output-dir",
        str(tmp_path),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path / "selection.json").read_text())
    assert result["candidates"][0]["decision"] == "insufficient"
    assert any(
        "frozen_task_identity_mismatch" in gap
        for gap in result["candidates"][0]["gaps"]
    )
    assert result["student_snapshot"] is None


def test_candidate_without_harness_records_retains_its_identity_and_evidence_gap(
    paired_acceptance: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    training, original_base, original_candidate = paired_acceptance
    base = json.loads(original_base.read_text())
    base.update(
        evidence_kind="experimental", snapshot="vllm/Qwen/Qwen3.5-4B@" + "a" * 40
    )
    # A declared base is kept as insufficient; it cannot produce an OPSD start.
    for refs in base["runs"].values():
        for seed, ref in list(refs.items()):
            raw = json.loads(Path(ref["path"]).read_text())
            raw["snapshot_details"] = {
                "snapshot_kind": "base",
                "base_model": "Qwen/Qwen3.5-4B",
                "model_revision": "a" * 40,
            }
            path = (
                tmp_path / f"declared-base-{seed}-{len(list(tmp_path.iterdir()))}.json"
            )
            path.write_text(json.dumps(raw))
            refs[seed] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    control = tmp_path / "baseline.json"
    control.write_text(json.dumps(base))
    current = json.loads(original_candidate.read_text())
    current.update(
        evidence_kind="experimental",
        status="incomplete",
        snapshot="vllm/Qwen/Qwen3.5-4B@trio://run/sampler-12:" + "b" * 64,
        runs={track: {} for track in current["runs"]},
    )
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(current))
    process = cli(
        "select-sft",
        str(training),
        "--baseline",
        str(control),
        "--evaluation",
        str(candidate),
        "--output-dir",
        str(tmp_path),
        cwd=tmp_path,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads((tmp_path / "selection.json").read_text())
    row = result["candidates"][0]
    assert row["decision"] == "insufficient"
    assert row["snapshot_identity"] == current["snapshot"]
    assert "missing:harness:seed-4" in row["gaps"]
    assert result["student_snapshot"] is None


@dataclass
class AcceptanceExperiment:
    root: Path
    policy: Path
    config: Path
    training: Path

    def evaluate(self, scenario: str, step: int = 0) -> Path:
        destination = self.root / f"{scenario}-{step}"
        path = destination / "acceptance.json"
        if path.exists():
            return path
        snapshot = "pytrio/Qwen/Qwen3.5-4B@" + (
            f"trio://run/sampler-{step}" if step else "unversioned-base"
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("PYTHONPATH", str(Path(__file__).parent.resolve()))
            process = cli(
                "evaluate-acceptance",
                str(self.policy),
                "--snapshot",
                snapshot,
                "--evidence-kind",
                "test-double",
                "--stage",
                "confirmation",
                "-c",
                str(self.config),
                "--set",
                f"model.identity={snapshot}",
                "--set",
                f"model.scenario={scenario}",
                "--output-dir",
                str(destination),
                cwd=self.root,
                timeout=120,
            )
        assert process.returncode == (2 if scenario == "backend-error" else 0), (
            process.stdout + process.stderr
        )
        return path

    def review(self, candidate: Path) -> Path:
        baseline = self.evaluate("base")
        record = json.loads(candidate.read_text())
        items = []
        for seed, ref in record["runs"]["harness"].items():
            evaluation = json.loads(Path(ref["path"]).read_text())
            for row in evaluation["tasks"]:
                if row["task_type"] != "workspace":
                    continue
                items.append(
                    {
                        "task_id": row["task_id"],
                        "seed": int(seed),
                        "observation_step": 0,
                        "implementation_step": 1,
                        "observed_paths": [
                            "docs/" + row["task_id"].split("/")[-1] + ".md"
                        ],
                        "reason": (
                            "Observed instructions identify actual input paths; "
                            "the code reads them and passes changed-input grading."
                        ),
                        "author": "test-reviewer",
                    }
                )
        path = candidate.parent / "review.json"
        path.write_text(
            json.dumps(
                {
                    "version": "workspace-benefit-review-v1",
                    "baseline_sha256": hashlib.sha256(
                        baseline.read_bytes()
                    ).hexdigest(),
                    "candidate_sha256": hashlib.sha256(
                        candidate.read_bytes()
                    ).hexdigest(),
                    "items": items,
                }
            )
        )
        return path


@pytest.fixture(scope="module")
def acceptance_experiment(
    tmp_path_factory: pytest.TempPathFactory,
) -> AcceptanceExperiment:
    root = tmp_path_factory.mktemp("acceptance-boundaries")
    mbpp = prepare(
        root,
        [
            *[double_row(511 + index) for index in range(6)],
            double_row(602),
            double_row(603),
        ],
    )
    cohort = root / "cohort.json"
    catalog = json.loads(
        (Path(__file__).parents[1] / "configs/harness-authored-v1.json").read_text()
    )
    training_ids = ["mbpp/602", "mbpp/603"]
    for kind in ("code_edit", "workspace"):
        training_ids.extend(
            family["family_id"]
            for family in [
                family
                for family in catalog["families"]
                if family["split"] == "train" and family["task_type"] == kind
            ][:2]
        )
    cohort.write_text(
        json.dumps(
            {
                "version": "harness-task-cohort-v1",
                "name": "acceptance-fixture",
                "harness_development": {
                    "per_type": 2,
                    "task_ids": [
                        "mbpp/511",
                        "mbpp/512",
                        "harness/nested-setting",
                        "harness/duration-units",
                        "harness/license-audit",
                        "harness/deployment-targets",
                    ],
                },
                "training_candidates": {"per_type": 2, "task_ids": training_ids},
                "coding_development": {
                    "task_ids": [f"mbpp/{511 + index}" for index in range(6)]
                },
                "environment_variants": [],
                "rules": {"purpose": "test double; never a baseline"},
            }
        )
    )
    prepared = root / "harness-tasks"
    process = cli(
        "prepare-harness",
        "--mbpp-manifest",
        str(mbpp / "manifest.json"),
        "--cohort",
        str(cohort),
        "--output-dir",
        str(prepared),
        cwd=root,
        timeout=120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    manifest = prepared / "manifest.json"
    data = json.loads(manifest.read_text())
    ids = [
        *[f"mbpp/{511 + index}" for index in range(6)],
        "harness/nested-setting",
        "harness/license-audit",
        "harness/deployment-targets",
    ]
    programs = [
        {
            "keyword": "double(n)",
            "target": "solution.py",
            "kind": "short_program",
            "code": "def double(n): return n * 2\n",
            "read_paths": [],
        }
    ]
    for entry in data["tasks"]:
        if entry["task_id"] not in ids or entry["task_type"] == "short_program":
            continue
        task = json.loads((prepared / entry["task_file"]).read_text())
        target = task["implementation_paths"][0]
        programs.append(
            {
                "keyword": target,
                "target": target,
                "kind": entry["task_type"],
                "code": task["reference_implementation"],
                "read_paths": [
                    name
                    for name in task["initial_files"]
                    if name.endswith((".md", ".json"))
                ],
            }
        )
    plans = []
    for name, selected_ids in [
        ("harness", ids),
        ("coding", [f"mbpp/{511 + index}" for index in range(6)]),
    ]:
        path = root / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "version": "development-evaluation-v1",
                    "name": name,
                    "manifest": str(manifest),
                    "split": "dev",
                    "task_ids": selected_ids,
                }
            )
        )
        plans.append(path)
    policy = json.loads(
        (Path(__file__).parents[1] / "configs/sft-acceptance-v1.json").read_text()
    )
    policy.update(
        harness_plan=str(plans[0]),
        coding_plan=str(plans[1]),
        variant_plan=None,
        harness_counts={"short_program": 6, "code_edit": 1, "workspace": 2},
        coding_count=6,
    )
    policy_path = root / "policy.json"
    policy_path.write_text(json.dumps(policy))
    config = root / "model.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "class": "adapters.AcceptanceModel",
                    "programs": programs,
                    "scenario": "base",
                    "identity": "unused",
                }
            }
        )
    )
    training = root / "training"
    training.mkdir()
    (training / "training.json").write_text(
        json.dumps(
            {
                "metadata": {"base_model": "Qwen/Qwen3.5-4B"},
                "checkpoints": [
                    {
                        "optimizer_step": step,
                        "sampler": {
                            "kind": "sampler_weights",
                            "path": f"trio://run/sampler-{step}",
                        },
                        "training_state": {
                            "kind": "training_state",
                            "path": f"trio://run/state-{step}",
                        },
                    }
                    for step in (12, 18, 24, 36, 48, 60, 72)
                ],
            }
        )
    )
    return AcceptanceExperiment(root, policy_path, config, training)


def selection(
    experiment: AcceptanceExperiment,
    output: Path,
    candidates: list[Path],
    reviews: list[Path] | None = None,
) -> dict[str, Any]:
    args = [
        "select-sft",
        str(experiment.training),
        "--baseline",
        str(experiment.evaluate("base")),
        "--evidence-kind",
        "test-double",
        "--output-dir",
        str(output),
    ]
    for path in candidates:
        args.extend(["--evaluation", str(path)])
    for path in reviews or []:
        args.extend(["--benefit-review", str(path)])
    process = cli(*args, cwd=output)
    assert process.returncode == 0, process.stdout + process.stderr
    record: dict[str, Any] = json.loads((output / "selection.json").read_text())
    return record


@pytest.mark.parametrize(
    ("scenario", "step", "decision", "reason"),
    [
        ("format-only", 24, "ineligible", "confirmed_regression:short_program"),
        ("coding-regression", 36, "ineligible", "confirmed_regression:coding"),
        ("class-regression", 48, "insufficient", "unconfirmed_regression:code_edit"),
        (
            "no-observation",
            60,
            "insufficient",
            "reviewed_workspace_task_gain_not_established",
        ),
        (
            "backend-error",
            72,
            "insufficient",
            "backend_token_evidence_error:coding:seed-4",
        ),
    ],
)
def test_task_benefit_cannot_hide_regression_or_missing_state_evidence(
    acceptance_experiment: AcceptanceExperiment,
    tmp_path: Path,
    scenario: str,
    step: int,
    decision: str,
    reason: str,
) -> None:
    experiment = acceptance_experiment
    candidate = experiment.evaluate(scenario, step)
    record = selection(experiment, tmp_path, [candidate])
    assessment = record["candidates"][0]
    assert assessment["decision"] == decision
    assert reason in assessment["reasons"] + assessment["gaps"]
    assert record["student_snapshot"] is None
    if scenario == "class-regression":
        assert assessment["metrics"]["harness"]["net_successes"] == 3
        assert assessment["metrics"]["by_task_type"]["code_edit"]["net_successes"] == -3
    if scenario == "format-only":
        baseline = json.loads(experiment.evaluate("base").read_text())
        for seed in ("4", "17", "29"):
            before = json.loads(
                Path(baseline["runs"]["harness"][seed]["path"]).read_text()
            )
            after = json.loads(
                Path(
                    json.loads(candidate.read_text())["runs"]["harness"][seed]["path"]
                ).read_text()
            )
            assert (
                before["summary"]["by_task_type"]["short_program"]["protocol_errors"]
                == 6
            )
            assert (
                after["summary"]["by_task_type"]["short_program"]["protocol_errors"]
                == 0
            )


def test_qualified_ties_prefer_the_earlier_snapshot_and_test_doubles_never_start_opsd(
    acceptance_experiment: AcceptanceExperiment,
    tmp_path: Path,
) -> None:
    experiment = acceptance_experiment
    candidates = [experiment.evaluate("good", step) for step in (18, 12)]
    record = selection(
        experiment,
        tmp_path,
        candidates,
        [experiment.review(path) for path in candidates],
    )
    assert [item["decision"] for item in record["candidates"]] == [
        "qualified",
        "qualified",
    ]
    assert record["status"] == "test_double_selection"
    assert record["simulation_selected_snapshot"] == "trio://run/sampler-12"
    assert (
        record["student_snapshot"] is None and record["frozen_teacher_snapshot"] is None
    )
    assert record["candidates"][0]["metrics"]["coding"]["task_families"] == 6


@pytest.mark.parametrize(
    "problem", ["conditions", "incomplete", "reused_seed", "screening"]
)
def test_evidence_gaps_never_become_a_qualified_snapshot(
    acceptance_experiment: AcceptanceExperiment,
    tmp_path: Path,
    problem: str,
) -> None:
    experiment = acceptance_experiment
    source = experiment.evaluate("good", 12)
    record = json.loads(source.read_text())
    if problem == "conditions":
        record["conditions"]["harness"]["sampling"]["temperature"] = 0.5
    elif problem == "incomplete":
        record["status"] = "incomplete"
        del record["runs"]["coding"]["17"]
    elif problem == "reused_seed":
        record["runs"]["coding"]["17"] = record["runs"]["coding"]["4"]
    else:
        record["stage"] = "screening"
        record["seeds"] = [4]
        record["runs"] = {
            track: {"4": refs["4"]} for track, refs in record["runs"].items()
        }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(record))
    selected = selection(experiment, tmp_path, [path])
    assessment = selected["candidates"][0]
    assert assessment["decision"] == "insufficient"
    assert assessment["gaps"]
    assert selected["status"] == "no_qualified_snapshot"
    if problem == "screening":
        assert assessment["metrics"]["harness"]["net_successes"] == 2
        assert assessment["screening"]["promising"] is True
