"""Join recorded experiment artifacts without reimplementing selection rules."""

import hashlib
import json
from pathlib import Path

from bashlet.evaluation.harness import effective_comparison

root = Path("runs/issue26-bounded-sft")

def read(path):
    return json.loads(path.read_text())

def reference(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

baseline_path = Path("runs/issue25-acceptance/base-v1/acceptance.json")
baseline = read(baseline_path)
selection = read(root / "selection-verified/selection.json")
training = read(root / "training/training.json")
records = {"base": (baseline_path, baseline)}
for step in (1, 4, 8):
    path = root / f"step{step}-screening/acceptance.json"
    records[f"step{step}"] = (path, read(path))

tracks = {}
task_rows = []
for name, (path, suite) in records.items():
    assert suite["status"] == "complete" and suite["seeds"] == [4]
    tracks[name] = {"suite": reference(path), "snapshot": suite["snapshot"], "tracks": {}}
    for track, refs in suite["runs"].items():
        ref = refs["4"]
        evaluation_path = Path(ref["path"])
        evaluation = read(evaluation_path)
        assert evaluation["status"] == "complete"
        if name != "base":
            control = read(Path(baseline["runs"][track]["4"]["path"]))
            assert effective_comparison(control) == effective_comparison(evaluation)
            assert evaluation["snapshot_details"]["identity_verified"]
        tracks[name]["tracks"][track] = reference(evaluation_path) | {
            "summary": evaluation["summary"],
        }
        for row in evaluation["tasks"]:
            task_rows.append({"condition": name, "track": track, "seed": 4,
                              "task_id": row["task_id"],
                              "family_id": row["task_origin"]["family_id"],
                              "task_type": row["task_type"], "outcome": row["outcome"],
                              "run": str(evaluation_path.parent / row["run"]),
                              "episode_sha256": row["episode_sha256"]})

assert all(selection[key] is None for key in (
    "student_snapshot", "frozen_teacher_snapshot", "sft_training_state",
))
assert not any(candidate["screening"]["promising"] for candidate in selection["candidates"])
assert training["status"] == "complete" and training["data_progress"]["epochs_processed"] == 1
report = {
    "schema_version": 1, "issue": 26,
    "status": "completed-pilot; insufficient-qualification-evidence",
    "conclusion": "insufficient_evidence", "qualified_snapshot": None,
    "student_snapshot": None, "frozen_teacher_snapshot": None, "OPSD_executed": False,
    "preflight": reference(root / "preflight.json"),
    "config": reference(Path("configs/bounded-harness-sft-v1.yaml")),
    "dataset_manifest": reference(Path(training["config"]["dataset"])),
    "dataset_sha256": training["metadata"]["data_sha256"],
    "frozen_acceptance_sha256": baseline["frozen_sha256"],
    "training": reference(root / "training/training.json"),
    "training_config": training["metadata"]["training"],
    "training_volume": training["data_progress"],
    "checkpoints": training["checkpoints"], "loss_updates": training["updates"],
    "results": tracks, "per_task": "per-task.json", "selection": reference(root / "selection-verified/selection.json"),
    "selection_verification": reference(root / "selection-verification.json"),
    "code_review": reference(root / "code-review.md"),
    "confirmation": {"frozen_seeds": [4, 17, 29], "executed_seeds": [4],
                     "reason": "Every candidate fails the frozen promising screen: no workspace task gain. Do not add seeds to fish for qualification."},
    "environment_or_feedback_task_benefit": "not established under the frozen workspace criterion",
    "coding_preservation": "one-seed count matches baseline at steps4/8; no three-seed preservation claim",
    "spend": reference(root / "spend.json"),
    "new_paid_upper_bound_cny": training["cost"]["charged_upper_bound_cny"],
    "actual_paid_cny": None,
    "validation": reference(root / "validation.json"),
    "training_verification": reference(root / "training-verification.json"),
    "behavior_reviews": [reference(root / f"step{step}-behavior-review.json") for step in (1, 4, 8)],
    "coverage_limitation": reference(root / "coverage-diagnosis.json"),
    "historical_training_records": [
        reference(root / f"training/step{step}-record.json") for step in (1, 4)
    ],
    "historical_record_provenance": "Reconstructed from the completed journal; each byte hash matches the training-record hash captured by its original export.",
    "immutable_training_records": [
        reference(path) for path in sorted((root / "training/training-records").glob("*.json"))
    ],
    "monitoring_verification": reference(root / "wandb-verification.json") | {
        "verified_runs": len(read(root / "wandb-verification.json")["verified"])
    },
    "service_restoration": reference(root / "service-restored.json") | {
        "scope": "pre-existing diagnostic service; no qualified initializer",
    },
    "scope": "8 frozen training families (6/1/1 types), 30 Harness and 37 coding development tasks; shared families/seeds/variants are not independent tasks. No held-out testing or causal attribution to old templates or learning rate.",
}
(root / "per-task.json").write_text(json.dumps({"rows": task_rows}, indent=2))
(root / "acceptance.json").write_text(json.dumps(report, indent=2))
print(json.dumps({"conclusion": report["conclusion"], "per_task_rows": len(task_rows),
                  "new_paid_upper_cny": report["new_paid_upper_bound_cny"]}))
