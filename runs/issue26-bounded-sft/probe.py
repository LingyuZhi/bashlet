"""Run the frozen training-task diagnostics through the existing experiment seam."""

import argparse
import json
from pathlib import Path

from bashlet.config import load_config, load_model
from bashlet.experiment import run_experiment, task_from_record
from bashlet.tasks import task_path

parser = argparse.ArgumentParser()
parser.add_argument("step", type=int, choices=(4, 8))
options = parser.parse_args()
root = Path("runs/issue26-bounded-sft")
condition = load_config([
    Path("configs/local-base.yaml"),
    Path(f"configs/bounded-harness-step{options.step}.yaml"),
    Path("configs/acceptance-local-v1.yaml"),
])
model = load_model(condition.model)
results = []
for kind, task_id in (
    ("short_program", "mbpp/623"),
    ("code_edit", "harness/last-record"),
    ("workspace", "harness/stock-value"),
):
    path = task_path(Path("runs/harness-tasks-v1/manifest.json"), task_id, "train")
    task = task_from_record(json.loads(path.read_text()))
    result = run_experiment(
        task=task, model=model, config=condition,
        output_dir=root / f"probes-step{options.step}" / kind,
    )
    record = {"task_id": task_id, "status": result.status,
              "passed": result.passed, "run_dir": str(result.run_dir)}
    results.append(record)
    print(json.dumps(record), flush=True)
(root / f"probes-step{options.step}" / "index.json").write_text(
    json.dumps({"scope": "training-split diagnostics only", "results": results}, indent=2)
)
