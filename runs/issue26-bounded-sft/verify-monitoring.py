"""Read back allowed metric histories from the existing W&B project."""

import json
from pathlib import Path

import wandb

from bashlet.monitoring import MonitorConfig

root = Path("runs/issue26-bounded-sft")
config = MonitorConfig.read(Path("configs/bounded-harness-monitoring.yaml"))
api = wandb.Api(api_key=config.credential(), timeout=20)
verified = []
for path in sorted(root.rglob("monitor.json")):
    state = json.loads(path.read_text())
    if state.get("sync_status") != "synced":
        continue
    target = f"{state['entity']}/{state['project']}/{state['id']}"
    remote = api.run(target)
    events = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
    history = list(remote.scan_history())
    assert len(history) == len(events), (target, len(history), len(events))
    assert [row["_step"] for row in history] == list(range(len(events)))
    for local, row in zip(events, history, strict=True):
        for key, value in local["metrics"].items():
            assert row.get(key) == value, (target, local["sequence"], key)
    verified.append({
        "local_run": str(path.parent), "url": state["url"],
        "history_rows": len(history), "all_local_metric_rows_match": True,
        "summary": {key: value for key, value in remote.summary.items()
                    if key.startswith(("train/", "eval/", "spend/", "checkpoint/", "run/"))},
    })
(root / "wandb-verification.json").write_text(json.dumps({"verified": verified}, indent=2))
print(json.dumps({"verified_runs": len(verified)}))
