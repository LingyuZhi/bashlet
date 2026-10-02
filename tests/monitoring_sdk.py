"""External W&B SDK double used by CLI integration tests."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class Config(dict[str, Any]):
    def update(self, values: Any = (), /, **kwargs: Any) -> None:
        super().update(values)


class Run:
    def __init__(self, **kwargs: Any) -> None:
        self.id = kwargs["id"]
        self.entity = kwargs.get("entity") or "test-account"
        self.project = kwargs["project"]
        self.url = f"https://wandb.invalid/{self.entity}/{self.project}/{self.id}"
        self.root = Path(kwargs["dir"])
        self.settings = SimpleNamespace(sync_dir=str(self.root / "wandb/cache"))
        self.step = 0
        self.config = Config(kwargs["config"])
        self.settings_input = {
            k: v for k, v in vars(kwargs["settings"]).items() if k != "api_key"
        }
        self.rows: list[dict[str, Any]] = []
        if (self.root / "remote.json").exists():
            previous = json.loads((self.root / "remote.json").read_text())
            if previous["id"] == self.id:
                self.rows = previous["rows"]
                self.step = max((r["_step"] + 1 for r in self.rows), default=0)
        self._save()

    def _save(self) -> None:
        (self.root / "remote.json").write_text(
            json.dumps(
                {
                    "config": self.config,
                    "settings": self.settings_input,
                    "rows": self.rows,
                    "id": self.id,
                }
            )
        )

    def log(self, data: dict[str, Any], *, step: int) -> None:
        if os.environ.get("FAKE_WANDB_LOG_FAIL"):
            raise ConnectionError("network unavailable")
        self.rows.append(data | {"_step": step})
        self.step = step + 1
        self._save()

    def define_metric(self, *args: Any, **kwargs: Any) -> None:
        pass

    def finish(self, **kwargs: Any) -> None:
        self._save()


current_run: Run


class CommError(Exception):
    def __init__(self) -> None:
        self.exc = SimpleNamespace(response=SimpleNamespace(status_code=403))


def init(**kwargs: Any) -> Run:
    if os.environ.get("FAKE_WANDB_AUTH_FAIL"):
        raise PermissionError("authentication failed")
    if os.environ.get("FAKE_WANDB_REVOKED"):
        raise CommError()
    global current_run
    current_run = Run(**kwargs)
    return current_run


Settings = SimpleNamespace


class Api:
    def __init__(self, **kwargs: Any) -> None:
        self.default_entity = "test-account"

    def run(self, path: str) -> Any:
        if os.environ.get("FAKE_WANDB_HISTORY_FAIL"):
            raise ConnectionError("history unavailable after successful init")
        return SimpleNamespace(lastHistoryStep=current_run.step - 1)
