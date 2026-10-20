"""Isolated source registrations for data-role and grading regression tests."""

import hashlib
import json
import shutil
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from bashlet.tasks import preparation, sources

DATA = Path(__file__).parents[1] / "data"


def role_cohort() -> dict[str, Any]:
    return {
        "version": "harness-task-cohort-v2",
        "name": "test-role-cohort",
        "training_candidates": {
            "task_ids": [
                "fixture/sign-runs",
                "harness/header-key",
                "harness/nested-setting",
                "harness/stock-value",
                "harness/license-audit",
            ],
            "per_type": {"short_program": 1, "code_edit": 2, "workspace": 2},
        },
        "development": {
            "task_ids": [
                "fixture/interval-union",
                "fixture/cache-expiry/v2",
                "fixture/release-order",
            ],
            "per_type": 1,
        },
        "test": {
            "task_ids": [
                "fixture/circular-windows",
                "fixture/bounded-drain",
                "fixture/checksum-audit",
            ],
            "per_type": 1,
        },
        "environment_variants": ["harness/license-audit/policy-green"],
    }


@pytest.fixture(scope="module")
def role_sources(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("role-sources")
    for path in DATA.glob("*.json"):
        shutil.copyfile(path, root / path.name)
    registry_path = root / "sources.json"
    registry = json.loads(registry_path.read_text())
    catalog = json.loads(
        (Path(__file__).parent / "fixtures/task-roles.json").read_text()
    )
    revised = deepcopy(catalog)
    revised["revision"] = "fixture-v2"
    revised["families"] = [
        family
        for family in revised["families"]
        if family["original_task_id"] == "cache-expiry"
    ]
    revised["families"][0]["variants"][0]["task"]["task_id"] += "/v2"
    for version, content in enumerate((catalog, revised), start=1):
        filename = f"roles-v{version}.json"
        raw = json.dumps(content).encode()
        (root / filename).write_bytes(raw)
        lock_key = f"roles-v{version}"
        registry["locks"][lock_key] = {
            "dataset": content["dataset"],
            "revision": content["revision"],
            "file": filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        registry["sources"].append(
            {
                "format": "catalog",
                "kind": "model_assisted",
                "lock": lock_key,
                "generator": {"model": "unknown", "model_revision": None},
            }
        )
    registry_path.write_text(json.dumps(registry))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sources, "_DATA", root)
        patch.setattr(sources, "SOURCE_REGISTRY", registry_path)
        patch.setattr(preparation, "SOURCE_REGISTRY", registry_path)
        patch.setenv("BASHLET_TEST_SOURCE_DATA", str(root))
        yield root
