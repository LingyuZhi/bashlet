"""Source-owned admission and manifest selection shared by every task consumer."""

import hashlib
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..mbpp import mbpp_task_path
from ..mbpp import validate_origin as validate_mbpp_origin
from ..models import TaskOrigin

AUTHORED_DATASET = "bashlet/harness-authored"
CONVERSION = "harness-task-families-v1"
_CONFIGS = Path(__file__).resolve().parents[3] / "configs"
if not _CONFIGS.is_dir():
    _CONFIGS = Path(__file__).resolve().parents[1] / "configs"
AUTHORED_LOCK = _CONFIGS / "harness-authored-v1.lock.json"


@lru_cache(maxsize=1)
def authored_source() -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the explicitly admitted, content-locked local source registry."""
    lock = json.loads(AUTHORED_LOCK.read_text())
    data = (AUTHORED_LOCK.parent / lock["file"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != lock["sha256"]:
        raise ValueError("Authored source SHA-256 mismatch")
    source = json.loads(data)
    if (
        lock["dataset"] != AUTHORED_DATASET
        or source["dataset"] != lock["dataset"]
        or source["revision"] != lock["revision"]
    ):
        raise ValueError("Unsupported authored source version")
    families: dict[str, dict[str, Any]] = {}
    task_ids: set[str] = set()
    for family in source["families"]:
        original = family["original_task_id"]
        if (
            family["family_id"] != f"harness/{original}"
            or family["split"] not in ("train", "dev")
            or family["task_type"] not in ("code_edit", "workspace")
            or original in families
        ):
            raise ValueError("Invalid or duplicate authored family registration")
        families[original] = family
        for variant in family["variants"]:
            task_id = variant["task"]["task_id"]
            if task_id in task_ids:
                raise ValueError(f"Duplicate task ID: {task_id}")
            task_ids.add(task_id)
    return lock, families


def validate_origin(origin: TaskOrigin) -> None:
    """Unknown labels are never trusted; variants inherit source-owned splits."""
    if origin.dataset == "google-research/mbpp":
        validate_mbpp_origin(origin)
        return
    if origin.dataset != AUTHORED_DATASET:
        raise ValueError(f"Unsupported task-origin dataset: {origin.dataset}")
    lock, families = authored_source()
    if origin.revision != lock["revision"] or origin.source_sha256 != lock["sha256"]:
        raise ValueError("Unregistered authored source version or content identity")
    family = families.get(origin.original_task_id)
    if family is None:
        raise ValueError(f"Unregistered authored family: {origin.original_task_id}")
    if origin.family_id != family["family_id"] or origin.split != family["split"]:
        raise ValueError(f"Authored family split mismatch: {origin.family_id}")


def task_path(manifest_path: Path, task_id: str, split: str) -> Path:
    """Select a verified task after checking every family's source and split.

    The original MBPP format remains supported with its source-owned ID rules.
    Multi-source manifests use an explicit origin per task, bound to an admitted
    source version. Renames cannot bypass either source or family validation.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("conversion") == "mbpp-json-functions-v1":
        return mbpp_task_path(manifest_path, task_id, split)
    if manifest.get("status") != "complete" or manifest.get("conversion") != CONVERSION:
        raise ValueError("Task set is incomplete or has an unsupported conversion")
    sources = {
        (source["dataset"], source["revision"], source["sha256"])
        for source in manifest["sources"]
    }
    tasks: dict[str, dict[str, Any]] = {}
    families: dict[str, str] = {}
    for entry in manifest["tasks"]:
        origin = TaskOrigin(**entry["origin"])
        validate_origin(origin)
        if (origin.dataset, origin.revision, origin.source_sha256) not in sources:
            raise ValueError("Task origin is absent from the admitted manifest sources")
        if entry["family_id"] != origin.family_id or entry["split"] != origin.split:
            raise ValueError(f"Task family split mismatch: {origin.family_id}")
        if origin.family_id in families and families[origin.family_id] != origin.split:
            raise ValueError(f"Task family split conflict: {origin.family_id}")
        families[origin.family_id] = origin.split
        if entry["task_id"] in tasks:
            raise ValueError(f"Duplicate task ID: {entry['task_id']}")
        if entry["task_type"] not in ("short_program", "code_edit", "workspace"):
            raise ValueError("Unknown task type")
        tasks[entry["task_id"]] = entry
    if task_id not in tasks:
        raise ValueError(f"Unknown task ID: {task_id}")
    selected = tasks[task_id]
    if selected["split"] != split:
        raise ValueError(
            f"Task {task_id} belongs to split {selected['split']}, not {split}"
        )
    if selected["status"] != "verified":
        raise ValueError(f"Task {task_id} is not verified")
    root = manifest_path.parent.resolve()
    path = (root / Path(selected["task_file"])).resolve()
    if not path.is_relative_to(root / "private/tasks"):
        raise ValueError("Task file is outside the private task set")
    if hashlib.sha256(path.read_bytes()).hexdigest() != selected["task_sha256"]:
        raise ValueError("Prepared Task SHA-256 mismatch")
    task = json.loads(path.read_text())
    if task["task_id"] != task_id or task["origin"] != asdict(
        TaskOrigin(**selected["origin"])
    ):
        raise ValueError("Task identity or origin differs from its family manifest")
    if task.get("task_type") is not None and task["task_type"] != selected["task_type"]:
        raise ValueError("Task type differs from its manifest")
    if task.get("variant_id", "base") != selected["variant_id"]:
        raise ValueError("Task variant differs from its manifest")
    return path
