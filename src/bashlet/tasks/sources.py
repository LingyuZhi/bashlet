"""Explicit content locks admit public, authored and model-assisted sources."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ..models import Task, TaskOrigin
from ..protocol import fingerprint
from . import _CONFIGS

SOURCE_REGISTRY = _CONFIGS / "task-sources-v2.json"
TASK_TYPES = ("short_program", "code_edit", "workspace")
type SourceRegistry = dict[
    tuple[str, str, str], tuple[dict[str, Any], dict[str, Any] | None]
]

WORKSPACE_GRADING_DRIVER = """import os, shutil, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    shutil.copytree('/workspace', directory + '/task')
    os.chdir(directory + '/task')
    for path in Path('.').rglob('*'):
        if path.is_file(): path.chmod(0o600)
    source = Path('_grading/cases.py').read_text()
    exec(compile(source, '_grading/cases.py', 'exec'))
"""


def _local_file(name: str) -> Path:
    path = (_CONFIGS / name).resolve()
    if not path.is_relative_to(_CONFIGS.resolve()):
        raise ValueError("Source registration must remain inside configs")
    return path


def admitted_sources() -> SourceRegistry:
    """Read repository-owned registrations; a manifest cannot admit itself."""
    registry = json.loads(SOURCE_REGISTRY.read_text())
    if registry["version"] != "task-source-registry-v2":
        raise ValueError("Unsupported source registry version")
    sources: SourceRegistry = {}
    mappings: dict[tuple[str, str], tuple[str, str]] = {}
    task_ids: set[str] = set()
    for registration in registry["sources"]:
        if registration["kind"] not in ("public", "authored", "model_assisted"):
            raise ValueError("Unknown source provenance kind")
        lock = json.loads(_local_file(registration["lock_file"]).read_text())
        identity = (lock["dataset"], lock["revision"], lock["sha256"])
        if identity in sources:
            raise ValueError("Duplicate source registration")
        families = None
        if registration["format"] == "catalog":
            content = _local_file(lock["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != lock["sha256"]:
                raise ValueError("Registered source content SHA-256 mismatch")
            catalog = json.loads(content)
            if (catalog["dataset"], catalog["revision"]) != identity[:2]:
                raise ValueError("Registered source version mismatch")
            families = {}
            for family in catalog["families"]:
                original = family["original_task_id"]
                if (
                    original in families
                    or not original
                    or not family["family_id"]
                    or family["split"] not in ("train", "dev", "test", "unsplit")
                    or family["task_type"] not in TASK_TYPES
                ):
                    raise ValueError("Invalid registered source family")
                mapping = (family["family_id"], family["split"])
                key = (lock["dataset"], original)
                if key in mappings and mappings[key] != mapping:
                    raise ValueError("Conflicting registered family mapping")
                mappings[key] = mapping
                families[original] = family
                for variant in family["variants"]:
                    task_id = variant["task"]["task_id"]
                    if task_id in task_ids:
                        raise ValueError("Duplicate registered task identity")
                    task_ids.add(task_id)
        elif (
            registration["format"] != "mbpp"
            or lock["dataset"] != "google-research/mbpp"
        ):
            raise ValueError("Unsupported registered source format")
        sources[identity] = (lock, families)
    return sources


def registered_origin(
    origin: TaskOrigin,
    registrations: SourceRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    identity = (origin.dataset, origin.revision, origin.source_sha256)
    source = (admitted_sources() if registrations is None else registrations).get(
        identity
    )
    if source is None:
        raise ValueError("Unregistered source, source version or content identity")
    lock, families = source
    family = None if families is None else families.get(origin.original_task_id)
    if families is not None and (
        family is None
        or origin.family_id != family["family_id"]
        or origin.split != family["split"]
    ):
        raise ValueError("Conflicting registered family mapping or original split")
    return lock, family


def catalog_task_record(
    lock: dict[str, Any],
    family: dict[str, Any],
    variant: dict[str, Any],
    environment_id: str,
) -> dict[str, Any]:
    data: dict[str, Any] = variant["task"] | {
        "origin": asdict(
            TaskOrigin(
                lock["dataset"],
                lock["revision"],
                lock["sha256"],
                family["original_task_id"],
                family["family_id"],
                family["split"],
            )
        ),
        "task_type": family["task_type"],
        "variant_id": variant["variant_id"],
        "environment_id": environment_id,
    }
    if family["task_type"] == "workspace":
        data["grader"] = data["grader"] | {
            "hidden_files": data["grader"]["hidden_files"]
            | {
                "_grading/check.py": WORKSPACE_GRADING_DRIVER,
            },
        }
    return data


def validate_registered_task(task: Task) -> None:
    """Bind catalog tasks to reviewed content, including references and graders."""
    assert task.origin is not None
    lock, family = registered_origin(task.origin)
    if family is None:
        from ..mbpp import _convert
        from .versions import source_bytes

        row = next(
            (
                row
                for row in json.loads(source_bytes(task))
                if str(row["task_id"]) == task.origin.original_task_id
            ),
            None,
        )
        if row is None:
            raise ValueError("Task original identity is absent from its native source")
        native_expected, _ = _convert(row, task.environment_id, task.origin)
        native = replace(
            task,
            assignment=None,
            task_type=None,
            feedback_start_id=None,
            environment_variant_id=None,
        )
        if (
            fingerprint(asdict(native)) != fingerprint(asdict(native_expected))
            or task.task_type != "short_program"
        ):
            raise ValueError(
                "Task content differs from its registered MBPP source record"
            )
        return
    variant = next(
        (v for v in family["variants"] if v["task"]["task_id"] == task.task_id), None
    )
    if variant is None:
        raise ValueError(
            "Unregistered task variant; register its inherited family first"
        )
    expected = catalog_task_record(lock, family, variant, task.environment_id)
    actual = asdict(task)
    if any(
        fingerprint(actual[key]) != fingerprint(value)
        for key, value in expected.items()
    ):
        raise ValueError("Task content differs from its registered source record")


def validate_source_entry(
    origin: TaskOrigin,
    entry: dict[str, Any],
    registrations: SourceRegistry,
) -> None:
    """Check every pool row, even when it is not selected for this run."""
    _, family = registered_origin(origin, registrations)
    if family is None:
        return
    variant = next(
        (v for v in family["variants"] if v["task"]["task_id"] == entry["task_id"]),
        None,
    )
    if (
        variant is None
        or entry["source_record_sha256"] != fingerprint(variant)
        or entry["task_type"] != family["task_type"]
        or entry["variant_id"] != variant["variant_id"]
    ):
        raise ValueError("Task entry differs from its registered source record")
