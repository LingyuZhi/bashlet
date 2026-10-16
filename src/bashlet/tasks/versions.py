"""Bind standalone tasks to frozen roles and scope unpublished quality checks."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bashlet.harness.models import Task
from bashlet.harness.protocol import fingerprint

GROUP_ROLES = (
    ("training_candidates", "train_candidate"),
    ("development", "dev"),
    ("test", "test"),
)


@dataclass(frozen=True)
class _Preparation:
    manifest_path: Path
    cohort: dict[str, Any]
    cohort_sha256: str
    mbpp_manifest: Path | None


_PREPARING: ContextVar[_Preparation | None] = ContextVar(
    "task_preparation", default=None
)


@contextmanager
def preparing_cohort(
    cohort_path: Path,
    manifest_path: Path,
    mbpp_manifest: Path | None,
) -> Iterator[None]:
    """Only the preparation entry point admits unpublished role assignments."""
    content = cohort_path.read_bytes()
    cohort = json.loads(content)
    if cohort["version"] != "harness-task-cohort-v2":
        yield
        return
    state = _Preparation(
        manifest_path.resolve(),
        cohort,
        hashlib.sha256(content).hexdigest(),
        mbpp_manifest,
    )
    token = _PREPARING.set(state)
    try:
        yield
    finally:
        _PREPARING.reset(token)


def validate_assignment(task: Task) -> None:
    """Role claims need the admitted cohort or a complete frozen manifest."""
    from bashlet.tasks import task_path, task_split
    from bashlet.tasks.sources import registered_origin

    assignment = task.assignment
    assert assignment is not None and task.origin is not None
    manifest_path = Path(assignment.manifest_path)
    if not assignment.manifest_path or not manifest_path.is_absolute():
        raise ValueError("Versioned role requires its frozen manifest path")
    preparation = _PREPARING.get()
    if preparation is not None and manifest_path == preparation.manifest_path:
        cohort = preparation.cohort
        ids = {task.task_id}
        if task.task_id in cohort["environment_variants"]:
            _, family = registered_origin(task.origin)
            if family is None:
                raise ValueError("Unregistered environment variant")
            ids = {
                v["task"]["task_id"]
                for v in family["variants"]
                if v["variant_id"] == "base"
            }
        roles = [
            role
            for group, role in GROUP_ROLES
            if ids.intersection(cohort[group]["task_ids"])
        ]
        if (
            assignment.version != cohort["name"]
            or assignment.cohort_sha256 != preparation.cohort_sha256
            or roles != [assignment.role]
        ):
            raise ValueError("Task role differs from the admitted cohort")
        return
    path = task_path(manifest_path, task.task_id, str(task_split(task)))
    if fingerprint(asdict(task)) != fingerprint(json.loads(path.read_text())):
        raise ValueError("Task content or role differs from its frozen manifest")


def source_bytes(task: Task) -> bytes:
    """Read the pinned native source from preparation or its frozen artifact."""
    assignment, origin = task.assignment, task.origin
    assert assignment is not None and origin is not None
    manifest_path = Path(assignment.manifest_path)
    if not assignment.manifest_path or not manifest_path.is_absolute():
        raise ValueError("Versioned role requires its frozen manifest path")
    preparation = _PREPARING.get()
    if preparation is not None and manifest_path == preparation.manifest_path:
        if preparation.mbpp_manifest is None:
            raise ValueError("Native MBPP tasks require their original source")
        path = preparation.mbpp_manifest.parent / "private/source.json"
    else:
        manifest = json.loads(manifest_path.read_text())
        artifact = next(
            (
                source
                for source in manifest["source_artifacts"]
                if (source["dataset"], source["revision"], source["sha256"])
                == (origin.dataset, origin.revision, origin.source_sha256)
            ),
            None,
        )
        if artifact is None:
            raise ValueError("Task source is absent from the frozen manifest")
        path = (manifest_path.parent / artifact["file"]).resolve()
        if not path.is_relative_to(
            (manifest_path.parent / "private/sources").resolve()
        ):
            raise ValueError("Frozen source is outside the private source set")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != origin.source_sha256:
        raise ValueError("Native source content identity changed")
    return content
