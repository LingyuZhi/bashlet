"""Source-owned admission and manifest selection shared by every task consumer."""

import hashlib
import json
import re
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..mbpp import mbpp_task_path
from ..mbpp import validate_origin as validate_mbpp_origin
from ..models import Task, TaskAssignment, TaskOrigin
from ..protocol import fingerprint

AUTHORED_DATASET = "bashlet/harness-authored"
CONVERSION = "harness-task-families-v1"
VERSIONED_CONVERSION = "harness-task-families-v2"
_CONFIGS = Path(__file__).resolve().parents[3] / "configs"
if not _CONFIGS.is_dir():
    _CONFIGS = Path(__file__).resolve().parents[1] / "configs"
AUTHORED_LOCK = _CONFIGS / "harness-authored-v1.lock.json"
FAMILY_GUARD = _CONFIGS / "harness-family-guard-v1.json"


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


def validate_origin(origin: TaskOrigin, *, artifact_dir: Path | None = None) -> None:
    """Unknown labels are never trusted; variants inherit source-owned splits."""
    if origin.dataset == "google-research/mbpp":
        validate_mbpp_origin(origin)
        return
    if origin.dataset != AUTHORED_DATASET:
        from .sources import registered_origin

        registered_origin(origin, artifact_dir=artifact_dir)
        return
    lock, families = authored_source()
    if origin.revision != lock["revision"] or origin.source_sha256 != lock["sha256"]:
        raise ValueError("Unregistered authored source version or content identity")
    family = families.get(origin.original_task_id)
    if family is None:
        raise ValueError(f"Unregistered authored family: {origin.original_task_id}")
    if origin.family_id != family["family_id"] or origin.split != family["split"]:
        raise ValueError(f"Authored family split mismatch: {origin.family_id}")


@lru_cache(maxsize=1)
def _family_reservations() -> dict[tuple[str, str], tuple[str, str]]:
    guard = json.loads(FAMILY_GUARD.read_text())
    if guard["version"] != "harness-family-guard-v1":
        raise ValueError("Unsupported family guard version")
    reservations: dict[tuple[str, str], tuple[str, str]] = {}
    for group in guard["equivalent_families"]:
        for member in group["members"]:
            key = (group["dataset"], member)
            value = (group["split"], group["family_id"])
            if key in reservations and reservations[key] != value:
                raise ValueError("Conflicting equivalent family reservations")
            reservations[key] = value
    return reservations


def validate_harness_origin(origin: TaskOrigin) -> None:
    """Apply the frozen cohort's equivalence reservations to supplements too.

    Native MBPP origins retain their original ID/split for legacy data. Harness
    cohorts additionally reject known train duplicates of reserved dev families.
    """
    validate_origin(origin)
    reservation = _family_reservations().get((origin.dataset, origin.family_id))
    if reservation is not None and origin.split != reservation[0]:
        raise ValueError(
            f"MBPP equivalent family {origin.family_id} is reserved to "
            f"{reservation[0]} with {reservation[1]}"
        )


def canonical_family(origin: TaskOrigin) -> str:
    """Known equivalents keep native origins but share one counting identity."""
    reservation = _family_reservations().get((origin.dataset, origin.family_id))
    return reservation[1] if reservation is not None else origin.family_id


def task_family(task: Task) -> str:
    if task.origin is None:
        raise ValueError("Task family requires an origin")
    return task.assignment.family_id if task.assignment else task.origin.family_id


def task_split(task: Task) -> str | None:
    if task.assignment is not None:
        return (
            "train"
            if task.assignment.role == "train_candidate"
            else task.assignment.role
        )
    return task.origin.split if task.origin else None


def validate_task(task: Task, *, harness: bool = False) -> None:
    """Validate original lineage and the separately declared current role."""
    if task.origin is None:
        if task.assignment is not None:
            raise ValueError("A versioned task requires a registered origin")
        return
    artifact_dir = (
        Path(task.assignment.manifest_path).parent / "private/sources"
        if task.assignment is not None
        else None
    )
    validate_origin(task.origin, artifact_dir=artifact_dir)
    assignment = task.assignment
    if assignment is None:
        if task.origin.dataset not in ("google-research/mbpp", AUTHORED_DATASET):
            raise ValueError("New task sources require a versioned role assignment")
        if harness:
            validate_harness_origin(task.origin)
        return
    from .sources import validate_registered_task
    from .versions import validate_assignment

    validate_registered_task(task)
    if (
        not assignment.version.strip()
        or re.fullmatch(r"[0-9a-f]{64}", assignment.cohort_sha256) is None
        or assignment.role not in ("train_candidate", "dev", "test")
        or assignment.family_id != canonical_family(task.origin)
        or task.task_type not in ("short_program", "code_edit", "workspace")
    ):
        raise ValueError("Invalid versioned task role or conflicting family mapping")
    validate_assignment(task)


def validate_training_task(task: Task) -> None:
    validate_task(task)
    if task.origin is None or task_split(task) != "train":
        raise ValueError(
            "Training requires a training-split task or training candidate"
        )
    if task.assignment is not None:
        path = Path(task.assignment.manifest_path)
        if (
            not path.is_file()
            or json.loads(path.read_text()).get("status") != "complete"
        ):
            raise ValueError("Training requires a complete frozen task manifest")


def _verified_task_path(
    manifest_path: Path,
    selected: dict[str, Any],
    expected: TaskOrigin,
    task_id: str,
    split: str,
) -> Path:
    """Shared private-path, content and identity checks for both manifest formats."""
    if selected["split"] != split:
        raise ValueError(
            f"Task {task_id} belongs to split {selected['split']}, not {split}"
        )
    if selected["status"] != "verified":
        reason = selected.get("reason", "not verified")
        raise ValueError(f"Task {task_id} is excluded: {reason}")
    root = manifest_path.parent.resolve()
    path = (root / Path(selected["task_file"])).resolve()
    if not path.is_relative_to(root / "private/tasks"):
        raise ValueError("Task file is outside the private task set")
    if hashlib.sha256(path.read_bytes()).hexdigest() != selected["task_sha256"]:
        raise ValueError("Prepared Task SHA-256 mismatch")
    task = json.loads(path.read_text())
    if task["task_id"] != task_id or task["origin"] != asdict(expected):
        raise ValueError("Task identity or origin differs from its family manifest")
    if task.get("assignment") != selected.get("assignment"):
        raise ValueError("Task role differs from its versioned manifest")
    validate_origin(expected, artifact_dir=root / "private/sources")
    if "task_type" in selected and task.get("task_type") is not None:
        if task["task_type"] != selected["task_type"]:
            raise ValueError("Task type differs from its manifest")
    if (
        "variant_id" in selected
        and task.get("variant_id", "base") != selected["variant_id"]
    ):
        raise ValueError("Task variant differs from its manifest")
    return path


def _versioned_roles(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, str]:
    """Every frozen candidate remains reserved, including unselected tasks."""
    private = manifest_path.parent / "private"
    cohort_bytes = (private / "cohort.json").read_bytes()
    cohort = json.loads(cohort_bytes)
    registry_bytes = (private / "source-registry.json").read_bytes()
    if hashlib.sha256(registry_bytes).hexdigest() != manifest["source_registry_sha256"]:
        raise ValueError("Frozen source registration content identity changed")
    identities = {
        (source["dataset"], source["revision"], source["sha256"])
        for source in manifest["sources"]
    }
    artifacts = manifest["source_artifacts"]
    if identities != {
        (source["dataset"], source["revision"], source["sha256"])
        for source in artifacts
    }:
        raise ValueError("Frozen source content is missing")
    for source in artifacts:
        path = (manifest_path.parent / source["file"]).resolve()
        if (
            not path.is_relative_to((private / "sources").resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
        ):
            raise ValueError("Frozen source content identity changed")
    if (
        hashlib.sha256(cohort_bytes).hexdigest() != manifest["source"]["sha256"]
        or cohort != manifest["cohort"]
        or cohort["version"] != "harness-task-cohort-v2"
        or cohort["name"] != manifest["source"]["revision"]
        or hashlib.sha256((private / "family-guard.json").read_bytes()).hexdigest()
        != manifest["family_guard_sha256"]
    ):
        raise ValueError("Frozen cohort or family guard content identity changed")
    roles: dict[str, str] = {}
    expected_audits = {d["sha256"] for d in cohort.get("audit_documents", [])}
    artifacts_audits = manifest.get("audit_artifacts", [])
    if expected_audits != {d["sha256"] for d in artifacts_audits}:
        raise ValueError("Frozen cohort audit evidence is missing")
    for document in artifacts_audits:
        path = (manifest_path.parent / document["file"]).resolve()
        if (
            not path.is_relative_to((private / "audits").resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != document["sha256"]
        ):
            raise ValueError("Frozen cohort audit evidence changed")
    for group, role in (
        ("training_candidates", "train_candidate"),
        ("development", "dev"),
        ("test", "test"),
    ):
        ids = cohort[group]["task_ids"]
        if len(ids) != len(set(ids)) or any(identity in roles for identity in ids):
            raise ValueError("Conflicting frozen family roles")
        roles.update(dict.fromkeys(ids, role))
    variants = cohort["environment_variants"]
    entries = {entry["task_id"]: entry for entry in manifest["tasks"]}
    if set(entries) != set(roles) | set(variants) or len(entries) != len(
        manifest["tasks"]
    ):
        raise ValueError(
            "Task set differs from the whole frozen candidate/dev/test pool"
        )
    family_roles: dict[str, str] = {}
    from .audit import validate_independence

    validate_independence(cohort, [entries[i]["family_id"] for i in roles])
    for identity, role in roles.items():
        family = entries[identity]["family_id"]
        if family in family_roles:
            raise ValueError("Frozen lists require independent task families")
        family_roles[family] = role
    for identity in variants:
        family = entries[identity]["family_id"]
        if identity in roles or family not in family_roles:
            raise ValueError("Variant must inherit a frozen base family")
        roles[identity] = family_roles[family]
    from .freeze import validate_selection

    validate_selection(manifest_path.parent, manifest)
    return roles


def task_path(manifest_path: Path, task_id: str, split: str) -> Path:
    """Select a verified task after checking every family's source and split.

    The original MBPP format remains supported with its source-owned ID rules.
    Multi-source manifests use an explicit origin per task, bound to an admitted
    source version. Renames cannot bypass either source or family validation.
    """
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("conversion") == "mbpp-json-functions-v1":
        return mbpp_task_path(manifest_path, task_id, split)
    versioned = manifest.get("conversion") == VERSIONED_CONVERSION
    if manifest.get("status") != "complete" or manifest.get("conversion") not in (
        CONVERSION,
        VERSIONED_CONVERSION,
    ):
        raise ValueError("Task set is incomplete or has an unsupported conversion")
    sources = {
        (source["dataset"], source["revision"], source["sha256"])
        for source in manifest["sources"]
    }
    tasks: dict[str, dict[str, Any]] = {}
    families: dict[str, str] = {}
    roles = _versioned_roles(manifest_path, manifest) if versioned else {}
    from .sources import admitted_sources, validate_source_entry

    registrations = (
        admitted_sources(manifest_path.parent / "private/sources") if versioned else {}
    )
    native_rows: dict[tuple[str, str, str, str], str] = {}
    native_records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    if versioned:
        if not sources <= registrations.keys():
            raise ValueError("Manifest contains an unregistered source version")
        for source in manifest["source_artifacts"]:
            if source["dataset"] == "google-research/mbpp":
                rows = json.loads((manifest_path.parent / source["file"]).read_text())
                for row in rows:
                    native_records[
                        (
                            source["dataset"],
                            source["revision"],
                            source["sha256"],
                            str(row["task_id"]),
                        )
                    ] = row
                    native_rows[
                        (
                            source["dataset"],
                            source["revision"],
                            source["sha256"],
                            str(row["task_id"]),
                        )
                    ] = fingerprint(row)
    for entry in manifest["tasks"]:
        origin = TaskOrigin(**entry["origin"])
        assignment = TaskAssignment(**entry["assignment"]) if versioned else None
        if versioned:
            assert assignment is not None
            if (
                assignment.version != manifest["source"]["revision"]
                or assignment.cohort_sha256 != manifest["source"]["sha256"]
                or assignment.family_id != canonical_family(origin)
                or assignment.role not in ("train_candidate", "dev", "test")
                or assignment.role != roles[entry["task_id"]]
            ):
                raise ValueError(
                    "Conflicting versioned family mapping or data identity"
                )
            if origin.dataset == "google-research/mbpp":
                validate_mbpp_origin(origin)
                if (
                    native_rows.get(
                        (
                            origin.dataset,
                            origin.revision,
                            origin.source_sha256,
                            origin.original_task_id,
                        )
                    )
                    != entry["source_record_sha256"]
                ):
                    raise ValueError("MBPP original source record identity changed")
            record = None
            if "freeze" in manifest["cohort"]:
                record = json.loads(
                    (manifest_path.parent / entry["task_file"]).read_text()
                )
            validate_source_entry(
                origin,
                entry,
                registrations,
                record,
                native_records.get(
                    (
                        origin.dataset,
                        origin.revision,
                        origin.source_sha256,
                        origin.original_task_id,
                    )
                ),
            )
        else:
            validate_harness_origin(origin)
        if (origin.dataset, origin.revision, origin.source_sha256) not in sources:
            raise ValueError("Task origin is absent from the admitted manifest sources")
        family = assignment.family_id if assignment else origin.family_id
        role_split = (
            ("train" if assignment.role == "train_candidate" else assignment.role)
            if assignment
            else origin.split
        )
        if entry["family_id"] != family or entry["split"] != role_split:
            raise ValueError(f"Task family split mismatch: {origin.family_id}")
        if family in families and families[family] != role_split:
            raise ValueError(f"Task family split conflict: {origin.family_id}")
        families[family] = role_split
        if entry["task_id"] in tasks:
            raise ValueError(f"Duplicate task ID: {entry['task_id']}")
        if entry["task_type"] not in ("short_program", "code_edit", "workspace"):
            raise ValueError("Unknown task type")
        tasks[entry["task_id"]] = entry
    if task_id not in tasks:
        raise ValueError(f"Unknown task ID: {task_id}")
    selected = tasks[task_id]
    return _verified_task_path(
        manifest_path,
        selected,
        TaskOrigin(**selected["origin"]),
        task_id,
        "train" if split == "train_candidate" else split,
    )
