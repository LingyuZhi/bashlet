"""Explicit content locks admit public, authored and model-assisted sources."""

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from bashlet.harness.models import Task, TaskOrigin
from bashlet.harness.protocol import fingerprint
from bashlet.tasks import _DATA

SOURCE_REGISTRY = _DATA / "sources.json"
TASK_TYPES = ("short_program", "code_edit", "workspace")
type SourceRegistry = dict[
    tuple[str, str, str], tuple[dict[str, Any], dict[str, Any] | None]
]
_ADMISSIONS: ContextVar[SourceRegistry | None] = ContextVar(
    "preparing_sources", default=None
)
_ARTIFACT_ROOT: ContextVar[Path | None] = ContextVar(
    "source_artifact_root", default=None
)


def definition_bytes(value: object) -> bytes:
    """Canonical bytes preserve the pinned import and generation plan identities."""
    return (
        json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode()


def pinned_definition(record: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = record.get("definition")
    if not isinstance(value, dict):
        raise ValueError("Pinned data definition must contain a mapping")
    content = definition_bytes(value)
    if hashlib.sha256(content).hexdigest() != record.get("sha256"):
        raise ValueError("Pinned data definition SHA-256 mismatch")
    return value, content


def source_definitions(path: Path | None = None) -> dict[str, Any]:
    return _source_definitions((path or SOURCE_REGISTRY).read_bytes())


def _source_definitions(content: bytes) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(content)
    if (
        not isinstance(value, dict)
        or value.get("version") != "task-source-registry-v3"
        or not isinstance(value.get("sources"), list)
        or not isinstance(value.get("locks"), dict)
        or not isinstance(value.get("imports"), dict)
    ):
        raise ValueError("Unsupported or incomplete source definitions")
    return value


def mbpp_source_lock(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_bytes())
    if "locks" in value:
        value = source_definitions(path)["locks"]["mbpp"]
    return value


def import_definition(
    path: Path, catalog: str | None
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    value = json.loads(path.read_bytes())
    if "imports" not in value:
        if catalog is not None:
            raise ValueError("--catalog requires combined source definitions")
        return value, path.read_bytes(), {}
    definitions = source_definitions(path)
    if catalog not in definitions["imports"]:
        choices = ", ".join(definitions["imports"])
        raise ValueError(f"Select --catalog from: {choices}")
    plan, content = pinned_definition(definitions["imports"][catalog])
    return plan, content, definitions["locks"]


def artifact_root(path: Path | None = None) -> Path:
    return (path or _ARTIFACT_ROOT.get() or _DATA.parent / "runs/data").resolve()


def family_guard_bytes() -> bytes:
    cohort = json.loads((_DATA / "cohort.json").read_bytes())
    return definition_bytes(cohort["family_guard"])


def audit_contents(
    cohort: dict[str, Any], root: Path | None = None
) -> list[tuple[dict[str, Any], bytes]]:
    """Require every declared audit before any preparation output is created."""
    verified = []
    missing = []
    for document in cohort.get("audit_documents", []):
        if document.get("embedded") == "generation_plan":
            _, content = pinned_definition(cohort["generation_plan"])
        else:
            path = source_file(document, artifact_directory=root)
            if not path.is_file():
                missing.append(str(path))
                continue
            content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != document["sha256"]:
            raise ValueError("Cohort audit document content changed")
        verified.append((document, content))
    if missing:
        raise FileNotFoundError(
            f"Missing cohort audit documents ({len(missing)}):\n" + "\n".join(missing)
        )
    return verified


@contextmanager
def preparing_sources(artifact_directory: Path | None = None) -> Iterator[None]:
    """Share verified admission within preparation; ordinary loads read bytes."""
    root_token = _ARTIFACT_ROOT.set(artifact_root(artifact_directory))
    try:
        token = _ADMISSIONS.set(admitted_sources())
        try:
            yield
        finally:
            _ADMISSIONS.reset(token)
    finally:
        _ARTIFACT_ROOT.reset(root_token)


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


def source_file(
    lock: dict[str, Any], *, artifact_directory: Path | None = None
) -> Path:
    """Repository registrations may pin generated private catalogs in runs."""
    if "artifact" in lock:
        root = artifact_root(artifact_directory)
        path = (root / str(lock["artifact"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source artifact must remain inside its artifact root")
        return path
    path = (_DATA / Path(str(lock["file"]))).resolve()
    if not any(
        path.is_relative_to(root.resolve())
        for root in (
            _DATA,
            _DATA.parent / "runs",
            _DATA.parent / "outputs",
        )
    ):
        raise ValueError("Registered source must be in data, runs or outputs")
    return path


def admitted_sources(artifact_dir: Path | None = None) -> SourceRegistry:
    """Read repository-owned registrations; a manifest cannot admit itself."""
    snapshot = _ADMISSIONS.get()
    if snapshot is not None:
        return snapshot
    raw = SOURCE_REGISTRY.read_bytes()
    definitions = _source_definitions(raw)
    inputs = []
    for registration in definitions["sources"]:
        key = registration.get("lock")
        if not isinstance(key, str) or key not in definitions["locks"]:
            raise ValueError("Source registration has no pinned lock")
        lock_bytes = definition_bytes(definitions["locks"][key])
        content = b""
        if registration["format"] == "catalog":
            path = source_file(json.loads(lock_bytes))
            if registration.get("optional_artifact") is True and not path.exists():
                lock = json.loads(lock_bytes)
                if artifact_dir is None:
                    continue
                path = artifact_dir / (lock["sha256"] + ".json")
                if not path.is_file():
                    continue
            content = path.read_bytes()
        inputs.append((json.dumps(registration), lock_bytes, content))
    return _admitted_sources(raw, tuple(inputs))


@lru_cache(maxsize=2)
def _admitted_sources(
    raw: bytes, inputs: tuple[tuple[str, bytes, bytes], ...]
) -> SourceRegistry:
    # Cache parsing, never filesystem reads or content identities.
    registry = json.loads(raw)
    if registry["version"] != "task-source-registry-v3":
        raise ValueError("Unsupported source registry version")
    sources: SourceRegistry = {}
    mappings: dict[tuple[str, str], tuple[str, str]] = {}
    task_ids: set[str] = set()
    for registration_json, lock_bytes, content in inputs:
        registration = json.loads(registration_json)
        if registration["kind"] not in ("public", "authored", "model_assisted"):
            raise ValueError("Unknown source provenance kind")
        if registration["kind"] == "model_assisted":
            generator = registration.get("generator")
            if (
                not isinstance(generator, dict)
                or not isinstance(generator.get("model"), str)
                or not generator["model"].strip()
                or "model_revision" not in generator
            ):
                raise ValueError(
                    "Model-assisted admission requires a generator model "
                    "or explicit unknown"
                )
        lock = json.loads(lock_bytes)
        identity = (lock["dataset"], lock["revision"], lock["sha256"])
        if identity in sources:
            raise ValueError("Duplicate source registration")
        families = None
        if registration["format"] == "catalog":
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
    *,
    artifact_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    identity = (origin.dataset, origin.revision, origin.source_sha256)
    source = (
        admitted_sources(artifact_dir) if registrations is None else registrations
    ).get(identity)
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


def _validate_native_task(
    record: dict[str, Any], row: dict[str, Any], origin: TaskOrigin
) -> None:
    from bashlet.tasks.mbpp import _convert

    expected, _ = _convert(row, record["environment_id"], origin)
    native = record | {
        "assignment": None,
        "task_type": None,
        "feedback_start_id": None,
        "environment_variant_id": None,
    }
    if (
        fingerprint(native) != fingerprint(asdict(expected))
        or record["task_type"] != "short_program"
    ):
        raise ValueError("Task content differs from its registered MBPP source record")


def validate_registered_task(task: Task) -> None:
    """Bind catalog tasks to reviewed content, including references and graders."""
    assert task.origin is not None
    artifact_dir = (
        Path(task.assignment.manifest_path).parent / "private/sources"
        if task.assignment is not None
        else None
    )
    lock, family = registered_origin(task.origin, artifact_dir=artifact_dir)
    if family is None:
        from bashlet.tasks.versions import source_bytes

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
        _validate_native_task(asdict(task), row, task.origin)
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
    task_record: dict[str, Any] | None = None,
    native_record: dict[str, Any] | None = None,
) -> None:
    """Check every pool row, even when it is not selected for this run."""
    lock, family = registered_origin(origin, registrations)
    if family is None:
        if task_record is not None:
            if native_record is None:
                raise ValueError("Frozen native task source record is missing")
            _validate_native_task(task_record, native_record, origin)
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
    if task_record is not None:
        expected_record = catalog_task_record(
            lock, family, variant, task_record["environment_id"]
        )
        if any(
            fingerprint(task_record[key]) != fingerprint(value)
            for key, value in expected_record.items()
        ):
            raise ValueError("Frozen pool task content changed")
