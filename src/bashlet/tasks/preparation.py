"""Freeze mixed-source cohorts with actual reference and mutation execution."""

import ast
import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from ..config import ExperimentConfig, config_record
from ..experiment import run_experiment, serialize_task, task_from_record
from ..mbpp import _convert
from ..models import FunctionGrader, Task, TaskAssignment, TaskOrigin
from ..protocol import fingerprint
from ..replay import ReplayModel, Utf8Encoding
from . import (
    _CONFIGS,
    AUTHORED_LOCK,
    CONVERSION,
    FAMILY_GUARD,
    VERSIONED_CONVERSION,
    authored_source,
    canonical_family,
    task_family,
    task_path,
    task_split,
    validate_task,
)
from .sources import (
    SOURCE_REGISTRY,
    admitted_sources,
    catalog_task_record,
    registered_origin,
)

DEFAULT_COHORT = _CONFIGS / "harness-cohort-v1.json"
_TYPES = ("short_program", "code_edit", "workspace")


def _family(task: Task) -> str:
    return task_family(task)


@dataclass(frozen=True)
class _Definition:
    task: Task
    wrong_files: dict[str, str]
    wrong_reason: str
    source_record_sha256: str


def _write(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    temporary.replace(path)


def _identities(task: Task) -> Task:
    """A broken start and an environment variant have distinct content IDs."""
    return replace(
        task,
        feedback_start_id=fingerprint(
            {
                "initial_files": task.initial_files,
                "feedback_command": task.feedback_command,
            }
        ),
        environment_variant_id=fingerprint(
            {
                "environment_id": task.environment_id,
                "context_files": {
                    p: c
                    for p, c in task.initial_files.items()
                    if p not in task.implementation_paths
                },
            }
        ),
    )


def _mbpp_definition(
    manifest: Path,
    task_id: str,
    split: str,
    *,
    versioned: bool = False,
) -> _Definition:
    task = task_from_record(json.loads(task_path(manifest, task_id, split).read_text()))
    entries = json.loads(manifest.read_text())["tasks"]
    source_hash = next(
        e["source_record_sha256"] for e in entries if e["task_id"] == task_id
    )
    if versioned:
        assert task.origin is not None
        registered_origin(task.origin)
        raw = (manifest.parent / "private/source.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != task.origin.source_sha256:
            raise ValueError("MBPP original source content identity changed")
        row = next(
            (
                row
                for row in json.loads(raw)
                if str(row["task_id"]) == task.origin.original_task_id
            ),
            None,
        )
        if row is None or fingerprint(row) != source_hash:
            raise ValueError("MBPP original source record identity changed")
        expected, _ = _convert(row, task.environment_id, task.origin)
        if task != expected:
            raise ValueError("Prepared MBPP task differs from the registered source")
    if not isinstance(task.grader, FunctionGrader) or not task.feedback_command:
        raise ValueError("MBPP requires function grading and public feedback")
    if len(task.implementation_paths) != 1:
        raise ValueError("MBPP reference must have one implementation path")
    # A feedback-example memorizer should pass feedback and fail independent
    # inputs. Its answer and arguments come only from the public assertion.
    assertions = [
        node
        for content in task.initial_files.values()
        for node in ast.parse(content).body
        if isinstance(node, ast.Assert)
    ]
    if len(assertions) != 1:
        raise ValueError("MBPP conversion requires exactly one feedback assertion")
    test = assertions[0].test
    if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Call):
        raise ValueError("Unsupported MBPP feedback assertion")
    args = tuple(ast.literal_eval(arg) for arg in test.left.args)
    expected = ast.literal_eval(test.comparators[0])
    wrong = (
        f"def {task.grader.function_name}(*args):\n"
        f"    return {expected!r} if args == {args!r} else None\n"
    )
    return _Definition(
        _identities(replace(task, task_type="short_program")),
        {task.implementation_paths[0]: wrong},
        "Memorizes the feedback example and returns None for other inputs.",
        source_hash,
    )


def _definitions(
    cohort: dict[str, Any],
    mbpp_manifest: Path | None,
    config: ExperimentConfig,
    cohort_sha256: str,
) -> tuple[list[_Definition], list[dict[str, Any]]]:
    versioned = cohort["version"] == "harness-task-cohort-v2"
    if cohort["version"] not in ("harness-task-cohort-v1", "harness-task-cohort-v2"):
        raise ValueError("Unsupported Harness cohort version")
    requested: dict[str, Literal["train", "dev", "test"]] = {}
    groups: tuple[tuple[str, Literal["train", "dev", "test"]], ...] = (
        (
            ("training_candidates", "train"),
            ("development", "dev"),
            ("test", "test"),
        )
        if versioned
        else (
            ("training_candidates", "train"),
            ("harness_development", "dev"),
            ("coding_development", "dev"),
        )
    )
    for name, split in groups:
        ids = cohort[name]["task_ids"]
        if (not ids and not versioned) or len(ids) != len(set(ids)):
            raise ValueError("Cohort requires nonempty unique task IDs")
        for task_id in ids:
            if task_id in requested and requested[task_id] != split:
                raise ValueError(f"Task family split conflict: {task_id}")
            requested[task_id] = split
    lock, families = authored_source()
    catalogs = (
        [
            (source_lock, source_families)
            for source_lock, source_families in admitted_sources().values()
            if source_families is not None
        ]
        if versioned
        else [(lock, families)]
    )
    authored = {
        v["task"]["task_id"]: (source_lock, family, v)
        for source_lock, source_families in catalogs
        for family in source_families.values()
        for v in family["variants"]
    }
    for task_id in cohort["environment_variants"]:
        if task_id in requested or task_id not in authored:
            raise ValueError("Environment variants need a registered family")
        role: Literal["train", "dev", "test"] = "dev"
        if versioned:
            family_id = authored[task_id][1]["family_id"]
            parent_roles = {
                split
                for identity, split in requested.items()
                if identity in authored
                and authored[identity][1]["family_id"] == family_id
            }
            if len(parent_roles) != 1:
                raise ValueError("Variant must inherit one frozen base family role")
            role = parent_roles.pop()
        requested[task_id] = role
    definitions = []
    for task_id, split in requested.items():
        if task_id.startswith("mbpp/"):
            if mbpp_manifest is None:
                raise ValueError("MBPP tasks require --mbpp-manifest")
            source_split: str = split
            if versioned:
                entries = json.loads(mbpp_manifest.read_text())["tasks"]
                source_split = next(
                    e["split"] for e in entries if e["task_id"] == task_id
                )
            definition = _mbpp_definition(
                mbpp_manifest,
                task_id,
                source_split,
                versioned=versioned,
            )
        elif task_id in authored:
            source_lock, family, variant = authored[task_id]
            if not versioned and family["split"] != split:
                raise ValueError(
                    f"Authored family split mismatch: {family['family_id']}"
                )
            data = catalog_task_record(
                source_lock, family, variant, config.environment.id
            )
            if versioned:
                data["assignment"] = asdict(
                    TaskAssignment(
                        cohort["name"],
                        cohort_sha256,
                        canonical_family(TaskOrigin(**data["origin"])),
                        "train_candidate" if split == "train" else split,
                    )
                )
            definition = _Definition(
                _identities(task_from_record(data)),
                variant["wrong_files"],
                variant["wrong_reason"],
                fingerprint(variant),
            )
        else:
            raise ValueError(f"Unregistered task: {task_id}")
        task = definition.task
        assert task.origin is not None
        if versioned and task.assignment is None:
            task = replace(
                task,
                assignment=TaskAssignment(
                    cohort["name"],
                    cohort_sha256,
                    canonical_family(task.origin),
                    "train_candidate" if split == "train" else split,
                ),
            )
            definition = replace(definition, task=task)
        validate_task(task, harness=True)
        if task.environment_id != config.environment.id:
            raise ValueError("Prepared task environment differs from validation config")
        if not task.reference_implementation or not task.feedback_command:
            raise ValueError("Every task needs a reference and a feedback command")
        if len(task.implementation_paths) != 1:
            raise ValueError(
                "This source conversion requires one reference entry point"
            )
        if set(definition.wrong_files) != set(task.implementation_paths):
            raise ValueError("Mutation must cover the declared implementation")
        definitions.append(definition)
    _validate_cohorts(cohort, [d.task for d in definitions])
    sources = [source_lock for source_lock, _ in catalogs]
    if mbpp_manifest is not None:
        sources.insert(0, json.loads(mbpp_manifest.read_text())["source"])
    if versioned:
        needed = {
            (d.task.origin.dataset, d.task.origin.revision, d.task.origin.source_sha256)
            for d in definitions
            if d.task.origin is not None
        }
        sources = [
            source
            for source in sources
            if (source["dataset"], source["revision"], source["sha256"]) in needed
        ]
    return definitions, sources


def _validate_cohorts(cohort: dict[str, Any], tasks: list[Task]) -> None:
    versioned = cohort["version"] == "harness-task-cohort-v2"
    by_id = {t.task_id: t for t in tasks}
    splits: dict[str, str] = {}
    for task in tasks:
        assert task.origin is not None
        family, split = task_family(task), str(task_split(task))
        if family in splits and splits[family] != split:
            raise ValueError(f"Task family split conflict: {family}")
        splits[family] = split
    groups = (
        ("training_candidates", "development", "test")
        if versioned
        else (
            "harness_development",
            "training_candidates",
            "coding_development",
        )
    )
    for name in groups:
        selected = [by_id[task_id] for task_id in cohort[name]["task_ids"]]
        family_ids = [task_family(t) for t in selected]
        if len(family_ids) != len(set(family_ids)) or any(
            t.variant_id != "base" for t in selected
        ):
            raise ValueError("Checklists require independent base task families")
        if name == "coding_development":
            if any(
                t.origin is None or t.origin.dataset != "google-research/mbpp"
                for t in selected
            ):
                raise ValueError("Coding checklist must retain MBPP development tasks")
        else:
            target = cohort[name]["per_type"]
            counts = (
                {kind: target for kind in _TYPES} if type(target) is int else target
            )
            if (
                not isinstance(counts, dict)
                or set(counts) != set(_TYPES)
                or any(
                    type(value) is not int or value < (0 if versioned else 1)
                    for value in counts.values()
                )
                or any(
                    sum(t.task_type == kind for t in selected) != counts[kind]
                    for kind in _TYPES
                )
            ):
                raise ValueError(f"Incorrect task-type counts in {name}")
    base_families = {
        _family(by_id[i])
        for name in (groups if versioned else ("harness_development",))
        for i in cohort[name]["task_ids"]
    }
    for task_id in cohort["environment_variants"]:
        task = by_id[task_id]
        if (
            task.variant_id == "base"
            or task.origin is None
            or task_family(task) not in base_families
        ):
            raise ValueError("Environment variant must retain a frozen base family")


def _replay(task: Task, files: dict[str, str], config: ExperimentConfig) -> ReplayModel:
    script = (
        "from pathlib import Path; "
        f"files = {files!r}; "
        "[(Path(p).parent.mkdir(parents=True, exist_ok=True), "
        "Path(p).write_text(c, encoding='utf-8')) for p, c in files.items()]"
    )
    commands = [
        "python3 -c " + shlex.quote(script),
        str(task.feedback_command),
        "printf '%s\\n' " + shlex.quote(config.protocol.submission_marker),
    ]
    return ReplayModel(
        [f"```bash\n{command}\n```" for command in commands],
        encoding=Utf8Encoding(),
        identity="replay/reference-and-mutation-validation",
    )


def prepare_harness(
    *,
    mbpp_manifest: Path | None,
    output_dir: Path,
    config: ExperimentConfig,
    cohort_path: Path = DEFAULT_COHORT,
) -> Path:
    """Publish a frozen candidate pool and separate development checklists.

    Reference and mutation failures abort the build rather than change its fixed
    denominator. Preparation reads only selected MBPP train/dev Task documents;
    the held-out materials in an existing source manifest remain untouched.
    """
    cohort_bytes = cohort_path.read_bytes()
    cohort = json.loads(cohort_bytes)
    versioned = cohort["version"] == "harness-task-cohort-v2"
    definitions, sources = _definitions(
        cohort,
        mbpp_manifest,
        config,
        hashlib.sha256(cohort_bytes).hexdigest(),
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    lock, _ = authored_source()
    (private / "authored-source.json").write_bytes(
        (AUTHORED_LOCK.parent / lock["file"]).read_bytes()
    )
    (private / "cohort.json").write_bytes(cohort_bytes)
    (private / "family-guard.json").write_bytes(FAMILY_GUARD.read_bytes())
    by_id = {d.task.task_id: d.task for d in definitions}
    harness_ids = cohort["development" if versioned else "harness_development"][
        "task_ids"
    ]
    coding_ids = [] if versioned else cohort["coding_development"]["task_ids"]
    overlap = sorted(
        {_family(by_id[i]) for i in harness_ids}
        & {_family(by_id[i]) for i in coding_ids}
    )
    manifest: dict[str, Any] = {
        "schema_version": 2 if versioned else 1,
        "status": "preparing",
        "conversion": VERSIONED_CONVERSION if versioned else CONVERSION,
        "source": {
            "dataset": "bashlet/harness-cohort",
            "revision": cohort["name"],
            "sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        },
        "sources": sources,
        "cohort": cohort,
        "family_guard_sha256": hashlib.sha256(FAMILY_GUARD.read_bytes()).hexdigest(),
        "upstream_manifest_sha256": (
            hashlib.sha256(mbpp_manifest.read_bytes()).hexdigest()
            if mbpp_manifest
            else None
        ),
        "evidence_kind": "reference-and-mutation-replay",
        "model_baseline": False,
        "validation_config": config_record(config),
        "checklists": {
            "harness_development": harness_ids,
            "coding_development": coding_ids,
            "overlap_family_ids": overlap,
            "combined_independent_samples": None,
            "environment_variants": cohort["environment_variants"],
        },
        "tasks": [],
    }
    if versioned:
        registry_bytes = SOURCE_REGISTRY.read_bytes()
        (private / "source-registry.json").write_bytes(registry_bytes)
        manifest["source_registry_sha256"] = hashlib.sha256(registry_bytes).hexdigest()
        artifacts = []
        source_dir = private / "sources"
        source_dir.mkdir(mode=0o700)
        for source in sources:
            if source["dataset"] == "google-research/mbpp":
                assert mbpp_manifest is not None
                content = (mbpp_manifest.parent / "private/source.json").read_bytes()
            else:
                content = (_CONFIGS / source["file"]).read_bytes()
            if hashlib.sha256(content).hexdigest() != source["sha256"]:
                raise ValueError("Source changed before its frozen copy was written")
            path = source_dir / (source["sha256"] + ".json")
            path.write_bytes(content)
            artifacts.append(
                {key: source[key] for key in ("dataset", "revision", "sha256")}
                | {"file": str(path.relative_to(output_dir))}
            )
        manifest["source_artifacts"] = artifacts
        manifest["checklists"] = {
            "training_candidates": cohort["training_candidates"]["task_ids"],
            "development": harness_ids,
            "test": cohort["test"]["task_ids"],
            "environment_variants": cohort["environment_variants"],
        }
    destination = output_dir / "manifest.json"
    _write(destination, manifest)
    try:
        for definition in definitions:
            task = definition.task
            assert task.origin is not None
            entry: dict[str, Any] = {
                "task_id": task.task_id,
                "origin": asdict(task.origin),
                "family_id": task_family(task),
                "split": task_split(task),
                "task_type": task.task_type,
                "variant_id": task.variant_id,
                "feedback_start_id": task.feedback_start_id,
                "environment_variant_id": task.environment_variant_id,
                "source_record_sha256": definition.source_record_sha256,
                "wrong_reason": definition.wrong_reason,
                "status": "validating",
            }
            if task.assignment is not None:
                entry["assignment"] = asdict(task.assignment)
            manifest["tasks"].append(entry)
            for name, files in (
                (
                    "reference",
                    {task.implementation_paths[0]: task.reference_implementation},
                ),
                ("wrong", definition.wrong_files),
            ):
                result = run_experiment(
                    task=task,
                    model=_replay(task, files, config),
                    config=config,
                    output_dir=private / "validation",
                )
                entry[name + "_run"] = str(result.run_dir.relative_to(output_dir))
                entry[name + "_run_sha256"] = hashlib.sha256(
                    (result.run_dir / "run.json").read_bytes()
                ).hexdigest()
                _write(destination, manifest)
                if result.status != "completed" or result.grade is None:
                    raise ValueError(
                        f"{task.task_id}: {name} execution incomplete: "
                        f"{result.status}; {result.reason}"
                    )
                submission = result.submission
                if name == "wrong" and (
                    len(result.trajectory) != 3
                    or result.trajectory[0].observation is None
                    or result.trajectory[0].observation.status != "ok"
                    or submission is None
                    or any(
                        path not in submission.manifest
                        or submission.manifest[path].sha256
                        != hashlib.sha256(content.encode()).hexdigest()
                        for path, content in files.items()
                    )
                ):
                    raise ValueError(f"{task.task_id}: mutation was not executed")
                if name == "reference" and (
                    not result.passed
                    or len(result.trajectory) != 3
                    or any(
                        s.observation is None or s.observation.status != "ok"
                        for s in result.trajectory
                    )
                ):
                    raise ValueError(
                        f"{task.task_id}: reference feedback or grading failed"
                    )
                if name == "wrong" and result.passed:
                    raise ValueError(
                        f"{task.task_id}: targeted wrong implementation passed grading"
                    )
            task_file = (
                private
                / "tasks"
                / str(task_split(task))
                / (fingerprint(task.task_id) + ".json")
            )
            task_file.parent.mkdir(parents=True, exist_ok=True)
            task_file.write_text(serialize_task(task))
            entry.update(
                status="verified",
                task_file=str(task_file.relative_to(output_dir)),
                task_sha256=hashlib.sha256(task_file.read_bytes()).hexdigest(),
            )
            _write(destination, manifest)
        manifest["counts"] = {
            split: {
                kind: sum(
                    t.task_type == kind
                    for t in (by_id[i] for i in cohort[name]["task_ids"])
                )
                for kind in _TYPES
            }
            for split, name in (
                (
                    ("train_candidate", "training_candidates"),
                    ("dev", "development"),
                    ("test", "test"),
                )
                if versioned
                else (
                    ("train", "training_candidates"),
                    ("dev", "harness_development"),
                )
            )
        }
        plans = (
            (("harness-dev", harness_ids),)
            if versioned
            else (
                ("harness-dev", harness_ids),
                ("coding-dev", coding_ids),
            )
        )
        for name, ids in plans:
            _write(
                output_dir / (name + ".json"),
                {
                    "version": "development-evaluation-v1",
                    "name": cohort["name"] + "/" + name,
                    "manifest": "manifest.json",
                    "split": "dev",
                    "task_ids": ids,
                },
            )
        manifest["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        manifest["status"], manifest["reason"] = "incomplete", str(error)
        raise
    finally:
        _write(destination, manifest)
    return destination
