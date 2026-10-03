"""Freeze mixed-source cohorts with actual reference and mutation execution."""

import ast
import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, config_record
from ..experiment import run_experiment, serialize_task, task_from_record
from ..models import FunctionGrader, Task, TaskOrigin
from ..protocol import fingerprint
from ..replay import ReplayModel, Utf8Encoding
from . import (
    _CONFIGS,
    AUTHORED_LOCK,
    CONVERSION,
    FAMILY_GUARD,
    authored_source,
    task_path,
    validate_harness_origin,
)

DEFAULT_COHORT = _CONFIGS / "harness-cohort-v1.json"
_TYPES = ("short_program", "code_edit", "workspace")

_WORKSPACE_GRADING_DRIVER = """import os, shutil, tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as directory:
    shutil.copytree('/workspace', directory + '/task')
    os.chdir(directory + '/task')
    for path in Path('.').rglob('*'):
        if path.is_file(): path.chmod(0o600)
    source = Path('_grading/cases.py').read_text()
    exec(compile(source, '_grading/cases.py', 'exec'))
"""


def _family(task: Task) -> str:
    assert task.origin is not None
    return task.origin.family_id


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


def _mbpp_definition(manifest: Path, task_id: str, split: str) -> _Definition:
    task = task_from_record(json.loads(task_path(manifest, task_id, split).read_text()))
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
    entries = json.loads(manifest.read_text())["tasks"]
    source_hash = next(
        e["source_record_sha256"] for e in entries if e["task_id"] == task_id
    )
    return _Definition(
        _identities(replace(task, task_type="short_program")),
        {task.implementation_paths[0]: wrong},
        "Memorizes the feedback example and returns None for other inputs.",
        source_hash,
    )


def _definitions(
    cohort: dict[str, Any], mbpp_manifest: Path, config: ExperimentConfig
) -> tuple[list[_Definition], list[dict[str, Any]]]:
    if cohort["version"] != "harness-task-cohort-v1":
        raise ValueError("Unsupported Harness cohort version")
    requested: dict[str, str] = {}
    for name, split in (
        ("training_candidates", "train"),
        ("harness_development", "dev"),
        ("coding_development", "dev"),
    ):
        ids = cohort[name]["task_ids"]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Cohort requires nonempty unique task IDs")
        for task_id in ids:
            if task_id in requested and requested[task_id] != split:
                raise ValueError(f"Task family split conflict: {task_id}")
            requested[task_id] = split
    for task_id in cohort["environment_variants"]:
        if task_id in requested:
            raise ValueError("Environment variants cannot enlarge a checklist")
        requested[task_id] = "dev"
    lock, families = authored_source()
    authored = {
        v["task"]["task_id"]: (family, v)
        for family in families.values()
        for v in family["variants"]
    }
    definitions = []
    for task_id, split in requested.items():
        if task_id.startswith("mbpp/"):
            definition = _mbpp_definition(mbpp_manifest, task_id, split)
        elif task_id in authored:
            family, variant = authored[task_id]
            if family["split"] != split:
                raise ValueError(
                    f"Authored family split mismatch: {family['family_id']}"
                )
            data = variant["task"] | {
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
                "environment_id": config.environment.id,
            }
            if family["task_type"] == "workspace":
                data["grader"] = data["grader"] | {
                    "hidden_files": (
                        data["grader"]["hidden_files"]
                        | {"_grading/check.py": _WORKSPACE_GRADING_DRIVER}
                    )
                }
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
        validate_harness_origin(task.origin)
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
    mbpp = json.loads(mbpp_manifest.read_text())
    return definitions, [mbpp["source"], lock]


def _validate_cohorts(cohort: dict[str, Any], tasks: list[Task]) -> None:
    by_id = {t.task_id: t for t in tasks}
    splits: dict[str, str] = {}
    for task in tasks:
        assert task.origin is not None
        origin = task.origin
        if origin.family_id in splits and splits[origin.family_id] != origin.split:
            raise ValueError(f"Task family split conflict: {origin.family_id}")
        splits[origin.family_id] = origin.split
    for name in ("harness_development", "training_candidates", "coding_development"):
        selected = [by_id[task_id] for task_id in cohort[name]["task_ids"]]
        family_ids = [t.origin.family_id for t in selected if t.origin is not None]
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
            if (
                type(target) is not int
                or target < 1
                or any(
                    sum(t.task_type == kind for t in selected) != target
                    for kind in _TYPES
                )
            ):
                raise ValueError(f"Incorrect task-type counts in {name}")
    development = {_family(by_id[i]) for i in cohort["harness_development"]["task_ids"]}
    for task_id in cohort["environment_variants"]:
        task = by_id[task_id]
        if (
            task.variant_id == "base"
            or task.origin is None
            or task.origin.family_id not in development
        ):
            raise ValueError(
                "Environment variant must retain a Harness development family"
            )


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
    mbpp_manifest: Path,
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
    definitions, sources = _definitions(cohort, mbpp_manifest, config)
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
    harness_ids = cohort["harness_development"]["task_ids"]
    coding_ids = cohort["coding_development"]["task_ids"]
    overlap = sorted(
        {_family(by_id[i]) for i in harness_ids}
        & {_family(by_id[i]) for i in coding_ids}
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "conversion": CONVERSION,
        "source": {
            "dataset": "bashlet/harness-cohort",
            "revision": cohort["name"],
            "sha256": hashlib.sha256(cohort_bytes).hexdigest(),
        },
        "sources": sources,
        "cohort": cohort,
        "family_guard_sha256": hashlib.sha256(FAMILY_GUARD.read_bytes()).hexdigest(),
        "upstream_manifest_sha256": hashlib.sha256(
            mbpp_manifest.read_bytes()
        ).hexdigest(),
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
    destination = output_dir / "manifest.json"
    _write(destination, manifest)
    try:
        for definition in definitions:
            task = definition.task
            assert task.origin is not None
            entry: dict[str, Any] = {
                "task_id": task.task_id,
                "origin": asdict(task.origin),
                "family_id": task.origin.family_id,
                "split": task.origin.split,
                "task_type": task.task_type,
                "variant_id": task.variant_id,
                "feedback_start_id": task.feedback_start_id,
                "environment_variant_id": task.environment_variant_id,
                "source_record_sha256": definition.source_record_sha256,
                "wrong_reason": definition.wrong_reason,
                "status": "validating",
            }
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
                / task.origin.split
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
                ("train", "training_candidates"),
                ("dev", "harness_development"),
            )
        }
        for name, ids in (("harness-dev", harness_ids), ("coding-dev", coding_ids)):
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
