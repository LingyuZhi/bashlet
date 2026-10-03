"""Pinned MBPP preparation. All source, reference and grading data is operator-only."""

import ast
import hashlib
import json
import math
import re
import shlex
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import httpx

from .config import ExperimentConfig, config_record
from .models import FunctionGrader, GradingCase, JSONValue, Task, TaskOrigin
from .protocol import fingerprint
from .replay import ReplayModel, Utf8Encoding

CONVERSION = "mbpp-json-functions-v1"


def _write_json(path: Path, value: object) -> None:
    text = json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def _download(url: str) -> bytes:
    # An explicit HTTPTransport disables HTTPX's environment proxy discovery.
    # Keep the default client and retry bounded transient failures here instead.
    with httpx.Client(follow_redirects=True, timeout=30) as client:
        for attempt in range(3):
            try:
                response = client.get(url)
                response.raise_for_status()
                return response.content
            except httpx.HTTPError as error:
                transient = isinstance(error, httpx.TransportError) or (
                    isinstance(error, httpx.HTTPStatusError)
                    and error.response.status_code in {408, 429, 500, 502, 503, 504}
                )
                if not transient or attempt == 2:
                    raise ValueError(
                        "MBPP download failed; retry or use a cached --source-file. "
                        f"The pinned SHA-256 is required: {error}"
                    ) from error
                time.sleep(0.5 * 2**attempt)
    raise AssertionError("Download retry loop ended without a result")


def _original_id(value: object) -> int:
    match = re.fullmatch(r"(?:[Mm][Bb][Pp][Pp]/)?([1-9][0-9]*)", str(value))
    if match is None:
        raise ValueError(f"Unknown original MBPP task ID: {value!r}")
    return int(match[1])


def _split(task_id: int) -> Literal["train", "dev", "test", "prompt"]:
    if 1 <= task_id <= 10:
        return "prompt"
    if 11 <= task_id <= 510:
        return "test"
    if 511 <= task_id <= 600:
        return "dev"
    if 601 <= task_id <= 974:
        return "train"
    raise ValueError(f"Unknown original MBPP task ID: {task_id}")


def validate_origin(origin: TaskOrigin) -> None:
    """Validate source-owned lineage even when a derived Task bypasses a manifest."""
    if origin.dataset != "google-research/mbpp":
        raise ValueError(f"Unsupported task-origin dataset: {origin.dataset}")
    original_id = _original_id(origin.original_task_id)
    split = _split(original_id)
    if (
        split == "prompt"
        or origin.family_id != f"mbpp/{original_id}"
        or origin.split != split
    ):
        raise ValueError(f"MBPP family split mismatch: mbpp/{original_id}")


def _json_literal(value: object, *, output: bool = False) -> JSONValue:
    """Inputs must retain Python types; output tuples follow json-function-v1."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, list) or (output and isinstance(value, tuple)):
        return [_json_literal(item, output=output) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _json_literal(item, output=output) for key, item in value.items()}
    raise ValueError(f"unsupported_json_type: {type(value).__name__}")


def _case(source: str) -> tuple[str, GradingCase]:
    statements = ast.parse(source).body
    if len(statements) != 1 or not isinstance(statements[0], ast.Assert):
        raise ValueError("unsupported_assertion: expected one assert")
    comparison = statements[0].test
    if not (
        isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.Eq)
        and isinstance(comparison.left, ast.Call)
        and isinstance(comparison.left.func, ast.Name)
        and not comparison.left.keywords
    ):
        raise ValueError("unsupported_assertion: expected a literal equality call")
    call = comparison.left
    assert isinstance(call.func, ast.Name)
    args = tuple(_json_literal(ast.literal_eval(arg)) for arg in call.args)
    expected = _json_literal(ast.literal_eval(comparison.comparators[0]), output=True)
    return call.func.id, GradingCase(args, expected)


def _convert(
    row: dict[str, Any],
    environment_id: str,
    origin: TaskOrigin,
) -> tuple[Task, str]:
    tests = row["test_list"]
    if not isinstance(tests, list) or len(tests) < 2:
        raise ValueError("insufficient_tests")
    parsed = [_case(test) for test in tests]
    name = parsed[0][0]
    if any(function != name for function, _ in parsed):
        raise ValueError("multiple_tested_functions")
    feedback = parsed[0][1]
    hidden: dict[str, GradingCase] = {}
    for _, case in parsed[1:]:
        if case.args == feedback.args:
            if case.expected != feedback.expected:
                raise ValueError("conflicting_expected_outputs")
            continue
        key = json.dumps(case.args, sort_keys=True)
        if key in hidden and hidden[key].expected != case.expected:
            raise ValueError("conflicting_expected_outputs")
        hidden[key] = case
    if not hidden:
        raise ValueError("no_independent_grading_inputs")
    reference = row["code"]
    functions = [
        node
        for node in ast.parse(reference).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(functions) != 1:
        raise ValueError("missing_function_signature")
    signature = ast.unparse(functions[0].args)
    prompt = (
        f"{row['prompt']}\n\nImplement `{name}({signature})` in `solution.py`.\n"
        "Run `python3 feedback_tests.py` to check the public example.\n"
        "Your submission is graded on separate inputs."
    )
    imports = "\n".join(row["test_imports"])
    header = f"from solution import {name}\n{imports}\n"
    return Task(
        task_id=origin.family_id,
        prompt=prompt,
        initial_files={"feedback_tests.py": header + tests[0] + "\n"},
        grader=FunctionGrader("solution.py", name, tuple(hidden.values())),
        environment_id=environment_id,
        reference_implementation=reference,
        implementation_paths=("solution.py",),
        feedback_command="python3 feedback_tests.py",
        origin=origin,
    ), header + "\n".join(tests) + "\n"


def prepare_mbpp(
    *,
    source_lock: Path,
    source_file: Path | None,
    output_dir: Path,
    config: ExperimentConfig,
) -> Path:
    """Build a new private task set, publishing completeness only after validation.

    Reference runs use fixed replay commands, never a sampled policy. The original
    assertions and the actual function grader must both pass in real sandboxes.
    Infrastructure failures abort preparation rather than filter the dataset.
    """
    # Preparation orchestrates experiments; experiment-side origin validation
    # imports only the data rules above, not this orchestration dependency.
    from .experiment import run_experiment

    lock = json.loads(source_lock.read_text())
    if lock["variant"] != "sanitized" or lock["conversion"] != CONVERSION:
        raise ValueError("Only sanitized MBPP with mbpp-json-functions-v1 is supported")
    if (
        lock["dataset"] != "google-research/mbpp"
        or re.fullmatch(r"[0-9a-f]{40}", lock["revision"]) is None
        or re.fullmatch(r"[0-9a-f]{64}", lock["sha256"]) is None
    ):
        raise ValueError("MBPP source lock requires a dataset, commit and SHA-256")
    source = _download(lock["url"]) if source_file is None else source_file.read_bytes()
    if hashlib.sha256(source).hexdigest() != lock["sha256"]:
        raise ValueError("MBPP source SHA-256 mismatch")
    rows = json.loads(source)
    if not isinstance(rows, list):
        raise ValueError("Sanitized MBPP source must be a JSON list")
    identities = [_original_id(row["task_id"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate MBPP family in source")
    for task_id in identities:
        _split(task_id)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, mode=0o700)
    private = output_dir / "private"
    private.mkdir(mode=0o700)
    (private / "source.json").write_bytes(source)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "source": lock,
        "conversion": CONVERSION,
        "rules": {
            "split": "IDs: prompt 1-10; test 11-510; dev 511-600; train 601-974",
            "feedback": "first original assertion",
            "grading": "remaining distinct inputs; json-function-v1; literal outputs",
            "filter": "supported assertions and JSON inputs; verified reference only",
        },
        "validation_config": config_record(config),
        "tasks": [],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    for row in sorted(rows, key=lambda row: _original_id(row["task_id"])):
        task_id = _original_id(row["task_id"])
        split = _split(task_id)
        entry: dict[str, Any] = {
            "original_task_id": task_id,
            "task_id": f"mbpp/{task_id}",
            "family_id": f"mbpp/{task_id}",
            "split": split,
            "source_record_sha256": fingerprint(row),
            "status": "excluded",
        }
        manifest["tasks"].append(entry)
        if split == "prompt":
            entry["reason"] = "reserved_prompt_example"
        else:
            origin = TaskOrigin(
                dataset=lock["dataset"],
                revision=lock["revision"],
                source_sha256=lock["sha256"],
                original_task_id=str(task_id),
                family_id=f"mbpp/{task_id}",
                split=split,
            )
            try:
                task, reference_tests = _convert(row, config.environment.id, origin)
            except (ValueError, SyntaxError, TypeError) as error:
                entry["reason"] = str(error)
            else:
                verification = run_experiment(
                    task=replace(
                        task,
                        initial_files=task.initial_files
                        | {
                            "solution.py": task.reference_implementation,
                            "_reference_tests.py": reference_tests,
                        },
                    ),
                    model=ReplayModel(
                        [
                            "```bash\npython3 _reference_tests.py\n```",
                            "```bash\nprintf '%s\\n' "
                            + shlex.quote(config.protocol.submission_marker)
                            + "\n```",
                        ],
                        encoding=Utf8Encoding(),
                    ),
                    output_dir=private / "reference-validation",
                    config=config,
                )
                entry["verification_run"] = str(
                    verification.run_dir.relative_to(output_dir)
                )
                if verification.status != "completed":
                    entry["reason"] = verification.reason or verification.status
                    _write_json(manifest_path, manifest)
                    raise ValueError(
                        f"Reference validation did not complete: {entry['reason']}"
                    )
                observed = verification.trajectory[0].observation
                if (
                    not verification.passed
                    or observed is None
                    or observed.status != "ok"
                ):
                    entry["reason"] = "reference_failed"
                else:
                    path = private / "tasks" / split / f"{task_id}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _write_json(path, asdict(task))
                    entry.update(
                        status="verified",
                        task_file=str(path.relative_to(output_dir)),
                        task_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        grading_cases=len(task.grader.cases)
                        if isinstance(task.grader, FunctionGrader)
                        else 0,
                    )
        _write_json(manifest_path, manifest)
    manifest["status"] = "complete"
    manifest["counts"] = {
        split: sum(
            entry["split"] == split and entry["status"] == "verified"
            for entry in manifest["tasks"]
        )
        for split in ("train", "dev", "test")
    }
    _write_json(manifest_path, manifest)
    return manifest_path


def mbpp_task_path(manifest_path: Path, task_id: str, split: str) -> Path:
    """Select from a completed task set, enforcing original families and splits.

    Derived tasks must retain their parent's origin. Published source labels such
    as MBPP+'s `test` do not change the original MBPP family split.
    """
    # The source owns family rules; the task package owns shared file admission.
    from .tasks import _verified_task_path

    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "complete" or manifest["conversion"] != CONVERSION:
        raise ValueError("MBPP task set is incomplete or has an unsupported conversion")
    families: dict[str, str] = {}
    tasks: dict[str, dict[str, Any]] = {}
    for entry in manifest["tasks"]:
        original_id = _original_id(entry["original_task_id"])
        family = f"mbpp/{original_id}"
        if entry["family_id"] != family or entry["split"] != _split(original_id):
            raise ValueError(f"MBPP family split mismatch: {family}")
        if family in families and families[family] != entry["split"]:
            raise ValueError(f"MBPP family split conflict: {family}")
        families[family] = entry["split"]
        if entry["task_id"] in tasks:
            raise ValueError(f"Duplicate task ID: {entry['task_id']}")
        tasks[entry["task_id"]] = entry
    selected = tasks[task_id]
    expected = TaskOrigin(
        dataset=manifest["source"]["dataset"],
        revision=manifest["source"]["revision"],
        source_sha256=manifest["source"]["sha256"],
        original_task_id=str(selected["original_task_id"]),
        family_id=selected["family_id"],
        split=selected["split"],
    )
    return _verified_task_path(manifest_path, selected, expected, task_id, split)
