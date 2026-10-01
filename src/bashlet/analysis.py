"""Derive candidates, verifications and repairs from recorded episode evidence."""

import hashlib
import json
from dataclasses import replace

from .models import (
    Bash,
    Candidate,
    ExperimentResult,
    RepairAttempt,
    Step,
    Task,
    WorkspaceEntry,
)


def _track_candidate(
    result: ExperimentResult, task: Task, step: Step, index: int, current: int | None
) -> int | None:
    files: dict[str, str] = {}
    size_bytes = 0
    for path in sorted(set(task.implementation_paths)):
        entry = step.workspace_manifest.get(path)
        if (
            entry is None
            or entry.kind != "file"
            or entry.sha256 is None
            or entry.size_bytes is None
        ):
            return current
        files[path] = entry.sha256
        size_bytes += entry.size_bytes
    if size_bytes > result.limits.submission_bytes:
        return current
    payload = json.dumps(files, sort_keys=True, ensure_ascii=False, indent=2).encode()
    digest = (
        next(iter(files.values()))
        if len(files) == 1
        else hashlib.sha256(payload).hexdigest()
    )
    for candidate in result.candidates:
        if candidate.sha256 == digest:
            return candidate.index
    candidate_index = len(result.candidates)
    if len(files) == 1:
        snapshot_path = f"blobs/{digest}"
    else:
        directory = result.run_dir / "candidates"
        directory.mkdir(exist_ok=True)
        target = directory / f"{candidate_index:03d}.json"
        target.write_bytes(payload)
        target.chmod(0o400)
        snapshot_path = str(target.relative_to(result.run_dir))
    result.candidates.append(
        Candidate(candidate_index, digest, snapshot_path, index, files)
    )
    return candidate_index


def _track_repair(
    attempts: list[RepairAttempt], candidate: int | None, step: int, passed: bool
) -> None:
    """Keep v1 semantics: repeated verification of one candidate is one attempt."""
    open_attempt = attempts[-1] if attempts and attempts[-1].passed is None else None
    if open_attempt is not None and candidate != open_attempt.from_candidate:
        open_attempt.to_candidate = candidate
        open_attempt.verified_at_step = step
        open_attempt.passed = passed
        open_attempt = None
    if not passed and open_attempt is None:
        attempts.append(RepairAttempt(len(attempts), step, candidate))


def analyze_episode(
    result: ExperimentResult,
    task: Task,
    feedback_materials: dict[str, WorkspaceEntry],
) -> None:
    """Analyze recorded manifests and observations, without reading live files."""
    if not task.implementation_paths:
        return
    current: int | None = None
    for index, step in enumerate(result.trajectory):
        verification = False
        observation = step.observation
        if (
            isinstance(step.action, Bash)
            and observation is not None
            and observation.status != "infrastructure_error"
        ):
            current = _track_candidate(result, task, step, index, current)
            verification = (
                task.feedback_command is not None
                and step.action.command.strip() == task.feedback_command.strip()
                and bool(feedback_materials)
                and all(
                    step.workspace_manifest.get(path) == entry
                    for path, entry in feedback_materials.items()
                )
            )
            if verification:
                _track_repair(
                    result.repair_attempts, current, index, observation.status == "ok"
                )
        result.trajectory[index] = replace(
            step, candidate=current, verification=verification
        )
