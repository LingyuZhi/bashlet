"""Record Task Workspace states with shared, content-addressed file blobs."""

import hashlib
import json
import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path

from bashlet.harness.models import Limits, Submission, WorkspaceEntry, WorkspaceKind

_KINDS: dict[int, WorkspaceKind] = {
    stat.S_IFREG: "file",
    stat.S_IFDIR: "directory",
    stat.S_IFLNK: "symlink",
    stat.S_IFIFO: "fifo",
    stat.S_IFSOCK: "socket",
    stat.S_IFBLK: "block_device",
    stat.S_IFCHR: "character_device",
}


def capture_workspace(workspace: Path, run_dir: Path) -> dict[str, WorkspaceEntry]:
    blobs = run_dir / "blobs"
    blobs.mkdir(exist_ok=True)
    manifest: dict[str, WorkspaceEntry] = {}

    def visit(directory: int, prefix: str = "") -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
        except OSError as error:
            manifest[prefix.rstrip("/") or "."] = WorkspaceEntry(
                "directory", error=str(error)
            )
            return
        for entry in entries:
            relative = prefix + entry.name
            manifest[relative] = WorkspaceEntry("unknown")
            subdirectory: int | None = None
            content: bytes | None = None
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
                kind = _KINDS.get(stat.S_IFMT(mode), "unknown")
                manifest[relative] = WorkspaceEntry(kind, mode=stat.S_IMODE(mode))
                if kind == "directory":
                    subdirectory = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory,
                    )
                elif kind == "symlink":
                    manifest[relative] = WorkspaceEntry(
                        kind,
                        target=os.readlink(entry.name, dir_fd=directory),
                        mode=stat.S_IMODE(mode),
                    )
                elif kind == "file":
                    descriptor = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                        dir_fd=directory,
                    )
                    with os.fdopen(descriptor, "rb") as source:
                        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                            continue
                        content = source.read()
            except OSError as error:
                manifest[relative] = replace(manifest[relative], error=str(error))
                continue
            if subdirectory is not None:
                try:
                    visit(subdirectory, relative + "/")
                finally:
                    os.close(subdirectory)
            if content is not None:
                digest = hashlib.sha256(content).hexdigest()
                target = blobs / digest
                if not target.exists():
                    pending = blobs / f"{digest}.tmp"
                    try:
                        pending.write_bytes(content)
                        pending.chmod(0o400)
                        pending.replace(target)
                    finally:
                        pending.unlink(missing_ok=True)
                manifest[relative] = WorkspaceEntry(
                    "file", digest, len(content), mode=manifest[relative].mode
                )

    try:
        descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        return {".": WorkspaceEntry("directory", error=str(error))}
    try:
        visit(descriptor)
    finally:
        os.close(descriptor)
    return manifest


def validate_workspace_path(path: str) -> None:
    """Trusted task paths must remain inside the workspace."""
    if (
        not path
        or "\x00" in path
        or path.startswith("/")
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"Invalid workspace path {path!r}")


def snapshot_workspace(
    manifest: dict[str, WorkspaceEntry],
    run_dir: Path,
    limits: Limits,
    *,
    destination: str = "submission",
) -> Submission:
    """Materialize the recorded submission, independently of the live workspace."""
    if len(manifest) > limits.submission_files:
        raise ValueError("Workspace exceeds the submission file count limit")
    size_bytes = 0
    for path, entry in manifest.items():
        validate_workspace_path(path)
        if entry.error:
            raise ValueError(f"Cannot snapshot {path!r}: {entry.error}")
        if entry.kind not in {"file", "directory"}:
            raise ValueError(f"{path!r} must be a regular file or directory")
        if entry.kind == "file":
            if entry.sha256 is None or entry.size_bytes is None:
                raise ValueError(f"Cannot snapshot unreadable file {path!r}")
            size_bytes += entry.size_bytes
    if size_bytes > limits.submission_bytes:
        raise ValueError("Workspace exceeds the submission size limit")
    validate_workspace_path(destination)
    target = run_dir / destination
    target.mkdir(parents=True)
    for path, entry in sorted(manifest.items()):
        file_path = target / path
        if entry.kind == "directory":
            file_path.mkdir(parents=True, exist_ok=True)
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(run_dir / "blobs" / str(entry.sha256), file_path)
            file_path.chmod(0o400 | ((entry.mode or 0) & 0o111))
    return Submission(
        str(target.relative_to(run_dir)),
        _submission_sha256(manifest),
        size_bytes,
        len(manifest),
        dict(manifest),
    )


def _submission_sha256(manifest: dict[str, WorkspaceEntry]) -> str:
    payload = json.dumps(
        {
            path: {"kind": entry.kind, "sha256": entry.sha256, "mode": entry.mode}
            for path, entry in manifest.items()
        },
        sort_keys=True,
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_submission(submission: Submission, run_dir: Path) -> None:
    """Verify the retained grading snapshot against its recorded content identity."""
    validate_workspace_path(submission.path)
    root = run_dir / submission.path
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Submission snapshot is missing or replaced")
    actual = {str(path.relative_to(root)) for path in root.rglob("*")}
    if actual != set(submission.manifest):
        raise ValueError("Submission snapshot paths differ from its manifest")
    size_bytes = 0
    for name, entry in submission.manifest.items():
        validate_workspace_path(name)
        path = root / name
        kind = _KINDS.get(stat.S_IFMT(path.lstat().st_mode), "unknown")
        if kind != entry.kind or kind not in ("file", "directory"):
            raise ValueError("Submission snapshot entry type differs")
        if kind == "file":
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            size = path.stat().st_size
            if digest != entry.sha256 or size != entry.size_bytes:
                raise ValueError("Submission snapshot content differs")
            if stat.S_IMODE(path.stat().st_mode) & 0o111 != (entry.mode or 0) & 0o111:
                raise ValueError("Submission snapshot executable mode differs")
            size_bytes += size
    if (
        _submission_sha256(submission.manifest) != submission.sha256
        or len(actual) != submission.file_count
        or size_bytes != submission.size_bytes
    ):
        raise ValueError("Submission snapshot identity or counts differ")
