"""Record Task Workspace states with shared, content-addressed file blobs."""

import hashlib
import os
import stat
from dataclasses import replace
from pathlib import Path

from .models import WorkspaceEntry, WorkspaceKind

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
                manifest[relative] = WorkspaceEntry(kind)
                if kind == "directory":
                    subdirectory = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory,
                    )
                elif kind == "symlink":
                    manifest[relative] = WorkspaceEntry(
                        kind, target=os.readlink(entry.name, dir_fd=directory)
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
                    target.write_bytes(content)
                    target.chmod(0o400)
                manifest[relative] = WorkspaceEntry("file", digest, len(content))

    try:
        descriptor = os.open(workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        return {".": WorkspaceEntry("directory", error=str(error))}
    try:
        visit(descriptor)
    finally:
        os.close(descriptor)
    return manifest
