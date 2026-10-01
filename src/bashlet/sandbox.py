"""Linux execution boundary. No host fallback is permitted."""

import json
import math
import os
import selectors
import signal
import subprocess
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .config import EnvironmentConfig
from .models import Limits, Observation


class Sandbox:
    def __init__(self, environment: EnvironmentConfig) -> None:
        self.environment = environment

    @property
    def limits(self) -> Limits:
        environment = self.environment
        return Limits(
            timeout_seconds=environment.timeout_seconds,
            output_bytes=environment.output_bytes,
            submission_bytes=environment.submission_bytes,
            submission_files=environment.submission_files,
            memory_bytes=environment.memory_bytes,
            file_bytes=environment.file_bytes,
        )

    @property
    def bwrap(self) -> str:
        return self.environment.bwrap

    @property
    def resource_limits(self) -> dict[str, int]:
        """The prlimit settings used both for launching and for the run record."""
        return {
            "as": self.environment.memory_bytes,
            "fsize": self.environment.file_bytes,
            "cpu": max(1, math.ceil(self.environment.timeout_seconds)),
            "core": 0,
            "nofile": self.environment.nofile,
        }

    def execute(
        self,
        workspace: Path,
        argv: list[str],
        *,
        stdin: bytes = b"",
        readonly: bool = False,
        merge_output: bool = False,
    ) -> Observation:
        started = time.monotonic()
        if any("\x00" in argument for argument in argv):
            message = "Command contains a NUL byte"
            return Observation(
                "command_failed",
                output=message if merge_output else "",
                stderr="" if merge_output else message,
            )
        with (
            tempfile.TemporaryFile() as status_file,
            tempfile.TemporaryFile() as input_file,
        ):
            input_file.write(stdin)
            input_file.seek(0)
            command = [
                "/usr/bin/prlimit",
                *(f"--{name}={value}" for name, value in self.resource_limits.items()),
                "--",
                self.bwrap,
                "--unshare-all",
                "--unshare-user",
                "--die-with-parent",
                "--new-session",
                "--cap-drop",
                "ALL",
                "--json-status-fd",
                str(status_file.fileno()),
            ]
            for bind in self.environment.ro_bind:
                command.extend(["--ro-bind", bind.source, bind.target])
            for link in self.environment.symlink:
                command.extend(["--symlink", link.source, link.target])
            command.extend(
                [
                    "--proc",
                    "/proc",
                    "--dev",
                    "/dev",
                    "--tmpfs",
                    "/tmp",
                    "--ro-bind" if readonly else "--bind",
                    str(workspace),
                    "/workspace",
                    "--chdir",
                    "/workspace",
                    "--clearenv",
                    "--setenv",
                    "PATH",
                    self.environment.path,
                    "--setenv",
                    "LANG",
                    self.environment.lang,
                    "--",
                    *argv,
                ]
            )
            try:
                process = subprocess.Popen(
                    command,
                    stdin=input_file,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT if merge_output else subprocess.PIPE,
                    env={
                        "PATH": self.environment.path,
                        "LANG": self.environment.lang,
                    },
                    pass_fds=(status_file.fileno(),),
                    start_new_session=True,
                )
            except (OSError, ValueError) as error:
                return Observation(
                    "infrastructure_error",
                    output=str(error) if merge_output else "",
                    stderr="" if merge_output else str(error),
                )
            stdout, stderr = bytearray(), bytearray()
            output_truncated = False
            outcome: Observation | None = None
            assert process.stdout is not None
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ, stdout)
                    if process.stderr is not None:
                        selector.register(process.stderr, selectors.EVENT_READ, stderr)
                    while selector.get_map() or process.poll() is None:
                        remaining = self.limits.timeout_seconds - (
                            time.monotonic() - started
                        )
                        if remaining <= 0:
                            outcome = Observation("timeout")
                            break
                        for key, _ in selector.select(min(remaining, 0.05)):
                            chunk = os.read(key.fd, 8192)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            available = (
                                self.limits.output_bytes - len(stdout) - len(stderr)
                            )
                            key.data.extend(chunk[:available])
                            if len(chunk) > available:
                                output_truncated = True
            except KeyboardInterrupt:
                # Keep captured output while the finally block stops all children.
                outcome = Observation("interrupted")
            finally:
                # Killing bwrap also kills the namespace's init and detached children.
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                process.stdout.close()
                if process.stderr is not None:
                    process.stderr.close()
            if outcome is None:
                status_file.seek(0)
                reports = [json.loads(line) for line in status_file if line.strip()]
                if not any("exit-code" in report for report in reports):
                    outcome = Observation("infrastructure_error")
                else:
                    outcome = Observation(
                        "ok" if process.returncode == 0 else "command_failed"
                    )
            return replace(
                outcome,
                output=stdout.decode("utf-8", errors="replace") if merge_output else "",
                stdout="" if merge_output else stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
                output_truncated=output_truncated,
                exit_code=process.returncode,
                duration_seconds=time.monotonic() - started,
            )
