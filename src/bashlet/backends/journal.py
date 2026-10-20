"""Durable remote request identities, outcomes and token usage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class RemoteFuture[T](Protocol):
    def result(self, timeout: float | None = None) -> T: ...


class RequestConflict(ValueError):
    """A request cannot proceed under the recorded remote identity or state."""


class RequestJournal:
    """Persist each attempt before submission and block matching pending work.

    Transactions serialize callers across processes. Unknown remote outcomes
    remain pending after errors or interruption, including after reopening.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, fingerprint TEXT, operation TEXT,
                request_group TEXT, state TEXT, evidence TEXT, created_at TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                request_id TEXT, recorded_at TEXT, evidence TEXT
            )""")
        self.path.chmod(0o600)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def bind_account(self, workspace_cuid: str) -> None:
        if not workspace_cuid:
            raise RequestConflict("A server workspace identity is required")
        with self._transaction() as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key='workspace'"
            ).fetchone()
            if row is not None and row[0] != workspace_cuid:
                raise RequestConflict(
                    "This journal belongs to another PyTRIO workspace"
                )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('workspace', ?)",
                (workspace_cuid,),
            )

    def begin(self, *, operation: str, group: str, payload: dict[str, Any]) -> str:
        if not operation or not group:
            raise ValueError("A request operation and group are required")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"operation": operation, "payload": payload},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self._transaction() as db:
            if db.execute(
                "SELECT 1 FROM requests WHERE state='pending' AND fingerprint=?",
                (fingerprint,),
            ).fetchone():
                raise RequestConflict(
                    "Matching remote work is still pending; "
                    "resolve it before resubmitting"
                )
            identity = uuid.uuid4().hex
            db.execute(
                """INSERT INTO requests
                (id, fingerprint, operation, request_group, state, evidence, created_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
                (
                    identity,
                    fingerprint,
                    operation,
                    group,
                    json.dumps({"payload": payload}, allow_nan=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
        return identity

    def _annotate(
        self, db: sqlite3.Connection, request_id: str, evidence: dict[str, Any]
    ) -> None:
        row = db.execute(
            "SELECT evidence FROM requests WHERE id=?", (request_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown journal request")
        merged = json.loads(row[0]) | evidence
        db.execute(
            "UPDATE requests SET evidence=? WHERE id=?",
            (json.dumps(merged, allow_nan=False), request_id),
        )
        db.execute(
            "INSERT INTO events VALUES (?, ?, ?)",
            (request_id, datetime.now(UTC).isoformat(), json.dumps(evidence)),
        )

    def annotate(self, request_id: str, evidence: dict[str, Any]) -> None:
        with self._transaction() as db:
            self._annotate(db, request_id, evidence)

    def complete(self, request_id: str, *, failed: bool = False) -> None:
        with self._transaction() as db:
            if (
                db.execute(
                    "SELECT 1 FROM requests WHERE id=?", (request_id,)
                ).fetchone()
                is None
            ):
                raise ValueError("Unknown journal request")
            db.execute(
                "UPDATE requests SET state=? WHERE id=? AND state='pending'",
                ("failed" if failed else "completed", request_id),
            )

    def execute[T](
        self,
        *,
        operation: str,
        group: str,
        payload: dict[str, Any],
        submit: Callable[[str], RemoteFuture[T]],
        timeout: float,
        terminal_failure: Callable[[BaseException], bool] = lambda error: False,
    ) -> tuple[T, str]:
        request_id = self.begin(operation=operation, group=group, payload=payload)
        try:
            value = submit(request_id).result(timeout=timeout)
        except BaseException as error:
            self.annotate(request_id, {"error_type": type(error).__name__})
            if terminal_failure(error):
                self.complete(request_id, failed=True)
            raise
        self.complete(request_id)
        return value, request_id

    def resolve(
        self, request_id: str, *, evidence: dict[str, Any], outcome: str
    ) -> None:
        """Resolve uncertain work only with explicit remote outcome evidence."""
        if (
            outcome not in ("completed", "failed")
            or not isinstance(evidence, dict)
            or not evidence
        ):
            raise ValueError("A terminal outcome and completion evidence are required")
        with self._transaction() as db:
            self._annotate(db, request_id, {"resolution": evidence})
            db.execute("UPDATE requests SET state=? WHERE id=?", (outcome, request_id))

    def snapshot(self) -> dict[str, Any]:
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM requests ORDER BY rowid").fetchall()
        entries = [
            {
                "id": row["id"],
                "fingerprint": row["fingerprint"],
                "operation": row["operation"],
                "group": row["request_group"],
                "state": row["state"],
                "created_at": row["created_at"],
                "evidence": json.loads(row["evidence"]),
            }
            for row in rows
        ]
        return {
            "journal": str(self.path),
            "entries": entries,
            "pending_count": sum(e["state"] == "pending" for e in entries),
        }


def summarize_requests(
    snapshot: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "journal": snapshot.get("journal"),
        "request_ids": [entry["id"] for entry in entries],
        "count": len(entries),
        "pending_count": sum(entry["state"] == "pending" for entry in entries),
    }
