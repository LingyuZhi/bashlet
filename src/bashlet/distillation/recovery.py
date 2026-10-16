"""Verify explicit reconciliation evidence for rejected rollouts without updates."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Reconciliation evidence must be an object")
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _sampling_request_reconciled(
    path: Path, receipt: dict[str, Any], before: dict[str, Any],
    record: dict[str, Any], batch: dict[str, Any],
) -> bool:
    resolution = receipt["sampling_request"]
    entry = _read(Path(receipt["archived"]["pending-sampling-request.json"]["copy"]))
    original = json.loads(entry["evidence"])
    payload = original["payload"]
    operation = entry.get("operation")
    if operation == "capability_probe":
        if (
            payload.get("scoring") is not False
            or payload.get("sampling", {}).get("max_tokens") != 1
            or not payload.get("prompt_tokens")
            or record.get("context_tokens") != []
            or record.get("trajectory") != []
            or record.get("usage", {}).get("steps") != 0
            or record.get("usage", {}).get("generated_tokens") != 0
        ):
            return False
        prefix = "Capability probe failed: PyTRIO actor.request_failed (HTTP 0,"
    elif operation == "sampling":
        if payload.get("prompt_tokens") != record.get("context_tokens"):
            return False
        prefix = "RuntimeError: PyTRIO actor.request_failed (HTTP 0,"
    else:
        return False
    if (
        resolution.get("id") != entry.get("id")
        or not original.get("request_id")
        or resolution.get("request_id") != original["request_id"]
        or resolution.get("response_abandoned") is not True
        or resolution.get("remote_outcome") != "unknown"
        or resolution.get("update_submitted") is not False
        or resolution.get("full_ceiling_retained") is not True
        or entry.get("state") != "pending"
        or payload.get("run_id") != before.get("run_id")
        or not record.get("reason", "").startswith(prefix)
        or not payload.get("model_path")
        or not batch["student_policy"].endswith(payload["model_path"])
        or record.get("policy") != batch["student_policy"]
    ):
        return False
    ledger = Path(before["ledger"]).resolve()
    with closing(
        sqlite3.connect(ledger.as_uri() + "?mode=ro", uri=True, timeout=2)
    ) as db:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only = ON")
        current = db.execute(
            "SELECT * FROM requests WHERE id=?", (entry["id"],)
        ).fetchone()
    if current is None:
        return False
    evidence = json.loads(current["evidence"])
    abandoned = evidence.get("sampling_abandoned", {})
    return (
        current["operation"] == operation and current["state"] == "failed"
        and current["fingerprint"] == entry["fingerprint"]
        and current["upper_micros"] == entry["upper_micros"]
        and current["charged_micros"] == current["upper_micros"]
        and current["billing_basis"] == "upper_bound"
        and evidence.get("payload") == payload
        and evidence.get("request_id") == original["request_id"]
        and abandoned.get("recovery_receipt") == str(path)
        and abandoned.get("remote_outcome") == "unknown"
        and abandoned.get("response_will_not_be_used") is True
    )


def _receipt_matches(
    path: Path, batch: dict[str, Any], row: dict[str, Any], root: Path, step: int,
) -> bool:
    receipt = _read(path)
    episode = Path(batch["episode"])
    index = batch["index"]
    sampling_failure = receipt.get("action") == (
        "explicit user-authorized retry after sampling transport failure without update"
    )
    if (
        not path.resolve().is_relative_to(root.parent / "recoveries")
        or not episode.resolve().is_relative_to(root / "rollouts")
        or (
            not sampling_failure and receipt.get("action") != (
                "explicit user-authorized retry of a visit rejected before task rollout"
            )
        )
        or receipt.get("status") != "reconciled; waiting for existing resume entrypoint"
        or receipt.get("visit_id") != row.get("visit_id")
        or type(receipt.get("optimizer_step")) is not int
        or receipt["optimizer_step"] != step
        or type(receipt.get("pending_requests")) is not int
        or receipt["pending_requests"] != 0
        or "pending_update" not in receipt
        or receipt["pending_update"] is not None
        or receipt.get("rejected_batch_index") != index
        or receipt.get("rejected_episode") != str(episode)
        or receipt.get("rejected_episode_sha256") != _digest(episode / "run.json")
    ):
        return False
    record = _read(episode / "run.json")
    if (
        record.get("task_id") != row.get("task_id")
        or not receipt.get("reason")
        or record.get("reason") != receipt["reason"]
    ):
        return False
    if sampling_failure:
        if (
            record.get("status") != "infrastructure_error"
            or record.get("usage", {}).get("steps") != batch.get("interaction_steps")
            or len(record.get("trajectory", [])) != batch.get("interaction_steps")
        ):
            return False
    else:
        if (
            record.get("status") not in {"model_error", "infrastructure_error"}
            or record.get("trajectory") != []
            or batch.get("interaction_steps") != 0
            or record.get("usage", {}).get("steps") != 0
            or record.get("usage", {}).get("generated_tokens") != 0
        ):
            return False
        # The Harness checks isolation before probing or sampling the model.
        # Other infrastructure errors require their own reconciliation proof.
        if record["status"] == "infrastructure_error" and (
            record.get("context_tokens") != []
            or record.get("isolation", {}).get("passed") is not False
            or record.get("isolation", {}).get("reason") != record["reason"]
        ):
            return False
    archived = receipt["archived"]
    prefix = root.name
    required = {
        f"{prefix}/pilot.json", f"{prefix}/training/opsd.json",
        f"{prefix}/training/backend.json", "workflow.json",
        f"{prefix}/rollouts/{episode.name}/run.json",
    }
    if not isinstance(archived, dict) or not required <= archived.keys():
        return False
    for relative, reference in archived.items():
        copy = Path(reference["copy"])
        if (
            copy.resolve() != (path.parent / "before" / relative).resolve()
            or not copy.resolve().is_relative_to(path.parent / "before")
            or _digest(copy) != reference["sha256"]
        ):
            return False
    before = _read(Path(archived[f"{prefix}/training/opsd.json"]["copy"]))
    visits = _read(Path(archived[f"{prefix}/pilot.json"]["copy"]))
    if sampling_failure and not _sampling_request_reconciled(
        path, receipt, before, record, batch,
    ):
        return False
    return (
        before.get("optimizer_step") == step
        and "pending_update" in before
        and before["pending_update"] is None
        and before["batches"][index] == batch
        and any(
            visit.get("visit_id") == row.get("visit_id")
            and visit.get("task_id") == row.get("task_id")
            and visit.get("batch_index") == index
            and visit.get("episode") == str(episode)
            and visit.get("status") == "rejected"
            for visit in visits["training_tasks"]
        )
    )


def reconciled_rejection(
    batch: dict[str, Any], pilot: dict[str, Any], root: Path, optimizer_step: int,
) -> bool:
    """Accept a rejected batch only with a preserved, explicit no-update retry proof.

    The proof records the optimizer step *at rejection*. Later updates therefore
    do not invalidate it, and it cannot authorize a new failed attempt.
    """
    if (
        batch.get("status") != "rejected" or "update" in batch
        or batch.get("supervised_tokens") != 0
    ):
        return False
    try:
        for row in pilot.get("training_tasks", []):
            for attempt in [row, *row.get("attempts", [])]:
                path = attempt.get("retry_receipt")
                if isinstance(path, str) and _receipt_matches(
                    Path(path), batch, row, root.resolve(), optimizer_step,
                ):
                    return True
    except (
        OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
        sqlite3.Error,
    ):
        return False
    return False
