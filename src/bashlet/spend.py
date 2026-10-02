"""Durable cash reservations shared by every paid experiment operation."""

import hashlib
import json
import math
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

import httpx
import yaml

_MICROS = 1_000_000
DEFAULT_POLICY = Path(__file__).with_name("configs") / "spend.yaml"
_SOURCE_POLICY = Path(__file__).resolve().parents[2] / "configs" / "spend.yaml"
if _SOURCE_POLICY.is_file():
    DEFAULT_POLICY = _SOURCE_POLICY


class BudgetRefused(Exception):
    """No paid work may be submitted under the current cash constraints."""


def _micros(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise BudgetRefused("A finite, non-negative cost upper bound is required")
    try:
        money = Decimal(str(value))
    except InvalidOperation:
        raise BudgetRefused("Invalid cash amount") from None
    if not money.is_finite() or money < 0:
        raise BudgetRefused("Cash amounts must be finite and non-negative")
    return int((money * _MICROS).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class SpendPolicy:
    regular_cny: float = 150
    reserve_cny: float = 50
    project_cny: float = 200
    funded_cny: float = 10
    groups: dict[str, float] = field(
        default_factory=lambda: {
            "pytrio-smoke": 0.50,
            "opsd-pilot": 10,
            "development-baseline": 2,
        }
    )

    @classmethod
    def read(cls, path: Path) -> "SpendPolicy":
        data = yaml.safe_load(path.read_text())
        return cls(**data["limits"])

    def validate(self) -> None:
        for value in (
            self.regular_cny,
            self.reserve_cny,
            self.project_cny,
            self.funded_cny,
            *self.groups.values(),
        ):
            _micros(value)
        if self.regular_cny + self.reserve_cny != self.project_cny:
            raise ValueError("Regular plan plus reserve must equal the project cap")
        if not self.groups or any(not name for name in self.groups):
            raise ValueError("Named sub-budgets are required")


class RemoteFuture[T](Protocol):
    def result(self, timeout: float | None = None) -> T: ...


class SpendLedger:
    """SQLite transactions make reservations atomic across threads and processes.

    Completion consumes the full reserved ceiling until a bill is reconciled.
    Unknown outcomes stay pending. A matching pending payload cannot be sent
    again, even from another process or after restarting the experiment.
    """

    def __init__(self, path: Path, *, policy: SpendPolicy) -> None:
        policy.validate()
        self.path = path.resolve()
        self.policy = policy
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with self._transaction() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
            )
            db.execute("""CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, fingerprint TEXT, operation TEXT,
                budget_group TEXT,
                pool TEXT, upper_micros INTEGER, charged_micros INTEGER DEFAULT 0,
                state TEXT, billing_basis TEXT, evidence TEXT, created_at TEXT
            )""")
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                request_id TEXT, recorded_at TEXT, evidence TEXT
            )""")
            encoded = json.dumps(asdict(policy), sort_keys=True)
            existing = db.execute(
                "SELECT value FROM metadata WHERE key='policy'"
            ).fetchone()
            if existing is not None and existing[0] != encoded:
                previous = json.loads(existing[0])
                current = asdict(policy)
                previous_groups = previous.pop("groups")
                current_groups = current.pop("groups")
                if previous != current or any(
                    current_groups.get(name) != limit
                    for name, limit in previous_groups.items()
                ):
                    raise ValueError(
                        "Ledger policy differs; restarting cannot reset cash limits"
                    )
                # Add allocations under unchanged global caps; retain every charge.
                db.execute("UPDATE metadata SET value=? WHERE key='policy'", (encoded,))
                db.execute(
                    "INSERT INTO events VALUES (?, ?, ?)",
                    (
                        "policy",
                        datetime.now(UTC).isoformat(),
                        json.dumps(
                            {
                                "previous": json.loads(existing[0]),
                                "current": asdict(policy),
                            }
                        ),
                    ),
                )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('policy', ?)", (encoded,)
            )
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
            raise BudgetRefused("A server workspace identity is required")
        with self._transaction() as db:
            row = db.execute(
                "SELECT value FROM metadata WHERE key='workspace'"
            ).fetchone()
            if row is not None and row[0] != workspace_cuid:
                raise BudgetRefused("This ledger belongs to another PyTRIO workspace")
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES ('workspace', ?)",
                (workspace_cuid,),
            )

    def reserve(
        self,
        *,
        operation: str,
        group: str,
        payload: dict[str, Any],
        upper_cny: object,
        pool: str = "regular",
    ) -> str:
        upper = _micros(upper_cny)
        if upper <= 0:
            raise BudgetRefused("Paid work needs a positive cost upper bound")
        if pool not in ("regular", "reserve") or not operation or not group:
            raise BudgetRefused(
                "A cost category, sub-budget and valid pool are required"
            )
        if group not in self.policy.groups:
            raise BudgetRefused(f"Unknown sub-budget {group!r}")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"operation": operation, "payload": payload},
                sort_keys=True,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        with self._transaction() as db:
            entries = db.execute("SELECT * FROM requests").fetchall()
            if any(row["charged_micros"] > row["upper_micros"] for row in entries):
                raise BudgetRefused(
                    "A reconciled bill exceeded its ceiling; pricing must be reviewed"
                )
            if any(
                row["state"] == "pending" and row["fingerprint"] == fingerprint
                for row in entries
            ):
                raise BudgetRefused(
                    "Matching paid work is still pending; "
                    "reconcile it before resubmitting"
                )

            def committed(rows: list[sqlite3.Row]) -> int:
                return sum(
                    row["upper_micros"]
                    if row["state"] == "pending"
                    else row["charged_micros"]
                    for row in rows
                )

            total = committed(entries)
            # Names describe checks; a sub-budget cannot replace a hard limit.
            checks = [
                ("project", total, _micros(self.policy.project_cny)),
                ("funded", total, _micros(self.policy.funded_cny)),
                (
                    pool,
                    committed([row for row in entries if row["pool"] == pool]),
                    _micros(
                        self.policy.regular_cny
                        if pool == "regular"
                        else self.policy.reserve_cny
                    ),
                ),
                (
                    group,
                    committed([row for row in entries if row["budget_group"] == group]),
                    _micros(self.policy.groups[group]),
                ),
            ]
            for name, used, cap in checks:
                if used + upper > cap:
                    raise BudgetRefused(
                        f"Cash budget {name!r} cannot cover this request"
                    )
            identity = uuid.uuid4().hex
            db.execute(
                """INSERT INTO requests
                (id, fingerprint, operation, budget_group, pool, upper_micros,
                 state, billing_basis, evidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', 'unsettled', ?, ?)""",
                (
                    identity,
                    fingerprint,
                    operation,
                    group,
                    pool,
                    upper,
                    json.dumps({"payload": payload}, allow_nan=False),
                    datetime.now(UTC).isoformat(),
                ),
            )
            return identity

    def annotate(self, request_id: str, evidence: dict[str, Any]) -> None:
        with self._transaction() as db:
            row = db.execute(
                "SELECT evidence FROM requests WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                raise ValueError("Unknown ledger request")
            merged = json.loads(row[0]) | evidence
            db.execute(
                "UPDATE requests SET evidence=? WHERE id=?",
                (json.dumps(merged), request_id),
            )
            db.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                (
                    request_id,
                    datetime.now(UTC).isoformat(),
                    json.dumps(evidence),
                ),
            )

    def complete(self, request_id: str, *, failed: bool = False) -> None:
        """Keep a conservative charge even for completed failures or discarded data."""
        with self._transaction() as db:
            db.execute(
                """UPDATE requests SET state=?, charged_micros=upper_micros,
                billing_basis='upper_bound' WHERE id=? AND state='pending'""",
                ("failed" if failed else "completed", request_id),
            )

    def execute[T](
        self,
        *,
        operation: str,
        group: str,
        payload: dict[str, Any],
        upper_cny: object,
        submit: Callable[[str], RemoteFuture[T]],
        timeout: float,
        terminal_failure: Callable[[BaseException], bool] = lambda error: False,
        pool: str = "regular",
    ) -> tuple[T, str]:
        """Reserve durably *before* calling an SDK/API, including every retry."""
        request_id = self.reserve(
            operation=operation,
            group=group,
            payload=payload,
            upper_cny=upper_cny,
            pool=pool,
        )
        try:
            value = submit(request_id).result(timeout=timeout)
        except BaseException as error:
            self.annotate(request_id, {"error_type": type(error).__name__})
            if terminal_failure(error):
                self.complete(request_id, failed=True)
            raise
        self.complete(request_id)
        return value, request_id

    def reconcile(
        self,
        request_id: str,
        *,
        actual_cny: object,
        evidence: dict[str, Any],
        outcome: str,
    ) -> None:
        """Release a ceiling with verified completion and billing evidence."""
        actual = _micros(actual_cny)
        if outcome not in ("completed", "failed") or not evidence:
            raise ValueError("A terminal outcome and billing evidence are required")
        self.annotate(
            request_id, {"reconciliation": evidence, "actual_cny": actual / _MICROS}
        )
        with self._transaction() as db:
            db.execute(
                """UPDATE requests SET charged_micros=?, state=?,
                billing_basis='confirmed' WHERE id=?""",
                (actual, outcome, request_id),
            )

    def snapshot(self) -> dict[str, Any]:
        with self._transaction() as db:
            rows = db.execute("SELECT * FROM requests ORDER BY rowid").fetchall()
            entries = [
                {
                    "id": row["id"],
                    "operation": row["operation"],
                    "group": row["budget_group"],
                    "pool": row["pool"],
                    "upper_cny": row["upper_micros"] / _MICROS,
                    "charged_cny": row["charged_micros"] / _MICROS,
                    "state": row["state"],
                    "billing_basis": row["billing_basis"],
                    "created_at": row["created_at"],
                    "evidence": json.loads(row["evidence"]),
                }
                for row in rows
            ]
            charged = sum(row["charged_micros"] for row in rows)
            pending = sum(
                row["upper_micros"] for row in rows if row["state"] == "pending"
            )
            return {
                "ledger": str(self.path),
                "policy": asdict(self.policy),
                "entries": entries,
                "charged_cny": charged / _MICROS,
                "pending_cny": pending / _MICROS,
                "funded_remaining_cny": (
                    _micros(self.policy.funded_cny) - charged - pending
                )
                / _MICROS,
                "project_remaining_cny": (
                    _micros(self.policy.project_cny) - charged - pending
                )
                / _MICROS,
            }


def summarize_cost(
    snapshot: dict[str, Any], entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Report one run's entries against the same point-in-time project ledger."""
    return {
        "ledger": snapshot.get("ledger"),
        "request_ids": [entry["id"] for entry in entries],
        "charged_upper_bound_cny": round(sum(e["charged_cny"] for e in entries), 6),
        "pending_upper_bound_cny": round(
            sum(e["upper_cny"] for e in entries if e["state"] == "pending"), 6
        ),
        "actual_cny": round(sum(e["charged_cny"] for e in entries), 6)
        if entries and all(e["billing_basis"] == "confirmed" for e in entries)
        else None,
        "billing_note": "Unreconciled ceilings are not verified actual charges.",
        **{
            k: snapshot[k]
            for k in ("funded_remaining_cny", "project_remaining_cny")
            if k in snapshot
        },
        **(
            {
                "project_upper_cny": snapshot["charged_cny"] + snapshot["pending_cny"],
                "project_pending_cny": snapshot["pending_cny"],
            }
            if "charged_cny" in snapshot
            else {}
        ),
    }


class TokenPricing:
    """Shared live catalog checks and conservative token charges."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def verify(self, model: str, timeout: float) -> dict[str, Any]:
        """Read the same public catalog as the official price page before spending."""
        response = httpx.get(self.config["price_catalog_url"], timeout=timeout)
        response.raise_for_status()
        catalog = response.json()
        if (
            catalog.get("currency") != "CNY"
            or catalog.get("token_unit") != "per_1m_tokens"
            or catalog.get("rounding") != "ceil"
        ):
            raise BudgetRefused("Unknown live PyTRIO billing units")
        item = next(
            (item for item in catalog["items"] if item["display_name"] == model), None
        )
        if item is None:
            raise BudgetRefused("Qwen3.5-4B pricing is not available")
        configured = self.config["prices_cny_per_million"]
        for remote, local in (
            ("prefill", "prefill"),
            ("sample", "sample"),
            ("train", "train"),
            ("cached_prefill", "cache"),
        ):
            price = item["prices"][remote]
            if price["billing_unit"] != "per_1m_tokens" or Decimal(
                str(price["unit_price"])
            ) / 100 > Decimal(str(configured[local])):
                raise BudgetRefused(
                    "Live prices exceed the configured ceilings; review spend.yaml"
                )
        return {
            "version": catalog["version"],
            "currency": "CNY",
            "unit_price_currency": "fen",
            "rounding": catalog["rounding"],
            "model": item,
        }

    def upper_bound(
        self, prompt_tokens: int, max_tokens: int, *, scoring: bool = False
    ) -> Decimal:
        prices = self.config.get("prices_cny_per_million")
        if not isinstance(prices, dict) or any(
            type(prices.get(name)) not in (int, float)
            or not math.isfinite(prices[name])
            or prices[name] <= 0
            for name in ("prefill", "sample", "train")
        ):
            raise BudgetRefused("Unknown token pricing; cannot bound paid work")
        multiplier = self.config.get("ceiling_multiplier")
        if (
            not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier)
            or multiplier < 1
        ):
            raise BudgetRefused("A finite cost ceiling multiplier >= 1 is required")
        try:
            quantum = Decimal(str(self.config.get("billing_quantum_cny")))
        except InvalidOperation:
            raise BudgetRefused("Unknown billing rounding quantum") from None
        if not quantum.is_finite() or quantum < Decimal("0.01"):
            raise BudgetRefused(
                "A billing rounding quantum of at least one fen is required"
            )
        prefill = (
            max(prices[name] for name in ("prefill", "sample", "train"))
            if scoring
            else prices["prefill"]
        )

        def rounded(rate: float, tokens: int) -> Decimal:
            cost = Decimal(str(rate)) * tokens / 1_000_000
            return (cost / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum

        return Decimal(str(multiplier)) * (
            rounded(prefill, prompt_tokens) + rounded(prices["sample"], max_tokens)
        )

    def training_bound(self, tokens: int) -> Decimal:
        # Bound all context tokens, including masked positions. Account for
        # both training and possible prefill categories without cache discounts.
        return 2 * self.upper_bound(tokens, 0, scoring=True)
