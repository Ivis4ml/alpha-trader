"""SQLite-based trade history tracking for Alpha Trader."""

from __future__ import annotations

import pathlib
import sqlite3
from datetime import datetime

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "trades.db"


def _ensure_dir():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol          TEXT NOT NULL,
            action          TEXT NOT NULL DEFAULT 'sell_call',
            strike          REAL NOT NULL,
            expiry          TEXT NOT NULL,
            contracts       INTEGER NOT NULL,
            premium_per_contract REAL NOT NULL,
            total_premium   REAL NOT NULL,
            delta           REAL,
            otm_pct         REAL,
            opened_at       TEXT NOT NULL,
            closed_at       TEXT,
            close_price     REAL,
            pnl             REAL,
            status          TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'expired', 'closed', 'assigned'))
        );

        CREATE TABLE IF NOT EXISTS weekly_summary (
            week_start      TEXT PRIMARY KEY,
            total_premium   REAL NOT NULL DEFAULT 0,
            target          REAL NOT NULL DEFAULT 0,
            trades_count    INTEGER NOT NULL DEFAULT 0,
            symbols         TEXT NOT NULL DEFAULT ''
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Core CRUD
# ---------------------------------------------------------------------------

def record_trade(
    symbol: str,
    strike: float,
    expiry: str,
    contracts: int,
    premium_per_contract: float,
    delta: float | None = None,
    otm_pct: float | None = None,
    action: str = "sell_call",
) -> int:
    """Insert a new trade and return its row id."""
    total_premium = premium_per_contract * contracts * 100
    opened_at = datetime.now().isoformat()

    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO trades
               (symbol, action, strike, expiry, contracts,
                premium_per_contract, total_premium, delta, otm_pct,
                opened_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')""",
            (symbol, action, strike, expiry, contracts,
             premium_per_contract, total_premium, delta, otm_pct,
             opened_at),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def close_trade(
    symbol: str,
    expiry: str,
    strike: float,
    status: str = "expired",
    close_price: float | None = None,
) -> int:
    """Close matching open trade(s). Returns number of rows updated."""
    closed_at = datetime.now().isoformat()
    conn = _connect()
    try:
        # Find matching open trades
        rows = conn.execute(
            """SELECT id, premium_per_contract, contracts FROM trades
               WHERE symbol = ? AND expiry = ? AND strike = ? AND status = 'open'""",
            (symbol, expiry, strike),
        ).fetchall()

        updated = 0
        for row in rows:
            pnl = _calc_pnl(
                premium_per_contract=row["premium_per_contract"],
                contracts=row["contracts"],
                status=status,
                close_price=close_price,
            )
            conn.execute(
                """UPDATE trades
                   SET status = ?, closed_at = ?, close_price = ?, pnl = ?
                   WHERE id = ?""",
                (status, closed_at, close_price, pnl, row["id"]),
            )
            updated += 1

        conn.commit()
        return updated
    finally:
        conn.close()


def _calc_pnl(
    premium_per_contract: float,
    contracts: int,
    status: str,
    close_price: float | None = None,
) -> float:
    """Calculate P&L for a trade.

    - expired:  full premium kept  -> premium * contracts * 100
    - closed:   partial premium    -> (premium - close_price) * contracts * 100
    - assigned: track premium P&L  -> premium * contracts * 100
      (stock-level P&L tracked separately)
    """
    multiplier = contracts * 100
    if status == "expired":
        return premium_per_contract * multiplier
    elif status == "closed":
        cp = close_price if close_price is not None else 0.0
        return (premium_per_contract - cp) * multiplier
    elif status == "assigned":
        # Just track the premium portion
        return premium_per_contract * multiplier
    return 0.0


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_open_trades() -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY expiry"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_history(limit: int = 50) -> list[dict]:
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY opened_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_weekly_summary(weeks: int = 8) -> list[dict]:
    """Return premium totals grouped by ISO week for the last N weeks."""
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT
                 date(opened_at, 'weekday 0', '-6 days') AS week_start,
                 SUM(total_premium) AS total_premium,
                 COUNT(*) AS trades_count,
                 GROUP_CONCAT(DISTINCT symbol) AS symbols
               FROM trades
               GROUP BY week_start
               ORDER BY week_start DESC
               LIMIT ?""",
            (weeks,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_monthly_summary(months: int = 6) -> list[dict]:
    """Return premium totals grouped by month."""
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT
                 strftime('%%Y-%%m', opened_at) AS month,
                 SUM(total_premium) AS total_premium,
                 COUNT(*) AS trades_count,
                 GROUP_CONCAT(DISTINCT symbol) AS symbols,
                 SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) AS realized_pnl
               FROM trades
               GROUP BY month
               ORDER BY month DESC
               LIMIT ?""",
            (months,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_cumulative_pnl() -> dict:
    """Return cumulative P&L statistics."""
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS total_trades,
                 SUM(total_premium) AS total_premium_collected,
                 SUM(CASE WHEN pnl IS NOT NULL THEN pnl ELSE 0 END) AS realized_pnl,
                 SUM(CASE WHEN status = 'open' THEN total_premium ELSE 0 END) AS unrealized_premium,
                 SUM(CASE WHEN status = 'expired' THEN 1 ELSE 0 END) AS expired_count,
                 SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) AS closed_count,
                 SUM(CASE WHEN status = 'assigned' THEN 1 ELSE 0 END) AS assigned_count,
                 SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) AS open_count
               FROM trades"""
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()
