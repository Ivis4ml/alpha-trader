"""SQLite-based trade history + candidate observation tracking for Alpha Trader."""

from __future__ import annotations

import json
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

        CREATE TABLE IF NOT EXISTS candidate_observations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_ts         TEXT NOT NULL,
            scan_id         TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            expiry          TEXT NOT NULL,
            strike          REAL NOT NULL,
            stock_price     REAL NOT NULL,
            -- Full feature vector as JSON (CandidateFeatures fields)
            features        TEXT NOT NULL,
            -- Hard filter result
            hard_filter_passed INTEGER NOT NULL,
            reject_reasons  TEXT,
            -- Scoring (NULL for rejected candidates)
            score           REAL,
            score_breakdown TEXT,
            flags           TEXT,
            -- Market context at scan time
            market_regime   TEXT,
            vix             REAL,
            iv_rank         REAL,
            -- Selection tracking (updated post-scan)
            chosen_by_llm   INTEGER NOT NULL DEFAULT 0,
            chosen_by_user  INTEGER NOT NULL DEFAULT 0,
            -- Outcome tracking (filled after expiry via backfill)
            realized_outcome TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_cobs_scan_id ON candidate_observations(scan_id);
        CREATE INDEX IF NOT EXISTS idx_cobs_symbol_expiry ON candidate_observations(symbol, expiry);
        CREATE INDEX IF NOT EXISTS idx_cobs_scan_ts ON candidate_observations(scan_ts);

        -- Policy layer: daily decision tracking
        CREATE TABLE IF NOT EXISTS policy_decisions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id     TEXT NOT NULL UNIQUE,
            decision_ts     TEXT NOT NULL,
            scan_id         TEXT,
            -- Portfolio state snapshot
            market_regime   TEXT,
            vix             REAL,
            spy_price       REAL,
            -- Portfolio context
            open_shorts     TEXT,           -- JSON: [{symbol, strike, expiry, dte, delta, pnl_pct}]
            shares_available TEXT,          -- JSON: {symbol: contracts_available}
            weekly_premium_so_far REAL,
            weekly_target   REAL,
            target_gap_pct  REAL,           -- (target - collected) / target * 100
            -- Summary
            actions_total   INTEGER NOT NULL DEFAULT 0,
            actions_chosen  INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_pd_decision_ts ON policy_decisions(decision_ts);

        CREATE TABLE IF NOT EXISTS policy_actions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id     TEXT NOT NULL,
            -- Action identity
            action_type     TEXT NOT NULL,   -- OPEN, ADD, HOLD, CLOSE, ROLL, SKIP, LET_EXPIRE
            symbol          TEXT NOT NULL,
            -- For OPEN/ADD/ROLL target
            target_expiry   TEXT,
            target_strike   REAL,
            target_premium  REAL,
            target_delta    REAL,
            target_contracts INTEGER,
            -- For CLOSE/ROLL source (existing short)
            source_expiry   TEXT,
            source_strike   REAL,
            source_entry_premium REAL,
            -- Scoring
            score           REAL,
            score_breakdown TEXT,            -- JSON
            urgency         TEXT,            -- HIGH, MEDIUM, LOW
            reason          TEXT,
            flags           TEXT,            -- JSON
            -- Selection
            chosen          INTEGER NOT NULL DEFAULT 0,
            chosen_by       TEXT,            -- llm, user, auto
            -- Trade linkage (set on promotion to realized — persists 1:1 match)
            matched_trade_id INTEGER,       -- trades.id that this action was promoted from
            matched_trade_id_source INTEGER, -- trades.id for ROLL source leg
            -- Outcome (filled on terminal event)
            rollout_source  TEXT NOT NULL DEFAULT 'recommendation',  -- recommendation | realized | counterfactual
            terminal_status TEXT,            -- expired, closed, assigned, rolled, skipped
            terminal_pnl    REAL,
            terminal_ts     TEXT,
            reward          REAL,            -- utility-weighted reward
            reward_confidence TEXT,          -- high, medium, low
            -- Full context
            features        TEXT,            -- JSON: full CandidateFeatures if applicable
            market_snapshot TEXT,            -- JSON: {vix, regime, iv_rank, stock_price}

            FOREIGN KEY (decision_id) REFERENCES policy_decisions(decision_id)
        );

        CREATE INDEX IF NOT EXISTS idx_pa_decision_id ON policy_actions(decision_id);
        CREATE INDEX IF NOT EXISTS idx_pa_symbol ON policy_actions(symbol);
        CREATE INDEX IF NOT EXISTS idx_pa_chosen ON policy_actions(chosen);

        CREATE TABLE IF NOT EXISTS policy_action_updates (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            action_id       INTEGER NOT NULL,
            update_ts       TEXT NOT NULL,
            -- Mark-to-market
            stock_price     REAL,
            option_price    REAL,
            delta_now       REAL,
            moneyness_pct   REAL,           -- (stock - strike) / strike * 100
            pnl_pct         REAL,           -- current P&L as % of entry premium
            -- Current advice from strategy.py
            current_advice  TEXT,            -- HOLD, CLOSE, ROLL, etc.
            advice_urgency  TEXT,
            -- Context
            vix             REAL,
            days_to_expiry  INTEGER,

            FOREIGN KEY (action_id) REFERENCES policy_actions(id)
        );

        CREATE INDEX IF NOT EXISTS idx_pau_action_id ON policy_action_updates(action_id);
        CREATE INDEX IF NOT EXISTS idx_pau_update_ts ON policy_action_updates(update_ts);
    """)
    # Add market context columns if missing (safe migration for existing DBs)
    for col, coltype in [
        ("regime", "TEXT"),
        ("iv_rank", "REAL"),
        ("vix", "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass  # column already exists
    # Add trade linkage columns to policy_actions (safe migration)
    for col, coltype in [
        ("matched_trade_id", "INTEGER"),
        ("matched_trade_id_source", "INTEGER"),
    ]:
        try:
            conn.execute(f"ALTER TABLE policy_actions ADD COLUMN {col} {coltype}")
        except sqlite3.OperationalError:
            pass
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
    regime: str | None = None,
    iv_rank: float | None = None,
    vix: float | None = None,
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
                opened_at, status, regime, iv_rank, vix)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
            (symbol, action, strike, expiry, contracts,
             premium_per_contract, total_premium, delta, otm_pct,
             opened_at, regime, iv_rank, vix),
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
        # Find the earliest matching open trade (LIMIT 1 aligns with YAML which
        # also removes only the first match, preventing state divergence on
        # partial closes or duplicate symbol/expiry/strike positions).
        row = conn.execute(
            """SELECT id, premium_per_contract, contracts FROM trades
               WHERE symbol = ? AND expiry = ? AND strike = ? AND status = 'open'
               ORDER BY id LIMIT 1""",
            (symbol, expiry, strike),
        ).fetchone()

        updated = 0
        if row:
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
            updated = 1

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
                 strftime('%Y-%m', opened_at) AS month,
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


# ---------------------------------------------------------------------------
# Candidate observations — records every candidate from each scan
# ---------------------------------------------------------------------------

def record_scan_candidates(
    scan_id: str,
    candidates: list[dict],
    market_regime: str | None = None,
    vix: float | None = None,
) -> int:
    """Bulk-insert candidate observations from a scan run.

    Each entry in *candidates* should have:
        symbol, expiry, strike, stock_price, features (dict),
        hard_filter_passed (bool), reject_reasons (list[str] | None),
        score (float | None), score_breakdown (dict | None),
        flags (list[str] | None), iv_rank (float | None)

    Returns the number of rows inserted.
    """
    scan_ts = datetime.now().isoformat()
    conn = _connect()
    try:
        rows = []
        for c in candidates:
            rows.append((
                scan_ts,
                scan_id,
                c["symbol"],
                c["expiry"],
                c["strike"],
                c["stock_price"],
                json.dumps(c["features"]),
                1 if c["hard_filter_passed"] else 0,
                json.dumps(c["reject_reasons"]) if c.get("reject_reasons") else None,
                c.get("score"),
                json.dumps(c["score_breakdown"]) if c.get("score_breakdown") else None,
                json.dumps(c["flags"]) if c.get("flags") else None,
                market_regime,
                vix,
                c.get("iv_rank"),
            ))
        conn.executemany(
            """INSERT INTO candidate_observations
               (scan_ts, scan_id, symbol, expiry, strike, stock_price,
                features, hard_filter_passed, reject_reasons,
                score, score_breakdown, flags,
                market_regime, vix, iv_rank)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def get_scan_observations(
    scan_id: str | None = None,
    symbol: str | None = None,
    passed_only: bool = False,
    limit: int = 500,
) -> list[dict]:
    """Query candidate observations with optional filters."""
    conn = _connect()
    try:
        clauses = []
        params: list = []
        if scan_id:
            clauses.append("scan_id = ?")
            params.append(scan_id)
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if passed_only:
            clauses.append("hard_filter_passed = 1")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = conn.execute(
            f"""SELECT * FROM candidate_observations
                {where}
                ORDER BY scan_ts DESC, score DESC
                LIMIT ?""",
            params + [limit],
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            # Deserialize JSON fields
            for field in ("features", "reject_reasons", "score_breakdown",
                          "flags", "realized_outcome"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results
    finally:
        conn.close()


def get_observation_stats() -> dict:
    """Summary statistics for candidate observations."""
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(*) AS total_observations,
                 COUNT(DISTINCT scan_id) AS total_scans,
                 COUNT(DISTINCT symbol) AS symbols_observed,
                 SUM(hard_filter_passed) AS passed_count,
                 SUM(CASE WHEN hard_filter_passed = 0 THEN 1 ELSE 0 END) AS rejected_count,
                 SUM(chosen_by_user) AS user_chosen_count,
                 SUM(CASE WHEN realized_outcome IS NOT NULL THEN 1 ELSE 0 END) AS outcomes_filled,
                 MIN(scan_ts) AS first_scan,
                 MAX(scan_ts) AS last_scan
               FROM candidate_observations"""
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def mark_chosen(scan_id: str, symbol: str, strike: float, expiry: str,
                by: str = "user") -> int:
    """Mark a candidate as chosen (by LLM or user). Returns rows updated."""
    col = "chosen_by_llm" if by == "llm" else "chosen_by_user"
    conn = _connect()
    try:
        cur = conn.execute(
            f"""UPDATE candidate_observations SET {col} = 1
                WHERE scan_id = ? AND symbol = ? AND strike = ? AND expiry = ?""",
            (scan_id, symbol, strike, expiry),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def backfill_outcomes(expiry: str) -> int:
    """Backfill realized outcomes for candidates with a given expiry date.

    Matches against the trades table to determine what actually happened.
    For candidates that weren't traded, checks if the stock closed above/below
    the strike (requires stock price at expiry — caller provides via update).

    Returns the number of rows updated.
    """
    conn = _connect()
    try:
        # For candidates that WERE traded: join with trades table
        cur = conn.execute(
            """UPDATE candidate_observations SET realized_outcome = (
                 SELECT json_object(
                   'traded', 1,
                   'status', t.status,
                   'pnl', t.pnl,
                   'premium_collected', t.total_premium
                 )
                 FROM trades t
                 WHERE t.symbol = candidate_observations.symbol
                   AND t.expiry = candidate_observations.expiry
                   AND t.strike = candidate_observations.strike
                   AND t.status != 'open'
               )
               WHERE expiry = ?
                 AND realized_outcome IS NULL
                 AND EXISTS (
                   SELECT 1 FROM trades t
                   WHERE t.symbol = candidate_observations.symbol
                     AND t.expiry = candidate_observations.expiry
                     AND t.strike = candidate_observations.strike
                     AND t.status != 'open'
                 )""",
            (expiry,),
        )
        updated = cur.rowcount
        conn.commit()
        return updated
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Policy decisions — daily portfolio-level decision tracking
# ---------------------------------------------------------------------------

def record_policy_decision(
    decision_id: str,
    scan_id: str | None,
    market_regime: str | None,
    vix: float | None,
    spy_price: float | None,
    open_shorts: list[dict],
    shares_available: dict[str, int],
    weekly_premium_so_far: float,
    weekly_target: float,
    actions_total: int = 0,
    actions_chosen: int = 0,
) -> None:
    """Record a daily policy decision (portfolio state snapshot)."""
    decision_ts = datetime.now().isoformat()
    target_gap = ((weekly_target - weekly_premium_so_far) / weekly_target * 100
                  if weekly_target > 0 else 0)
    conn = _connect()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO policy_decisions
               (decision_id, decision_ts, scan_id, market_regime, vix, spy_price,
                open_shorts, shares_available, weekly_premium_so_far, weekly_target,
                target_gap_pct, actions_total, actions_chosen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (decision_id, decision_ts, scan_id, market_regime, vix, spy_price,
             json.dumps(open_shorts), json.dumps(shares_available),
             weekly_premium_so_far, weekly_target, round(target_gap, 1),
             actions_total, actions_chosen),
        )
        conn.commit()
    finally:
        conn.close()


def record_policy_actions(decision_id: str, actions: list[dict]) -> int:
    """Bulk-insert policy actions for a decision. Returns row count.

    Each action dict should have:
        action_type, symbol, score, urgency, reason,
        and optionally: target_expiry, target_strike, target_premium, target_delta,
        target_contracts, source_expiry, source_strike, source_entry_premium,
        score_breakdown, flags, features, market_snapshot,
        chosen, chosen_by, rollout_source
    """
    conn = _connect()
    try:
        rows = []
        for a in actions:
            rows.append((
                decision_id,
                a["action_type"],
                a["symbol"],
                a.get("target_expiry"),
                a.get("target_strike"),
                a.get("target_premium"),
                a.get("target_delta"),
                a.get("target_contracts"),
                a.get("source_expiry"),
                a.get("source_strike"),
                a.get("source_entry_premium"),
                a.get("score"),
                json.dumps(a["score_breakdown"]) if a.get("score_breakdown") else None,
                a.get("urgency"),
                a.get("reason"),
                json.dumps(a["flags"]) if a.get("flags") else None,
                1 if a.get("chosen") else 0,
                a.get("chosen_by"),
                a.get("rollout_source", "recommendation"),
                json.dumps(a["features"]) if a.get("features") else None,
                json.dumps(a["market_snapshot"]) if a.get("market_snapshot") else None,
            ))
        conn.executemany(
            """INSERT INTO policy_actions
               (decision_id, action_type, symbol,
                target_expiry, target_strike, target_premium, target_delta,
                target_contracts, source_expiry, source_strike, source_entry_premium,
                score, score_breakdown, urgency, reason, flags,
                chosen, chosen_by, rollout_source, features, market_snapshot)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def record_action_update(
    action_id: int,
    stock_price: float | None = None,
    option_price: float | None = None,
    delta_now: float | None = None,
    moneyness_pct: float | None = None,
    pnl_pct: float | None = None,
    current_advice: str | None = None,
    advice_urgency: str | None = None,
    vix: float | None = None,
    days_to_expiry: int | None = None,
) -> None:
    """Record a daily mark-to-market update for a policy action."""
    conn = _connect()
    try:
        conn.execute(
            """INSERT INTO policy_action_updates
               (action_id, update_ts, stock_price, option_price, delta_now,
                moneyness_pct, pnl_pct, current_advice, advice_urgency,
                vix, days_to_expiry)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (action_id, datetime.now().isoformat(), stock_price, option_price,
             delta_now, moneyness_pct, pnl_pct, current_advice, advice_urgency,
             vix, days_to_expiry),
        )
        conn.commit()
    finally:
        conn.close()


def get_open_policy_actions(symbol: str | None = None) -> list[dict]:
    """Get policy actions that haven't reached terminal state yet."""
    conn = _connect()
    try:
        clause = "WHERE terminal_status IS NULL"
        params: list = []
        if symbol:
            clause += " AND symbol = ?"
            params.append(symbol)

        rows = conn.execute(
            f"""SELECT pa.*, pd.decision_ts, pd.market_regime
                FROM policy_actions pa
                JOIN policy_decisions pd ON pa.decision_id = pd.decision_id
                {clause}
                ORDER BY pd.decision_ts DESC""",
            params,
        ).fetchall()

        results = []
        for r in rows:
            d = dict(r)
            for fld in ("score_breakdown", "flags", "features", "market_snapshot"):
                if d.get(fld):
                    try:
                        d[fld] = json.loads(d[fld])
                    except (json.JSONDecodeError, TypeError):
                        pass
            results.append(d)
        return results
    finally:
        conn.close()


def finalize_policy_action(
    action_id: int,
    terminal_status: str,
    terminal_pnl: float | None = None,
    reward: float | None = None,
    reward_confidence: str = "high",
) -> None:
    """Mark a policy action as terminal (closed/expired/assigned/rolled/skipped)."""
    conn = _connect()
    try:
        conn.execute(
            """UPDATE policy_actions
               SET terminal_status = ?, terminal_pnl = ?, terminal_ts = ?,
                   reward = ?, reward_confidence = ?
               WHERE id = ?""",
            (terminal_status, terminal_pnl, datetime.now().isoformat(),
             reward, reward_confidence, action_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_policy_stats() -> dict:
    """Summary statistics for the policy layer.

    Counts meaningful actions (OPEN/ADD/CLOSE/ROLL) separately from
    passive actions (HOLD/SKIP) to avoid inflating optimization triggers.
    """
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT
                 COUNT(DISTINCT pd.decision_id) AS total_decisions,
                 COUNT(pa.id) AS total_actions,
                 SUM(pa.chosen) AS selected_actions,
                 -- Realized = user actually executed the trade
                 SUM(CASE WHEN pa.rollout_source = 'realized' THEN 1 ELSE 0 END) AS realized_actions,
                 -- Meaningful = OPEN/ADD/CLOSE/ROLL (not HOLD/SKIP/LET_EXPIRE)
                 SUM(CASE WHEN pa.action_type IN ('OPEN', 'ADD', 'CLOSE', 'ROLL')
                          AND pa.terminal_status IS NOT NULL THEN 1 ELSE 0 END) AS meaningful_terminal,
                 SUM(CASE WHEN pa.action_type IN ('OPEN', 'ADD', 'CLOSE', 'ROLL')
                          AND pa.rollout_source = 'realized' THEN 1 ELSE 0 END) AS meaningful_realized,
                 SUM(CASE WHEN pa.terminal_status IS NOT NULL THEN 1 ELSE 0 END) AS terminal_actions,
                 SUM(CASE WHEN pa.reward IS NOT NULL THEN 1 ELSE 0 END) AS rewarded_actions,
                 MIN(pd.decision_ts) AS first_decision,
                 MAX(pd.decision_ts) AS last_decision
               FROM policy_actions pa
               JOIN policy_decisions pd ON pa.decision_id = pd.decision_id"""
        ).fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()
