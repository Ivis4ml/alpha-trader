"""Regression tests for Stage 2 data integrity invariants.

These tests verify the 5 critical properties that prevent training label
corruption. Each test uses an isolated in-memory SQLite DB.
"""

from __future__ import annotations

import json
import sqlite3
import datetime as dt

import pytest

# ---------------------------------------------------------------------------
# Fixtures: isolated in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def db(monkeypatch, tmp_path):
    """Provide an isolated DB and patch all db module paths to use it."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.db.DB_PATH", db_path)

    from src.db import _connect
    conn = _connect()  # creates tables
    conn.close()
    return db_path


def _insert_observation(db, **kwargs):
    """Helper to insert a candidate_observation row."""
    from src.db import _connect
    defaults = {
        "scan_ts": "2026-01-10T10:00:00",
        "scan_id": "scan_001",
        "symbol": "TEST",
        "expiry": "2026-01-17",
        "strike": 100.0,
        "stock_price": 95.0,
        "features": json.dumps({"premium": 2.0, "delta": 0.2}),
        "hard_filter_passed": 1,
        "reject_reasons": None,
        "score": 0.7,
        "score_breakdown": json.dumps({"income": 0.8}),
        "flags": None,
        "market_regime": "balanced",
        "vix": 18.0,
        "iv_rank": 50.0,
        "chosen_by_user": 0,
    }
    defaults.update(kwargs)
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO candidate_observations
               (scan_ts, scan_id, symbol, expiry, strike, stock_price,
                features, hard_filter_passed, reject_reasons,
                score, score_breakdown, flags,
                market_regime, vix, iv_rank, chosen_by_user)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (defaults["scan_ts"], defaults["scan_id"], defaults["symbol"],
             defaults["expiry"], defaults["strike"], defaults["stock_price"],
             defaults["features"], defaults["hard_filter_passed"],
             defaults["reject_reasons"], defaults["score"],
             defaults["score_breakdown"], defaults["flags"],
             defaults["market_regime"], defaults["vix"], defaults["iv_rank"],
             defaults["chosen_by_user"]),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _insert_trade(db, **kwargs):
    """Helper to insert a trade row."""
    from src.db import _connect
    defaults = {
        "symbol": "TEST",
        "action": "sell_call",
        "strike": 100.0,
        "expiry": "2026-01-17",
        "contracts": 1,
        "premium_per_contract": 2.0,
        "total_premium": 200.0,
        "delta": 0.2,
        "otm_pct": 5.0,
        "opened_at": "2026-01-10T14:00:00",
        "status": "expired",
        "closed_at": "2026-01-17T16:00:00",
        "pnl": 200.0,
    }
    defaults.update(kwargs)
    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO trades
               (symbol, action, strike, expiry, contracts,
                premium_per_contract, total_premium, delta, otm_pct,
                opened_at, status, closed_at, pnl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (defaults["symbol"], defaults["action"], defaults["strike"],
             defaults["expiry"], defaults["contracts"],
             defaults["premium_per_contract"], defaults["total_premium"],
             defaults["delta"], defaults["otm_pct"], defaults["opened_at"],
             defaults["status"], defaults["closed_at"], defaults["pnl"]),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _get_obs(db, obs_id):
    from src.db import _connect
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM candidate_observations WHERE id = ?", (obs_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test 1: Realized label only matches trade within 2 days BEFORE observation
# ---------------------------------------------------------------------------

class TestNoLookahead:
    """Realized label must not use future information."""

    def test_observation_after_trade_is_not_labeled(self, db):
        """An observation from AFTER the trade should not get realized label."""
        from src.db import backfill_outcomes

        # Trade opened Jan 10
        _insert_trade(db, opened_at="2026-01-10T14:00:00")
        # Observation from Jan 13 (3 days AFTER trade) — should NOT match
        obs_id = _insert_observation(db, scan_ts="2026-01-13T10:00:00")

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        assert obs["outcome_source"] != "realized", \
            "Observation after trade should not get realized label (lookahead)"

    def test_observation_before_trade_is_labeled(self, db):
        """An observation from before the trade (within 2 days) gets labeled."""
        from src.db import backfill_outcomes

        # Trade opened Jan 11
        _insert_trade(db, opened_at="2026-01-11T14:00:00")
        # Observation from Jan 10 (1 day before trade) — should match
        obs_id = _insert_observation(db, scan_ts="2026-01-10T10:00:00")

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        assert obs["outcome_source"] == "realized"

    def test_observation_too_old_not_labeled(self, db):
        """An observation from 5 days before trade should not match (>2 day window)."""
        from src.db import backfill_outcomes

        _insert_trade(db, opened_at="2026-01-15T14:00:00")
        # Observation from Jan 10 (5 days before) — outside 2-day window
        obs_id = _insert_observation(db, scan_ts="2026-01-10T10:00:00")

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        assert obs["outcome_source"] != "realized", \
            "Observation 5 days before trade should not match (outside 2-day window)"

    def test_user_chosen_preferred(self, db):
        """chosen_by_user=1 observation should be preferred over closer one."""
        from src.db import backfill_outcomes

        _insert_trade(db, opened_at="2026-01-11T14:00:00")
        # Closer observation but not user-chosen
        obs_close = _insert_observation(db, scan_ts="2026-01-11T09:00:00",
                                        scan_id="scan_a", chosen_by_user=0)
        # Further observation but user-chosen
        obs_chosen = _insert_observation(db, scan_ts="2026-01-10T10:00:00",
                                         scan_id="scan_b", chosen_by_user=1)

        backfill_outcomes("2026-01-17")
        obs_c = _get_obs(db, obs_chosen)
        obs_f = _get_obs(db, obs_close)
        assert obs_c["outcome_source"] == "realized", "User-chosen should be preferred"
        # The non-chosen observation may get a counterfactual label (that's fine),
        # but it must NOT get outcome_source='realized'
        assert obs_f.get("outcome_source") != "realized", \
            "Non-chosen should not be labeled as realized (counterfactual is OK)"


# ---------------------------------------------------------------------------
# Test 2: Multi-contract trade utility normalization
# ---------------------------------------------------------------------------

class TestUtilityNormalization:
    """PnL must be normalized by actual contract count."""

    def test_single_contract(self, db):
        """1 contract, $200 PnL → per-share = $2.00."""
        from src.db import backfill_outcomes

        _insert_trade(db, contracts=1, pnl=200.0,
                      opened_at="2026-01-10T14:00:00")
        obs_id = _insert_observation(db)

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        assert obs["outcome_source"] == "realized"
        # utility = capture_ratio - 0 (not assigned) = 200/(1*100) / 2.0 = 1.0
        assert obs["utility_label"] == 1.0

    def test_multi_contract_normalized(self, db):
        """3 contracts, $600 PnL → per-share = $2.00, same utility as 1 contract."""
        from src.db import backfill_outcomes

        _insert_trade(db, contracts=3, pnl=600.0,
                      premium_per_contract=2.0, total_premium=600.0,
                      opened_at="2026-01-10T14:00:00")
        obs_id = _insert_observation(db)

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        assert obs["outcome_source"] == "realized"
        # per-share = 600 / (3*100) = 2.0; capture_ratio = 2.0/2.0 = 1.0
        assert obs["utility_label"] == 1.0

    def test_multi_contract_loss(self, db):
        """3 contracts, -$900 PnL → per-share = -$3.00, clamped utility."""
        from src.db import backfill_outcomes

        _insert_trade(db, contracts=3, pnl=-900.0, status="assigned",
                      opened_at="2026-01-10T14:00:00")
        obs_id = _insert_observation(db)

        backfill_outcomes("2026-01-17")
        obs = _get_obs(db, obs_id)
        # per-share = -900/(3*100) = -3.0; capture_ratio = -3.0/2.0 = -1.5
        # clamp(-2.0, 1.0) → -1.5; utility = -1.5 - 0.5 = -2.0
        assert obs["utility_label"] == -2.0


# ---------------------------------------------------------------------------
# Test 3: heuristic_score not contaminated by blended score
# ---------------------------------------------------------------------------

class TestHeuristicScorePurity:
    """candidate_observations.score must always be pure heuristic."""

    def test_observation_stores_heuristic_not_blended(self):
        """_candidate_to_observation must use heuristic_score, not score."""
        from src.scan_engine import CandidateDecision, CandidateFeatures, _candidate_to_observation

        feat = CandidateFeatures(
            symbol="TEST", expiry="2026-01-17", strike=100.0,
            bid=1.9, ask=2.1, premium=2.0, delta=0.2, theta=0.05,
            implied_vol=0.3, dte=7, otm_pct=5.0, spread_pct=5.0,
            annualized_yield=50.0, open_interest=500, volume=100,
            atr_distance=1.5, earnings_gap=30, iv_rank=50.0,
            stock_price=95.0, cost_basis=80.0, allow_assignment=False,
            off_hours=False,
        )
        decision = CandidateDecision(
            features=feat, passed=True, reject_reasons=[],
            score=0.85,  # blended score (if model active)
            breakdown={"income": 0.8},
            flags=[],
            heuristic_score=0.70,  # pure heuristic
            model_score=0.95,
        )

        obs = _candidate_to_observation(decision)
        assert obs["score"] == 0.70, \
            f"Observation should store heuristic_score (0.70), not blended (0.85), got {obs['score']}"


# ---------------------------------------------------------------------------
# Test 4: VIX consistent between train and inference
# ---------------------------------------------------------------------------

class TestVixConsistency:
    """VIX must flow from market context to inference features."""

    def test_apply_learner_scores_passes_vix(self):
        """_apply_learner_scores must include vix in feature rows."""
        import inspect
        from src.scan_engine import _apply_learner_scores

        src = inspect.getsource(_apply_learner_scores)
        # Must accept vix parameter
        assert "vix: float" in src
        # Must put it in the feature dict
        assert '"vix": vix' in src
        # Must NOT hardcode None
        assert '"vix": None' not in src

    def test_scan_symbol_passes_market_vix(self):
        """scan_symbol must pass market.vix to _apply_learner_scores."""
        import inspect
        from src.scan_engine import scan_symbol

        src = inspect.getsource(scan_symbol)
        assert "vix=market.vix" in src


# ---------------------------------------------------------------------------
# Test 5: Same contract across multiple trades doesn't reuse observation
# ---------------------------------------------------------------------------

class TestOneTradeOneObservation:
    """Each trade must label at most one observation, even for repeated contracts."""

    def test_two_trades_same_contract_label_different_observations(self, db):
        """Two trades on same contract should each label their own closest observation."""
        from src.db import backfill_outcomes

        # Trade 1: opened Jan 10
        _insert_trade(db, opened_at="2026-01-10T14:00:00",
                      pnl=200.0, status="expired")
        # Trade 2: opened Jan 11 (same contract, different trade)
        _insert_trade(db, opened_at="2026-01-11T14:00:00",
                      pnl=150.0, status="expired")

        # Observation from Jan 10 scan → should match trade 1
        obs1 = _insert_observation(db, scan_ts="2026-01-10T10:00:00", scan_id="scan_a")
        # Observation from Jan 11 scan → should match trade 2
        obs2 = _insert_observation(db, scan_ts="2026-01-11T10:00:00", scan_id="scan_b")

        backfill_outcomes("2026-01-17")

        o1 = _get_obs(db, obs1)
        o2 = _get_obs(db, obs2)

        # Both should be labeled
        assert o1["outcome_source"] == "realized"
        assert o2["outcome_source"] == "realized"

        # They should have different PnL values (from different trades)
        # Trade 1: pnl=200 → per-share=2.0 → per-contract=200
        # Trade 2: pnl=150 → per-share=1.5 → per-contract=150
        assert o1["terminal_pnl"] != o2["terminal_pnl"], \
            "Two trades should label observations with their own PnL"

    def test_one_trade_does_not_label_two_observations(self, db):
        """A single trade must label at most one observation."""
        from src.db import backfill_outcomes

        _insert_trade(db, opened_at="2026-01-10T14:00:00", pnl=200.0)

        # Two observations from different scans, both within 2-day window
        obs1 = _insert_observation(db, scan_ts="2026-01-09T10:00:00", scan_id="scan_a")
        obs2 = _insert_observation(db, scan_ts="2026-01-10T09:00:00", scan_id="scan_b")

        backfill_outcomes("2026-01-17")

        o1 = _get_obs(db, obs1)
        o2 = _get_obs(db, obs2)

        realized_count = sum(1 for o in [o1, o2] if o["outcome_source"] == "realized")
        assert realized_count == 1, \
            f"One trade should label exactly 1 observation, got {realized_count}"


# ---------------------------------------------------------------------------
# Test: Utility label clamping
# ---------------------------------------------------------------------------

class TestUtilityLabelClamping:
    """Utility labels must be clamped to [-2.5, 1.0]."""

    def test_deep_itm_clamped(self):
        from src.db import _compute_utility_label
        # Deep ITM: premium $2, lost $42 → capture_ratio = -42/2 = -21
        u = _compute_utility_label(entry_premium=2.0, terminal_pnl=-42.0, assigned=True)
        assert u == -2.5, f"Deep ITM should be clamped to -2.5, got {u}"

    def test_full_premium_captured(self):
        from src.db import _compute_utility_label
        u = _compute_utility_label(entry_premium=2.0, terminal_pnl=2.0, assigned=False)
        assert u == 1.0

    def test_assigned_penalty_applied(self):
        from src.db import _compute_utility_label
        # Assigned but kept some premium: capture_ratio = 0.5, penalty = 0.5 → 0.0
        u = _compute_utility_label(entry_premium=2.0, terminal_pnl=1.0, assigned=True)
        assert u == 0.0
