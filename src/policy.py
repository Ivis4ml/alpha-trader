"""Unified daily action slate generator + portfolio policy engine.

Merges three decision sources into one coherent action slate:
  - scan_engine.py: OPEN / ADD candidates (new covered call positions)
  - strategy.py:    HOLD / CLOSE / LET_EXPIRE advice (existing shorts)
  - roll.py:        ROLL candidates (replace expiring/ITM shorts)

Pipeline per /scan:
  1. Enumerate all feasible actions (action slate)
  2. Score each action via deterministic_prior + portfolio adjustments
  3. Portfolio allocator selects optimal subset
  4. Persist all actions (chosen + unchosen) for learning
  5. Backfill prior open actions (daily MTM + terminal outcomes)
  6. Compute 3-layer rewards (next_day, interim, terminal)
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field

from .config import (
    load_config, get_symbols, get_short_calls, get_position,
    contracts_available, get_delta_range, get_weekly_target,
)
from .data.fetcher import (
    fetch_stock, fetch_events, fetch_iv_stats, fetch_option_chain,
    MarketContext, RISK_FREE_RATE,
)
from .data.greeks import bs_call_price, black_scholes_greeks
from .data.events_calendar import assess_event_risk
from .scan_engine import (
    PortfolioScanResult,
    build_features, apply_hard_filters, score_candidate,
)
from .strategy import advise_position


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class PolicyAction:
    """One possible action in a daily decision slate."""
    action_type: str        # OPEN, ADD, HOLD, CLOSE, ROLL, SKIP, LET_EXPIRE
    symbol: str
    # For OPEN / ADD / ROLL target
    target_expiry: str | None = None
    target_strike: float | None = None
    target_premium: float | None = None
    target_delta: float | None = None
    target_contracts: int | None = None
    # For CLOSE / ROLL source
    source_expiry: str | None = None
    source_strike: float | None = None
    source_entry_premium: float | None = None
    # Scoring and context
    score: float = 0.0
    score_breakdown: dict = field(default_factory=dict)
    urgency: str = "LOW"        # HIGH, MEDIUM, LOW
    reason: str = ""
    flags: list[str] = field(default_factory=list)
    # Selection tracking
    # "selected" = allocator recommends this action (shadow mode output)
    # "executed" = user actually performed this trade (set later via mark_executed)
    # Only executed actions get rollout_source="realized" — selected-but-not-executed
    # are still "recommendation" and should be treated as counterfactual.
    selected: bool = False
    selected_by: str | None = None    # auto, llm
    executed: bool = False             # set later when user confirms trade
    # Feature context (for modeling)
    features: dict | None = None
    market_snapshot: dict | None = None


@dataclass
class PolicyDecision:
    """Complete daily action slate: all possible actions for the portfolio."""
    decision_id: str
    decision_ts: str
    scan_id: str | None
    # Market state
    market_regime: str | None = None
    vix: float | None = None
    spy_price: float | None = None
    # Portfolio state
    open_shorts_summary: list[dict] = field(default_factory=list)
    shares_available: dict[str, int] = field(default_factory=dict)
    weekly_premium_so_far: float = 0.0
    weekly_target: float = 0.0
    # Actions
    actions: list[PolicyAction] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Action generators
# ---------------------------------------------------------------------------

def _generate_open_actions(
    scan_result: PortfolioScanResult,
    config: dict,
) -> list[PolicyAction]:
    """Generate OPEN/ADD actions from scan results (top candidates per symbol)."""
    actions: list[PolicyAction] = []
    market = scan_result.market

    for symbol, sym_result in scan_result.symbols.items():
        avail = contracts_available(config, symbol)
        if avail <= 0:
            # SKIP: no contracts available
            actions.append(PolicyAction(
                action_type="SKIP",
                symbol=symbol,
                reason=f"No contracts available (all covered or at max %)",
                urgency="LOW",
                market_snapshot=_market_snap(market, sym_result.briefing.iv_stats.iv_rank,
                                            sym_result.briefing.stock.price),
            ))
            continue

        if not sym_result.candidates:
            actions.append(PolicyAction(
                action_type="SKIP",
                symbol=symbol,
                reason=f"No candidates passed hard filters ({sym_result.rejected_count} rejected)",
                urgency="LOW",
                market_snapshot=_market_snap(market, sym_result.briefing.iv_stats.iv_rank,
                                            sym_result.briefing.stock.price),
            ))
            continue

        # Check if symbol already has open shorts
        shorts = get_short_calls(config)
        has_existing = any(s["symbol"] == symbol for s in shorts)
        action_type = "ADD" if has_existing else "OPEN"

        # Top N candidates as possible actions
        for candidate in sym_result.candidates[:5]:
            f = candidate.features
            actions.append(PolicyAction(
                action_type=action_type,
                symbol=symbol,
                target_expiry=f.expiry,
                target_strike=f.strike,
                target_premium=f.premium,
                target_delta=f.delta,
                target_contracts=min(avail, 1),  # conservative: 1 at a time
                score=candidate.score,
                score_breakdown=candidate.breakdown,
                urgency="LOW",
                reason=f"Score {candidate.score:.3f} | Δ{f.delta:.2f} | {f.dte}DTE | ${f.premium:.2f}",
                flags=candidate.flags,
                features=_features_to_dict(f),
                market_snapshot=_market_snap(market, f.iv_rank, f.stock_price),
            ))

    return actions


def _generate_position_actions(
    config: dict,
    market: MarketContext,
) -> list[PolicyAction]:
    """Generate HOLD/CLOSE/ROLL/LET_EXPIRE actions for existing short calls."""
    actions: list[PolicyAction] = []
    shorts = get_short_calls(config)

    if not shorts:
        return actions

    today = dt.date.today()

    for sc in shorts:
        symbol = sc["symbol"]
        strike = sc["strike"]
        expiry_str = sc["expiry"]
        premium_received = sc.get("premium_received", 0)

        expiry = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        dte = (expiry - today).days

        stock = fetch_stock(symbol)
        price = stock.price
        events = fetch_events(symbol)
        is_itm = price >= strike

        # Estimate current option value
        T = max(dte, 0) / 365.0
        iv_est = 0.30
        estimated_value = bs_call_price(price, strike, T, RISK_FREE_RATE, iv_est)
        greeks = black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv_est, "call")

        # Get advice from strategy.py
        earnings_before_expiry = (
            events.days_to_earnings is not None
            and 0 <= events.days_to_earnings <= dte
        )
        advice = advise_position(
            entry_premium=premium_received,
            current_premium=estimated_value,
            dte=dte,
            is_itm=is_itm,
            earnings_before_expiry=earnings_before_expiry,
            config=config,
        )

        moneyness = (price - strike) / strike * 100
        pnl_pct = ((premium_received - estimated_value) / premium_received * 100
                   if premium_received > 0 else 0)

        snap = {
            "vix": market.vix,
            "regime": market.regime,
            "stock_price": price,
            "delta_now": round(greeks.delta, 4),
            "moneyness_pct": round(moneyness, 1),
            "pnl_pct": round(pnl_pct, 1),
            "dte": dte,
        }

        if advice.action == "ROLL":
            # Generate ROLL actions with scored candidates
            roll_actions = _generate_roll_candidates(
                symbol, price, strike, dte, premium_received,
                expiry_str, config, market, advice, snap,
            )
            actions.extend(roll_actions)

            # Also add a CLOSE alternative (user might prefer to just close)
            actions.append(PolicyAction(
                action_type="CLOSE",
                symbol=symbol,
                source_expiry=expiry_str,
                source_strike=strike,
                source_entry_premium=premium_received,
                score=0.3,  # lower than roll candidates
                urgency=advice.urgency,
                reason=f"Alternative to roll: buy to close at ~${estimated_value:.2f}",
                market_snapshot=snap,
            ))
        else:
            # Map strategy advice to action type
            action_type_map = {
                "HOLD": "HOLD",
                "CLOSE_PROFIT": "CLOSE",
                "CLOSE_STOP": "CLOSE",
                "CLOSE_EARNINGS": "CLOSE",
                "LET_EXPIRE": "LET_EXPIRE",
            }
            action_type = action_type_map.get(advice.action, "HOLD")

            # Score: urgency-based for position management
            urgency_score = {"HIGH": 0.9, "MEDIUM": 0.6, "LOW": 0.3}
            score = urgency_score.get(advice.urgency, 0.3)

            actions.append(PolicyAction(
                action_type=action_type,
                symbol=symbol,
                source_expiry=expiry_str,
                source_strike=strike,
                source_entry_premium=premium_received,
                score=score,
                urgency=advice.urgency,
                reason=advice.reason,
                market_snapshot=snap,
            ))

    return actions


def _generate_roll_candidates(
    symbol: str,
    price: float,
    current_strike: float,
    current_dte: int,
    entry_premium: float,
    source_expiry: str,
    config: dict,
    market: MarketContext,
    advice,
    snap: dict,
) -> list[PolicyAction]:
    """Generate scored ROLL candidates for a position that needs rolling."""
    actions: list[PolicyAction] = []

    position = get_position(config, symbol)
    iv_stats = fetch_iv_stats(symbol)
    events = fetch_events(symbol)
    stock = fetch_stock(symbol)
    strat = config.get("strategy", {})
    delta_range = get_delta_range(config, market.regime)

    try:
        candidates = fetch_option_chain(symbol, price, config, "call")
    except Exception:
        actions.append(PolicyAction(
            action_type="ROLL",
            symbol=symbol,
            source_expiry=source_expiry,
            source_strike=current_strike,
            source_entry_premium=entry_premium,
            score=0.5,
            urgency=advice.urgency,
            reason=f"{advice.reason} (no roll candidates fetched)",
            market_snapshot=snap,
        ))
        return actions

    # Filter: further out expiry, same or higher strike
    rollable = [c for c in candidates
                if c.dte > current_dte + 4 and c.strike >= current_strike]

    if not rollable:
        actions.append(PolicyAction(
            action_type="ROLL",
            symbol=symbol,
            source_expiry=source_expiry,
            source_strike=current_strike,
            source_entry_premium=entry_premium,
            score=0.5,
            urgency=advice.urgency,
            reason=f"{advice.reason} (no suitable roll targets)",
            market_snapshot=snap,
        ))
        return actions

    # Score through scan_engine pipeline
    passed_features = []
    for opt in rollable:
        feat = build_features(opt, stock, iv_stats, events, position)
        reasons = apply_hard_filters(feat, config)
        if not reasons:
            passed_features.append(feat)

    if not passed_features:
        return actions

    all_yields = [f.annualized_yield for f in passed_features]
    all_thetas = [f.theta for f in passed_features]
    all_ois = [f.open_interest for f in passed_features]
    earnings_buffer = strat.get("blackout_earnings_days", 7)

    scored = []
    for feat in passed_features:
        event_risk = assess_event_risk(
            symbol, feat.expiry,
            earnings_dte=events.days_to_earnings,
            earnings_buffer=earnings_buffer,
        )
        sc, breakdown, flags = score_candidate(
            feat, config, event_risk, all_yields, all_thetas, all_ois,
            delta_range=delta_range,
        )
        scored.append((sc, feat, breakdown, flags))

    scored.sort(key=lambda x: x[0], reverse=True)

    for sc, feat, breakdown, flags in scored[:3]:
        actions.append(PolicyAction(
            action_type="ROLL",
            symbol=symbol,
            target_expiry=feat.expiry,
            target_strike=feat.strike,
            target_premium=feat.premium,
            target_delta=feat.delta,
            target_contracts=1,
            source_expiry=source_expiry,
            source_strike=current_strike,
            source_entry_premium=entry_premium,
            score=sc,
            score_breakdown=breakdown,
            urgency=advice.urgency,
            reason=f"{advice.reason} → roll to {feat.expiry} ${feat.strike}",
            flags=flags,
            features=_features_to_dict(feat),
            market_snapshot=snap,
        ))

    return actions


# ---------------------------------------------------------------------------
# Deterministic prior scoring
# ---------------------------------------------------------------------------

@dataclass
class _PortfolioState:
    """Transient state for scoring and allocation — not persisted."""
    weekly_target: float
    weekly_so_far: float
    target_gap_pct: float           # (target - collected) / target * 100
    total_open_shorts: int          # number of open short positions
    symbol_short_count: dict        # {symbol: count of open shorts}
    total_available_contracts: int
    regime: str
    vix: float


def _build_portfolio_state(config: dict, market: MarketContext) -> _PortfolioState:
    from .db import get_weekly_summary
    shorts = get_short_calls(config)
    symbols = get_symbols(config)

    weekly_target = get_weekly_target(config)
    week_summary = get_weekly_summary(weeks=1)
    weekly_so_far = week_summary[0]["total_premium"] if week_summary else 0.0
    gap_pct = ((weekly_target - weekly_so_far) / weekly_target * 100
               if weekly_target > 0 else 0)

    sym_count: dict[str, int] = {}
    for s in shorts:
        sym_count[s["symbol"]] = sym_count.get(s["symbol"], 0) + s.get("contracts", 1)

    total_avail = sum(contracts_available(config, sym) for sym in symbols)

    return _PortfolioState(
        weekly_target=weekly_target,
        weekly_so_far=weekly_so_far,
        target_gap_pct=gap_pct,
        total_open_shorts=len(shorts),
        symbol_short_count=sym_count,
        total_available_contracts=total_avail,
        regime=market.regime,
        vix=market.vix,
    )


def _target_gap_bonus(state: _PortfolioState) -> float:
    """Bonus for opening new positions when behind weekly target. [0, 0.15]"""
    if state.target_gap_pct <= 0:
        return 0.0  # already met target, no bonus for chasing
    # Scale: 0 at 0% gap, 0.15 at 100% gap, capped
    return min(0.15, state.target_gap_pct / 100 * 0.15)


def _concentration_penalty(action: PolicyAction, state: _PortfolioState) -> float:
    """Penalty for over-concentrating in one symbol. [0, 0.20]"""
    existing = state.symbol_short_count.get(action.symbol, 0)
    if existing == 0:
        return 0.0
    # 0.05 per existing short in the same symbol, max 0.20
    return min(0.20, existing * 0.05)


def _contract_overuse_penalty(action: PolicyAction, state: _PortfolioState) -> float:
    """Penalty when portfolio is already heavily committed. [0, 0.10]"""
    if state.total_available_contracts <= 0:
        return 0.10
    # Ratio of used contracts
    total_open = state.total_open_shorts
    utilization = total_open / max(total_open + state.total_available_contracts, 1)
    if utilization > 0.7:
        return 0.10
    elif utilization > 0.5:
        return 0.05
    return 0.0


def _theta_carry_value(snap: dict) -> float:
    """Value of continuing to hold for theta decay. [0, 0.8]"""
    dte = snap.get("dte", 0)
    pnl_pct = snap.get("pnl_pct", 0)
    if dte <= 0:
        return 0.0
    # More value when there's still significant theta to capture
    # and position is profitable
    time_value = min(0.4, dte / 30 * 0.4)  # decays as DTE shrinks
    profit_bonus = min(0.4, max(0, pnl_pct) / 100 * 0.4)
    return time_value + profit_bonus


def _assignment_risk_penalty(snap: dict) -> float:
    """Penalty for ITM or near-ITM positions. [0, 0.4]"""
    moneyness = snap.get("moneyness_pct", 0)
    delta = snap.get("delta_now", 0)
    if moneyness <= -5:  # safely OTM
        return 0.0
    if moneyness > 0:  # ITM
        return min(0.4, 0.2 + moneyness / 100 * 0.5)
    # Near ATM: use delta as proxy
    return min(0.3, delta * 0.5)


def _premium_capture_value(snap: dict) -> float:
    """Value of taking profit now. [0, 0.8]"""
    pnl_pct = snap.get("pnl_pct", 0)
    if pnl_pct <= 0:
        return 0.0
    return min(0.8, pnl_pct / 100)


def _buyback_cost(snap: dict, entry_premium: float) -> float:
    """Cost of buying back the option. [0, 0.3]"""
    if entry_premium <= 0:
        return 0.0
    # Estimated buyback as fraction of entry premium
    pnl_pct = snap.get("pnl_pct", 0)
    remaining_frac = max(0, (100 - pnl_pct)) / 100
    return min(0.3, remaining_frac * 0.3)


def _tail_risk_reduction(snap: dict) -> float:
    """Bonus for closing positions that are risky. [0, 0.2]"""
    moneyness = snap.get("moneyness_pct", 0)
    dte = snap.get("dte", 0)
    if moneyness > -2 and dte > 3:  # near-ATM with time left
        return 0.15
    if moneyness > 0:  # ITM
        return 0.20
    return 0.0


def deterministic_prior(action: PolicyAction, state: _PortfolioState) -> float:
    """Compute the base utility score for an action.

    Returns a score in [0, 1] range. This is the heuristic prior —
    a learned model score can be blended in later via combine_scores().
    """
    snap = action.market_snapshot or {}

    if action.action_type in ("OPEN", "ADD"):
        base = action.score  # from scan_engine candidate scoring
        bonus = _target_gap_bonus(state)
        penalty = (_concentration_penalty(action, state)
                   + _contract_overuse_penalty(action, state))
        return max(0.0, min(1.0, base + bonus - penalty))

    if action.action_type == "HOLD":
        theta = _theta_carry_value(snap)
        risk = _assignment_risk_penalty(snap)
        return max(0.0, min(1.0, theta - risk))

    if action.action_type == "CLOSE":
        entry = action.source_entry_premium or 0
        capture = _premium_capture_value(snap)
        tail_bonus = _tail_risk_reduction(snap)
        cost = _buyback_cost(snap, entry)
        return max(0.0, min(1.0, capture + tail_bonus - cost))

    if action.action_type == "ROLL":
        # New candidate score + net roll credit value
        new_score = action.score  # from scan_engine scoring of roll target
        old_risk = _assignment_risk_penalty(snap)
        friction = 0.05  # fixed friction penalty for double-leg trade
        return max(0.0, min(1.0, new_score + old_risk * 0.5 - friction))

    if action.action_type == "LET_EXPIRE":
        pnl_pct = snap.get("pnl_pct", 0)
        dte = snap.get("dte", 0)
        if dte <= 3 and pnl_pct > 75:
            return 0.85  # high confidence: let it expire
        return 0.5

    if action.action_type == "SKIP":
        # Baseline: slightly positive to avoid always-trade bias
        return 0.3

    return 0.0


def combine_scores(
    base_score: float,
    model_score: float = 0.0,
    model_weight: float = 0.0,
) -> float:
    """Blend heuristic prior with model score.

    v1: model_weight=0 → pure heuristic.
    Phase 2: model_weight=0.3 → 70% heuristic + 30% model.
    """
    if model_weight <= 0:
        return base_score
    return (1 - model_weight) * base_score + model_weight * model_score


# ---------------------------------------------------------------------------
# Portfolio allocator
# ---------------------------------------------------------------------------

def portfolio_allocator(
    actions: list[PolicyAction],
    state: _PortfolioState,
    config: dict,
) -> list[PolicyAction]:
    """Select the optimal action subset from ranked feasible actions.

    Constraints:
    - Mutual exclusion: only ONE action per source position (same symbol +
      source_strike + source_expiry). CLOSE, ROLL, LET_EXPIRE, HOLD all
      compete for the same slot — highest-scoring wins.
    - Max new opens per day (default 2)
    - Don't over-concentrate in one symbol
    - Don't chase target beyond reasonable risk
    """
    strat = config.get("strategy", {})
    max_new_per_day = strat.get("max_new_positions_per_day", 2)
    max_per_symbol = strat.get("max_shorts_per_symbol", 3)

    selected: list[PolicyAction] = []
    new_opens = 0
    symbol_adds: dict[str, int] = {}

    # Track which source positions already have a selected action.
    # Key: (symbol, source_strike, source_expiry) → prevents selecting
    # both ROLL-to-A and ROLL-to-B for the same old short.
    source_claimed: set[tuple] = set()

    def _source_key(a: PolicyAction) -> tuple | None:
        """Unique key for the source position an action operates on."""
        if a.source_strike and a.source_expiry:
            return (a.symbol, a.source_strike, a.source_expiry)
        return None

    # Sort by score descending — highest-scored action for each source wins
    ranked = sorted(actions, key=lambda a: a.score, reverse=True)

    for action in ranked:
        src_key = _source_key(action)

        # ── Position management: CLOSE / ROLL / LET_EXPIRE / HOLD ──
        if action.action_type in ("CLOSE", "ROLL", "LET_EXPIRE", "HOLD"):
            # Mutual exclusion: only one action per source position
            if src_key and src_key in source_claimed:
                continue  # a higher-scored action already claimed this position

            # HIGH urgency: always select (but still respect mutual exclusion)
            if action.urgency == "HIGH" and action.action_type != "HOLD":
                action.selected = True
                action.selected_by = "auto"
                selected.append(action)
                if src_key:
                    source_claimed.add(src_key)
                continue

            # HOLD: select if no better action claimed this position
            if action.action_type == "HOLD":
                action.selected = True
                action.selected_by = "auto"
                selected.append(action)
                if src_key:
                    source_claimed.add(src_key)
                continue

            # Non-HIGH CLOSE / ROLL: select if score is high enough
            if action.score >= 0.5:
                action.selected = True
                action.selected_by = "auto"
                selected.append(action)
                if src_key:
                    source_claimed.add(src_key)
            continue

        # ── OPEN / ADD: subject to allocation limits ──
        if action.action_type in ("OPEN", "ADD"):
            if new_opens >= max_new_per_day:
                continue
            sym_existing = state.symbol_short_count.get(action.symbol, 0)
            sym_today = symbol_adds.get(action.symbol, 0)
            if sym_existing + sym_today >= max_per_symbol:
                continue

            # Don't force trades when target is already met
            if state.target_gap_pct <= 0 and action.score < 0.6:
                continue

            action.selected = True
            action.selected_by = "auto"
            selected.append(action)
            new_opens += 1
            symbol_adds[action.symbol] = sym_today + 1
            continue

        # ── SKIP ──
        if action.action_type == "SKIP":
            action.selected = True
            action.selected_by = "auto"
            selected.append(action)

    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_action_slate(
    scan_result: PortfolioScanResult,
    config: dict | None = None,
) -> PolicyDecision:
    """Generate a complete daily action slate from a scan result.

    Pipeline:
    1. Enumerate all actions (OPEN/ADD/HOLD/CLOSE/ROLL/SKIP)
    2. Score each via deterministic_prior + portfolio adjustments
    3. Run portfolio allocator to select optimal subset
    4. Package as PolicyDecision for persistence
    """
    if config is None:
        config = load_config()

    market = scan_result.market
    decision_id = uuid.uuid4().hex[:12]

    # Build portfolio state for scoring
    state = _build_portfolio_state(config, market)

    # 1. Enumerate all actions
    open_actions = _generate_open_actions(scan_result, config)
    position_actions = _generate_position_actions(config, market)
    all_actions = position_actions + open_actions

    # 2. Score each action with portfolio-aware prior
    for action in all_actions:
        base = deterministic_prior(action, state)
        action.score = combine_scores(base)
        action.score_breakdown["policy_prior"] = round(base, 4)

    # 3. Allocate: select best actions subject to portfolio constraints
    # (modifies actions in-place via action.selected = True)
    portfolio_allocator(all_actions, state, config)

    # Build portfolio state summary for persistence
    shorts = get_short_calls(config)
    today = dt.date.today()
    open_shorts_summary = []
    for sc in shorts:
        expiry = dt.datetime.strptime(sc["expiry"], "%Y-%m-%d").date()
        dte = (expiry - today).days
        open_shorts_summary.append({
            "symbol": sc["symbol"],
            "strike": sc["strike"],
            "expiry": sc["expiry"],
            "dte": dte,
            "contracts": sc.get("contracts", 1),
            "premium_received": sc.get("premium_received", 0),
        })

    symbols = get_symbols(config)
    shares_avail = {sym: contracts_available(config, sym) for sym in symbols}

    return PolicyDecision(
        decision_id=decision_id,
        decision_ts=dt.datetime.now().isoformat(),
        scan_id=scan_result.scan_id,
        market_regime=market.regime,
        vix=market.vix,
        spy_price=market.spy_price,
        open_shorts_summary=open_shorts_summary,
        shares_available=shares_avail,
        weekly_premium_so_far=state.weekly_so_far,
        weekly_target=state.weekly_target,
        actions=all_actions,
    )


def persist_decision(decision: PolicyDecision) -> str:
    """Save a PolicyDecision and its actions to the database. Returns decision_id."""
    from .db import record_policy_decision, record_policy_actions

    record_policy_decision(
        decision_id=decision.decision_id,
        scan_id=decision.scan_id,
        market_regime=decision.market_regime,
        vix=decision.vix,
        spy_price=decision.spy_price,
        open_shorts=decision.open_shorts_summary,
        shares_available=decision.shares_available,
        weekly_premium_so_far=decision.weekly_premium_so_far,
        weekly_target=decision.weekly_target,
        actions_total=len(decision.actions),
        actions_chosen=sum(1 for a in decision.actions if a.selected),
    )

    action_dicts = []
    for a in decision.actions:
        action_dicts.append({
            "action_type": a.action_type,
            "symbol": a.symbol,
            "target_expiry": a.target_expiry,
            "target_strike": a.target_strike,
            "target_premium": a.target_premium,
            "target_delta": a.target_delta,
            "target_contracts": a.target_contracts,
            "source_expiry": a.source_expiry,
            "source_strike": a.source_strike,
            "source_entry_premium": a.source_entry_premium,
            "score": a.score,
            "score_breakdown": a.score_breakdown,
            "urgency": a.urgency,
            "reason": a.reason,
            "flags": a.flags,
            "chosen": a.selected,       # DB column is still "chosen" for compat
            "chosen_by": a.selected_by,
            # IMPORTANT: everything starts as "recommendation". Only mark_executed()
            # promotes to "realized". This prevents allocator selections from being
            # treated as ground-truth training labels.
            "rollout_source": "recommendation",
            "features": a.features,
            "market_snapshot": a.market_snapshot,
        })

    record_policy_actions(decision.decision_id, action_dicts)
    return decision.decision_id


def mark_executed(action_id: int) -> None:
    """Promote a policy action from recommendation to realized.

    Call this when the user actually executes a trade that matches
    a policy action. This is the ONLY path to rollout_source="realized".
    """
    from .db import _connect
    conn = _connect()
    try:
        conn.execute(
            """UPDATE policy_actions
               SET rollout_source = 'realized'
               WHERE id = ?""",
            (action_id,),
        )
        conn.commit()
    finally:
        conn.close()


def auto_mark_executed_from_trades() -> int:
    """Match trades against policy actions and promote to realized.

    Design constraints:
    - Each trade (by id) promotes at most ONE policy action. This is
      enforced DURABLY via matched_trade_id on policy_actions — once
      a trade is linked, it's excluded from future runs. The in-memory
      consumed_trades set handles within-run dedup across passes.
    - Each policy action is promoted at most once (enforced by the
      rollout_source='recommendation' filter in queries).
    - ROLL requires target opened AFTER source closed (no ABS — strict
      chronological ordering prevents false matches from unrelated
      close/open sequences).

    Returns count of actions promoted.
    """
    from .db import _connect
    conn = _connect()
    try:
        promoted = 0

        # Load trade IDs already permanently linked from prior runs.
        already_linked = conn.execute(
            """SELECT matched_trade_id FROM policy_actions
               WHERE matched_trade_id IS NOT NULL
               UNION
               SELECT matched_trade_id_source FROM policy_actions
               WHERE matched_trade_id_source IS NOT NULL"""
        ).fetchall()
        consumed_trade_ids: set[int] = {r[0] for r in already_linked}

        # ── Pass 1: ROLL (most specific — requires two-leg match) ──
        roll_pairs = conn.execute(
            """SELECT t_src.id AS source_trade_id,
                      t_tgt.id AS target_trade_id,
                      t_src.symbol,
                      t_src.strike AS source_strike,
                      t_src.expiry AS source_expiry,
                      t_tgt.strike AS target_strike,
                      t_tgt.expiry AS target_expiry,
                      t_tgt.opened_at AS target_opened_at
               FROM trades t_src
               JOIN trades t_tgt
                 ON t_tgt.symbol = t_src.symbol
                AND t_tgt.strike != t_src.strike
                -- Target must open AFTER source closes (forward chronology)
                AND julianday(t_tgt.opened_at) >= julianday(t_src.closed_at)
                AND julianday(t_tgt.opened_at) - julianday(t_src.closed_at) <= 2
               WHERE t_src.status IN ('expired', 'closed', 'assigned')
                 AND t_src.closed_at IS NOT NULL"""
        ).fetchall()

        for rp in roll_pairs:
            p = dict(rp)
            if (p["source_trade_id"] in consumed_trade_ids
                    or p["target_trade_id"] in consumed_trade_ids):
                continue

            best_roll = conn.execute(
                """SELECT pa.id,
                          ABS(julianday(pd.decision_ts) - julianday(?)) AS dt_diff
                   FROM policy_actions pa
                   JOIN policy_decisions pd ON pa.decision_id = pd.decision_id
                   WHERE pa.rollout_source = 'recommendation'
                     AND pa.chosen = 1
                     AND pa.action_type = 'ROLL'
                     AND pa.symbol = ?
                     AND pa.source_strike = ?
                     AND pa.source_expiry = ?
                     AND pa.target_strike = ?
                     AND pa.target_expiry = ?
                     AND ABS(julianday(pd.decision_ts) - julianday(?)) <= 2
                   ORDER BY dt_diff ASC
                   LIMIT 1""",
                (p["target_opened_at"], p["symbol"],
                 p["source_strike"], p["source_expiry"],
                 p["target_strike"], p["target_expiry"],
                 p["target_opened_at"]),
            ).fetchone()

            if best_roll:
                conn.execute(
                    """UPDATE policy_actions
                       SET rollout_source = 'realized',
                           matched_trade_id = ?,
                           matched_trade_id_source = ?
                       WHERE id = ?""",
                    (p["target_trade_id"], p["source_trade_id"],
                     best_roll["id"]),
                )
                consumed_trade_ids.add(p["source_trade_id"])
                consumed_trade_ids.add(p["target_trade_id"])
                promoted += 1

        # ── Pass 2: OPEN / ADD (by opened_at) ──
        rows = conn.execute(
            """SELECT t.id AS trade_id, t.symbol, t.strike, t.expiry,
                      t.opened_at, t.closed_at, t.status
               FROM trades t"""
        ).fetchall()

        for trade in rows:
            t = dict(trade)
            if t["trade_id"] in consumed_trade_ids:
                continue

            best = conn.execute(
                """SELECT pa.id, pa.action_type,
                          ABS(julianday(pd.decision_ts) - julianday(?)) AS dt_diff
                   FROM policy_actions pa
                   JOIN policy_decisions pd ON pa.decision_id = pd.decision_id
                   WHERE pa.rollout_source = 'recommendation'
                     AND pa.chosen = 1
                     AND pa.action_type IN ('OPEN', 'ADD')
                     AND pa.symbol = ?
                     AND pa.target_strike = ?
                     AND pa.target_expiry = ?
                     AND ABS(julianday(pd.decision_ts) - julianday(?)) <= 2
                   ORDER BY dt_diff ASC
                   LIMIT 1""",
                (t["opened_at"], t["symbol"], t["strike"], t["expiry"],
                 t["opened_at"]),
            ).fetchone()

            if best:
                conn.execute(
                    """UPDATE policy_actions
                       SET rollout_source = 'realized', matched_trade_id = ?
                       WHERE id = ?""",
                    (t["trade_id"], best["id"]),
                )
                consumed_trade_ids.add(t["trade_id"])
                promoted += 1
                continue

            # ── Pass 3: CLOSE / LET_EXPIRE (by closed_at) ──
            if t["status"] in ("expired", "closed", "assigned"):
                if t["trade_id"] in consumed_trade_ids:
                    continue
                close_ts = t.get("closed_at") or t["opened_at"]

                best_close = conn.execute(
                    """SELECT pa.id, pa.action_type,
                              ABS(julianday(pd.decision_ts) - julianday(?)) AS dt_diff
                       FROM policy_actions pa
                       JOIN policy_decisions pd ON pa.decision_id = pd.decision_id
                       WHERE pa.rollout_source = 'recommendation'
                         AND pa.chosen = 1
                         AND pa.action_type IN ('CLOSE', 'LET_EXPIRE')
                         AND pa.symbol = ?
                         AND pa.source_strike = ?
                         AND pa.source_expiry = ?
                         AND ABS(julianday(pd.decision_ts) - julianday(?)) <= 2
                       ORDER BY dt_diff ASC
                       LIMIT 1""",
                    (close_ts, t["symbol"], t["strike"], t["expiry"], close_ts),
                ).fetchone()

                if best_close:
                    conn.execute(
                        """UPDATE policy_actions
                           SET rollout_source = 'realized', matched_trade_id = ?
                           WHERE id = ?""",
                        (t["trade_id"], best_close["id"]),
                    )
                    consumed_trade_ids.add(t["trade_id"])
                    promoted += 1

        conn.commit()
        return promoted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reward computation (3 layers)
# ---------------------------------------------------------------------------

def compute_reward(
    action: dict,
    entry_premium: float,
    current_option_price: float | None = None,
    terminal_pnl: float | None = None,
    terminal_status: str | None = None,
    is_counterfactual: bool = False,
) -> dict:
    """Compute 3-layer reward for a policy action.

    Returns dict with next_day, interim, terminal, total, confidence.
    All rewards are normalized to per-contract scale.
    """
    contracts = action.get("target_contracts") or 1
    multiplier = contracts * 100

    # ── Next-day reward (MTM change from entry) ────────────────
    next_day = 0.0
    if current_option_price is not None and entry_premium > 0:
        # For short call: profit when option price drops
        next_day = (entry_premium - current_option_price) * multiplier

    # ── Interim reward (running P&L) ──────────────────────────
    interim = 0.0
    if current_option_price is not None and entry_premium > 0:
        capture_pct = (entry_premium - current_option_price) / entry_premium
        interim = capture_pct * entry_premium * multiplier
        # Penalize high assignment risk (proxy: option near or ITM)
        if current_option_price > entry_premium:
            interim -= (current_option_price - entry_premium) * multiplier * 0.5

    # ── Terminal reward (final outcome) ───────────────────────
    terminal = 0.0
    if terminal_pnl is not None:
        terminal = terminal_pnl
    elif terminal_status == "expired":
        terminal = entry_premium * multiplier  # kept full premium
    elif terminal_status == "assigned":
        terminal = entry_premium * multiplier * 0.3  # premium kept but assignment cost

    # Friction adjustment for counterfactual
    if is_counterfactual:
        # Conservative: assume worse execution
        friction = entry_premium * multiplier * 0.03  # ~3% slippage
        next_day -= friction
        interim -= friction
        terminal -= friction

    # Confidence
    if terminal_pnl is not None:
        confidence = "high"
    elif is_counterfactual:
        confidence = "low"
    else:
        confidence = "medium"

    total = 0.2 * next_day + 0.3 * interim + 0.5 * terminal

    return {
        "next_day": round(next_day, 2),
        "interim": round(interim, 2),
        "terminal": round(terminal, 2),
        "total": round(total, 2),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Daily backfill
# ---------------------------------------------------------------------------

def run_daily_backfill(config: dict | None = None, vix: float | None = None) -> int:
    """Update mark-to-market for all open (non-terminal) policy actions.

    Called during each /scan to keep the trajectory data fresh.
    Returns number of actions updated.
    """
    from .db import get_open_policy_actions, record_action_update

    if config is None:
        config = load_config()

    open_actions = get_open_policy_actions()
    updated = 0
    today = dt.date.today()

    for action in open_actions:
        symbol = action["symbol"]
        strike = action.get("target_strike") or action.get("source_strike")
        expiry_str = action.get("target_expiry") or action.get("source_expiry")

        if not strike or not expiry_str:
            continue

        try:
            stock = fetch_stock(symbol)
            price = stock.price
            expiry = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry - today).days

            if dte < 0:
                continue  # Past expiry → handled by terminal backfill

            T = max(dte, 0) / 365.0
            iv_est = 0.30
            option_price = bs_call_price(price, strike, T, RISK_FREE_RATE, iv_est)
            greeks = black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv_est, "call")

            moneyness = (price - strike) / strike * 100
            entry = action.get("target_premium") or action.get("source_entry_premium") or 0
            pnl_pct = ((entry - option_price) / entry * 100) if entry > 0 else 0

            # Current advice
            events = fetch_events(symbol)
            is_itm = price >= strike
            earnings_before = (
                events.days_to_earnings is not None
                and 0 <= events.days_to_earnings <= dte
            )
            advice = advise_position(
                entry_premium=entry,
                current_premium=option_price,
                dte=dte,
                is_itm=is_itm,
                earnings_before_expiry=earnings_before,
                config=config,
            )

            record_action_update(
                action_id=action["id"],
                stock_price=round(price, 2),
                option_price=round(option_price, 4),
                delta_now=round(greeks.delta, 4),
                moneyness_pct=round(moneyness, 1),
                pnl_pct=round(pnl_pct, 1),
                current_advice=advice.action,
                advice_urgency=advice.urgency,
                vix=vix,
                days_to_expiry=dte,
            )
            updated += 1
        except Exception:
            continue

    return updated


def run_terminal_backfill() -> int:
    """Backfill terminal outcomes for policy actions.

    Three paths:
    1. Realized: ONLY for actions with rollout_source="realized" — match
       against trades table for trades the user actually executed.
    2. Counterfactual: for all other past-expiry actions, estimate outcome
       using the LAST daily update's stock price (not today's price, which
       would introduce future information leakage).
    3. Auto-promote: first run auto_mark_executed_from_trades() to detect
       which recommendations were actually traded.
    """
    from .db import get_open_policy_actions, finalize_policy_action

    # Step 0: Auto-promote recommendations → realized where trades match
    try:
        auto_mark_executed_from_trades()
    except Exception:
        pass

    open_actions = get_open_policy_actions()
    finalized = 0
    today = dt.date.today()

    for action in open_actions:
        strike = action.get("target_strike") or action.get("source_strike")
        expiry_str = action.get("target_expiry") or action.get("source_expiry")
        entry_premium = action.get("target_premium") or action.get("source_entry_premium") or 0
        rollout_source = action.get("rollout_source", "recommendation")

        if not strike or not expiry_str:
            continue

        # ── Path 1: Realized — use persisted matched_trade_id for exact lookup ──
        if rollout_source == "realized":
            action_type = action.get("action_type", "")
            trade_id = action.get("matched_trade_id")
            trade_id_source = action.get("matched_trade_id_source")

            if action_type == "ROLL":
                # Lookup both legs by their persisted trade IDs (direct DB query)
                source_trade = _get_trade_by_id(trade_id_source)
                target_trade = _get_trade_by_id(trade_id)

                if source_trade and source_trade.get("status") != "open":
                    source_pnl = source_trade.get("pnl") or 0
                    if target_trade and target_trade.get("status") != "open":
                        total_pnl = source_pnl + (target_trade.get("pnl") or 0)
                        confidence = "high"
                        status = f"rolled_{target_trade['status']}"
                    else:
                        total_pnl = source_pnl
                        confidence = "medium"
                        status = "rolled_open"

                    reward_data = compute_reward(
                        action=action,
                        entry_premium=entry_premium,
                        terminal_pnl=total_pnl,
                        terminal_status=status,
                        is_counterfactual=False,
                    )
                    finalize_policy_action(
                        action_id=action["id"],
                        terminal_status=status,
                        terminal_pnl=total_pnl,
                        reward=reward_data["total"],
                        reward_confidence=confidence,
                    )
                    finalized += 1
                continue

            # OPEN/ADD/CLOSE/LET_EXPIRE: single trade lookup by ID (direct DB query)
            matched_trade = _get_trade_by_id(trade_id)

            if matched_trade and matched_trade.get("status") != "open":
                reward_data = compute_reward(
                    action=action,
                    entry_premium=entry_premium,
                    terminal_pnl=matched_trade.get("pnl"),
                    terminal_status=matched_trade["status"],
                    is_counterfactual=False,
                )
                finalize_policy_action(
                    action_id=action["id"],
                    terminal_status=matched_trade["status"],
                    terminal_pnl=matched_trade.get("pnl"),
                    reward=reward_data["total"],
                    reward_confidence="high",
                )
                finalized += 1
            continue  # realized actions only finalize from trades

        # ── Path 2: Counterfactual — recommendations and non-executed actions ──
        try:
            exp_date = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        if exp_date >= today:
            continue  # not yet expired

        last_update_price = _get_last_update_price(action["id"])
        action_type = action.get("action_type", "")

        if action_type == "ROLL" and last_update_price is not None:
            # Counterfactual ROLL: both legs, matching realized semantics.
            #
            # Source leg: bought back around the roll date (decision time).
            #   Use the FIRST daily update price — closest to when the roll
            #   would have been executed.
            # Target leg: new short expires at target expiry.
            #   Use the LAST daily update price — closest to target expiry.
            source_entry = action.get("source_entry_premium") or 0
            target_premium = action.get("target_premium") or 0
            target_strike = action.get("target_strike") or strike
            source_strike = action.get("source_strike") or strike
            contracts = action.get("target_contracts") or 1
            mult = contracts * 100

            # Source leg price: first update (near roll/decision date)
            roll_date_price = _get_first_update_price(action["id"])
            if roll_date_price is None:
                roll_date_price = last_update_price  # fallback

            # Source leg PnL: short at source_entry, buyback at intrinsic
            source_moneyness = roll_date_price - source_strike
            source_buyback = max(0, source_moneyness)
            source_pnl = (source_entry - source_buyback) * mult

            # Target leg: last update price (near target expiry)
            target_assigned = last_update_price > target_strike
            if target_assigned:
                target_pnl = (target_premium - (last_update_price - target_strike)) * mult
            else:
                target_pnl = target_premium * mult

            cf_pnl = source_pnl + target_pnl
            cf_status = f"cf_rolled_{'assigned' if target_assigned else 'expired'}"

            reward_data = compute_reward(
                action=action,
                entry_premium=target_premium,
                terminal_pnl=cf_pnl,
                terminal_status=cf_status,
                is_counterfactual=True,
            )
            finalize_policy_action(
                action_id=action["id"],
                terminal_status=cf_status,
                terminal_pnl=cf_pnl,
                reward=reward_data["total"],
                reward_confidence="low",
            )
            finalized += 1

        elif action_type in ("OPEN", "ADD") and last_update_price is not None:
            # Single-leg counterfactual
            assigned = last_update_price > strike
            contracts = action.get("target_contracts") or 1
            if assigned:
                cf_pnl = (entry_premium - (last_update_price - strike)) * contracts * 100
            else:
                cf_pnl = entry_premium * contracts * 100

            reward_data = compute_reward(
                action=action,
                entry_premium=entry_premium,
                terminal_pnl=cf_pnl,
                terminal_status="assigned" if assigned else "expired",
                is_counterfactual=True,
            )
            finalize_policy_action(
                action_id=action["id"],
                terminal_status=f"cf_{'assigned' if assigned else 'expired'}",
                terminal_pnl=cf_pnl,
                reward=reward_data["total"],
                reward_confidence="low",
            )
            finalized += 1

        elif action_type in ("OPEN", "ADD", "ROLL"):
            # No daily updates — can't price counterfactual reliably
            finalize_policy_action(
                action_id=action["id"],
                terminal_status="cf_unknown",
                reward_confidence="low",
            )
            finalized += 1

        else:
            # HOLD/CLOSE/SKIP past expiry
            finalize_policy_action(
                action_id=action["id"],
                terminal_status="cf_expired",
                reward_confidence="low",
            )
            finalized += 1

    return finalized


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_last_update_price(action_id: int) -> float | None:
    """Get the stock price from the most recent daily update for an action.

    Used for counterfactual terminal pricing — avoids future info leakage
    by using the last known price before/at expiry, not today's price.
    """
    from .db import _connect
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT stock_price FROM policy_action_updates
               WHERE action_id = ?
               ORDER BY update_ts DESC LIMIT 1""",
            (action_id,),
        ).fetchone()
        return row["stock_price"] if row else None
    finally:
        conn.close()


def _get_first_update_price(action_id: int) -> float | None:
    """Get the stock price from the earliest daily update for an action.

    Used for counterfactual ROLL source-leg pricing — the source leg
    is bought back around the roll date, which is closest to the first
    daily update (near decision time), not the last (near target expiry).
    """
    from .db import _connect
    conn = _connect()
    try:
        row = conn.execute(
            """SELECT stock_price FROM policy_action_updates
               WHERE action_id = ?
               ORDER BY update_ts ASC LIMIT 1""",
            (action_id,),
        ).fetchone()
        return row["stock_price"] if row else None
    finally:
        conn.close()


def _get_trade_by_id(trade_id: int | None) -> dict | None:
    """Fetch a single trade row by primary key. Direct DB lookup — no limit bias."""
    if trade_id is None:
        return None
    from .db import _connect
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM trades WHERE id = ?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def _features_to_dict(f) -> dict:
    """Convert CandidateFeatures to a serializable dict."""
    return {
        "bid": f.bid, "ask": f.ask, "premium": f.premium,
        "delta": f.delta, "theta": f.theta, "implied_vol": f.implied_vol,
        "dte": f.dte, "otm_pct": f.otm_pct, "spread_pct": f.spread_pct,
        "annualized_yield": f.annualized_yield,
        "open_interest": f.open_interest, "volume": f.volume,
        "atr_distance": f.atr_distance, "earnings_gap": f.earnings_gap,
        "iv_rank": f.iv_rank, "cost_basis": f.cost_basis,
    }


def _market_snap(market: MarketContext, iv_rank: float, stock_price: float) -> dict:
    return {
        "vix": market.vix,
        "regime": market.regime,
        "iv_rank": iv_rank,
        "stock_price": stock_price,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_action_slate(decision: PolicyDecision) -> str:
    """Format the action slate for display / briefing."""
    lines = [
        f"# Action Slate — {decision.decision_ts[:10]}",
        f"Regime: {decision.market_regime} | VIX: {decision.vix:.1f} | "
        f"SPY: ${decision.spy_price:.2f}" if decision.vix and decision.spy_price else "",
        f"Weekly target: ${decision.weekly_target:,.0f} | "
        f"Collected: ${decision.weekly_premium_so_far:,.0f}",
        "",
    ]

    # Group by type
    by_type: dict[str, list[PolicyAction]] = {}
    for a in decision.actions:
        by_type.setdefault(a.action_type, []).append(a)

    # Position management first
    for action_type in ["CLOSE", "ROLL", "HOLD", "LET_EXPIRE"]:
        group = by_type.get(action_type, [])
        if not group:
            continue
        lines.append(f"## {action_type} ({len(group)})")
        for a in sorted(group, key=lambda x: -x.score):
            target = ""
            if a.target_strike:
                target = f" → {a.target_expiry} ${a.target_strike:.0f}"
            source = ""
            if a.source_strike:
                source = f"{a.source_expiry} ${a.source_strike:.0f}"
            lines.append(
                f"  [{a.urgency}] {a.symbol} {source}{target} "
                f"| score={a.score:.3f} | {a.reason}"
            )
        lines.append("")

    # New positions
    for action_type in ["OPEN", "ADD"]:
        group = by_type.get(action_type, [])
        if not group:
            continue
        lines.append(f"## {action_type} ({len(group)})")
        for a in sorted(group, key=lambda x: -x.score)[:10]:
            lines.append(
                f"  {a.symbol} {a.target_expiry} ${a.target_strike:.0f} "
                f"Δ{a.target_delta:.2f} ${a.target_premium:.2f} "
                f"| score={a.score:.3f}"
            )
        lines.append("")

    # Skips
    skips = by_type.get("SKIP", [])
    if skips:
        lines.append(f"## SKIP ({len(skips)})")
        for a in skips:
            lines.append(f"  {a.symbol}: {a.reason}")
        lines.append("")

    lines.append(f"_Total actions: {len(decision.actions)} | "
                 f"Shares available: {decision.shares_available}_")
    return "\n".join(lines)
