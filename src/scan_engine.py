"""Deterministic covered-call selection engine.

Pipeline:  fetch raw → build features → hard filter → score → rank
Each stage is a pure function so the same logic can later drive backtests.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .data.fetcher import (
    OptionRow,
    IVStats,
    StockSnapshot,
    EventInfo,
    SymbolBriefing,
    MarketContext,
    fetch_market_context,
    fetch_symbol_briefing,
)
from .data.events_calendar import assess_event_risk, EventRisk


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class CandidateFeatures:
    """Uniform feature vector for one option contract."""
    symbol: str
    expiry: str
    strike: float
    bid: float
    ask: float
    premium: float              # bid (or last if off-hours)
    delta: float
    theta: float
    implied_vol: float
    dte: int
    otm_pct: float
    spread_pct: float
    annualized_yield: float
    open_interest: int
    volume: int
    atr_distance: float | None  # (strike - price) / ATR, None if ATR unavailable
    earnings_gap: int | None    # days between earnings and expiry (negative = earnings after)
    iv_rank: float
    stock_price: float
    cost_basis: float
    allow_assignment: bool
    off_hours: bool


@dataclass
class CandidateDecision:
    """Result of hard-filter + scoring for one contract."""
    features: CandidateFeatures
    passed: bool
    reject_reasons: list[str]
    score: float
    breakdown: dict[str, float]  # income, assignment_risk, execution_quality, event_risk
    flags: list[str]


@dataclass
class SymbolScanResult:
    """All scan output for one symbol."""
    symbol: str
    briefing: SymbolBriefing             # raw data for report rendering
    candidates: list[CandidateDecision]  # passed hard filters, sorted by score desc
    rejected_count: int                  # how many dropped by hard filters
    symbol_flags: list[str]              # symbol-level context (iv_rank, news, etc.)


@dataclass
class PortfolioScanResult:
    """Full scan across all portfolio symbols."""
    market: MarketContext
    symbols: dict[str, SymbolScanResult]


# ---------------------------------------------------------------------------
# Build features
# ---------------------------------------------------------------------------

def build_features(
    opt: OptionRow,
    stock: StockSnapshot,
    iv_stats: IVStats,
    events: EventInfo,
    position: dict,
) -> CandidateFeatures:
    """Convert an OptionRow + context into a flat feature vector."""
    atr = stock.technicals.atr_14 if stock.technicals else None
    atr_distance = (opt.strike - stock.price) / atr if atr and atr > 0 else None

    earnings_gap = None
    if events.days_to_earnings is not None:
        earnings_gap = events.days_to_earnings - opt.dte  # negative = earnings falls within window

    return CandidateFeatures(
        symbol=stock.symbol,
        expiry=opt.expiry,
        strike=opt.strike,
        bid=opt.bid,
        ask=opt.ask,
        premium=opt.last if opt.off_hours else opt.bid,
        delta=abs(opt.greeks.delta),
        theta=abs(opt.greeks.theta),
        implied_vol=opt.implied_vol,
        dte=opt.dte,
        otm_pct=opt.otm_pct,
        spread_pct=opt.spread_pct,
        annualized_yield=opt.annualized_yield,
        open_interest=opt.open_interest,
        volume=opt.volume,
        atr_distance=round(atr_distance, 2) if atr_distance is not None else None,
        earnings_gap=earnings_gap,
        iv_rank=iv_stats.iv_rank,
        stock_price=stock.price,
        cost_basis=position.get("cost_basis", 0),
        allow_assignment=position.get("allow_assignment", False),
        off_hours=opt.off_hours,
    )


# ---------------------------------------------------------------------------
# Hard filters (deterministic, non-negotiable)
# ---------------------------------------------------------------------------

def apply_hard_filters(feat: CandidateFeatures, config: dict) -> list[str]:
    """Return rejection reasons. Empty list = candidate passed."""
    strat = config.get("strategy", {})
    reasons: list[str] = []

    # Min premium
    min_prem = strat.get("min_premium", 0.25)
    if feat.premium < min_prem:
        reasons.append(f"premium ${feat.premium:.2f} < min ${min_prem:.2f}")

    # Min OI (skip during off-hours when OI may be stale)
    min_oi = strat.get("min_open_interest", 50)
    liquidity = max(feat.open_interest, feat.volume) if feat.off_hours else feat.open_interest
    if liquidity < min_oi and not feat.off_hours:
        reasons.append(f"OI {feat.open_interest} < min {min_oi}")

    # Max spread (skip during off-hours)
    max_spread = strat.get("max_spread_pct", 20)
    if not feat.off_hours and feat.spread_pct > max_spread:
        reasons.append(f"spread {feat.spread_pct:.1f}% > max {max_spread}%")

    # Earnings blackout: if earnings falls within the option's life
    blackout_days = strat.get("blackout_earnings_days", 7)
    if feat.earnings_gap is not None and feat.earnings_gap < blackout_days:
        # earnings_gap < blackout means earnings is too close to (or before) expiry
        reasons.append(f"earnings blackout ({feat.earnings_gap}d gap, need {blackout_days}d)")

    # ATR floor: strike too close to current price
    if feat.atr_distance is not None and feat.atr_distance < 0.5:
        reasons.append(f"ATR distance {feat.atr_distance:.1f} < 0.5 (assignment risk)")

    # Cost basis: if assignment not allowed, strike must be above cost basis
    if not feat.allow_assignment and feat.cost_basis > 0:
        if feat.strike <= feat.cost_basis:
            reasons.append(f"strike ${feat.strike:.0f} <= cost basis ${feat.cost_basis:.0f}")

    return reasons


# ---------------------------------------------------------------------------
# Scoring (4 categories)
# ---------------------------------------------------------------------------

def _normalize(value: float, lo: float, hi: float) -> float:
    """Clamp value into [0, 1] between lo and hi."""
    if hi <= lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def score_candidate(
    feat: CandidateFeatures,
    config: dict,
    event_risk: EventRisk,
    all_yields: list[float],
    all_thetas: list[float],
    all_ois: list[int],
    delta_range: tuple[float, float] = (0.15, 0.25),
) -> tuple[float, dict[str, float], list[str]]:
    """Score a candidate across 4 categories. Returns (score, breakdown, flags).

    all_yields / all_thetas / all_ois are from the full candidate pool for
    this symbol so we can normalize relative to peers.
    """
    strat = config.get("strategy", {})
    weights = strat.get("scoring_weights", {})
    w_income = weights.get("income", 0.35)
    w_assign = weights.get("assignment_risk", 0.30)
    w_exec = weights.get("execution_quality", 0.20)
    w_event = weights.get("event_risk", 0.15)

    # Normalize weights
    total_w = w_income + w_assign + w_exec + w_event
    if total_w > 0:
        w_income /= total_w
        w_assign /= total_w
        w_exec /= total_w
        w_event /= total_w

    # Regime-aware delta target: use the midpoint of the active regime's range
    # (e.g. conservative 0.08-0.15 → target 0.115, aggressive 0.25-0.35 → 0.30)
    delta_lo, delta_hi = delta_range
    target_delta = (delta_lo + delta_hi) / 2
    preferred_dte = strat.get("preferred_dte", 10)
    flags: list[str] = []

    # ── Income ──────────────────────────────────────────────────
    max_yield = max(all_yields) if all_yields else 1
    min_yield = min(all_yields) if all_yields else 0
    yield_score = _normalize(feat.annualized_yield, min_yield, max_yield)

    max_theta = max(all_thetas) if all_thetas else 1
    theta_score = feat.theta / max_theta if max_theta > 0 else 0.5

    income = 0.6 * yield_score + 0.4 * theta_score

    # ── Assignment Risk (higher = safer) ────────────────────────
    # Delta proximity: closer to regime target midpoint = better
    # Score falls off as delta moves away from the regime's ideal range
    delta_dev = abs(feat.delta - target_delta) / target_delta if target_delta > 0 else 0
    delta_score = max(0.0, 1.0 - delta_dev)

    # ATR distance: more ATRs away = safer
    if feat.atr_distance is not None:
        atr_score = _normalize(feat.atr_distance, 0.5, 2.5)
    else:
        atr_score = 0.5  # neutral when unavailable

    # OTM%: further OTM = safer
    otm_score = _normalize(feat.otm_pct, 1.0, 10.0)

    # DTE: closer to preferred = better
    dte_dev = abs(feat.dte - preferred_dte) / preferred_dte if preferred_dte > 0 else 0
    dte_score = max(0.0, 1.0 - dte_dev)

    assignment_risk = 0.35 * delta_score + 0.25 * atr_score + 0.20 * otm_score + 0.20 * dte_score

    # Flag if delta is in target range
    if delta_lo <= feat.delta <= delta_hi:
        flags.append(">>>")

    # ── Execution Quality ───────────────────────────────────────
    # Spread tightness (lower spread = better)
    if feat.off_hours:
        spread_score = 0.5  # unreliable during off-hours
    else:
        spread_score = max(0.0, 1.0 - feat.spread_pct / 20.0)

    # OI depth (normalized within peer pool)
    max_oi = max(all_ois) if all_ois else 1
    oi_score = feat.open_interest / max_oi if max_oi > 0 else 0.5

    execution_quality = 0.5 * spread_score + 0.5 * oi_score

    if feat.open_interest < strat.get("min_open_interest", 50) * 2:
        flags.append("LOW_OI")
    if not feat.off_hours and feat.spread_pct > 10:
        flags.append("WIDE_SPREAD")

    # ── Event Risk (higher = less risky) ────────────────────────
    event_score = 1.0 - event_risk.risk_score

    if event_risk.risk_level == "HIGH":
        flags.append("EVENT_RISK")
    if feat.earnings_gap is not None and feat.earnings_gap < 14:
        flags.append("EARNINGS_NEAR")

    # ── Composite ───────────────────────────────────────────────
    composite = (
        w_income * income
        + w_assign * assignment_risk
        + w_exec * execution_quality
        + w_event * event_score
    )

    breakdown = {
        "income": round(income, 2),
        "assignment_risk": round(assignment_risk, 2),
        "execution_quality": round(execution_quality, 2),
        "event_risk": round(event_score, 2),
    }

    return round(composite, 4), breakdown, flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _progress(msg: str):
    print(f"  ⏳ {msg}", file=sys.stderr, flush=True)


def scan_symbol(
    symbol: str,
    config: dict,
    market: MarketContext,
    quick: bool = False,
) -> SymbolScanResult:
    """Full scan pipeline for one symbol: fetch → features → filter → score → rank."""
    from .config import get_position, get_delta_range

    briefing = fetch_symbol_briefing(symbol, config, quick=quick)
    position = get_position(config, symbol)
    strat = config.get("strategy", {})
    delta_range = get_delta_range(config, market.regime)

    # Build features for all call candidates
    features_list = [
        build_features(opt, briefing.stock, briefing.iv_stats, briefing.events, position)
        for opt in briefing.call_chains
    ]

    # Hard filter
    passed_features: list[CandidateFeatures] = []
    rejected_count = 0
    decisions: list[CandidateDecision] = []

    for feat in features_list:
        reasons = apply_hard_filters(feat, config)
        if reasons:
            rejected_count += 1
            continue
        passed_features.append(feat)

    # Compute peer stats for normalization (from passed candidates only)
    all_yields = [f.annualized_yield for f in passed_features]
    all_thetas = [f.theta for f in passed_features]
    all_ois = [f.open_interest for f in passed_features]

    # News sentiment for event risk context
    news_sentiment = None
    if briefing.news:
        from .data.news import score_news_sentiment
        news_sentiment = score_news_sentiment(briefing.news)

    # Score each passed candidate
    earnings_buffer = strat.get("blackout_earnings_days", 7)
    for feat in passed_features:
        event_risk = assess_event_risk(
            symbol, feat.expiry,
            news=briefing.news,
            earnings_dte=briefing.events.days_to_earnings,
            earnings_buffer=earnings_buffer,
        )
        score, breakdown, flags = score_candidate(
            feat, config, event_risk, all_yields, all_thetas, all_ois,
            delta_range=delta_range,
        )
        decisions.append(CandidateDecision(
            features=feat,
            passed=True,
            reject_reasons=[],
            score=score,
            breakdown=breakdown,
            flags=flags,
        ))

    # Sort by score descending
    decisions.sort(key=lambda d: d.score, reverse=True)

    # Symbol-level flags (for context, not scoring)
    sym_flags: list[str] = []
    iv_r = briefing.iv_stats.iv_rank
    if iv_r >= 50:
        sym_flags.append(f"IV_RANK_HIGH ({iv_r:.0f}%)")
    elif iv_r <= 20:
        sym_flags.append(f"IV_RANK_LOW ({iv_r:.0f}%)")
    if news_sentiment:
        sym_flags.append(f"NEWS_{news_sentiment['label']}")
    if briefing.events.days_to_earnings is not None and briefing.events.days_to_earnings <= 14:
        sym_flags.append(f"EARNINGS_{briefing.events.days_to_earnings}d")

    return SymbolScanResult(
        symbol=symbol,
        briefing=briefing,
        candidates=decisions,
        rejected_count=rejected_count,
        symbol_flags=sym_flags,
    )


def scan_portfolio(config: dict, quick: bool = False) -> PortfolioScanResult:
    """Scan all portfolio symbols in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .config import get_symbols

    _progress("Market context (VIX, SPY)...")
    market = fetch_market_context(config)
    symbols = get_symbols(config)

    results: dict[str, SymbolScanResult] = {}

    def _scan_one(sym: str) -> tuple[str, SymbolScanResult]:
        _progress(f"{sym}...")
        result = scan_symbol(sym, config, market, quick=quick)
        _progress(f"{sym} done — {len(result.candidates)} candidates, {result.rejected_count} filtered.")
        return sym, result

    with ThreadPoolExecutor(max_workers=max(len(symbols), 1)) as pool:
        futures = {pool.submit(_scan_one, s): s for s in symbols}
        for f in as_completed(futures):
            sym, result = f.result()
            results[sym] = result

    return PortfolioScanResult(market=market, symbols=results)
