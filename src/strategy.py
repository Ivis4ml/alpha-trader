"""5-factor scoring model for covered call candidate selection.

Factors:
  1. Delta proximity — how close to target delta
  2. Premium richness — annualized yield relative to peers
  3. Theta decay — daily time decay efficiency
  4. IV Rank — selling when IV is rich
  5. DTE proximity — how close to preferred DTE

All weights are configurable and auto-tunable by the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data.fetcher import OptionRow, IVStats


@dataclass
class ScoredCandidate:
    """An option candidate with its component scores."""
    option: OptionRow
    symbol: str
    score: float
    delta_score: float
    premium_score: float
    theta_score: float
    iv_score: float
    dte_score: float
    flags: list[str]  # e.g. ["EARNINGS_NEAR", "LOW_OI", "WIDE_SPREAD"]


def score_candidates(
    candidates: list[OptionRow],
    symbol: str,
    iv_stats: IVStats,
    config: dict,
    earnings_dte: int | None = None,
) -> list[ScoredCandidate]:
    """Score and rank option candidates using the 5-factor model.

    Returns candidates sorted by score descending.
    """
    strat = config.get("strategy", {})
    weights = strat.get("scoring_weights", {})
    w_delta = weights.get("delta", 0.30)
    w_premium = weights.get("premium", 0.25)
    w_theta = weights.get("theta", 0.20)
    w_iv = weights.get("iv", 0.15)
    w_dte = weights.get("dte", 0.10)

    target_delta = strat.get("target_delta", 0.20)
    preferred_dte = strat.get("preferred_dte", 10)
    profit_take_pct = strat.get("profit_take_pct", 50)
    min_oi = strat.get("min_open_interest", 50)
    max_spread = strat.get("max_spread_pct", 20)
    earnings_buffer = strat.get("earnings_buffer_days", 7)

    if not candidates:
        return []

    # Normalize premium yields for relative comparison
    yields = [c.annualized_yield for c in candidates]
    max_yield = max(yields) if yields else 1
    min_yield = min(yields) if yields else 0
    yield_range = max_yield - min_yield if max_yield != min_yield else 1

    # Normalize theta values
    thetas = [abs(c.greeks.theta) for c in candidates]
    max_theta = max(thetas) if thetas else 1

    scored: list[ScoredCandidate] = []

    for c in candidates:
        delta = abs(c.greeks.delta)
        flags = []

        # ── Hard filters (disqualify) ────────────────────────────
        if c.open_interest < min_oi and not c.off_hours:
            flags.append("LOW_OI")
        if c.spread_pct > max_spread and not c.off_hours:
            flags.append("WIDE_SPREAD")
        if earnings_dte is not None and c.dte >= earnings_dte - earnings_buffer:
            flags.append("EARNINGS_NEAR")

        # ── Factor 1: Delta proximity (0-1, 1 = exactly on target) ──
        delta_dev = abs(delta - target_delta) / target_delta if target_delta > 0 else 0
        delta_score = max(0, 1 - delta_dev)

        # ── Factor 2: Premium richness (0-1, relative to peers) ──
        premium_score = (c.annualized_yield - min_yield) / yield_range if yield_range > 0 else 0.5

        # ── Factor 3: Theta decay efficiency (0-1) ──
        theta_score = abs(c.greeks.theta) / max_theta if max_theta > 0 else 0

        # ── Factor 4: IV Rank (0-1, higher = better for selling) ──
        iv_score = iv_stats.iv_rank / 100 if iv_stats.iv_rank else 0.5

        # ── Factor 5: DTE proximity (0-1, 1 = exactly preferred) ──
        dte_dev = abs(c.dte - preferred_dte) / preferred_dte if preferred_dte > 0 else 0
        dte_score = max(0, 1 - dte_dev)

        # ── Composite score ──────────────────────────────────────
        score = (
            w_delta * delta_score +
            w_premium * premium_score +
            w_theta * theta_score +
            w_iv * iv_score +
            w_dte * dte_score
        )

        # Penalty for flagged issues
        if "EARNINGS_NEAR" in flags:
            score *= 0.3  # Heavy penalty
        if "LOW_OI" in flags:
            score *= 0.7
        if "WIDE_SPREAD" in flags:
            score *= 0.8

        scored.append(ScoredCandidate(
            option=c,
            symbol=symbol,
            score=round(score, 4),
            delta_score=round(delta_score, 3),
            premium_score=round(premium_score, 3),
            theta_score=round(theta_score, 3),
            iv_score=round(iv_score, 3),
            dte_score=round(dte_score, 3),
            flags=flags,
        ))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ── Position management rules ────────────────────────────────────────────────

@dataclass
class PositionAdvice:
    action: str        # HOLD, CLOSE_PROFIT, CLOSE_STOP, LET_EXPIRE, ROLL, CLOSE_EARNINGS
    reason: str
    urgency: str       # LOW, MEDIUM, HIGH


def advise_position(
    entry_premium: float,
    current_premium: float,
    dte: int,
    is_itm: bool,
    earnings_before_expiry: bool,
    config: dict,
) -> PositionAdvice:
    """Generate hold/close/roll advice for an open short call position."""
    strat = config.get("strategy", {})
    profit_take = strat.get("profit_take_pct", 50) / 100
    max_loss_mult = strat.get("max_loss_multiple", 2.0)
    roll_dte = strat.get("roll_when_dte", 5)

    captured = (entry_premium - current_premium) / entry_premium if entry_premium > 0 else 0

    # Priority order (first match wins)

    # 1. Earnings approaching
    if earnings_before_expiry and strat.get("avoid_earnings", True):
        return PositionAdvice("CLOSE_EARNINGS", "Earnings before expiry — close to avoid gap risk", "HIGH")

    # 2. Stop loss
    if current_premium >= entry_premium * max_loss_mult:
        return PositionAdvice("CLOSE_STOP",
            f"Stop loss triggered — current premium {current_premium:.2f} is "
            f"{current_premium/entry_premium:.1f}x entry ({entry_premium:.2f})", "HIGH")

    # 3. ITM + low DTE → roll
    if is_itm and dte <= roll_dte:
        return PositionAdvice("ROLL", f"ITM with {dte} DTE — high assignment risk, roll out", "HIGH")

    # 4. Profit take
    if captured >= profit_take:
        return PositionAdvice("CLOSE_PROFIT",
            f"Captured {captured:.0%} profit (target: {profit_take:.0%})", "MEDIUM")

    # 5. Low DTE, OTM, mostly decayed → let expire
    if dte <= 3 and not is_itm and captured > 0.75:
        return PositionAdvice("LET_EXPIRE",
            f"{captured:.0%} captured with {dte} DTE — let expire", "LOW")

    # 6. Low DTE → roll even if OTM (to keep rolling the wheel)
    if dte <= roll_dte and not is_itm:
        return PositionAdvice("ROLL",
            f"DTE={dte} approaching — consider rolling to next cycle", "MEDIUM")

    # 7. Default: hold
    return PositionAdvice("HOLD",
        f"On track — {captured:.0%} captured, {dte} DTE remaining", "LOW")
