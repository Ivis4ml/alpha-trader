"""Position management rules for open short calls.

Scoring / candidate selection logic has moved to scan_engine.py.
This module only handles hold / close / roll advice for existing positions.
"""

from __future__ import annotations

from dataclasses import dataclass


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

    if earnings_before_expiry and strat.get("avoid_earnings", True):
        return PositionAdvice("CLOSE_EARNINGS", "Earnings before expiry — close to avoid gap risk", "HIGH")

    if current_premium >= entry_premium * max_loss_mult:
        return PositionAdvice("CLOSE_STOP",
            f"Stop loss — current {current_premium:.2f} is "
            f"{current_premium/entry_premium:.1f}x entry ({entry_premium:.2f})", "HIGH")

    if is_itm and dte <= roll_dte:
        return PositionAdvice("ROLL", f"ITM with {dte} DTE — high assignment risk, roll out", "HIGH")

    if captured >= profit_take:
        return PositionAdvice("CLOSE_PROFIT",
            f"Captured {captured:.0%} profit (target: {profit_take:.0%})", "MEDIUM")

    if dte <= 3 and not is_itm and captured > 0.75:
        return PositionAdvice("LET_EXPIRE",
            f"{captured:.0%} captured with {dte} DTE — let expire", "LOW")

    if dte <= roll_dte and not is_itm:
        return PositionAdvice("ROLL",
            f"DTE={dte} approaching — consider rolling to next cycle", "MEDIUM")

    return PositionAdvice("HOLD",
        f"On track — {captured:.0%} captured, {dte} DTE remaining", "LOW")
