"""Roll analysis for existing short call positions.

Uses advise_position() from strategy.py for deterministic hold/close/roll
decisions, and scan_engine for scoring roll candidates.
"""

from __future__ import annotations

import datetime as dt

from .config import load_config, get_short_calls, get_position, get_delta_range
from .data.fetcher import fetch_stock, fetch_events, fetch_iv_stats, fetch_option_chain, RISK_FREE_RATE
from .data.greeks import black_scholes_greeks, bs_call_price
from .strategy import advise_position


def analyze_rolls(config: dict | None = None, regime: str = "balanced") -> str:
    """Analyze existing short calls and generate roll/close recommendations.

    Uses the same advise_position() logic as ``./at daily`` so hold/close/roll
    decisions are consistent across all commands.
    """
    if config is None:
        config = load_config()

    shorts = get_short_calls(config)
    if not shorts:
        return "No open short calls to analyze.\n"

    today = dt.date.today()
    lines = ["# Roll & Position Analysis", ""]

    for sc in shorts:
        symbol = sc["symbol"]
        strike = sc["strike"]
        expiry_str = sc["expiry"]
        contracts = sc["contracts"]
        premium_received = sc.get("premium_received", 0)

        expiry = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        dte = (expiry - today).days

        stock = fetch_stock(symbol)
        price = stock.price
        events = fetch_events(symbol)

        is_itm = price >= strike
        moneyness = (price - strike) / strike * 100

        # Estimate current option value
        T = max(dte, 0) / 365.0
        iv_est = 0.30
        estimated_value = bs_call_price(price, strike, T, RISK_FREE_RATE, iv_est)
        greeks = black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv_est, "call")

        profit_pct = 0.0
        if premium_received > 0:
            profit_pct = (premium_received - estimated_value) / premium_received * 100

        # Position header
        lines.append(f"## {symbol} {expiry_str} ${strike} Call x{contracts}")
        lines.append(f"- Stock: ${price:.2f} | Strike: ${strike} | "
                      f"{'**ITM**' if is_itm else 'OTM'} ({moneyness:+.1f}%)")
        lines.append(f"- DTE: {dte} | Delta: {greeks.delta:.3f}")
        lines.append(f"- Premium received: ${premium_received:.2f} | "
                      f"Est. current value: ${estimated_value:.2f}")
        lines.append(f"- P&L: {profit_pct:.0f}% of max profit captured")

        # Deterministic advice from strategy.py (single source of truth)
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

        # Format advice
        urgency_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(advice.urgency, "")
        lines.append("")
        lines.append(f"**→ {advice.action}** {urgency_icon} — {advice.reason}")

        # For ROLL / CLOSE_EARNINGS: show scored roll candidates
        if advice.action in ("ROLL", "CLOSE_EARNINGS"):
            _add_scored_roll_candidates(lines, symbol, price, strike, dte, config, regime)

        # For CLOSE_PROFIT / CLOSE_STOP: show close cost
        if advice.action in ("CLOSE_PROFIT", "CLOSE_STOP"):
            close_cost = estimated_value * contracts * 100
            lines.append(f"  Buy to close at ~${estimated_value:.2f}/contract "
                          f"(${close_cost:,.0f} total)")

        lines.append("")

    return "\n".join(lines)


def _add_scored_roll_candidates(
    lines: list[str],
    symbol: str,
    price: float,
    current_strike: float,
    current_dte: int,
    config: dict,
    regime: str,
) -> None:
    """Find roll candidates and score them via scan_engine."""
    from .scan_engine import build_features, apply_hard_filters, score_candidate
    from .data.events_calendar import assess_event_risk

    position = get_position(config, symbol)
    iv_stats = fetch_iv_stats(symbol)
    events = fetch_events(symbol)
    stock = fetch_stock(symbol)
    strat = config.get("strategy", {})
    delta_range = get_delta_range(config, regime)

    candidates = fetch_option_chain(symbol, price, config, "call")
    # Roll = further out expiry, same or higher strike
    rollable = [c for c in candidates
                if c.dte > current_dte + 4 and c.strike >= current_strike]

    if not rollable:
        lines.append("  No roll candidates found in the current DTE window.")
        return

    # Build features and hard-filter first, then score from passed pool only
    # (same as scan_engine.scan_symbol — normalize against clean candidates)
    passed_features = []
    for opt in rollable:
        feat = build_features(opt, stock, iv_stats, events, position)
        reasons = apply_hard_filters(feat, config)
        if not reasons:
            passed_features.append(feat)

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
        score, breakdown, flags = score_candidate(
            feat, config, event_risk, all_yields, all_thetas, all_ois,
            delta_range=delta_range,
        )
        scored.append((score, feat, flags))

    scored.sort(key=lambda x: x[0], reverse=True)

    lines.append("  **Roll candidates (scored):**")
    lines.append("  | Expiry | Strike | Price | Score | Delta | OTM% | DTE | Flags |")
    lines.append("  |--------|--------|-------|-------|-------|------|-----|-------|")
    for score, feat, flags in scored[:5]:
        flags_str = " ".join(flags)
        lines.append(
            f"  | {feat.expiry} | ${feat.strike:.1f} | ${feat.premium:.2f} | "
            f"{int(score * 100):>3d} | {feat.delta:.3f} | {feat.otm_pct:.1f}% | "
            f"{feat.dte} | {flags_str} |"
        )
