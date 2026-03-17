"""Roll analysis for existing short call positions."""

from __future__ import annotations

import datetime as dt

from .config import load_config, get_short_calls
from .data.fetcher import fetch_stock, fetch_option_chain, RISK_FREE_RATE
from .data.greeks import black_scholes_greeks, bs_call_price


def analyze_rolls(config: dict | None = None) -> str:
    """Analyze existing short calls and generate roll/close recommendations."""
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

        is_itm = price >= strike
        moneyness = (price - strike) / strike * 100

        # Estimate current option value (use 30% IV as rough default)
        T = max(dte, 0) / 365.0
        iv_est = 0.30
        estimated_value = bs_call_price(price, strike, T, RISK_FREE_RATE, iv_est)
        greeks = black_scholes_greeks(price, strike, T, RISK_FREE_RATE, iv_est, "call")

        profit_pct = 0.0
        if premium_received > 0:
            profit_pct = (premium_received - estimated_value) / premium_received * 100

        lines.append(f"## {symbol} {expiry_str} ${strike} Call x{contracts}")
        lines.append(f"- Stock: ${price:.2f} | Strike: ${strike} | "
                      f"{'**ITM**' if is_itm else 'OTM'} ({moneyness:+.1f}%)")
        lines.append(f"- DTE: {dte} | Delta: {greeks.delta:.3f}")
        lines.append(f"- Premium received: ${premium_received:.2f} | "
                      f"Est. current value: ${estimated_value:.2f}")
        lines.append(f"- P&L: {profit_pct:.0f}% of max profit captured")

        # ── decision logic ───────────────────────────────────────────
        if dte <= 0:
            lines.append("")
            lines.append("**→ EXPIRED** — Remove from config:")
            lines.append(f"```\npython -m src.cli close-short {symbol} {expiry_str} {strike}\n```")

        elif dte <= 2 and is_itm:
            lines.append("")
            lines.append("**→ ROLL NOW** — ITM with ≤2 days left. High assignment risk.")
            _add_roll_candidates(lines, symbol, price, strike, dte, config)

        elif dte <= 3 and not is_itm and profit_pct > 75:
            lines.append("")
            lines.append(f"**→ LET EXPIRE** — OTM, {profit_pct:.0f}% profit captured. "
                          "Theta is working hard. Safe to let expire worthless.")

        elif profit_pct >= 50 and dte > 3:
            lines.append("")
            lines.append(f"**→ CONSIDER EARLY CLOSE** — {profit_pct:.0f}% profit with "
                          f"{dte} days remaining.")
            lines.append(f"  Buy to close at ~${estimated_value:.2f}/contract "
                          f"(${estimated_value * contracts * 100:.0f} total).")
            lines.append("  Then sell next cycle for fresh premium and reset theta.")

        elif is_itm and dte > 3:
            lines.append("")
            lines.append(f"**→ WATCH CLOSELY** — ITM with {dte} days left.")
            lines.append(f"  If stock stays above ${strike} by DTE=2, roll out and up.")
            _add_roll_candidates(lines, symbol, price, strike, dte, config)

        else:
            lines.append("")
            lines.append(f"**→ HOLD** — On track. {dte} days left, {profit_pct:.0f}% profit. "
                          "No action needed.")

        lines.append("")

    return "\n".join(lines)


def _add_roll_candidates(
    lines: list[str],
    symbol: str,
    price: float,
    current_strike: float,
    current_dte: int,
    config: dict,
) -> None:
    """Find and display roll candidates for the next expiry cycle."""
    candidates = fetch_option_chain(symbol, price, config, "call")
    # Filter: further out expiry, same or higher strike
    good = [c for c in candidates
            if c.dte > current_dte + 4 and c.strike >= current_strike]

    if not good:
        lines.append("  No roll candidates found in the current DTE window.")
        return

    lines.append("  **Roll candidates:**")
    lines.append("  | Expiry | Strike | Price* | Delta | OTM% | DTE |")
    lines.append("  |--------|--------|--------|-------|------|-----|")
    for c in good[:5]:
        p = c.last if c.off_hours else c.bid
        lines.append(
            f"  | {c.expiry} | ${c.strike:.1f} | ${p:.2f} | "
            f"{abs(c.greeks.delta):.3f} | {c.otm_pct:.1f}% | {c.dte} |"
        )
