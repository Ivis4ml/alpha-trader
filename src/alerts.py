"""Proactive alerts engine — monitors positions and market conditions."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .config import load_config, get_short_calls, get_symbols, get_position
from .data.fetcher import fetch_stock, StockSnapshot


@dataclass
class Alert:
    severity: str   # URGENT, WARNING, INFO
    symbol: str
    title: str
    detail: str


def check_all_alerts(config: dict | None = None) -> list[Alert]:
    """Run all alert checks. Returns list of alerts sorted by severity."""
    if config is None:
        config = load_config()

    alerts: list[Alert] = []
    symbols = get_symbols(config)
    shorts = get_short_calls(config)

    # Fetch stock data for all symbols
    stocks: dict[str, StockSnapshot] = {}
    for sym in symbols:
        try:
            stocks[sym] = fetch_stock(sym)
        except Exception:
            pass

    # ── Short call alerts ────────────────────────────────────────
    today = dt.date.today()
    for sc in shorts:
        sym = sc["symbol"]
        strike = sc["strike"]
        expiry_str = sc["expiry"]
        contracts = sc["contracts"]
        stock = stocks.get(sym)
        if not stock:
            continue

        expiry = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
        dte = (expiry - today).days

        # 1. DTE <= 2 roll reminder
        if dte <= 2 and dte >= 0:
            alerts.append(Alert(
                severity="URGENT",
                symbol=sym,
                title=f"Roll needed — {sym} ${strike}C expires in {dte}d",
                detail=f"{contracts}x {sym} {expiry_str} ${strike}C — DTE={dte}. "
                       f"Stock at ${stock.price:.2f}. "
                       f"{'ITM — high assignment risk!' if stock.price >= strike else 'OTM — may expire worthless.'}",
            ))

        # 2. Expired — needs cleanup
        if dte < 0:
            alerts.append(Alert(
                severity="WARNING",
                symbol=sym,
                title=f"Expired — {sym} ${strike}C past expiry",
                detail=f"Expired on {expiry_str}. Run: close-short {sym} {expiry_str} {strike}",
            ))

        # 3. Stock crosses above strike
        if stock.price >= strike and dte > 0:
            alerts.append(Alert(
                severity="URGENT",
                symbol=sym,
                title=f"ITM! {sym} ${stock.price:.2f} > ${strike} strike",
                detail=f"{contracts}x {sym} {expiry_str} ${strike}C is in-the-money. "
                       f"DTE={dte}. Consider rolling up and out.",
            ))
        elif stock.price >= strike * 0.98 and dte > 0:
            alerts.append(Alert(
                severity="WARNING",
                symbol=sym,
                title=f"Near strike — {sym} ${stock.price:.2f} approaching ${strike}",
                detail=f"Stock is within 2% of strike. Monitor closely.",
            ))

    # ── Technical alerts ─────────────────────────────────────────
    for sym, stock in stocks.items():
        t = stock.technicals
        if not t:
            continue

        # 4. RSI > 70 overbought
        if t.rsi_14 is not None and t.rsi_14 > 70:
            alerts.append(Alert(
                severity="WARNING",
                symbol=sym,
                title=f"Overbought — {sym} RSI {t.rsi_14:.0f}",
                detail=f"RSI > 70 suggests potential pullback. "
                       f"If selling calls, consider higher strikes. "
                       f"If holding short calls, stock may pull back in your favor.",
            ))

        # 5. RSI < 30 oversold
        if t.rsi_14 is not None and t.rsi_14 < 30:
            alerts.append(Alert(
                severity="INFO",
                symbol=sym,
                title=f"Oversold — {sym} RSI {t.rsi_14:.0f}",
                detail=f"RSI < 30 suggests potential bounce. "
                       f"Good time to sell puts (CSP) or wait before selling calls.",
            ))

        # 6. Bollinger Band breakout
        if t.bb_upper is not None and stock.price > t.bb_upper:
            alerts.append(Alert(
                severity="WARNING",
                symbol=sym,
                title=f"Above Bollinger — {sym} ${stock.price:.2f} > BB ${t.bb_upper:.2f}",
                detail=f"Price broke above upper Bollinger Band. "
                       f"Overbought signal. May revert to mean. "
                       f"Caution selling calls at current levels.",
            ))

        # 7. Bollinger squeeze (low BB width = breakout imminent)
        if t.bb_width is not None and t.bb_width < 6:
            alerts.append(Alert(
                severity="INFO",
                symbol=sym,
                title=f"BB Squeeze — {sym} width {t.bb_width:.1f}%",
                detail=f"Bollinger Bands are very narrow — breakout likely. "
                       f"Direction unknown. Consider waiting before selling calls.",
            ))

        # 8. ATR spike (sudden volatility increase)
        if t.atr_14 is not None:
            atr_pct = t.atr_14 / stock.price * 100
            if atr_pct > 4:  # ATR > 4% of price = high volatility
                alerts.append(Alert(
                    severity="WARNING",
                    symbol=sym,
                    title=f"High volatility — {sym} ATR ${t.atr_14:.2f} ({atr_pct:.1f}%)",
                    detail=f"ATR is elevated. Premiums should be rich but assignment risk higher. "
                           f"Consider wider OTM strikes.",
                ))

    # Sort: URGENT first, then WARNING, then INFO
    severity_order = {"URGENT": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: severity_order.get(a.severity, 3))
    return alerts


def format_alerts(alerts: list[Alert]) -> str:
    """Format alerts as readable text."""
    if not alerts:
        return "No alerts. All positions look good."

    lines = ["# Alerts\n"]
    for a in alerts:
        icon = {"URGENT": "🔴", "WARNING": "🟡", "INFO": "🔵"}.get(a.severity, "⚪")
        lines.append(f"{icon} **{a.severity}** [{a.symbol}] {a.title}")
        lines.append(f"   {a.detail}")
        lines.append("")

    return "\n".join(lines)
