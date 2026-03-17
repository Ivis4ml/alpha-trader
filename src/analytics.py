"""Advanced analytics — P&L attribution, correlation, earnings crush, IV surface."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import yfinance as yf
import pandas as pd

from .data.greeks import black_scholes_greeks, bs_call_price, implied_volatility
from .data.fetcher import RISK_FREE_RATE


# ── P&L Attribution ─────────────────────────────────────────────────────────

@dataclass
class PnLAttribution:
    """Break down option P&L into Greek components."""
    total_pnl: float
    theta_pnl: float       # time decay contribution
    delta_pnl: float       # stock move contribution
    vega_pnl: float        # IV change contribution
    gamma_pnl: float       # convexity contribution
    unexplained: float     # residual


def attribute_pnl(
    stock_price_open: float,
    stock_price_close: float,
    iv_open: float,
    iv_close: float,
    strike: float,
    dte_open: int,
    dte_close: int,
    premium_open: float,
    premium_close: float,
    contracts: int = 1,
) -> PnLAttribution:
    """Decompose option P&L into Greek contributions."""
    T_open = max(dte_open, 1) / 365.0
    T_close = max(dte_close, 1) / 365.0
    dt_years = (dte_open - dte_close) / 365.0

    g = black_scholes_greeks(stock_price_open, strike, T_open, RISK_FREE_RATE, iv_open, "call")
    dS = stock_price_close - stock_price_open
    dIV = iv_close - iv_open

    # P&L components (per share, seller perspective — negative for seller means profit)
    delta_pnl = -g.delta * dS
    gamma_pnl = -0.5 * g.gamma * dS ** 2
    theta_pnl = -g.theta * (dte_open - dte_close)  # theta is daily, already negative for calls
    vega_pnl = -g.vega * (dIV * 100)  # vega is per 1% IV

    total = (premium_open - premium_close)
    unexplained = total - (delta_pnl + gamma_pnl + theta_pnl + vega_pnl)

    mult = contracts * 100
    return PnLAttribution(
        total_pnl=round(total * mult, 2),
        theta_pnl=round(theta_pnl * mult, 2),
        delta_pnl=round(delta_pnl * mult, 2),
        vega_pnl=round(vega_pnl * mult, 2),
        gamma_pnl=round(gamma_pnl * mult, 2),
        unexplained=round(unexplained * mult, 2),
    )


# ── Correlation Analysis ────────────────────────────────────────────────────

def analyze_correlation(symbols: list[str], period: str = "6mo") -> dict:
    """Analyze correlation between positions."""
    if len(symbols) < 2:
        return {"error": "Need at least 2 symbols"}

    data = yf.download(symbols, period=period, progress=False)["Close"]
    if data.empty:
        return {"error": "No data"}

    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()
    rolling_corr = {}

    # Pairwise correlations
    pairs = []
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i+1:]:
            try:
                c = corr_matrix.loc[s1, s2]
                roll_30 = returns[s1].rolling(30).corr(returns[s2]).iloc[-1]
                pairs.append({
                    "pair": f"{s1}/{s2}",
                    "correlation_6mo": round(c, 3),
                    "correlation_30d": round(roll_30, 3),
                    "diversification": "LOW" if abs(c) > 0.7 else ("MODERATE" if abs(c) > 0.4 else "GOOD"),
                })
            except Exception:
                pass

    # Beta to SPY
    spy = yf.download("SPY", period=period, progress=False)["Close"].pct_change().dropna()
    betas = {}
    for sym in symbols:
        try:
            sym_ret = returns[sym].dropna()
            common = pd.concat([sym_ret, spy], axis=1, join="inner").dropna()
            if len(common) > 20:
                cov = common.iloc[:, 0].cov(common.iloc[:, 1])
                var = common.iloc[:, 1].var()
                betas[sym] = round(cov / var, 2) if var > 0 else 0
        except Exception:
            pass

    return {"pairs": pairs, "betas": betas}


# ── Earnings Volatility Crush ───────────────────────────────────────────────

def analyze_earnings_crush(symbol: str) -> dict:
    """Analyze historical IV behavior around earnings for a symbol."""
    t = yf.Ticker(symbol)
    hist = t.history(period="2y")

    if len(hist) < 100:
        return {"error": "Insufficient history"}

    # Calculate 20-day rolling HV
    returns = hist["Close"].pct_change()
    hv = returns.rolling(20).std() * math.sqrt(252)

    # Get earnings dates — try multiple yfinance methods
    edates = []
    try:
        earnings = t.earnings_dates
        if earnings is not None and not earnings.empty:
            edates = sorted([d.date() if hasattr(d, 'date') else d for d in earnings.index])
    except Exception:
        pass

    if not edates:
        # Fallback: detect earnings from large 1-day HV spikes
        daily_ret = returns.abs()
        threshold = daily_ret.mean() + 2.5 * daily_ret.std()
        spikes = daily_ret[daily_ret > threshold]
        edates = sorted([d.date() if hasattr(d, 'date') else d for d in spikes.index])

    if not edates:
        return {"error": "No earnings data found"}

    # Analyze IV around each earnings
    crushes = []
    for ed in edates:
        try:
            # Find HV 5 days before and 5 days after earnings
            ed_ts = pd.Timestamp(ed)
            before_mask = (hv.index >= ed_ts - pd.Timedelta(days=10)) & (hv.index < ed_ts)
            after_mask = (hv.index > ed_ts) & (hv.index <= ed_ts + pd.Timedelta(days=10))

            hv_before = hv[before_mask]
            hv_after = hv[after_mask]

            if len(hv_before) > 0 and len(hv_after) > 0:
                iv_pre = hv_before.iloc[-1]
                iv_post = hv_after.iloc[0]
                crush_pct = (iv_pre - iv_post) / iv_pre * 100 if iv_pre > 0 else 0

                # Price move on earnings
                price_before = hist["Close"][before_mask]
                price_after = hist["Close"][after_mask]
                if len(price_before) > 0 and len(price_after) > 0:
                    move_pct = (price_after.iloc[0] - price_before.iloc[-1]) / price_before.iloc[-1] * 100
                else:
                    move_pct = 0

                crushes.append({
                    "date": str(ed),
                    "iv_before": round(iv_pre * 100, 1),
                    "iv_after": round(iv_post * 100, 1),
                    "crush_pct": round(crush_pct, 1),
                    "price_move_pct": round(move_pct, 1),
                })
        except Exception:
            continue

    avg_crush = sum(c["crush_pct"] for c in crushes) / len(crushes) if crushes else 0
    avg_move = sum(abs(c["price_move_pct"]) for c in crushes) / len(crushes) if crushes else 0

    return {
        "symbol": symbol,
        "earnings_analyzed": len(crushes),
        "avg_iv_crush_pct": round(avg_crush, 1),
        "avg_earnings_move_pct": round(avg_move, 1),
        "history": crushes[-8:],  # last 8 earnings
        "recommendation": (
            "High crush — selling calls/puts before earnings captures vol premium"
            if avg_crush > 15 else
            "Moderate crush — earnings play viable but size conservatively"
            if avg_crush > 5 else
            "Low crush — earnings vol premium is thin, be cautious"
        ),
    }


# ── Tax Lot Optimization ────────────────────────────────────────────────────

def suggest_tax_lots(
    symbol: str,
    lots: list[dict],  # [{"shares": 100, "cost_basis": 150.00, "date": "2024-01-15"}, ...]
    current_price: float,
) -> list[dict]:
    """Suggest which tax lots to let get assigned if called away.

    Strategy: Let lots with highest cost basis get assigned first
    (minimizes capital gains tax).
    """
    today = dt.date.today()

    for lot in lots:
        lot["gain"] = (current_price - lot["cost_basis"]) * lot["shares"]
        lot["gain_pct"] = (current_price - lot["cost_basis"]) / lot["cost_basis"] * 100

        acq = dt.datetime.strptime(lot["date"], "%Y-%m-%d").date()
        lot["holding_days"] = (today - acq).days
        lot["long_term"] = lot["holding_days"] >= 365
        lot["tax_type"] = "LTCG" if lot["long_term"] else "STCG"

    # Sort: prefer to assign STCG with lowest gain first (minimizes tax)
    sorted_lots = sorted(lots, key=lambda l: (l["long_term"], l["gain"]))

    for i, lot in enumerate(sorted_lots):
        lot["assign_priority"] = i + 1
        lot["recommendation"] = (
            "ASSIGN FIRST — short-term, low gain"
            if not lot["long_term"] and lot["gain"] < 0 else
            "ASSIGN FIRST — short-term loss (tax benefit)"
            if not lot["long_term"] and lot["gain"] < 0 else
            "PREFER TO KEEP — long-term, favorable tax rate"
            if lot["long_term"] else
            "NEUTRAL"
        )

    return sorted_lots


# ── CLI Formatters ───────────────────────────────────────────────────────────

def format_correlation(result: dict) -> str:
    lines = ["## Position Correlation Analysis\n"]
    if "error" in result:
        return result["error"]

    lines.append("| Pair | 6mo Corr | 30d Corr | Diversification |")
    lines.append("|------|----------|----------|-----------------|")
    for p in result.get("pairs", []):
        lines.append(f"| {p['pair']} | {p['correlation_6mo']:.3f} | {p['correlation_30d']:.3f} | {p['diversification']} |")

    lines.append("\n| Symbol | Beta to SPY |")
    lines.append("|--------|------------|")
    for sym, beta in result.get("betas", {}).items():
        lines.append(f"| {sym} | {beta:.2f} |")

    return "\n".join(lines)


def format_earnings_crush(result: dict) -> str:
    if "error" in result:
        return result["error"]

    lines = [f"## Earnings Volatility Crush — {result['symbol']}\n"]
    lines.append(f"Analyzed {result['earnings_analyzed']} earnings events\n")
    lines.append(f"- Avg IV crush: **{result['avg_iv_crush_pct']}%**")
    lines.append(f"- Avg price move: **{result['avg_earnings_move_pct']}%**")
    lines.append(f"- {result['recommendation']}\n")

    if result.get("history"):
        lines.append("| Date | IV Before | IV After | Crush% | Move% |")
        lines.append("|------|-----------|----------|--------|-------|")
        for h in result["history"]:
            lines.append(
                f"| {h['date']} | {h['iv_before']}% | {h['iv_after']}% | "
                f"{h['crush_pct']}% | {h['price_move_pct']:+.1f}% |"
            )

    return "\n".join(lines)
