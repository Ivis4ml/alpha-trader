"""Additional free data sources — no API keys required.

All functions use yfinance exclusively.
"""

from __future__ import annotations

import math

import yfinance as yf
import pandas as pd


# ── Unusual Options Activity ────────────────────────────────────────────────

def fetch_unusual_options_activity(symbol: str) -> list[dict]:
    """Detect strikes with unusually high volume vs open interest.

    Flags strikes where volume > 3x open_interest OR volume > 5000.
    Uses the nearest expiry option chain from yfinance.
    """
    t = yf.Ticker(symbol)
    try:
        expiries = t.options
    except Exception:
        return []

    if not expiries:
        return []

    # Use nearest expiry
    exp_str = expiries[0]
    try:
        chain = t.option_chain(exp_str)
    except Exception:
        return []

    results: list[dict] = []

    for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
        if df is None or df.empty:
            continue
        for _, row in df.iterrows():
            import math
            _vol = row.get("volume", 0)
            _oi = row.get("openInterest", 0)
            vol = 0 if (_vol is None or (isinstance(_vol, float) and math.isnan(_vol))) else int(_vol)
            oi = 0 if (_oi is None or (isinstance(_oi, float) and math.isnan(_oi))) else int(_oi)
            strike = float(row.get("strike", 0) or 0)

            if vol == 0:
                continue

            ratio = vol / oi if oi > 0 else float("inf")

            if ratio > 3.0 or vol > 5000:
                results.append({
                    "strike": strike,
                    "volume": vol,
                    "open_interest": oi,
                    "ratio": round(ratio, 1) if oi > 0 else None,
                    "type": opt_type,
                    "expiry": exp_str,
                })

    # Sort by volume descending — most notable first
    results.sort(key=lambda r: r["volume"], reverse=True)
    return results


# ── Put/Call Ratio ──────────────────────────────────────────────────────────

def fetch_put_call_ratio(symbol: str) -> float | None:
    """Calculate put/call volume ratio for the nearest expiry.

    >1.0 = bearish sentiment, <0.7 = bullish.
    Returns None if data is unavailable.
    """
    t = yf.Ticker(symbol)
    try:
        expiries = t.options
    except Exception:
        return None

    if not expiries:
        return None

    exp_str = expiries[0]
    try:
        chain = t.option_chain(exp_str)
    except Exception:
        return None

    call_vol = 0
    put_vol = 0

    if chain.calls is not None and not chain.calls.empty:
        call_vol = int(chain.calls["volume"].fillna(0).sum())
    if chain.puts is not None and not chain.puts.empty:
        put_vol = int(chain.puts["volume"].fillna(0).sum())

    if call_vol == 0:
        return None

    return round(put_vol / call_vol, 2)


# ── Sector Performance ──────────────────────────────────────────────────────

def fetch_sector_performance() -> dict:
    """Fetch XLK, SPY, QQQ performance over 1d, 5d, 1mo.

    Returns dict keyed by ticker, each containing period returns.
    """
    tickers = {"XLK": "Tech Sector", "SPY": "S&P 500", "QQQ": "Nasdaq 100"}
    result: dict = {}

    for sym, label in tickers.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="1mo")
            if hist.empty or len(hist) < 2:
                continue

            close = hist["Close"]
            current = close.iloc[-1]

            # 1-day return
            ret_1d = (current / close.iloc[-2] - 1) * 100 if len(close) >= 2 else 0
            # 5-day return
            ret_5d = (current / close.iloc[-6] - 1) * 100 if len(close) >= 6 else None
            # 1-month return (first available in this period)
            ret_1mo = (current / close.iloc[0] - 1) * 100

            result[sym] = {
                "label": label,
                "1d": round(ret_1d, 2),
                "5d": round(ret_5d, 2) if ret_5d is not None else None,
                "1mo": round(ret_1mo, 2),
            }
        except Exception:
            continue

    return result


# ── Institutional Holdings ──────────────────────────────────────────────────

def fetch_institutional_holdings(symbol: str) -> dict:
    """Fetch institutional holder summary from yfinance.

    Returns a summary dict with top holders and major holder percentages.
    """
    t = yf.Ticker(symbol)
    result: dict = {}

    # Major holders (% insiders, % institutions, etc.)
    try:
        mh = t.major_holders
        if mh is not None and not mh.empty:
            holders = {}
            for _, row in mh.iterrows():
                # major_holders has two columns: value and description
                val = row.iloc[0] if len(row) > 0 else None
                desc = str(row.iloc[1]).strip() if len(row) > 1 else ""
                if desc:
                    holders[desc] = str(val)
            result["major"] = holders
    except Exception:
        pass

    # Top institutional holders
    try:
        ih = t.institutional_holders
        if ih is not None and not ih.empty:
            top = []
            for _, row in ih.head(5).iterrows():
                holder = str(row.get("Holder", row.get("holder", "")))
                shares = row.get("Shares", row.get("shares", 0))
                try:
                    shares = int(shares) if shares and str(shares) != "nan" else 0
                except (ValueError, TypeError):
                    shares = 0
                pct = row.get("pctHeld", row.get("% Out", None))
                try:
                    pct = round(float(pct) * 100, 2) if pct and str(pct) != "nan" else None
                except (ValueError, TypeError):
                    pct = None
                top.append({"holder": holder, "shares": shares, "pct": pct})
            result["top_holders"] = top
    except Exception:
        pass

    return result
