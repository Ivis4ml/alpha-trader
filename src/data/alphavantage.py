"""Alpha Vantage data integration — fundamentals, economic indicators, earnings.

Supplements Yahoo Finance with higher-quality data:
- Company fundamentals (P/E, EPS, revenue growth, profit margin)
- Economic indicators (Fed funds rate, CPI, unemployment, GDP)
- Earnings calendar with estimates vs actuals
- Top gainers/losers/most active

API key from .env: ALPHAVANTAGE_API_KEY
Free tier: 25 requests/day. Cache aggressively.
"""

from __future__ import annotations

import os
import json
import datetime as dt
import pathlib
import requests

_API_KEY = None
_BASE = "https://www.alphavantage.co/query"
_CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "av_cache"


def _get_key() -> str:
    global _API_KEY
    if _API_KEY is None:
        _API_KEY = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    return _API_KEY


def _cached_fetch(function: str, params: dict, ttl_hours: int = 12) -> dict | None:
    """Fetch from AV with file-based cache to stay within free tier limits."""
    key = _get_key()
    if not key:
        return None

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = f"{function}_{'_'.join(f'{k}={v}' for k, v in sorted(params.items()))}"
    cache_file = _CACHE_DIR / f"{cache_key}.json"

    # Check cache
    if cache_file.exists():
        age = dt.datetime.now().timestamp() - cache_file.stat().st_mtime
        if age < ttl_hours * 3600:
            try:
                return json.loads(cache_file.read_text())
            except Exception:
                pass

    # Fetch
    all_params = {"function": function, "apikey": key, **params}
    try:
        resp = requests.get(_BASE, params=all_params, timeout=15)
        data = resp.json()
        if "Error Message" in data or "Note" in data:
            return None
        cache_file.write_text(json.dumps(data))
        return data
    except Exception:
        return None


# ── Company Fundamentals ────────────────────────────────────────────────────

def fetch_fundamentals(symbol: str) -> dict:
    """Get company overview: P/E, EPS, revenue, margins, etc."""
    data = _cached_fetch("OVERVIEW", {"symbol": symbol}, ttl_hours=24)
    if not data:
        return {}

    def _f(key):
        v = data.get(key)
        if v and v != "None" and v != "-":
            try:
                return float(v)
            except (ValueError, TypeError):
                return v
        return None

    return {
        "pe_ratio": _f("PERatio"),
        "forward_pe": _f("ForwardPE"),
        "eps": _f("EPS"),
        "revenue_growth_yoy": _f("QuarterlyRevenueGrowthYOY"),
        "earnings_growth_yoy": _f("QuarterlyEarningsGrowthYOY"),
        "profit_margin": _f("ProfitMargin"),
        "operating_margin": _f("OperatingMarginTTM"),
        "return_on_equity": _f("ReturnOnEquityTTM"),
        "dividend_yield": _f("DividendYield"),
        "beta": _f("Beta"),
        "market_cap": _f("MarketCapitalization"),
        "52w_high": _f("52WeekHigh"),
        "52w_low": _f("52WeekLow"),
        "analyst_target": _f("AnalystTargetPrice"),
        "sector": data.get("Sector"),
        "industry": data.get("Industry"),
    }


# ── Earnings Calendar ───────────────────────────────────────────────────────

def fetch_earnings(symbol: str) -> list[dict]:
    """Get upcoming and recent earnings with estimates."""
    data = _cached_fetch("EARNINGS", {"symbol": symbol}, ttl_hours=24)
    if not data:
        return []

    results = []
    for q in (data.get("quarterlyEarnings") or [])[:8]:
        results.append({
            "date": q.get("fiscalDateEnding"),
            "reported_date": q.get("reportedDate"),
            "estimated_eps": _safe_float(q.get("estimatedEPS")),
            "actual_eps": _safe_float(q.get("reportedEPS")),
            "surprise": _safe_float(q.get("surprise")),
            "surprise_pct": _safe_float(q.get("surprisePercentage")),
        })
    return results


# ── Economic Indicators ─────────────────────────────────────────────────────

def fetch_fed_funds_rate() -> dict:
    """Current federal funds rate (affects risk-free rate for BS model)."""
    data = _cached_fetch("FEDERAL_FUNDS_RATE", {"interval": "monthly"}, ttl_hours=48)
    if not data or "data" not in data:
        return {}
    recent = data["data"][:3]
    return {
        "current": float(recent[0]["value"]) if recent else None,
        "previous": float(recent[1]["value"]) if len(recent) > 1 else None,
        "trend": "rising" if len(recent) > 1 and float(recent[0]["value"]) > float(recent[1]["value"]) else "falling",
        "history": [{"date": r["date"], "rate": float(r["value"])} for r in recent[:6]],
    }


def fetch_cpi() -> dict:
    """Consumer Price Index (inflation indicator)."""
    data = _cached_fetch("CPI", {"interval": "monthly"}, ttl_hours=48)
    if not data or "data" not in data:
        return {}
    recent = data["data"][:3]
    if len(recent) < 2:
        return {}
    current = float(recent[0]["value"])
    previous = float(recent[1]["value"])
    return {
        "current": current,
        "previous": previous,
        "mom_change": round(current - previous, 2),
        "date": recent[0]["date"],
    }


def fetch_unemployment() -> dict:
    """Unemployment rate."""
    data = _cached_fetch("UNEMPLOYMENT", {}, ttl_hours=48)
    if not data or "data" not in data:
        return {}
    recent = data["data"][:3]
    return {
        "current": float(recent[0]["value"]) if recent else None,
        "date": recent[0]["date"] if recent else None,
    }


def fetch_treasury_yield() -> dict:
    """10-year Treasury yield (benchmark for risk-free rate)."""
    data = _cached_fetch("TREASURY_YIELD", {"interval": "monthly", "maturity": "10year"}, ttl_hours=48)
    if not data or "data" not in data:
        return {}
    recent = data["data"][:3]
    return {
        "current": float(recent[0]["value"]) if recent else None,
        "date": recent[0]["date"] if recent else None,
    }


# ── Market Movers ───────────────────────────────────────────────────────────

def fetch_market_movers() -> dict:
    """Top gainers, losers, and most active stocks today."""
    data = _cached_fetch("TOP_GAINERS_LOSERS", {}, ttl_hours=4)
    if not data:
        return {}
    return {
        "top_gainers": [
            {"ticker": g["ticker"], "price": g.get("price"), "change_pct": g.get("change_percentage")}
            for g in (data.get("top_gainers") or [])[:5]
        ],
        "top_losers": [
            {"ticker": g["ticker"], "price": g.get("price"), "change_pct": g.get("change_percentage")}
            for g in (data.get("top_losers") or [])[:5]
        ],
        "most_active": [
            {"ticker": g["ticker"], "price": g.get("price"), "volume": g.get("volume")}
            for g in (data.get("most_actively_traded") or [])[:5]
        ],
    }


# ── Aggregated Macro Snapshot ────────────────────────────────────────────────

def fetch_macro_snapshot() -> dict:
    """One-call macro overview: fed rate, CPI, unemployment, treasury yield."""
    return {
        "fed_funds": fetch_fed_funds_rate(),
        "cpi": fetch_cpi(),
        "unemployment": fetch_unemployment(),
        "treasury_10y": fetch_treasury_yield(),
    }


def format_macro_snapshot(macro: dict) -> str:
    lines = ["**Macro:**"]
    ff = macro.get("fed_funds", {})
    if ff.get("current"):
        lines.append(f"Fed Rate {ff['current']:.2f}% ({ff.get('trend', '?')})")
    ty = macro.get("treasury_10y", {})
    if ty.get("current"):
        lines.append(f"| 10Y Yield {ty['current']:.2f}%")
    cpi = macro.get("cpi", {})
    if cpi.get("current"):
        lines.append(f"| CPI {cpi['current']:.1f} ({cpi.get('mom_change', 0):+.1f} MoM)")
    ue = macro.get("unemployment", {})
    if ue.get("current"):
        lines.append(f"| Unemp {ue['current']:.1f}%")
    return " ".join(lines)


def format_fundamentals(fund: dict, symbol: str) -> str:
    if not fund:
        return ""
    parts = [f"**{symbol} Fundamentals:**"]
    if fund.get("pe_ratio"):
        parts.append(f"P/E {fund['pe_ratio']:.1f}")
    if fund.get("eps"):
        parts.append(f"EPS ${fund['eps']:.2f}")
    if fund.get("revenue_growth_yoy"):
        parts.append(f"Rev Growth {fund['revenue_growth_yoy']:.1%}")
    if fund.get("profit_margin"):
        parts.append(f"Margin {fund['profit_margin']:.1%}")
    if fund.get("beta"):
        parts.append(f"Beta {fund['beta']:.2f}")
    return " | ".join(parts)


def _safe_float(v):
    if v is None or v == "None" or v == "-":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None
