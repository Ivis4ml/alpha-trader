"""Fetch market data, option chains, and events from Yahoo Finance."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import yfinance as yf
import pandas as pd

from .greeks import black_scholes_greeks, Greeks, implied_volatility


def _safe_float(val, default=0.0) -> float:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def _safe_int(val, default=0) -> int:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return int(val)

# Approximate risk-free rate — updated periodically or fetched
RISK_FREE_RATE = 0.043


# ── dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class TechnicalIndicators:
    rsi_14: float | None       # 0-100, >70 overbought, <30 oversold
    macd: float | None         # MACD line value
    macd_signal: float | None  # Signal line
    macd_hist: float | None    # Histogram (macd - signal)
    bb_upper: float | None     # Bollinger upper band (20,2)
    bb_lower: float | None     # Bollinger lower band
    bb_width: float | None     # (upper-lower)/mid as %, squeeze detection
    atr_14: float | None       # Average True Range, for OTM distance sizing


@dataclass
class StockSnapshot:
    symbol: str
    price: float
    prev_close: float
    day_change_pct: float
    sma_20: float | None
    sma_50: float | None
    high_52w: float
    low_52w: float
    market_cap: float | None
    volume: int | None
    technicals: TechnicalIndicators | None = None


@dataclass
class OptionRow:
    """One strike in an option chain, enriched with Greeks."""
    expiry: str          # YYYY-MM-DD
    strike: float
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    implied_vol: float
    greeks: Greeks
    dte: int             # calendar days to expiry
    annualized_yield: float   # annualized premium / stock price
    otm_pct: float       # % out of the money
    spread_pct: float    # bid-ask spread as % of mid
    off_hours: bool = False  # True when using lastPrice (market closed)


@dataclass
class EventInfo:
    next_earnings: str | None    # YYYY-MM-DD or None
    days_to_earnings: int | None
    next_ex_div: str | None
    days_to_ex_div: int | None


@dataclass
class IVStats:
    current_iv: float        # current implied vol (ATM approx)
    iv_high_52w: float
    iv_low_52w: float
    iv_rank: float           # 0-100
    iv_percentile: float     # 0-100


@dataclass
class SymbolBriefing:
    stock: StockSnapshot
    iv_stats: IVStats
    events: EventInfo
    call_chains: list[OptionRow] = field(default_factory=list)
    put_chains: list[OptionRow] = field(default_factory=list)
    news: list[dict] = field(default_factory=list)
    insider: list[dict] = field(default_factory=list)
    analyst: dict = field(default_factory=dict)
    unusual_activity: list[dict] = field(default_factory=list)
    put_call_ratio: float | None = None
    sector: dict = field(default_factory=dict)


@dataclass
class MarketContext:
    vix: float
    vix_change: float
    spy_price: float
    spy_change_pct: float
    spy_above_sma20: bool
    regime: str              # conservative / balanced / aggressive
    timestamp: str


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe_get(info: dict, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None:
            return v
    return default


def _years_to_expiry(expiry_str: str) -> float:
    exp = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    today = dt.date.today()
    days = (exp - today).days
    return max(days, 0) / 365.0


def _dte(expiry_str: str) -> int:
    exp = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
    return max((exp - dt.date.today()).days, 0)


# ── fetchers ─────────────────────────────────────────────────────────────────

def fetch_market_context(config: dict) -> MarketContext:
    """Fetch VIX, SPY, and determine market regime."""
    vix_t = yf.Ticker("^VIX")
    spy_t = yf.Ticker("SPY")

    vix_info = vix_t.fast_info
    spy_info = spy_t.fast_info

    vix_price = vix_info.get("lastPrice", 0) or vix_info.get("previousClose", 18)
    vix_prev = vix_info.get("previousClose", vix_price)
    spy_price = spy_info.get("lastPrice", 0) or spy_info.get("previousClose", 0)
    spy_prev = spy_info.get("previousClose", spy_price)
    spy_change = ((spy_price - spy_prev) / spy_prev * 100) if spy_prev else 0

    # SPY 20-day SMA
    spy_hist = spy_t.history(period="1mo")
    spy_sma20 = spy_hist["Close"].tail(20).mean() if len(spy_hist) >= 20 else spy_price

    # Determine regime from config thresholds
    regimes = config.get("strategy", {}).get("regimes", {})
    if vix_price < regimes.get("conservative", {}).get("vix_below", 15):
        regime = "conservative"
    elif vix_price > regimes.get("aggressive", {}).get("vix_above", 25):
        regime = "aggressive"
    else:
        regime = "balanced"

    if not config.get("strategy", {}).get("auto_regime", True):
        regime = config.get("strategy", {}).get("default_regime", "balanced")

    return MarketContext(
        vix=round(vix_price, 2),
        vix_change=round(vix_price - vix_prev, 2),
        spy_price=round(spy_price, 2),
        spy_change_pct=round(spy_change, 2),
        spy_above_sma20=spy_price > spy_sma20,
        regime=regime,
        timestamp=dt.datetime.now().strftime("%Y-%m-%d %H:%M PST"),
    )


def _calc_technicals(hist: pd.DataFrame) -> TechnicalIndicators:
    """Calculate RSI, MACD, Bollinger Bands, ATR from price history."""
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    # RSI-14
    rsi = None
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-10)
        rsi_series = 100 - (100 / (1 + rs))
        rsi = round(rsi_series.iloc[-1], 1)

    # MACD (12, 26, 9)
    macd_val = macd_sig = macd_h = None
    if len(close) >= 35:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_val = round(macd_line.iloc[-1], 3)
        macd_sig = round(signal_line.iloc[-1], 3)
        macd_h = round((macd_line - signal_line).iloc[-1], 3)

    # Bollinger Bands (20, 2)
    bb_up = bb_lo = bb_w = None
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_up = round((sma20 + 2 * std20).iloc[-1], 2)
        bb_lo = round((sma20 - 2 * std20).iloc[-1], 2)
        mid = sma20.iloc[-1]
        bb_w = round((bb_up - bb_lo) / mid * 100, 1) if mid else None

    # ATR-14
    atr = None
    if len(close) >= 15:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = round(tr.rolling(14).mean().iloc[-1], 2)

    return TechnicalIndicators(
        rsi_14=rsi, macd=macd_val, macd_signal=macd_sig, macd_hist=macd_h,
        bb_upper=bb_up, bb_lower=bb_lo, bb_width=bb_w, atr_14=atr,
    )


def fetch_stock(symbol: str) -> StockSnapshot:
    """Fetch current stock data + technical indicators."""
    t = yf.Ticker(symbol)
    info = t.fast_info

    price = info.get("lastPrice", 0) or info.get("previousClose", 0)
    prev = info.get("previousClose", price)
    change = ((price - prev) / prev * 100) if prev else 0

    hist = t.history(period="3mo")
    sma_20 = round(hist["Close"].tail(20).mean(), 2) if len(hist) >= 20 else None
    sma_50 = round(hist["Close"].tail(50).mean(), 2) if len(hist) >= 50 else None
    high_52w = info.get("yearHigh", 0) or 0
    low_52w = info.get("yearLow", 0) or 0
    technicals = _calc_technicals(hist) if len(hist) >= 15 else None

    return StockSnapshot(
        symbol=symbol,
        price=round(price, 2),
        prev_close=round(prev, 2),
        day_change_pct=round(change, 2),
        sma_20=sma_20,
        sma_50=sma_50,
        high_52w=round(high_52w, 2),
        low_52w=round(low_52w, 2),
        market_cap=info.get("marketCap"),
        volume=info.get("lastVolume"),
        technicals=technicals,
    )


def fetch_iv_stats(symbol: str) -> IVStats:
    """Calculate IV rank and percentile from historical option-implied vol."""
    t = yf.Ticker(symbol)
    hist = t.history(period="1y")

    if len(hist) < 20:
        return IVStats(0, 0, 0, 0, 0)

    # Use historical volatility as IV proxy for ranking
    returns = hist["Close"].pct_change().dropna()
    rolling_vol = returns.rolling(20).std() * math.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    if len(rolling_vol) < 5:
        return IVStats(0, 0, 0, 0, 0)

    current = rolling_vol.iloc[-1]
    high = rolling_vol.max()
    low = rolling_vol.min()
    iv_range = high - low

    rank = ((current - low) / iv_range * 100) if iv_range > 0 else 50
    percentile = (rolling_vol < current).mean() * 100

    return IVStats(
        current_iv=round(current, 4),
        iv_high_52w=round(high, 4),
        iv_low_52w=round(low, 4),
        iv_rank=round(rank, 1),
        iv_percentile=round(percentile, 1),
    )


def fetch_events(symbol: str) -> EventInfo:
    """Fetch next earnings date and ex-dividend date."""
    t = yf.Ticker(symbol)
    today = dt.date.today()

    earnings_date = None
    days_to_earn = None
    ex_div = None
    days_to_div = None

    try:
        cal = t.calendar
        if cal is not None:
            if isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:
                    ed = cal.loc["Earnings Date"].iloc[0]
                    if hasattr(ed, "date"):
                        ed = ed.date()
                    elif isinstance(ed, str):
                        ed = dt.datetime.strptime(ed, "%Y-%m-%d").date()
                    earnings_date = str(ed)
                    days_to_earn = (ed - today).days
            elif isinstance(cal, dict):
                eds = cal.get("Earnings Date", [])
                if eds:
                    ed = eds[0]
                    if hasattr(ed, "date"):
                        ed = ed.date()
                    earnings_date = str(ed)
                    days_to_earn = (ed - today).days
                ex = cal.get("Ex-Dividend Date")
                if ex is not None:
                    if hasattr(ex, "date"):
                        ex = ex.date()
                    ex_div = str(ex)
                    days_to_div = (ex - today).days
    except Exception:
        pass

    # Try info dict as fallback for ex-dividend
    if ex_div is None:
        try:
            info = t.info
            ex_ts = info.get("exDividendDate")
            if ex_ts:
                exd = dt.datetime.fromtimestamp(ex_ts).date()
                ex_div = str(exd)
                days_to_div = (exd - today).days
        except Exception:
            pass

    return EventInfo(
        next_earnings=earnings_date,
        days_to_earnings=days_to_earn,
        next_ex_div=ex_div,
        days_to_ex_div=days_to_div,
    )


def fetch_option_chain(
    symbol: str,
    stock_price: float,
    config: dict,
    option_type: str = "call",
) -> list[OptionRow]:
    """Fetch and filter option chain for a symbol."""
    t = yf.Ticker(symbol)
    strat = config.get("strategy", {})
    dte_min, dte_max = strat.get("dte_range", [3, 21])

    try:
        expiries = t.options
    except Exception:
        return []

    today = dt.date.today()
    rows: list[OptionRow] = []

    for exp_str in expiries:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < dte_min or dte > dte_max:
            continue

        try:
            chain = t.option_chain(exp_str)
        except Exception:
            continue

        df = chain.calls if option_type == "call" else chain.puts

        for _, row in df.iterrows():
            strike = _safe_float(row.get("strike"))
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last = _safe_float(row.get("lastPrice"))
            iv = _safe_float(row.get("impliedVolatility"))
            oi = _safe_int(row.get("openInterest"))
            vol = _safe_int(row.get("volume"))

            # Off-hours fallback: use lastPrice when bid/ask are 0
            is_off_hours = (bid == 0 and ask == 0 and last > 0)
            price_ref = last if is_off_hours else bid
            mid = last if is_off_hours else ((bid + ask) / 2 if (bid + ask) > 0 else 0)

            # Data-level filters only: OTM + has *some* price
            if option_type == "call" and strike <= stock_price:
                continue
            if option_type == "put" and strike >= stock_price:
                continue
            if price_ref <= 0:
                continue

            spread_pct = ((ask - bid) / mid * 100) if (mid > 0 and not is_off_hours) else 0

            T = max(dte, 1) / 365.0

            # If IV looks unreliable (< 10% for equities = clearly wrong),
            # solve for real IV from the last traded price
            if iv < 0.10 and last > 0:
                iv = implied_volatility(last, stock_price, strike, T, RISK_FREE_RATE, option_type)

            greeks = black_scholes_greeks(stock_price, strike, T, RISK_FREE_RATE, iv, option_type)

            otm_pct = abs(strike - stock_price) / stock_price * 100
            ann_yield = (price_ref / stock_price) * (365.0 / max(dte, 1)) * 100

            rows.append(OptionRow(
                expiry=exp_str,
                strike=strike,
                bid=round(bid, 2),
                ask=round(ask, 2),
                mid=round(mid, 2),
                last=round(last, 2),
                volume=vol,
                open_interest=oi,
                implied_vol=round(iv * 100, 1),  # as percentage
                greeks=greeks,
                dte=dte,
                annualized_yield=round(ann_yield, 1),
                otm_pct=round(otm_pct, 1),
                spread_pct=round(spread_pct, 1),
                off_hours=is_off_hours,
            ))

    # Sort by expiry then by strike
    rows.sort(key=lambda r: (r.expiry, r.strike))
    return rows


def fetch_symbol_briefing(symbol: str, config: dict, quick: bool = False) -> SymbolBriefing:
    """Briefing for one symbol. quick=True skips news/insider/analyst/puts (~2x faster)."""
    stock = fetch_stock(symbol)
    iv_stats = fetch_iv_stats(symbol)
    events = fetch_events(symbol)
    calls = fetch_option_chain(symbol, stock.price, config, "call")

    if quick:
        return SymbolBriefing(stock=stock, iv_stats=iv_stats, events=events,
                              call_chains=calls)

    from .news import fetch_news, fetch_insider_transactions, fetch_analyst_data
    from .enhanced import (
        fetch_unusual_options_activity,
        fetch_put_call_ratio,
        fetch_sector_performance,
    )
    puts = fetch_option_chain(symbol, stock.price, config, "put")
    news = fetch_news(symbol)
    insider = fetch_insider_transactions(symbol)
    analyst = fetch_analyst_data(symbol)
    unusual = fetch_unusual_options_activity(symbol)
    pc_ratio = fetch_put_call_ratio(symbol)
    sector = fetch_sector_performance()
    return SymbolBriefing(
        stock=stock, iv_stats=iv_stats, events=events,
        call_chains=calls, put_chains=puts,
        news=news, insider=insider, analyst=analyst,
        unusual_activity=unusual, put_call_ratio=pc_ratio, sector=sector,
    )
