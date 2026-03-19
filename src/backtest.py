"""Replay-based backtesting engine for covered call strategies.

Uses yfinance historical prices and Black-Scholes pricing to generate
synthetic option candidates, then runs them through the SAME pipeline
as live scans: build_features → apply_hard_filters → score_candidate.

This ensures backtest validates the same decision logic that runs live.
BS pricing is still used for premium estimation (no historical chains
from yfinance), but candidate selection/scoring matches scan_engine.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import yfinance as yf

from .data.greeks import bs_call_price, black_scholes_greeks, Greeks
from .data.fetcher import (
    OptionRow, StockSnapshot, IVStats, EventInfo, TechnicalIndicators,
)
from .data.events_calendar import assess_event_risk, EventRisk
from .scan_engine import (
    CandidateFeatures, CandidateDecision,
    build_features, apply_hard_filters, score_candidate,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class WeekResult:
    week_start: str          # YYYY-MM-DD (Monday)
    week_end: str            # YYYY-MM-DD (Friday / expiry)
    stock_price_at_sell: float
    strike: float
    dte: int
    delta: float
    premium_per_share: float
    premium_total: float     # premium_per_share * shares
    stock_price_at_expiry: float
    assigned: bool           # price > strike at expiry
    pnl: float              # premium collected (minus assignment loss if any)
    score: float = 0.0      # scan_engine score of selected candidate
    score_breakdown: dict = field(default_factory=dict)
    candidates_passed: int = 0   # how many passed hard filters this week
    candidates_total: int = 0    # total candidates generated


@dataclass
class BacktestResult:
    symbol: str
    shares: int
    start_date: str
    end_date: str
    delta_target: float
    dte_target: int

    total_premium: float = 0.0
    num_trades: int = 0
    num_assigned: int = 0
    win_rate: float = 0.0
    avg_weekly_premium: float = 0.0
    annualized_return: float = 0.0
    max_drawdown: float = 0.0
    stock_return_pct: float = 0.0     # buy-and-hold return over period
    strategy_return_pct: float = 0.0  # covered call total return over period
    # Utility metrics
    avg_score: float = 0.0
    premium_capture_rate: float = 0.0     # avg fraction of premium kept
    assignment_rate: float = 0.0
    called_away_downside: float = 0.0     # total forgone upside from assignments
    weeks_skipped: int = 0                # weeks where no candidate passed filters
    results_by_week: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RISK_FREE_RATE = 0.045


def _historical_vol(prices: np.ndarray, window: int = 20) -> float:
    """Annualized historical volatility from a price array (most recent *window* days)."""
    if len(prices) < window + 1:
        window = max(len(prices) - 1, 1)
    recent = prices[-(window + 1):]
    log_returns = np.diff(np.log(recent))
    if len(log_returns) == 0:
        return 0.30  # fallback
    return float(np.std(log_returns, ddof=1) * math.sqrt(252))


def _compute_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 14) -> float | None:
    """ATR from arrays ending at current day."""
    if len(closes) < period + 1:
        return None
    tr_values = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        tr_values.append(tr)
    return float(np.mean(tr_values))


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """RSI from a price array."""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _fridays_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return all Fridays (inclusive) between start and end."""
    d = start
    while d.weekday() != 4:  # 4 = Friday
        d += dt.timedelta(days=1)
    fridays = []
    while d <= end:
        fridays.append(d)
        d += dt.timedelta(days=7)
    return fridays


def _nearest_price(
    date_price: dict[dt.date, float],
    target: dt.date,
    sorted_dates: list[dt.date],
) -> float | None:
    """Get the price on *target* or the nearest prior trading day (up to 5 days back)."""
    for offset in range(6):
        d = target - dt.timedelta(days=offset)
        if d in date_price:
            return date_price[d]
    for offset in range(1, 4):
        d = target + dt.timedelta(days=offset)
        if d in date_price:
            return date_price[d]
    return None


def _date_index(sorted_dates: list[dt.date], target: dt.date) -> int | None:
    """Find index of target (or nearest prior date) in sorted_dates."""
    best = None
    for i, d in enumerate(sorted_dates):
        if d <= target:
            best = i
        else:
            break
    return best


# ---------------------------------------------------------------------------
# Synthetic option chain generation
# ---------------------------------------------------------------------------

def _generate_synthetic_chain(
    price: float,
    dte: int,
    sigma: float,
    num_strikes: int = 15,
) -> list[OptionRow]:
    """Generate synthetic OTM call option candidates using BS pricing.

    Creates strikes from ~ATM to ~15% OTM, spaced at realistic increments.
    """
    T = dte / 365.0
    if T <= 0:
        return []

    # Determine strike increment based on price level
    if price < 50:
        incr = 0.50
    elif price < 200:
        incr = 1.00
    elif price < 500:
        incr = 2.50
    else:
        incr = 5.00

    # Generate strikes from ATM to ~15% OTM
    base_strike = math.ceil(price / incr) * incr  # first strike at or above price
    strikes = [base_strike + i * incr for i in range(num_strikes)]

    expiry_date = (dt.date.today() + dt.timedelta(days=dte)).isoformat()
    chain: list[OptionRow] = []

    for strike in strikes:
        greeks = black_scholes_greeks(price, strike, T, _RISK_FREE_RATE, sigma)
        premium = bs_call_price(price, strike, T, _RISK_FREE_RATE, sigma)

        if premium < 0.01 or greeks.delta < 0.01:
            continue  # too far OTM to be useful

        # Simulate realistic bid/ask spread (wider for lower OI / higher OTM)
        otm_pct = (strike - price) / price * 100
        spread_factor = 0.02 + 0.005 * otm_pct  # 2-7% spread typical
        half_spread = premium * spread_factor / 2
        bid = max(0.01, premium - half_spread)
        ask = premium + half_spread
        mid = (bid + ask) / 2

        # Synthetic OI: higher near ATM, drops off OTM
        base_oi = 2000
        oi = max(50, int(base_oi * math.exp(-0.3 * otm_pct)))
        volume = max(10, oi // 3)

        spread_pct = (ask - bid) / mid * 100 if mid > 0 else 0
        annualized_yield = (bid / price) * (365 / dte) * 100 if dte > 0 else 0

        chain.append(OptionRow(
            expiry=expiry_date,
            strike=round(strike, 2),
            bid=round(bid, 4),
            ask=round(ask, 4),
            mid=round(mid, 4),
            last=round(premium, 4),
            volume=volume,
            open_interest=oi,
            implied_vol=round(sigma, 4),
            greeks=greeks,
            dte=dte,
            annualized_yield=round(annualized_yield, 2),
            otm_pct=round(otm_pct, 2),
            spread_pct=round(spread_pct, 2),
            off_hours=False,
        ))

    return chain


def _build_synthetic_context(
    symbol: str,
    price: float,
    prices_up_to: np.ndarray,
    highs_up_to: np.ndarray,
    lows_up_to: np.ndarray,
    sell_date: dt.date,
    earnings_dates: list[dt.date],
) -> tuple[StockSnapshot, IVStats, EventInfo]:
    """Build the data objects scan_engine.build_features needs, from historical data."""

    # Technicals
    atr = _compute_atr(highs_up_to, lows_up_to, prices_up_to, 14)
    rsi = _compute_rsi(prices_up_to, 14)

    sma_20 = float(np.mean(prices_up_to[-20:])) if len(prices_up_to) >= 20 else None
    sma_50 = float(np.mean(prices_up_to[-50:])) if len(prices_up_to) >= 50 else None

    technicals = TechnicalIndicators(
        rsi_14=rsi,
        macd=None, macd_signal=None, macd_hist=None,
        bb_upper=None, bb_lower=None, bb_width=None,
        atr_14=atr,
    )

    prev_close = float(prices_up_to[-2]) if len(prices_up_to) >= 2 else price
    high_52w = float(np.max(prices_up_to[-252:])) if len(prices_up_to) >= 252 else float(np.max(prices_up_to))
    low_52w = float(np.min(prices_up_to[-252:])) if len(prices_up_to) >= 252 else float(np.min(prices_up_to))

    stock = StockSnapshot(
        symbol=symbol,
        price=price,
        prev_close=prev_close,
        day_change_pct=(price - prev_close) / prev_close * 100 if prev_close else 0,
        sma_20=sma_20,
        sma_50=sma_50,
        high_52w=high_52w,
        low_52w=low_52w,
        market_cap=None,
        volume=None,
        technicals=technicals,
    )

    # IV stats from historical vol
    hv20 = _historical_vol(prices_up_to, 20)
    hv_series = []
    for i in range(max(20, len(prices_up_to) - 252), len(prices_up_to)):
        if i >= 20:
            hv_series.append(_historical_vol(prices_up_to[:i + 1], 20))
    if hv_series:
        iv_high = max(hv_series)
        iv_low = min(hv_series)
        iv_range = iv_high - iv_low
        iv_rank = ((hv20 - iv_low) / iv_range * 100) if iv_range > 0 else 50.0
        iv_pctile = sum(1 for h in hv_series if h < hv20) / len(hv_series) * 100
    else:
        iv_high, iv_low, iv_rank, iv_pctile = hv20, hv20, 50.0, 50.0

    iv_stats = IVStats(
        current_iv=hv20 * 1.10,  # IV premium over realized
        iv_high_52w=iv_high * 1.10,
        iv_low_52w=iv_low * 1.10,
        iv_rank=iv_rank,
        iv_percentile=iv_pctile,
    )

    # Earnings proximity
    days_to_earnings = None
    next_earnings_str = None
    for ed in earnings_dates:
        if ed >= sell_date:
            days_to_earnings = (ed - sell_date).days
            next_earnings_str = ed.isoformat()
            break

    events = EventInfo(
        next_earnings=next_earnings_str,
        days_to_earnings=days_to_earnings,
        next_ex_div=None,
        days_to_ex_div=None,
    )

    return stock, iv_stats, events


# ---------------------------------------------------------------------------
# Main replay engine
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str,
    shares: int,
    start_date: str,
    end_date: str,
    delta_target: float = 0.20,
    dte_target: int = 7,
    config: dict | None = None,
) -> BacktestResult:
    """Run a covered call backtest using the scan_engine pipeline.

    For each week:
    1. Generate synthetic option chain (BS-priced)
    2. Run through build_features → apply_hard_filters → score_candidate
    3. Select the top-scoring candidate
    4. Evaluate at expiry with actual stock price
    """
    from .config import load_config, get_delta_range

    if config is None:
        config = load_config()

    contracts = shares // 100
    if contracts <= 0:
        raise ValueError(f"Need at least 100 shares for covered calls, got {shares}")

    # Fetch historical data (add buffer for HV + indicator calculation)
    buf_start = (dt.date.fromisoformat(start_date) - dt.timedelta(days=400)).isoformat()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=buf_start, end=end_date, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No historical data for {symbol} between {start_date} and {end_date}")

    # Build date -> OHLC lookups
    date_close: dict[dt.date, float] = {}
    date_high: dict[dt.date, float] = {}
    date_low: dict[dt.date, float] = {}
    for ts in hist.index:
        d = ts.date() if hasattr(ts, 'date') else ts
        date_close[d] = float(hist.loc[ts, "Close"])
        date_high[d] = float(hist.loc[ts, "High"])
        date_low[d] = float(hist.loc[ts, "Low"])

    all_dates_sorted = sorted(date_close.keys())
    all_closes = np.array([date_close[d] for d in all_dates_sorted])
    all_highs = np.array([date_high[d] for d in all_dates_sorted])
    all_lows = np.array([date_low[d] for d in all_dates_sorted])

    # Fetch earnings dates for event risk context
    earnings_dates: list[dt.date] = []
    try:
        ed = ticker.earnings_dates
        if ed is not None and not ed.empty:
            for ts in ed.index:
                earnings_dates.append(ts.date() if hasattr(ts, "date") else ts)
        earnings_dates = sorted(set(earnings_dates))
    except Exception:
        pass

    # Determine trading Fridays in the backtest window
    bt_start = dt.date.fromisoformat(start_date)
    bt_end = dt.date.fromisoformat(end_date)
    fridays = _fridays_between(bt_start, bt_end)

    if len(fridays) < 2:
        raise ValueError("Not enough trading weeks in the specified range")

    # Position context for build_features
    position = {
        "cost_basis": 0,
        "allow_assignment": False,
    }

    # Determine regime for delta range (use static config-based heuristic)
    # In backtest we don't have live VIX per-week, so use a default
    regime = "balanced"
    delta_range = get_delta_range(config, regime)
    strat = config.get("strategy", {})
    earnings_buffer = strat.get("blackout_earnings_days", 7)

    results: list[WeekResult] = []
    weeks_skipped = 0
    total_called_away_loss = 0.0

    for i in range(len(fridays) - 1):
        sell_friday = fridays[i]
        expiry_friday = fridays[i + 1]

        # Handle DTE targets > 7
        if dte_target > 7:
            skip = (dte_target - 1) // 7
            if i % (skip + 1) != 0 and skip > 0:
                continue
            expiry_friday = fridays[min(i + skip + 1, len(fridays) - 1)]

        sell_price = _nearest_price(date_close, sell_friday, all_dates_sorted)
        expiry_price = _nearest_price(date_close, expiry_friday, all_dates_sorted)

        if sell_price is None or expiry_price is None:
            continue

        idx = _date_index(all_dates_sorted, sell_friday)
        if idx is None or idx < 30:
            continue

        prices_up_to = all_closes[:idx + 1]
        highs_up_to = all_highs[:idx + 1]
        lows_up_to = all_lows[:idx + 1]

        # Historical vol for BS pricing
        hv = _historical_vol(prices_up_to, 20)
        iv_estimate = hv * 1.10

        actual_dte = (expiry_friday - sell_friday).days

        # 1. Generate synthetic option chain
        chain = _generate_synthetic_chain(sell_price, actual_dte, iv_estimate)
        if not chain:
            weeks_skipped += 1
            continue

        # 2. Build context objects
        stock, iv_stats, events = _build_synthetic_context(
            symbol, sell_price, prices_up_to, highs_up_to, lows_up_to,
            sell_friday, earnings_dates,
        )

        # 3. Run through scan_engine pipeline: features → filter → score
        features_list = [
            build_features(opt, stock, iv_stats, events, position)
            for opt in chain
        ]

        passed: list[CandidateFeatures] = []
        for feat in features_list:
            reasons = apply_hard_filters(feat, config)
            if not reasons:
                passed.append(feat)

        candidates_total = len(features_list)

        if not passed:
            weeks_skipped += 1
            continue

        # Peer stats for normalization
        all_yields = [f.annualized_yield for f in passed]
        all_thetas = [f.theta for f in passed]
        all_ois = [f.open_interest for f in passed]

        # Score each passed candidate
        scored: list[tuple[CandidateFeatures, float, dict]] = []
        for feat in passed:
            event_risk = assess_event_risk(
                symbol, feat.expiry,
                earnings_dte=events.days_to_earnings,
                earnings_buffer=earnings_buffer,
            )
            sc, breakdown, flags = score_candidate(
                feat, config, event_risk, all_yields, all_thetas, all_ois,
                delta_range=delta_range,
            )
            scored.append((feat, sc, breakdown))

        # 4. Select top-scoring candidate
        scored.sort(key=lambda x: x[1], reverse=True)
        best_feat, best_score, best_breakdown = scored[0]

        # 5. Evaluate at expiry
        assigned = expiry_price > best_feat.strike
        premium_per_share = best_feat.premium
        premium_total = premium_per_share * contracts * 100

        if assigned:
            option_pnl = (premium_per_share - (expiry_price - best_feat.strike)) * contracts * 100
            total_called_away_loss += (expiry_price - best_feat.strike) * contracts * 100
        else:
            option_pnl = premium_total

        week_result = WeekResult(
            week_start=sell_friday.isoformat(),
            week_end=expiry_friday.isoformat(),
            stock_price_at_sell=round(sell_price, 2),
            strike=round(best_feat.strike, 2),
            dte=actual_dte,
            delta=round(best_feat.delta, 4),
            premium_per_share=round(premium_per_share, 4),
            premium_total=round(premium_total, 2),
            stock_price_at_expiry=round(expiry_price, 2),
            assigned=assigned,
            pnl=round(option_pnl, 2),
            score=round(best_score, 4),
            score_breakdown=best_breakdown,
            candidates_passed=len(passed),
            candidates_total=candidates_total,
        )
        results.append(week_result)

    # Aggregate
    total_premium = sum(r.premium_total for r in results)
    num_trades = len(results)
    num_assigned = sum(1 for r in results if r.assigned)
    win_rate = (num_trades - num_assigned) / num_trades * 100 if num_trades else 0
    avg_weekly_premium = total_premium / num_trades if num_trades else 0

    # Utility metrics
    avg_score = sum(r.score for r in results) / num_trades if num_trades else 0
    assignment_rate = num_assigned / num_trades * 100 if num_trades else 0

    # Premium capture rate: fraction of collected premium that was actually kept
    total_collected = sum(r.premium_total for r in results)
    total_pnl = sum(r.pnl for r in results)
    premium_capture_rate = total_pnl / total_collected * 100 if total_collected > 0 else 0

    # Stock buy-and-hold return
    first_price = _nearest_price(date_close, bt_start, all_dates_sorted) or all_closes[0]
    last_price = _nearest_price(date_close, bt_end, all_dates_sorted) or all_closes[-1]
    stock_return_pct = (last_price - first_price) / first_price * 100

    # Position value for return calculation
    position_value = first_price * shares
    strategy_return_pct = stock_return_pct + (total_pnl / position_value * 100)

    # Annualized return
    days_in_period = (bt_end - bt_start).days
    if days_in_period > 0 and position_value > 0:
        total_return_frac = strategy_return_pct / 100
        years = days_in_period / 365.0
        if total_return_frac > -1.0:
            annualized_return = ((1 + total_return_frac) ** (1 / years) - 1) * 100
        else:
            annualized_return = -100.0
    else:
        annualized_return = 0.0

    max_drawdown = _compute_max_drawdown(results, position_value)

    return BacktestResult(
        symbol=symbol,
        shares=shares,
        start_date=start_date,
        end_date=end_date,
        delta_target=delta_target,
        dte_target=dte_target,
        total_premium=round(total_premium, 2),
        num_trades=num_trades,
        num_assigned=num_assigned,
        win_rate=round(win_rate, 1),
        avg_weekly_premium=round(avg_weekly_premium, 2),
        annualized_return=round(annualized_return, 2),
        max_drawdown=round(max_drawdown, 2),
        stock_return_pct=round(stock_return_pct, 2),
        strategy_return_pct=round(strategy_return_pct, 2),
        avg_score=round(avg_score, 4),
        premium_capture_rate=round(premium_capture_rate, 1),
        assignment_rate=round(assignment_rate, 1),
        called_away_downside=round(total_called_away_loss, 2),
        weeks_skipped=weeks_skipped,
        results_by_week=[_week_to_dict(r) for r in results],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_max_drawdown(results: list[WeekResult], position_value: float) -> float:
    """Max drawdown of cumulative strategy P&L as percentage of initial position value."""
    if not results or position_value <= 0:
        return 0.0

    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0

    for r in results:
        cumulative += r.pnl
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    return max_dd / position_value * 100


def _week_to_dict(r: WeekResult) -> dict:
    return {
        "week_start": r.week_start,
        "week_end": r.week_end,
        "stock_price": r.stock_price_at_sell,
        "strike": r.strike,
        "dte": r.dte,
        "delta": r.delta,
        "premium_per_share": r.premium_per_share,
        "premium_total": r.premium_total,
        "expiry_price": r.stock_price_at_expiry,
        "assigned": r.assigned,
        "pnl": r.pnl,
        "score": r.score,
        "candidates_passed": r.candidates_passed,
    }


# ---------------------------------------------------------------------------
# Formatting for CLI output
# ---------------------------------------------------------------------------

def format_summary(result: BacktestResult) -> str:
    """Format a BacktestResult as a readable summary table."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  COVERED CALL BACKTEST (REPLAY): {result.symbol}")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Period:          {result.start_date} to {result.end_date}")
    lines.append(f"  Shares:          {result.shares:,}  ({result.shares // 100} contracts/week)")
    lines.append(f"  Target Delta:    {result.delta_target:.0%}")
    lines.append(f"  Target DTE:      {result.dte_target} days")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Total Trades:    {result.num_trades}")
    lines.append(f"  Assigned:        {result.num_assigned}  ({result.assignment_rate:.1f}%)")
    lines.append(f"  Expired OTM:     {result.num_trades - result.num_assigned}  ({result.win_rate:.1f}%)")
    lines.append(f"  Weeks Skipped:   {result.weeks_skipped}  (no candidate passed filters)")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Total Premium:   ${result.total_premium:>12,.2f}")
    lines.append(f"  Avg Weekly Prem: ${result.avg_weekly_premium:>12,.2f}")
    lines.append(f"  Premium Capture: {result.premium_capture_rate:>10.1f}%")
    lines.append(f"  Called-Away Loss:${result.called_away_downside:>12,.2f}")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Stock Return:    {result.stock_return_pct:>+10.2f}%  (buy & hold)")
    lines.append(f"  Strategy Return: {result.strategy_return_pct:>+10.2f}%  (covered call)")
    lines.append(f"  Annualized:      {result.annualized_return:>+10.2f}%")
    lines.append(f"  Max Drawdown:    {result.max_drawdown:>10.2f}%  (option overlay)")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Avg Score:       {result.avg_score:>10.4f}  (scan_engine composite)")
    lines.append(f"{'=' * 70}\n")
    return "\n".join(lines)


def format_weekly_detail(result: BacktestResult) -> str:
    """Format week-by-week results as a table."""
    lines = []
    lines.append(
        f"{'Week':<12} {'Price':>8} {'Strike':>8} {'Delta':>6} "
        f"{'Premium':>10} {'Expiry':>8} {'Asgn':>5} {'P&L':>10} {'Score':>6}"
    )
    lines.append("─" * 88)

    for w in result.results_by_week:
        assigned_str = "YES" if w["assigned"] else ""
        lines.append(
            f"{w['week_start']:<12} "
            f"${w['stock_price']:>7.2f} "
            f"${w['strike']:>7.2f} "
            f"{w['delta']:>5.2f} "
            f"${w['premium_total']:>9,.2f} "
            f"${w['expiry_price']:>7.2f} "
            f"{assigned_str:>5} "
            f"${w['pnl']:>9,.2f} "
            f"{w['score']:>5.3f}"
        )

    lines.append("─" * 88)
    return "\n".join(lines)
