"""Backtesting engine — simulate covered call selling over historical data.

Uses yfinance historical prices and Black-Scholes pricing (from src.data.greeks)
to estimate option premiums. Since historical option chains are not available via
yfinance, we use 20-day historical volatility as an IV proxy and price calls
using BS formula.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import yfinance as yf

from .data.greeks import bs_call_price, black_scholes_greeks


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
    results_by_week: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RISK_FREE_RATE = 0.045  # approximate; not critical for short-dated options


def _historical_vol(prices: np.ndarray, window: int = 20) -> float:
    """Annualized historical volatility from a price array (most recent *window* days)."""
    if len(prices) < window + 1:
        window = max(len(prices) - 1, 1)
    recent = prices[-(window + 1):]
    log_returns = np.diff(np.log(recent))
    if len(log_returns) == 0:
        return 0.30  # fallback
    return float(np.std(log_returns, ddof=1) * math.sqrt(252))


def _find_strike_for_delta(
    S: float,
    T: float,
    sigma: float,
    delta_target: float,
    r: float = _RISK_FREE_RATE,
) -> float:
    """Binary-search for a call strike that yields approximately *delta_target*.

    Returns the strike price rounded to the nearest 0.50.
    """
    # Search between S * 0.90 and S * 1.30
    lo, hi = S * 0.90, S * 1.30

    for _ in range(80):
        mid = (lo + hi) / 2.0
        g = black_scholes_greeks(S, mid, T, r, sigma, option_type="call")
        if g.delta > delta_target:
            lo = mid
        else:
            hi = mid

    raw_strike = (lo + hi) / 2.0
    # Round to nearest $0.50 (common strike increment)
    return round(raw_strike * 2) / 2


def _fridays_between(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return all Fridays (inclusive) between start and end."""
    # Find first Friday on or after start
    d = start
    while d.weekday() != 4:  # 4 = Friday
        d += dt.timedelta(days=1)
    fridays = []
    while d <= end:
        fridays.append(d)
        d += dt.timedelta(days=7)
    return fridays


# ---------------------------------------------------------------------------
# Main engine
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
    """Run a covered call backtest simulation.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AMZN").
    shares : int
        Number of shares held (must be multiple of 100 for covered calls).
    start_date, end_date : str
        ISO date strings "YYYY-MM-DD".
    delta_target : float
        Target call delta (e.g. 0.20 for 20-delta OTM calls).
    dte_target : int
        Days to expiration for each simulated call (typically 7 for weeklies).
    config : dict, optional
        Config dict (unused for now, reserved for future strategy params).

    Returns
    -------
    BacktestResult
    """
    contracts = shares // 100
    if contracts <= 0:
        raise ValueError(f"Need at least 100 shares for covered calls, got {shares}")

    # Fetch historical data (add buffer for HV calculation)
    buf_start = (dt.date.fromisoformat(start_date) - dt.timedelta(days=60)).isoformat()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=buf_start, end=end_date, auto_adjust=True)

    if hist.empty:
        raise ValueError(f"No historical data for {symbol} between {start_date} and {end_date}")

    # Build a date -> price lookup (using Close)
    # yfinance index is tz-aware Timestamp; normalize to date
    prices_series = hist["Close"]
    date_price: dict[dt.date, float] = {}
    for ts, price in prices_series.items():
        date_price[ts.date() if hasattr(ts, 'date') else ts] = float(price)

    all_dates_sorted = sorted(date_price.keys())
    all_prices = np.array([date_price[d] for d in all_dates_sorted])

    # Determine trading Fridays in the backtest window
    bt_start = dt.date.fromisoformat(start_date)
    bt_end = dt.date.fromisoformat(end_date)
    fridays = _fridays_between(bt_start, bt_end)

    if len(fridays) < 2:
        raise ValueError("Not enough trading weeks in the specified range")

    results: list[WeekResult] = []

    for i in range(len(fridays) - 1):
        sell_friday = fridays[i]
        expiry_friday = fridays[i + 1]

        # If dte_target != 7 (e.g. 14 = biweekly), skip intermediate Fridays
        # For simplicity, use consecutive Fridays when dte_target <= 7,
        # and skip one for dte_target in (8..14), etc.
        if dte_target > 7:
            skip = (dte_target - 1) // 7
            expiry_idx = i + 1 + (skip - 1) * 1  # approximate
            if expiry_idx >= len(fridays):
                break
            # Only process if this is a "sell" week
            if i % (skip + 1) != 0 and skip > 0:
                continue
            expiry_friday = fridays[min(i + skip + 1, len(fridays) - 1)]

        # Find nearest trading day price for sell_friday
        sell_price = _nearest_price(date_price, sell_friday, all_dates_sorted)
        expiry_price = _nearest_price(date_price, expiry_friday, all_dates_sorted)

        if sell_price is None or expiry_price is None:
            continue

        # Compute historical vol up to sell date
        idx = _date_index(all_dates_sorted, sell_friday)
        if idx is None or idx < 5:
            continue
        hv = _historical_vol(all_prices[:idx + 1], window=20)
        # Add a small vol premium (market IV typically > realized vol)
        iv_estimate = hv * 1.10

        actual_dte = (expiry_friday - sell_friday).days
        T = actual_dte / 365.0

        # Find strike for target delta
        strike = _find_strike_for_delta(sell_price, T, iv_estimate, delta_target)

        # Price the call
        premium_per_share = bs_call_price(sell_price, strike, T, _RISK_FREE_RATE, iv_estimate)

        # Get actual delta for reporting
        greeks = black_scholes_greeks(sell_price, strike, T, _RISK_FREE_RATE, iv_estimate)

        # Premium collected
        premium_total = premium_per_share * contracts * 100

        # Assignment check
        assigned = expiry_price > strike

        # P&L: premium collected minus any assignment loss
        # If assigned: forced to sell at strike, lose upside above strike
        # P&L = premium + min(strike - sell_price, 0) ... no, for covered call:
        # The stock position gains/loses regardless. The call overlay adds:
        #   +premium, and if assigned, caps the upside at strike.
        # For the OPTION OVERLAY P&L specifically:
        #   If not assigned: +premium (call expires worthless)
        #   If assigned: +premium - (expiry_price - strike) [obligation to sell at strike]
        if assigned:
            option_pnl = (premium_per_share - (expiry_price - strike)) * contracts * 100
        else:
            option_pnl = premium_total

        week_result = WeekResult(
            week_start=sell_friday.isoformat(),
            week_end=expiry_friday.isoformat(),
            stock_price_at_sell=round(sell_price, 2),
            strike=round(strike, 2),
            dte=actual_dte,
            delta=round(greeks.delta, 4),
            premium_per_share=round(premium_per_share, 4),
            premium_total=round(premium_total, 2),
            stock_price_at_expiry=round(expiry_price, 2),
            assigned=assigned,
            pnl=round(option_pnl, 2),
        )
        results.append(week_result)

    # Aggregate
    total_premium = sum(r.premium_total for r in results)
    num_trades = len(results)
    num_assigned = sum(1 for r in results if r.assigned)
    win_rate = (num_trades - num_assigned) / num_trades * 100 if num_trades else 0
    avg_weekly_premium = total_premium / num_trades if num_trades else 0

    # Calculate total covered call P&L (option overlay)
    total_option_pnl = sum(r.pnl for r in results)

    # Stock buy-and-hold return
    first_price = _nearest_price(date_price, bt_start, all_dates_sorted) or all_prices[0]
    last_price = _nearest_price(date_price, bt_end, all_dates_sorted) or all_prices[-1]
    stock_return_pct = (last_price - first_price) / first_price * 100

    # Position value for return calculation
    position_value = first_price * shares

    # Strategy total return = stock return + option overlay P&L
    strategy_return_pct = stock_return_pct + (total_option_pnl / position_value * 100)

    # Annualized return
    days_in_period = (bt_end - bt_start).days
    if days_in_period > 0 and position_value > 0:
        total_return_frac = (strategy_return_pct / 100)
        years = days_in_period / 365.0
        if total_return_frac > -1.0:
            annualized_return = ((1 + total_return_frac) ** (1 / years) - 1) * 100
        else:
            annualized_return = -100.0
    else:
        annualized_return = 0.0

    # Max drawdown (on cumulative option P&L curve)
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
        results_by_week=[_week_to_dict(r) for r in results],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    # Try forward too (e.g. Monday after a holiday Friday)
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
    }


# ---------------------------------------------------------------------------
# Formatting for CLI output
# ---------------------------------------------------------------------------

def format_summary(result: BacktestResult) -> str:
    """Format a BacktestResult as a readable summary table."""
    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  COVERED CALL BACKTEST: {result.symbol}")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Period:          {result.start_date} to {result.end_date}")
    lines.append(f"  Shares:          {result.shares:,}  ({result.shares // 100} contracts/week)")
    lines.append(f"  Target Delta:    {result.delta_target:.0%}")
    lines.append(f"  Target DTE:      {result.dte_target} days")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Total Trades:    {result.num_trades}")
    lines.append(f"  Assigned:        {result.num_assigned}  ({100 - result.win_rate:.1f}%)")
    lines.append(f"  Expired OTM:     {result.num_trades - result.num_assigned}  ({result.win_rate:.1f}%)")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Total Premium:   ${result.total_premium:>12,.2f}")
    lines.append(f"  Avg Weekly Prem: ${result.avg_weekly_premium:>12,.2f}")
    lines.append(f"{'─' * 70}")
    lines.append(f"  Stock Return:    {result.stock_return_pct:>+10.2f}%  (buy & hold)")
    lines.append(f"  Strategy Return: {result.strategy_return_pct:>+10.2f}%  (covered call)")
    lines.append(f"  Annualized:      {result.annualized_return:>+10.2f}%")
    lines.append(f"  Max Drawdown:    {result.max_drawdown:>10.2f}%  (option overlay)")
    lines.append(f"{'=' * 70}\n")
    return "\n".join(lines)


def format_weekly_detail(result: BacktestResult) -> str:
    """Format week-by-week results as a table."""
    lines = []
    lines.append(
        f"{'Week':<12} {'Price':>8} {'Strike':>8} {'Delta':>6} "
        f"{'Premium':>10} {'Expiry':>8} {'Assigned':>8} {'P&L':>10}"
    )
    lines.append("─" * 82)

    for w in result.results_by_week:
        assigned_str = "YES" if w["assigned"] else ""
        lines.append(
            f"{w['week_start']:<12} "
            f"${w['stock_price']:>7.2f} "
            f"${w['strike']:>7.2f} "
            f"{w['delta']:>5.2f} "
            f"${w['premium_total']:>9,.2f} "
            f"${w['expiry_price']:>7.2f} "
            f"{assigned_str:>8} "
            f"${w['pnl']:>9,.2f}"
        )

    lines.append("─" * 82)
    return "\n".join(lines)
