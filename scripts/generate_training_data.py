#!/usr/bin/env python3
"""Generate synthetic training data for Stage 2 learner validation.

Replays historical weeks through the scan_engine pipeline, populates
candidate_observations with synthetic data, and labels them with actual
stock prices at expiry.

Usage:
    python scripts/generate_training_data.py --months 12 --symbols AMZN,FTNT
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import uuid

import numpy as np
import yfinance as yf

# Add project root to path
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.backtest import (
    _historical_vol, _generate_synthetic_chain, _build_synthetic_context,
    _nearest_price, _date_index, _fridays_between,
)
from src.scan_engine import (
    build_features, apply_hard_filters, score_candidate,
)
from src.data.events_calendar import assess_event_risk
from src.data.fetcher import EventInfo
from src.config import load_config, get_position, get_delta_range
from src.db import _connect, _compute_utility_label


def generate(symbols: list[str], months: int = 12, config: dict | None = None):
    if config is None:
        config = load_config()

    strat = config.get("strategy", {})
    delta_range = get_delta_range(config, "balanced")
    earnings_buffer = strat.get("blackout_earnings_days", 7)

    bt_end = dt.date.today()
    bt_start = bt_end - dt.timedelta(days=months * 30)
    buf_start = bt_start - dt.timedelta(days=400)

    total_obs = 0
    total_labeled = 0

    for symbol in symbols:
        print(f"\n{'=' * 50}")
        print(f"  {symbol}: fetching {months}mo history...")

        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=buf_start.isoformat(), end=bt_end.isoformat(), auto_adjust=True)

        if hist.empty or len(hist) < 60:
            print(f"  Skipping {symbol}: insufficient data ({len(hist)} rows)")
            continue

        # Build lookups
        date_close = {}
        date_high = {}
        date_low = {}
        for ts in hist.index:
            d = ts.date() if hasattr(ts, "date") else ts
            date_close[d] = float(hist.loc[ts, "Close"])
            date_high[d] = float(hist.loc[ts, "High"])
            date_low[d] = float(hist.loc[ts, "Low"])

        all_dates = sorted(date_close.keys())
        all_closes = np.array([date_close[d] for d in all_dates])
        all_highs = np.array([date_high[d] for d in all_dates])
        all_lows = np.array([date_low[d] for d in all_dates])

        # Earnings dates
        earnings_dates = []
        try:
            ed = ticker.earnings_dates
            if ed is not None and not ed.empty:
                for ts in ed.index:
                    earnings_dates.append(ts.date() if hasattr(ts, "date") else ts)
            earnings_dates = sorted(set(earnings_dates))
        except Exception:
            pass

        position = get_position(config, symbol)
        fridays = _fridays_between(bt_start, bt_end)

        sym_obs = 0
        sym_labeled = 0

        for i in range(len(fridays) - 1):
            sell_friday = fridays[i]
            expiry_friday = fridays[i + 1]

            sell_price = _nearest_price(date_close, sell_friday, all_dates)
            expiry_price = _nearest_price(date_close, expiry_friday, all_dates)
            if sell_price is None or expiry_price is None:
                continue

            idx = _date_index(all_dates, sell_friday)
            if idx is None or idx < 30:
                continue

            prices_up_to = all_closes[:idx + 1]
            highs_up_to = all_highs[:idx + 1]
            lows_up_to = all_lows[:idx + 1]

            hv = _historical_vol(prices_up_to, 20)
            iv_estimate = hv * 1.10
            actual_dte = (expiry_friday - sell_friday).days

            # Generate synthetic option chain
            chain = _generate_synthetic_chain(sell_price, actual_dte, iv_estimate)
            if not chain:
                continue

            # Build context
            stock, iv_stats, events = _build_synthetic_context(
                symbol, sell_price, prices_up_to, highs_up_to, lows_up_to,
                sell_friday, earnings_dates,
            )

            # Run through pipeline
            features_list = [
                build_features(opt, stock, iv_stats, events, position)
                for opt in chain
            ]

            passed = []
            rejected = []
            for feat in features_list:
                reasons = apply_hard_filters(feat, config)
                if reasons:
                    rejected.append((feat, reasons))
                else:
                    passed.append(feat)

            if not passed:
                continue

            # Peer stats for scoring
            all_yields = [f.annualized_yield for f in passed]
            all_thetas = [f.theta for f in passed]
            all_ois = [f.open_interest for f in passed]

            # Generate scan_id and timestamp for this simulated scan
            scan_id = uuid.uuid4().hex[:12]
            scan_ts = dt.datetime.combine(sell_friday, dt.time(10, 0)).isoformat()

            # Determine VIX estimate from historical vol
            vix_estimate = hv * 100  # rough proxy

            # Build observation rows
            obs_rows = []

            for feat in passed:
                event_risk = assess_event_risk(
                    symbol, feat.expiry,
                    earnings_dte=events.days_to_earnings,
                    earnings_buffer=earnings_buffer,
                )
                score, breakdown, flags = score_candidate(
                    feat, config, event_risk, all_yields, all_thetas, all_ois,
                    delta_range=delta_range,
                )

                # Compute label: did this candidate expire OTM?
                was_assigned = expiry_price > feat.strike
                entry_premium = feat.premium
                if was_assigned:
                    pnl_per_share = entry_premium - (expiry_price - feat.strike)
                else:
                    pnl_per_share = entry_premium

                # Apply counterfactual friction
                pnl_per_share -= entry_premium * 0.03

                utility = _compute_utility_label(
                    entry_premium=entry_premium,
                    terminal_pnl=pnl_per_share,
                    assigned=was_assigned,
                )

                obs_rows.append((
                    scan_ts, scan_id, symbol, feat.expiry, feat.strike,
                    feat.stock_price,
                    json.dumps({
                        "bid": feat.bid, "ask": feat.ask, "premium": feat.premium,
                        "delta": feat.delta, "theta": feat.theta,
                        "implied_vol": feat.implied_vol, "dte": feat.dte,
                        "otm_pct": feat.otm_pct, "spread_pct": feat.spread_pct,
                        "annualized_yield": feat.annualized_yield,
                        "open_interest": feat.open_interest, "volume": feat.volume,
                        "atr_distance": feat.atr_distance, "earnings_gap": feat.earnings_gap,
                        "iv_rank": feat.iv_rank, "cost_basis": feat.cost_basis,
                        "allow_assignment": feat.allow_assignment, "off_hours": False,
                    }),
                    1,  # hard_filter_passed
                    None,  # reject_reasons
                    score,
                    json.dumps(breakdown),
                    json.dumps(flags),
                    "balanced",  # market_regime
                    round(vix_estimate, 1),
                    round(iv_stats.iv_rank, 1),
                    # Structured outcome fields
                    "counterfactual",  # outcome_source
                    round(pnl_per_share * 100, 2),  # terminal_pnl (per-contract)
                    1 if was_assigned else 0,
                    0 if was_assigned else 1,
                    round(expiry_price, 2),  # stock_at_expiry
                    utility,
                    "medium",  # label_confidence
                ))

            # Insert into DB
            conn = _connect()
            try:
                conn.executemany(
                    """INSERT INTO candidate_observations
                       (scan_ts, scan_id, symbol, expiry, strike, stock_price,
                        features, hard_filter_passed, reject_reasons,
                        score, score_breakdown, flags,
                        market_regime, vix, iv_rank,
                        outcome_source, terminal_pnl, assigned, expired_otm,
                        stock_at_expiry, utility_label, label_confidence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                               ?, ?, ?, ?, ?, ?, ?)""",
                    obs_rows,
                )
                conn.commit()
            finally:
                conn.close()

            sym_obs += len(obs_rows)
            sym_labeled += len(obs_rows)

        print(f"  {symbol}: {sym_obs} observations, {sym_labeled} labeled")
        total_obs += sym_obs
        total_labeled += sym_labeled

    print(f"\n{'=' * 50}")
    print(f"  Total: {total_obs} observations, {total_labeled} labeled")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate training data for Stage 2")
    parser.add_argument("--months", type=int, default=12, help="Months of history")
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated symbols (default: from portfolio)")
    args = parser.parse_args()

    config = load_config()
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        from src.config import get_symbols
        symbols = get_symbols(config)

    if not symbols:
        print("No symbols. Add positions or use --symbols AMZN,FTNT")
        sys.exit(1)

    print(f"Generating training data: {', '.join(symbols)}, {args.months} months")
    generate(symbols, months=args.months, config=config)
