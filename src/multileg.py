"""Multi-leg option strategy scanner.

Supported strategies:
  1. Bull Put Spread — sell put, buy lower put
  2. Bear Call Spread — sell call, buy higher call
  3. Iron Condor — bull put spread + bear call spread
  4. Collar — own stock + buy put + sell call
  5. Poor Man's Covered Call — buy deep ITM LEAPS + sell OTM call
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data.fetcher import OptionRow, fetch_option_chain, RISK_FREE_RATE
from .data.greeks import black_scholes_greeks


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpreadCandidate:
    """A single multi-leg strategy candidate with risk/reward metrics."""
    strategy: str
    symbol: str
    legs: list[dict]           # each leg: {side, type, strike, expiry, premium, delta}
    net_premium: float         # positive = credit received
    max_profit: float
    max_loss: float
    breakeven: list[float]     # one or two breakeven prices
    probability_of_profit: float   # estimated from delta
    risk_reward_ratio: float       # max_profit / max_loss
    dte: int
    notes: str = ""


# ---------------------------------------------------------------------------
# Helper: build a relaxed option chain for spread scanning
# ---------------------------------------------------------------------------

def _fetch_chain_for_spreads(
    symbol: str,
    stock_price: float,
    config: dict,
    option_type: str,
    include_itm: bool = False,
    dte_override: tuple[int, int] | None = None,
) -> list[OptionRow]:
    """Fetch option chain with relaxed filters suitable for spread legs.

    The standard fetch_option_chain() only returns OTM options with tight
    liquidity filters.  For spread construction we sometimes need ITM
    options (e.g. for the long leg of a PMCC) and wider DTE windows.
    We call the existing fetcher but patch the config temporarily.
    """
    import copy
    cfg = copy.deepcopy(config)
    strat = cfg.setdefault("strategy", {})

    if dte_override:
        strat["dte_range"] = list(dte_override)

    # Lower minimum premium for protective legs
    strat["min_premium"] = 0.05
    strat["min_open_interest"] = 10

    # Temporarily remove the OTM filter by monkey-patching is not great,
    # so we fetch normally and also grab ITM options when requested.
    otm_rows = fetch_option_chain(symbol, stock_price, cfg, option_type)

    if not include_itm:
        return otm_rows

    # For ITM: flip the OTM filter by pretending stock price is far away
    if option_type == "call":
        itm_rows = fetch_option_chain(symbol, 0.01, cfg, option_type)
        # Keep only strikes below current price (true ITM calls)
        itm_rows = [r for r in itm_rows if r.strike < stock_price]
    else:
        itm_rows = fetch_option_chain(symbol, stock_price * 100, cfg, option_type)
        itm_rows = [r for r in itm_rows if r.strike > stock_price]

    # Recalculate greeks with the real stock price
    for row in itm_rows:
        T = max(row.dte, 1) / 365.0
        iv = row.implied_vol / 100.0  # stored as percentage
        row.greeks = black_scholes_greeks(stock_price, row.strike, T, RISK_FREE_RATE, iv, option_type)
        row.otm_pct = round(abs(row.strike - stock_price) / stock_price * 100, 1)

    # Merge and deduplicate by (expiry, strike)
    seen = {(r.expiry, r.strike) for r in otm_rows}
    combined = list(otm_rows)
    for r in itm_rows:
        if (r.expiry, r.strike) not in seen:
            combined.append(r)
            seen.add((r.expiry, r.strike))

    combined.sort(key=lambda r: (r.expiry, r.strike))
    return combined


def _mid_price(row: OptionRow) -> float:
    """Best estimate of fair value."""
    if row.mid > 0:
        return row.mid
    return row.last


def _make_leg(side: str, opt_type: str, row: OptionRow) -> dict:
    return {
        "side": side,
        "type": opt_type,
        "strike": row.strike,
        "expiry": row.expiry,
        "premium": _mid_price(row),
        "delta": round(row.greeks.delta, 4),
        "bid": row.bid,
        "ask": row.ask,
    }


# ---------------------------------------------------------------------------
# Strategy scanners
# ---------------------------------------------------------------------------

def scan_bull_put_spreads(
    symbol: str,
    stock_price: float,
    put_chain: list[OptionRow],
    config: dict,
) -> list[SpreadCandidate]:
    """Scan for bull put spreads (credit put spreads).

    Structure: sell higher put, buy lower put (both OTM).
    Profit when stock stays above short put strike.
    """
    strat = config.get("strategy", {})
    min_credit = strat.get("min_premium", 0.25)

    # Group by expiry
    by_expiry: dict[str, list[OptionRow]] = {}
    for row in put_chain:
        by_expiry.setdefault(row.expiry, []).append(row)

    candidates: list[SpreadCandidate] = []

    for expiry, puts in by_expiry.items():
        puts_sorted = sorted(puts, key=lambda r: r.strike, reverse=True)

        for i, short_put in enumerate(puts_sorted):
            short_delta = abs(short_put.greeks.delta)
            # Sell puts with delta between 0.15 and 0.40
            if short_delta < 0.10 or short_delta > 0.45:
                continue

            for long_put in puts_sorted[i + 1:]:
                if long_put.strike >= short_put.strike:
                    continue

                width = short_put.strike - long_put.strike
                if width <= 0:
                    continue

                credit = _mid_price(short_put) - _mid_price(long_put)
                if credit < min_credit:
                    continue

                max_profit = credit * 100  # per contract
                max_loss = (width - credit) * 100
                if max_loss <= 0:
                    continue

                breakeven_price = short_put.strike - credit
                # P(profit) ~ 1 - |delta of short put|
                pop = round((1 - short_delta) * 100, 1)
                rr = round(max_profit / max_loss, 3)

                candidates.append(SpreadCandidate(
                    strategy="Bull Put Spread",
                    symbol=symbol,
                    legs=[
                        _make_leg("SELL", "put", short_put),
                        _make_leg("BUY", "put", long_put),
                    ],
                    net_premium=round(credit, 2),
                    max_profit=round(max_profit, 2),
                    max_loss=round(max_loss, 2),
                    breakeven=[round(breakeven_price, 2)],
                    probability_of_profit=pop,
                    risk_reward_ratio=rr,
                    dte=short_put.dte,
                    notes=f"Width ${width:.0f}",
                ))

    # Sort by risk-reward, then by probability of profit
    candidates.sort(key=lambda c: (c.risk_reward_ratio, c.probability_of_profit), reverse=True)
    return candidates[:20]


def scan_bear_call_spreads(
    symbol: str,
    stock_price: float,
    call_chain: list[OptionRow],
    config: dict,
) -> list[SpreadCandidate]:
    """Scan for bear call spreads (credit call spreads).

    Structure: sell lower call, buy higher call (both OTM).
    Profit when stock stays below short call strike.
    """
    strat = config.get("strategy", {})
    min_credit = strat.get("min_premium", 0.25)

    by_expiry: dict[str, list[OptionRow]] = {}
    for row in call_chain:
        by_expiry.setdefault(row.expiry, []).append(row)

    candidates: list[SpreadCandidate] = []

    for expiry, calls in by_expiry.items():
        calls_sorted = sorted(calls, key=lambda r: r.strike)

        for i, short_call in enumerate(calls_sorted):
            short_delta = abs(short_call.greeks.delta)
            if short_delta < 0.10 or short_delta > 0.45:
                continue

            for long_call in calls_sorted[i + 1:]:
                if long_call.strike <= short_call.strike:
                    continue

                width = long_call.strike - short_call.strike
                if width <= 0:
                    continue

                credit = _mid_price(short_call) - _mid_price(long_call)
                if credit < min_credit:
                    continue

                max_profit = credit * 100
                max_loss = (width - credit) * 100
                if max_loss <= 0:
                    continue

                breakeven_price = short_call.strike + credit
                pop = round((1 - short_delta) * 100, 1)
                rr = round(max_profit / max_loss, 3)

                candidates.append(SpreadCandidate(
                    strategy="Bear Call Spread",
                    symbol=symbol,
                    legs=[
                        _make_leg("SELL", "call", short_call),
                        _make_leg("BUY", "call", long_call),
                    ],
                    net_premium=round(credit, 2),
                    max_profit=round(max_profit, 2),
                    max_loss=round(max_loss, 2),
                    breakeven=[round(breakeven_price, 2)],
                    probability_of_profit=pop,
                    risk_reward_ratio=rr,
                    dte=short_call.dte,
                    notes=f"Width ${width:.0f}",
                ))

    candidates.sort(key=lambda c: (c.risk_reward_ratio, c.probability_of_profit), reverse=True)
    return candidates[:20]


def scan_iron_condors(
    symbol: str,
    stock_price: float,
    call_chain: list[OptionRow],
    put_chain: list[OptionRow],
    config: dict,
) -> list[SpreadCandidate]:
    """Scan for iron condors (bull put spread + bear call spread).

    Profit when stock stays in a range between the short strikes.
    """
    bull_puts = scan_bull_put_spreads(symbol, stock_price, put_chain, config)
    bear_calls = scan_bear_call_spreads(symbol, stock_price, call_chain, config)

    candidates: list[SpreadCandidate] = []

    for bp in bull_puts[:10]:
        for bc in bear_calls[:10]:
            # Must be same expiry
            if bp.legs[0]["expiry"] != bc.legs[0]["expiry"]:
                continue

            # Put spread short strike must be below call spread short strike
            put_short_strike = bp.legs[0]["strike"]
            call_short_strike = bc.legs[0]["strike"]
            if put_short_strike >= call_short_strike:
                continue

            total_credit = bp.net_premium + bc.net_premium
            put_width = bp.legs[0]["strike"] - bp.legs[1]["strike"]
            call_width = bc.legs[1]["strike"] - bc.legs[0]["strike"]
            # Max loss = wider spread width - total credit
            max_width = max(put_width, call_width)
            max_loss = (max_width - total_credit) * 100
            if max_loss <= 0:
                continue

            max_profit = total_credit * 100

            lower_be = put_short_strike - total_credit
            upper_be = call_short_strike + total_credit

            # POP ~ prob stock stays between short strikes
            put_delta = abs(bp.legs[0]["delta"])
            call_delta = abs(bc.legs[0]["delta"])
            pop = round((1 - put_delta - call_delta) * 100, 1)
            pop = max(pop, 0)

            rr = round(max_profit / max_loss, 3) if max_loss > 0 else 0

            all_legs = bp.legs + bc.legs

            candidates.append(SpreadCandidate(
                strategy="Iron Condor",
                symbol=symbol,
                legs=all_legs,
                net_premium=round(total_credit, 2),
                max_profit=round(max_profit, 2),
                max_loss=round(max_loss, 2),
                breakeven=[round(lower_be, 2), round(upper_be, 2)],
                probability_of_profit=pop,
                risk_reward_ratio=rr,
                dte=bp.dte,
                notes=f"Range ${put_short_strike:.0f}-${call_short_strike:.0f}",
            ))

    candidates.sort(key=lambda c: (c.probability_of_profit, c.risk_reward_ratio), reverse=True)
    return candidates[:15]


def scan_collars(
    symbol: str,
    stock_price: float,
    call_chain: list[OptionRow],
    put_chain: list[OptionRow],
    config: dict,
) -> list[SpreadCandidate]:
    """Scan for collar strategies (own stock + buy put + sell call).

    Protects downside with the put, caps upside with the call.
    Net cost = put premium paid - call premium received.
    """
    by_expiry_calls: dict[str, list[OptionRow]] = {}
    for row in call_chain:
        by_expiry_calls.setdefault(row.expiry, []).append(row)

    by_expiry_puts: dict[str, list[OptionRow]] = {}
    for row in put_chain:
        by_expiry_puts.setdefault(row.expiry, []).append(row)

    candidates: list[SpreadCandidate] = []

    for expiry in set(by_expiry_calls) & set(by_expiry_puts):
        calls = sorted(by_expiry_calls[expiry], key=lambda r: r.strike)
        puts = sorted(by_expiry_puts[expiry], key=lambda r: r.strike, reverse=True)

        for short_call in calls:
            call_delta = abs(short_call.greeks.delta)
            if call_delta < 0.15 or call_delta > 0.40:
                continue

            for long_put in puts:
                put_delta = abs(long_put.greeks.delta)
                if put_delta < 0.15 or put_delta > 0.40:
                    continue
                if long_put.strike >= short_call.strike:
                    continue

                call_credit = _mid_price(short_call)
                put_cost = _mid_price(long_put)
                net_debit = put_cost - call_credit  # positive = cost, negative = credit

                # Max profit: stock goes to call strike
                max_profit_per_share = (short_call.strike - stock_price) - net_debit
                # Max loss: stock goes to put strike
                max_loss_per_share = (stock_price - long_put.strike) + net_debit

                max_profit = max_profit_per_share * 100
                max_loss = max_loss_per_share * 100

                if max_loss <= 0 or max_profit <= 0:
                    continue

                rr = round(max_profit / max_loss, 3)

                # Breakeven = stock_price + net_debit
                be = stock_price + net_debit

                # POP is less intuitive for collars; approximate as prob above breakeven
                # Using short call delta as rough guide
                pop = round((1 - abs(long_put.greeks.delta)) * 100, 1)

                candidates.append(SpreadCandidate(
                    strategy="Collar",
                    symbol=symbol,
                    legs=[
                        {"side": "HOLD", "type": "stock", "strike": stock_price,
                         "expiry": expiry, "premium": 0, "delta": 1.0,
                         "bid": 0, "ask": 0},
                        _make_leg("BUY", "put", long_put),
                        _make_leg("SELL", "call", short_call),
                    ],
                    net_premium=round(-net_debit, 2),
                    max_profit=round(max_profit, 2),
                    max_loss=round(max_loss, 2),
                    breakeven=[round(be, 2)],
                    probability_of_profit=pop,
                    risk_reward_ratio=rr,
                    dte=short_call.dte,
                    notes=f"Net {'credit' if net_debit < 0 else 'debit'} ${abs(net_debit):.2f}",
                ))

    candidates.sort(key=lambda c: c.risk_reward_ratio, reverse=True)
    return candidates[:15]


def scan_pmcc(
    symbol: str,
    stock_price: float,
    config: dict,
) -> list[SpreadCandidate]:
    """Scan for Poor Man's Covered Call (diagonal spread).

    Structure: buy deep ITM LEAPS call + sell short-term OTM call.
    The LEAPS acts as a stock substitute at lower capital cost.
    """
    # Short-term OTM calls (near-term expiry)
    short_calls = _fetch_chain_for_spreads(symbol, stock_price, config, "call")

    # LEAPS: deep ITM calls, 90-365 DTE
    leaps = _fetch_chain_for_spreads(
        symbol, stock_price, config, "call",
        include_itm=True, dte_override=(90, 365),
    )
    leaps_itm = [r for r in leaps if r.strike < stock_price and abs(r.greeks.delta) >= 0.70]

    if not leaps_itm or not short_calls:
        return []

    candidates: list[SpreadCandidate] = []

    for leap in leaps_itm[:8]:  # limit to top 8 LEAPS to avoid combinatorial explosion
        leap_cost = _mid_price(leap)

        for short in short_calls:
            if short.strike <= stock_price:
                continue
            if short.expiry >= leap.expiry:
                continue

            short_credit = _mid_price(short)
            net_debit = leap_cost - short_credit

            if net_debit <= 0:
                continue

            # Max profit on the short call cycle:
            # short call expires worthless => keep credit
            # Best case if stock at short strike at short expiry:
            #   gain on LEAPS ~ (short_strike - leap_strike) minus time value lost
            #   This is approximate since LEAPS still has time value
            intrinsic_leap = max(stock_price - leap.strike, 0)
            extrinsic_leap = leap_cost - intrinsic_leap

            # Approximate max profit for the short call cycle
            max_profit_approx = short_credit * 100

            # Max loss: LEAPS loses all extrinsic + stock drops to leap strike
            # Simplified: net debit is the total at risk
            max_loss = net_debit * 100

            if max_loss <= 0:
                continue

            rr = round(max_profit_approx / max_loss, 3)
            breakeven_price = leap.strike + net_debit

            # POP ~ probability short call expires OTM
            short_delta = abs(short.greeks.delta)
            pop = round((1 - short_delta) * 100, 1)

            candidates.append(SpreadCandidate(
                strategy="Poor Man's Covered Call",
                symbol=symbol,
                legs=[
                    _make_leg("BUY", "call (LEAPS)", leap),
                    _make_leg("SELL", "call", short),
                ],
                net_premium=round(-net_debit, 2),
                max_profit=round(max_profit_approx, 2),
                max_loss=round(max_loss, 2),
                breakeven=[round(breakeven_price, 2)],
                probability_of_profit=pop,
                risk_reward_ratio=rr,
                dte=short.dte,
                notes=(
                    f"LEAPS {leap.expiry} ${leap.strike:.0f}C "
                    f"(delta {abs(leap.greeks.delta):.2f}, {leap.dte} DTE)"
                ),
            ))

    candidates.sort(key=lambda c: (c.risk_reward_ratio, c.probability_of_profit), reverse=True)
    return candidates[:15]


# ---------------------------------------------------------------------------
# Unified scanner
# ---------------------------------------------------------------------------

STRATEGY_MAP = {
    "bull-put": "Bull Put Spread",
    "bear-call": "Bear Call Spread",
    "iron-condor": "Iron Condor",
    "collar": "Collar",
    "pmcc": "Poor Man's Covered Call",
}


def scan_strategies(
    symbol: str,
    stock_price: float,
    config: dict,
    strategy: str | None = None,
) -> dict[str, list[SpreadCandidate]]:
    """Scan one or all strategies for a symbol.

    Args:
        strategy: one of the STRATEGY_MAP keys, or None for all.

    Returns:
        dict mapping strategy display name to list of candidates.
    """
    results: dict[str, list[SpreadCandidate]] = {}

    strategies = [strategy] if strategy else list(STRATEGY_MAP.keys())

    # Fetch chains once and reuse
    call_chain = None
    put_chain = None

    need_calls = any(s in ("bear-call", "iron-condor", "collar", "pmcc") for s in strategies)
    need_puts = any(s in ("bull-put", "iron-condor", "collar") for s in strategies)

    if need_calls:
        call_chain = _fetch_chain_for_spreads(symbol, stock_price, config, "call")
    if need_puts:
        put_chain = _fetch_chain_for_spreads(symbol, stock_price, config, "put")

    for strat_key in strategies:
        name = STRATEGY_MAP.get(strat_key, strat_key)

        if strat_key == "bull-put" and put_chain is not None:
            results[name] = scan_bull_put_spreads(symbol, stock_price, put_chain, config)
        elif strat_key == "bear-call" and call_chain is not None:
            results[name] = scan_bear_call_spreads(symbol, stock_price, call_chain, config)
        elif strat_key == "iron-condor" and call_chain is not None and put_chain is not None:
            results[name] = scan_iron_condors(symbol, stock_price, call_chain, put_chain, config)
        elif strat_key == "collar" and call_chain is not None and put_chain is not None:
            results[name] = scan_collars(symbol, stock_price, call_chain, put_chain, config)
        elif strat_key == "pmcc":
            results[name] = scan_pmcc(symbol, stock_price, config)

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_multileg_report(candidates: list[SpreadCandidate], strategy_name: str) -> str:
    """Format a list of spread candidates as a markdown table."""
    if not candidates:
        return f"### {strategy_name}\n\nNo candidates found.\n"

    lines = [f"### {strategy_name}\n"]

    # Header
    lines.append(
        "| # | DTE | Legs | Net Prem | Max Profit | Max Loss "
        "| Breakeven | POP | R:R | Notes |"
    )
    lines.append(
        "|---|-----|------|----------|------------|----------"
        "|-----------|-----|-----|-------|"
    )

    for idx, c in enumerate(candidates, 1):
        legs_str = " / ".join(
            f"{l['side']} {l['type']} ${l['strike']:.0f}" for l in c.legs
        )
        be_str = ", ".join(f"${b:.2f}" for b in c.breakeven)
        credit_sign = "+" if c.net_premium > 0 else ""

        lines.append(
            f"| {idx} | {c.dte} | {legs_str} "
            f"| {credit_sign}${c.net_premium:.2f} | ${c.max_profit:,.0f} "
            f"| ${c.max_loss:,.0f} | {be_str} "
            f"| {c.probability_of_profit:.0f}% | {c.risk_reward_ratio:.2f} "
            f"| {c.notes} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_all_strategies(results: dict[str, list[SpreadCandidate]], symbol: str) -> str:
    """Format all strategy results into a combined report."""
    parts = [f"## Multi-Leg Strategy Scan: {symbol}\n"]

    for strategy_name, candidates in results.items():
        parts.append(format_multileg_report(candidates, strategy_name))

    if not any(results.values()):
        parts.append("No spread candidates found. Try widening DTE range or lowering min premium.\n")

    return "\n".join(parts)
