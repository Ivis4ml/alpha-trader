"""Portfolio margin calculation and optimization.

Calculates margin requirements for various option strategies and provides
tools to optimize capital usage against a weekly premium target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import load_config, get_symbols, get_position, get_short_calls


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MarginRequirement:
    """Margin detail for a single position or strategy."""
    symbol: str
    strategy: str
    description: str
    margin_required: float   # buying power consumed
    max_risk: float          # maximum possible loss
    premium_collected: float # premium received (or expected)
    return_on_capital: float # premium / margin as %


@dataclass
class MarginSummary:
    """Portfolio-wide margin overview."""
    positions: list[MarginRequirement]
    total_buying_power_used: float
    account_buying_power: float  # from config or estimated
    available_buying_power: float
    margin_utilization_pct: float
    total_premium: float
    portfolio_roc: float  # total premium / total margin used as %


@dataclass
class OptimizationResult:
    """Result of margin-aware strategy optimization."""
    suggestions: list[dict]
    total_premium: float
    total_margin: float
    target_premium: float
    shortfall: float
    capital_efficiency: float  # premium / margin %


# ---------------------------------------------------------------------------
# Margin calculators for individual strategies
# ---------------------------------------------------------------------------

def calculate_covered_call_margin(stock_price: float, shares: int) -> float:
    """Margin for a covered call position.

    For covered calls, the margin requirement is the full stock value
    (the shares serve as collateral for the short call).
    """
    return stock_price * shares


def calculate_spread_margin(
    short_strike: float,
    long_strike: float,
    contracts: int,
) -> float:
    """Margin for a vertical spread (bull put or bear call).

    Max risk = spread width x contracts x 100 (multiplier).
    This is the margin required since the long leg caps the risk.
    """
    width = abs(short_strike - long_strike)
    return width * contracts * 100


def calculate_iron_condor_margin(
    put_short_strike: float,
    put_long_strike: float,
    call_short_strike: float,
    call_long_strike: float,
    contracts: int,
) -> float:
    """Margin for an iron condor.

    Only one side can lose at a time, so margin = max of put spread
    width or call spread width, times contracts x 100.
    """
    put_width = abs(put_short_strike - put_long_strike)
    call_width = abs(call_long_strike - call_short_strike)
    return max(put_width, call_width) * contracts * 100


def calculate_collar_margin(stock_price: float, shares: int) -> float:
    """Margin for a collar (stock + protective put + covered call).

    The margin is primarily the stock value. The protective put reduces
    risk but brokers still require stock margin. In a margin account
    the put reduces the maintenance requirement.
    """
    # Reg-T: stock margin is 50% initial, but the put reduces maintenance
    # For simplicity, use the full stock value as buying power consumed.
    return stock_price * shares


def calculate_pmcc_margin(
    leaps_cost: float,
    contracts: int,
) -> float:
    """Margin for a Poor Man's Covered Call.

    The max risk is the LEAPS debit paid. The short call is covered by
    the LEAPS, so no additional margin beyond the LEAPS cost.
    """
    return leaps_cost * contracts * 100


# ---------------------------------------------------------------------------
# Margin for existing positions from config
# ---------------------------------------------------------------------------

def _margin_for_existing_positions(config: dict) -> list[MarginRequirement]:
    """Calculate margin for all positions currently in config."""
    from .data.fetcher import fetch_stock

    positions = []
    shorts = get_short_calls(config)
    symbols = get_symbols(config)

    for sym in symbols:
        pos = get_position(config, sym)
        shares = pos.get("shares", 0)
        if shares <= 0:
            continue

        try:
            stock = fetch_stock(sym)
            price = stock.price
        except Exception:
            price = pos.get("cost_basis", 0)

        # Stock position margin
        stock_margin = calculate_covered_call_margin(price, shares)

        # Premium from short calls on this symbol
        sym_shorts = [s for s in shorts if s.get("symbol") == sym]
        total_prem = sum(
            s.get("premium_received", 0) * s.get("contracts", 0) * 100
            for s in sym_shorts
        )

        if sym_shorts:
            strategy_desc = "Covered Call"
            short_desc = ", ".join(
                f"${s['strike']}C {s['expiry'][5:]}" for s in sym_shorts
            )
            description = f"{shares} shares + SHORT {short_desc}"
        else:
            strategy_desc = "Long Stock"
            description = f"{shares} shares @ ${price:.2f}"

        roc = (total_prem / stock_margin * 100) if stock_margin > 0 else 0

        positions.append(MarginRequirement(
            symbol=sym,
            strategy=strategy_desc,
            description=description,
            margin_required=round(stock_margin, 2),
            max_risk=round(stock_margin, 2),  # stock can go to 0
            premium_collected=round(total_prem, 2),
            return_on_capital=round(roc, 2),
        ))

    return positions


# ---------------------------------------------------------------------------
# Portfolio margin summary
# ---------------------------------------------------------------------------

def portfolio_margin_summary(config: dict | None = None) -> MarginSummary:
    """Build a full portfolio margin summary.

    Shows buying power used, available, and margin utilization for all
    positions in the config.
    """
    if config is None:
        config = load_config()

    positions = _margin_for_existing_positions(config)

    total_bp_used = sum(p.margin_required for p in positions)
    total_premium = sum(p.premium_collected for p in positions)

    # Account buying power: use config or default estimate
    account_bp = config.get("account", {}).get("buying_power", 0)
    if account_bp <= 0:
        # Estimate from position values + 20% buffer
        account_bp = total_bp_used * 1.2 if total_bp_used > 0 else 100_000

    available_bp = max(account_bp - total_bp_used, 0)
    utilization = (total_bp_used / account_bp * 100) if account_bp > 0 else 0
    portfolio_roc = (total_premium / total_bp_used * 100) if total_bp_used > 0 else 0

    return MarginSummary(
        positions=positions,
        total_buying_power_used=round(total_bp_used, 2),
        account_buying_power=round(account_bp, 2),
        available_buying_power=round(available_bp, 2),
        margin_utilization_pct=round(utilization, 1),
        total_premium=round(total_premium, 2),
        portfolio_roc=round(portfolio_roc, 2),
    )


# ---------------------------------------------------------------------------
# Format margin report
# ---------------------------------------------------------------------------

def format_margin_summary(summary: MarginSummary) -> str:
    """Format the margin summary as a markdown report."""
    lines = ["## Portfolio Margin Summary\n"]

    # Overview
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|------:|")
    lines.append(f"| Account Buying Power | ${summary.account_buying_power:,.0f} |")
    lines.append(f"| Buying Power Used | ${summary.total_buying_power_used:,.0f} |")
    lines.append(f"| Available | ${summary.available_buying_power:,.0f} |")
    lines.append(f"| Utilization | {summary.margin_utilization_pct:.1f}% |")
    lines.append(f"| Total Premium Collected | ${summary.total_premium:,.0f} |")
    lines.append(f"| Portfolio ROC | {summary.portfolio_roc:.2f}% |")
    lines.append("")

    # Position detail
    if summary.positions:
        lines.append("### Position Detail\n")
        lines.append(
            "| Symbol | Strategy | Description | Margin | Premium | ROC |"
        )
        lines.append(
            "|--------|----------|-------------|-------:|--------:|----:|"
        )
        for p in summary.positions:
            lines.append(
                f"| {p.symbol} | {p.strategy} | {p.description} "
                f"| ${p.margin_required:,.0f} | ${p.premium_collected:,.0f} "
                f"| {p.return_on_capital:.2f}% |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Margin-aware optimization
# ---------------------------------------------------------------------------

def optimize_margin(
    config: dict | None = None,
    target_premium: float | None = None,
) -> OptimizationResult:
    """Suggest the most capital-efficient combination of strategies
    to hit the weekly premium target.

    Analyzes available buying power and suggests a mix of covered calls,
    spreads, and iron condors ranked by return-on-capital.
    """
    if config is None:
        config = load_config()

    if target_premium is None:
        target_premium = config.get("strategy", {}).get("weekly_target", 1500)

    summary = portfolio_margin_summary(config)
    available_bp = summary.available_buying_power

    # Gather potential strategy suggestions
    suggestions: list[dict] = []
    total_suggested_premium = 0.0
    total_suggested_margin = 0.0

    symbols = get_symbols(config)

    for sym in symbols:
        pos = get_position(config, sym)
        shares = pos.get("shares", 0)

        try:
            from .data.fetcher import fetch_stock, fetch_option_chain
            stock = fetch_stock(sym)
            price = stock.price
        except Exception:
            continue

        # 1. Covered calls on available shares
        from .config import contracts_available
        avail_contracts = contracts_available(config, sym)
        if avail_contracts > 0:
            try:
                calls = fetch_option_chain(sym, price, config, "call")
                if calls:
                    # Pick the best-premium OTM call near target delta
                    target_delta = config.get("strategy", {}).get("target_delta", 0.20)
                    best = min(calls, key=lambda c: abs(abs(c.greeks.delta) - target_delta))
                    prem_per = best.mid if best.mid > 0 else best.last
                    total_prem = prem_per * avail_contracts * 100
                    margin = calculate_covered_call_margin(price, avail_contracts * 100)
                    roc = (total_prem / margin * 100) if margin > 0 else 0

                    suggestions.append({
                        "strategy": "Covered Call",
                        "symbol": sym,
                        "detail": (
                            f"Sell {avail_contracts}x {best.expiry} "
                            f"${best.strike:.0f}C @ ${prem_per:.2f}"
                        ),
                        "premium": round(total_prem, 2),
                        "margin_required": round(margin, 2),
                        "roc_pct": round(roc, 2),
                        "additional_margin": 0,  # already own the stock
                    })
                    total_suggested_premium += total_prem
            except Exception:
                pass

        # 2. Bull put spreads (uses only spread margin from available BP)
        if available_bp > 0:
            try:
                puts = fetch_option_chain(sym, price, config, "put")
                if len(puts) >= 2:
                    # Simple: sell highest-premium put, buy next lower strike
                    puts_sorted = sorted(puts, key=lambda p: p.bid, reverse=True)
                    short_p = puts_sorted[0]
                    long_p = next(
                        (p for p in puts_sorted[1:] if p.strike < short_p.strike),
                        None,
                    )
                    if long_p:
                        credit = (short_p.mid or short_p.last) - (long_p.mid or long_p.last)
                        if credit > 0:
                            width = short_p.strike - long_p.strike
                            # How many contracts fit in available BP?
                            margin_per = width * 100
                            max_contracts = int(available_bp / margin_per) if margin_per > 0 else 0
                            max_contracts = min(max_contracts, 5)  # cap at 5

                            if max_contracts > 0:
                                total_prem = credit * max_contracts * 100
                                margin = margin_per * max_contracts
                                roc = (total_prem / margin * 100) if margin > 0 else 0

                                suggestions.append({
                                    "strategy": "Bull Put Spread",
                                    "symbol": sym,
                                    "detail": (
                                        f"{max_contracts}x {short_p.expiry} "
                                        f"sell ${short_p.strike:.0f}P / "
                                        f"buy ${long_p.strike:.0f}P "
                                        f"@ ${credit:.2f} credit"
                                    ),
                                    "premium": round(total_prem, 2),
                                    "margin_required": round(margin, 2),
                                    "roc_pct": round(roc, 2),
                                    "additional_margin": round(margin, 2),
                                })
                                total_suggested_premium += total_prem
                                total_suggested_margin += margin
            except Exception:
                pass

    # Sort suggestions by ROC (most capital-efficient first)
    suggestions.sort(key=lambda s: s["roc_pct"], reverse=True)

    total_margin = sum(s["margin_required"] for s in suggestions)
    shortfall = max(target_premium - total_suggested_premium, 0)
    efficiency = (total_suggested_premium / total_margin * 100) if total_margin > 0 else 0

    return OptimizationResult(
        suggestions=suggestions,
        total_premium=round(total_suggested_premium, 2),
        total_margin=round(total_margin, 2),
        target_premium=target_premium,
        shortfall=round(shortfall, 2),
        capital_efficiency=round(efficiency, 2),
    )


def format_optimization(result: OptimizationResult) -> str:
    """Format optimization results as markdown."""
    lines = ["## Margin Optimization\n"]

    lines.append(f"**Target weekly premium:** ${result.target_premium:,.0f}")
    lines.append(f"**Achievable premium:** ${result.total_premium:,.0f}")
    if result.shortfall > 0:
        lines.append(f"**Shortfall:** ${result.shortfall:,.0f}")
    lines.append(f"**Total margin needed:** ${result.total_margin:,.0f}")
    lines.append(f"**Capital efficiency:** {result.capital_efficiency:.2f}%\n")

    if result.suggestions:
        lines.append(
            "| # | Strategy | Symbol | Detail | Premium | Margin | ROC |"
        )
        lines.append(
            "|---|----------|--------|--------|--------:|-------:|----:|"
        )
        for i, s in enumerate(result.suggestions, 1):
            lines.append(
                f"| {i} | {s['strategy']} | {s['symbol']} "
                f"| {s['detail']} | ${s['premium']:,.0f} "
                f"| ${s['margin_required']:,.0f} | {s['roc_pct']:.2f}% |"
            )
        lines.append("")
    else:
        lines.append("No optimization suggestions available.\n")

    return "\n".join(lines)
