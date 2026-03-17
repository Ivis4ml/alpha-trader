"""Portfolio management — positions, targets, cash tracking.

Separate from config.yaml (strategy settings only).
portfolio.yaml is the user's live portfolio state.
"""

from __future__ import annotations

import pathlib
import yaml

PORTFOLIO_PATH = pathlib.Path(__file__).resolve().parent.parent / "portfolio.yaml"

_DEFAULT = {
    "positions": {},
    "short_calls": [],
    "weekly_target": 1500,
    "cash_from_premiums": 0.0,
    "realized_pnl": 0.0,
    "total_premium_collected": 0.0,
    "total_trades": 0,
}


def load_portfolio(path: pathlib.Path | str | None = None) -> dict:
    p = pathlib.Path(path) if path else PORTFOLIO_PATH
    if not p.exists():
        return dict(_DEFAULT)
    with open(p) as f:
        data = yaml.safe_load(f) or {}
    # Ensure all keys exist
    for k, v in _DEFAULT.items():
        data.setdefault(k, v)
    return data


def save_portfolio(data: dict, path: pathlib.Path | str | None = None):
    p = pathlib.Path(path) if path else PORTFOLIO_PATH
    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_positions(portfolio: dict) -> dict:
    return portfolio.get("positions", {})


def get_symbols(portfolio: dict) -> list[str]:
    return list(portfolio.get("positions", {}).keys())


def get_position(portfolio: dict, symbol: str) -> dict:
    return portfolio.get("positions", {}).get(symbol, {})


def get_short_calls(portfolio: dict) -> list[dict]:
    return portfolio.get("short_calls", []) or []


def get_weekly_target(portfolio: dict) -> float:
    return portfolio.get("weekly_target", 1500)


def contracts_available(portfolio: dict, symbol: str, config: dict) -> int:
    pos = get_position(portfolio, symbol)
    shares = pos.get("shares", 0)
    max_pct = config.get("strategy", {}).get("max_contracts_pct", 75)
    max_contracts = int(shares / 100 * max_pct / 100)
    existing = sum(
        sc.get("contracts", 0)
        for sc in get_short_calls(portfolio)
        if sc.get("symbol") == symbol
    )
    return max(max_contracts - existing, 0)


def add_premium(amount: float):
    """Record premium collected from selling a call."""
    p = load_portfolio()
    p["cash_from_premiums"] = round(p.get("cash_from_premiums", 0) + amount, 2)
    p["total_premium_collected"] = round(p.get("total_premium_collected", 0) + amount, 2)
    p["total_trades"] = p.get("total_trades", 0) + 1
    save_portfolio(p)


def deduct_premium(amount: float):
    """Record premium paid to close a position (buy-to-close)."""
    p = load_portfolio()
    p["cash_from_premiums"] = round(p.get("cash_from_premiums", 0) - amount, 2)
    save_portfolio(p)


def record_pnl(pnl: float):
    """Record realized P&L from a closed trade."""
    p = load_portfolio()
    p["realized_pnl"] = round(p.get("realized_pnl", 0) + pnl, 2)
    save_portfolio(p)


def add_short_call(symbol: str, expiry: str, strike: float, contracts: int, premium: float):
    """Add a short call and update cash."""
    p = load_portfolio()
    p.setdefault("short_calls", []).append({
        "symbol": symbol,
        "strike": strike,
        "expiry": expiry,
        "contracts": contracts,
        "premium_received": premium,
    })
    total = premium * contracts * 100
    p["cash_from_premiums"] = round(p.get("cash_from_premiums", 0) + total, 2)
    p["total_premium_collected"] = round(p.get("total_premium_collected", 0) + total, 2)
    p["total_trades"] = p.get("total_trades", 0) + 1
    save_portfolio(p)


def close_short_call(symbol: str, expiry: str, strike: float, close_price: float | None = None, contracts: int | None = None):
    """Remove a short call and update cash if bought to close."""
    p = load_portfolio()
    shorts = p.get("short_calls", [])

    # Find and remove the matching short call
    removed = None
    new_shorts = []
    for sc in shorts:
        if (not removed and sc.get("symbol") == symbol
                and sc.get("expiry") == expiry and sc.get("strike") == strike):
            removed = sc
        else:
            new_shorts.append(sc)
    p["short_calls"] = new_shorts

    # Update cash: if closed early, deduct the cost
    if removed and close_price is not None and close_price > 0:
        n = contracts or removed.get("contracts", 0)
        cost = close_price * n * 100
        p["cash_from_premiums"] = round(p.get("cash_from_premiums", 0) - cost, 2)

        # Realized P&L for this trade
        entry_prem = removed.get("premium_received", 0)
        pnl = (entry_prem - close_price) * n * 100
        p["realized_pnl"] = round(p.get("realized_pnl", 0) + pnl, 2)
    elif removed and close_price is None:
        # Expired worthless — full premium is profit
        n = contracts or removed.get("contracts", 0)
        pnl = removed.get("premium_received", 0) * n * 100
        p["realized_pnl"] = round(p.get("realized_pnl", 0) + pnl, 2)

    save_portfolio(p)
    return removed


def format_portfolio_summary(portfolio: dict) -> str:
    """One-line portfolio summary."""
    positions = get_positions(portfolio)
    shorts = get_short_calls(portfolio)
    cash = portfolio.get("cash_from_premiums", 0)
    pnl = portfolio.get("realized_pnl", 0)
    total = portfolio.get("total_premium_collected", 0)
    target = get_weekly_target(portfolio)

    lines = [
        f"## Portfolio Summary",
        f"",
        f"| Symbol | Shares | Cost Basis | Allow Assignment |",
        f"|--------|--------|-----------|-----------------|",
    ]
    for sym, pos in positions.items():
        lines.append(
            f"| {sym} | {pos.get('shares', 0):,} | "
            f"${pos.get('cost_basis', 0):,.2f} | "
            f"{'Yes' if pos.get('allow_assignment', False) else 'No'} |"
        )

    lines.append(f"\n**Weekly Target:** ${target:,}")
    lines.append(f"**Cash from Premiums:** ${cash:,.2f}")
    lines.append(f"**Realized P&L:** ${pnl:,.2f}")
    lines.append(f"**Total Premium Collected:** ${total:,.2f}")
    lines.append(f"**Total Trades:** {portfolio.get('total_trades', 0)}")

    if shorts:
        lines.append(f"\n**Open Short Calls:**")
        for sc in shorts:
            lines.append(f"  - {sc['symbol']} {sc['expiry']} ${sc['strike']}C "
                        f"x{sc['contracts']} (${sc.get('premium_received', '?')})")

    return "\n".join(lines)


# ── Interactive Setup ────────────────────────────────────────────────────────

def interactive_setup() -> dict:
    """Walk a new user through portfolio setup."""
    import sys

    print("\n" + "=" * 50)
    print("  Welcome to Alpha Trader!")
    print("  Let's set up your portfolio.")
    print("=" * 50 + "\n")

    positions = {}

    print("Enter your stock positions (one per line).")
    print("Format: SYMBOL SHARES COST_BASIS")
    print("Example: AAPL 3000 180.00")
    print("(empty line to finish)\n")

    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            break

        parts = line.split()
        if len(parts) < 2:
            print("  Need at least SYMBOL and SHARES. Try again.")
            continue

        symbol = parts[0].upper()
        try:
            shares = int(parts[1])
        except ValueError:
            print("  Shares must be a number. Try again.")
            continue

        cost_basis = 0.0
        if len(parts) >= 3:
            try:
                cost_basis = float(parts[2])
            except ValueError:
                pass

        # Ask about assignment preference
        try:
            allow = input(f"  Allow {symbol} to be called away? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            allow = "n"

        positions[symbol] = {
            "shares": shares,
            "cost_basis": cost_basis,
            "allow_assignment": allow == "y",
        }
        print(f"  Added: {symbol} — {shares:,} shares @ ${cost_basis:.2f}\n")

    if not positions:
        print("No positions entered. You can add them later with 'update-position'.")
        positions = {}

    # Weekly target
    print("\nWeekly premium target (default: $1500)?")
    try:
        target_input = input("  > $").strip()
    except (EOFError, KeyboardInterrupt):
        target_input = ""

    try:
        weekly_target = float(target_input) if target_input else 1500
    except ValueError:
        weekly_target = 1500

    portfolio = {
        "positions": positions,
        "short_calls": [],
        "weekly_target": weekly_target,
        "cash_from_premiums": 0.0,
        "realized_pnl": 0.0,
        "total_premium_collected": 0.0,
        "total_trades": 0,
    }

    save_portfolio(portfolio)

    print(f"\nPortfolio saved to {PORTFOLIO_PATH}")
    print(f"  Positions: {len(positions)} symbols")
    print(f"  Target: ${weekly_target:,}/week")
    print(f"\nRun './at scan' to get your first action list!")
    print()

    return portfolio
