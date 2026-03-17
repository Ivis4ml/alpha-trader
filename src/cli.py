"""Alpha Trader CLI — AI-powered covered call advisor."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def _load_dotenv():
    """Load .env file without extra dependency."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_scan(args):
    """Fetch data and generate briefing, optionally pipe to Claude for analysis."""
    from .config import load_config
    from .report import generate_briefing

    config = load_config(args.config)

    print("Fetching market data..." + (" (quick mode)" if args.quick else ""))
    briefing = generate_briefing(config, quick=args.quick)

    if args.data_only:
        print(briefing)
        return

    # Save briefing to file
    REPORTS_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    briefing_path = REPORTS_DIR / f"briefing_{ts}.md"
    briefing_path.write_text(briefing)
    print(f"Briefing saved: {briefing_path}")

    if args.no_ai:
        print(briefing)
        return

    # Pipe to Claude Code for analysis
    print("Sending to Claude for analysis...")
    report_path = REPORTS_DIR / f"action_{ts}.md"

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", args.model, "--reasoning-effort", "high", briefing],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode == 0 and result.stdout.strip():
            report = result.stdout.strip()
            report_path.write_text(report)
            print(f"\nAction list saved: {report_path}\n")
            print(report)
        else:
            print("Claude analysis failed. Falling back to data-only output.")
            if result.stderr:
                print(f"Error: {result.stderr[:500]}", file=sys.stderr)
            print(briefing)
    except FileNotFoundError:
        print("'claude' CLI not found. Showing raw briefing instead.")
        print(briefing)
    except subprocess.TimeoutExpired:
        print("Claude analysis timed out. Showing raw briefing.")
        print(briefing)

    # Send notification if requested
    if args.notify:
        from .notify import send_report
        text = report_path.read_text() if report_path.exists() else briefing
        if send_report(text):
            print("Sent to Telegram.")
        else:
            print("Telegram notification failed (check .env).")


def cmd_setup(args):
    """Interactive portfolio setup for new users."""
    from .portfolio import interactive_setup, PORTFOLIO_PATH
    if PORTFOLIO_PATH.exists() and not args.force:
        print(f"Portfolio already exists: {PORTFOLIO_PATH}")
        print("Use --force to overwrite, or edit portfolio.yaml directly.")
        return
    interactive_setup()


def cmd_portfolio(args):
    """Show current portfolio state: positions, cash, P&L."""
    from .portfolio import load_portfolio, format_portfolio_summary
    p = load_portfolio()
    print(format_portfolio_summary(p))


def cmd_preview(args):
    """Ultra-fast market snapshot (<2s). Shows prices + positions while full scan loads."""
    import yfinance as yf
    from .config import load_config, get_symbols, get_short_calls, get_position, contracts_available

    config = load_config(args.config)
    symbols = get_symbols(config)
    shorts = get_short_calls(config)

    # Use fast_info for each ticker — more reliable than batch download
    lines = ["## Market Snapshot\n"]

    try:
        vix_fi = yf.Ticker("^VIX").fast_info
        spy_fi = yf.Ticker("SPY").fast_info
        vix = vix_fi.get("lastPrice") or vix_fi.get("previousClose") or 0
        spy = spy_fi.get("lastPrice") or spy_fi.get("previousClose") or 0
        spy_prev = spy_fi.get("previousClose") or spy
        spy_chg = (spy - spy_prev) / spy_prev * 100 if spy_prev else 0
        lines.append(f"VIX **{vix:.1f}** | SPY **${spy:.2f}** ({'+' if spy_chg >= 0 else ''}{spy_chg:.1f}%)\n")
    except Exception:
        lines.append("VIX/SPY: loading...\n")

    lines.append("| Symbol | Price | Change | Shares | Open Shorts | Avail |")
    lines.append("|--------|-------|--------|--------|-------------|-------|")
    for sym in symbols:
        try:
            fi = yf.Ticker(sym).fast_info
            price = fi.get("lastPrice") or fi.get("previousClose") or 0
            prev = fi.get("previousClose") or price
            chg = (price - prev) / prev * 100 if prev else 0
        except Exception:
            price = 0
            chg = 0
        pos = get_position(config, sym)
        avail = contracts_available(config, sym)
        sym_shorts = [s for s in shorts if s["symbol"] == sym]
        short_str = ", ".join(f"${s['strike']}C {s['expiry'][5:]}" for s in sym_shorts) or "—"
        lines.append(
            f"| {sym} | ${price:.2f} | {'+' if chg >= 0 else ''}{chg:.1f}% | "
            f"{pos.get('shares', 0):,} | {short_str} | {avail} |"
        )

    lines.append("")
    lines.append("_Fetching option chains & technicals..._")
    print("\n".join(lines))


def cmd_roll(args):
    """Analyze existing short calls for roll/close opportunities."""
    from .config import load_config
    from .roll import analyze_rolls

    config = load_config(args.config)
    print("Analyzing positions...")
    report = analyze_rolls(config)
    print(report)

    if args.notify:
        from .notify import send_report
        send_report(report)


def cmd_notify(args):
    """Send the latest report via Telegram."""
    from .notify import send_telegram, verify_bot

    if args.test:
        bot = verify_bot()
        if bot:
            print(f"Bot OK: @{bot['username']}")
            send_telegram("Alpha Trader test message. Bot is working!")
            print("Test message sent.")
        else:
            print("Bot verification failed. Check TELEGRAM_BOT_TOKEN in .env")
        return

    # Send latest report
    reports = sorted(REPORTS_DIR.glob("action_*.md"), reverse=True)
    if not reports:
        reports = sorted(REPORTS_DIR.glob("briefing_*.md"), reverse=True)
    if not reports:
        print("No reports found. Run 'scan' first.")
        return

    text = reports[0].read_text()
    from .notify import send_report
    if send_report(text):
        print(f"Sent {reports[0].name} to Telegram.")
    else:
        print("Failed. Check .env for TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")


def cmd_bot(args):
    """Run interactive Telegram bot (long-polling daemon)."""
    from .bot import run
    run()


def cmd_update_position(args):
    """Update a position in portfolio.yaml."""
    from .portfolio import load_portfolio, save_portfolio

    p = load_portfolio()
    sym = args.symbol.upper()
    if sym not in p.get("positions", {}):
        p.setdefault("positions", {})[sym] = {}

    p["positions"][sym]["shares"] = args.shares
    if args.cost_basis:
        p["positions"][sym]["cost_basis"] = args.cost_basis

    save_portfolio(p)
    print(f"Updated {sym}: {args.shares:,} shares")


def cmd_add_short(args):
    """Record a new short call position."""
    from .portfolio import add_short_call

    # Save to portfolio.yaml + update cash
    add_short_call(args.symbol, args.expiry, args.strike, args.contracts, args.premium)

    # Record in SQLite trade history
    from .db import record_trade
    trade_id = record_trade(
        symbol=args.symbol,
        strike=args.strike,
        expiry=args.expiry,
        contracts=args.contracts,
        premium_per_contract=args.premium,
        delta=getattr(args, "delta", None),
        otm_pct=getattr(args, "otm_pct", None),
    )

    total = args.premium * args.contracts * 100
    print(f"Added: SHORT {args.contracts}x {args.symbol} {args.expiry} ${args.strike} Call @ ${args.premium}")
    print(f"  Premium collected: ${total:,.2f}")
    print(f"  (trade #{trade_id} recorded)")


def cmd_close_short(args):
    """Remove a short call from portfolio (expired or closed)."""
    from .portfolio import close_short_call
    close_price = getattr(args, "close_price", None)
    removed = close_short_call(args.symbol, args.expiry, args.strike, close_price)

    # Record close in SQLite trade history
    if removed:
        from .db import close_trade
        status = getattr(args, "status", "expired")
        close_trade(
            symbol=args.symbol,
            expiry=args.expiry,
            strike=args.strike,
            status=status,
            close_price=close_price,
        )

    print(f"{'Closed' if removed else 'Not found:'} {args.symbol} {args.expiry} ${args.strike}")
    if removed and close_price:
        print(f"  Bought to close @ ${close_price}")
    elif removed:
        print(f"  Expired worthless — full premium captured")


def cmd_report(args):
    """Show P&L reports from trade history."""
    from .db import get_weekly_summary, get_monthly_summary, get_cumulative_pnl, get_open_trades
    from .config import load_config

    config = load_config(getattr(args, "config", None))
    weekly_target = config.get("strategy", {}).get("weekly_target", 0)

    period = args.period if hasattr(args, "period") and args.period else "weekly"

    if period in ("weekly", "all"):
        weeks = get_weekly_summary(weeks=52 if period == "all" else 8)
        print("\n## Weekly Premium Summary\n")
        print(f"| Week Start | Premium | Target | % of Target | Trades | Symbols |")
        print(f"|------------|--------:|-------:|------------:|-------:|---------|")
        for w in weeks:
            prem = w["total_premium"] or 0
            pct = (prem / weekly_target * 100) if weekly_target else 0
            print(
                f"| {w['week_start']} | ${prem:,.0f} | ${weekly_target:,.0f} "
                f"| {pct:.0f}% | {w['trades_count']} | {w['symbols'] or ''} |"
            )
        if not weeks:
            print("| (no trades yet) | | | | | |")

    if period in ("monthly", "all"):
        months = get_monthly_summary(months=24 if period == "all" else 6)
        monthly_target = weekly_target * 4.33  # ~weeks per month
        print("\n## Monthly Premium Summary\n")
        print(f"| Month | Premium | Target | % of Target | Realized P&L | Trades | Symbols |")
        print(f"|-------|--------:|-------:|------------:|-------------:|-------:|---------|")
        for m in months:
            prem = m["total_premium"] or 0
            rpnl = m["realized_pnl"] or 0
            pct = (prem / monthly_target * 100) if monthly_target else 0
            print(
                f"| {m['month']} | ${prem:,.0f} | ${monthly_target:,.0f} "
                f"| {pct:.0f}% | ${rpnl:,.0f} | {m['trades_count']} | {m['symbols'] or ''} |"
            )
        if not months:
            print("| (no trades yet) | | | | | | |")

    # Always show cumulative stats
    cum = get_cumulative_pnl()
    if cum and cum.get("total_trades", 0) > 0:
        print("\n## Cumulative Stats\n")
        print(f"- **Total trades:** {cum['total_trades']}")
        print(f"- **Total premium collected:** ${cum['total_premium_collected']:,.0f}")
        print(f"- **Realized P&L:** ${cum['realized_pnl']:,.0f}")
        print(f"- **Unrealized premium (open):** ${cum['unrealized_premium']:,.0f}")
        print(f"- **Breakdown:** {cum['open_count']} open, {cum['expired_count']} expired, "
              f"{cum['closed_count']} closed, {cum['assigned_count']} assigned")

    # Show open trades
    open_trades = get_open_trades()
    if open_trades:
        print("\n## Open Trades\n")
        print("| # | Symbol | Strike | Expiry | Contracts | Premium/ct | Total | Status |")
        print("|---|--------|-------:|--------|----------:|-----------:|------:|--------|")
        for t in open_trades:
            print(
                f"| {t['id']} | {t['symbol']} | ${t['strike']:.2f} | {t['expiry']} "
                f"| {t['contracts']} | ${t['premium_per_contract']:.2f} "
                f"| ${t['total_premium']:,.0f} | {t['status']} |"
            )

    print()


def cmd_backtest(args):
    """Run covered call backtest simulation over historical data."""
    import datetime as _dt
    from .backtest import run_backtest, format_summary, format_weekly_detail

    end = _dt.date.today()
    start = end - _dt.timedelta(days=args.months * 30)

    symbols = [s.strip().upper() for s in args.symbol.split(",")]

    for sym in symbols:
        print(f"\nRunning backtest for {sym} ({args.months} months, "
              f"delta={args.delta:.0%}, dte={args.dte})...")
        try:
            result = run_backtest(
                symbol=sym,
                shares=args.shares,
                start_date=start.isoformat(),
                end_date=end.isoformat(),
                delta_target=args.delta,
                dte_target=args.dte,
            )
            print(format_summary(result))

            if args.weekly:
                print(format_weekly_detail(result))

        except Exception as e:
            print(f"  Error: {e}")

    if len(symbols) > 1:
        print("(Backtest complete for all symbols)")


def cmd_daily(args):
    """Combined daily view: position advice + scan prompt."""
    from .config import load_config, get_short_calls
    from .data.fetcher import fetch_stock, fetch_events
    from .data.greeks import bs_call_price
    from .strategy import advise_position
    from .optimizer import should_optimize

    config = load_config(getattr(args, "config", None))
    shorts = get_short_calls(config)
    today = dt.date.today()

    print(f"# Daily Review — {today}\n")

    if shorts:
        print("## Open Positions\n")
        print("| Symbol | Strike | Expiry | DTE | Entry | Est.Now | Captured | Action | Reason |")
        print("|--------|--------|--------|-----|-------|---------|----------|--------|--------|")
        for sc in shorts:
            sym, strike, expiry_str = sc["symbol"], sc["strike"], sc["expiry"]
            premium = sc.get("premium_received", 0)
            expiry = dt.datetime.strptime(expiry_str, "%Y-%m-%d").date()
            dte = (expiry - today).days
            stock = fetch_stock(sym)
            is_itm = stock.price >= strike
            T = max(dte, 0) / 365.0
            est_now = bs_call_price(stock.price, strike, T, 0.043, 0.30)
            captured = (premium - est_now) / premium * 100 if premium > 0 else 0
            events = fetch_events(sym)
            earnings_before = (events.days_to_earnings is not None
                             and 0 <= events.days_to_earnings <= dte)
            advice = advise_position(premium, est_now, dte, is_itm, earnings_before, config)
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(advice.urgency, "")
            print(f"| {sym} | ${strike} | {expiry_str[5:]} | {dte} | "
                  f"${premium:.2f} | ${est_now:.2f} | {captured:.0f}% | "
                  f"{icon} **{advice.action}** | {advice.reason} |")
        print()
    else:
        print("No open positions.\n")

    if should_optimize():
        print("Optimization available — run `optimize` to tune parameters.\n")
    print("_Run `/scan` for today's covered call candidates._")


def cmd_optimize(args):
    """Run strategy optimizer — analyze trades and suggest parameter changes."""
    from .optimizer import analyze_and_suggest, format_optimization, apply_suggestions, log_optimization
    print("Analyzing trade history...")
    result = analyze_and_suggest()
    print(format_optimization(result))
    if args.apply and result.suggestions:
        apply_suggestions(result.suggestions)
        log_optimization(result)
        print("Suggestions applied and logged.")


def cmd_alerts(args):
    """Check all alert conditions on current positions."""
    from .config import load_config
    from .alerts import check_all_alerts, format_alerts

    config = load_config(getattr(args, "config", None))
    print("Checking alerts...")
    alerts = check_all_alerts(config)
    print(format_alerts(alerts))

    if args.notify and alerts:
        from .notify import send_telegram
        send_telegram(format_alerts(alerts))


def cmd_correlation(args):
    """Analyze correlation between portfolio positions."""
    from .config import load_config, get_symbols
    from .analytics import analyze_correlation, format_correlation

    config = load_config(getattr(args, "config", None))
    symbols = get_symbols(config)
    print(f"Analyzing correlation: {', '.join(symbols)}...")
    result = analyze_correlation(symbols)
    print(format_correlation(result))


def cmd_news(args):
    """Portfolio news digest with sentiment scoring."""
    from .config import load_config, get_symbols
    from .data.news import fetch_news, score_news_sentiment
    from .data.events_calendar import fetch_macro_calendar, scan_news_for_risks

    config = load_config(getattr(args, "config", None))
    symbols = [args.symbol.upper()] if args.symbol else get_symbols(config)

    print("# News Digest\n")

    for sym in symbols:
        news = fetch_news(sym, max_items=8)
        if not news:
            print(f"## {sym} — no recent news\n")
            continue

        sentiment = score_news_sentiment(news)
        icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(sentiment["label"], "")
        print(f"## {sym} {icon} {sentiment['label']} (score: {sentiment['score']:+.2f})\n")

        for i, n in enumerate(news, 1):
            print(f"  {i}. {n['title']}")
            if n.get("publisher"):
                print(f"     — {n['publisher']}")

        # Risk flags from headlines
        risks = scan_news_for_risks(news)
        high_risks = [r for r in risks if r.impact == "HIGH"]
        if high_risks:
            print(f"\n  ⚠ Risk flags:")
            for r in high_risks:
                print(f"    - {r.description[:70]}")

        if sentiment.get("signals"):
            print(f"\n  Signals: {' | '.join(sentiment['signals'])}")
        print()

    # Macro events
    events = fetch_macro_calendar(lookahead_days=14)
    if events:
        print("## Macro Calendar (next 14 days)\n")
        for e in events:
            icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(e.impact, "")
            print(f"  {icon} {e.date} — **{e.event}** ({e.days_away}d) {e.description[:50]}")
        print()

    print("_Ask about any headline for impact analysis._")


def cmd_calendar(args):
    """Show upcoming macro events calendar."""
    from .data.events_calendar import fetch_macro_calendar, format_event_calendar
    days = args.days if hasattr(args, "days") else 30
    events = fetch_macro_calendar(lookahead_days=days)
    print(format_event_calendar(events))


def cmd_earnings_crush(args):
    """Analyze historical IV crush around earnings."""
    from .analytics import analyze_earnings_crush, format_earnings_crush

    symbols = [s.strip().upper() for s in args.symbol.split(",")]
    for sym in symbols:
        print(f"Analyzing earnings crush for {sym}...")
        result = analyze_earnings_crush(sym)
        print(format_earnings_crush(result))
        print()


def cmd_paper(args):
    """Alpaca paper trading — validate orders before executing on Robinhood."""
    from .paper import PaperTrader

    trader = PaperTrader()
    action = args.paper_action

    if action == "status":
        acct = trader.get_account()
        print("## Alpaca Paper Account\n")
        print(f"  Status:          {acct['status']}")
        print(f"  Portfolio Value:  ${acct['portfolio_value']:,.2f}")
        print(f"  Cash:            ${acct['cash']:,.2f}")
        print(f"  Buying Power:    ${acct['buying_power']:,.2f}")
        print(f"  Equity:          ${acct['equity']:,.2f}")
        print(f"  Long MV:         ${acct['long_market_value']:,.2f}")
        print(f"  Short MV:        ${acct['short_market_value']:,.2f}")

        positions = trader.get_positions()
        if positions:
            print("\n## Positions\n")
            print("| Symbol | Qty | Side | Avg Entry | Current | P&L | Class |")
            print("|--------|----:|------|----------:|--------:|----:|-------|")
            for p in positions:
                print(
                    f"| {p['symbol']} | {p['qty']:.0f} | {p['side']} "
                    f"| ${p['avg_entry_price']:.2f} | ${p['current_price']:.2f} "
                    f"| ${p['unrealized_pl']:,.2f} | {p['asset_class']} |"
                )
        else:
            print("\nNo open positions.")

    elif action == "submit":
        symbol = args.symbol.upper()
        result = trader.submit_covered_call(
            symbol=symbol,
            strike=args.strike,
            expiry=args.expiry,
            contracts=args.contracts,
            limit_price=args.limit_price,
        )
        print(f"Order submitted: SELL {args.contracts}x {symbol} "
              f"{args.expiry} ${args.strike} Call @ ${args.limit_price}")
        print(f"  Order ID: {result['id']}")
        print(f"  Status:   {result['status']}")

    elif action == "close":
        symbol = args.symbol.upper()
        result = trader.close_option(
            symbol=symbol,
            strike=args.strike,
            expiry=args.expiry,
            contracts=args.contracts,
            limit_price=args.limit_price,
        )
        print(f"Order submitted: BUY {args.contracts}x {symbol} "
              f"{args.expiry} ${args.strike} Call @ ${args.limit_price}")
        print(f"  Order ID: {result['id']}")
        print(f"  Status:   {result['status']}")

    elif action == "orders":
        orders = trader.get_orders(status="open")
        if not orders:
            print("No open orders.")
            return
        print("## Open Orders\n")
        print("| ID (short) | Symbol | Side | Qty | Type | Limit | Status | Created |")
        print("|------------|--------|------|----:|------|------:|--------|---------|")
        for o in orders:
            short_id = (o["id"] or "")[:8]
            limit = f"${float(o['limit_price']):,.2f}" if o.get("limit_price") else "—"
            created = (o.get("created_at") or "")[:19]
            print(
                f"| {short_id}... | {o['symbol']} | {o['side']} "
                f"| {o['qty']} | {o['type']} | {limit} "
                f"| {o['status']} | {created} |"
            )

    elif action == "cancel":
        trader.cancel_order(args.order_id)
        print(f"Order {args.order_id} cancelled.")

    elif action == "chain":
        chain = trader.get_option_chain(args.symbol.upper(), args.expiry)
        if not chain:
            print(f"No option data for {args.symbol.upper()} {args.expiry}")
            return
        print(f"## Option Chain: {args.symbol.upper()} Calls — {args.expiry}\n")
        print("| Contract | Bid | Ask | Last | Delta | Theta | IV |")
        print("|----------|----:|----:|-----:|------:|------:|---:|")
        for c in chain:
            print(
                f"| {c['osi_symbol']} | ${c['bid']:.2f} | ${c['ask']:.2f} "
                f"| ${c['last']:.2f} | {c['delta']:.3f} | {c['theta']:.3f} "
                f"| {c['iv']:.1%} |"
            )


def cmd_spreads(args):
    """Scan for multi-leg option strategy candidates."""
    from .config import load_config
    from .data.fetcher import fetch_stock
    from .multileg import scan_strategies, format_all_strategies, STRATEGY_MAP

    config = load_config(getattr(args, "config", None))
    symbol = args.symbol.upper()

    strategy = args.strategy if hasattr(args, "strategy") and args.strategy else None
    if strategy and strategy not in STRATEGY_MAP:
        print(f"Unknown strategy '{strategy}'. Available: {', '.join(STRATEGY_MAP.keys())}")
        return

    print(f"Fetching data for {symbol}...")
    try:
        stock = fetch_stock(symbol)
    except Exception as e:
        print(f"Failed to fetch {symbol}: {e}")
        return

    print(f"  {symbol} @ ${stock.price:.2f}")
    strategy_label = STRATEGY_MAP.get(strategy, "all strategies") if strategy else "all strategies"
    print(f"Scanning {strategy_label}...\n")

    results = scan_strategies(symbol, stock.price, config, strategy=strategy)
    print(format_all_strategies(results, symbol))


def cmd_margin(args):
    """Show portfolio margin summary and optional optimization."""
    from .config import load_config
    from .margin import (
        portfolio_margin_summary, format_margin_summary,
        optimize_margin, format_optimization,
    )

    config = load_config(getattr(args, "config", None))

    print("Calculating portfolio margin...\n")
    summary = portfolio_margin_summary(config)
    print(format_margin_summary(summary))

    if getattr(args, "optimize", False):
        target = getattr(args, "target", None)
        print("Running margin optimization...\n")
        result = optimize_margin(config, target_premium=target)
        print(format_optimization(result))


def cmd_spark(args):
    """Inline sparkline chart — shows directly in Claude Code Remote."""
    from .sparkline import generate_sparkline_report
    from .config import load_config, get_symbols, get_short_calls

    config = load_config(getattr(args, "config", None))
    symbols = [args.symbol.upper()] if args.symbol else get_symbols(config)
    shorts = get_short_calls(config)

    for sym in symbols:
        sym_shorts = [s for s in shorts if s.get("symbol") == sym]
        report = generate_sparkline_report(sym, args.period, sym_shorts)
        print(report)
        print()

    print("_Interactive chart: http://127.0.0.1:8080/chart/SYMBOL_")


def cmd_chart(args):
    """Generate interactive candlestick chart with technical overlays."""
    import webbrowser
    from .charts import generate_chart
    from .config import load_config, get_short_calls
    from .db import get_trade_history

    config = load_config(getattr(args, "config", None))
    symbol = args.symbol.upper()
    indicators = [i.strip() for i in args.indicators.split(",")] if args.indicators else ["sma", "bb", "rsi", "volume"]

    # Get trades and short calls for markers
    trades = []
    try:
        all_trades = get_trade_history(limit=50)
        trades = [t for t in all_trades if t.get("symbol") == symbol]
    except Exception:
        pass

    short_calls = [sc for sc in get_short_calls(config) if sc.get("symbol") == symbol]

    print(f"Generating chart for {symbol} ({args.period}, indicators: {', '.join(indicators)})...")
    path = generate_chart(
        symbol=symbol,
        period=args.period,
        indicators=indicators,
        trades=trades,
        show_short_calls=short_calls,
    )

    if path:
        print(f"Saved: {path}")
        # Always show the dashboard URL for Remote access
        import subprocess
        try:
            ip = subprocess.run(["ipconfig", "getifaddr", "en0"],
                              capture_output=True, text=True).stdout.strip() or "127.0.0.1"
        except Exception:
            ip = "127.0.0.1"
        print(f"\nOpen on iPhone: http://{ip}:8080/chart/{symbol}")
        if not args.no_open:
            webbrowser.open(f"file://{path}")
    else:
        print("No data available.")


def cmd_iv_surface(args):
    """Generate IV surface visualization."""
    from .iv_surface import plot_iv_surface_html, plot_iv_smile_html

    symbol = args.symbol.upper()

    if args.smile:
        print(f"Generating IV smile for {symbol} (nearest expiry)...")
        path = plot_iv_smile_html(symbol)
    else:
        print(f"Generating IV surface for {symbol} (all expiries)...")
        path = plot_iv_surface_html(symbol)

    if path:
        print(f"Saved: {path}")
        if not args.no_open:
            import webbrowser
            webbrowser.open(f"file://{path}")
    else:
        print("No data available. Market may be closed or symbol has no options.")


def cmd_dashboard(args):
    """Launch the web dashboard."""
    from .dashboard import run_dashboard
    run_dashboard(host=args.host, port=args.port, debug=args.debug)


def cmd_ml(args):
    """Machine learning signal generation for covered call timing."""
    action = args.ml_action

    if action == "train":
        from .config import load_config, get_symbols
        from .ml_signals import train_model, format_train_result

        config = load_config(getattr(args, "config", None))
        symbols = get_symbols(config)
        if not symbols:
            print("No symbols in portfolio. Add positions first.")
            return

        print(f"Training ML model on {', '.join(symbols)}...")
        print(f"  Lookback: {args.months} months | Delta: {args.delta} | DTE: {args.dte}\n")
        try:
            result = train_model(
                symbols=symbols,
                lookback_months=args.months,
                delta_target=args.delta,
                dte=args.dte,
            )
            print(format_train_result(result))
        except Exception as e:
            print(f"Training failed: {e}")

    elif action == "predict":
        from .config import load_config, get_symbols
        from .ml_signals import predict_signal, format_prediction

        config = load_config(getattr(args, "config", None))
        symbols = get_symbols(config)
        if not symbols:
            print("No symbols in portfolio.")
            return

        print(f"\n{'=' * 60}")
        print("  ML SIGNAL PREDICTIONS")
        print(f"{'=' * 60}\n")
        for sym in symbols:
            try:
                result = predict_signal(sym)
                print(format_prediction(sym, result))
                print()
            except Exception as e:
                print(f"  [?] {sym:<6s}  Error: {e}\n")
        print(f"{'=' * 60}\n")

    elif action == "features":
        from .ml_signals import build_features_current, format_features

        symbol = args.symbol.upper()
        print(f"Fetching features for {symbol}...")
        try:
            features = build_features_current(symbol)
            print(format_features(symbol, features))
        except Exception as e:
            print(f"Error: {e}")

    else:
        print("Usage: alpha-trader ml {train|predict|features}")


def cmd_cron(args):
    """Install or remove cron jobs for automated scanning."""
    cron_script = PROJECT_ROOT / "scripts" / "cron_scan.sh"
    if not cron_script.exists():
        print(f"Missing {cron_script}")
        return

    project_path = str(PROJECT_ROOT)

    if args.action == "install":
        morning = f'0 8 * * 1-5 cd {project_path} && TZ=US/Pacific {cron_script} morning >> {project_path}/reports/cron.log 2>&1'
        midday = f'0 12 * * 1-5 cd {project_path} && TZ=US/Pacific {cron_script} midday >> {project_path}/reports/cron.log 2>&1'

        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        existing = result.stdout if result.returncode == 0 else ""

        lines = [l for l in existing.splitlines() if "alpha-trader" not in l and "cron_scan" not in l]
        lines.append("# alpha-trader morning scan")
        lines.append(morning)
        lines.append("# alpha-trader midday scan")
        lines.append(midday)

        new_crontab = "\n".join(lines) + "\n"
        proc = subprocess.run(["crontab", "-"], input=new_crontab, text=True, capture_output=True)
        if proc.returncode == 0:
            print("Cron jobs installed (Mon-Fri 8AM & 12PM PST)")
        else:
            print(f"Failed: {proc.stderr}")

    elif args.action == "remove":
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            print("No crontab.")
            return
        lines = [l for l in result.stdout.splitlines()
                 if "alpha-trader" not in l and "cron_scan" not in l]
        new_crontab = "\n".join(lines) + "\n" if lines else ""
        subprocess.run(["crontab", "-"], input=new_crontab, text=True)
        print("Cron jobs removed.")

    elif args.action == "status":
        result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            print("No crontab.")
            return
        found = [l for l in result.stdout.splitlines()
                 if "alpha-trader" in l or "cron_scan" in l]
        if found:
            for l in found:
                print(f"  {l}")
        else:
            print("No alpha-trader cron jobs.")


def main():
    _load_dotenv()

    # Auto-run setup if portfolio.yaml doesn't exist
    from .portfolio import PORTFOLIO_PATH
    if not PORTFOLIO_PATH.exists():
        print("No portfolio found. Let's set one up.\n")
        from .portfolio import interactive_setup
        interactive_setup()

    parser = argparse.ArgumentParser(
        prog="alpha-trader",
        description="AI-powered covered call & options advisor",
    )
    sub = parser.add_subparsers(dest="command")

    # setup
    p_setup = sub.add_parser("setup", help="Interactive portfolio setup for new users")
    p_setup.add_argument("--force", action="store_true", help="Overwrite existing portfolio")
    p_setup.set_defaults(func=cmd_setup)

    # portfolio
    p_pf = sub.add_parser("portfolio", help="Show portfolio: positions, cash, P&L")
    p_pf.set_defaults(func=cmd_portfolio)

    # scan
    p_scan = sub.add_parser("scan", help="Fetch data + generate action list")
    p_scan.add_argument("--data-only", action="store_true", help="Raw briefing, no AI")
    p_scan.add_argument("--no-ai", action="store_true", help="Save briefing, skip Claude")
    p_scan.add_argument("--quick", action="store_true", help="Fast mode: skip news/insider/puts")
    p_scan.add_argument("--notify", action="store_true", help="Send result via Telegram")
    p_scan.add_argument("--config", default=None)
    p_scan.add_argument("--model", default="claude-opus-4-6")
    p_scan.set_defaults(func=cmd_scan)

    # preview
    p_prev = sub.add_parser("preview", help="Ultra-fast market snapshot (<2s)")
    p_prev.add_argument("--config", default=None)
    p_prev.set_defaults(func=cmd_preview)

    # roll
    p_roll = sub.add_parser("roll", help="Analyze short calls for roll/close")
    p_roll.add_argument("--notify", action="store_true")
    p_roll.add_argument("--config", default=None)
    p_roll.set_defaults(func=cmd_roll)

    # notify
    p_notify = sub.add_parser("notify", help="Send latest report via Telegram")
    p_notify.add_argument("--test", action="store_true", help="Test Telegram connection")
    p_notify.set_defaults(func=cmd_notify)

    # bot
    p_bot = sub.add_parser("bot", help="Run interactive Telegram bot daemon")
    p_bot.set_defaults(func=cmd_bot)

    # update-position
    p_pos = sub.add_parser("update-position", help="Update share count")
    p_pos.add_argument("symbol")
    p_pos.add_argument("shares", type=int)
    p_pos.add_argument("--cost-basis", type=float, default=None)
    p_pos.set_defaults(func=cmd_update_position)

    # add-short
    p_short = sub.add_parser("add-short", help="Record a new short call")
    p_short.add_argument("symbol")
    p_short.add_argument("expiry", help="YYYY-MM-DD")
    p_short.add_argument("strike", type=float)
    p_short.add_argument("contracts", type=int)
    p_short.add_argument("premium", type=float)
    p_short.set_defaults(func=cmd_add_short)

    # close-short
    p_close = sub.add_parser("close-short", help="Remove expired/closed short call")
    p_close.add_argument("symbol")
    p_close.add_argument("expiry")
    p_close.add_argument("strike", type=float)
    p_close.add_argument("--status", choices=["expired", "closed", "assigned"], default="expired",
                         help="How the trade was closed (default: expired)")
    p_close.add_argument("--close-price", type=float, default=None,
                         help="Price paid to close (for buy-to-close)")
    p_close.set_defaults(func=cmd_close_short)

    # backtest
    p_bt = sub.add_parser("backtest", help="Simulate covered call strategy on historical data")
    p_bt.add_argument("--symbol", "-s", default="NVDA,TSLA",
                       help="Ticker symbol(s), comma-separated (default: NVDA,TSLA)")
    p_bt.add_argument("--shares", type=int, default=3000,
                       help="Number of shares held (default: 3000)")
    p_bt.add_argument("--months", type=int, default=12,
                       help="Lookback period in months (default: 12)")
    p_bt.add_argument("--delta", type=float, default=0.20,
                       help="Target call delta (default: 0.20)")
    p_bt.add_argument("--dte", type=int, default=7,
                       help="Days to expiration per trade (default: 7)")
    p_bt.add_argument("--weekly", "-w", action="store_true",
                       help="Show week-by-week detail table")
    p_bt.set_defaults(func=cmd_backtest)

    # daily
    p_daily = sub.add_parser("daily", help="Daily review: position advice + alerts")
    p_daily.add_argument("--config", default=None)
    p_daily.set_defaults(func=cmd_daily)

    # optimize
    p_opt = sub.add_parser("optimize", help="Analyze trades and suggest parameter changes")
    p_opt.add_argument("--apply", action="store_true", help="Apply suggestions to config")
    p_opt.set_defaults(func=cmd_optimize)

    # alerts
    p_alerts = sub.add_parser("alerts", help="Check all alert conditions")
    p_alerts.add_argument("--notify", action="store_true", help="Send alerts via Telegram")
    p_alerts.add_argument("--config", default=None)
    p_alerts.set_defaults(func=cmd_alerts)

    # correlation
    p_corr = sub.add_parser("correlation", help="Analyze position correlation & beta")
    p_corr.add_argument("--config", default=None)
    p_corr.set_defaults(func=cmd_correlation)

    # earnings-crush
    # calendar
    # news
    p_news = sub.add_parser("news", help="Portfolio news digest with sentiment")
    p_news.add_argument("--symbol", "-s", default=None, help="Single symbol (default: all portfolio)")
    p_news.add_argument("--config", default=None)
    p_news.set_defaults(func=cmd_news)

    # calendar
    p_cal = sub.add_parser("calendar", help="Upcoming macro events (FOMC, NFP, CPI, etc.)")
    p_cal.add_argument("--days", type=int, default=30, help="Lookahead days (default: 30)")
    p_cal.set_defaults(func=cmd_calendar)

    # earnings-crush
    p_ec = sub.add_parser("earnings-crush", help="Analyze historical IV crush around earnings")
    p_ec.add_argument("--symbol", "-s", default="NVDA,TSLA")
    p_ec.set_defaults(func=cmd_earnings_crush)

    # spreads — multi-leg strategy scanner
    p_spreads = sub.add_parser("spreads", help="Scan for multi-leg option strategy candidates")
    p_spreads.add_argument("--symbol", "-s", required=True,
                           help="Ticker symbol (e.g. NVDA)")
    p_spreads.add_argument("--strategy", default=None,
                           choices=["bull-put", "bear-call", "iron-condor", "collar", "pmcc"],
                           help="Specific strategy (default: scan all)")
    p_spreads.add_argument("--config", default=None)
    p_spreads.set_defaults(func=cmd_spreads)

    # margin — portfolio margin summary & optimization
    p_margin = sub.add_parser("margin", help="Show portfolio margin summary")
    p_margin.add_argument("--optimize", action="store_true",
                          help="Run margin-aware optimization to hit premium target")
    p_margin.add_argument("--target", type=float, default=None,
                          help="Override weekly premium target (default: from config)")
    p_margin.add_argument("--config", default=None)
    p_margin.set_defaults(func=cmd_margin)

    # spark (inline text chart for Remote)
    p_spark = sub.add_parser("spark", help="Inline sparkline chart (shows in CC Remote)")
    p_spark.add_argument("--symbol", "-s", default=None, help="Ticker (default: all portfolio)")
    p_spark.add_argument("--period", "-p", default="3mo", help="1mo, 3mo, 6mo, 1y")
    p_spark.add_argument("--config", default=None)
    p_spark.set_defaults(func=cmd_spark)

    # chart
    p_chart = sub.add_parser("chart", help="Candlestick chart with technicals + trade markers")
    p_chart.add_argument("--symbol", "-s", default="AAPL", help="Ticker symbol")
    p_chart.add_argument("--period", "-p", default="3mo", help="Period: 1mo, 3mo, 6mo, 1y (default: 3mo)")
    p_chart.add_argument("--indicators", "-i", default=None,
                          help="Comma-separated: sma,bb,rsi,macd,volume,atr (default: sma,bb,rsi,volume)")
    p_chart.add_argument("--no-open", action="store_true", help="Don't auto-open in browser")
    p_chart.add_argument("--config", default=None)
    p_chart.set_defaults(func=cmd_chart)

    # iv-surface
    p_iv = sub.add_parser("iv-surface", help="Generate IV surface or smile visualization")
    p_iv.add_argument("--symbol", "-s", required=True, help="Ticker symbol (e.g. NVDA)")
    p_iv.add_argument("--smile", action="store_true",
                       help="Generate 2D IV smile for nearest expiry instead of full 3D surface")
    p_iv.add_argument("--no-open", action="store_true",
                       help="Don't auto-open the HTML file in browser")
    p_iv.set_defaults(func=cmd_iv_surface)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch web dashboard")
    p_dash.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    p_dash.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p_dash.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    p_dash.set_defaults(func=cmd_dashboard)

    # cron
    p_cron = sub.add_parser("cron", help="Manage scheduled scans")
    p_cron.add_argument("action", choices=["install", "remove", "status"])
    p_cron.set_defaults(func=cmd_cron)

    # paper — Alpaca paper trading
    p_paper = sub.add_parser("paper", help="Alpaca paper trading (order validation)")
    paper_sub = p_paper.add_subparsers(dest="paper_action")

    paper_sub.add_parser("status", help="Show account + positions")

    p_paper_submit = paper_sub.add_parser("submit", help="Submit covered call (sell-to-open)")
    p_paper_submit.add_argument("symbol", help="Underlying ticker (e.g. NVDA)")
    p_paper_submit.add_argument("expiry", help="Expiry YYYY-MM-DD")
    p_paper_submit.add_argument("strike", type=float, help="Strike price")
    p_paper_submit.add_argument("contracts", type=int, help="Number of contracts")
    p_paper_submit.add_argument("limit_price", type=float, help="Limit price per contract")

    p_paper_close = paper_sub.add_parser("close", help="Close option (buy-to-close)")
    p_paper_close.add_argument("symbol", help="Underlying ticker")
    p_paper_close.add_argument("expiry", help="Expiry YYYY-MM-DD")
    p_paper_close.add_argument("strike", type=float, help="Strike price")
    p_paper_close.add_argument("contracts", type=int, help="Number of contracts")
    p_paper_close.add_argument("limit_price", type=float, help="Limit price per contract")

    paper_sub.add_parser("orders", help="Show open orders")

    p_paper_cancel = paper_sub.add_parser("cancel", help="Cancel an order")
    p_paper_cancel.add_argument("order_id", help="Alpaca order ID")

    p_paper_chain = paper_sub.add_parser("chain", help="View option chain with Greeks")
    p_paper_chain.add_argument("symbol", help="Underlying ticker")
    p_paper_chain.add_argument("expiry", help="Expiry YYYY-MM-DD")

    p_paper.set_defaults(func=cmd_paper)

    # ml — machine learning signals
    p_ml = sub.add_parser("ml", help="ML signal generation for covered call timing")
    ml_sub = p_ml.add_subparsers(dest="ml_action")

    p_ml_train = ml_sub.add_parser("train", help="Train model on portfolio symbols")
    p_ml_train.add_argument("--months", type=int, default=24,
                            help="Lookback months for training data (default: 24)")
    p_ml_train.add_argument("--delta", type=float, default=0.20,
                            help="Target call delta for labels (default: 0.20)")
    p_ml_train.add_argument("--dte", type=int, default=7,
                            help="Days to expiration for labels (default: 7)")
    p_ml_train.add_argument("--config", default=None)

    p_ml_predict = ml_sub.add_parser("predict", help="Show ML signals for portfolio")
    p_ml_predict.add_argument("--config", default=None)

    p_ml_features = ml_sub.add_parser("features", help="Show current feature values")
    p_ml_features.add_argument("--symbol", "-s", required=True,
                               help="Ticker symbol (e.g. NVDA)")

    p_ml.set_defaults(func=cmd_ml)

    # report
    p_report = sub.add_parser("report", help="Show P&L summaries from trade history")
    p_report.add_argument("period", nargs="?", choices=["weekly", "monthly", "all"],
                          default="weekly", help="Report period (default: weekly)")
    p_report.add_argument("--config", default=None)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
