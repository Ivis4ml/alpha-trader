"""Generate structured briefing text for Claude analysis."""

from __future__ import annotations

import sys
import datetime as dt
from textwrap import dedent

from .config import load_config, get_symbols, contracts_available, get_short_calls, get_delta_range, get_position, get_language, LANGUAGES
from .data.fetcher import (
    fetch_market_context,
    fetch_symbol_briefing,
    MarketContext,
    SymbolBriefing,
)


def _progress(msg: str):
    """Print progress to stderr so it streams in real-time without polluting stdout."""
    print(f"  ⏳ {msg}", file=sys.stderr, flush=True)


def _fmt_option_table(rows, label: str, delta_lo: float, delta_hi: float, top_n: int = 10) -> str:
    """Format top N option candidates into a compact table."""
    if not rows:
        return f"No {label} candidates match filters.\n"

    # Sort by delta proximity to target range midpoint, then filter top N
    target_mid = (delta_lo + delta_hi) / 2
    ranked = sorted(rows, key=lambda r: abs(abs(r.greeks.delta) - target_mid))
    top = ranked[:top_n]
    # Re-sort by expiry/strike for display
    top.sort(key=lambda r: (r.expiry, r.strike))

    off_hours = any(r.off_hours for r in top)
    lines = [f"### {label}"]
    if off_hours:
        lines.append("*(Market closed — last traded prices)*")
    lines.append("")
    lines.append("| Expiry | Strike | Price | Delta | Theta | IV% | OI | Vol | Yield% | OTM% | Fit |")
    lines.append("|--------|--------|-------|-------|-------|-----|-----|-----|--------|------|-----|")

    for r in top:
        d = abs(r.greeks.delta)
        fit = ">>>" if delta_lo <= d <= delta_hi else ""
        price = r.last if r.off_hours else r.bid
        lines.append(
            f"| {r.expiry[5:]} | {r.strike:.1f} | {price:.2f} | "
            f"{d:.2f} | {r.greeks.theta:.3f} | {r.implied_vol:.0f} | "
            f"{r.open_interest} | {r.volume} | {r.annualized_yield:.1f} | {r.otm_pct:.1f} | {fit} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt_symbol_compact(briefing: SymbolBriefing, config: dict, regime: str) -> str:
    """Compact per-symbol section: one-line summary + filtered candidates."""
    s = briefing.stock
    iv = briefing.iv_stats
    ev = briefing.events
    a = briefing.analyst
    pos = get_position(config, s.symbol)
    avail = contracts_available(config, s.symbol)
    delta_lo, delta_hi = get_delta_range(config, regime)
    trend = "↑" if (s.sma_20 and s.price > s.sma_20) else "↓"

    # One-line stock summary
    analyst_str = ""
    if a and a.get("target_mean"):
        analyst_str = f" | Analyst: {(a.get('recommendation') or '?').upper()} → ${a['target_mean']:.0f}"
    earnings_str = f"Earnings: {ev.next_earnings or '?'} ({ev.days_to_earnings or '?'}d)"

    # Technicals one-liner
    tech_str = ""
    t = s.technicals
    if t:
        rsi_label = ""
        if t.rsi_14 is not None:
            if t.rsi_14 > 70: rsi_label = " OVERBOUGHT"
            elif t.rsi_14 < 30: rsi_label = " OVERSOLD"
            rsi_part = f"RSI {t.rsi_14:.0f}{rsi_label}"
        else:
            rsi_part = "RSI ?"

        macd_part = ""
        if t.macd_hist is not None:
            macd_part = f" | MACD {'↑' if t.macd_hist > 0 else '↓'}"

        bb_part = ""
        if t.bb_width is not None:
            squeeze = " SQUEEZE" if t.bb_width < 8 else ""
            bb_part = f" | BB {t.bb_width:.0f}%w{squeeze}"

        atr_part = ""
        if t.atr_14 is not None:
            atr_part = f" | ATR ${t.atr_14:.2f}"

        tech_str = f"\n**Technicals:** {rsi_part}{macd_part}{bb_part}{atr_part}"

    # Unusual options activity (only if any found)
    unusual_str = ""
    if briefing.unusual_activity:
        alerts = []
        for ua in briefing.unusual_activity[:5]:
            ratio_s = f"{ua['ratio']}x" if ua.get("ratio") is not None else "HIGH"
            alerts.append(f"{ua['type'].upper()} ${ua['strike']:.0f} vol={ua['volume']} OI={ua['open_interest']} ({ratio_s})")
        unusual_str = f"\n**Unusual Activity:** {' | '.join(alerts)}"

    # Put/call ratio
    pcr_str = ""
    if briefing.put_call_ratio is not None:
        sentiment = "BEARISH" if briefing.put_call_ratio > 1.0 else ("BULLISH" if briefing.put_call_ratio < 0.7 else "NEUTRAL")
        pcr_str = f"\n**P/C Ratio:** {briefing.put_call_ratio:.2f} ({sentiment})"

    header = (
        f"## {s.symbol} — ${s.price:.2f} ({'+' if s.day_change_pct >= 0 else ''}"
        f"{s.day_change_pct:.1f}%) {trend}20SMA | "
        f"IV Rank {iv.iv_rank:.0f}% | {earnings_str} | "
        f"{pos.get('shares', 0):,} shares ({avail} contracts avail)"
        f"{analyst_str}{tech_str}{unusual_str}{pcr_str}\n"
    )

    # News + sentiment
    news = ""
    if briefing.news:
        from .data.news import score_news_sentiment
        sentiment = score_news_sentiment(briefing.news)
        sent_label = sentiment.get("label", "NEUTRAL")
        sent_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(sent_label, "")
        news = f"**News {sent_icon}{sent_label}:** " + " | ".join(n['title'][:60] for n in briefing.news[:3]) + "\n\n"

    # Insider (compact)
    insider = ""
    if briefing.insider:
        recent = briefing.insider[:3]
        insider = "**Insider:** " + " | ".join(
            f"{i['date']}: {i['transaction'][:40]}" for i in recent
        ) + "\n\n"

    # Alpha Vantage fundamentals (only in full mode — when news is present)
    av_fund_str = ""
    if briefing.news:  # proxy for full mode
        try:
            from .data.alphavantage import fetch_fundamentals, format_fundamentals
            fund = fetch_fundamentals(s.symbol)
            av_fund_str = format_fundamentals(fund, s.symbol)
            if av_fund_str:
                av_fund_str += "\n\n"
        except Exception:
            pass

    calls = _fmt_option_table(briefing.call_chains, "Call Candidates", delta_lo, delta_hi)
    puts = _fmt_option_table(briefing.put_chains, "Put Candidates", delta_lo, delta_hi, top_n=5)

    return header + "\n" + av_fund_str + news + insider + calls + "\n" + puts


def generate_briefing(config: dict | None = None, quick: bool = False) -> str:
    """Generate the full market briefing text. quick=True skips news/insider/analyst/puts."""
    if config is None:
        config = load_config()

    symbols = get_symbols(config)

    _progress("Market context (VIX, SPY)...")
    mkt = fetch_market_context(config)
    delta_lo, delta_hi = get_delta_range(config, mkt.regime)
    regime_info = config.get("strategy", {}).get("regimes", {}).get(mkt.regime, {})

    short_calls = get_short_calls(config)
    short_calls_text = "None"
    if short_calls:
        short_calls_text = "\n".join(
            f"  - {sc['symbol']} {sc['expiry']} ${sc['strike']}C x{sc['contracts']} "
            f"(${sc.get('premium_received', '?')})"
            for sc in short_calls
        )

    # Macro events
    from .data.events_calendar import fetch_macro_calendar
    macro_events = fetch_macro_calendar(lookahead_days=21)
    macro_str = ""
    if macro_events:
        upcoming = [f"{e.event} {e.date[5:]} ({e.days_away}d)" for e in macro_events[:5]]
        macro_str = f"\n**Upcoming:** {' | '.join(upcoming)}"

    from .portfolio import load_portfolio, get_weekly_target
    pf = load_portfolio()
    target = get_weekly_target(pf)

    # Rolling average from trade history
    rolling_avg_str = ""
    try:
        from .db import get_weekly_summary
        recent_weeks = get_weekly_summary(weeks=8)
        if recent_weeks:
            premiums = [w["total_premium"] or 0 for w in recent_weeks]
            rolling_avg = sum(premiums) / len(premiums)
            pct_of_target = (rolling_avg / target * 100) if target else 0
            pace = "ON TRACK" if pct_of_target >= 85 else "BEHIND" if pct_of_target >= 50 else "WELL BEHIND"
            rolling_avg_str = f" | Rolling avg: ${rolling_avg:,.0f}/wk ({pct_of_target:.0f}% — {pace})"
    except Exception:
        pass

    # Alpha Vantage macro data (if available)
    av_macro_str = ""
    if not quick:
        try:
            from .data.alphavantage import fetch_macro_snapshot, format_macro_snapshot
            _progress("Macro indicators (AV)...")
            macro_data = fetch_macro_snapshot()
            av_macro_str = "\n" + format_macro_snapshot(macro_data)
        except Exception:
            pass

    header = dedent(f"""\
    # Alpha Trader Briefing — {mkt.timestamp}
    VIX {mkt.vix} ({'+' if mkt.vix_change >= 0 else ''}{mkt.vix_change}) | SPY ${mkt.spy_price} ({'+' if mkt.spy_change_pct >= 0 else ''}{mkt.spy_change_pct}%) {'↑' if mkt.spy_above_sma20 else '↓'}20SMA | Regime: **{mkt.regime.upper()}** (Δ {delta_lo:.2f}–{delta_hi:.2f})
    Target: ${target:,}/wk (yearly avg){rolling_avg_str} | Open shorts: {short_calls_text}{macro_str}{av_macro_str}

    ---
    """)

    # Fetch all symbols in parallel
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch_one(sym):
        _progress(f"{sym}...")
        return sym, fetch_symbol_briefing(sym, config, quick=quick)

    briefings = {}
    with ThreadPoolExecutor(max_workers=len(symbols)) as pool:
        futures = {pool.submit(_fetch_one, s): s for s in symbols}
        for f in as_completed(futures):
            sym, b = f.result()
            briefings[sym] = b
            _progress(f"{sym} done.")

    sections = [_fmt_symbol_compact(briefings[s], config, mkt.regime) for s in symbols]

    # Roll analysis
    roll_section = ""
    if short_calls:
        _progress("Roll analysis...")
        from .roll import analyze_rolls
        roll_section = "\n" + analyze_rolls(config) + "\n---\n\n"

    _progress("Done fetching. Generating analysis...")
    policy = _get_policy(config, mkt)

    return header + "\n".join(sections) + "\n---\n\n" + roll_section + policy


def _get_policy(config: dict, mkt: MarketContext) -> str:
    strat = config.get("strategy", {})
    delta_lo, delta_hi = get_delta_range(config, mkt.regime)

    return dedent(f"""\
    ## INSTRUCTIONS FOR CLAUDE

    You are Alpha Trader. Analyze the data above and output in **{LANGUAGES.get(get_language(config), 'English')}**, in EXACTLY this format.
    Be concise. Tables first, explanations only if asked.

    ### Rules
    - Covered calls on existing shares. RSU positions — minimize assignment risk.
    - Income goal: ${strat.get('weekly_target', 1500):,}/wk **as a yearly rolling average** (~${strat.get('weekly_target', 1500) * 52:,}/yr). Individual weeks can miss — do NOT force bad trades to hit the weekly number. Judge success by trailing 4-8 week average. If behind pace, lean slightly more aggressive (within regime bounds); if ahead, preserve gains. HARD CAP: never recommend more than ${int(strat.get('weekly_target', 1500) * strat.get('catchup_cap_pct', 200) / 100):,}/wk ({strat.get('catchup_cap_pct', 200)}% of target) in a single week — chasing losses destroys risk management.
    - Regime: {mkt.regime.upper()} → delta {delta_lo:.2f}–{delta_hi:.2f}
    - IV rank < 30 → lean lower delta. IV rank > 50 → lean higher delta.
    - RSI > 70 (overbought): stock may pull back — use higher strike or skip.
    - RSI < 30 (oversold): stock may bounce — sell further OTM.
    - MACD ↑ (bullish momentum): use higher strikes to avoid assignment.
    - BB SQUEEZE (narrow Bollinger): breakout likely — be conservative or wait.
    - ATR guides OTM distance: strike should be ≥1 ATR above current price ideally.
    - No calls within {strat.get('blackout_earnings_days', 7)} days of earnings.
    - DTE range: {strat.get('dte_range', [3, 21])[0]}–{strat.get('dte_range', [3, 21])[1]}
    - Prefer ">>>" rows (in target delta range), high OI, tight spread.
    - Leave buffer — don't sell on all available contracts.
    - Use BID price (or last traded if market closed) for premium estimates.

    ### Output format (STRICT)

    ```
    ## Positions
    | Symbol | Shares | Open Shorts | Avail Contracts |
    |--------|--------|-------------|-----------------|
    [fill from data]

    ## Trades
    | # | Action | Expiry | Strike | Contracts | Premium | Delta | OTM% |
    |---|--------|--------|--------|-----------|---------|-------|------|
    [each recommended trade as one row]
    | | | | **Total** | | **$X,XXX** | | |

    This week: $X,XXX | Rolling avg: $X,XXX/wk (X% of target) — ON TRACK or BEHIND PACE

    ## Risks
    [2-3 bullet points max: earnings proximity, IV environment, trend concerns]

    ## Open Position Status
    [only if there are existing short calls — roll/hold/close recommendations]

    ---
    _Ask "why [symbol] [strike]?" or "explain" for detailed reasoning on any trade._
    ```

    Do NOT add lengthy explanations unless the user asks. Keep the entire response under 40 lines.
    """)
