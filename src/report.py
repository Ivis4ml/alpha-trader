"""Generate structured briefing text for Claude analysis.

report.py is a pure render layer — it takes scan_engine output and formats
it into markdown that both humans (--data-only) and Claude can read.
"""

from __future__ import annotations

import sys
import datetime as dt
from textwrap import dedent

from .config import (
    load_config, get_symbols, contracts_available, get_short_calls,
    get_delta_range, get_position, get_language, get_weekly_target, LANGUAGES,
)
from .data.fetcher import MarketContext, SymbolBriefing
from .scan_engine import (
    CandidateDecision, SymbolScanResult, PortfolioScanResult, scan_portfolio,
)


def _progress(msg: str):
    print(f"  ⏳ {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Option table formatting
# ---------------------------------------------------------------------------

def _fmt_scored_table(candidates: list[CandidateDecision], label: str, top_n: int = 10) -> str:
    """Format scored call candidates into a table with score breakdown + flags."""
    if not candidates:
        return f"No {label} match filters.\n"

    top = candidates[:top_n]
    off_hours = any(d.features.off_hours for d in top)

    lines = [f"### {label}"]
    if off_hours:
        lines.append("*(Market closed — last traded prices)*")
    lines.append("")
    lines.append(
        "| Expiry | Strike | Price | Score | Income | Risk | Quality | Event "
        "| Delta | Theta | Yield% | OTM% | ATR | Flags |"
    )
    lines.append(
        "|--------|--------|-------|-------|--------|------|---------|-------"
        "|-------|-------|--------|------|-----|-------|"
    )

    for d in top:
        f = d.features
        b = d.breakdown
        atr_str = f"{f.atr_distance:.1f}" if f.atr_distance is not None else "?"
        flags_str = " ".join(d.flags)
        lines.append(
            f"| {f.expiry[5:]} | {f.strike:.1f} | {f.premium:.2f} "
            f"| {int(d.score * 100):>3d} "
            f"| {b.get('income', 0):.2f} | {b.get('assignment_risk', 0):.2f} "
            f"| {b.get('execution_quality', 0):.2f} | {b.get('event_risk', 0):.2f} "
            f"| {f.delta:.2f} | {f.theta:.3f} | {f.annualized_yield:.1f} "
            f"| {f.otm_pct:.1f} | {atr_str} | {flags_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def _fmt_option_table(rows, label: str, delta_lo: float, delta_hi: float, top_n: int = 5) -> str:
    """Format put candidates (unscored, secondary strategy)."""
    if not rows:
        return f"No {label} candidates match filters.\n"

    target_mid = (delta_lo + delta_hi) / 2
    ranked = sorted(rows, key=lambda r: abs(abs(r.greeks.delta) - target_mid))
    top = ranked[:top_n]
    top.sort(key=lambda r: (r.expiry, r.strike))

    lines = [f"### {label}", ""]
    lines.append("| Expiry | Strike | Price | Delta | Theta | IV% | OI | Yield% | OTM% |")
    lines.append("|--------|--------|-------|-------|-------|-----|-----|--------|------|")

    for r in top:
        d = abs(r.greeks.delta)
        price = r.last if r.off_hours else r.bid
        lines.append(
            f"| {r.expiry[5:]} | {r.strike:.1f} | {price:.2f} | "
            f"{d:.2f} | {r.greeks.theta:.3f} | {r.implied_vol:.0f} | "
            f"{r.open_interest} | {r.annualized_yield:.1f} | {r.otm_pct:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-symbol section
# ---------------------------------------------------------------------------

def _fmt_symbol_section(result: SymbolScanResult, config: dict, regime: str) -> str:
    """Render one symbol: header + context + scored call table + put table."""
    briefing = result.briefing
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

    # Unusual options activity
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

    # Symbol-level scan flags
    sym_flags_str = ""
    if result.symbol_flags:
        sym_flags_str = f"\n**Scan:** {' | '.join(result.symbol_flags)} | {result.rejected_count} candidates filtered"

    header = (
        f"## {s.symbol} — ${s.price:.2f} ({'+' if s.day_change_pct >= 0 else ''}"
        f"{s.day_change_pct:.1f}%) {trend}20SMA | "
        f"IV Rank {iv.iv_rank:.0f}% | {earnings_str} | "
        f"{pos.get('shares', 0):,} shares ({avail} contracts avail)"
        f"{analyst_str}{tech_str}{unusual_str}{pcr_str}{sym_flags_str}\n"
    )

    # News + sentiment
    news = ""
    if briefing.news:
        from .data.news import score_news_sentiment
        all_news = list(briefing.news)

        try:
            from .data.alphavantage import fetch_news_sentiment as _av_news
            av_items = _av_news(s.symbol, max_items=3)
            seen = {n["title"][:40].lower() for n in all_news}
            for av in av_items:
                if av["title"][:40].lower() not in seen:
                    all_news.append(av)
                    seen.add(av["title"][:40].lower())
        except Exception:
            pass

        sentiment = score_news_sentiment(all_news)
        sent_label = sentiment.get("label", "NEUTRAL")
        sent_icon = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "⚪"}.get(sent_label, "")

        headlines = []
        for n in all_news[:5]:
            age = n.get("age", "")
            title = n["title"][:60]
            headlines.append(f"{title} ({age})" if age else title)
        news = f"**News {sent_icon}{sent_label}:** " + " | ".join(headlines) + "\n\n"

    # Insider
    insider = ""
    if briefing.insider:
        recent = briefing.insider[:3]
        insider = "**Insider:** " + " | ".join(
            f"{i['date']}: {i['transaction'][:40]}" for i in recent
        ) + "\n\n"

    # AV fundamentals
    av_fund_str = ""
    if briefing.news:
        try:
            from .data.alphavantage import fetch_fundamentals, format_fundamentals
            fund = fetch_fundamentals(s.symbol)
            av_fund_str = format_fundamentals(fund, s.symbol)
            if av_fund_str:
                av_fund_str += "\n\n"
        except Exception:
            pass

    # Scored call candidates (from scan_engine)
    calls = _fmt_scored_table(result.candidates, "Call Candidates")

    # Puts (secondary, unscored)
    puts = _fmt_option_table(briefing.put_chains, "Put Candidates", delta_lo, delta_hi)

    return header + "\n" + av_fund_str + news + insider + calls + "\n" + puts


# ---------------------------------------------------------------------------
# Full briefing
# ---------------------------------------------------------------------------

def generate_briefing(config: dict | None = None, quick: bool = False) -> str:
    """Generate the full market briefing text via scan_engine."""
    if config is None:
        config = load_config()

    # Run the full scan pipeline (fetch + filter + score)
    scan_result = scan_portfolio(config, quick=quick)
    mkt = scan_result.market

    delta_lo, delta_hi = get_delta_range(config, mkt.regime)
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

    target = get_weekly_target(config)

    # Rolling average
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

    # AV macro
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

    # Per-symbol sections from scan results
    symbols = list(scan_result.symbols.keys())
    sections = [_fmt_symbol_section(scan_result.symbols[s], config, mkt.regime) for s in symbols]

    # Roll analysis
    roll_section = ""
    if short_calls:
        _progress("Roll analysis...")
        from .roll import analyze_rolls
        roll_section = "\n" + analyze_rolls(config, regime=mkt.regime) + "\n---\n\n"

    _progress("Done. Generating analysis...")
    policy = _get_policy(config, mkt)

    return header + "\n".join(sections) + "\n---\n\n" + roll_section + policy


def _get_policy(config: dict, mkt: MarketContext) -> str:
    strat = config.get("strategy", {})
    delta_lo, delta_hi = get_delta_range(config, mkt.regime)
    target = get_weekly_target(config)
    catchup_cap_pct = strat.get('catchup_cap_pct', 200)

    return dedent(f"""\
    ## INSTRUCTIONS FOR CLAUDE

    You are Alpha Trader. You are given:
    1. **Hard-filtered** covered-call candidates (earnings blackout, ATR floor, cost basis constraints already enforced — these cannot be overridden)
    2. A **composite score and sub-scores** (Income, Assignment Risk, Execution Quality, Event Risk) for each candidate
    3. **Technical, event, news, and portfolio context** for your analysis

    Use both the scoring model and your own analysis to produce the final recommendation.
    Do not violate hard constraints — filtered candidates are excluded for a reason.
    If you choose a lower-scored candidate or skip a high-scored one, explain why.
    Output in **{LANGUAGES.get(get_language(config), 'English')}**.

    ### Rules
    - Covered calls on existing shares. RSU positions — minimize assignment risk.
    - Income goal: ${target:,.0f}/wk **as a yearly rolling average** (~${target * 52:,.0f}/yr). Individual weeks can miss — do NOT force bad trades to hit the weekly number. Judge success by trailing 4-8 week average. If behind pace, lean slightly more aggressive (within regime bounds); if ahead, preserve gains. HARD CAP: never recommend more than ${int(target * catchup_cap_pct / 100):,}/wk ({catchup_cap_pct}% of target) in a single week.
    - Regime: {mkt.regime.upper()} → delta {delta_lo:.2f}–{delta_hi:.2f}
    - IV rank < 30 → lean lower delta. IV rank > 50 → lean higher delta.
    - RSI > 70 (overbought): consider higher strike or skip. RSI < 30 (oversold): sell further OTM.
    - MACD ↑ (bullish momentum): higher strikes. BB SQUEEZE: be conservative or wait.
    - Prefer >>> rows (in target delta range). Leave buffer — don't sell on all available contracts.
    - Use BID price (or last traded if market closed) for premium estimates.
    ### How to read the Score table
    - **Score** (0-100): composite rank. **Income**: premium yield + theta. **Risk**: delta + ATR + OTM safety.
    - **Quality**: spread + OI liquidity. **Event**: macro/earnings window safety.
    - **Flags**: >>> = delta in range, EVENT_RISK, EARNINGS_NEAR, LOW_OI, WIDE_SPREAD.
    - Use the breakdown to understand *why* a candidate scored the way it did, then combine with your own analysis.

    ### Output format (STRICT)

    ```
    ## Positions
    | Symbol | Shares | Open Shorts | Avail Contracts |
    |--------|--------|-------------|-----------------|
    [fill from data]

    ## Trades
    | # | Action | Expiry | Strike | Contracts | Premium | Delta | Score | OTM% |
    |---|--------|--------|--------|-----------|---------|-------|-------|------|
    [each recommended trade as one row]
    | | | | **Total** | | **$X,XXX** | | | |

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
