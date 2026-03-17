"""Macro economic events calendar + stock-specific event detection.

Fetches and flags market-moving events that affect options selling decisions:
- FOMC meetings (rate decisions → vol spike)
- NFP / jobs report (first Friday of month)
- CPI / inflation data
- Tariff / trade policy announcements
- Earnings dates (per-stock)
- Ex-dividend dates
- Government policy / regulatory events

Data sources: Yahoo Finance calendar, hardcoded FOMC/NFP schedule, news keyword scanning.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class MacroEvent:
    date: str           # YYYY-MM-DD
    event: str          # NFP, FOMC, CPI, TARIFF, etc.
    description: str
    impact: str         # HIGH, MEDIUM, LOW
    days_away: int


@dataclass
class EventRisk:
    """Aggregated event risk for a symbol + expiry window."""
    risk_level: str     # HIGH, MEDIUM, LOW, NONE
    risk_score: float   # 0.0 (no risk) to 1.0 (extreme risk)
    events: list[MacroEvent]
    warnings: list[str]


# ── FOMC Schedule (hardcoded, update annually) ──────────────────────────────
# These are the announcement dates. The actual 2-day meetings end on these dates.
FOMC_DATES_2025_2026 = [
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
    # 2026
    "2026-01-28", "2026-03-18", "2026-05-06", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
]

# ── NFP: First Friday of each month ────────────────────────────────────────
def _first_fridays(year: int) -> list[str]:
    dates = []
    for month in range(1, 13):
        d = dt.date(year, month, 1)
        # Find first Friday
        while d.weekday() != 4:  # 4 = Friday
            d += dt.timedelta(days=1)
        dates.append(d.isoformat())
    return dates

# ── CPI: Usually ~10th-13th of each month ──────────────────────────────────
def _approx_cpi_dates(year: int) -> list[str]:
    # CPI is typically released on the 2nd Tuesday-Wednesday of the month
    dates = []
    for month in range(1, 13):
        d = dt.date(year, month, 10)  # approximate
        # Adjust to nearest weekday
        while d.weekday() > 4:
            d += dt.timedelta(days=1)
        dates.append(d.isoformat())
    return dates


def fetch_macro_calendar(lookahead_days: int = 30) -> list[MacroEvent]:
    """Return upcoming macro events within the lookahead window."""
    today = dt.date.today()
    cutoff = today + dt.timedelta(days=lookahead_days)
    events = []

    # FOMC
    for d in FOMC_DATES_2025_2026:
        ed = dt.date.fromisoformat(d)
        if today <= ed <= cutoff:
            events.append(MacroEvent(
                date=d,
                event="FOMC",
                description="Federal Reserve rate decision — expect IV spike around announcement",
                impact="HIGH",
                days_away=(ed - today).days,
            ))

    # NFP (first Friday)
    for year in [today.year, today.year + 1]:
        for d in _first_fridays(year):
            ed = dt.date.fromisoformat(d)
            if today <= ed <= cutoff:
                events.append(MacroEvent(
                    date=d,
                    event="NFP",
                    description="Non-Farm Payrolls (jobs report) — market-moving, expect morning volatility",
                    impact="HIGH",
                    days_away=(ed - today).days,
                ))

    # CPI
    for year in [today.year, today.year + 1]:
        for d in _approx_cpi_dates(year):
            ed = dt.date.fromisoformat(d)
            if today <= ed <= cutoff:
                events.append(MacroEvent(
                    date=d,
                    event="CPI",
                    description="Consumer Price Index (inflation) — can shift rate expectations",
                    impact="MEDIUM",
                    days_away=(ed - today).days,
                ))

    events.sort(key=lambda e: e.date)
    return events


def scan_news_for_risks(news: list[dict]) -> list[MacroEvent]:
    """Scan news headlines for risk keywords (tariffs, regulation, policy)."""
    risk_keywords = {
        "HIGH": [
            "tariff", "trade war", "ban", "sanction", "antitrust", "lawsuit",
            "SEC investigation", "recall", "data breach", "downgrade",
            "guidance cut", "layoff", "restructur",
        ],
        "MEDIUM": [
            "regulation", "policy", "government", "congress", "legislation",
            "probe", "inquiry", "fine", "penalty", "trade deal",
            "executive order", "import duty", "export control",
        ],
        "LOW": [
            "analyst", "upgrade", "rating", "acquisition", "merger",
            "partnership", "expansion", "launch",
        ],
    }

    events = []
    today = dt.date.today().isoformat()

    for article in news:
        title = (article.get("title") or "").lower()
        for severity, keywords in risk_keywords.items():
            for kw in keywords:
                if kw.lower() in title:
                    events.append(MacroEvent(
                        date=today,
                        event="NEWS",
                        description=article.get("title", "")[:80],
                        impact=severity,
                        days_away=0,
                    ))
                    break
            else:
                continue
            break

    return events


def assess_event_risk(
    symbol: str,
    expiry: str,
    news: list[dict] | None = None,
    earnings_dte: int | None = None,
    earnings_buffer: int = 7,
) -> EventRisk:
    """Assess aggregate event risk for selling a call with given expiry.

    Checks:
    1. Macro events (FOMC, NFP, CPI) falling before expiry
    2. News-based risks (tariffs, regulation, etc.)
    3. Earnings proximity
    """
    today = dt.date.today()
    exp_date = dt.date.fromisoformat(expiry)
    dte = (exp_date - today).days

    all_events: list[MacroEvent] = []
    warnings: list[str] = []

    # 1. Macro calendar
    macro = fetch_macro_calendar(lookahead_days=dte + 5)
    for e in macro:
        ed = dt.date.fromisoformat(e.date)
        if today <= ed <= exp_date:
            all_events.append(e)
            if e.impact == "HIGH":
                warnings.append(f"{e.event} on {e.date} ({e.days_away}d) — {e.description[:60]}")

    # 2. News risks
    if news:
        news_risks = scan_news_for_risks(news)
        all_events.extend(news_risks)
        for nr in news_risks:
            if nr.impact == "HIGH":
                warnings.append(f"NEWS: {nr.description[:70]}")

    # 3. Earnings
    if earnings_dte is not None and 0 <= earnings_dte <= dte:
        all_events.append(MacroEvent(
            date=(today + dt.timedelta(days=earnings_dte)).isoformat(),
            event="EARNINGS",
            description=f"{symbol} earnings in {earnings_dte} days — within expiry window",
            impact="HIGH",
            days_away=earnings_dte,
        ))
        warnings.append(f"EARNINGS in {earnings_dte}d — consider avoiding or shorter DTE")

    # Calculate risk score
    high_count = sum(1 for e in all_events if e.impact == "HIGH")
    med_count = sum(1 for e in all_events if e.impact == "MEDIUM")
    risk_score = min(1.0, high_count * 0.35 + med_count * 0.15)

    if risk_score >= 0.5:
        risk_level = "HIGH"
    elif risk_score >= 0.2:
        risk_level = "MEDIUM"
    elif risk_score > 0:
        risk_level = "LOW"
    else:
        risk_level = "NONE"

    return EventRisk(
        risk_level=risk_level,
        risk_score=round(risk_score, 2),
        events=all_events,
        warnings=warnings,
    )


def format_event_calendar(events: list[MacroEvent]) -> str:
    if not events:
        return "No major macro events in the next 30 days."
    lines = ["## Upcoming Macro Events\n"]
    lines.append("| Date | Event | Impact | Days | Description |")
    lines.append("|------|-------|--------|------|-------------|")
    for e in events:
        icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}.get(e.impact, "")
        lines.append(f"| {e.date} | {e.event} | {icon} {e.impact} | {e.days_away}d | {e.description[:50]} |")
    return "\n".join(lines)
