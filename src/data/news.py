"""Fetch news, insider transactions, and analyst data from Yahoo Finance."""

from __future__ import annotations

import datetime as dt

import yfinance as yf

# Only keep news published within this window
_MAX_AGE_HOURS = 48


def _parse_pub_date(content: dict, item: dict) -> dt.datetime | None:
    """Extract publish time from yfinance news item."""
    # New format (>= 0.2.31): ISO string in content.pubDate
    raw = content.get("pubDate") or item.get("providerPublishTime")
    if isinstance(raw, str):
        try:
            return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(raw, (int, float)):
        # Legacy format: unix timestamp
        return dt.datetime.fromtimestamp(raw, tz=dt.timezone.utc)
    return None


def _format_age(published: dt.datetime) -> str:
    """Format a publish time as a human-readable age string like '2h ago'."""
    delta = dt.datetime.now(dt.timezone.utc) - published
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() / 60)}m ago"
    if hours < 24:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def fetch_news(symbol: str, max_items: int = 8) -> list[dict]:
    """Recent news headlines for a symbol (filtered to last 48h)."""
    t = yf.Ticker(symbol)
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=_MAX_AGE_HOURS)
    try:
        raw = t.news or []
        results = []
        for n in raw:
            content = n.get("content", {})
            title = content.get("title") or n.get("title", "")
            publisher = (content.get("provider", {}).get("displayName")
                         or n.get("publisher", ""))
            link = (content.get("canonicalUrl", {}).get("url")
                    or n.get("link", ""))
            published = _parse_pub_date(content, n)

            if not title:
                continue
            # Filter out stale articles
            if published and published < cutoff:
                continue

            age_str = _format_age(published) if published else ""
            results.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "published": published.isoformat() if published else "",
                "age": age_str,
            })
            if len(results) >= max_items:
                break
        return results
    except Exception:
        return []


def fetch_insider_transactions(symbol: str, max_items: int = 5) -> list[dict]:
    """Recent insider buys/sells."""
    t = yf.Ticker(symbol)
    try:
        df = t.insider_transactions
        if df is None or df.empty:
            return []
        results = []
        for _, row in df.head(max_items).iterrows():
            shares = row.get("Shares", row.get("shares", 0))
            try:
                shares = int(shares) if shares and str(shares) != "nan" else 0
            except (ValueError, TypeError):
                shares = 0
            results.append({
                "insider": str(row.get("Insider Trading", row.get("insider", ""))),
                "transaction": str(row.get("Text", row.get("transaction", ""))),
                "shares": shares,
                "date": str(row.get("Start Date", row.get("startDate", "")))[:10],
            })
        return results
    except Exception:
        return []


def score_news_sentiment(news: list[dict]) -> dict:
    """Simple keyword-based sentiment scoring for news headlines.

    Returns:
        score: -1.0 (very bearish) to +1.0 (very bullish)
        label: BULLISH, BEARISH, NEUTRAL
        signals: list of detected signals
    """
    bullish_kw = [
        "beat", "exceed", "upgrade", "raise", "growth", "record revenue",
        "strong demand", "outperform", "buy rating", "bullish", "surge",
        "breakout", "all-time high", "expansion", "partnership", "deal",
        "approval", "launches", "innovation", "ai growth",
    ]
    bearish_kw = [
        "miss", "disappoint", "downgrade", "cut", "decline", "weak",
        "layoff", "restructur", "lawsuit", "investigation", "recall",
        "tariff", "ban", "sanction", "sell rating", "bearish", "crash",
        "warning", "guidance lower", "debt concern", "default",
        "trade war", "antitrust", "fine", "penalty",
    ]

    bull_hits = []
    bear_hits = []

    for article in news:
        title = (article.get("title") or "").lower()
        for kw in bullish_kw:
            if kw in title:
                bull_hits.append(kw)
                break
        for kw in bearish_kw:
            if kw in title:
                bear_hits.append(kw)
                break

    total = len(bull_hits) + len(bear_hits)
    if total == 0:
        return {"score": 0.0, "label": "NEUTRAL", "signals": []}

    score = (len(bull_hits) - len(bear_hits)) / total
    label = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")

    signals = []
    if bull_hits:
        signals.append(f"+{len(bull_hits)} bullish: {', '.join(set(bull_hits))}")
    if bear_hits:
        signals.append(f"-{len(bear_hits)} bearish: {', '.join(set(bear_hits))}")

    return {"score": round(score, 2), "label": label, "signals": signals}


def fetch_analyst_data(symbol: str) -> dict:
    """Analyst price targets and recommendation."""
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
        return {
            "target_mean": info.get("targetMeanPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_high": info.get("targetHighPrice"),
            "recommendation": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions"),
        }
    except Exception:
        return {}
