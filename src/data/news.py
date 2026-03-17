"""Fetch news, insider transactions, and analyst data from Yahoo Finance."""

from __future__ import annotations

import yfinance as yf


def fetch_news(symbol: str, max_items: int = 5) -> list[dict]:
    """Recent news headlines for a symbol."""
    t = yf.Ticker(symbol)
    try:
        raw = t.news or []
        results = []
        for n in raw[:max_items]:
            # yfinance >= 0.2.31 uses nested content structure
            content = n.get("content", {})
            title = content.get("title") or n.get("title", "")
            publisher = (content.get("provider", {}).get("displayName")
                         or n.get("publisher", ""))
            link = (content.get("canonicalUrl", {}).get("url")
                    or n.get("link", ""))
            if title:
                results.append({"title": title, "publisher": publisher, "link": link})
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
