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
