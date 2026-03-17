"""Telegram push notifications."""

from __future__ import annotations

import os
import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def _get_creds() -> tuple[str, str]:
    return (
        os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        os.environ.get("TELEGRAM_CHAT_ID", ""),
    )


def _split(text: str, limit: int = 4096) -> list[str]:
    """Split long text into Telegram-safe chunks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut == -1:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


def send_telegram(text: str, parse_mode: str | None = "Markdown") -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    token, chat_id = _get_creds()
    if not token or not chat_id:
        print("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see .env.example)")
        return False

    for chunk in _split(text):
        url = TELEGRAM_API.format(token=token, method="sendMessage")
        payload: dict = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        resp = requests.post(url, json=payload, timeout=15)
        if not resp.ok:
            # Retry without parse_mode (markdown can fail on special chars)
            payload.pop("parse_mode", None)
            resp = requests.post(url, json=payload, timeout=15)
            if not resp.ok:
                print(f"Telegram error: {resp.status_code} {resp.text[:200]}")
                return False
    return True


def send_report(report: str) -> bool:
    """Send a full report — tries Markdown first, falls back to plain."""
    return send_telegram(report)


def verify_bot() -> dict | None:
    """Check if bot token is valid. Returns bot info or None."""
    token, _ = _get_creds()
    if not token:
        return None
    try:
        url = TELEGRAM_API.format(token=token, method="getMe")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        return data.get("result") if data.get("ok") else None
    except Exception:
        return None
