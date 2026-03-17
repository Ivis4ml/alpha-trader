"""Interactive Telegram bot for Alpha Trader.

Run as a daemon:  python -m src.bot
Supports commands: /scan /data /positions /roll /help
"""

from __future__ import annotations

import os
import subprocess
import time

import requests

from .config import load_config, get_short_calls

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


def _call(token: str, method: str, **kwargs) -> dict:
    url = TELEGRAM_API.format(token=token, method=method)
    resp = requests.post(url, json=kwargs, timeout=60)
    return resp.json()


def _send(token: str, chat_id: int | str, text: str, parse_mode: str | None = "Markdown"):
    """Send message, split if >4096 chars."""
    while text:
        chunk = text[:4096]
        if len(text) > 4096:
            cut = chunk.rfind("\n")
            if cut > 0:
                chunk = text[:cut]
        text = text[len(chunk):].lstrip("\n")

        payload: dict = {"chat_id": chat_id, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        resp = _call(token, "sendMessage", **payload)
        if not resp.get("ok") and parse_mode:
            # Fallback: strip markdown
            payload.pop("parse_mode", None)
            _call(token, "sendMessage", **payload)


def _handle(token: str, chat_id: int | str, text: str):
    cmd = text.strip().split()[0].lower()
    config = load_config()

    if cmd == "/scan":
        _send(token, chat_id, "Scanning... (30-60s)", None)
        try:
            from .report import generate_briefing
            briefing = generate_briefing(config)
            result = subprocess.run(
                ["claude", "-p", "--model", "claude-opus-4-6", briefing],
                capture_output=True, text=True, timeout=180,
            )
            output = result.stdout.strip() if result.returncode == 0 else ""
            _send(token, chat_id, output or briefing, None)
        except Exception as e:
            _send(token, chat_id, f"Scan error: {e}", None)

    elif cmd == "/data":
        _send(token, chat_id, "Fetching raw data...", None)
        try:
            from .report import generate_briefing
            briefing = generate_briefing(config)
            _send(token, chat_id, briefing, None)
        except Exception as e:
            _send(token, chat_id, f"Error: {e}", None)

    elif cmd in ("/positions", "/status"):
        lines = ["Portfolio Status\n"]
        for sym, pos in config.get("positions", {}).items():
            lines.append(f"  {sym}: {pos.get('shares', 0):,} shares")
        shorts = get_short_calls(config)
        if shorts:
            lines.append("\nOpen Short Calls:")
            for sc in shorts:
                lines.append(
                    f"  {sc['symbol']} {sc['expiry']} ${sc['strike']} "
                    f"x{sc['contracts']} (rcvd ${sc.get('premium_received', '?')})"
                )
        else:
            lines.append("\nNo open short calls.")
        _send(token, chat_id, "\n".join(lines), None)

    elif cmd == "/roll":
        _send(token, chat_id, "Checking roll candidates...", None)
        try:
            from .roll import analyze_rolls
            report = analyze_rolls(config)
            _send(token, chat_id, report, None)
        except Exception as e:
            _send(token, chat_id, f"Roll error: {e}", None)

    elif cmd == "/help":
        _send(token, chat_id,
              "/scan  — Full scan + AI analysis\n"
              "/data  — Raw market data only\n"
              "/positions — Portfolio & short calls\n"
              "/roll  — Check roll candidates\n"
              "/help  — This message",
              None)
    else:
        _send(token, chat_id, "Unknown command. /help for list.", None)


def run():
    """Long-polling bot loop."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN in .env")
        return

    me = _call(token, "getMe")
    if not me.get("ok"):
        print(f"Invalid bot token: {me}")
        return
    username = me["result"]["username"]
    print(f"Alpha Trader bot running as @{username}  (Ctrl+C to stop)")

    offset = 0
    while True:
        try:
            data = _call(token, "getUpdates", offset=offset, timeout=30)
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message", {})
                txt = msg.get("text", "")
                cid = msg.get("chat", {}).get("id")
                if cid and txt.startswith("/"):
                    _handle(token, cid, txt)
        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    run()
