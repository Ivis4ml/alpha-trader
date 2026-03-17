"""Unicode sparkline charts for inline display in Claude Code Remote.

Renders directly in the iOS app conversation — no browser needed.
"""

from __future__ import annotations

import math
import yfinance as yf

# Unicode block elements for sparklines: ▁▂▃▄▅▆▇█
BLOCKS = " ▁▂▃▄▅▆▇█"


def _spark(values: list[float], width: int = 40) -> str:
    """Convert a list of values into a sparkline string."""
    if not values:
        return ""
    # Resample to target width
    if len(values) > width:
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]

    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1
    return "".join(BLOCKS[min(8, max(0, int((v - mn) / rng * 8)))] for v in values)


def _rsi_bar(rsi: float) -> str:
    """Visual RSI bar."""
    filled = int(rsi / 100 * 15)
    bar = "█" * filled + "░" * (15 - filled)
    label = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else "neutral")
    return f"{bar} {rsi:.0f} ({label})"


def _pct_bar(value: float, mn: float, mx: float, width: int = 12) -> str:
    """Position bar within a range."""
    if mx == mn:
        return "█" * (width // 2)
    pos = (value - mn) / (mx - mn)
    filled = int(pos * width)
    return "░" * filled + "█" + "░" * (width - filled - 1)


def generate_sparkline_report(
    symbol: str,
    period: str = "3mo",
    show_short_calls: list[dict] | None = None,
) -> str:
    """Generate a text-based chart that renders inline in Claude Code.

    Returns a multi-line string with sparkline price chart, RSI bar,
    MACD direction, volume profile, and 52w range position.
    """
    t = yf.Ticker(symbol)
    hist = t.history(period=period)
    if hist.empty:
        return f"{symbol}: no data"

    close = hist["Close"].tolist()
    volume = hist["Volume"].tolist()
    current = close[-1]
    prev = close[-2] if len(close) > 1 else current
    change = current - prev
    change_pct = (change / prev * 100) if prev else 0

    # Price sparkline
    price_spark = _spark(close)

    # Range labels
    period_low = min(close)
    period_high = max(close)

    # RSI
    delta = hist["Close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    rsi_series = 100 - (100 / (1 + rs))
    rsi = rsi_series.iloc[-1] if not math.isnan(rsi_series.iloc[-1]) else 50

    # MACD
    ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
    ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal
    macd_val = macd_hist.iloc[-1]
    macd_trend = "▲ bullish" if macd_val > 0 else "▼ bearish"
    macd_spark = _spark(macd_hist.dropna().tolist()[-20:], width=20)

    # Volume sparkline
    vol_spark = _spark(volume[-20:], width=20)
    avg_vol = sum(volume[-20:]) / len(volume[-20:])

    # SMA
    sma20 = hist["Close"].tail(20).mean()
    sma50 = hist["Close"].tail(50).mean() if len(hist) >= 50 else None
    sma_status = "↑20SMA" if current > sma20 else "↓20SMA"
    if sma50:
        sma_status += f" {'↑' if current > sma50 else '↓'}50SMA"

    # 52w range position
    info = t.fast_info
    high_52w = info.get("yearHigh", period_high)
    low_52w = info.get("yearLow", period_low)
    range_bar = _pct_bar(current, low_52w, high_52w)

    # Short call strikes
    strike_lines = ""
    if show_short_calls:
        for sc in show_short_calls:
            strike = sc.get("strike", 0)
            dist = (strike - current) / current * 100
            strike_lines += f"\n  ⚡ Short ${strike}C ({dist:+.1f}% away) exp {sc.get('expiry', '?')[5:]}"

    sign = "+" if change >= 0 else ""
    color_arrow = "🟢" if change >= 0 else "🔴"

    return f"""{color_arrow} **{symbol}** ${current:.2f} ({sign}{change:.2f}, {sign}{change_pct:.1f}%)

```
Price  {price_spark}
       ${period_low:.0f}{' ' * 30}${period_high:.0f}
```
RSI 14 {_rsi_bar(rsi)}
MACD   {macd_spark} {macd_trend}
Volume {vol_spark} avg {avg_vol/1e6:.1f}M
52w    {range_bar} ${low_52w:.0f}—${high_52w:.0f}
Trend  {sma_status}{strike_lines}"""
