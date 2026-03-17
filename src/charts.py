"""Robinhood-style interactive stock charts with technical overlays and trade markers.

Generates standalone HTML files using Plotly.js (CDN, no pip install).
Works in any browser — Mac, iPhone Safari, dashboard.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib

import yfinance as yf
import pandas as pd

REPORTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _safe_round(v, decimals=2):
    """Round a value, returning None for NaN."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return round(v, decimals)


def generate_chart(
    symbol: str,
    period: str = "3mo",
    indicators: list[str] | None = None,
    trades: list[dict] | None = None,
    show_short_calls: list[dict] | None = None,
) -> str:
    """Generate Robinhood-style interactive chart with overlays.

    Args:
        symbol: Ticker symbol
        period: Default view period (1d, 1wk, 1mo, 3mo, 1y, all).
                Internally fetches 1Y of data; period sets the default filter.
        indicators: List of overlays: "sma", "bb", "rsi", "macd", "volume"
        trades: Trade journal entries to mark as sell/close points
        show_short_calls: Open short calls to show as horizontal strike lines

    Returns: Path to saved HTML file.
    """
    if indicators is None:
        indicators = ["sma", "bb", "rsi", "volume"]

    # Always fetch 1 year of data so all client-side period filters work
    t = yf.Ticker(symbol)
    hist = t.history(period="1y")
    if hist.empty:
        return ""

    # Also fetch 1d intraday data for the 1D view
    hist_1d = t.history(period="1d", interval="5m")

    # Get current price info
    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass

    current_price = hist["Close"].iloc[-1] if not hist.empty else 0
    prev_close = info.get("previousClose") or (
        hist["Close"].iloc[-2] if len(hist) > 1 else current_price
    )

    # Calculate all indicators on the full 1Y dataframe
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    # SMA
    hist["SMA20"] = close.rolling(20).mean()
    hist["SMA50"] = close.rolling(50).mean()

    # Bollinger Bands
    hist["BB_mid"] = close.rolling(20).mean()
    hist["BB_std"] = close.rolling(20).std()
    hist["BB_upper"] = hist["BB_mid"] + 2 * hist["BB_std"]
    hist["BB_lower"] = hist["BB_mid"] - 2 * hist["BB_std"]

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    hist["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    hist["MACD"] = ema12 - ema26
    hist["MACD_signal"] = hist["MACD"].ewm(span=9, adjust=False).mean()
    hist["MACD_hist"] = hist["MACD"] - hist["MACD_signal"]

    # Build daily data payload
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]
    data_payload = {
        "dates": dates,
        "open": [_safe_round(v) for v in hist["Open"]],
        "high": [_safe_round(v) for v in hist["High"]],
        "low": [_safe_round(v) for v in hist["Low"]],
        "close": [_safe_round(v) for v in hist["Close"]],
        "volume": [int(v) for v in hist["Volume"]],
        "sma20": [_safe_round(v) for v in hist["SMA20"]],
        "sma50": [_safe_round(v) for v in hist["SMA50"]],
        "bb_upper": [_safe_round(v) for v in hist["BB_upper"]],
        "bb_lower": [_safe_round(v) for v in hist["BB_lower"]],
        "rsi": [_safe_round(v, 1) for v in hist["RSI"]],
        "macd": [_safe_round(v, 3) for v in hist["MACD"]],
        "macd_signal": [_safe_round(v, 3) for v in hist["MACD_signal"]],
        "macd_hist": [_safe_round(v, 3) for v in hist["MACD_hist"]],
    }

    # Build intraday data payload for 1D view
    intraday_payload = {
        "dates": [], "open": [], "high": [], "low": [],
        "close": [], "volume": [],
    }
    if not hist_1d.empty:
        intraday_payload = {
            "dates": [d.strftime("%Y-%m-%dT%H:%M") for d in hist_1d.index],
            "open": [_safe_round(v) for v in hist_1d["Open"]],
            "high": [_safe_round(v) for v in hist_1d["High"]],
            "low": [_safe_round(v) for v in hist_1d["Low"]],
            "close": [_safe_round(v) for v in hist_1d["Close"]],
            "volume": [int(v) for v in hist_1d["Volume"]],
        }

    # Trade markers
    trade_markers = []
    if trades:
        for tr in trades:
            trade_markers.append({
                "date": tr.get("opened_at", tr.get("expiry", ""))[:10],
                "price": tr.get("strike", 0),
                "type": "sell",
                "label": (
                    f"SELL ${tr.get('strike')}C"
                    f" @${tr.get('premium_per_contract', '?')}"
                ),
            })
            if tr.get("closed_at"):
                trade_markers.append({
                    "date": tr["closed_at"][:10],
                    "price": tr.get("strike", 0),
                    "type": "close",
                    "label": (
                        f"CLOSE ${tr.get('strike')}C"
                        f" (${tr.get('close_price', '?')})"
                    ),
                })

    # Short call strike lines
    strike_lines = []
    if show_short_calls:
        for sc in show_short_calls:
            strike_lines.append({
                "strike": sc.get("strike", 0),
                "expiry": sc.get("expiry", ""),
                "label": f"${sc['strike']}C exp {sc.get('expiry', '?')[5:]}",
            })

    # Map period param to a JS default key
    period_map = {
        "1d": "1D", "1wk": "1W", "1mo": "1M", "3mo": "3M",
        "6mo": "3M", "1y": "1Y", "all": "ALL",
    }
    default_period = period_map.get(period, "3M")

    html = _build_html(
        symbol=symbol,
        current_price=round(current_price, 2),
        prev_close=round(prev_close, 2),
        data_payload=data_payload,
        intraday_payload=intraday_payload,
        trade_markers=trade_markers,
        strike_lines=strike_lines,
        default_period=default_period,
        default_indicators=indicators,
    )

    date_str = dt.datetime.now().strftime("%Y%m%d")
    path = REPORTS_DIR / f"chart_{symbol}_{date_str}.html"
    path.write_text(html)
    return str(path)


def _build_html(
    symbol: str,
    current_price: float,
    prev_close: float,
    data_payload: dict,
    intraday_payload: dict,
    trade_markers: list,
    strike_lines: list,
    default_period: str,
    default_indicators: list,
) -> str:
    """Build the complete standalone HTML string."""

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>{symbol} — Alpha Trader</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #000000;
    --bg-secondary: #1C1C1E;
    --green: #00C805;
    --red: #FF5000;
    --text-primary: #FFFFFF;
    --text-secondary: #9B9B9B;
    --text-tertiary: #6B6B6B;
    --border: #2C2C2E;
    --pill-bg: #1C1C1E;
    --pill-active: #2C2C2E;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
                 "Helvetica Neue", Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    overflow-x: hidden;
  }}

  /* Header */
  .rh-header {{
    padding: 16px 20px 0 20px;
    position: relative;
    z-index: 10;
  }}

  .rh-symbol {{
    font-size: 14px;
    font-weight: 700;
    color: var(--text-secondary);
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}

  .rh-price {{
    font-size: 38px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 4px;
    transition: all 0.15s ease;
  }}

  .rh-change {{
    font-size: 15px;
    font-weight: 500;
    margin-bottom: 14px;
    transition: all 0.15s ease;
  }}

  .rh-change-text {{
    transition: color 0.15s ease;
  }}

  .rh-hover-date {{
    font-size: 12px;
    color: var(--text-tertiary);
    height: 16px;
    margin-bottom: 4px;
    transition: opacity 0.15s ease;
  }}

  /* Period selector */
  .rh-periods {{
    display: flex;
    gap: 0;
    padding: 0 20px;
    margin-bottom: 2px;
    border-bottom: 1px solid var(--border);
  }}

  .rh-period-btn {{
    background: none;
    border: none;
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 13px;
    font-weight: 600;
    padding: 10px 14px;
    cursor: pointer;
    position: relative;
    transition: color 0.2s ease;
    -webkit-tap-highlight-color: transparent;
  }}

  .rh-period-btn:hover {{
    color: var(--text-primary);
  }}

  .rh-period-btn.active {{
    color: var(--green);
  }}

  .rh-period-btn.active::after {{
    content: '';
    position: absolute;
    bottom: -1px;
    left: 50%;
    transform: translateX(-50%);
    width: 100%;
    height: 2px;
    background: var(--green);
    border-radius: 1px;
  }}

  .rh-period-btn.active.negative {{
    color: var(--red);
  }}

  .rh-period-btn.active.negative::after {{
    background: var(--red);
  }}

  /* Chart container */
  .rh-chart-wrap {{
    width: 100%;
    padding: 0;
  }}

  #main-chart {{
    width: 100%;
    height: 320px;
  }}

  #rsi-chart {{
    width: 100%;
    height: 80px;
    display: none;
  }}

  #macd-chart {{
    width: 100%;
    height: 80px;
    display: none;
  }}

  #volume-chart {{
    width: 100%;
    height: 60px;
    display: none;
  }}

  /* Indicator toggles */
  .rh-controls {{
    display: flex;
    gap: 8px;
    padding: 8px 20px 12px 20px;
    flex-wrap: wrap;
    align-items: center;
  }}

  .rh-toggle {{
    background: var(--pill-bg);
    border: 1px solid var(--border);
    border-radius: 20px;
    color: var(--text-secondary);
    font-family: inherit;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 14px;
    cursor: pointer;
    transition: all 0.2s ease;
    -webkit-tap-highlight-color: transparent;
    user-select: none;
  }}

  .rh-toggle:hover {{
    border-color: var(--text-secondary);
  }}

  .rh-toggle.active {{
    background: var(--pill-active);
    border-color: var(--text-primary);
    color: var(--text-primary);
  }}

  .rh-toggle.chart-mode {{
    background: var(--pill-bg);
    border-color: var(--green);
    color: var(--green);
    margin-left: auto;
  }}

  .rh-toggle.chart-mode:hover {{
    background: rgba(0, 200, 5, 0.1);
  }}

  /* Responsive */
  @media (max-width: 480px) {{
    .rh-price {{ font-size: 32px; }}
    .rh-period-btn {{ padding: 10px 10px; font-size: 12px; }}
    .rh-toggle {{ font-size: 11px; padding: 5px 10px; }}
    #main-chart {{ height: 280px; }}
  }}

  /* Plotly overrides */
  .js-plotly-plot .plotly .modebar {{ display: none !important; }}
</style>
</head><body>

<!-- Header -->
<div class="rh-header">
  <div class="rh-symbol">{symbol}</div>
  <div class="rh-hover-date" id="hover-date"></div>
  <div class="rh-price" id="price-display">${current_price:,.2f}</div>
  <div class="rh-change" id="change-display"><span class="rh-change-text" id="change-text"></span></div>
</div>

<!-- Period selector -->
<div class="rh-periods" id="period-bar">
  <button class="rh-period-btn" data-period="1D">1D</button>
  <button class="rh-period-btn" data-period="1W">1W</button>
  <button class="rh-period-btn" data-period="1M">1M</button>
  <button class="rh-period-btn" data-period="3M">3M</button>
  <button class="rh-period-btn" data-period="1Y">1Y</button>
  <button class="rh-period-btn" data-period="ALL">ALL</button>
</div>

<!-- Charts -->
<div class="rh-chart-wrap">
  <div id="main-chart"></div>
  <div id="rsi-chart"></div>
  <div id="macd-chart"></div>
  <div id="volume-chart"></div>
</div>

<!-- Controls -->
<div class="rh-controls" id="indicator-bar">
  <button class="rh-toggle" data-ind="sma">SMA</button>
  <button class="rh-toggle" data-ind="bb">BB</button>
  <button class="rh-toggle" data-ind="rsi">RSI</button>
  <button class="rh-toggle" data-ind="macd">MACD</button>
  <button class="rh-toggle" data-ind="volume">VOL</button>
  <button class="rh-toggle chart-mode" id="chart-mode-btn">Candles</button>
</div>

<script>
// ===========================================================
// DATA
// ===========================================================
var SYMBOL = {json.dumps(symbol)};
var CURRENT_PRICE = {current_price};
var PREV_CLOSE = {prev_close};
var D = {json.dumps(data_payload)};
var INTRADAY = {json.dumps(intraday_payload)};
var TRADES = {json.dumps(trade_markers)};
var STRIKES = {json.dumps(strike_lines)};
var GREEN = '#00C805';
var RED = '#FF5000';
var GREEN_FILL = 'rgba(0, 200, 5, 0.08)';
var RED_FILL = 'rgba(255, 80, 0, 0.08)';

// ===========================================================
// STATE
// ===========================================================
var activePeriod = {json.dumps(default_period)};
var chartMode = 'line'; // 'line' or 'candle'
var activeIndicators = new Set({json.dumps(default_indicators)});
var isHovering = false;

// ===========================================================
// PERIOD FILTERING
// ===========================================================
function filterByPeriod(period) {{
  if (period === '1D') {{
    return {{
      dates: INTRADAY.dates,
      open: INTRADAY.open,
      high: INTRADAY.high,
      low: INTRADAY.low,
      close: INTRADAY.close,
      volume: INTRADAY.volume,
      sma20: new Array(INTRADAY.dates.length).fill(null),
      sma50: new Array(INTRADAY.dates.length).fill(null),
      bb_upper: new Array(INTRADAY.dates.length).fill(null),
      bb_lower: new Array(INTRADAY.dates.length).fill(null),
      rsi: new Array(INTRADAY.dates.length).fill(null),
      macd: new Array(INTRADAY.dates.length).fill(null),
      macd_signal: new Array(INTRADAY.dates.length).fill(null),
      macd_hist: new Array(INTRADAY.dates.length).fill(null),
    }};
  }}

  var now = new Date();
  var cutoff = new Date();
  switch (period) {{
    case '1W': cutoff.setDate(now.getDate() - 7); break;
    case '1M': cutoff.setMonth(now.getMonth() - 1); break;
    case '3M': cutoff.setMonth(now.getMonth() - 3); break;
    case '1Y': cutoff.setFullYear(now.getFullYear() - 1); break;
    case 'ALL': cutoff = new Date('1900-01-01'); break;
    default: cutoff.setMonth(now.getMonth() - 3);
  }}

  var cutoffStr = cutoff.toISOString().slice(0, 10);
  var startIdx = D.dates.findIndex(function(d) {{ return d >= cutoffStr; }});
  var idx = startIdx === -1 ? 0 : startIdx;

  var filtered = {{}};
  var keys = Object.keys(D);
  for (var k = 0; k < keys.length; k++) {{
    filtered[keys[k]] = D[keys[k]].slice(idx);
  }}
  return filtered;
}}

function getPeriodStartPrice(data) {{
  for (var i = 0; i < data.close.length; i++) {{
    if (data.close[i] !== null) return data.close[i];
  }}
  return CURRENT_PRICE;
}}

// ===========================================================
// COLOR HELPERS
// ===========================================================
function isUp(data) {{
  var start = getPeriodStartPrice(data);
  var end = data.close[data.close.length - 1] || CURRENT_PRICE;
  return end >= start;
}}

function accentColor(data) {{
  return isUp(data) ? GREEN : RED;
}}

// ===========================================================
// HEADER UPDATE (safe DOM — no innerHTML)
// ===========================================================
function updateHeader(price, refPrice, dateStr) {{
  var priceEl = document.getElementById('price-display');
  var changeTextEl = document.getElementById('change-text');
  var hoverDateEl = document.getElementById('hover-date');

  priceEl.textContent = '$' + price.toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});

  var diff = price - refPrice;
  var pct = refPrice !== 0 ? ((diff / refPrice) * 100) : 0;
  var sign = diff >= 0 ? '+' : '';
  var color = diff >= 0 ? GREEN : RED;

  changeTextEl.textContent = sign + diff.toFixed(2) + ' (' + sign + pct.toFixed(2) + '%)';
  changeTextEl.style.color = color;
  priceEl.style.color = 'var(--text-primary)';

  if (dateStr) {{
    hoverDateEl.textContent = dateStr;
    hoverDateEl.style.opacity = '1';
  }} else {{
    hoverDateEl.textContent = '';
    hoverDateEl.style.opacity = '0';
  }}
}}

function resetHeader() {{
  var data = filterByPeriod(activePeriod);
  var refPrice = getPeriodStartPrice(data);
  updateHeader(CURRENT_PRICE, refPrice, null);
}}

// ===========================================================
// RENDER: MAIN CHART
// ===========================================================
function renderMainChart() {{
  var data = filterByPeriod(activePeriod);
  var color = accentColor(data);
  var fillColor = isUp(data) ? GREEN_FILL : RED_FILL;
  var traces = [];

  if (chartMode === 'line') {{
    // Main price line
    traces.push({{
      x: data.dates,
      y: data.close,
      type: 'scatter',
      mode: 'lines',
      name: 'Price',
      line: {{ color: color, width: 1.8, shape: 'spline', smoothing: 0.3 }},
      fill: 'tozeroy',
      fillcolor: fillColor,
      fillgradient: {{
        type: 'vertical',
        colorscale: [[0, 'rgba(0,0,0,0)'], [1, color === GREEN ? 'rgba(0,200,5,0.12)' : 'rgba(255,80,0,0.12)']],
      }},
      hoverinfo: 'x+y',
      hoverlabel: {{ bgcolor: '#1C1C1E', bordercolor: color, font: {{ color: '#fff', family: '-apple-system, sans-serif', size: 13 }} }},
    }});
  }} else {{
    // Candlestick
    traces.push({{
      x: data.dates,
      open: data.open,
      high: data.high,
      low: data.low,
      close: data.close,
      type: 'candlestick',
      name: SYMBOL,
      increasing: {{ line: {{ color: GREEN, width: 1 }}, fillcolor: GREEN }},
      decreasing: {{ line: {{ color: RED, width: 1 }}, fillcolor: RED }},
      hoverinfo: 'x+text',
    }});
  }}

  // SMA overlays
  if (activeIndicators.has('sma')) {{
    traces.push({{
      x: data.dates, y: data.sma20, type: 'scatter', mode: 'lines',
      name: 'SMA 20', line: {{ color: '#FFAA00', width: 1, shape: 'spline' }},
      hoverinfo: 'skip',
    }});
    traces.push({{
      x: data.dates, y: data.sma50, type: 'scatter', mode: 'lines',
      name: 'SMA 50', line: {{ color: '#5AC8FA', width: 1, shape: 'spline' }},
      hoverinfo: 'skip',
    }});
  }}

  // Bollinger Bands
  if (activeIndicators.has('bb')) {{
    traces.push({{
      x: data.dates, y: data.bb_upper, type: 'scatter', mode: 'lines',
      name: 'BB Upper', line: {{ color: 'rgba(155,155,155,0.4)', width: 0.8 }},
      hoverinfo: 'skip',
    }});
    traces.push({{
      x: data.dates, y: data.bb_lower, type: 'scatter', mode: 'lines',
      name: 'BB Lower', line: {{ color: 'rgba(155,155,155,0.4)', width: 0.8 }},
      fill: 'tonexty', fillcolor: 'rgba(155,155,155,0.04)',
      hoverinfo: 'skip',
    }});
  }}

  // Strike lines for open short calls
  if (STRIKES.length > 0) {{
    for (var si = 0; si < STRIKES.length; si++) {{
      var s = STRIKES[si];
      traces.push({{
        x: [data.dates[0], data.dates[data.dates.length - 1]],
        y: [s.strike, s.strike],
        type: 'scatter', mode: 'lines', name: s.label,
        line: {{ color: 'rgba(255,80,0,0.5)', width: 1, dash: 'dash' }},
        hoverinfo: 'skip',
      }});
      // Label on right edge
      traces.push({{
        x: [data.dates[data.dates.length - 1]],
        y: [s.strike],
        type: 'scatter', mode: 'text',
        text: [s.label],
        textposition: 'middle left',
        textfont: {{ size: 10, color: 'rgba(255,80,0,0.7)', family: '-apple-system, sans-serif' }},
        showlegend: false, hoverinfo: 'skip',
      }});
    }}
  }}

  // Trade markers
  var sells = TRADES.filter(function(t) {{ return t.type === 'sell'; }});
  var closes = TRADES.filter(function(t) {{ return t.type === 'close'; }});

  if (sells.length) {{
    var visibleSells = sells.filter(function(sv) {{ return data.dates.indexOf(sv.date) !== -1; }});
    if (visibleSells.length) {{
      traces.push({{
        x: visibleSells.map(function(t) {{ return t.date; }}),
        y: visibleSells.map(function(t) {{ return t.price; }}),
        type: 'scatter', mode: 'markers+text', name: 'Sell',
        marker: {{ color: RED, size: 7, symbol: 'circle' }},
        text: visibleSells.map(function(t) {{ return t.label; }}),
        textposition: 'top center',
        textfont: {{ size: 9, color: RED, family: '-apple-system, sans-serif' }},
        hoverinfo: 'text',
      }});
    }}
  }}
  if (closes.length) {{
    var visibleCloses = closes.filter(function(cv) {{ return data.dates.indexOf(cv.date) !== -1; }});
    if (visibleCloses.length) {{
      traces.push({{
        x: visibleCloses.map(function(t) {{ return t.date; }}),
        y: visibleCloses.map(function(t) {{ return t.price; }}),
        type: 'scatter', mode: 'markers+text', name: 'Close',
        marker: {{ color: GREEN, size: 7, symbol: 'circle' }},
        text: visibleCloses.map(function(t) {{ return t.label; }}),
        textposition: 'bottom center',
        textfont: {{ size: 9, color: GREEN, family: '-apple-system, sans-serif' }},
        hoverinfo: 'text',
      }});
    }}
  }}

  // Compute Y range with padding
  var validCloses = data.close.filter(function(v) {{ return v !== null; }});
  var yMin = Math.min.apply(null, validCloses);
  var yMax = Math.max.apply(null, validCloses);
  if (chartMode === 'candle') {{
    var validHighs = data.high.filter(function(v) {{ return v !== null; }});
    var validLows = data.low.filter(function(v) {{ return v !== null; }});
    if (validLows.length) yMin = Math.min(yMin, Math.min.apply(null, validLows));
    if (validHighs.length) yMax = Math.max(yMax, Math.max.apply(null, validHighs));
  }}
  for (var sti = 0; sti < STRIKES.length; sti++) {{
    yMin = Math.min(yMin, STRIKES[sti].strike);
    yMax = Math.max(yMax, STRIKES[sti].strike);
  }}
  var pad = (yMax - yMin) * 0.08;

  var layout = {{
    paper_bgcolor: '#000000',
    plot_bgcolor: '#000000',
    font: {{ color: '#fff', family: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif' }},
    margin: {{ l: 0, r: 0, t: 8, b: 28 }},
    showlegend: false,
    xaxis: {{
      rangeslider: {{ visible: false }},
      showgrid: false,
      zeroline: false,
      showticklabels: true,
      tickfont: {{ size: 10, color: '#6B6B6B' }},
      tickformat: activePeriod === '1D' ? '%H:%M' : (activePeriod === '1W' ? '%a' : '%b %d'),
      nticks: 5,
      fixedrange: true,
      linecolor: 'rgba(0,0,0,0)',
      type: activePeriod === '1D' ? 'date' : 'category',
    }},
    yaxis: {{
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      fixedrange: true,
      range: [yMin - pad, yMax + pad],
    }},
    hovermode: 'x unified',
    hoverlabel: {{
      bgcolor: 'rgba(0,0,0,0)',
      bordercolor: 'rgba(0,0,0,0)',
      font: {{ color: 'rgba(0,0,0,0)', size: 1 }},
    }},
    dragmode: false,
  }};

  var config = {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: false,
    staticPlot: false,
  }};

  Plotly.react('main-chart', traces, layout, config);

  // Hover events — update header like Robinhood
  var mainEl = document.getElementById('main-chart');
  mainEl.on('plotly_hover', function(eventData) {{
    if (eventData.points && eventData.points.length > 0) {{
      isHovering = true;
      var pt = eventData.points[0];
      var price = pt.y || pt.close || CURRENT_PRICE;
      var dateStr = pt.x || '';
      var refPrice = getPeriodStartPrice(data);

      // Format date for display
      var displayDate = dateStr;
      try {{
        var d = new Date(dateStr);
        if (activePeriod === '1D') {{
          displayDate = d.toLocaleTimeString('en-US', {{ hour: 'numeric', minute: '2-digit' }});
        }} else {{
          displayDate = d.toLocaleDateString('en-US', {{ month: 'short', day: 'numeric', year: 'numeric' }});
        }}
      }} catch(e) {{}}

      updateHeader(price, refPrice, displayDate);
    }}
  }});

  mainEl.on('plotly_unhover', function() {{
    isHovering = false;
    resetHeader();
  }});
}}

// ===========================================================
// RENDER: RSI SUB-CHART
// ===========================================================
function renderRSI() {{
  var el = document.getElementById('rsi-chart');
  if (!activeIndicators.has('rsi')) {{
    el.style.display = 'none';
    Plotly.purge('rsi-chart');
    return;
  }}

  el.style.display = 'block';
  var data = filterByPeriod(activePeriod);

  var traces = [
    // 70-zone fill
    {{
      x: data.dates, y: data.dates.map(function() {{ return 70; }}),
      type: 'scatter', mode: 'lines', showlegend: false,
      line: {{ color: 'rgba(0,0,0,0)', width: 0 }}, hoverinfo: 'skip',
    }},
    {{
      x: data.dates, y: data.dates.map(function() {{ return 100; }}),
      type: 'scatter', mode: 'lines', showlegend: false,
      fill: 'tonexty', fillcolor: 'rgba(255,80,0,0.06)',
      line: {{ color: 'rgba(0,0,0,0)', width: 0 }}, hoverinfo: 'skip',
    }},
    // 30-zone fill
    {{
      x: data.dates, y: data.dates.map(function() {{ return 0; }}),
      type: 'scatter', mode: 'lines', showlegend: false,
      line: {{ color: 'rgba(0,0,0,0)', width: 0 }}, hoverinfo: 'skip',
    }},
    {{
      x: data.dates, y: data.dates.map(function() {{ return 30; }}),
      type: 'scatter', mode: 'lines', showlegend: false,
      fill: 'tonexty', fillcolor: 'rgba(0,200,5,0.06)',
      line: {{ color: 'rgba(0,0,0,0)', width: 0 }}, hoverinfo: 'skip',
    }},
    // 70 line
    {{
      x: [data.dates[0], data.dates[data.dates.length-1]], y: [70, 70],
      type: 'scatter', mode: 'lines', showlegend: false,
      line: {{ color: 'rgba(155,155,155,0.2)', width: 0.5, dash: 'dot' }}, hoverinfo: 'skip',
    }},
    // 30 line
    {{
      x: [data.dates[0], data.dates[data.dates.length-1]], y: [30, 30],
      type: 'scatter', mode: 'lines', showlegend: false,
      line: {{ color: 'rgba(155,155,155,0.2)', width: 0.5, dash: 'dot' }}, hoverinfo: 'skip',
    }},
    // RSI line
    {{
      x: data.dates, y: data.rsi,
      type: 'scatter', mode: 'lines', name: 'RSI',
      line: {{ color: '#BF5AF2', width: 1.5, shape: 'spline' }},
      hoverinfo: 'y',
      hoverlabel: {{ bgcolor: '#1C1C1E', font: {{ color: '#BF5AF2', size: 11 }} }},
    }},
  ];

  var layout = {{
    paper_bgcolor: '#000000', plot_bgcolor: '#000000',
    font: {{ color: '#6B6B6B', family: '-apple-system, sans-serif', size: 9 }},
    margin: {{ l: 0, r: 38, t: 0, b: 0 }},
    showlegend: false,
    xaxis: {{
      showgrid: false, zeroline: false, showticklabels: false,
      fixedrange: true, type: activePeriod === '1D' ? 'date' : 'category',
    }},
    yaxis: {{
      showgrid: false, zeroline: false,
      showticklabels: true, tickvals: [30, 70], tickfont: {{ size: 9, color: '#6B6B6B' }},
      fixedrange: true, range: [0, 100], side: 'right',
    }},
    hovermode: 'x', dragmode: false,
    annotations: [{{
      x: 0.01, y: 0.92, xref: 'paper', yref: 'paper',
      text: 'RSI', showarrow: false,
      font: {{ size: 10, color: '#6B6B6B', family: '-apple-system, sans-serif' }},
    }}],
  }};

  Plotly.react('rsi-chart', traces, layout, {{ responsive: true, displayModeBar: false, scrollZoom: false, staticPlot: false }});
}}

// ===========================================================
// RENDER: MACD SUB-CHART
// ===========================================================
function renderMACD() {{
  var el = document.getElementById('macd-chart');
  if (!activeIndicators.has('macd')) {{
    el.style.display = 'none';
    Plotly.purge('macd-chart');
    return;
  }}

  el.style.display = 'block';
  var data = filterByPeriod(activePeriod);

  var histColors = data.macd_hist.map(function(v) {{
    if (v === null) return 'rgba(0,0,0,0)';
    return v >= 0 ? 'rgba(0,200,5,0.5)' : 'rgba(255,80,0,0.5)';
  }});

  var traces = [
    // Histogram
    {{
      x: data.dates, y: data.macd_hist, type: 'bar', name: 'Hist',
      marker: {{ color: histColors }}, hoverinfo: 'skip',
    }},
    // MACD line
    {{
      x: data.dates, y: data.macd, type: 'scatter', mode: 'lines', name: 'MACD',
      line: {{ color: '#5AC8FA', width: 1.2, shape: 'spline' }}, hoverinfo: 'y',
      hoverlabel: {{ bgcolor: '#1C1C1E', font: {{ color: '#5AC8FA', size: 11 }} }},
    }},
    // Signal line
    {{
      x: data.dates, y: data.macd_signal, type: 'scatter', mode: 'lines', name: 'Signal',
      line: {{ color: '#FF9500', width: 1, shape: 'spline' }}, hoverinfo: 'y',
      hoverlabel: {{ bgcolor: '#1C1C1E', font: {{ color: '#FF9500', size: 11 }} }},
    }},
    // Zero line
    {{
      x: [data.dates[0], data.dates[data.dates.length-1]], y: [0, 0],
      type: 'scatter', mode: 'lines', showlegend: false,
      line: {{ color: 'rgba(155,155,155,0.15)', width: 0.5 }}, hoverinfo: 'skip',
    }},
  ];

  var layout = {{
    paper_bgcolor: '#000000', plot_bgcolor: '#000000',
    font: {{ color: '#6B6B6B', family: '-apple-system, sans-serif', size: 9 }},
    margin: {{ l: 0, r: 38, t: 0, b: 0 }},
    showlegend: false,
    xaxis: {{
      showgrid: false, zeroline: false, showticklabels: false,
      fixedrange: true, type: activePeriod === '1D' ? 'date' : 'category',
    }},
    yaxis: {{
      showgrid: false, zeroline: false,
      showticklabels: true, tickfont: {{ size: 9, color: '#6B6B6B' }},
      fixedrange: true, side: 'right', nticks: 3,
    }},
    hovermode: 'x', dragmode: false, bargap: 0.3,
    annotations: [{{
      x: 0.01, y: 0.92, xref: 'paper', yref: 'paper',
      text: 'MACD', showarrow: false,
      font: {{ size: 10, color: '#6B6B6B', family: '-apple-system, sans-serif' }},
    }}],
  }};

  Plotly.react('macd-chart', traces, layout, {{ responsive: true, displayModeBar: false, scrollZoom: false, staticPlot: false }});
}}

// ===========================================================
// RENDER: VOLUME SUB-CHART
// ===========================================================
function renderVolume() {{
  var el = document.getElementById('volume-chart');
  if (!activeIndicators.has('volume')) {{
    el.style.display = 'none';
    Plotly.purge('volume-chart');
    return;
  }}

  el.style.display = 'block';
  var data = filterByPeriod(activePeriod);

  var barColors = data.close.map(function(c, i) {{
    if (i === 0) return 'rgba(155,155,155,0.25)';
    return c >= data.close[i - 1] ? 'rgba(0,200,5,0.25)' : 'rgba(255,80,0,0.25)';
  }});

  var traces = [{{
    x: data.dates, y: data.volume, type: 'bar', name: 'Volume',
    marker: {{ color: barColors }}, hoverinfo: 'y',
    hoverlabel: {{ bgcolor: '#1C1C1E', font: {{ color: '#9B9B9B', size: 11 }} }},
  }}];

  var layout = {{
    paper_bgcolor: '#000000', plot_bgcolor: '#000000',
    font: {{ color: '#6B6B6B', family: '-apple-system, sans-serif', size: 9 }},
    margin: {{ l: 0, r: 38, t: 0, b: 0 }},
    showlegend: false,
    xaxis: {{
      showgrid: false, zeroline: false, showticklabels: false,
      fixedrange: true, type: activePeriod === '1D' ? 'date' : 'category',
    }},
    yaxis: {{
      showgrid: false, zeroline: false,
      showticklabels: true, tickfont: {{ size: 9, color: '#6B6B6B' }},
      fixedrange: true, side: 'right', nticks: 2,
    }},
    hovermode: 'x', dragmode: false, bargap: 0.2,
    annotations: [{{
      x: 0.01, y: 0.88, xref: 'paper', yref: 'paper',
      text: 'VOL', showarrow: false,
      font: {{ size: 10, color: '#6B6B6B', family: '-apple-system, sans-serif' }},
    }}],
  }};

  Plotly.react('volume-chart', traces, layout, {{ responsive: true, displayModeBar: false, scrollZoom: false, staticPlot: false }});
}}

// ===========================================================
// RENDER ALL
// ===========================================================
function renderAll() {{
  renderMainChart();
  renderRSI();
  renderMACD();
  renderVolume();
  resetHeader();
  updatePeriodButtons();
  updateIndicatorButtons();

  // Adjust main chart height based on active sub-charts
  var mainEl = document.getElementById('main-chart');
  var subHeight = 0;
  if (activeIndicators.has('rsi')) subHeight += 80;
  if (activeIndicators.has('macd')) subHeight += 80;
  if (activeIndicators.has('volume')) subHeight += 60;
  var mainHeight = Math.max(200, 340 - subHeight * 0.3);
  mainEl.style.height = mainHeight + 'px';
  Plotly.Plots.resize(mainEl);
}}

// ===========================================================
// UI: Period buttons
// ===========================================================
function updatePeriodButtons() {{
  var data = filterByPeriod(activePeriod);
  var negative = !isUp(data);

  var btns = document.querySelectorAll('.rh-period-btn');
  for (var i = 0; i < btns.length; i++) {{
    btns[i].classList.remove('active', 'negative');
    if (btns[i].getAttribute('data-period') === activePeriod) {{
      btns[i].classList.add('active');
      if (negative) btns[i].classList.add('negative');
    }}
  }}
}}

var periodBtns = document.querySelectorAll('.rh-period-btn');
for (var pi = 0; pi < periodBtns.length; pi++) {{
  periodBtns[pi].addEventListener('click', function() {{
    activePeriod = this.getAttribute('data-period');
    renderAll();
  }});
}}

// ===========================================================
// UI: Indicator toggles
// ===========================================================
function updateIndicatorButtons() {{
  var indBtns = document.querySelectorAll('.rh-toggle[data-ind]');
  for (var i = 0; i < indBtns.length; i++) {{
    var ind = indBtns[i].getAttribute('data-ind');
    if (activeIndicators.has(ind)) {{
      indBtns[i].classList.add('active');
    }} else {{
      indBtns[i].classList.remove('active');
    }}
  }}
  document.getElementById('chart-mode-btn').textContent = chartMode === 'line' ? 'Candles' : 'Line';
}}

var indToggles = document.querySelectorAll('.rh-toggle[data-ind]');
for (var ii = 0; ii < indToggles.length; ii++) {{
  indToggles[ii].addEventListener('click', function() {{
    var ind = this.getAttribute('data-ind');
    if (activeIndicators.has(ind)) {{
      activeIndicators.delete(ind);
    }} else {{
      activeIndicators.add(ind);
    }}
    renderAll();
  }});
}}

document.getElementById('chart-mode-btn').addEventListener('click', function() {{
  chartMode = chartMode === 'line' ? 'candle' : 'line';
  renderAll();
}});

// ===========================================================
// INIT
// ===========================================================
renderAll();
</script>
</body></html>"""


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    path = generate_chart(sym)
    if path:
        print(f"Chart saved: {path}")
        import webbrowser
        webbrowser.open(f"file://{path}")
