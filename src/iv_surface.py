"""IV Surface visualization — 3D implied volatility surface and 2D smile plots.

Generates standalone HTML files using Plotly.js (loaded from CDN, no pip install).
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib

import yfinance as yf

from .data.greeks import implied_volatility
from .data.fetcher import RISK_FREE_RATE, _safe_float, _safe_int

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


def generate_iv_surface(symbol: str) -> list[dict]:
    """Fetch option chains for ALL available expiries and extract IV for each (strike, expiry) pair.

    Returns a list of dicts with keys: strike, dte, iv, expiry, option_type.
    """
    t = yf.Ticker(symbol)
    try:
        expiries = t.options
    except Exception:
        return []

    # Get current stock price
    fi = t.fast_info
    stock_price = fi.get("lastPrice") or fi.get("previousClose") or 0
    if stock_price <= 0:
        return []

    today = dt.date.today()
    surface_points = []

    for exp_str in expiries:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte <= 0:
            continue

        T = dte / 365.0

        try:
            chain = t.option_chain(exp_str)
        except Exception:
            continue

        # Process both calls and puts; prefer calls for OTM above spot, puts for OTM below
        for _, row in chain.calls.iterrows():
            strike = _safe_float(row.get("strike"))
            if strike <= 0:
                continue
            iv = _safe_float(row.get("impliedVolatility"))
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last = _safe_float(row.get("lastPrice"))
            mid = (bid + ask) / 2 if (bid + ask) > 0 else last

            # Skip strikes with no meaningful price data
            if mid <= 0 and last <= 0:
                continue

            # If IV looks unreliable, solve from market price
            if iv < 0.01 and mid > 0:
                iv = implied_volatility(mid, stock_price, strike, T, RISK_FREE_RATE, "call")
            elif iv < 0.01 and last > 0:
                iv = implied_volatility(last, stock_price, strike, T, RISK_FREE_RATE, "call")

            if iv <= 0.001:
                continue

            surface_points.append({
                "strike": round(strike, 2),
                "dte": dte,
                "iv": round(iv * 100, 2),  # as percentage
                "expiry": exp_str,
                "option_type": "call",
            })

        for _, row in chain.puts.iterrows():
            strike = _safe_float(row.get("strike"))
            if strike <= 0:
                continue
            iv = _safe_float(row.get("impliedVolatility"))
            bid = _safe_float(row.get("bid"))
            ask = _safe_float(row.get("ask"))
            last = _safe_float(row.get("lastPrice"))
            mid = (bid + ask) / 2 if (bid + ask) > 0 else last

            if mid <= 0 and last <= 0:
                continue

            if iv < 0.01 and mid > 0:
                iv = implied_volatility(mid, stock_price, strike, T, RISK_FREE_RATE, "put")
            elif iv < 0.01 and last > 0:
                iv = implied_volatility(last, stock_price, strike, T, RISK_FREE_RATE, "put")

            if iv <= 0.001:
                continue

            surface_points.append({
                "strike": round(strike, 2),
                "dte": dte,
                "iv": round(iv * 100, 2),
                "expiry": exp_str,
                "option_type": "put",
            })

    return surface_points


def _get_smile_data(symbol: str) -> tuple[list[dict], str, float]:
    """Get IV smile data for the nearest expiry.

    Returns (points, expiry_str, stock_price).
    """
    t = yf.Ticker(symbol)
    fi = t.fast_info
    stock_price = fi.get("lastPrice") or fi.get("previousClose") or 0

    try:
        expiries = t.options
    except Exception:
        return [], "", stock_price

    today = dt.date.today()

    # Find nearest expiry with at least 1 DTE
    nearest_exp = None
    for exp_str in expiries:
        exp_date = dt.datetime.strptime(exp_str, "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte > 0:
            nearest_exp = exp_str
            break

    if not nearest_exp:
        return [], "", stock_price

    dte = (dt.datetime.strptime(nearest_exp, "%Y-%m-%d").date() - today).days
    T = dte / 365.0

    try:
        chain = t.option_chain(nearest_exp)
    except Exception:
        return [], nearest_exp, stock_price

    points = []

    for _, row in chain.calls.iterrows():
        strike = _safe_float(row.get("strike"))
        if strike <= 0:
            continue
        iv = _safe_float(row.get("impliedVolatility"))
        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        last = _safe_float(row.get("lastPrice"))
        mid = (bid + ask) / 2 if (bid + ask) > 0 else last

        if mid <= 0 and last <= 0:
            continue

        if iv < 0.01 and mid > 0:
            iv = implied_volatility(mid, stock_price, strike, T, RISK_FREE_RATE, "call")
        elif iv < 0.01 and last > 0:
            iv = implied_volatility(last, stock_price, strike, T, RISK_FREE_RATE, "call")

        if iv <= 0.001:
            continue

        points.append({
            "strike": round(strike, 2),
            "iv": round(iv * 100, 2),
            "type": "call",
        })

    for _, row in chain.puts.iterrows():
        strike = _safe_float(row.get("strike"))
        if strike <= 0:
            continue
        iv = _safe_float(row.get("impliedVolatility"))
        bid = _safe_float(row.get("bid"))
        ask = _safe_float(row.get("ask"))
        last = _safe_float(row.get("lastPrice"))
        mid = (bid + ask) / 2 if (bid + ask) > 0 else last

        if mid <= 0 and last <= 0:
            continue

        if iv < 0.01 and mid > 0:
            iv = implied_volatility(mid, stock_price, strike, T, RISK_FREE_RATE, "put")
        elif iv < 0.01 and last > 0:
            iv = implied_volatility(last, stock_price, strike, T, RISK_FREE_RATE, "put")

        if iv <= 0.001:
            continue

        points.append({
            "strike": round(strike, 2),
            "iv": round(iv * 100, 2),
            "type": "put",
        })

    points.sort(key=lambda p: p["strike"])
    return points, nearest_exp, stock_price


def plot_iv_smile_html(symbol: str) -> str:
    """Generate a standalone HTML file with a 2D IV smile plot for the nearest expiry.

    Returns the file path.
    """
    points, expiry, stock_price = _get_smile_data(symbol)

    calls = [p for p in points if p["type"] == "call"]
    puts = [p for p in points if p["type"] == "put"]

    today_str = dt.date.today().strftime("%Y%m%d")
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"iv_smile_{symbol}_{today_str}.html"

    calls_json = json.dumps(calls)
    puts_json = json.dumps(puts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IV Smile - {symbol} ({expiry})</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    margin: 0; padding: 20px;
    background: #1a1a2e; color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }}
  h1 {{ color: #00ff88; font-size: 20px; margin-bottom: 4px; }}
  .subtitle {{ color: #8892b0; font-size: 14px; margin-bottom: 16px; }}
  #chart {{ width: 100%; height: 600px; }}
</style>
</head>
<body>
<h1>IV Smile - {symbol}</h1>
<div class="subtitle">Expiry: {expiry} | Stock: ${stock_price:.2f} | Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
<div id="chart"></div>
<script>
var calls = {calls_json};
var puts = {puts_json};
var stockPrice = {stock_price};

var traceCalls = {{
  x: calls.map(function(p) {{ return p.strike; }}),
  y: calls.map(function(p) {{ return p.iv; }}),
  mode: 'lines+markers',
  name: 'Calls',
  line: {{ color: '#4fc3f7', width: 2 }},
  marker: {{ size: 5 }}
}};

var tracePuts = {{
  x: puts.map(function(p) {{ return p.strike; }}),
  y: puts.map(function(p) {{ return p.iv; }}),
  mode: 'lines+markers',
  name: 'Puts',
  line: {{ color: '#ff7043', width: 2 }},
  marker: {{ size: 5 }}
}};

var traceSpot = {{
  x: [stockPrice, stockPrice],
  y: [0, Math.max.apply(null, calls.map(function(p) {{ return p.iv; }}).concat(puts.map(function(p) {{ return p.iv; }}))) * 1.05],
  mode: 'lines',
  name: 'Spot Price',
  line: {{ color: '#ffd700', width: 2, dash: 'dash' }},
  hoverinfo: 'skip'
}};

var layout = {{
  paper_bgcolor: '#1a1a2e',
  plot_bgcolor: '#16213e',
  font: {{ color: '#e0e0e0', family: 'monospace' }},
  xaxis: {{
    title: 'Strike Price ($)',
    gridcolor: '#2a3a5c',
    zerolinecolor: '#2a3a5c'
  }},
  yaxis: {{
    title: 'Implied Volatility (%)',
    gridcolor: '#2a3a5c',
    zerolinecolor: '#2a3a5c'
  }},
  legend: {{ x: 0.02, y: 0.98 }},
  margin: {{ t: 20, r: 40, b: 60, l: 70 }},
  hovermode: 'closest'
}};

Plotly.newPlot('chart', [traceCalls, tracePuts, traceSpot], layout, {{responsive: true}});
</script>
</body>
</html>"""

    out_path.write_text(html)
    return str(out_path)


def plot_iv_surface_html(symbol: str) -> str:
    """Generate a standalone HTML file with a 3D IV surface plot using Plotly.js.

    X axis: Strike price
    Y axis: Days to expiry
    Z axis: Implied Volatility %
    Color: IV level (cool=low, hot=high)

    Returns the file path.
    """
    print(f"Fetching option chains for {symbol} (all expiries)...")
    surface_points = generate_iv_surface(symbol)

    if not surface_points:
        print(f"No IV surface data available for {symbol}")
        return ""

    # Get stock price for reference
    fi = yf.Ticker(symbol).fast_info
    stock_price = fi.get("lastPrice") or fi.get("previousClose") or 0

    # Build the surface: use calls for OTM (strike > spot) and puts for OTM (strike < spot)
    # This gives a cleaner composite surface
    composite = []
    for p in surface_points:
        if p["option_type"] == "call" and p["strike"] >= stock_price:
            composite.append(p)
        elif p["option_type"] == "put" and p["strike"] < stock_price:
            composite.append(p)

    # If composite is too sparse, fall back to all calls
    if len(composite) < 10:
        composite = [p for p in surface_points if p["option_type"] == "call"]

    if not composite:
        composite = surface_points

    # Deduplicate: keep one IV per (strike, dte) pair
    seen = {}
    for p in composite:
        key = (p["strike"], p["dte"])
        if key not in seen:
            seen[key] = p
    composite = list(seen.values())

    # Get unique sorted strikes and DTEs for the grid
    strikes = sorted(set(p["strike"] for p in composite))
    dtes = sorted(set(p["dte"] for p in composite))

    # Build lookup
    iv_lookup = {(p["strike"], p["dte"]): p["iv"] for p in composite}

    # Build 2D grid for surface plot (dtes x strikes)
    z_grid = []
    for dte_val in dtes:
        row = []
        for strike_val in strikes:
            iv_val = iv_lookup.get((strike_val, dte_val))
            row.append(iv_val)  # None = gap, Plotly handles it
        z_grid.append(row)

    today_str = dt.date.today().strftime("%Y%m%d")
    REPORTS_DIR.mkdir(exist_ok=True)
    out_path = REPORTS_DIR / f"iv_surface_{symbol}_{today_str}.html"

    # Also generate the smile data for nearest expiry (embedded in same page)
    smile_points, smile_expiry, _ = _get_smile_data(symbol)
    smile_calls = [p for p in smile_points if p["type"] == "call"]
    smile_puts = [p for p in smile_points if p["type"] == "put"]

    # Count expiries
    unique_expiries = sorted(set(p["expiry"] for p in composite))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IV Surface - {symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    margin: 0; padding: 20px;
    background: #1a1a2e; color: #e0e0e0;
    font-family: 'SF Mono', 'Fira Code', monospace;
  }}
  h1 {{ color: #00ff88; font-size: 22px; margin-bottom: 4px; }}
  h2 {{ color: #4fc3f7; font-size: 17px; margin-top: 30px; margin-bottom: 8px; }}
  .subtitle {{ color: #8892b0; font-size: 13px; margin-bottom: 16px; }}
  .stats {{
    display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 20px;
  }}
  .stat-box {{
    background: #1e2a4a; border: 1px solid #2a3a5c; border-radius: 8px;
    padding: 12px 18px; min-width: 140px;
  }}
  .stat-label {{ font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }}
  .stat-value {{ font-size: 20px; font-weight: bold; margin-top: 4px; }}
  .positive {{ color: #00ff88; }}
  .blue {{ color: #4fc3f7; }}
  .yellow {{ color: #ffd700; }}
  #surface {{ width: 100%; height: 650px; }}
  #smile {{ width: 100%; height: 450px; }}
</style>
</head>
<body>
<h1>IV Surface - {symbol}</h1>
<div class="subtitle">Stock: ${stock_price:.2f} | {len(unique_expiries)} expiries, {len(composite)} data points | Generated: {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}</div>

<div class="stats">
  <div class="stat-box">
    <div class="stat-label">Strike Range</div>
    <div class="stat-value blue">${min(strikes):.0f} - ${max(strikes):.0f}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">DTE Range</div>
    <div class="stat-value yellow">{min(dtes)} - {max(dtes)}d</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">IV Range</div>
    <div class="stat-value positive">{min(p['iv'] for p in composite):.1f}% - {max(p['iv'] for p in composite):.1f}%</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Expiries</div>
    <div class="stat-value">{len(unique_expiries)}</div>
  </div>
</div>

<h2>3D IV Surface</h2>
<div id="surface"></div>

<h2>IV Smile - Nearest Expiry ({smile_expiry})</h2>
<div id="smile"></div>

<script>
// ---- 3D Surface ----
var strikes = {json.dumps(strikes)};
var dtes = {json.dumps(dtes)};
var zGrid = {json.dumps(z_grid)};
var stockPrice = {stock_price};

var surfaceTrace = {{
  type: 'surface',
  x: strikes,
  y: dtes,
  z: zGrid,
  colorscale: [
    [0, '#0d47a1'],
    [0.2, '#1565c0'],
    [0.4, '#42a5f5'],
    [0.5, '#66bb6a'],
    [0.65, '#fdd835'],
    [0.8, '#ff7043'],
    [1.0, '#d32f2f']
  ],
  colorbar: {{
    title: {{ text: 'IV %', font: {{ color: '#e0e0e0' }} }},
    tickfont: {{ color: '#e0e0e0' }},
    thickness: 18,
    len: 0.8
  }},
  hovertemplate:
    'Strike: $%{{x:.0f}}<br>' +
    'DTE: %{{y}}d<br>' +
    'IV: %{{z:.1f}}%<extra></extra>',
  connectgaps: true
}};

var surfaceLayout = {{
  paper_bgcolor: '#1a1a2e',
  plot_bgcolor: '#16213e',
  font: {{ color: '#e0e0e0', family: 'monospace', size: 11 }},
  scene: {{
    xaxis: {{
      title: {{ text: 'Strike ($)', font: {{ size: 12 }} }},
      gridcolor: '#2a3a5c',
      backgroundcolor: '#16213e',
      color: '#8892b0'
    }},
    yaxis: {{
      title: {{ text: 'Days to Expiry', font: {{ size: 12 }} }},
      gridcolor: '#2a3a5c',
      backgroundcolor: '#16213e',
      color: '#8892b0'
    }},
    zaxis: {{
      title: {{ text: 'IV (%)', font: {{ size: 12 }} }},
      gridcolor: '#2a3a5c',
      backgroundcolor: '#16213e',
      color: '#8892b0'
    }},
    camera: {{
      eye: {{ x: 1.6, y: -1.6, z: 0.8 }}
    }}
  }},
  margin: {{ t: 20, r: 20, b: 20, l: 20 }}
}};

Plotly.newPlot('surface', [surfaceTrace], surfaceLayout, {{responsive: true}});

// ---- 2D Smile ----
var smileCalls = {json.dumps(smile_calls)};
var smilePuts = {json.dumps(smile_puts)};

var traceSmileCalls = {{
  x: smileCalls.map(function(p) {{ return p.strike; }}),
  y: smileCalls.map(function(p) {{ return p.iv; }}),
  mode: 'lines+markers',
  name: 'Calls',
  line: {{ color: '#4fc3f7', width: 2 }},
  marker: {{ size: 5 }}
}};

var traceSmilePuts = {{
  x: smilePuts.map(function(p) {{ return p.strike; }}),
  y: smilePuts.map(function(p) {{ return p.iv; }}),
  mode: 'lines+markers',
  name: 'Puts',
  line: {{ color: '#ff7043', width: 2 }},
  marker: {{ size: 5 }}
}};

var allIV = smileCalls.map(function(p) {{ return p.iv; }}).concat(smilePuts.map(function(p) {{ return p.iv; }}));
var maxIV = allIV.length > 0 ? Math.max.apply(null, allIV) * 1.05 : 100;

var traceSpot = {{
  x: [stockPrice, stockPrice],
  y: [0, maxIV],
  mode: 'lines',
  name: 'Spot Price',
  line: {{ color: '#ffd700', width: 2, dash: 'dash' }},
  hoverinfo: 'skip'
}};

var smileLayout = {{
  paper_bgcolor: '#1a1a2e',
  plot_bgcolor: '#16213e',
  font: {{ color: '#e0e0e0', family: 'monospace' }},
  xaxis: {{
    title: 'Strike Price ($)',
    gridcolor: '#2a3a5c',
    zerolinecolor: '#2a3a5c'
  }},
  yaxis: {{
    title: 'Implied Volatility (%)',
    gridcolor: '#2a3a5c',
    zerolinecolor: '#2a3a5c'
  }},
  legend: {{ x: 0.02, y: 0.98 }},
  margin: {{ t: 20, r: 40, b: 60, l: 70 }},
  hovermode: 'closest'
}};

Plotly.newPlot('smile', [traceSmileCalls, traceSmilePuts, traceSpot], smileLayout, {{responsive: true}});
</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"IV surface saved: {out_path}")
    return str(out_path)
