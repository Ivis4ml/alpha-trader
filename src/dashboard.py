"""Alpha Trader Web Dashboard — single-file Flask server with embedded HTML/JS.

This is a LOCAL-ONLY dashboard (binds to 127.0.0.1 by default).
All data rendered in the UI comes from local config.yaml and SQLite — no
untrusted user input is processed, so innerHTML usage is safe in this context.
"""

from __future__ import annotations

import datetime as dt
import json
import queue
import threading

from flask import Flask, jsonify, Response

from .config import load_config, get_symbols, get_short_calls, get_position, contracts_available

app = Flask(__name__)

# ---------------------------------------------------------------------------
# SSE streaming infrastructure
# ---------------------------------------------------------------------------

# Thread-safe set of subscriber queues for SSE
_sse_subscribers: list[queue.Queue] = []
_sse_lock = threading.Lock()
_stream_thread: threading.Timer | None = None
_stream_running = False

STREAM_INTERVAL = 30  # seconds between price updates


def _fetch_price_update() -> dict:
    """Fetch latest prices for VIX, SPY, and all portfolio symbols."""
    try:
        import yfinance as yf
        config = load_config()
        symbols = get_symbols(config)

        vix_fi = yf.Ticker("^VIX").fast_info
        spy_fi = yf.Ticker("SPY").fast_info
        vix = vix_fi.get("lastPrice") or vix_fi.get("previousClose") or 0
        vix_prev = vix_fi.get("previousClose") or vix
        spy = spy_fi.get("lastPrice") or spy_fi.get("previousClose") or 0
        spy_prev = spy_fi.get("previousClose") or spy

        regimes = config.get("strategy", {}).get("regimes", {})
        if vix < regimes.get("conservative", {}).get("vix_below", 15):
            regime = "conservative"
        elif vix > regimes.get("aggressive", {}).get("vix_above", 25):
            regime = "aggressive"
        else:
            regime = "balanced"

        stocks = {}
        for sym in symbols:
            try:
                fi = yf.Ticker(sym).fast_info
                price = fi.get("lastPrice") or fi.get("previousClose") or 0
                prev = fi.get("previousClose") or price
                chg = (price - prev) / prev * 100 if prev else 0
                stocks[sym] = {
                    "price": round(price, 2),
                    "change_pct": round(chg, 2),
                    "prev_close": round(prev, 2),
                }
            except Exception:
                stocks[sym] = {"price": 0, "change_pct": 0, "prev_close": 0}

        return {
            "vix": round(vix, 2),
            "vix_change": round(vix - vix_prev, 2),
            "spy": round(spy, 2),
            "spy_change_pct": round((spy - spy_prev) / spy_prev * 100, 2) if spy_prev else 0,
            "regime": regime,
            "stocks": stocks,
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _broadcast_prices():
    """Fetch prices and push to all SSE subscribers. Reschedule via Timer."""
    global _stream_thread
    if not _stream_running:
        return

    data = _fetch_price_update()
    payload = json.dumps(data)

    with _sse_lock:
        dead = []
        for i, q in enumerate(_sse_subscribers):
            try:
                q.put_nowait(payload)
            except queue.Full:
                dead.append(i)
        # Clean up dead queues
        for i in reversed(dead):
            _sse_subscribers.pop(i)

    # Reschedule
    if _stream_running:
        _stream_thread = threading.Timer(STREAM_INTERVAL, _broadcast_prices)
        _stream_thread.daemon = True
        _stream_thread.start()


def _start_stream():
    """Start the background price streaming thread (idempotent)."""
    global _stream_running, _stream_thread
    if _stream_running:
        return
    _stream_running = True
    _stream_thread = threading.Timer(STREAM_INTERVAL, _broadcast_prices)
    _stream_thread.daemon = True
    _stream_thread.start()


def _stop_stream():
    """Stop the background streaming thread."""
    global _stream_running, _stream_thread
    _stream_running = False
    if _stream_thread:
        _stream_thread.cancel()
        _stream_thread = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj):
    """Convert objects to JSON-safe types."""
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _safe_json(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Auth: simple token-based access control for LAN exposure
# Set DASHBOARD_TOKEN in .env to require ?token=xxx on all requests
# If not set, dashboard is open (localhost-only use)
# ---------------------------------------------------------------------------
import os as _os
import functools as _functools

_DASHBOARD_TOKEN = _os.environ.get("DASHBOARD_TOKEN", "")


def _require_token(f):
    @_functools.wraps(f)
    def wrapper(*args, **kwargs):
        if _DASHBOARD_TOKEN:
            tok = request.args.get("token", "") or request.headers.get("X-Dashboard-Token", "")
            if tok != _DASHBOARD_TOKEN:
                return jsonify({"error": "unauthorized"}), 401
        return f(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# API Routes (all protected by token when DASHBOARD_TOKEN is set)
# ---------------------------------------------------------------------------

@app.route("/api/positions")
@_require_token
def api_positions():
    config = load_config()
    symbols = get_symbols(config)
    shorts = get_short_calls(config)
    positions = []
    for sym in symbols:
        pos = get_position(config, sym)
        avail = contracts_available(config, sym)
        sym_shorts = [s for s in shorts if s.get("symbol") == sym]
        positions.append({
            "symbol": sym,
            "shares": pos.get("shares", 0),
            "cost_basis": pos.get("cost_basis", 0),
            "allow_assignment": pos.get("allow_assignment", False),
            "available_contracts": avail,
            "short_calls": sym_shorts,
        })
    return jsonify({
        "positions": positions,
        "short_calls": shorts,
        "strategy": config.get("strategy", {}),
    })


@app.route("/api/trades")
@_require_token
def api_trades():
    try:
        from .db import get_trade_history, get_open_trades
        history = get_trade_history(limit=50)
        open_trades = get_open_trades()
        return jsonify({"trades": history, "open_trades": open_trades})
    except Exception as e:
        return jsonify({"trades": [], "open_trades": [], "error": str(e)})


@app.route("/api/summary")
@_require_token
def api_summary():
    try:
        from .db import get_weekly_summary, get_monthly_summary, get_cumulative_pnl
        weekly = get_weekly_summary(weeks=12)
        monthly = get_monthly_summary(months=6)
        cumulative = get_cumulative_pnl()
        config = load_config()
        strat = config.get("strategy", {})
        target = strat.get("weekly_target", 1500)

        # Rolling average and pace
        rolling_avg = 0
        annual_pace = 0
        pace_status = "NO_DATA"
        if weekly:
            premiums = [w["total_premium"] or 0 for w in weekly]
            rolling_avg = round(sum(premiums) / len(premiums), 2)
            annual_pace = round(rolling_avg * 52, 2)
            pct = (rolling_avg / target * 100) if target else 0
            pace_status = "ON_TRACK" if pct >= 85 else "BEHIND" if pct >= 50 else "WELL_BEHIND"

        return jsonify({
            "weekly": weekly,
            "monthly": monthly,
            "cumulative": cumulative,
            "weekly_target": target,
            "annual_target": strat.get("annual_target", target * 52),
            "rolling_avg": rolling_avg,
            "annual_pace": annual_pace,
            "pace_status": pace_status,
        })
    except Exception as e:
        return jsonify({
            "weekly": [], "monthly": [], "cumulative": {},
            "weekly_target": 1500, "error": str(e),
        })


@app.route("/api/market")
@_require_token
def api_market():
    try:
        import yfinance as yf
        config = load_config()
        symbols = get_symbols(config)

        # VIX and SPY
        vix_fi = yf.Ticker("^VIX").fast_info
        spy_fi = yf.Ticker("SPY").fast_info
        vix = vix_fi.get("lastPrice") or vix_fi.get("previousClose") or 0
        vix_prev = vix_fi.get("previousClose") or vix
        spy = spy_fi.get("lastPrice") or spy_fi.get("previousClose") or 0
        spy_prev = spy_fi.get("previousClose") or spy

        # Determine regime
        regimes = config.get("strategy", {}).get("regimes", {})
        if vix < regimes.get("conservative", {}).get("vix_below", 15):
            regime = "conservative"
        elif vix > regimes.get("aggressive", {}).get("vix_above", 25):
            regime = "aggressive"
        else:
            regime = "balanced"

        # Stock prices
        stocks = {}
        for sym in symbols:
            try:
                fi = yf.Ticker(sym).fast_info
                price = fi.get("lastPrice") or fi.get("previousClose") or 0
                prev = fi.get("previousClose") or price
                chg = (price - prev) / prev * 100 if prev else 0
                stocks[sym] = {
                    "price": round(price, 2),
                    "change_pct": round(chg, 2),
                    "prev_close": round(prev, 2),
                }
            except Exception:
                stocks[sym] = {"price": 0, "change_pct": 0, "prev_close": 0}

        return jsonify({
            "vix": round(vix, 2),
            "vix_change": round(vix - vix_prev, 2),
            "spy": round(spy, 2),
            "spy_change_pct": round((spy - spy_prev) / spy_prev * 100, 2) if spy_prev else 0,
            "regime": regime,
            "stocks": stocks,
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/stream")
@_require_token
def api_stream():
    """Server-Sent Events endpoint for real-time price updates.

    Every 30 seconds, pushes latest prices for VIX, SPY, and portfolio symbols.
    """
    _start_stream()

    def event_stream():
        q = queue.Queue(maxsize=50)
        with _sse_lock:
            _sse_subscribers.append(q)
        try:
            # Send an initial update immediately
            initial = json.dumps(_fetch_price_update())
            yield f"data: {initial}\n\n"

            while True:
                try:
                    data = q.get(timeout=60)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # Send keepalive comment to prevent timeout
                    yield ": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            with _sse_lock:
                try:
                    _sse_subscribers.remove(q)
                except ValueError:
                    pass

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/chart/<symbol>")
@_require_token
def chart_page(symbol):
    """Serve a Robinhood-style chart for any symbol. Reusable URL: /chart/AAPL"""
    from .charts import generate_chart
    from .config import get_short_calls

    symbol = symbol.upper()
    config = load_config()
    short_calls = [sc for sc in get_short_calls(config) if sc.get("symbol") == symbol]

    trades = []
    try:
        from .db import get_trade_history
        all_trades = get_trade_history(limit=50)
        trades = [t for t in all_trades if t.get("symbol") == symbol]
    except Exception:
        pass

    path = generate_chart(
        symbol=symbol,
        period="3mo",
        indicators=["sma", "bb", "rsi", "volume"],
        trades=trades,
        show_short_calls=short_calls,
    )

    if path:
        return open(path).read()
    return f"<h1>No data for {symbol}</h1>", 404


@app.route("/api/scan")
@_require_token
def api_scan():
    """Trigger a quick scan and return data as JSON."""
    try:
        config = load_config()
        from .data.fetcher import fetch_market_context, fetch_stock, fetch_iv_stats
        from .config import get_delta_range

        mkt = fetch_market_context(config)
        symbols = get_symbols(config)
        delta_lo, delta_hi = get_delta_range(config, mkt.regime)

        symbol_data = []
        for sym in symbols:
            stock = fetch_stock(sym)
            iv = fetch_iv_stats(sym)
            pos = get_position(config, sym)
            avail = contracts_available(config, sym)
            symbol_data.append({
                "symbol": sym,
                "price": stock.price,
                "change_pct": stock.day_change_pct,
                "sma_20": stock.sma_20,
                "sma_50": stock.sma_50,
                "high_52w": stock.high_52w,
                "low_52w": stock.low_52w,
                "iv_rank": iv.iv_rank,
                "iv_percentile": iv.iv_percentile,
                "current_iv": round(iv.current_iv * 100, 1),
                "shares": pos.get("shares", 0),
                "cost_basis": pos.get("cost_basis", 0),
                "available_contracts": avail,
                "technicals": _safe_json(stock.technicals) if stock.technicals else None,
            })

        return jsonify({
            "market": {
                "vix": mkt.vix,
                "vix_change": mkt.vix_change,
                "spy_price": mkt.spy_price,
                "spy_change_pct": mkt.spy_change_pct,
                "regime": mkt.regime,
                "timestamp": mkt.timestamp,
            },
            "delta_range": [delta_lo, delta_hi],
            "symbols": symbol_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)})


# ---------------------------------------------------------------------------
# Dashboard HTML (local-only; all data from local config/SQLite, not user input)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alpha Trader Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root {
  --bg: #1a1a2e;
  --bg2: #16213e;
  --bg3: #0f3460;
  --card: #1e2a4a;
  --border: #2a3a5c;
  --text: #e0e0e0;
  --text2: #8892b0;
  --green: #00ff88;
  --red: #ff4444;
  --yellow: #ffd700;
  --blue: #4fc3f7;
  --cyan: #00e5ff;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
  font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  line-height: 1.5;
}
a { color: var(--blue); text-decoration: none; }

/* Header bar */
.header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}
.logo {
  font-size: 18px;
  font-weight: bold;
  color: var(--green);
  letter-spacing: 1px;
}
.market-bar {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}
.market-item {
  display: flex;
  align-items: baseline;
  gap: 6px;
}
.market-label {
  color: var(--text2);
  font-size: 12px;
  text-transform: uppercase;
}
.market-value {
  font-size: 16px;
  font-weight: bold;
}
.regime-badge {
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 1px;
}
.regime-conservative { background: var(--blue); color: #000; }
.regime-balanced { background: var(--green); color: #000; }
.regime-aggressive { background: var(--red); color: #fff; }

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}
.btn {
  padding: 8px 16px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--bg3);
  color: var(--text);
  cursor: pointer;
  font-family: inherit;
  font-size: 13px;
  transition: all 0.2s;
}
.btn:hover { background: var(--border); }
.btn-primary {
  background: var(--green);
  color: #000;
  border-color: var(--green);
  font-weight: bold;
}
.btn-primary:hover { background: #00cc6a; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.timestamp {
  font-size: 11px;
  color: var(--text2);
}
.stream-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text2);
}
.stream-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text2);
  transition: background 0.3s;
}
.stream-dot.connected {
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
  animation: pulse 2s ease-in-out infinite;
}
.stream-dot.error {
  background: var(--red);
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}
.flash-update {
  animation: flashHighlight 0.6s ease-out;
}
@keyframes flashHighlight {
  0% { background-color: rgba(0, 255, 136, 0.15); }
  100% { background-color: transparent; }
}

/* Layout */
.container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.grid-full { grid-column: 1 / -1; }
@media (max-width: 900px) {
  .grid { grid-template-columns: 1fr; }
}

/* Cards */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
}
.card-title {
  font-size: 14px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-bottom: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.card-title .badge {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 8px;
  background: var(--bg3);
  color: var(--text2);
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
th {
  text-align: left;
  color: var(--text2);
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
}
td {
  padding: 8px 10px;
  border-bottom: 1px solid rgba(42, 58, 92, 0.4);
}
tr:hover td { background: rgba(79, 195, 247, 0.04); }
.mono { font-variant-numeric: tabular-nums; }
.positive { color: var(--green); }
.negative { color: var(--red); }
.neutral { color: var(--text2); }

/* Stats row */
.stats-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 20px;
}
.stat-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 16px 20px;
  flex: 1;
  min-width: 150px;
}
.stat-label {
  font-size: 11px;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 1px;
}
.stat-value {
  font-size: 24px;
  font-weight: bold;
  margin-top: 4px;
}

/* Chart */
.chart-container {
  position: relative;
  width: 100%;
  height: 250px;
}

/* DTE bar */
.dte-bar {
  background: var(--bg);
  border-radius: 4px;
  height: 6px;
  width: 100%;
  margin-top: 4px;
}
.dte-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}

/* Loading & error */
.loading {
  text-align: center;
  padding: 40px;
  color: var(--text2);
}
.loading::after {
  content: '';
  display: inline-block;
  width: 16px;
  height: 16px;
  margin-left: 8px;
  border: 2px solid var(--text2);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }
.error-msg {
  color: var(--red);
  font-size: 12px;
  padding: 8px;
  background: rgba(255, 68, 68, 0.1);
  border-radius: 6px;
  margin-top: 8px;
}
</style>
</head>
<body>

<div class="header">
  <div class="header-left">
    <div class="logo">ALPHA TRADER</div>
    <div class="market-bar" id="marketBar">
      <div class="market-item">
        <span class="market-label">VIX</span>
        <span class="market-value" id="vixValue">--</span>
        <span class="market-value" id="vixChange" style="font-size:12px">--</span>
      </div>
      <div class="market-item">
        <span class="market-label">SPY</span>
        <span class="market-value" id="spyValue">--</span>
        <span class="market-value" id="spyChange" style="font-size:12px">--</span>
      </div>
      <div class="market-item">
        <span class="market-label">Regime</span>
        <span class="regime-badge" id="regimeBadge">--</span>
      </div>
    </div>
  </div>
  <div class="header-right">
    <span class="stream-status">
      <span class="stream-dot" id="streamDot"></span>
      <span id="streamLabel">Connecting...</span>
    </span>
    <span class="timestamp" id="lastUpdate">--</span>
    <button class="btn" onclick="refreshAll()">Refresh</button>
    <button class="btn btn-primary" id="scanBtn" onclick="runScan()">Scan Now</button>
  </div>
</div>

<div class="container">
  <!-- Stats -->
  <div class="stats-row" id="statsRow">
    <div class="stat-box">
      <div class="stat-label">Total Premium</div>
      <div class="stat-value positive" id="statPremium">--</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Realized P&L</div>
      <div class="stat-value" id="statPnl">--</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Open Trades</div>
      <div class="stat-value" id="statOpen">--</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Total Trades</div>
      <div class="stat-value" id="statTotal">--</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Rolling Avg</div>
      <div class="stat-value" id="statRollingAvg">--</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Annual Pace</div>
      <div class="stat-value" id="statAnnualPace">--</div>
    </div>
  </div>

  <div class="grid">
    <!-- Positions -->
    <div class="card">
      <div class="card-title">
        Portfolio Positions
        <span class="badge" id="posCount">--</span>
      </div>
      <div id="positionsTable"><div class="loading">Loading positions</div></div>
    </div>

    <!-- Short Calls -->
    <div class="card">
      <div class="card-title">
        Open Short Calls
        <span class="badge" id="shortsCount">--</span>
      </div>
      <div id="shortsTable"><div class="loading">Loading short calls</div></div>
    </div>

    <!-- Premium Chart -->
    <div class="card">
      <div class="card-title">Weekly Premium</div>
      <div class="chart-container">
        <canvas id="premiumChart"></canvas>
      </div>
    </div>

    <!-- Stock Prices -->
    <div class="card">
      <div class="card-title">Stock Prices</div>
      <div id="stocksTable"><div class="loading">Loading prices</div></div>
    </div>

    <!-- Scan Results -->
    <div class="card grid-full" id="scanCard" style="display:none">
      <div class="card-title">
        Scan Results
        <span class="badge" id="scanTime">--</span>
      </div>
      <div id="scanResults"></div>
    </div>

    <!-- Trade History -->
    <div class="card grid-full">
      <div class="card-title">
        Trade History
        <span class="badge" id="tradeCount">--</span>
      </div>
      <div id="tradesTable"><div class="loading">Loading trades</div></div>
    </div>
  </div>
</div>

<script>
/* All data rendered in this dashboard comes from local config.yaml and
   local SQLite via our own Flask API endpoints on 127.0.0.1.
   No untrusted user input is processed. */

// ── Globals ──
let premiumChart = null;

// ── Helpers ──
function fmt(n, dec) {
  if (dec === undefined) dec = 2;
  if (n == null || isNaN(n)) return '--';
  return Number(n).toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec});
}
function fmtDollar(n, dec) {
  if (dec === undefined) dec = 2;
  if (n == null || isNaN(n)) return '--';
  return '$' + fmt(n, dec);
}
function fmtPct(n, dec) {
  if (dec === undefined) dec = 1;
  if (n == null || isNaN(n)) return '--';
  var sign = n >= 0 ? '+' : '';
  return sign + fmt(n, dec) + '%';
}
function colorClass(n) {
  if (n == null || isNaN(n)) return 'neutral';
  return n >= 0 ? 'positive' : 'negative';
}
function daysUntil(dateStr) {
  if (!dateStr) return null;
  var d = new Date(dateStr + 'T16:00:00');
  var now = new Date();
  return Math.ceil((d - now) / (1000 * 60 * 60 * 24));
}
function dteColor(dte) {
  if (dte <= 2) return 'var(--red)';
  if (dte <= 5) return 'var(--yellow)';
  return 'var(--green)';
}
function escapeHtml(str) {
  var div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.textContent;
}

// ── Safe DOM builder helpers ──
function setText(id, text) {
  document.getElementById(id).textContent = text;
}
function setHtml(id, html) {
  // NOTE: Only used with locally-generated content from our own API,
  // never with external/untrusted data
  document.getElementById(id).innerHTML = html;
}

// ── Market data ──
async function loadMarket() {
  try {
    var r = await fetch('/api/market');
    var d = await r.json();
    if (d.error) throw new Error(d.error);

    var vixEl = document.getElementById('vixValue');
    var vixChEl = document.getElementById('vixChange');
    var spyEl = document.getElementById('spyValue');
    var spyChEl = document.getElementById('spyChange');
    var regEl = document.getElementById('regimeBadge');

    vixEl.textContent = fmt(d.vix, 1);
    vixEl.className = 'market-value ' + (d.vix > 25 ? 'negative' : d.vix < 15 ? 'positive' : '');
    var vixPrev = d.vix - d.vix_change;
    vixChEl.textContent = '(' + fmtPct(vixPrev ? d.vix_change / vixPrev * 100 : 0, 1) + ')';
    vixChEl.className = 'market-value ' + colorClass(-d.vix_change);

    spyEl.textContent = fmtDollar(d.spy);
    spyChEl.textContent = '(' + fmtPct(d.spy_change_pct) + ')';
    spyChEl.className = 'market-value ' + colorClass(d.spy_change_pct);

    regEl.textContent = d.regime;
    regEl.className = 'regime-badge regime-' + d.regime;

    setText('lastUpdate', 'Updated: ' + d.timestamp);

    // Stocks table
    renderStocks(d.stocks);
  } catch (e) {
    console.error('Market load error:', e);
  }
}

function renderStocks(stocks) {
  var container = document.getElementById('stocksTable');
  if (!stocks || Object.keys(stocks).length === 0) {
    container.textContent = 'No stock data';
    return;
  }
  var html = '<table><thead><tr><th>Symbol</th><th>Price</th><th>Change</th></tr></thead><tbody>';
  for (var sym in stocks) {
    var data = stocks[sym];
    html += '<tr>'
      + '<td style="font-weight:bold">' + escapeHtml(sym) + '</td>'
      + '<td class="mono">' + fmtDollar(data.price) + '</td>'
      + '<td class="mono ' + colorClass(data.change_pct) + '">' + fmtPct(data.change_pct) + '</td>'
      + '</tr>';
  }
  html += '</tbody></table>';
  setHtml('stocksTable', html);
}

// ── Positions ──
async function loadPositions() {
  try {
    var r = await fetch('/api/positions');
    var d = await r.json();
    renderPositions(d.positions);
    renderShorts(d.short_calls);
  } catch (e) {
    setHtml('positionsTable', '<div class="error-msg">Failed to load positions</div>');
  }
}

function renderPositions(positions) {
  var container = document.getElementById('positionsTable');
  if (!positions || positions.length === 0) {
    container.textContent = 'No positions configured';
    setText('posCount', '0');
    return;
  }
  setText('posCount', String(positions.length));
  var html = '<table><thead><tr><th>Symbol</th><th>Shares</th><th>Cost Basis</th><th>Avail</th><th>Shorts</th></tr></thead><tbody>';
  for (var i = 0; i < positions.length; i++) {
    var p = positions[i];
    var shortCount = p.short_calls ? p.short_calls.length : 0;
    html += '<tr>'
      + '<td style="font-weight:bold">' + escapeHtml(p.symbol) + '</td>'
      + '<td class="mono">' + p.shares.toLocaleString() + '</td>'
      + '<td class="mono">' + fmtDollar(p.cost_basis) + '</td>'
      + '<td class="mono">' + p.available_contracts + '</td>'
      + '<td class="mono">' + shortCount + '</td>'
      + '</tr>';
  }
  html += '</tbody></table>';
  setHtml('positionsTable', html);
}

function renderShorts(shorts) {
  var container = document.getElementById('shortsTable');
  if (!shorts || shorts.length === 0) {
    container.textContent = 'No open short calls';
    setText('shortsCount', '0');
    return;
  }
  setText('shortsCount', String(shorts.length));
  var html = '<table><thead><tr><th>Symbol</th><th>Strike</th><th>Expiry</th><th>DTE</th><th>Qty</th><th>Premium</th></tr></thead><tbody>';
  for (var i = 0; i < shorts.length; i++) {
    var s = shorts[i];
    var dte = daysUntil(s.expiry);
    var dteStr = dte !== null ? String(dte) : '--';
    var dteFillPct = dte !== null ? Math.max(0, Math.min(100, dte / 21 * 100)) : 0;
    html += '<tr>'
      + '<td style="font-weight:bold">' + escapeHtml(s.symbol) + '</td>'
      + '<td class="mono">' + fmtDollar(s.strike) + '</td>'
      + '<td class="mono">' + escapeHtml(s.expiry) + '</td>'
      + '<td>'
      +   '<span class="mono" style="color:' + dteColor(dte) + '">' + dteStr + 'd</span>'
      +   '<div class="dte-bar"><div class="dte-fill" style="width:' + dteFillPct + '%;background:' + dteColor(dte) + '"></div></div>'
      + '</td>'
      + '<td class="mono">' + s.contracts + 'x</td>'
      + '<td class="mono positive">' + fmtDollar(s.premium_received) + '</td>'
      + '</tr>';
  }
  html += '</tbody></table>';
  setHtml('shortsTable', html);
}

// ── Trades & Summary ──
async function loadTrades() {
  try {
    var r = await fetch('/api/trades');
    var d = await r.json();
    renderTrades(d.trades);
  } catch (e) {
    setHtml('tradesTable', '<div class="error-msg">Failed to load trades</div>');
  }
}

function renderTrades(trades) {
  var container = document.getElementById('tradesTable');
  if (!trades || trades.length === 0) {
    container.textContent = 'No trade history yet. Use "add-short" CLI command to record trades.';
    setText('tradeCount', '0');
    return;
  }
  setText('tradeCount', String(trades.length));
  var statusColors = {
    'open': 'var(--blue)',
    'expired': 'var(--green)',
    'closed': 'var(--yellow)',
    'assigned': 'var(--red)'
  };
  var html = '<table><thead><tr>'
    + '<th>Date</th><th>Symbol</th><th>Action</th><th>Strike</th><th>Expiry</th>'
    + '<th>Qty</th><th>Premium</th><th>P&amp;L</th><th>Status</th>'
    + '</tr></thead><tbody>';
  for (var i = 0; i < trades.length; i++) {
    var t = trades[i];
    var date = t.opened_at ? t.opened_at.substring(0, 10) : '--';
    var statusColor = statusColors[t.status] || 'var(--text2)';
    html += '<tr>'
      + '<td class="mono">' + escapeHtml(date) + '</td>'
      + '<td style="font-weight:bold">' + escapeHtml(t.symbol) + '</td>'
      + '<td>' + escapeHtml(t.action) + '</td>'
      + '<td class="mono">' + fmtDollar(t.strike) + '</td>'
      + '<td class="mono">' + escapeHtml(t.expiry) + '</td>'
      + '<td class="mono">' + t.contracts + 'x</td>'
      + '<td class="mono positive">' + fmtDollar(t.total_premium) + '</td>'
      + '<td class="mono ' + colorClass(t.pnl) + '">' + (t.pnl != null ? fmtDollar(t.pnl) : '--') + '</td>'
      + '<td><span style="color:' + statusColor + ';font-weight:bold;text-transform:uppercase;font-size:11px">' + escapeHtml(t.status) + '</span></td>'
      + '</tr>';
  }
  html += '</tbody></table>';
  setHtml('tradesTable', html);
}

async function loadSummary() {
  try {
    var r = await fetch('/api/summary');
    var d = await r.json();

    // Stats boxes
    var c = d.cumulative || {};
    setText('statPremium', fmtDollar(c.total_premium_collected || 0, 0));
    var pnl = c.realized_pnl || 0;
    var pnlEl = document.getElementById('statPnl');
    pnlEl.textContent = fmtDollar(pnl, 0);
    pnlEl.className = 'stat-value ' + colorClass(pnl);
    setText('statOpen', String(c.open_count || 0));
    setText('statTotal', String(c.total_trades || 0));

    // Rolling avg + pace
    var avgEl = document.getElementById('statRollingAvg');
    if (avgEl) {
      avgEl.textContent = fmtDollar(d.rolling_avg || 0, 0) + '/wk';
      avgEl.className = 'stat-value ' + (d.pace_status === 'ON_TRACK' ? 'positive' : 'negative');
    }
    var paceEl = document.getElementById('statAnnualPace');
    if (paceEl) paceEl.textContent = fmtDollar(d.annual_pace || 0, 0) + '/yr';

    // Premium chart
    renderPremiumChart(d.weekly || [], d.weekly_target, d.rolling_avg);
  } catch (e) {
    console.error('Summary load error:', e);
  }
}

function renderPremiumChart(weekly, target, rollingAvg) {
  var ctx = document.getElementById('premiumChart');
  if (!ctx) return;

  // Reverse so oldest is first
  var data = weekly.slice().reverse();
  var labels = data.map(function(w) { return w.week_start ? w.week_start.substring(5) : '?'; });
  var values = data.map(function(w) { return w.total_premium || 0; });

  if (premiumChart) {
    premiumChart.destroy();
  }

  premiumChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Premium ($)',
        data: values,
        backgroundColor: values.map(function(v) { return v >= target ? 'rgba(0,255,136,0.6)' : 'rgba(79,195,247,0.6)'; }),
        borderColor: values.map(function(v) { return v >= target ? 'rgba(0,255,136,1)' : 'rgba(79,195,247,1)'; }),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          grid: { color: 'rgba(42,58,92,0.4)' },
          ticks: { color: '#8892b0', font: { family: 'monospace', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(42,58,92,0.4)' },
          ticks: {
            color: '#8892b0',
            font: { family: 'monospace', size: 11 },
            callback: function(v) { return '$' + v.toLocaleString(); }
          }
        }
      }
    },
    plugins: [{
      // Target line plugin
      id: 'targetLine',
      afterDatasetsDraw: function(chart) {
        if (!target) return;
        var yAxis = chart.scales.y;
        var y = yAxis.getPixelForValue(target);
        var ctx2 = chart.ctx;
        ctx2.save();
        ctx2.beginPath();
        ctx2.moveTo(chart.chartArea.left, y);
        ctx2.lineTo(chart.chartArea.right, y);
        ctx2.strokeStyle = 'rgba(255,215,0,0.6)';
        ctx2.lineWidth = 2;
        ctx2.setLineDash([6, 4]);
        ctx2.stroke();
        ctx2.fillStyle = 'rgba(255,215,0,0.8)';
        ctx2.font = '11px monospace';
        ctx2.fillText('Target $' + target.toLocaleString(), chart.chartArea.right - 100, y - 6);
        // Rolling average line
        if (rollingAvg) {
          var yAvg = yAxis.getPixelForValue(rollingAvg);
          ctx2.beginPath();
          ctx2.moveTo(chart.chartArea.left, yAvg);
          ctx2.lineTo(chart.chartArea.right, yAvg);
          ctx2.strokeStyle = 'rgba(0,255,136,0.5)';
          ctx2.lineWidth = 2;
          ctx2.setLineDash([3, 3]);
          ctx2.stroke();
          ctx2.fillStyle = 'rgba(0,255,136,0.7)';
          ctx2.fillText('Avg $' + rollingAvg.toLocaleString(), chart.chartArea.left + 4, yAvg - 6);
        }
        ctx2.restore();
      }
    }]
  });
}

// ── Scan ──
async function runScan() {
  var btn = document.getElementById('scanBtn');
  var card = document.getElementById('scanCard');
  btn.disabled = true;
  btn.textContent = 'Scanning...';
  card.style.display = 'block';
  setHtml('scanResults', '<div class="loading">Running scan (may take 30-60 seconds)</div>');

  try {
    var r = await fetch('/api/scan');
    var d = await r.json();
    if (d.error) throw new Error(d.error);
    renderScanResults(d);
    setText('scanTime', d.market && d.market.timestamp ? d.market.timestamp : 'now');
  } catch (e) {
    setHtml('scanResults', '<div class="error-msg">Scan failed: ' + escapeHtml(e.message) + '</div>');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Scan Now';
  }
}

function renderScanResults(data) {
  var mkt = data.market || {};
  var html = '<div style="margin-bottom:16px">'
    + '<strong>Market:</strong> VIX ' + fmt(mkt.vix, 1) + ' | SPY ' + fmtDollar(mkt.spy_price)
    + ' (' + fmtPct(mkt.spy_change_pct) + ') | Regime: <strong>' + escapeHtml((mkt.regime || '').toUpperCase()) + '</strong>'
    + ' | Delta Range: ' + fmt(data.delta_range ? data.delta_range[0] : null, 2) + ' - ' + fmt(data.delta_range ? data.delta_range[1] : null, 2)
    + '</div>';

  if (data.symbols && data.symbols.length > 0) {
    html += '<table><thead><tr>'
      + '<th>Symbol</th><th>Price</th><th>Change</th><th>IV Rank</th><th>RSI</th>'
      + '<th>MACD</th><th>Shares</th><th>Cost</th><th>Avail</th>'
      + '</tr></thead><tbody>';
    for (var i = 0; i < data.symbols.length; i++) {
      var s = data.symbols[i];
      var tech = s.technicals || {};
      var rsiClass = tech.rsi_14 > 70 ? 'negative' : tech.rsi_14 < 30 ? 'positive' : '';
      var macdDir = tech.macd_hist > 0 ? 'positive' : tech.macd_hist < 0 ? 'negative' : '';
      var macdLabel = tech.macd_hist != null ? (tech.macd_hist > 0 ? 'Bullish' : 'Bearish') : '--';
      html += '<tr>'
        + '<td style="font-weight:bold">' + escapeHtml(s.symbol) + '</td>'
        + '<td class="mono">' + fmtDollar(s.price) + '</td>'
        + '<td class="mono ' + colorClass(s.change_pct) + '">' + fmtPct(s.change_pct) + '</td>'
        + '<td class="mono">' + fmt(s.iv_rank, 0) + '%</td>'
        + '<td class="mono ' + rsiClass + '">' + (tech.rsi_14 != null ? fmt(tech.rsi_14, 0) : '--') + '</td>'
        + '<td class="mono ' + macdDir + '">' + macdLabel + '</td>'
        + '<td class="mono">' + s.shares.toLocaleString() + '</td>'
        + '<td class="mono">' + fmtDollar(s.cost_basis) + '</td>'
        + '<td class="mono">' + s.available_contracts + '</td>'
        + '</tr>';
    }
    html += '</tbody></table>';
  }

  setHtml('scanResults', html);
}

// ── Refresh all ──
async function refreshAll() {
  await Promise.all([loadMarket(), loadPositions(), loadTrades(), loadSummary()]);
}

// ── Real-time SSE streaming ──
function applyStreamUpdate(d) {
  if (d.error) {
    console.warn('Stream update error:', d.error);
    return;
  }

  // Update VIX
  var vixEl = document.getElementById('vixValue');
  var vixChEl = document.getElementById('vixChange');
  if (vixEl && d.vix != null) {
    vixEl.textContent = fmt(d.vix, 1);
    vixEl.className = 'market-value ' + (d.vix > 25 ? 'negative' : d.vix < 15 ? 'positive' : '');
    vixEl.classList.add('flash-update');
    setTimeout(function() { vixEl.classList.remove('flash-update'); }, 700);
  }
  if (vixChEl && d.vix_change != null) {
    var vixPrev = d.vix - d.vix_change;
    vixChEl.textContent = '(' + fmtPct(vixPrev ? d.vix_change / vixPrev * 100 : 0, 1) + ')';
    vixChEl.className = 'market-value ' + colorClass(-d.vix_change);
  }

  // Update SPY
  var spyEl = document.getElementById('spyValue');
  var spyChEl = document.getElementById('spyChange');
  if (spyEl && d.spy != null) {
    spyEl.textContent = fmtDollar(d.spy);
    spyEl.classList.add('flash-update');
    setTimeout(function() { spyEl.classList.remove('flash-update'); }, 700);
  }
  if (spyChEl && d.spy_change_pct != null) {
    spyChEl.textContent = '(' + fmtPct(d.spy_change_pct) + ')';
    spyChEl.className = 'market-value ' + colorClass(d.spy_change_pct);
  }

  // Update regime
  var regEl = document.getElementById('regimeBadge');
  if (regEl && d.regime) {
    regEl.textContent = d.regime;
    regEl.className = 'regime-badge regime-' + d.regime;
  }

  // Update timestamp
  if (d.timestamp) {
    setText('lastUpdate', 'Live: ' + d.timestamp);
  }

  // Update stock prices table
  if (d.stocks) {
    renderStocks(d.stocks);
  }
}

var sseSource = null;
var sseReconnectDelay = 1000;

function connectSSE() {
  if (sseSource) {
    sseSource.close();
  }

  var dot = document.getElementById('streamDot');
  var label = document.getElementById('streamLabel');

  sseSource = new EventSource('/api/stream');

  sseSource.onopen = function() {
    dot.className = 'stream-dot connected';
    label.textContent = 'Live';
    sseReconnectDelay = 1000;
  };

  sseSource.onmessage = function(event) {
    try {
      var data = JSON.parse(event.data);
      applyStreamUpdate(data);
    } catch (e) {
      console.error('SSE parse error:', e);
    }
  };

  sseSource.onerror = function() {
    dot.className = 'stream-dot error';
    label.textContent = 'Reconnecting...';
    sseSource.close();
    // Exponential backoff reconnect
    setTimeout(function() {
      sseReconnectDelay = Math.min(sseReconnectDelay * 2, 30000);
      connectSSE();
    }, sseReconnectDelay);
  };
}

// ── Init ──
document.addEventListener('DOMContentLoaded', function() {
  refreshAll();
  connectSSE();
});
// Auto-refresh non-streaming data every 5 minutes
setInterval(function() {
  Promise.all([loadPositions(), loadTrades(), loadSummary()]);
}, 300000);
</script>
</body>
</html>
"""


@app.route("/")
@_require_token
def index():
    return Response(DASHBOARD_HTML, mimetype="text/html")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dashboard(host: str = "127.0.0.1", port: int = 8080, debug: bool = False):
    """Start the dashboard server."""
    print(f"\n  Alpha Trader Dashboard")
    print(f"  http://{host}:{port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=host, port=port, debug=debug)
