<div align="center">

# Alpha Trader

### AI-Powered Options Advisor That Learns From Your Trades

**`v0.0.1-alpha`**

*You decide. You execute. The system recommends, tracks, and gets smarter over time.*

![alt text](image.png)
![alt text](image-1.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB.svg)](https://python.org)
[![Claude Code](https://img.shields.io/badge/built%20with-Claude%20Code-6B4FBB.svg)](https://claude.ai)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**23 commands** · **5-factor scoring** · **self-learning optimizer** · **ML signals** · **multi-leg strategies**
**Robinhood-style charts** · **web dashboard** · **iOS remote** · **6 languages**

[Quick Start](#quick-start) · [How It Works](#how-it-works) · [Commands](#commands) · [Strategy](#strategy) · [iPhone Access](#access-from-iphone)

</div>

---

## Why Alpha Trader?

Most options tools either do too much (auto-trade with black-box logic) or too little (just show you data). Alpha Trader sits in the sweet spot:

| | Traditional Bot | Alpha Trader | Spreadsheet |
|---|---|---|---|
| Scans option chains | Yes | **Yes** | Manual |
| Scores candidates | Black box | **Transparent 5-factor model** | Manual formula |
| Learns from results | No | **Yes — auto-tunes parameters** | No |
| Tracks P&L | Sometimes | **Full SQLite journal** | Manual entry |
| Explains reasoning | No | **Claude Opus analyzes every trade** | N/A |
| Executes trades | Yes (risky) | **No — you stay in control** | N/A |

**Zero risk of rogue trades.** The system generates a concrete action list — exact strikes, quantities, prices — and you execute on Robinhood yourself.

---

## Quick Start

```bash
git clone https://github.com/Ivis4ml/alpha-trader.git && cd alpha-trader
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Configure your positions
vim config.yaml

# Your first scan
./at scan --data-only --quick
```

<details>
<summary><b>config.yaml example</b></summary>

```yaml
positions:
  AAPL:
    shares: 3000
    cost_basis: 180.0
    allow_assignment: false   # RSU — prefer not to be called away
  TSLA:
    shares: 4000
    cost_basis: 380.0
    allow_assignment: false

short_calls: []

strategy:
  weekly_target: 1500
  target_delta: 0.20
  preferred_dte: 10
  profit_take_pct: 50
  max_loss_multiple: 2.0
  auto_regime: true           # auto-switch based on VIX

language: en                  # en | zh | zh-tw | es | ja | ko
```
</details>

---

## How It Works

```
                          ┌─────────────────────┐
                          │    Yahoo Finance     │  Free. No API key.
                          │  (prices, options,   │
                          │   IV, news, insider) │
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    Data Pipeline     │  RSI, MACD, BB, ATR,
                          │   + 5-Factor Score   │  IV rank, earnings,
                          │                      │  unusual activity
                          └──────────┬──────────┘
                                     │
                          ┌──────────▼──────────┐
                          │    Claude Opus       │  Analyzes everything.
                          │   Decision Engine    │  Outputs action list.
                          └──────────┬──────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                   │
          ┌───────▼──────┐  ┌───────▼───────┐  ┌───────▼───────┐
          │  Action List  │  │  Sparkline    │  │  Dashboard    │
          │  (terminal)   │  │  (CC Remote)  │  │  (browser)    │
          └───────┬──────┘  └───────────────┘  └───────────────┘
                  │
          ┌───────▼──────┐
          │  You trade    │  On Robinhood.
          │  manually     │  Full control.
          └───────┬──────┘
                  │
          ┌───────▼──────┐
          │ Trade Journal │  SQLite. Every trade
          │   (SQLite)    │  tracked automatically.
          └───────┬──────┘
                  │
          ┌───────▼──────┐
          │  Optimizer    │  After 20 trades:
          │  (self-learn) │  tunes your parameters.
          └──────────────┘
```

---

## Daily Workflow

```bash
# ┌─── Morning (8 AM) ─────────────────────────────────┐
  /daily                    # What to do with open positions
  /scan                     # New covered call candidates
# └─────────────────────────────────────────────────────┘

# ┌─── You trade on Robinhood, then record it ─────────┐
  ./at add-short AAPL 2026-04-17 225 12 1.09
# └─────────────────────────────────────────────────────┘

# ┌─── Midweek ────────────────────────────────────────┐
  /monitor                  # Quick alert check
# └─────────────────────────────────────────────────────┘

# ┌─── Friday ─────────────────────────────────────────┐
  /daily                    # Expiry management
  ./at close-short AAPL 2026-04-17 225 --status expired
# └─────────────────────────────────────────────────────┘

# ┌─── Monthly ────────────────────────────────────────┐
  ./at report all           # P&L vs target
  ./at optimize             # Self-learning suggestions
# └─────────────────────────────────────────────────────┘
```

---

## Commands

### Scanning & Recommendations

| Command | Description | Speed |
|---------|-------------|-------|
| **`scan`** | Full market scan + Claude AI action list | ~3s (quick) / ~30s (full) |
| **`preview`** | Ultra-fast market snapshot | ~1.5s |
| **`daily`** | Open position advice: HOLD / CLOSE / ROLL / EXPIRE | ~3s |
| **`roll`** | Detailed roll analysis with specific candidates | ~5s |
| **`spreads`** | Multi-leg strategies (5 types) | ~10s |

<details>
<summary><b>scan — detailed usage</b></summary>

```bash
./at scan                              # Full scan → Claude analysis → action list
./at scan --quick                      # Fast: parallel fetch, skip news/insider
./at scan --data-only                  # Raw data only, no Claude
./at scan --notify                     # Scan + push to Telegram
```

Outputs: Market context (VIX, regime) → per-symbol technicals → scored option chain (marked `>>>`) → Claude's action list with exact strikes, quantities, premiums.
</details>

<details>
<summary><b>daily — position management rules</b></summary>

| Condition | Action |
|-----------|--------|
| Profit ≥ 50% captured | **CLOSE_PROFIT** |
| Premium ≥ 2x entry | **CLOSE_STOP** |
| DTE ≤ 5 + ITM | **ROLL** |
| DTE ≤ 3 + OTM + 75%+ captured | **LET_EXPIRE** |
| Earnings before expiry | **CLOSE_EARNINGS** |
| Otherwise | **HOLD** |

All thresholds configurable in `config.yaml`.
</details>

<details>
<summary><b>spreads — 5 multi-leg strategies</b></summary>

```bash
./at spreads --symbol AAPL                        # All strategies
./at spreads --symbol AAPL --strategy iron-condor  # Specific
```

| Strategy | What it does |
|----------|-------------|
| `bull-put` | Sell put + buy lower put. Defined risk premium collection. |
| `bear-call` | Sell call + buy higher call. Defined risk. |
| `iron-condor` | Both spreads combined. Profit from range-bound. |
| `collar` | Own stock + buy put + sell call. Hedge. |
| `pmcc` | Buy LEAPS + sell OTM call. Leveraged covered call. |
</details>

### Alerts & Monitoring

| Command | Description |
|---------|-------------|
| **`alerts`** | Check 8 alert conditions (DTE, ITM, RSI, BB, ATR) |
| **`/monitor`** | Same, as slash command for CC Remote |

<details>
<summary><b>Alert types</b></summary>

| Severity | Trigger |
|----------|---------|
| URGENT | DTE ≤ 2, ITM, or within 2% of strike |
| WARNING | RSI > 70, above Bollinger, ATR spike, expired position |
| INFO | RSI < 30, Bollinger squeeze |
</details>

### Trade Tracking & P&L

| Command | Description |
|---------|-------------|
| **`add-short`** | Record a new short call (saves to config + SQLite) |
| **`close-short`** | Record closure: expired / closed / assigned |
| **`update-position`** | Update share count |
| **`report`** | P&L summaries: weekly / monthly / all |

```bash
./at add-short AAPL 2026-04-17 225 12 1.09
./at close-short AAPL 2026-04-17 225 --status expired
./at report all
```

### Analysis & Visualization

| Command | Description |
|---------|-------------|
| **`spark`** | Unicode sparkline charts (inline in CC Remote conversation) |
| **`chart`** | Robinhood-style interactive charts (Plotly.js, browser) |
| **`iv-surface`** | 3D implied volatility surface |
| **`backtest`** | Simulate strategy on historical data |
| **`correlation`** | Position correlation + beta to SPY |
| **`earnings-crush`** | Historical IV crush around earnings |
| **`margin`** | Portfolio margin summary + optimization |

<details>
<summary><b>spark — inline chart example</b></summary>

Shows directly in Claude Code conversation, no browser needed:

```
🟢 AAPL $252.82 (+2.70, +1.1%)

Price  ▆▆▆▆▆▆▆▆▃▃▃▃▃▂   ▂▂▂▅▅▇█▆▇▂▄▃▄▆▇▄▄▄▃▃▃▂
       $246                              $278

RSI 14 ███░░░░░░░░░░░░ 24 (OVERSOLD)
MACD   ▄▃▂▃▃▅▇█▆▅▄▃▂▁▁▁▂▁   ▼ bearish
52w    ░░░░░░░░█░░░ $169—$289
```
</details>

<details>
<summary><b>chart — Robinhood-style</b></summary>

```bash
./at chart --symbol AAPL                    # Opens in browser
./at chart --symbol AAPL --no-open          # Generate + show dashboard URL
```

Features: Pure black background, Robinhood green/red, 1D/1W/1M/3M/1Y/ALL period selector, line ↔ candle toggle, SMA/BB/RSI/MACD/VOL toggles, trade markers, short call strike lines.

Also accessible via: `http://127.0.0.1:8080/chart/AAPL`
</details>

### Machine Learning

| Command | Description |
|---------|-------------|
| **`ml train`** | Train GradientBoosting model (14 features, 24mo history) |
| **`ml predict`** | Current signals: BUY_SIGNAL / NEUTRAL / CAUTION |
| **`ml features`** | Inspect feature values for a symbol |

### Self-Learning

| Command | Description |
|---------|-------------|
| **`optimize`** | Analyze trades → suggest parameter adjustments |

After every 20 closed trades, the optimizer: buckets by delta/DTE/IV → finds best-performing ranges → nudges parameters (max 15% per cycle) → you confirm → config updates.

### Paper Trading

| Command | Description |
|---------|-------------|
| **`paper status`** | Alpaca paper account balance + positions |
| **`paper chain`** | Real-time option chain with Greeks |
| **`paper submit`** | Submit covered call order |
| **`paper close`** | Close option position |
| **`paper orders`** | List open orders |

### Infrastructure

| Command | Description |
|---------|-------------|
| **`dashboard`** | Web UI: dark theme, real-time SSE, Chart.js |
| **`notify`** | Push latest report to Telegram |
| **`bot`** | Interactive Telegram bot daemon |
| **`cron`** | Schedule auto-scans (Mon-Fri 8AM + 12PM PST) |

---

## Access from iPhone

```bash
./scripts/start-remote.sh
# Scan QR code → opens in Claude iOS app
# Type /scan, /daily, /monitor — all work from your phone
```

Requires Claude Max subscription. Dashboard also starts automatically at `http://<mac-ip>:8080`.

---

## Strategy

### VIX-Based Regime Switching

| VIX | Regime | Delta | Approach |
|-----|--------|-------|----------|
| < 15 | Conservative | 0.08–0.15 | Safety first |
| 15–25 | Balanced | 0.15–0.25 | Premium vs protection |
| > 25 | Aggressive | 0.25–0.35 | Fat premiums |

### 5-Factor Scoring Model

```
score = delta_proximity  × 0.30    ← distance to your target delta
      + premium_richness × 0.25    ← annualized yield vs peers
      + theta_decay      × 0.20    ← time decay efficiency
      + iv_rank          × 0.15    ← selling when vol is rich
      + dte_preference   × 0.10    ← distance to preferred DTE
```

All weights auto-tune based on your trade results.

### Self-Learning Loop

```
Trade history (SQLite)
       ↓
Bucket by delta / DTE / IV rank
       ↓
Best win rate + avg P&L ranges
       ↓
Bounded nudge (max 15%)
       ↓
You approve → config updates → next scan improves
```

---

## Data Sources

All free. No API keys required for core functionality.

| Source | What |
|--------|------|
| **Yahoo Finance** | Prices, option chains, IV, earnings, news, insider, analyst |
| **Black-Scholes** | IV solver, Greeks calculation (works off-hours) |
| **Computed** | RSI-14, MACD, Bollinger Bands, ATR-14, IV rank |
| **Enhanced** | Unusual activity, put/call ratio, sector rotation, institutional |
| **Alpaca** *(optional)* | Paper trading validation with real-time Greeks |

---

## Architecture

```
alpha-trader/
├── at                       # Wrapper script (auto-activates venv)
├── config.yaml              # Your positions + strategy (you maintain)
├── .env                     # Credentials (not committed)
├── data/
│   ├── trades.db            # Trade journal (SQLite)
│   ├── ml_model.joblib      # Trained ML model
│   └── optimizer_log.json   # Parameter change log
├── src/
│   ├── cli.py               # 23 CLI commands
│   ├── strategy.py          # 5-factor scoring + position rules
│   ├── optimizer.py         # Self-learning parameter tuner
│   ├── db.py                # SQLite trade journal
│   ├── report.py            # Briefing generator for Claude
│   ├── alerts.py            # 8 alert conditions
│   ├── analytics.py         # Correlation, earnings crush, P&L attribution
│   ├── backtest.py          # Historical simulation
│   ├── multileg.py          # Spreads, condors, collars, PMCC
│   ├── margin.py            # Margin calculation + optimization
│   ├── ml_signals.py        # GradientBoosting signals
│   ├── charts.py            # Robinhood-style Plotly charts
│   ├── sparkline.py         # Unicode inline charts
│   ├── iv_surface.py        # 3D IV surface
│   ├── paper.py             # Alpaca paper trading
│   ├── dashboard.py         # Flask web UI + SSE streaming
│   ├── notify.py            # Telegram push
│   ├── bot.py               # Telegram interactive bot
│   └── data/
│       ├── fetcher.py       # Yahoo Finance pipeline + technicals
│       ├── greeks.py        # Black-Scholes + IV solver
│       ├── news.py          # News, insider, analyst
│       └── enhanced.py      # Unusual activity, P/C ratio
├── .claude/skills/          # /scan, /monitor, /lang
└── scripts/
    ├── start-remote.sh      # Claude Code Remote + Dashboard
    └── cron_scan.sh         # Scheduled scans
```

---

## Language

Set in `config.yaml` → `language: zh` or switch anytime with `/lang`:

| Code | Language |
|------|----------|
| `en` | English |
| `zh` | 简体中文 |
| `zh-tw` | 繁體中文 |
| `es` | Español |
| `ja` | 日本語 |
| `ko` | 한국어 |

---

## Security

- **Your data stays local** unless you explicitly use `/scan` (sends to Claude API) or Telegram. Use `--data-only` for fully offline analysis.
- **Dashboard auth**: Set `DASHBOARD_TOKEN` in `.env` when exposing on LAN.
- **Nothing secret in git**: `.env`, `trades.db`, model files all gitignored.

---

## Changelog

### v0.0.1-alpha (2026-03-17)

Initial release. Built in a single Claude Code session.

- 23 CLI commands: scan, trade tracking, analysis, visualization, automation
- 5-factor scoring model with auto-tunable weights
- Self-learning optimizer (bounded nudge, human-in-the-loop)
- GradientBoosting ML signals (14 features, TimeSeriesSplit CV)
- Technical indicators: RSI, MACD, Bollinger Bands, ATR
- Black-Scholes IV solver for off-hours Greeks
- Multi-leg strategies: bull put, bear call, iron condor, collar, PMCC
- Robinhood-style interactive charts + Unicode sparkline inline charts
- 3D IV surface visualization
- Web dashboard with SSE real-time streaming
- Claude Code Remote (iOS) + Telegram bot
- Alpaca paper trading integration
- SQLite trade journal with P&L reports
- Historical backtesting engine
- Portfolio correlation + margin optimization
- 6 output languages
- All data from free sources, no API keys required

---

<div align="center">

**Built with [Claude Code](https://claude.ai/claude-code)**

MIT License

</div>
