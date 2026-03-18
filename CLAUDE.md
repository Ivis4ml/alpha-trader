# Alpha Trader

AI-powered covered call & options advisor. Python data pipeline + Claude Opus decision engine.

## Project Structure
```
config.yaml            — strategy params (shared template)
portfolio.yaml         — user positions, short calls, cash, P&L (auto-created by setup)
.env                   — Telegram + Alpaca credentials (not committed)
src/cli.py             — CLI entry point (23+ commands)
src/config.py          — config loader (reads config.yaml + portfolio.yaml)
src/portfolio.py       — portfolio management, cash tracking, interactive setup
src/strategy.py        — 5-factor scoring model + position management rules
src/optimizer.py       — self-learning parameter tuner
src/db.py              — SQLite trade journal
src/data/fetcher.py    — market data, option chains, events, technicals (Yahoo Finance)
src/data/greeks.py     — Black-Scholes Greeks + IV solver
src/data/news.py       — news headlines, insider trades, analyst ratings
src/data/enhanced.py   — unusual options activity, P/C ratio, sector rotation
src/report.py          — briefing generator (structured markdown for Claude)
src/roll.py            — roll analysis for existing short calls
src/alerts.py          — 8 alert conditions (DTE, ITM, RSI, BB, ATR)
src/analytics.py       — correlation, earnings crush, tax lots, P&L attribution
src/backtest.py        — historical covered call simulation
src/multileg.py        — multi-leg strategies (spreads, condors, collars)
src/margin.py          — portfolio margin calculation + optimization
src/ml_signals.py      — ML signal generation (GradientBoosting)
src/charts.py          — Robinhood-style interactive charts (Plotly.js)
src/sparkline.py       — Unicode sparkline charts (inline in CC Remote)
src/iv_surface.py      — 3D IV surface visualization
src/paper.py           — Alpaca paper trading integration
src/dashboard.py       — Web UI (Flask + Chart.js + SSE streaming)
src/notify.py          — Telegram push notifications
src/bot.py             — interactive Telegram bot (long-polling daemon)
scripts/start-remote.sh — Claude Code Remote + Dashboard (iOS via tmux)
scripts/cron_scan.sh   — cron automation (8AM + 12PM PST Mon-Fri)
```

## Commands
```bash
# First time setup
./at setup                                # Interactive portfolio setup

# Core
./at scan                                 # Full scan → Claude analysis → action list
./at scan --data-only --quick             # Raw briefing only, fast mode
./at daily                                # Position advice: HOLD/CLOSE/ROLL/EXPIRE
./at preview                              # Ultra-fast market snapshot (<2s)
./at roll                                 # Check short calls for roll/close
./at alerts                               # Check all alert conditions

# Trade tracking
./at add-short NVDA 2026-04-17 225 10 2.50
./at close-short NVDA 2026-04-17 225 --status expired
./at update-position NVDA 3000
./at portfolio                            # Show positions, cash, P&L
./at report all                           # P&L reports

# Analysis
./at spark --symbol NVDA                  # Inline sparkline chart
./at chart --symbol NVDA                  # Robinhood-style interactive chart
./at backtest --symbol NVDA --months 12   # Historical simulation
./at correlation                          # Position correlation + beta
./at earnings-crush --symbol NVDA         # IV crush around earnings
./at iv-surface --symbol NVDA             # 3D IV surface
./at spreads --symbol NVDA                # Multi-leg strategies
./at margin                               # Portfolio margin summary

# ML
./at ml train                             # Train model
./at ml predict                           # Current signals

# Self-learning
./at review                               # Strategy review: perf vs target + backtest
./at optimize                             # Suggest parameter changes

# Paper trading
./at paper status                         # Alpaca paper account

# Telegram / Scheduling
./at notify --test
./at bot
./at cron install
```

## How It Works
1. `./at setup` — interactive portfolio setup (first time only)
2. Python fetches: stock prices, VIX, option chains, IV stats, earnings, news, insider trades, technicals
3. 5-factor scoring model ranks candidates
4. Pipes to `claude -p --reasoning-effort high` for AI analysis
5. Claude Opus outputs concrete action list
6. User executes trades manually, records with `add-short` / `close-short`
7. Cash and P&L auto-tracked in portfolio.yaml
8. After 20 trades, optimizer suggests parameter improvements

## Strategy
- **Primary**: Covered call selling on existing stock positions
- **Secondary**: Cash-secured puts, spreads, iron condors
- **Regime**: Auto-switches conservative/balanced/aggressive based on VIX
- **Self-learning**: Optimizer tunes delta/DTE/scoring weights from trade history

## Key Config
- `config.yaml` — strategy parameters (shared, in git)
- `portfolio.yaml` — your positions + state (private, gitignored, auto-created)
- `.env` — credentials (Telegram, Alpaca)
