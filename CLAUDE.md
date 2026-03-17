# Alpha Trader

AI-powered covered call & CSP advisor. Python data pipeline + Claude Opus decision engine.

## Project Structure
```
config.yaml            — positions, short calls, strategy params (user maintains)
.env                   — Telegram bot token & chat ID (not committed)
src/cli.py             — CLI entry point (all commands)
src/config.py          — config loader
src/data/fetcher.py    — market data, option chains, events (Yahoo Finance)
src/data/greeks.py     — Black-Scholes Greeks + IV solver
src/data/news.py       — news headlines, insider trades, analyst ratings
src/report.py          — briefing generator (structured markdown for Claude)
src/roll.py            — roll analysis for existing short calls
src/notify.py          — Telegram push notifications
src/bot.py             — interactive Telegram bot (long-polling daemon)
scripts/cron_scan.sh   — cron automation (8AM + 12PM PST Mon-Fri)
```

## Commands
```bash
# Core
python -m src.cli scan                    # Full scan → Claude analysis → action list
python -m src.cli scan --data-only        # Raw briefing only (no Claude)
python -m src.cli scan --notify           # Scan + send result to Telegram
python -m src.cli roll                    # Check short calls for roll/close opportunities

# Telegram
python -m src.cli notify --test           # Test Telegram connection
python -m src.cli notify                  # Send latest report to Telegram
python -m src.cli bot                     # Run interactive Telegram bot daemon

# Position management
python -m src.cli add-short AMZN 2026-04-17 225 10 2.50
python -m src.cli close-short AMZN 2026-04-17 225
python -m src.cli update-position AMZN 2900

# Scheduling
python -m src.cli cron install            # Mon-Fri 8AM + 12PM PST
python -m src.cli cron remove
python -m src.cli cron status
```

## Telegram Bot Commands
When running `python -m src.cli bot`:
- `/scan` — Full scan with AI analysis
- `/data` — Raw market data briefing
- `/positions` — Show portfolio & short calls
- `/roll` — Check roll candidates
- `/help` — Show commands

## How It Works
1. Python fetches: stock prices, VIX, option chains, IV stats, earnings, news, insider trades, analyst data
2. Generates structured briefing with all data + embedded trading policy
3. Pipes to `claude -p` (or used interactively via `/scan` skill)
4. Claude Opus analyzes and outputs concrete action list
5. Optionally sends to Telegram
6. User executes trades manually on Robinhood

## Data Sources (all free, no API keys)
- **Yahoo Finance** via yfinance: prices, option chains, IV, earnings, news, insider transactions, analyst ratings
- **Black-Scholes IV solver**: when Yahoo's reported IV is stale (weekends/after-hours), we solve for real IV from last traded price

## Strategy
- **Primary**: Covered call selling on existing RSU positions (AMZN, FTNT)
- **Secondary**: Cash-secured puts (only when appropriate)
- **Regime**: Auto-switches conservative/balanced/aggressive based on VIX
- **Goal**: ~$1,500/week premium, minimize assignment risk
- RSU positions — owner prefers NOT to be called away

## Key Config
- Edit `config.yaml` to update positions, record short calls, adjust strategy params
- After selling a call → `add-short` to record it
- After expiry/close → `close-short` to remove it
- Copy `.env.example` to `.env` and add Telegram credentials for notifications
