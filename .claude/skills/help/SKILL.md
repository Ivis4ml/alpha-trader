---
name: help
description: Show all available Alpha Trader commands and slash commands
user-invocable: true
---

# Alpha Trader — Command Reference

When the user types `/help`, display this command list directly (do NOT run any bash command):

---

## Slash Commands (Claude Code Remote)

| Command | Description |
|---------|-------------|
| `/scan` | Full options scan → preview + data fetch + AI action list |
| `/scan full` | Full scan with news/insider/analyst data (slower) |
| `/monitor` | Check alerts — roll reminders, ITM risk, technicals |
| `/news` | Daily news digest for portfolio symbols |
| `/setup` | Interactive portfolio setup (add/edit positions, target) |
| `/lang CODE` | Switch output language (en/zh/zh-tw/es/ja/ko) |
| `/learner` | Candidate ranker status report |
| `/learner train` | Train utility ranker on labeled observations |
| `/learner eval` | Dataset stats + model evaluation metrics |
| `/learner backfill` | Label all past-expiry candidates for training |
| `/learner shadow` | Compare model vs heuristic on historical scans |
| `/learner slices` | Break down uplift by symbol, IV, regime, DTE |
| `/learner promote` | Check all promotion gates before enabling model |
| `/help` | This command list |

## CLI Commands (`./at` prefix)

### Core — Daily Workflow
| Command | Description |
|---------|-------------|
| `./at scan` | Full scan → Claude analysis → action list |
| `./at scan --data-only --quick` | Raw briefing only, fast mode |
| `./at daily` | Position advice: HOLD / CLOSE / ROLL / EXPIRE |
| `./at preview` | Ultra-fast market snapshot (<2s) |
| `./at roll` | Check short calls for roll/close opportunities |
| `./at alerts` | Check all alert conditions |

### Trade Tracking
| Command | Description |
|---------|-------------|
| `./at add-short SYMBOL EXPIRY STRIKE CONTRACTS PREMIUM` | Record new short call |
| `./at close-short SYMBOL EXPIRY STRIKE [--status expired]` | Close/expire a short |
| `./at update-position SYMBOL SHARES` | Update share count |
| `./at portfolio` | Show positions, cash, P&L |
| `./at report weekly/monthly/all` | P&L reports |

### Analysis
| Command | Description |
|---------|-------------|
| `./at spark --symbol NVDA` | Inline sparkline chart |
| `./at chart --symbol NVDA` | Interactive Robinhood-style chart |
| `./at backtest --symbol NVDA --months 12` | Historical covered call simulation |
| `./at correlation` | Position correlation + beta |
| `./at earnings-crush --symbol NVDA` | IV crush around earnings |
| `./at iv-surface --symbol NVDA` | 3D IV surface visualization |
| `./at spreads --symbol NVDA` | Multi-leg strategy analysis |
| `./at margin` | Portfolio margin summary |

### ML & Learner
| Command | Description |
|---------|-------------|
| `./at ml train` | Train timing signal model |
| `./at ml predict` | Show current ML signals |
| `./at learner report` | Ranker readiness + model status |
| `./at learner backfill` | Label past-expiry candidates |
| `./at learner train` | Train candidate utility ranker |
| `./at learner eval` | Dataset stats + metrics |
| `./at learner shadow` | Model vs heuristic comparison |
| `./at learner slices` | Uplift by symbol/IV/regime/DTE |
| `./at learner promote` | Check promotion gates |

### Self-Learning & Optimization
| Command | Description |
|---------|-------------|
| `./at review` | Strategy review: perf vs target + backtest |
| `./at optimize` | Performance report + parameter analysis |

### Infrastructure
| Command | Description |
|---------|-------------|
| `./at setup` | Interactive portfolio setup |
| `./at notify --test` | Test Telegram notification |
| `./at bot` | Start Telegram bot (long-polling) |
| `./at cron install` | Install cron jobs (8AM + 12PM PST) |
| `./at paper status` | Alpaca paper trading account |
| `./at dashboard` | Start web dashboard |
| `./at lang CODE` | Set output language |
| `./at news` | Portfolio news digest |

## Automatic (runs during `/scan`)
- Candidate observations logged to DB (all passed + rejected)
- Policy action slate generated and persisted
- Daily MTM backfill on open policy actions
- Terminal backfill from closed trades
- Learner shadow scoring (if model exists)
- Optimization trigger check
