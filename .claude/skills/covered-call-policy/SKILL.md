---
name: scan
description: Fetch market data and generate today's covered call action list
user-invocable: true
---

# Alpha Trader — Daily Options Scan

## Steps (TWO-PHASE — show preview immediately, then full analysis)

### Phase 1: Instant preview (<2s) — show to user RIGHT AWAY
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at preview
```
After running this, IMMEDIATELY output the preview result to the user. Say "Running full option scan..."

### Phase 2: Full data fetch (~5-10s)
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at scan --data-only --quick
```
If user says `/scan full`, drop --quick for news/insider/analyst data:
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at scan --data-only
```

### Phase 3: Generate ACTION LIST
Analyze the Phase 2 data following the INSTRUCTIONS FOR CLAUDE section. Key rules:
- **Tables first** — Positions, then Trades table with totals
- **Concise** — under 40 lines, no lengthy explanations
- End with: `_Ask "why [trade]?" or "explain" for details._`

## Other commands (all use `./at` prefix)
- User asks "why", "explain" → give full reasoning for that trade
- User says "roll" → `./at roll`
- User says "daily" → `./at daily`
- User says "alerts" or "monitor" → `./at alerts`
- User confirms a trade → `./at add-short SYMBOL EXPIRY STRIKE CONTRACTS PREMIUM`
- User asks to close → `./at close-short SYMBOL EXPIRY STRIKE`
- User wants chart → run `./at spark --symbol SYMBOL` for inline preview, then `./at chart --symbol SYMBOL --no-open` which outputs a clickable `http://IP:8080/chart/SYMBOL` URL for the full interactive Robinhood-style chart (requires dashboard running via start-remote.sh)
- User wants backtest → `./at backtest --symbol SYMBOL`
- User wants spreads → `./at spreads --symbol SYMBOL`
- User wants ML → `./at ml predict`
- User wants report → `./at report weekly`
- User wants correlation → `./at correlation`

## Output Style
- Tables first, then 2-3 bullet risks
- No explanations unless asked
- Short, scannable, mobile-friendly
