---
name: monitor
description: Check alerts on all positions — roll reminders, ITM risk, technical signals
user-invocable: true
---

# Alpha Trader — Position Monitor

Quick check on all alert conditions.

## Steps

1. Run alert check:
   ```bash
   cd /Users/xxzhou/OSS/alpha-trader && ./at alerts
   ```

2. Present alerts to user. If no alerts, say "All clear."

3. For URGENT alerts, suggest specific actions.
