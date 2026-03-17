---
name: news
description: Daily news digest for your portfolio — headlines, sentiment, impact
user-invocable: true
---

# Alpha Trader — Portfolio News Digest

## Steps

1. Fetch news for all portfolio symbols:
   ```bash
   cd /Users/xxzhou/OSS/alpha-trader && ./at news
   ```

2. Present the output to the user as-is. It contains per-symbol news with sentiment scoring and a macro section.

3. If the user asks about a specific headline, explain its potential impact on options selling (e.g., bearish news = safer to sell calls, tariff news = higher IV expected).
