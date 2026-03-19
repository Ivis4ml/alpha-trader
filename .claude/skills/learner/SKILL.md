---
name: learner
description: Stage 2 candidate ranker — train, evaluate, backfill, shadow, slices, promote
user-invocable: true
---

# Alpha Trader — Candidate Ranker (Stage 2)

Manage the learned candidate ranker that improves option strike selection.

## Commands

Parse the user's intent and run the matching subcommand:

| User says | Run |
|-----------|-----|
| `/learner` or `/learner report` | `./at learner report` |
| `/learner train` | `./at learner backfill` then `./at learner train` |
| `/learner eval` | `./at learner eval` |
| `/learner backfill` | `./at learner backfill` |
| `/learner shadow` | `./at learner shadow` |
| `/learner slices` | `./at learner slices` |
| `/learner promote` | `./at learner promote` |

All commands use:
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at learner SUBCOMMAND
```

## Workflow

1. **First time**: `/learner backfill` → `/learner train` → `/learner eval`
2. **Regular check**: `/learner report` (shows readiness + model status)
3. **Before promoting**: `/learner shadow` + `/learner slices` + `/learner promote`
4. **After promoting**: Monitor via `/learner shadow` periodically

## Key metrics to highlight
- **Top-1 uplift**: model picks better than heuristic? (must be > 0)
- **Loss rate**: how often model picks worse? (must be < 20%)
- **Slice stability**: no symbol or regime with negative uplift?
- **Promotion gate**: all 5 checks must PASS before enabling model weight
