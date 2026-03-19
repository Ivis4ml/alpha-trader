---
name: setup
description: Interactive portfolio setup — add/edit positions, set weekly target
user-invocable: true
---

# Alpha Trader — Portfolio Setup

## Usage
- `/setup` — run interactive setup (creates portfolio.yaml if missing, or re-run with --force)
- `/setup --force` — overwrite existing portfolio

## How to execute
```bash
cd /Users/xxzhou/OSS/alpha-trader && ./at setup --force
```

If the user just typed `/setup` and portfolio.yaml already exists, still use `--force` so they can re-enter their positions.

After setup completes, show a summary of what was saved.
