#!/usr/bin/env bash
# Alpha Trader biweekly strategy review
# Called by cron every other Friday at 4 PM PST

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M")

echo "=== Alpha Trader strategy review — ${TIMESTAMP} ==="

# Load .env for Telegram credentials
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Activate venv if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run strategy review (backtest + performance analysis)
python -m src.cli review --months 3

# Also run optimizer to check for parameter tuning
python -m src.cli optimize

echo "=== Review complete ==="
