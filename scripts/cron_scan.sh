#!/usr/bin/env bash
# Alpha Trader scheduled scan
# Usage: cron_scan.sh [morning|midday]
# Called by cron at 8 AM and 12 PM PST, Mon-Fri

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SESSION="${1:-morning}"
TIMESTAMP=$(date +"%Y%m%d_%H%M")

echo "=== Alpha Trader ${SESSION} scan — ${TIMESTAMP} ==="

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

# Run scan with Claude analysis + Telegram notification
python -m src.cli scan --notify --model claude-opus-4-6

echo "=== Scan complete ==="
