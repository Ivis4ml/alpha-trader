#!/usr/bin/env bash
# Start Claude Code Remote + Dashboard in tmux
# Dashboard runs on 0.0.0.0:8080 so iPhone can access charts via http://<mac-ip>:8080/chart/AAPL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

tmux kill-session -t alpha-trader 2>/dev/null || true

cd "$PROJECT_DIR"

# Get local IP for display
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || echo "127.0.0.1")

# Window 0: Claude Remote Control
tmux new-session -d -s alpha-trader \
    "cd $PROJECT_DIR && unset ANTHROPIC_API_KEY && claude remote-control --name 'Alpha Trader'; echo '=== Exited ==='; read"

# Window 1: Dashboard server (background, accessible from iPhone)
tmux new-window -t alpha-trader \
    "cd $PROJECT_DIR && source .venv/bin/activate && python -m src.cli dashboard --host 0.0.0.0 --port 8080; read"

tmux select-window -t alpha-trader:0

echo "Started:"
echo "  Claude Remote: tmux window 0 (attaching now)"
echo "  Dashboard:     http://${LOCAL_IP}:8080"
echo "  Charts:        http://${LOCAL_IP}:8080/chart/AAPL"
echo ""
tmux attach -t alpha-trader
