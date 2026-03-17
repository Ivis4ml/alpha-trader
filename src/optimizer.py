"""Self-learning optimizer — analyzes trade history and suggests parameter adjustments.

Triggered every N closed trades. All changes logged and require user confirmation.

Approach:
  1. Bucket trades by delta/DTE/IV rank
  2. Calculate win rate and avg P&L per bucket
  3. Find best-performing parameter ranges
  4. Nudge current parameters toward winning ranges (bounded, max 15% per cycle)
  5. Use correlation analysis to adjust scoring weights
"""

from __future__ import annotations

import json
import datetime as dt
import pathlib
from dataclasses import dataclass, field, asdict

import numpy as np

from .db import get_trade_history, get_cumulative_pnl
from .config import load_config, CONFIG_PATH

OPTIMIZER_LOG = pathlib.Path(__file__).resolve().parent.parent / "data" / "optimizer_log.json"
TRADES_BEFORE_OPTIMIZE = 20  # auto-trigger threshold


@dataclass
class BucketStats:
    bucket: str
    trades: int
    wins: int
    win_rate: float
    avg_pnl: float
    total_pnl: float


@dataclass
class ParameterSuggestion:
    param: str
    current: float
    suggested: float
    reason: str
    confidence: str  # HIGH, MEDIUM, LOW


@dataclass
class OptimizationResult:
    timestamp: str
    trades_analyzed: int
    delta_buckets: list[BucketStats]
    dte_buckets: list[BucketStats]
    iv_buckets: list[BucketStats]
    suggestions: list[ParameterSuggestion]
    weight_suggestions: dict[str, float]  # factor -> suggested weight


def _bucket_stats(trades: list[dict], key_fn, bucket_name_fn) -> list[BucketStats]:
    """Group trades into buckets and calculate stats."""
    buckets: dict[str, list[dict]] = {}
    for t in trades:
        k = key_fn(t)
        if k is not None:
            name = bucket_name_fn(k)
            buckets.setdefault(name, []).append(t)

    results = []
    for name, group in sorted(buckets.items()):
        wins = sum(1 for t in group if (t.get("pnl") or 0) > 0)
        pnls = [(t.get("pnl") or 0) for t in group]
        results.append(BucketStats(
            bucket=name,
            trades=len(group),
            wins=wins,
            win_rate=round(wins / len(group) * 100, 1) if group else 0,
            avg_pnl=round(sum(pnls) / len(pnls), 2) if pnls else 0,
            total_pnl=round(sum(pnls), 2),
        ))
    return results


def _delta_bucket(delta):
    if delta is None:
        return None
    if delta < 0.10:
        return 0.05
    elif delta < 0.15:
        return 0.10
    elif delta < 0.20:
        return 0.15
    elif delta < 0.25:
        return 0.20
    elif delta < 0.30:
        return 0.25
    else:
        return 0.30


def _dte_bucket(dte):
    if dte is None:
        return None
    if dte <= 7:
        return 7
    elif dte <= 14:
        return 14
    elif dte <= 21:
        return 21
    else:
        return 30


def _bounded_nudge(current: float, target: float, max_pct: float = 0.15) -> float:
    """Move current toward target, but no more than max_pct of current."""
    max_move = abs(current * max_pct)
    diff = target - current
    if abs(diff) <= max_move:
        return target
    return current + (max_move if diff > 0 else -max_move)


def analyze_and_suggest(config: dict | None = None) -> OptimizationResult:
    """Analyze closed trades and generate parameter adjustment suggestions."""
    if config is None:
        config = load_config()

    strat = config.get("strategy", {})
    trades = get_trade_history(limit=200)

    # Only analyze closed trades with P&L data
    closed = [t for t in trades if t.get("status") in ("expired", "closed", "assigned") and t.get("pnl") is not None]

    # Delta buckets
    delta_buckets = _bucket_stats(
        closed,
        lambda t: t.get("delta"),
        lambda d: f"{d:.2f}-{d+0.05:.2f}",
    )

    # DTE buckets (use the DTE at open — approximate from expiry - opened_at)
    def _get_dte(t):
        try:
            exp = dt.datetime.strptime(t["expiry"], "%Y-%m-%d").date()
            opened = dt.datetime.fromisoformat(t["opened_at"]).date()
            return (exp - opened).days
        except Exception:
            return None

    dte_buckets = _bucket_stats(
        closed,
        _get_dte,
        lambda d: f"{_dte_bucket(d)}d" if d else "?",
    )

    # IV buckets (we don't store IV at open currently, skip if not available)
    iv_buckets = []

    # ── Generate suggestions ──────────────────────────────────────
    suggestions = []

    # 1. Suggest target_delta based on best-performing bucket
    if delta_buckets:
        best_delta = max(delta_buckets, key=lambda b: b.avg_pnl)
        if best_delta.trades >= 5:  # Need minimum sample
            # Parse bucket name to get midpoint
            try:
                lo = float(best_delta.bucket.split("-")[0])
                suggested_delta = lo + 0.025  # midpoint
                current_delta = strat.get("target_delta", 0.20)
                if abs(suggested_delta - current_delta) > 0.01:
                    nudged = round(_bounded_nudge(current_delta, suggested_delta), 3)
                    suggestions.append(ParameterSuggestion(
                        param="target_delta",
                        current=current_delta,
                        suggested=nudged,
                        reason=f"Delta {best_delta.bucket} has best avg P&L (${best_delta.avg_pnl:.0f}, "
                               f"win rate {best_delta.win_rate:.0f}%, {best_delta.trades} trades)",
                        confidence="HIGH" if best_delta.trades >= 10 else "MEDIUM",
                    ))
            except (ValueError, IndexError):
                pass

    # 2. Suggest preferred_dte
    if dte_buckets:
        best_dte = max(dte_buckets, key=lambda b: b.avg_pnl)
        if best_dte.trades >= 5:
            try:
                suggested_dte = int(best_dte.bucket.replace("d", ""))
                current_dte = strat.get("preferred_dte", 10)
                if abs(suggested_dte - current_dte) > 2:
                    nudged = round(_bounded_nudge(current_dte, suggested_dte))
                    suggestions.append(ParameterSuggestion(
                        param="preferred_dte",
                        current=current_dte,
                        suggested=nudged,
                        reason=f"DTE {best_dte.bucket} has best avg P&L (${best_dte.avg_pnl:.0f}, "
                               f"win rate {best_dte.win_rate:.0f}%, {best_dte.trades} trades)",
                        confidence="HIGH" if best_dte.trades >= 10 else "MEDIUM",
                    ))
            except ValueError:
                pass

    # 3. Suggest profit_take_pct — check if trades that hit 50% continued to 60%+
    current_pt = strat.get("profit_take_pct", 50)
    profit_takes = [t for t in closed if t.get("pnl") and t.get("pnl") > 0]
    if len(profit_takes) >= 10:
        high_captures = sum(1 for t in profit_takes
                           if t.get("premium_per_contract") and t.get("close_price") is not None
                           and (t["premium_per_contract"] - (t["close_price"] or 0)) / t["premium_per_contract"] > 0.6)
        if high_captures / len(profit_takes) > 0.6 and current_pt < 60:
            suggestions.append(ParameterSuggestion(
                param="profit_take_pct",
                current=current_pt,
                suggested=min(current_pt + 10, 75),
                reason=f"{high_captures}/{len(profit_takes)} profitable trades reached 60%+ capture — "
                       f"current {current_pt}% threshold may be too early",
                confidence="MEDIUM",
            ))

    # ── Scoring weight suggestions (correlation-based) ────────────
    weight_suggestions = {}
    if len(closed) >= 15:
        # Build feature matrix: [delta_score_proxy, premium, theta_proxy, iv_proxy, dte_proxy]
        # and correlate each with P&L
        pnls = []
        features = {"delta": [], "premium": [], "dte": []}

        for t in closed:
            pnl = t.get("pnl", 0) or 0
            pnls.append(pnl)
            features["delta"].append(t.get("delta") or 0.20)
            features["premium"].append(t.get("premium_per_contract", 0) or 0)
            try:
                exp = dt.datetime.strptime(t["expiry"], "%Y-%m-%d").date()
                opened = dt.datetime.fromisoformat(t["opened_at"]).date()
                features["dte"].append((exp - opened).days)
            except Exception:
                features["dte"].append(10)

        pnl_arr = np.array(pnls)
        if pnl_arr.std() > 0:
            corrs = {}
            for feat_name, feat_vals in features.items():
                feat_arr = np.array(feat_vals, dtype=float)
                if feat_arr.std() > 0:
                    corr = np.corrcoef(feat_arr, pnl_arr)[0, 1]
                    corrs[feat_name] = abs(corr)

            # Normalize correlations to sum to 1 for weight suggestion
            total_corr = sum(corrs.values())
            if total_corr > 0:
                for k, v in corrs.items():
                    weight_suggestions[k] = round(v / total_corr, 3)

    return OptimizationResult(
        timestamp=dt.datetime.now().isoformat(),
        trades_analyzed=len(closed),
        delta_buckets=delta_buckets,
        dte_buckets=dte_buckets,
        iv_buckets=iv_buckets,
        suggestions=suggestions,
        weight_suggestions=weight_suggestions,
    )


def apply_suggestions(suggestions: list[ParameterSuggestion]) -> None:
    """Apply parameter suggestions to config.yaml."""
    import yaml
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    for s in suggestions:
        if s.param in config.get("strategy", {}):
            config["strategy"][s.param] = s.suggested

    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def log_optimization(result: OptimizationResult) -> None:
    """Append optimization result to log file."""
    OPTIMIZER_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = []
    if OPTIMIZER_LOG.exists():
        try:
            log = json.loads(OPTIMIZER_LOG.read_text())
        except Exception:
            log = []

    entry = {
        "timestamp": result.timestamp,
        "trades_analyzed": result.trades_analyzed,
        "suggestions": [asdict(s) for s in result.suggestions],
        "weight_suggestions": result.weight_suggestions,
    }
    log.append(entry)
    OPTIMIZER_LOG.write_text(json.dumps(log, indent=2))


def format_optimization(result: OptimizationResult) -> str:
    """Format optimization results for display."""
    lines = [f"## Strategy Optimization Report",
             f"Analyzed {result.trades_analyzed} closed trades\n"]

    if result.delta_buckets:
        lines.append("### Performance by Delta Range")
        lines.append("| Delta | Trades | Win Rate | Avg P&L | Total P&L |")
        lines.append("|-------|--------|----------|---------|-----------|")
        for b in result.delta_buckets:
            lines.append(f"| {b.bucket} | {b.trades} | {b.win_rate}% | ${b.avg_pnl:,.0f} | ${b.total_pnl:,.0f} |")
        lines.append("")

    if result.dte_buckets:
        lines.append("### Performance by DTE")
        lines.append("| DTE | Trades | Win Rate | Avg P&L | Total P&L |")
        lines.append("|-----|--------|----------|---------|-----------|")
        for b in result.dte_buckets:
            lines.append(f"| {b.bucket} | {b.trades} | {b.win_rate}% | ${b.avg_pnl:,.0f} | ${b.total_pnl:,.0f} |")
        lines.append("")

    if result.suggestions:
        lines.append("### Parameter Suggestions\n")
        for s in result.suggestions:
            arrow = "↑" if s.suggested > s.current else "↓"
            lines.append(f"**{s.param}**: {s.current} → {s.suggested} {arrow} [{s.confidence}]")
            lines.append(f"  {s.reason}\n")
    else:
        lines.append("No parameter changes suggested (need more data or current params are optimal).\n")

    if result.weight_suggestions:
        lines.append("### Scoring Weight Suggestions (correlation-based)")
        for k, v in result.weight_suggestions.items():
            lines.append(f"  {k}: {v:.3f}")
        lines.append("")

    lines.append("_Apply suggestions? Tell me 'apply' or 'reject'._")
    return "\n".join(lines)


def should_optimize() -> bool:
    """Check if we've hit the threshold for auto-optimization."""
    cum = get_cumulative_pnl()
    if not cum:
        return False
    closed_count = (cum.get("expired_count") or 0) + (cum.get("closed_count") or 0) + (cum.get("assigned_count") or 0)

    # Check last optimization
    if OPTIMIZER_LOG.exists():
        try:
            log = json.loads(OPTIMIZER_LOG.read_text())
            if log:
                last = log[-1]
                last_analyzed = last.get("trades_analyzed", 0)
                return (closed_count - last_analyzed) >= TRADES_BEFORE_OPTIMIZE
        except Exception:
            pass

    return closed_count >= TRADES_BEFORE_OPTIMIZE
