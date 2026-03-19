"""Strategy performance reporter — analyzes trade history and reports findings.

Descriptive analysis only. Does NOT auto-suggest parameter changes, because
the trade history suffers from selection bias (we only see outcomes of trades
we chose to make, not counterfactuals). Use candidate_observations for
unbiased analysis once enough data accumulates.

Reports:
  1. Bucket trades by delta/DTE/IV rank/regime
  2. Calculate win rate, avg P&L, utility metrics per bucket
  3. Premium capture rate, assignment rate, called-away cost
  4. Observation coverage stats (how much unbiased data we have)
"""

from __future__ import annotations

import json
import datetime as dt
import pathlib
from dataclasses import dataclass, field, asdict

from .db import get_trade_history, get_cumulative_pnl, get_observation_stats
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
class UtilityMetrics:
    """Aggregate utility metrics across all closed trades."""
    total_premium_collected: float
    total_realized_pnl: float
    premium_capture_rate: float   # realized / collected as %
    assignment_rate: float        # assigned / total as %
    assignment_count: int
    called_away_cost: float       # premium lost to assignments (negative PnL trades)
    avg_pnl_per_trade: float
    best_trade_pnl: float
    worst_trade_pnl: float
    pnl_std: float                # trade PnL standard deviation
    sharpe_proxy: float           # avg_pnl / pnl_std (trade-level, not time-weighted)
    win_rate: float
    # Target tracking
    weeks_tracked: int
    weeks_on_target: int          # weeks where premium >= weekly target


@dataclass
class ParameterSuggestion:
    """Kept for backwards compatibility — but no longer auto-generated."""
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
    regime_buckets: list[BucketStats] = field(default_factory=list)
    utility: UtilityMetrics | None = None
    observation_stats: dict = field(default_factory=dict)
    suggestions: list[ParameterSuggestion] = field(default_factory=list)
    # Removed: weight_suggestions — was correlation-based on biased data
    weight_suggestions: dict[str, float] = field(default_factory=dict)


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


def _compute_utility_metrics(closed: list[dict], config: dict) -> UtilityMetrics:
    """Compute utility-oriented metrics from closed trades."""
    import numpy as np

    pnls = [(t.get("pnl") or 0) for t in closed]
    premiums = [(t.get("total_premium") or 0) for t in closed]
    total_collected = sum(premiums)
    total_realized = sum(pnls)
    assigned = [t for t in closed if t.get("status") == "assigned"]
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]

    # Called-away cost: sum of negative PnL from assignments
    called_away_cost = sum(
        abs(t.get("pnl") or 0) for t in assigned if (t.get("pnl") or 0) < 0
    )

    pnl_arr = np.array(pnls) if pnls else np.array([0.0])
    pnl_std = float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0
    avg_pnl = float(np.mean(pnl_arr))
    sharpe = avg_pnl / pnl_std if pnl_std > 0 else 0.0

    # Weekly target tracking
    from .db import get_weekly_summary
    strat = config.get("strategy", {})
    weekly_target = strat.get("income_goal", {}).get("weekly_target", 0)
    weeks = get_weekly_summary(weeks=52)
    weeks_on_target = sum(1 for w in weeks if (w.get("total_premium") or 0) >= weekly_target) if weekly_target > 0 else 0

    return UtilityMetrics(
        total_premium_collected=round(total_collected, 2),
        total_realized_pnl=round(total_realized, 2),
        premium_capture_rate=round(total_realized / total_collected * 100, 1) if total_collected > 0 else 0,
        assignment_rate=round(len(assigned) / len(closed) * 100, 1) if closed else 0,
        assignment_count=len(assigned),
        called_away_cost=round(called_away_cost, 2),
        avg_pnl_per_trade=round(avg_pnl, 2),
        best_trade_pnl=round(max(pnls), 2) if pnls else 0,
        worst_trade_pnl=round(min(pnls), 2) if pnls else 0,
        pnl_std=round(pnl_std, 2),
        sharpe_proxy=round(sharpe, 3),
        win_rate=round(len(wins) / len(closed) * 100, 1) if closed else 0,
        weeks_tracked=len(weeks),
        weeks_on_target=weeks_on_target,
    )


def analyze_and_suggest(config: dict | None = None) -> OptimizationResult:
    """Analyze closed trades and generate a performance report.

    NOTE: Parameter suggestions are disabled. The trade history has selection
    bias (we only see outcomes of trades we chose). Use candidate_observations
    for unbiased analysis. This function now returns descriptive stats only.
    """
    if config is None:
        config = load_config()

    trades = get_trade_history(limit=200)
    closed = [t for t in trades if t.get("status") in ("expired", "closed", "assigned") and t.get("pnl") is not None]

    # Delta buckets
    delta_buckets = _bucket_stats(
        closed,
        lambda t: t.get("delta"),
        lambda d: f"{d:.2f}-{d+0.05:.2f}",
    )

    # DTE buckets
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

    # IV rank buckets
    iv_trades = [t for t in closed if t.get("iv_rank") is not None]
    iv_buckets = _bucket_stats(
        iv_trades,
        lambda t: t.get("iv_rank"),
        lambda iv: f"IV {int(iv // 20) * 20}-{int(iv // 20) * 20 + 20}",
    ) if iv_trades else []

    # Regime buckets
    regime_trades = [t for t in closed if t.get("regime")]
    regime_buckets = _bucket_stats(
        regime_trades,
        lambda t: t.get("regime"),
        lambda r: r,
    ) if regime_trades else []

    # Utility metrics
    utility = _compute_utility_metrics(closed, config) if closed else None

    # Observation stats (how much unbiased data we have)
    obs_stats = {}
    try:
        obs_stats = get_observation_stats()
    except Exception:
        pass

    return OptimizationResult(
        timestamp=dt.datetime.now().isoformat(),
        trades_analyzed=len(closed),
        delta_buckets=delta_buckets,
        dte_buckets=dte_buckets,
        iv_buckets=iv_buckets,
        regime_buckets=regime_buckets,
        utility=utility,
        observation_stats=obs_stats,
        suggestions=[],   # Disabled: selection bias makes suggestions unreliable
        weight_suggestions={},  # Disabled: correlation on biased sample is misleading
    )


def apply_suggestions(suggestions: list[ParameterSuggestion]) -> None:
    """Apply parameter suggestions to config.yaml.

    Kept for backwards compatibility but suggestions list will be empty
    until we have unbiased candidate_observations data to learn from.
    """
    if not suggestions:
        return
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

    # Capture policy stats at optimization time for future delta comparison
    policy_terminal = 0
    policy_realized = 0
    try:
        from .db import get_policy_stats
        ps = get_policy_stats()
        policy_terminal = ps.get("meaningful_terminal", 0) or 0
        policy_realized = ps.get("meaningful_realized", 0) or 0
    except Exception:
        pass

    entry = {
        "timestamp": result.timestamp,
        "trades_analyzed": result.trades_analyzed,
        "suggestions": [asdict(s) for s in result.suggestions],
        "policy_terminal_at_optimize": policy_terminal,
        "policy_realized_at_optimize": policy_realized,
    }
    log.append(entry)
    OPTIMIZER_LOG.write_text(json.dumps(log, indent=2))


def format_optimization(result: OptimizationResult) -> str:
    """Format optimization results for display."""
    lines = [f"## Strategy Performance Report",
             f"Analyzed {result.trades_analyzed} closed trades\n"]

    # Utility metrics first — the most important section
    if result.utility:
        u = result.utility
        lines.append("### Utility Metrics")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Premium Collected | ${u.total_premium_collected:,.0f} |")
        lines.append(f"| Realized P&L | ${u.total_realized_pnl:,.0f} |")
        lines.append(f"| Premium Capture Rate | {u.premium_capture_rate:.1f}% |")
        lines.append(f"| Win Rate | {u.win_rate:.1f}% |")
        lines.append(f"| Assignment Rate | {u.assignment_rate:.1f}% ({u.assignment_count} trades) |")
        lines.append(f"| Called-Away Cost | ${u.called_away_cost:,.0f} |")
        lines.append(f"| Avg P&L/Trade | ${u.avg_pnl_per_trade:,.0f} |")
        lines.append(f"| Best Trade | ${u.best_trade_pnl:,.0f} |")
        lines.append(f"| Worst Trade | ${u.worst_trade_pnl:,.0f} |")
        lines.append(f"| P&L Std Dev | ${u.pnl_std:,.0f} |")
        lines.append(f"| Sharpe (trade-level) | {u.sharpe_proxy:.3f} |")
        if u.weeks_tracked > 0:
            lines.append(f"| Weeks on Target | {u.weeks_on_target}/{u.weeks_tracked} |")
        lines.append("")

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

    if result.regime_buckets:
        lines.append("### Performance by Market Regime")
        lines.append("| Regime | Trades | Win Rate | Avg P&L | Total P&L |")
        lines.append("|--------|--------|----------|---------|-----------|")
        for b in result.regime_buckets:
            lines.append(f"| {b.bucket} | {b.trades} | {b.win_rate}% | ${b.avg_pnl:,.0f} | ${b.total_pnl:,.0f} |")
        lines.append("")

    if result.iv_buckets:
        lines.append("### Performance by IV Rank")
        lines.append("| IV Rank | Trades | Win Rate | Avg P&L | Total P&L |")
        lines.append("|---------|--------|----------|---------|-----------|")
        for b in result.iv_buckets:
            lines.append(f"| {b.bucket} | {b.trades} | {b.win_rate}% | ${b.avg_pnl:,.0f} | ${b.total_pnl:,.0f} |")
        lines.append("")

    # Observation data coverage
    if result.observation_stats:
        obs = result.observation_stats
        total = obs.get("total_observations", 0)
        scans = obs.get("total_scans", 0)
        filled = obs.get("outcomes_filled", 0)
        lines.append("### Candidate Observation Coverage")
        lines.append(f"  Scans logged: {scans}")
        lines.append(f"  Total candidates observed: {total}")
        lines.append(f"  Outcomes backfilled: {filled}")
        if total > 0 and filled < total:
            lines.append(f"  (Run outcome backfill to enable unbiased analysis)")
        lines.append("")

    # Policy layer stats
    try:
        from .db import get_policy_stats
        ps = get_policy_stats()
        if ps and (ps.get("total_actions") or 0) > 0:
            lines.append("### Policy Layer Coverage")
            lines.append(f"  Decisions recorded: {ps.get('total_decisions', 0)}")
            lines.append(f"  Total actions: {ps.get('total_actions', 0)}")
            lines.append(f"  Selected (by allocator): {ps.get('selected_actions', 0)}")
            lines.append(f"  Realized (user executed): {ps.get('realized_actions', 0)}")
            lines.append(f"  Terminal (with outcome): {ps.get('terminal_actions', 0)}")
            lines.append(f"  Meaningful terminal: {ps.get('meaningful_terminal', 0)}")
            lines.append(f"  With reward: {ps.get('rewarded_actions', 0)}")
            lines.append("")
    except Exception:
        pass

    # No more auto-suggestions
    lines.append("_Note: Auto parameter suggestions disabled. Trade history has_")
    lines.append("_selection bias — use candidate observations for unbiased analysis._")

    return "\n".join(lines)


def should_optimize() -> bool:
    """Check if optimization should run, using data-driven triggers.

    Triggers (any one is sufficient):
    1. New terminal policy actions >= 40 since last optimization
    2. New realized (chosen) policy actions >= 25 since last optimization
    3. Legacy: new closed trades >= 20 (backwards compat)

    Safety: minimum 7 days since last optimization.
    """
    import datetime as _dt

    # Minimum cooldown: 7 days since last optimization
    if OPTIMIZER_LOG.exists():
        try:
            log = json.loads(OPTIMIZER_LOG.read_text())
            if log:
                last_ts = log[-1].get("timestamp", "")
                last_date = _dt.datetime.fromisoformat(last_ts).date()
                if (_dt.date.today() - last_date).days < 7:
                    return False
        except Exception:
            pass

    # Trigger 1 & 2: Policy layer data-driven triggers
    # Uses "meaningful" counts (OPEN/ADD/CLOSE/ROLL only) to avoid
    # HOLD/SKIP auto-selections inflating the trigger.
    try:
        from .db import get_policy_stats
        policy_stats = get_policy_stats()
        # meaningful_terminal: OPEN/ADD/CLOSE/ROLL with terminal outcome
        terminal_count = policy_stats.get("meaningful_terminal", 0) or 0
        # meaningful_realized: OPEN/ADD/CLOSE/ROLL that user actually executed
        realized_count = policy_stats.get("meaningful_realized", 0) or 0

        # Check against last optimization's counts
        last_terminal = 0
        last_realized = 0
        if OPTIMIZER_LOG.exists():
            try:
                log = json.loads(OPTIMIZER_LOG.read_text())
                if log:
                    last_terminal = log[-1].get("policy_terminal_at_optimize", 0)
                    last_realized = log[-1].get("policy_realized_at_optimize", 0)
            except Exception:
                pass

        new_terminal = terminal_count - last_terminal
        new_realized = realized_count - last_realized

        if new_terminal >= 40:
            return True
        if new_realized >= 25:
            return True
    except Exception:
        pass  # Policy layer may not exist yet

    # Trigger 3: Legacy trade-count threshold (backwards compat)
    cum = get_cumulative_pnl()
    if not cum:
        return False
    closed_count = (cum.get("expired_count") or 0) + (cum.get("closed_count") or 0) + (cum.get("assigned_count") or 0)

    if OPTIMIZER_LOG.exists():
        try:
            log = json.loads(OPTIMIZER_LOG.read_text())
            if log:
                last_analyzed = log[-1].get("trades_analyzed", 0)
                return (closed_count - last_analyzed) >= TRADES_BEFORE_OPTIMIZE
        except Exception:
            pass

    return closed_count >= TRADES_BEFORE_OPTIMIZE
