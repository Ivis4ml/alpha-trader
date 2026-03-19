"""Training dataset extractor for the Stage 2 candidate ranker.

Pulls labeled candidate observations from the database, flattens features
+ market context into tabular rows, and provides walk-forward splitting
grouped by scan_id.

Each row = one passed candidate from a historical /scan run.
Label = utility_label (pre-computed in db.py backfill).
"""

from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pandas as pd

from .db import _connect


# Feature columns extracted from CandidateFeatures JSON + market context
FEATURE_COLS = [
    # Option characteristics
    "premium",
    "delta",
    "theta",
    "implied_vol",
    "dte",
    "otm_pct",
    "spread_pct",
    "annualized_yield",
    "open_interest",
    "volume",
    "atr_distance",
    "earnings_gap",
    "iv_rank",
    # Market context
    "vix",
    # Heuristic score components (from score_breakdown)
    "sb_income",
    "sb_assignment_risk",
    "sb_execution_quality",
    "sb_event_risk",
    # Derived
    "bid_ask_ratio",       # bid / ask
    "premium_to_atr",      # premium / atr_distance if available
    "cost_basis_buffer",   # (strike - cost_basis) / stock_price if cost_basis > 0
]


class CandidateDataset(NamedTuple):
    """Training data for the candidate ranker."""
    X: pd.DataFrame          # features, columns = FEATURE_COLS (subset present)
    y: pd.Series             # utility_label
    scan_ids: pd.Series      # scan_id per row, for grouped walk-forward split
    scan_ts: pd.Series       # scan timestamp, for temporal ordering
    heuristic_scores: pd.Series  # original heuristic score, for baseline comparison
    meta: pd.DataFrame       # symbol, expiry, strike, scan_id for debugging


def load_dataset(
    min_label_confidence: str = "medium",
    include_counterfactual: bool = True,
) -> CandidateDataset:
    """Load labeled passed candidates as a flat DataFrame.

    Parameters
    ----------
    min_label_confidence : str
        Minimum confidence to include. "high" = realized only,
        "medium" = realized + counterfactual, "low" = all.
    include_counterfactual : bool
        If False, only include realized (traded) candidates.

    Returns
    -------
    CandidateDataset with X, y, scan_ids, scan_ts, heuristic_scores, meta.
    """
    conn = _connect()
    try:
        confidence_filter = ""
        if min_label_confidence == "high":
            confidence_filter = "AND label_confidence = 'high'"
        elif min_label_confidence == "medium":
            confidence_filter = "AND label_confidence IN ('high', 'medium')"

        source_filter = ""
        if not include_counterfactual:
            source_filter = "AND outcome_source = 'realized'"

        rows = conn.execute(
            f"""SELECT id, scan_id, scan_ts, symbol, expiry, strike, stock_price,
                       features, score, score_breakdown, flags,
                       market_regime, vix, iv_rank,
                       utility_label, label_confidence, outcome_source
                FROM candidate_observations
                WHERE hard_filter_passed = 1
                  AND utility_label IS NOT NULL
                  {confidence_filter}
                  {source_filter}
                ORDER BY scan_ts, scan_id"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        empty = pd.DataFrame()
        return CandidateDataset(
            X=empty, y=pd.Series(dtype=float),
            scan_ids=pd.Series(dtype=str), scan_ts=pd.Series(dtype=str),
            heuristic_scores=pd.Series(dtype=float),
            meta=empty,
        )

    records = []
    for row in rows:
        r = dict(row)
        features = json.loads(r["features"]) if r["features"] else {}
        breakdown = json.loads(r["score_breakdown"]) if r["score_breakdown"] else {}

        record = {
            # Raw features from CandidateFeatures
            "premium": features.get("premium"),
            "delta": features.get("delta"),
            "theta": features.get("theta"),
            "implied_vol": features.get("implied_vol"),
            "dte": features.get("dte"),
            "otm_pct": features.get("otm_pct"),
            "spread_pct": features.get("spread_pct"),
            "annualized_yield": features.get("annualized_yield"),
            "open_interest": features.get("open_interest"),
            "volume": features.get("volume"),
            "atr_distance": features.get("atr_distance"),
            "earnings_gap": features.get("earnings_gap"),
            "iv_rank": features.get("iv_rank", r.get("iv_rank")),
            # Market context
            "vix": r.get("vix"),
            # Score breakdown components
            "sb_income": breakdown.get("income"),
            "sb_assignment_risk": breakdown.get("assignment_risk"),
            "sb_execution_quality": breakdown.get("execution_quality"),
            "sb_event_risk": breakdown.get("event_risk"),
            # Derived features
            "bid_ask_ratio": _safe_div(features.get("bid"), features.get("ask")),
            "premium_to_atr": _safe_div(features.get("premium"),
                                        features.get("atr_distance")),
            "cost_basis_buffer": _cost_basis_buffer(
                features.get("cost_basis"), r["strike"], r["stock_price"]),
            # Label and meta
            "_utility_label": r["utility_label"],
            "_heuristic_score": r["score"],
            "_scan_id": r["scan_id"],
            "_scan_ts": r["scan_ts"],
            "_symbol": r["symbol"],
            "_expiry": r["expiry"],
            "_strike": r["strike"],
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Separate features, label, and meta
    feature_cols = [c for c in df.columns if not c.startswith("_")]
    X = df[feature_cols].copy()
    y = df["_utility_label"].astype(float)
    scan_ids = df["_scan_id"]
    scan_ts = df["_scan_ts"]
    heuristic_scores = df["_heuristic_score"].astype(float)
    meta = df[["_symbol", "_expiry", "_strike", "_scan_id"]].rename(
        columns=lambda c: c.lstrip("_"))

    return CandidateDataset(X=X, y=y, scan_ids=scan_ids, scan_ts=scan_ts,
                            heuristic_scores=heuristic_scores, meta=meta)


# ---------------------------------------------------------------------------
# Walk-forward splitting
# ---------------------------------------------------------------------------

def walk_forward_splits(
    dataset: CandidateDataset,
    n_splits: int = 5,
    min_train_scans: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate walk-forward train/test splits grouped by scan.

    Rules:
    - Same scan_id never split across train and test.
    - Ordered by scan_ts (temporal).
    - Each fold's test set is the next chronological block.

    Returns list of (train_indices, test_indices) arrays.
    """
    # Get unique scans in temporal order
    scan_order = (
        dataset.scan_ts.drop_duplicates()
        .sort_values()
        .index
    )
    unique_scans = dataset.scan_ids.iloc[scan_order].unique()

    if len(unique_scans) < min_train_scans + n_splits:
        # Not enough data — return single train/test split
        split_at = max(min_train_scans, len(unique_scans) * 3 // 4)
        if split_at >= len(unique_scans):
            return []
        train_scans = set(unique_scans[:split_at])
        test_scans = set(unique_scans[split_at:])
        train_idx = dataset.scan_ids[dataset.scan_ids.isin(train_scans)].index.values
        test_idx = dataset.scan_ids[dataset.scan_ids.isin(test_scans)].index.values
        return [(train_idx, test_idx)]

    # Expanding window: train grows, test is next block
    scans_per_fold = max(1, (len(unique_scans) - min_train_scans) // n_splits)
    splits = []

    for i in range(n_splits):
        train_end = min_train_scans + i * scans_per_fold
        test_end = train_end + scans_per_fold

        if test_end > len(unique_scans):
            break

        train_scans = set(unique_scans[:train_end])
        test_scans = set(unique_scans[train_end:test_end])

        train_idx = dataset.scan_ids[dataset.scan_ids.isin(train_scans)].index.values
        test_idx = dataset.scan_ids[dataset.scan_ids.isin(test_scans)].index.values

        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_div(a, b) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _cost_basis_buffer(cost_basis, strike, stock_price) -> float | None:
    if not cost_basis or cost_basis <= 0 or not stock_price or stock_price <= 0:
        return None
    return (strike - cost_basis) / stock_price


def format_dataset_summary(dataset: CandidateDataset) -> str:
    """Format a summary of the dataset for CLI output."""
    if dataset.y.empty:
        return "No labeled candidates available for training."

    n_scans = dataset.scan_ids.nunique()
    lines = [
        f"  Candidates: {len(dataset.y):,}",
        f"  Scan groups: {n_scans}",
        f"  Utility label: mean={dataset.y.mean():.3f} std={dataset.y.std():.3f} "
        f"min={dataset.y.min():.3f} max={dataset.y.max():.3f}",
        f"  Features: {dataset.X.shape[1]} columns, "
        f"{dataset.X.notna().mean().mean():.0%} non-null",
        f"  Heuristic score: mean={dataset.heuristic_scores.mean():.3f}",
    ]
    return "\n".join(lines)
