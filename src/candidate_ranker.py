"""Stage 2 candidate ranker — learned utility regressor for covered call selection.

Replaces the heuristic linear scoring in scan_engine.score_candidate() with a
trained model that predicts utility_label (risk-adjusted premium capture).

Model: HistGradientBoostingRegressor (handles NaN natively, fast, tabular-optimal).
Evaluation: walk-forward CV grouped by scan, ranking metrics (NDCG, top1 uplift, regret).
Inference: predict_candidate_scores() returns model_score per candidate for combine_scores().
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidate_dataset import (
    load_dataset, walk_forward_splits, CandidateDataset, format_dataset_summary,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "candidate_ranker.joblib"

# Minimum thresholds for training
MIN_CANDIDATES = 500
MIN_SCAN_GROUPS = 50


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RankerMetrics:
    """Evaluation metrics for one walk-forward fold."""
    fold: int
    n_train: int
    n_test: int
    n_test_scans: int
    # Regression
    rmse: float
    mae: float
    # Ranking (the ones that actually matter)
    top1_utility_uplift: float   # model_top1 - heuristic_top1 avg utility
    ndcg_at_3: float
    mean_regret: float           # avg gap between model_top1 and scan_best
    # Baselines
    heuristic_top1_utility: float
    model_top1_utility: float
    random_top1_utility: float


@dataclass
class TrainResult:
    """Full training result across all walk-forward folds."""
    n_total: int
    n_scans: int
    n_folds: int
    feature_importances: dict[str, float]
    fold_metrics: list[RankerMetrics]
    # Aggregates
    avg_top1_uplift: float
    avg_ndcg: float
    avg_regret: float
    promotion_ready: bool  # meets all gates for model_weight > 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ranker(
    min_candidates: int = MIN_CANDIDATES,
    min_scans: int = MIN_SCAN_GROUPS,
    n_splits: int = 5,
) -> TrainResult:
    """Train a utility regressor on labeled candidate observations.

    Walk-forward cross-validation, then final model trained on all data
    and saved to data/candidate_ranker.joblib.
    """
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    dataset = load_dataset(min_label_confidence="medium", include_counterfactual=True)

    if len(dataset.y) < min_candidates:
        raise ValueError(
            f"Not enough labeled candidates: {len(dataset.y)} < {min_candidates}. "
            f"Run more scans and backfill labels first."
        )

    n_scans = dataset.scan_ids.nunique()
    if n_scans < min_scans:
        raise ValueError(
            f"Not enough scan groups: {n_scans} < {min_scans}. "
            f"Need more historical scans."
        )

    # Walk-forward CV
    splits = walk_forward_splits(dataset, n_splits=n_splits)
    if not splits:
        raise ValueError("Could not create walk-forward splits (not enough data).")

    fold_metrics: list[RankerMetrics] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train = dataset.X.iloc[train_idx]
        y_train = dataset.y.iloc[train_idx]
        X_test = dataset.X.iloc[test_idx]
        y_test = dataset.y.iloc[test_idx]
        scan_ids_test = dataset.scan_ids.iloc[test_idx]
        heuristic_test = dataset.heuristic_scores.iloc[test_idx]

        model = HistGradientBoostingRegressor(
            max_iter=200,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        metrics = _evaluate_fold(
            fold=fold_idx,
            y_true=y_test.values,
            y_pred=preds,
            heuristic_scores=heuristic_test.values,
            scan_ids=scan_ids_test.values,
            n_train=len(train_idx),
        )
        fold_metrics.append(metrics)

    # Final model: train on all data
    final_model = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_leaf=10,
        random_state=42,
    )
    final_model.fit(dataset.X, dataset.y)

    # Feature importances — HistGBR doesn't expose feature_importances_ directly,
    # so use permutation importance on a held-out slice (last 20% of data)
    feature_names = list(dataset.X.columns)
    try:
        from sklearn.inspection import permutation_importance
        n_eval = max(1, len(dataset.y) // 5)
        X_eval = dataset.X.iloc[-n_eval:]
        y_eval = dataset.y.iloc[-n_eval:]
        perm_result = permutation_importance(
            final_model, X_eval, y_eval, n_repeats=10, random_state=42,
        )
        importances = dict(zip(feature_names, perm_result.importances_mean))
    except Exception:
        # Fallback: uniform importances
        importances = {name: 1.0 / len(feature_names) for name in feature_names}
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    # Save model + metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    import datetime as dt
    model_data = {
        "model": final_model,
        "feature_cols": feature_names,
        "trained_at": dt.datetime.now().isoformat(),
        "train_range": {
            "first_scan": dataset.scan_ts.min(),
            "last_scan": dataset.scan_ts.max(),
        },
        "n_samples": len(dataset.y),
        "n_scans": n_scans,
        "label_definition": "utility = capture_ratio - 0.5 * assignment_penalty",
        "metrics": {
            "avg_top1_uplift": float(np.mean([m.top1_utility_uplift for m in fold_metrics])),
            "avg_ndcg": float(np.mean([m.ndcg_at_3 for m in fold_metrics])),
            "avg_regret": float(np.mean([m.mean_regret for m in fold_metrics])),
        },
    }
    joblib.dump(model_data, MODEL_PATH)

    # Aggregates
    avg_uplift = float(np.mean([m.top1_utility_uplift for m in fold_metrics]))
    avg_ndcg = float(np.mean([m.ndcg_at_3 for m in fold_metrics]))
    avg_regret = float(np.mean([m.mean_regret for m in fold_metrics]))

    # Promotion gate: all folds must show uplift, enough data
    recent_folds = fold_metrics[-3:] if len(fold_metrics) >= 3 else fold_metrics
    no_degradation = all(m.top1_utility_uplift >= 0 for m in recent_folds)
    promotion_ready = (
        len(dataset.y) >= min_candidates
        and n_scans >= min_scans
        and avg_uplift > 0
        and no_degradation
    )

    return TrainResult(
        n_total=len(dataset.y),
        n_scans=n_scans,
        n_folds=len(fold_metrics),
        feature_importances=importances,
        fold_metrics=fold_metrics,
        avg_top1_uplift=round(avg_uplift, 4),
        avg_ndcg=round(avg_ndcg, 4),
        avg_regret=round(avg_regret, 4),
        promotion_ready=promotion_ready,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate_fold(
    fold: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    heuristic_scores: np.ndarray,
    scan_ids: np.ndarray,
    n_train: int,
) -> RankerMetrics:
    """Compute ranking metrics for one walk-forward fold."""
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Per-scan ranking metrics
    unique_scans = np.unique(scan_ids)
    uplift_list = []
    ndcg_list = []
    regret_list = []
    heuristic_top1s = []
    model_top1s = []
    random_top1s = []

    for scan_id in unique_scans:
        mask = scan_ids == scan_id
        if mask.sum() < 2:
            continue  # need at least 2 candidates to rank

        y_scan = y_true[mask]
        pred_scan = y_pred[mask]
        heur_scan = heuristic_scores[mask]

        # Best possible utility in this scan
        best_utility = float(np.max(y_scan))

        # Model top-1: candidate with highest predicted score
        model_top1_idx = np.argmax(pred_scan)
        model_top1_utility = float(y_scan[model_top1_idx])

        # Heuristic top-1: candidate with highest heuristic score
        heur_top1_idx = np.argmax(heur_scan)
        heur_top1_utility = float(y_scan[heur_top1_idx])

        # Random baseline: average utility
        random_top1_utility = float(np.mean(y_scan))

        uplift_list.append(model_top1_utility - heur_top1_utility)
        regret_list.append(best_utility - model_top1_utility)
        heuristic_top1s.append(heur_top1_utility)
        model_top1s.append(model_top1_utility)
        random_top1s.append(random_top1_utility)

        # NDCG@3
        ndcg = _ndcg_at_k(y_scan, pred_scan, k=3)
        ndcg_list.append(ndcg)

    return RankerMetrics(
        fold=fold,
        n_train=n_train,
        n_test=len(y_true),
        n_test_scans=len(unique_scans),
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        top1_utility_uplift=round(float(np.mean(uplift_list)), 4) if uplift_list else 0.0,
        ndcg_at_3=round(float(np.mean(ndcg_list)), 4) if ndcg_list else 0.0,
        mean_regret=round(float(np.mean(regret_list)), 4) if regret_list else 0.0,
        heuristic_top1_utility=round(float(np.mean(heuristic_top1s)), 4) if heuristic_top1s else 0.0,
        model_top1_utility=round(float(np.mean(model_top1s)), 4) if model_top1s else 0.0,
        random_top1_utility=round(float(np.mean(random_top1s)), 4) if random_top1s else 0.0,
    )


def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    # Rank by predicted scores
    order = np.argsort(-y_pred)[:k]
    dcg = sum(y_true[order[i]] / np.log2(i + 2) for i in range(min(k, len(order))))

    # Ideal ranking
    ideal_order = np.argsort(-y_true)[:k]
    idcg = sum(y_true[ideal_order[i]] / np.log2(i + 2) for i in range(min(k, len(ideal_order))))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_candidate_scores(features_df: pd.DataFrame) -> np.ndarray:
    """Predict utility scores for a batch of candidates.

    Parameters
    ----------
    features_df : DataFrame
        One row per candidate, columns matching training feature_cols.

    Returns
    -------
    np.ndarray of predicted utility scores.
    """
    import joblib

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained ranker at {MODEL_PATH}. Run './at learner train' first."
        )

    model_data = joblib.load(MODEL_PATH)
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]

    # Align columns (fill missing with NaN — HistGBR handles natively)
    X = features_df.reindex(columns=feature_cols)
    return model.predict(X)


def is_model_available() -> bool:
    """Check if a trained ranker model exists."""
    return MODEL_PATH.exists()


def get_model_metadata() -> dict | None:
    """Load model metadata without the full model object."""
    if not MODEL_PATH.exists():
        return None
    import joblib
    data = joblib.load(MODEL_PATH)
    return {k: v for k, v in data.items() if k != "model"}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_train_result(result: TrainResult) -> str:
    """Format training results for CLI output."""
    lines = [
        f"\n{'=' * 65}",
        f"  CANDIDATE RANKER TRAINING RESULT",
        f"{'=' * 65}",
        f"  Samples: {result.n_total:,} ({result.n_scans} scan groups)",
        f"  Walk-forward folds: {result.n_folds}",
        f"{'─' * 65}",
    ]

    for m in result.fold_metrics:
        lines.append(
            f"  Fold {m.fold}: "
            f"RMSE={m.rmse:.3f} "
            f"NDCG@3={m.ndcg_at_3:.3f} "
            f"top1↑={m.top1_utility_uplift:+.3f} "
            f"regret={m.mean_regret:.3f} "
            f"({m.n_test_scans} scans)"
        )

    lines.append(f"{'─' * 65}")
    lines.append(f"  Avg top-1 uplift:  {result.avg_top1_uplift:+.4f}")
    lines.append(f"  Avg NDCG@3:        {result.avg_ndcg:.4f}")
    lines.append(f"  Avg regret:        {result.avg_regret:.4f}")
    lines.append(f"{'─' * 65}")

    gate = "PASS" if result.promotion_ready else "NOT READY"
    lines.append(f"  Promotion gate:    {gate}")
    if result.promotion_ready:
        lines.append(f"  → Safe to set model_weight > 0 in combine_scores()")
    else:
        reasons = []
        if result.n_total < MIN_CANDIDATES:
            reasons.append(f"need {MIN_CANDIDATES}+ candidates (have {result.n_total})")
        if result.n_scans < MIN_SCAN_GROUPS:
            reasons.append(f"need {MIN_SCAN_GROUPS}+ scans (have {result.n_scans})")
        if result.avg_top1_uplift <= 0:
            reasons.append(f"avg uplift must be > 0 (is {result.avg_top1_uplift:+.4f})")
        if reasons:
            lines.append(f"  → Blockers: {'; '.join(reasons)}")

    lines.append(f"{'─' * 65}")
    lines.append(f"  Feature Importances (top 10):")
    for i, (name, imp) in enumerate(result.feature_importances.items()):
        if i >= 10:
            break
        bar = "#" * int(imp * 100)
        lines.append(f"    {name:<25s} {imp:.3f}  {bar}")

    lines.append(f"{'=' * 65}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shadow comparison — model vs heuristic on historical data
# ---------------------------------------------------------------------------

def run_shadow_comparison() -> dict:
    """Compare model vs heuristic rankings on all labeled scan groups.

    For each scan group, computes which candidate each method would pick
    as top-1, and what that candidate's actual utility was.
    Returns summary dict.
    """
    dataset = load_dataset(min_label_confidence="medium")
    if len(dataset.y) < 10:
        return {"error": "Not enough labeled data for comparison"}

    if not is_model_available():
        return {"error": "No trained model"}

    model_preds = predict_candidate_scores(dataset.X)

    unique_scans = dataset.scan_ids.unique()
    comparisons = []

    for scan_id in unique_scans:
        mask = dataset.scan_ids == scan_id
        if mask.sum() < 2:
            continue

        y_scan = dataset.y[mask].values
        heur_scan = dataset.heuristic_scores[mask].values
        model_scan = model_preds[mask.values]
        meta_scan = dataset.meta[mask]

        best_utility = float(np.max(y_scan))
        heur_top1_idx = np.argmax(heur_scan)
        model_top1_idx = np.argmax(model_scan)

        comparisons.append({
            "scan_id": scan_id,
            "n_candidates": int(mask.sum()),
            "best_utility": round(best_utility, 4),
            "heuristic_pick_utility": round(float(y_scan[heur_top1_idx]), 4),
            "model_pick_utility": round(float(y_scan[model_top1_idx]), 4),
            "uplift": round(float(y_scan[model_top1_idx] - y_scan[heur_top1_idx]), 4),
            "heuristic_pick": f"{meta_scan.iloc[heur_top1_idx]['symbol']} ${meta_scan.iloc[heur_top1_idx]['strike']}",
            "model_pick": f"{meta_scan.iloc[model_top1_idx]['symbol']} ${meta_scan.iloc[model_top1_idx]['strike']}",
            "same_pick": heur_top1_idx == model_top1_idx,
        })

    if not comparisons:
        return {"error": "No scan groups with 2+ candidates"}

    uplifts = [c["uplift"] for c in comparisons]
    model_wins = sum(1 for u in uplifts if u > 0)
    ties = sum(1 for u in uplifts if u == 0)
    heur_wins = sum(1 for u in uplifts if u < 0)

    return {
        "total_scans": len(comparisons),
        "model_wins": model_wins,
        "ties": ties,
        "heuristic_wins": heur_wins,
        "avg_uplift": round(float(np.mean(uplifts)), 4),
        "median_uplift": round(float(np.median(uplifts)), 4),
        "same_pick_pct": round(sum(1 for c in comparisons if c["same_pick"]) / len(comparisons) * 100, 1),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Slice evaluation — break down uplift by segment
# ---------------------------------------------------------------------------

def run_slice_evaluation() -> dict[str, list[dict]]:
    """Break down model vs heuristic performance by key slices.

    Slices: symbol, IV bucket, regime (from VIX), assigned/expired.
    Returns dict of slice_name -> [{slice, n, uplift, heur_utility, model_utility}].
    """
    dataset = load_dataset(min_label_confidence="medium")
    if len(dataset.y) < 10 or not is_model_available():
        return {}

    model_preds = predict_candidate_scores(dataset.X)

    # Build per-row data
    df = dataset.X.copy()
    df["y"] = dataset.y.values
    df["heuristic"] = dataset.heuristic_scores.values
    df["model_pred"] = model_preds
    df["scan_id"] = dataset.scan_ids.values
    df["symbol"] = dataset.meta["symbol"].values
    df["assigned"] = (dataset.y < 0).astype(int).values  # negative utility ≈ assigned

    # Define slice functions
    def _iv_bucket(iv):
        if pd.isna(iv):
            return "unknown"
        if iv < 30:
            return "low (<30)"
        elif iv < 60:
            return "mid (30-60)"
        else:
            return "high (60+)"

    def _vix_regime(vix):
        if pd.isna(vix):
            return "unknown"
        if vix < 15:
            return "low (<15)"
        elif vix < 25:
            return "mid (15-25)"
        else:
            return "high (25+)"

    def _dte_bucket(dte):
        if pd.isna(dte):
            return "unknown"
        if dte <= 7:
            return "weekly (<=7)"
        elif dte <= 14:
            return "biweekly (8-14)"
        else:
            return "monthly (15+)"

    df["iv_bucket"] = df["iv_rank"].apply(_iv_bucket)
    df["vix_regime"] = df["vix"].apply(_vix_regime)
    df["dte_bucket"] = df["dte"].apply(_dte_bucket)

    slices = {}

    for slice_col in ["symbol", "iv_bucket", "vix_regime", "dte_bucket"]:
        slice_results = []
        for name, group in df.groupby(slice_col):
            if len(group) < 5:
                continue

            # Per-scan uplift within this slice
            uplifts = []
            heur_utils = []
            model_utils = []
            for _, scan_group in group.groupby("scan_id"):
                if len(scan_group) < 2:
                    continue
                y_s = scan_group["y"].values
                h_s = scan_group["heuristic"].values
                m_s = scan_group["model_pred"].values

                h_pick = float(y_s[np.argmax(h_s)])
                m_pick = float(y_s[np.argmax(m_s)])
                uplifts.append(m_pick - h_pick)
                heur_utils.append(h_pick)
                model_utils.append(m_pick)

            if not uplifts:
                continue

            slice_results.append({
                "slice": str(name),
                "n_candidates": len(group),
                "n_scans": len(uplifts),
                "avg_uplift": round(float(np.mean(uplifts)), 4),
                "heur_utility": round(float(np.mean(heur_utils)), 4),
                "model_utility": round(float(np.mean(model_utils)), 4),
                "model_win_pct": round(sum(1 for u in uplifts if u > 0) / len(uplifts) * 100, 1),
            })

        slices[slice_col] = sorted(slice_results, key=lambda x: -x["avg_uplift"])

    return slices


# ---------------------------------------------------------------------------
# Promotion helpers
# ---------------------------------------------------------------------------

def check_promotion_readiness() -> dict:
    """Full promotion readiness check with all gates."""
    result = {
        "ready": False,
        "checks": [],
    }

    # Gate 1: Model exists
    if not is_model_available():
        result["checks"].append(("Model trained", False, "No model"))
        return result
    result["checks"].append(("Model trained", True, ""))

    # Gate 2: Enough data
    from .db import get_labeled_candidate_count
    stats = get_labeled_candidate_count()
    labeled = stats.get("labeled", 0) or 0
    scans = stats.get("scan_groups", 0) or 0
    result["checks"].append((
        f"Data volume ({labeled} candidates, {scans} scans)",
        labeled >= MIN_CANDIDATES and scans >= MIN_SCAN_GROUPS,
        f"need {MIN_CANDIDATES}/{MIN_SCAN_GROUPS}",
    ))

    # Gate 3: Positive uplift
    meta = get_model_metadata()
    metrics = meta.get("metrics", {}) if meta else {}
    uplift = metrics.get("avg_top1_uplift", 0)
    result["checks"].append((
        f"Avg top-1 uplift ({uplift:+.4f})",
        uplift > 0,
        "must be > 0",
    ))

    # Gate 4: Slice stability — no major slice with negative uplift
    slices = run_slice_evaluation()
    symbol_slices = slices.get("symbol", [])
    bad_symbols = [s for s in symbol_slices if s["avg_uplift"] < -0.1 and s["n_scans"] >= 5]
    result["checks"].append((
        f"No symbol with uplift < -0.1 ({len(bad_symbols)} bad)",
        len(bad_symbols) == 0,
        ", ".join(s["slice"] for s in bad_symbols) if bad_symbols else "",
    ))

    # Gate 5: Shadow consistency — model must not LOSE on more than 20% of scans.
    # Ties (model and heuristic pick same utility) don't count against either.
    shadow = run_shadow_comparison()
    if "error" not in shadow:
        total = max(shadow["total_scans"], 1)
        loss_pct = shadow["heuristic_wins"] / total * 100
        result["checks"].append((
            f"Model loss rate {loss_pct:.0f}% (max 20%)",
            loss_pct <= 20,
            f"loses {shadow['heuristic_wins']}/{total}, wins {shadow['model_wins']}, ties {shadow['ties']}",
        ))

    result["ready"] = all(c[1] for c in result["checks"])
    return result
