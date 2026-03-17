"""Machine learning signal generation for options trading.

Uses scikit-learn GradientBoosting to predict whether a given week is a
good time to sell covered calls (high probability the call expires OTM).

Features are derived from technical indicators, IV stats, and market context
data already available in the codebase.
"""

from __future__ import annotations

import datetime as dt
import math
import pathlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

from .data.greeks import (
    black_scholes_greeks,
    bs_call_price,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "ml_model.joblib"

_RISK_FREE_RATE = 0.043


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    signal: str           # BUY_SIGNAL / NEUTRAL / CAUTION
    confidence: float     # 0-100%
    top_features: list[dict]   # [{name, value, importance}, ...]
    explanation: str      # human-readable


@dataclass
class TrainResult:
    accuracy: float
    precision: float
    recall: float
    feature_importances: dict[str, float]
    n_samples: int
    n_positive: int
    cv_scores: list[float]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "rsi_14",
    "macd_hist",
    "bb_pct_b",
    "atr_price_ratio",
    "iv_rank",
    "iv_percentile",
    "hv20_hv60_ratio",
    "sma20_distance",
    "sma50_distance",
    "day_of_week",
    "days_to_earnings",
    "vix_level",
    "vix_change_5d",
    "put_call_ratio",
]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI over a rolling window."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))


def _compute_macd_hist(close: pd.Series) -> pd.Series:
    """MACD histogram (12, 26, 9)."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    return macd_line - signal_line


def _compute_bb_pct_b(close: pd.Series, period: int = 20, num_std: int = 2) -> pd.Series:
    """Bollinger %B: (price - lower) / (upper - lower). 0 = at lower, 1 = at upper."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = upper - lower
    return (close - lower) / width.replace(0, 1e-10)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _compute_hv(close: pd.Series, window: int) -> pd.Series:
    """Rolling annualized historical volatility."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * math.sqrt(252)


def _get_vix_history(start: str, end: str) -> pd.DataFrame:
    """Fetch VIX price history and compute 5-day change."""
    vix = yf.Ticker("^VIX")
    hist = vix.history(start=start, end=end, auto_adjust=True)
    if hist.empty:
        return pd.DataFrame()
    df = pd.DataFrame(index=hist.index)
    df["vix_level"] = hist["Close"]
    df["vix_change_5d"] = hist["Close"].pct_change(5) * 100
    return df


def _get_earnings_dates(symbol: str) -> list[dt.date]:
    """Retrieve historical earnings dates for a symbol (best-effort)."""
    t = yf.Ticker(symbol)
    dates: list[dt.date] = []
    try:
        # yfinance exposes earnings_dates as a DataFrame with DatetimeIndex
        ed = t.earnings_dates
        if ed is not None and not ed.empty:
            for ts in ed.index:
                d = ts.date() if hasattr(ts, "date") else ts
                dates.append(d)
    except Exception:
        pass

    # Fallback: quarterly calendar from t.calendar
    if not dates:
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                for d in cal.get("Earnings Date", []):
                    if hasattr(d, "date"):
                        dates.append(d.date())
        except Exception:
            pass

    return sorted(set(dates))


def _days_to_nearest_earnings(date: dt.date, earnings_dates: list[dt.date]) -> float:
    """Days to the nearest (past or future) earnings date. Returns 999 if unknown."""
    if not earnings_dates:
        return 999.0
    min_dist = min(abs((d - date).days) for d in earnings_dates)
    return float(min_dist)


def build_features(symbol: str, lookback_months: int = 24) -> pd.DataFrame:
    """Build a feature DataFrame for a symbol over the lookback window.

    Each row corresponds to one trading day. For training, rows are then
    aligned with weekly labels (typically using Friday dates).

    Returns a DataFrame indexed by date with columns matching FEATURE_COLS.
    """
    end_date = dt.date.today()
    # Extra buffer for indicator warm-up (60-day warmup for MACD/HV60)
    start_date = end_date - dt.timedelta(days=lookback_months * 30 + 90)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat(), auto_adjust=True)

    if hist.empty or len(hist) < 70:
        raise ValueError(f"Insufficient price history for {symbol} ({len(hist)} rows)")

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]

    df = pd.DataFrame(index=hist.index)

    # Technical indicators
    df["rsi_14"] = _compute_rsi(close, 14)
    df["macd_hist"] = _compute_macd_hist(close)
    df["bb_pct_b"] = _compute_bb_pct_b(close)

    atr = _compute_atr(high, low, close, 14)
    df["atr_price_ratio"] = atr / close

    # Volatility
    hv20 = _compute_hv(close, 20)
    hv60 = _compute_hv(close, 60)
    df["hv20_hv60_ratio"] = hv20 / hv60.replace(0, 1e-10)

    # IV rank and percentile (rolling from HV as proxy, same as fetcher.py)
    rolling_vol = hv20.copy()
    expanding_min = rolling_vol.expanding(min_periods=20).min()
    expanding_max = rolling_vol.expanding(min_periods=20).max()
    iv_range = expanding_max - expanding_min
    df["iv_rank"] = ((rolling_vol - expanding_min) / iv_range.replace(0, 1e-10)) * 100
    df["iv_percentile"] = rolling_vol.expanding(min_periods=20).apply(
        lambda x: (x[:-1] < x.iloc[-1]).mean() * 100 if len(x) > 1 else 50.0, raw=False
    )

    # SMA distances
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    df["sma20_distance"] = (close / sma20 - 1) * 100
    df["sma50_distance"] = (close / sma50 - 1) * 100

    # Day of week (0=Monday, 4=Friday)
    df["day_of_week"] = df.index.map(
        lambda ts: ts.weekday() if hasattr(ts, "weekday") else ts.date().weekday()
    )

    # Days to nearest earnings
    earnings_dates = _get_earnings_dates(symbol)
    df["days_to_earnings"] = df.index.map(
        lambda ts: _days_to_nearest_earnings(
            ts.date() if hasattr(ts, "date") else ts,
            earnings_dates,
        )
    )

    # VIX data
    vix_df = _get_vix_history(start_date.isoformat(), end_date.isoformat())
    if not vix_df.empty:
        # Align VIX to our index by date
        vix_df.index = vix_df.index.normalize()
        df.index = df.index.normalize()
        df["vix_level"] = vix_df["vix_level"].reindex(df.index, method="ffill")
        df["vix_change_5d"] = vix_df["vix_change_5d"].reindex(df.index, method="ffill")
    else:
        df["vix_level"] = np.nan
        df["vix_change_5d"] = np.nan

    # Put/call ratio — only available for current day via yfinance;
    # for historical data we fill NaN and let the model handle it.
    df["put_call_ratio"] = np.nan

    # Trim warm-up period (first ~65 rows will have NaNs from indicators)
    actual_start = end_date - dt.timedelta(days=lookback_months * 30)
    df = df.loc[df.index >= pd.Timestamp(actual_start)]

    # Ensure column order
    df = df[FEATURE_COLS]

    return df


def build_features_current(symbol: str) -> pd.Series:
    """Build features for the current (latest) day. Returns a single-row Series."""
    df = build_features(symbol, lookback_months=3)
    if df.empty:
        raise ValueError(f"No feature data for {symbol}")

    latest = df.iloc[-1].copy()

    # Try to fill put/call ratio from live data
    try:
        from .data.enhanced import fetch_put_call_ratio
        pc = fetch_put_call_ratio(symbol)
        if pc is not None:
            latest["put_call_ratio"] = pc
    except Exception:
        pass

    return latest


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def build_labels(
    symbol: str,
    delta_target: float = 0.20,
    dte: int = 7,
    lookback_months: int = 24,
) -> pd.Series:
    """Generate weekly labels: 1 = call expired OTM (good), 0 = assigned (bad).

    For each Friday in the lookback window, simulate selling a call at
    *delta_target* and check whether the stock stayed below the strike
    by the next Friday.

    Returns a Series indexed by sell-date (Friday) with values 0 or 1.
    """
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=lookback_months * 30 + 60)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.isoformat(), end=end_date.isoformat(), auto_adjust=True)

    if hist.empty or len(hist) < 60:
        raise ValueError(f"Insufficient history for label generation: {symbol}")

    close = hist["Close"]
    prices_arr = close.values.astype(float)
    dates_arr = [ts.date() if hasattr(ts, "date") else ts for ts in close.index]
    date_price = dict(zip(dates_arr, prices_arr))
    sorted_dates = sorted(date_price.keys())

    # Find Fridays in the actual lookback window
    actual_start = end_date - dt.timedelta(days=lookback_months * 30)
    fridays = _fridays_in_range(actual_start, end_date)

    labels: dict[pd.Timestamp, int] = {}

    for i in range(len(fridays) - 1):
        sell_day = fridays[i]
        # Determine expiry based on dte
        expiry_day = sell_day + dt.timedelta(days=dte)

        sell_price = _nearest_trading_price(date_price, sell_day, sorted_dates)
        expiry_price = _nearest_trading_price(date_price, expiry_day, sorted_dates)

        if sell_price is None or expiry_price is None:
            continue

        # Historical vol for BS pricing
        idx = _find_date_index(sorted_dates, sell_day)
        if idx is None or idx < 25:
            continue
        window = min(20, idx)
        recent_prices = prices_arr[idx - window:idx + 1]
        if len(recent_prices) < 5:
            continue
        log_ret = np.diff(np.log(recent_prices))
        hv = float(np.std(log_ret, ddof=1) * math.sqrt(252))
        iv_est = max(hv * 1.10, 0.10)  # floor at 10%

        T = dte / 365.0

        # Find strike for target delta
        strike = _find_strike_for_delta(sell_price, T, iv_est, delta_target)

        # Label: 1 if expired OTM (stock below strike), 0 if assigned
        label = 1 if expiry_price <= strike else 0

        # Index as Timestamp so it aligns with feature DataFrame
        ts = pd.Timestamp(sell_day)
        labels[ts] = label

    return pd.Series(labels, name="label", dtype=int)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    symbols: list[str],
    lookback_months: int = 24,
    delta_target: float = 0.20,
    dte: int = 7,
) -> TrainResult:
    """Train a GradientBoosting classifier on historical features and labels.

    Combines data from all *symbols*, uses TimeSeriesSplit for cross-validation,
    and saves the trained model to data/ml_model.joblib.

    Returns training metrics and feature importances.
    """
    import joblib
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import TimeSeriesSplit

    all_features: list[pd.DataFrame] = []
    all_labels: list[pd.Series] = []

    for sym in symbols:
        print(f"  Building features for {sym}...")
        try:
            feat_df = build_features(sym, lookback_months=lookback_months)
        except ValueError as e:
            print(f"    Skipping {sym} features: {e}")
            continue

        print(f"  Building labels for {sym}...")
        try:
            lbl_series = build_labels(sym, delta_target=delta_target, dte=dte,
                                      lookback_months=lookback_months)
        except ValueError as e:
            print(f"    Skipping {sym} labels: {e}")
            continue

        if lbl_series.empty:
            print(f"    No labels generated for {sym}")
            continue

        # Align: for each label date (Friday), pick the nearest feature row
        # Features are daily; labels are weekly (Fridays).
        feat_df.index = feat_df.index.normalize()
        aligned_rows = []
        aligned_labels = []

        for label_date, label_val in lbl_series.items():
            label_ts = pd.Timestamp(label_date).normalize()
            # Find exact or nearest prior feature row
            mask = feat_df.index <= label_ts
            if mask.any():
                nearest_idx = feat_df.index[mask][-1]
                aligned_rows.append(feat_df.loc[nearest_idx])
                aligned_labels.append(label_val)

        if aligned_rows:
            sym_features = pd.DataFrame(aligned_rows)
            sym_labels = pd.Series(aligned_labels, index=sym_features.index, name="label")
            all_features.append(sym_features)
            all_labels.append(sym_labels)
            print(f"    {sym}: {len(sym_features)} samples, "
                  f"{sym_labels.sum()} positive ({sym_labels.mean()*100:.0f}%)")

    if not all_features:
        raise ValueError("No training data could be generated from any symbol")

    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)

    # Handle missing values: fill NaN with median (robust to outliers)
    feature_medians = X.median()
    X = X.fillna(feature_medians)

    print(f"\n  Total training set: {len(X)} samples, {y.sum()} positive "
          f"({y.mean()*100:.1f}%)")

    # TimeSeriesSplit cross-validation (no lookahead bias)
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    cv_accuracies: list[float] = []
    cv_precisions: list[float] = []
    cv_recalls: list[float] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cv_accuracies.append(accuracy_score(y_test, y_pred))
            cv_precisions.append(precision_score(y_test, y_pred, zero_division=0))
            cv_recalls.append(recall_score(y_test, y_pred, zero_division=0))

    # Final model: train on all data
    model.fit(X, y)

    # Save model + metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "model": model,
        "feature_cols": FEATURE_COLS,
        "feature_medians": feature_medians.to_dict(),
        "trained_at": dt.datetime.now().isoformat(),
        "symbols": symbols,
        "lookback_months": lookback_months,
        "delta_target": delta_target,
        "dte": dte,
        "n_samples": len(X),
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n  Model saved to {MODEL_PATH}")

    # Feature importances
    importances = dict(zip(FEATURE_COLS, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda kv: -kv[1]))

    return TrainResult(
        accuracy=round(float(np.mean(cv_accuracies)), 4),
        precision=round(float(np.mean(cv_precisions)), 4),
        recall=round(float(np.mean(cv_recalls)), 4),
        feature_importances=importances,
        n_samples=len(X),
        n_positive=int(y.sum()),
        cv_scores=[round(a, 4) for a in cv_accuracies],
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _load_model() -> dict:
    """Load the trained model from disk."""
    import joblib
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}. Run 'alpha-trader ml train' first."
        )
    return joblib.load(MODEL_PATH)


def predict_signal(symbol: str) -> PredictionResult:
    """Predict whether now is a good time to sell covered calls on *symbol*.

    Returns a PredictionResult with signal, confidence, top features, and explanation.
    """
    model_data = _load_model()
    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    feature_medians = model_data["feature_medians"]

    # Build current features
    features = build_features_current(symbol)

    # Fill missing with training medians
    for col in feature_cols:
        if pd.isna(features.get(col)):
            features[col] = feature_medians.get(col, 0.0)

    X = pd.DataFrame([features[feature_cols].values], columns=feature_cols)

    # Predict
    proba = model.predict_proba(X)[0]
    # proba[1] = probability of class 1 (good time to sell)
    prob_good = float(proba[1]) * 100

    # Determine signal
    if prob_good >= 65:
        signal = "BUY_SIGNAL"
    elif prob_good >= 45:
        signal = "NEUTRAL"
    else:
        signal = "CAUTION"

    # Feature importances for this prediction (using model importances + current values)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda kv: -kv[1])[:5]
    top_feat_list = [
        {
            "name": name,
            "value": round(float(features[name]), 4),
            "importance": round(float(imp), 4),
        }
        for name, imp in top_features
    ]

    # Generate explanation
    explanation = _generate_explanation(signal, prob_good, features, top_feat_list)

    return PredictionResult(
        signal=signal,
        confidence=round(prob_good, 1),
        top_features=top_feat_list,
        explanation=explanation,
    )


def _generate_explanation(
    signal: str,
    confidence: float,
    features: pd.Series,
    top_features: list[dict],
) -> str:
    """Build a human-readable explanation of the prediction."""
    parts: list[str] = []

    if signal == "BUY_SIGNAL":
        parts.append(f"Model predicts {confidence:.0f}% probability the call expires OTM.")
    elif signal == "NEUTRAL":
        parts.append(f"Mixed signals ({confidence:.0f}% OTM probability).")
    else:
        parts.append(f"Elevated assignment risk ({confidence:.0f}% OTM probability).")

    # Contextual commentary based on feature values
    rsi = features.get("rsi_14", 50)
    if not pd.isna(rsi):
        if rsi > 70:
            parts.append(f"RSI is overbought ({rsi:.0f}), pullback may help calls expire OTM.")
        elif rsi < 30:
            parts.append(f"RSI is oversold ({rsi:.0f}), potential bounce increases assignment risk.")

    iv_rank = features.get("iv_rank", 50)
    if not pd.isna(iv_rank):
        if iv_rank > 70:
            parts.append(f"IV rank is elevated ({iv_rank:.0f}%), premium is rich.")
        elif iv_rank < 25:
            parts.append(f"IV rank is low ({iv_rank:.0f}%), premiums may be thin.")

    vix = features.get("vix_level", 0)
    if not pd.isna(vix) and vix > 0:
        if vix > 25:
            parts.append(f"VIX at {vix:.1f} suggests a volatile environment.")
        elif vix < 14:
            parts.append(f"VIX at {vix:.1f} indicates low market fear.")

    days_earn = features.get("days_to_earnings", 999)
    if not pd.isna(days_earn) and days_earn < 14:
        parts.append(f"Earnings in ~{int(days_earn)} days -- watch for IV crush or gap risk.")

    # Top drivers
    drivers = ", ".join(f["name"] for f in top_features[:3])
    parts.append(f"Top drivers: {drivers}.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# CLI formatting helpers
# ---------------------------------------------------------------------------

def format_train_result(result: TrainResult) -> str:
    """Format training results for CLI output."""
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append("  ML MODEL TRAINING RESULTS")
    lines.append(f"{'=' * 60}")
    lines.append(f"  Samples:          {result.n_samples} ({result.n_positive} positive, "
                 f"{result.n_samples - result.n_positive} negative)")
    lines.append(f"  CV Accuracy:      {result.accuracy:.1%}")
    lines.append(f"  CV Precision:     {result.precision:.1%}")
    lines.append(f"  CV Recall:        {result.recall:.1%}")
    lines.append(f"  CV Fold Scores:   {', '.join(f'{s:.1%}' for s in result.cv_scores)}")
    lines.append(f"{'─' * 60}")
    lines.append("  Feature Importances:")
    for name, imp in result.feature_importances.items():
        bar = "#" * int(imp * 100)
        lines.append(f"    {name:<22s} {imp:.3f}  {bar}")
    lines.append(f"{'=' * 60}\n")
    return "\n".join(lines)


def format_prediction(symbol: str, result: PredictionResult) -> str:
    """Format a prediction for CLI output."""
    icon = {"BUY_SIGNAL": "[+]", "NEUTRAL": "[=]", "CAUTION": "[!]"}.get(result.signal, "[ ]")
    lines = []
    lines.append(f"  {icon} {symbol:<6s}  {result.signal:<12s}  confidence: {result.confidence:.0f}%")
    for f in result.top_features[:3]:
        lines.append(f"      {f['name']:<22s} = {f['value']:>8.3f}  (importance: {f['importance']:.3f})")
    lines.append(f"      >> {result.explanation}")
    return "\n".join(lines)


def format_features(symbol: str, features: pd.Series) -> str:
    """Format current features for CLI output."""
    lines = []
    lines.append(f"\n  Features for {symbol}:")
    lines.append(f"  {'─' * 45}")
    for col in FEATURE_COLS:
        val = features.get(col, float("nan"))
        if pd.isna(val):
            lines.append(f"    {col:<22s}  N/A")
        else:
            lines.append(f"    {col:<22s}  {val:>10.4f}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers (shared with backtest logic)
# ---------------------------------------------------------------------------

def _fridays_in_range(start: dt.date, end: dt.date) -> list[dt.date]:
    """Return all Fridays between start and end (inclusive)."""
    d = start
    while d.weekday() != 4:
        d += dt.timedelta(days=1)
    fridays = []
    while d <= end:
        fridays.append(d)
        d += dt.timedelta(days=7)
    return fridays


def _nearest_trading_price(
    date_price: dict[dt.date, float],
    target: dt.date,
    sorted_dates: list[dt.date],
) -> float | None:
    """Get price on target date or nearest trading day (up to 5 days in either direction)."""
    for offset in range(6):
        d = target - dt.timedelta(days=offset)
        if d in date_price:
            return date_price[d]
    for offset in range(1, 4):
        d = target + dt.timedelta(days=offset)
        if d in date_price:
            return date_price[d]
    return None


def _find_date_index(sorted_dates: list[dt.date], target: dt.date) -> int | None:
    """Find index of target (or nearest prior date) in sorted_dates."""
    best = None
    for i, d in enumerate(sorted_dates):
        if d <= target:
            best = i
        else:
            break
    return best


def _find_strike_for_delta(
    S: float,
    T: float,
    sigma: float,
    delta_target: float,
    r: float = _RISK_FREE_RATE,
) -> float:
    """Binary-search for a call strike yielding approximately *delta_target*."""
    lo, hi = S * 0.90, S * 1.30
    for _ in range(80):
        mid = (lo + hi) / 2.0
        g = black_scholes_greeks(S, mid, T, r, sigma, option_type="call")
        if g.delta > delta_target:
            lo = mid
        else:
            hi = mid
    raw_strike = (lo + hi) / 2.0
    return round(raw_strike * 2) / 2
