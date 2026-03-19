"""Load config.yaml (strategy only) and portfolio.yaml (positions + state).

config.yaml  — strategy params, regimes, schedule, language (shared/template)
portfolio.yaml — positions, short calls, targets, cash (per-user, single-account)

The helpers (get_symbols, get_position, …) always read the repo-local
portfolio.yaml.  ``--config`` only swaps strategy parameters; it does NOT
isolate portfolio or trade data.  If multi-account is needed in the future,
design a dedicated ``--account-dir`` that owns config + portfolio + db together.
"""

from __future__ import annotations

import pathlib
import yaml

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"
PORTFOLIO_PATH = pathlib.Path(__file__).resolve().parent.parent / "portfolio.yaml"


def load_config(path: pathlib.Path | str | None = None) -> dict:
    p = pathlib.Path(path) if path else CONFIG_PATH
    with open(p) as f:
        config = yaml.safe_load(f)

    # Unify income_goal → strategy.weekly_target so all code reads one place
    goal = config.get("income_goal", {})
    if goal:
        strat = config.setdefault("strategy", {})
        strat.setdefault("weekly_target", goal.get("weekly_target", 1500))
        strat["annual_target"] = goal.get("annual_target", strat.get("weekly_target", 1500) * 52)
        strat["catchup_cap_pct"] = goal.get("catchup_cap_pct", 200)
        strat["review_interval"] = goal.get("review_interval", "biweekly")
        strat["evaluation"] = goal.get("evaluation", "rolling_average")

    return config


def _load_portfolio() -> dict:
    """Load portfolio.yaml (single-account). Falls back to config.yaml positions."""
    if PORTFOLIO_PATH.exists():
        with open(PORTFOLIO_PATH) as f:
            return yaml.safe_load(f) or {}
    # Fallback: synthesise from config.yaml positions (pre-migration)
    config = load_config()
    return {
        "positions": config.get("positions", {}),
        "short_calls": config.get("short_calls", []) or [],
        "weekly_target": config.get("strategy", {}).get("weekly_target", 1500),
    }


def get_symbols(config: dict) -> list[str]:
    """Get symbols from portfolio.yaml. ``config`` is accepted for API compat only."""
    pf = _load_portfolio()
    return list(pf.get("positions", {}).keys())


def get_position(config: dict, symbol: str) -> dict:
    pf = _load_portfolio()
    return pf.get("positions", {}).get(symbol, {})


def get_short_calls(config: dict) -> list[dict]:
    pf = _load_portfolio()
    return pf.get("short_calls", []) or []


def contracts_available(config: dict, symbol: str) -> int:
    pf = _load_portfolio()
    pos = pf.get("positions", {}).get(symbol, {})
    shares = pos.get("shares", 0)
    max_pct = config.get("strategy", {}).get("max_contracts_pct", 75)
    max_contracts = int(shares / 100 * max_pct / 100)
    existing = sum(
        sc.get("contracts", 0)
        for sc in (pf.get("short_calls", []) or [])
        if sc.get("symbol") == symbol
    )
    return max(max_contracts - existing, 0)


def get_weekly_target(config: dict) -> float:
    """Return weekly_target, preferring portfolio.yaml over config.yaml."""
    pf = _load_portfolio()
    pt = pf.get("weekly_target")
    if pt:
        return float(pt)
    return float(config.get("strategy", {}).get("weekly_target", 1500))


def get_delta_range(config: dict, regime: str) -> tuple[float, float]:
    regimes = config.get("strategy", {}).get("regimes", {})
    r = regimes.get(regime, regimes.get("balanced", {}))
    lo, hi = r.get("delta_range", [0.15, 0.25])
    return (lo, hi)


LANGUAGES = {
    "en": "English",
    "zh": "Simplified Chinese (简体中文)",
    "zh-tw": "Traditional Chinese (繁體中文)",
    "es": "Spanish (Español)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
}


def get_language(config: dict) -> str:
    return config.get("language", "en")


def set_language(lang_code: str) -> str:
    import yaml
    if lang_code not in LANGUAGES:
        return ""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["language"] = lang_code
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return LANGUAGES[lang_code]
