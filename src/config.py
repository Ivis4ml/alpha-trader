"""Load and validate config.yaml."""

from __future__ import annotations

import pathlib
import yaml

CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: pathlib.Path | str | None = None) -> dict:
    p = pathlib.Path(path) if path else CONFIG_PATH
    with open(p) as f:
        return yaml.safe_load(f)


def get_symbols(config: dict) -> list[str]:
    return list(config.get("positions", {}).keys())


def get_position(config: dict, symbol: str) -> dict:
    return config.get("positions", {}).get(symbol, {})


def get_short_calls(config: dict) -> list[dict]:
    return config.get("short_calls", []) or []


def contracts_available(config: dict, symbol: str) -> int:
    """How many new contracts can be sold (accounting for existing shorts)."""
    pos = get_position(config, symbol)
    shares = pos.get("shares", 0)
    max_pct = config.get("strategy", {}).get("max_contracts_pct", 75)
    max_contracts = int(shares / 100 * max_pct / 100)

    existing = sum(
        sc.get("contracts", 0)
        for sc in get_short_calls(config)
        if sc.get("symbol") == symbol
    )
    return max(max_contracts - existing, 0)


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
    """Set output language in config.yaml. Returns display name."""
    import yaml
    if lang_code not in LANGUAGES:
        return ""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    config["language"] = lang_code
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return LANGUAGES[lang_code]
