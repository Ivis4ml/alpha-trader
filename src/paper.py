"""Alpaca paper trading integration for order validation.

Lets users validate covered call ideas against Alpaca's paper trading
environment before executing on Robinhood.  Uses the Alpaca REST API
directly via ``requests`` — no extra SDK dependency required.
"""

from __future__ import annotations

import os
import sys
from typing import Any

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"

SETUP_INSTRUCTIONS = """\
Alpaca credentials not found.

To set up Alpaca paper trading:
  1. Sign up at https://alpaca.markets (free paper account)
  2. Go to Paper Trading → API Keys → Generate
  3. Add to your .env file:

     ALPACA_API_KEY=your_key_here
     ALPACA_SECRET_KEY=your_secret_here

Then re-run this command.
"""


class AlpacaError(Exception):
    """Raised when an Alpaca API call fails."""


class PaperTrader:
    """Thin wrapper around the Alpaca paper-trading REST API."""

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper: bool = True,
    ) -> None:
        if requests is None:
            print("The 'requests' package is required.  pip install requests", file=sys.stderr)
            raise SystemExit(1)

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY", "")

        if not self.api_key or not self.secret_key:
            print(SETUP_INSTRUCTIONS, file=sys.stderr)
            raise SystemExit(1)

        self.base = PAPER_BASE if paper else "https://api.alpaca.markets"
        self.data_base = DATA_BASE
        self._session = requests.Session()
        self._session.headers.update({
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
            "Accept": "application/json",
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        resp = self._session.request(method, url, **kwargs)
        if not resp.ok:
            detail = resp.text[:500]
            raise AlpacaError(f"Alpaca {method} {url} → {resp.status_code}: {detail}")
        if resp.status_code == 204:
            return None
        return resp.json()

    def _trading(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, f"{self.base}{path}", **kwargs)

    def _data(self, method: str, path: str, **kwargs: Any) -> Any:
        return self._request(method, f"{self.data_base}{path}", **kwargs)

    # ------------------------------------------------------------------
    # Account & Positions
    # ------------------------------------------------------------------
    def get_account(self) -> dict:
        """Return account balance, buying power, etc."""
        acct = self._trading("GET", "/v2/account")
        return {
            "status": acct.get("status"),
            "currency": acct.get("currency"),
            "cash": float(acct.get("cash", 0)),
            "portfolio_value": float(acct.get("portfolio_value", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "equity": float(acct.get("equity", 0)),
            "long_market_value": float(acct.get("long_market_value", 0)),
            "short_market_value": float(acct.get("short_market_value", 0)),
        }

    def get_positions(self) -> list[dict]:
        """Return current paper positions (stocks + options)."""
        raw = self._trading("GET", "/v2/positions")
        positions = []
        for p in raw:
            positions.append({
                "symbol": p.get("symbol"),
                "qty": float(p.get("qty", 0)),
                "side": p.get("side"),
                "market_value": float(p.get("market_value", 0)),
                "avg_entry_price": float(p.get("avg_entry_price", 0)),
                "current_price": float(p.get("current_price", 0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0)),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
                "asset_class": p.get("asset_class", ""),
            })
        return positions

    # ------------------------------------------------------------------
    # Option chain
    # ------------------------------------------------------------------
    def get_option_chain(self, symbol: str, expiry: str) -> list[dict]:
        """Fetch option chain with Greeks from Alpaca market data.

        Parameters
        ----------
        symbol : str
            Underlying ticker, e.g. ``"AMZN"``.
        expiry : str
            Expiration date ``"YYYY-MM-DD"``.

        Returns a list of dicts, one per contract, with Greeks when available.
        """
        params = {
            "underlying_symbols": symbol,
            "expiration_date": expiry,
            "type": "call",
            "limit": 100,
        }
        # Alpaca options snapshot endpoint
        data = self._data("GET", "/v1beta1/options/snapshots", params=params)
        chain: list[dict] = []
        snapshots = data.get("snapshots", {})
        for osi_symbol, snap in snapshots.items():
            greeks = snap.get("greeks", {}) or {}
            latest = snap.get("latestQuote", {}) or {}
            trade = snap.get("latestTrade", {}) or {}
            chain.append({
                "osi_symbol": osi_symbol,
                "bid": float(latest.get("bp", 0)),
                "ask": float(latest.get("ap", 0)),
                "last": float(trade.get("p", 0)),
                "volume": int(trade.get("s", 0)),
                "delta": float(greeks.get("delta", 0)),
                "gamma": float(greeks.get("gamma", 0)),
                "theta": float(greeks.get("theta", 0)),
                "vega": float(greeks.get("vega", 0)),
                "iv": float(greeks.get("implied_volatility", 0)),
            })
        return sorted(chain, key=lambda c: c.get("delta", 0), reverse=True)

    # ------------------------------------------------------------------
    # Option symbol helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _build_osi_symbol(symbol: str, expiry: str, strike: float, right: str = "C") -> str:
        """Build an OSI option symbol, e.g. ``AMZN260417C00225000``.

        Parameters
        ----------
        symbol : str  — underlying ticker (up to 6 chars, left-padded with spaces in OSI but
                        Alpaca accepts without padding).
        expiry : str  — ``"YYYY-MM-DD"``
        strike : float
        right  : str  — ``"C"`` or ``"P"``
        """
        # YYMMDD
        parts = expiry.split("-")
        date_part = parts[0][2:] + parts[1] + parts[2]
        # Strike is multiplied by 1000, zero-padded to 8 digits
        strike_int = int(round(strike * 1000))
        strike_part = f"{strike_int:08d}"
        return f"{symbol}{date_part}{right}{strike_part}"

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------
    def submit_covered_call(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        contracts: int,
        limit_price: float,
    ) -> dict:
        """Submit a sell-to-open call order (covered call).

        Returns the order dict from Alpaca.
        """
        osi = self._build_osi_symbol(symbol, expiry, strike, "C")
        order = {
            "symbol": osi,
            "qty": str(contracts),
            "side": "sell",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": str(limit_price),
        }
        result = self._trading("POST", "/v2/orders", json=order)
        return {
            "id": result.get("id"),
            "client_order_id": result.get("client_order_id"),
            "symbol": result.get("symbol"),
            "side": result.get("side"),
            "qty": result.get("qty"),
            "type": result.get("type"),
            "limit_price": result.get("limit_price"),
            "status": result.get("status"),
            "created_at": result.get("created_at"),
        }

    def close_option(
        self,
        symbol: str,
        strike: float,
        expiry: str,
        contracts: int,
        limit_price: float,
    ) -> dict:
        """Submit a buy-to-close call order."""
        osi = self._build_osi_symbol(symbol, expiry, strike, "C")
        order = {
            "symbol": osi,
            "qty": str(contracts),
            "side": "buy",
            "type": "limit",
            "time_in_force": "day",
            "limit_price": str(limit_price),
        }
        result = self._trading("POST", "/v2/orders", json=order)
        return {
            "id": result.get("id"),
            "client_order_id": result.get("client_order_id"),
            "symbol": result.get("symbol"),
            "side": result.get("side"),
            "qty": result.get("qty"),
            "type": result.get("type"),
            "limit_price": result.get("limit_price"),
            "status": result.get("status"),
            "created_at": result.get("created_at"),
        }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def get_orders(self, status: str = "open") -> list[dict]:
        """List orders filtered by status (open, closed, all)."""
        params = {"status": status, "limit": 50}
        raw = self._trading("GET", "/v2/orders", params=params)
        orders: list[dict] = []
        for o in raw:
            orders.append({
                "id": o.get("id"),
                "symbol": o.get("symbol"),
                "side": o.get("side"),
                "qty": o.get("qty"),
                "type": o.get("type"),
                "limit_price": o.get("limit_price"),
                "status": o.get("status"),
                "filled_qty": o.get("filled_qty"),
                "filled_avg_price": o.get("filled_avg_price"),
                "created_at": o.get("created_at"),
            })
        return orders

    def cancel_order(self, order_id: str) -> None:
        """Cancel a specific order by ID."""
        self._trading("DELETE", f"/v2/orders/{order_id}")
