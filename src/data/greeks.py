"""Black-Scholes Greeks calculation — no scipy dependency."""

import math
from dataclasses import dataclass


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta: float  # per calendar day
    vega: float   # per 1% IV move


def black_scholes_greeks(
    S: float,
    K: float,
    T: float,       # years to expiry
    r: float,       # risk-free rate (e.g. 0.045)
    sigma: float,   # implied volatility (e.g. 0.30)
    option_type: str = "call",
) -> Greeks:
    """Calculate BS Greeks for a European option."""
    if T <= 1e-10 or sigma <= 1e-10:
        intrinsic_delta = 1.0 if (S > K and option_type == "call") else 0.0
        if option_type == "put":
            intrinsic_delta = -1.0 if S < K else 0.0
        return Greeks(delta=intrinsic_delta, gamma=0.0, theta=0.0, vega=0.0)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    nd1 = norm_pdf(d1)

    if option_type == "call":
        delta = norm_cdf(d1)
        theta = (
            -(S * nd1 * sigma) / (2.0 * sqrt_T)
            - r * K * math.exp(-r * T) * norm_cdf(d2)
        ) / 365.0
    else:
        delta = norm_cdf(d1) - 1.0
        theta = (
            -(S * nd1 * sigma) / (2.0 * sqrt_T)
            + r * K * math.exp(-r * T) * norm_cdf(-d2)
        ) / 365.0

    gamma = nd1 / (S * sigma * sqrt_T)
    vega = S * nd1 * sqrt_T / 100.0  # per 1 percentage-point move in IV

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 1e-10 or sigma <= 1e-10:
        return max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-6,
    max_iter: int = 50,
) -> float:
    """Solve for implied volatility using bisection method."""
    if market_price <= 0 or T <= 1e-10:
        return 0.0

    price_fn = bs_call_price if option_type == "call" else bs_put_price

    # Bisection bounds
    lo, hi = 0.01, 5.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        p = price_fn(S, K, T, r, mid)
        if abs(p - market_price) < tol:
            return mid
        if p < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0
