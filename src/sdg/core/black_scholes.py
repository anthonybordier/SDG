"""Forward-based Black-Scholes formulas (numpy-vectorized).

All functions use the forward price (no drift term). The discount factor
is applied externally to convert from forward to present value.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def d1(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
) -> np.ndarray:
    """Compute Black-Scholes d1.

    d1 = (-log(K/F) + 0.5 * sigma^2 * T) / (sigma * sqrt(T))
    """
    k = np.log(strikes / forward)
    return (-k + 0.5 * vols**2 * T) / (vols * np.sqrt(T))


def d2(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
) -> np.ndarray:
    """Compute Black-Scholes d2.

    d2 = d1 - sigma * sqrt(T)
    """
    return d1(forward, strikes, vols, T) - vols * np.sqrt(T)


def call_price(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Black-Scholes call price (forward-based).

    C = df * [F * N(d1) - K * N(d2)]
    """
    d1_val = d1(forward, strikes, vols, T)
    d2_val = d1_val - vols * np.sqrt(T)
    return df * (forward * norm.cdf(d1_val) - strikes * norm.cdf(d2_val))


def put_price(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Black-Scholes put price (forward-based).

    P = df * [K * N(-d2) - F * N(-d1)]
    """
    d1_val = d1(forward, strikes, vols, T)
    d2_val = d1_val - vols * np.sqrt(T)
    return df * (strikes * norm.cdf(-d2_val) - forward * norm.cdf(-d1_val))


def digital_call_price(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Cash-or-nothing digital call price.

    DC = df * N(d2)
    """
    d2_val = d2(forward, strikes, vols, T)
    return df * norm.cdf(d2_val)


def digital_put_price(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Cash-or-nothing digital put price.

    DP = df * N(-d2)
    """
    d2_val = d2(forward, strikes, vols, T)
    return df * norm.cdf(-d2_val)


def vega(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Black-Scholes vega (derivative of price w.r.t. volatility).

    Vega = F * df * sqrt(T) * phi(d1)
    """
    d1_val = d1(forward, strikes, vols, T)
    return forward * df * np.sqrt(T) * norm.pdf(d1_val)
