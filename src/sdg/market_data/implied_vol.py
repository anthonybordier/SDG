"""Implied volatility solver using Newton-Raphson with Brent fallback."""

from __future__ import annotations

import enum

import numpy as np
from scipy.optimize import brentq

from sdg.core.black_scholes import call_price, put_price, vega


class OptionType(enum.Enum):
    """Option type for implied vol computation."""

    CALL = "call"
    PUT = "put"


def _brenner_subrahmanyam_guess(
    price: float,
    forward: float,
    T: float,
    df: float,
) -> float:
    """Brenner-Subrahmanyam initial guess for ATM-ish options.

    sigma ~ sqrt(2*pi / T) * price / (df * forward)
    """
    return np.sqrt(2.0 * np.pi / T) * price / (df * forward)


def _bs_price_scalar(
    vol: float,
    forward: float,
    strike: float,
    T: float,
    df: float,
    option_type: OptionType,
) -> float:
    """Scalar BS price for use in Brent solver."""
    strikes = np.array([strike])
    vols = np.array([vol])
    if option_type is OptionType.CALL:
        return float(call_price(forward, strikes, vols, T, df)[0])
    return float(put_price(forward, strikes, vols, T, df)[0])


def implied_vol(
    price: np.ndarray,
    forward: float,
    strikes: np.ndarray,
    T: float,
    df: float,
    option_type: OptionType,
    *,
    max_newton_iter: int = 20,
    newton_tol: float = 1e-10,
    vol_lower: float = 1e-4,
    vol_upper: float = 5.0,
) -> np.ndarray:
    """Compute implied volatilities from option prices.

    Uses Newton-Raphson with Brenner-Subrahmanyam initial guess.
    Falls back to Brent's method if Newton fails to converge.

    Args:
        price: Observed option prices (array).
        forward: Forward price.
        strikes: Strike prices (array, same length as price).
        T: Time to expiry in years.
        df: Discount factor.
        option_type: CALL or PUT.
        max_newton_iter: Maximum Newton-Raphson iterations.
        newton_tol: Convergence tolerance for Newton.
        vol_lower: Lower bound for vol search.
        vol_upper: Upper bound for vol search.

    Returns:
        Array of implied volatilities. NaN where IV cannot be determined
        (negative price, below intrinsic, above upper bound).
    """
    price = np.asarray(price, dtype=float)
    strikes = np.asarray(strikes, dtype=float)
    n = len(price)
    result = np.full(n, np.nan)

    for i in range(n):
        p = price[i]
        K = strikes[i]

        if np.isnan(p) or p <= 0.0:
            continue

        # Check intrinsic bounds
        if option_type is OptionType.CALL:
            intrinsic = df * max(forward - K, 0.0)
            upper_bound = df * forward
        else:
            intrinsic = df * max(K - forward, 0.0)
            upper_bound = df * K

        if p < intrinsic - 1e-12 or p > upper_bound + 1e-12:
            continue

        # Newton-Raphson
        sigma = _brenner_subrahmanyam_guess(p, forward, T, df)
        sigma = np.clip(sigma, vol_lower, vol_upper)
        converged = False

        for _ in range(max_newton_iter):
            bs_p = _bs_price_scalar(sigma, forward, K, T, df, option_type)
            v = float(vega(forward, np.array([K]), np.array([sigma]), T, df)[0])
            if v < 1e-20:
                break
            diff = bs_p - p
            if abs(diff) < newton_tol:
                converged = True
                break
            sigma -= diff / v
            if sigma < vol_lower or sigma > vol_upper:
                break

        if not converged:
            # Brent fallback
            try:
                def objective(vol: float) -> float:
                    return _bs_price_scalar(vol, forward, K, T, df, option_type) - p

                sigma = brentq(objective, vol_lower, vol_upper, xtol=newton_tol)
                converged = True
            except (ValueError, RuntimeError):
                continue

        if converged and vol_lower <= sigma <= vol_upper:
            result[i] = sigma

    return result
