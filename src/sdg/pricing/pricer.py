"""Option pricing using a calibrated CVI volatility surface."""

from __future__ import annotations

import enum

import numpy as np

from sdg.core.black_scholes import (
    call_price,
    put_price,
    digital_call_price,
    digital_put_price,
)
from sdg.volatility.cvi import evaluate_vol
from sdg.volatility.types import CVIResult


class OptionKind(enum.Enum):
    """Option kind for pricing."""

    VANILLA_CALL = "vanilla_call"
    VANILLA_PUT = "vanilla_put"
    DIGITAL_CALL = "digital_call"
    DIGITAL_PUT = "digital_put"


_PRICE_FUNCS = {
    OptionKind.VANILLA_CALL: call_price,
    OptionKind.VANILLA_PUT: put_price,
    OptionKind.DIGITAL_CALL: digital_call_price,
    OptionKind.DIGITAL_PUT: digital_put_price,
}


def price_option(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
    option_kind: OptionKind,
) -> np.ndarray:
    """Price an option using the calibrated vol surface.

    Evaluates implied volatility from the CVI result at each strike,
    then applies the appropriate Black-Scholes formula.

    Args:
        result: Calibrated CVI result.
        strikes: Strike prices to evaluate.
        expiry_index: Index of the expiry in the calibration.
        option_kind: Type of option to price.

    Returns:
        Array of option prices.
    """
    strikes = np.asarray(strikes, dtype=float)
    exp = result.expiries[expiry_index]
    vols = evaluate_vol(result, strikes, expiry_index)

    price_func = _PRICE_FUNCS[option_kind]
    return price_func(exp.forward, strikes, vols, exp.time_to_expiry, exp.discount_factor)


def price_vanilla_call(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Price a vanilla call using the calibrated vol surface."""
    return price_option(result, strikes, expiry_index, OptionKind.VANILLA_CALL)


def price_vanilla_put(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Price a vanilla put using the calibrated vol surface."""
    return price_option(result, strikes, expiry_index, OptionKind.VANILLA_PUT)


def price_digital_call(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Price a digital call using the calibrated vol surface."""
    return price_option(result, strikes, expiry_index, OptionKind.DIGITAL_CALL)


def price_digital_put(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Price a digital put using the calibrated vol surface."""
    return price_option(result, strikes, expiry_index, OptionKind.DIGITAL_PUT)
