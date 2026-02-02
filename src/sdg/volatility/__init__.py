"""CVI volatility surface calibration module.

Implements Convex Volatility Interpolation (Deschatres, 2025) for fitting
arbitrage-free implied volatility surfaces from market option bid/ask quotes.
"""

from sdg.volatility.types import CVIConfig, CVIResult, ExpiryData
from sdg.volatility.cvi import CVICalibrator, evaluate_vol, evaluate_total_variance

__all__ = [
    "CVICalibrator",
    "CVIConfig",
    "CVIResult",
    "ExpiryData",
    "evaluate_vol",
    "evaluate_total_variance",
]
