"""Data types for the CVI volatility surface calibrator.

References:
    Deschatres (2025), "Convex Volatility Interpolation"
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ExpiryData:
    """Market data for a single expiry.

    Attributes:
        time_to_expiry: Time to expiry in years (T).
        forward: Forward price (F).
        discount_factor: Discount factor for this expiry.
        strikes: Array of option strikes.
        bid_vols: Array of bid implied volatilities (NaN where missing).
        ask_vols: Array of ask implied volatilities (NaN where missing).
        anchor_atm_vol: Anchor ATM volatility estimate (sigma_star).
            If None, will be estimated from market data during calibration.
    """

    time_to_expiry: float
    forward: float
    discount_factor: float
    strikes: np.ndarray
    bid_vols: np.ndarray
    ask_vols: np.ndarray
    anchor_atm_vol: float | None = None


@dataclass
class CVIConfig:
    """Configuration for the CVI calibrator.

    Attributes:
        n_knots: Total number of cubic spline knots (rounded up to odd
            to ensure z=0 is included). Used when knot_spacing="uniform".
        z_range: Range of normalized log-moneyness for edge knots.
            Edge knots are placed at -z_range and +z_range.
        regularization: Strike regularization factor lambda (Section 3.2).
            Default 0.05 as recommended by the paper.
        n_calendar_strikes: Number of strikes for calendar spread
            arbitrage constraints (Section 3.3.3).
        n_butterfly_strikes: Number of strikes for butterfly arbitrage
            constraints (Section 3.3.4).
        n_positivity_points: Number of points for variance positivity
            constraints (Section 3.3.1).
        max_iterations: Maximum number of butterfly linearization
            iterations (Section 3.3.4). Typically 2 suffices.
        variance_floor: Minimum allowed variance for positivity constraints.
        calendar_penalty: Penalty weight for soft calendar constraints.
            If > 0, calendar constraints are soft (penalty in objective).
            If = 0, calendar constraints are hard (may fail on arbitrageable data).
            If < 0 (e.g. -1), calendar constraints are completely disabled.
        solver: CVXPY solver to use. Default "SCS".
        knot_spacing: Knot placement strategy:
            - "uniform": Evenly spaced knots from -z_range to +z_range.
            - "atm_dense": Denser knots near ATM (step of 1 for |z|<=2, step of 2
              beyond), following the paper's Appendix C example.
            - "market": Include knots at market z-points for exact interpolation.
              Additional evenly-spaced knots (up to n_knots) are added.
            - "market_only": Minimal set with only market z-points and edge knots.
              Fastest option while maintaining exact interpolation at market strikes.
        calibration_mode: How to calibrate multiple expiries:
            - "joint": All expiries calibrated together in one QP (default).
              Allows calendar spread constraints and uses shared breakpoints.
            - "independent": Each expiry calibrated separately. Faster (smaller
              problems), each expiry gets its own breakpoints. No calendar
              constraints possible.
    """

    n_knots: int = 21
    z_range: float = 8.0
    regularization: float = 0.05
    n_calendar_strikes: int = 20
    n_butterfly_strikes: int = 20
    n_positivity_points: int = 20
    max_iterations: int = 2
    variance_floor: float = 1e-6
    calendar_penalty: float = 1.0
    solver: str = "SCS"
    knot_spacing: str = "market"
    calibration_mode: str = "joint"


@dataclass
class CVIResult:
    """Result of a CVI calibration.

    Attributes:
        bspline_weights: B-spline weights per expiry. For joint mode, shape
            (m, n_basis). For independent mode, use bspline_weights_list instead.
        breakpoints: Cubic spline breakpoints (knots) in z-space. Shared across
            expiries in joint mode. For independent mode, use breakpoints_list.
        knot_vector: Augmented B-spline knot vector. Shared across expiries in
            joint mode. For independent mode, use knot_vector_list.
        expiries: List of ExpiryData used for calibration.
        config: CVIConfig used for calibration.
        bspline_weights_list: Per-expiry B-spline weights for independent mode.
            Each element may have different length.
        breakpoints_list: Per-expiry breakpoints for independent mode.
        knot_vector_list: Per-expiry knot vectors for independent mode.
    """

    bspline_weights: np.ndarray | None
    breakpoints: np.ndarray | None
    knot_vector: np.ndarray | None
    expiries: list[ExpiryData]
    config: CVIConfig
    bspline_weights_list: list[np.ndarray] | None = None
    breakpoints_list: list[np.ndarray] | None = None
    knot_vector_list: list[np.ndarray] | None = None
