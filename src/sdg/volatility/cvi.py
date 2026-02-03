"""CVI (Convex Volatility Interpolation) volatility surface calibrator.

Implements the CVI method from Deschatres (2025) for fitting arbitrage-free
implied volatility surfaces using quadratic programming with linear constraints.

The calibration solves for B-spline weights that parameterize the variance
surface, minimizing a weighted combination of:
- Least squares fit to mid-market variances (Section 3.2)
- Penalty for exceeding ask variances (Section 3.2)
- Penalty for falling below bid variances (Section 3.2)
- Strike regularization via L1 norm of convexity differences (Section 3.2)

Subject to linear constraints enforcing:
- Positivity of variance (Section 3.3.1)
- Linear extrapolation and Lee's tail bounds (Section 3.3.2)
- No calendar spread arbitrage (Section 3.3.3)
- Linearized no butterfly arbitrage (Section 3.3.4)

References:
    Deschatres (2025), "Convex Volatility Interpolation"
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from sdg.core.black_scholes import d1 as _bs_d1_core, vega as _bs_vega_core
from sdg.volatility.types import CVIConfig, CVIResult, ExpiryData
from sdg.volatility.bspline import (
    build_knot_vector,
    eval_basis,
    eval_basis_extrap,
    build_dual_transform,
)


# ---------------------------------------------------------------------------
# Black-Scholes helpers
# ---------------------------------------------------------------------------

def _bs_d1(forward: float, strikes: np.ndarray, vols: np.ndarray, T: float) -> np.ndarray:
    """Compute Black-Scholes d1."""
    return _bs_d1_core(forward, strikes, vols, T)


def _bs_vega(
    forward: float,
    strikes: np.ndarray,
    vols: np.ndarray,
    T: float,
    df: float,
) -> np.ndarray:
    """Compute Black-Scholes vega (derivative of price w.r.t. volatility)."""
    return _bs_vega_core(forward, strikes, vols, T, df)


def estimate_anchor_atm_vol(expiry: ExpiryData) -> float:
    """Estimate the anchor ATM volatility from option quotes near ATM.

    Uses the mid-vol of the option closest to the forward price.

    Args:
        expiry: Market data for one expiry.

    Returns:
        Estimated ATM volatility sigma_star.
    """
    mid_vols = np.where(
        np.isnan(expiry.bid_vols) | np.isnan(expiry.ask_vols),
        np.nan,
        (expiry.bid_vols + expiry.ask_vols) / 2.0,
    )
    log_moneyness = np.abs(np.log(expiry.strikes / expiry.forward))
    # Among options with valid mid, pick the one closest to ATM
    valid = ~np.isnan(mid_vols)
    if not np.any(valid):
        raise ValueError("No options with both bid and ask for ATM vol estimation")
    valid_idx = np.where(valid)[0]
    best = valid_idx[np.argmin(log_moneyness[valid_idx])]
    return float(mid_vols[best])


# ---------------------------------------------------------------------------
# Butterfly arbitrage helpers (Section 3.3.4, Appendix B)
# ---------------------------------------------------------------------------

def _compute_butterfly_betas(
    z: np.ndarray,
    v_ref: np.ndarray,
    s_ref: np.ndarray,
    v_star: float,
    sigma_star: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linearized no-butterfly-arbitrage coefficients beta0, beta1, beta2.

    The linearized constraint (Eq. 5) is:
        c >= beta0 + beta1 * (s - s_ref) + beta2 * (v - v_ref)

    These are derived from linearizing the PDF >= 0 condition (Eq. 13)
    at the reference solution.

    Args:
        z: Evaluation points in normalized log-moneyness space, shape (p,).
        v_ref: Reference variance at each z, shape (p,).
        s_ref: Reference normalized skew at each z, shape (p,).
        v_star: Anchor ATM variance (sigma_star^2).
        sigma_star: Anchor ATM volatility.
        T: Time to expiry.

    Returns:
        Tuple (beta0, beta1, beta2), each of shape (p,).
    """
    k = z * sigma_star * np.sqrt(T)  # log-moneyness
    sigma_ref = np.sqrt(np.maximum(v_ref, 1e-12))
    vT = v_ref * T
    sqrt_vT = np.sqrt(np.maximum(vT, 1e-12))

    # d1, d2 at reference (Black-Scholes)
    d1_ref = (-k + vT / 2.0) / sqrt_vT
    d2_ref = (-k - vT / 2.0) / sqrt_vT

    sigma_ratio = sigma_star / sigma_ref

    # Eq. 5 coefficients
    beta0 = (
        (v_star / (2.0 * v_ref)) * s_ref**2
        + sigma_star * np.sqrt(T) * s_ref
        - 2.0 * (
            1.0
            + d1_ref * sigma_ratio * s_ref
            + d1_ref * d2_ref * s_ref**2 * v_star / (4.0 * v_ref)
        )
    )
    beta1 = (
        (v_star / v_ref) * s_ref
        + sigma_star * np.sqrt(T)
        - 2.0 * (
            d1_ref * sigma_ratio
            + d1_ref * d2_ref * s_ref * v_star / (2.0 * v_ref)
        )
    )
    beta2 = (
        (s_ref / v_ref) * (
            -(v_star / (2.0 * v_ref)) * s_ref
            - (2.0 * k / (v_ref * T)) * (
                sigma_star * np.sqrt(T)
                - 0.5 * k * s_ref * v_star / v_ref
            )
        )
    )

    return beta0, beta1, beta2


def _compute_s_min(v: float, k: float, T: float, sigma_star: float) -> float:
    """Minimum normalized skew at the left edge knot z_0 (Eq. 18).

    This prevents negative probability density in the left tail where
    the convexity is zero.
    """
    # Clamp v to positive to handle edge cases from optimization
    v = max(v, 1e-12)
    vT = v * T
    sqrt_vT = np.sqrt(vT)
    sqrt_term = np.sqrt(1.0 + vT / 4.0)
    denom = k**2 - (v**2 * T**2) / 4.0 - vT
    if abs(denom) < 1e-15:
        return -1e10
    return 2.0 * v * np.sqrt(T) * (k + sqrt_vT * sqrt_term) / (sigma_star * denom)


def _compute_s_max(v: float, k: float, T: float, sigma_star: float) -> float:
    """Maximum normalized skew at the right edge knot z_{n-1} (Eq. 19).

    This prevents negative probability density in the right tail where
    the convexity is zero.
    """
    # Clamp v to positive to handle edge cases from optimization
    v = max(v, 1e-12)
    vT = v * T
    sqrt_vT = np.sqrt(vT)
    sqrt_term = np.sqrt(1.0 + vT / 4.0)
    denom = k**2 - (v**2 * T**2) / 4.0 - vT
    if abs(denom) < 1e-15:
        return 1e10
    return 2.0 * v * np.sqrt(T) * (k - sqrt_vT * sqrt_term) / (sigma_star * denom)


def _finite_diff_deriv(func, v: float, *args, eps_rel: float = 1e-6) -> float:
    """Central finite difference derivative df/dv."""
    eps = max(abs(v) * eps_rel, 1e-10)
    return (func(v + eps, *args) - func(v - eps, *args)) / (2.0 * eps)


# ---------------------------------------------------------------------------
# CVI Calibrator
# ---------------------------------------------------------------------------

class CVICalibrator:
    """CVI volatility surface calibrator.

    Fits an arbitrage-free volatility surface to market option bid/ask quotes
    using convex optimization (quadratic programming).

    Usage:
        config = CVIConfig(n_knots=11, regularization=0.05)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate(expiries)
        vols = evaluate_vol(result, strikes, expiry_index=0)
    """

    def __init__(self, config: CVIConfig | None = None) -> None:
        self.config = config or CVIConfig()

    def calibrate(self, expiries: list[ExpiryData]) -> CVIResult:
        """Calibrate the CVI volatility surface.

        Args:
            expiries: Market data per expiry, sorted by increasing
                time_to_expiry.

        Returns:
            CVIResult containing calibrated B-spline weights.
        """
        # Sort expiries by time
        expiries = sorted(expiries, key=lambda e: e.time_to_expiry)

        # Estimate anchor ATM vols where not provided
        for exp in expiries:
            if exp.anchor_atm_vol is None:
                exp.anchor_atm_vol = estimate_anchor_atm_vol(exp)

        # Dispatch based on calibration mode
        if self.config.calibration_mode == "independent":
            return self._calibrate_independent(expiries)
        else:
            return self._calibrate_joint(expiries)

    def _calibrate_joint(self, expiries: list[ExpiryData]) -> CVIResult:
        """Calibrate all expiries jointly in a single QP."""
        # Build knot structure (shared across all expiries)
        breakpoints = self._build_breakpoints(expiries)
        knot_vector = build_knot_vector(breakpoints)
        M = build_dual_transform(breakpoints, knot_vector)

        # First iteration: solve without no-butterfly-arbitrage constraints
        # (Section 3.3.4, pseudo-code)
        weights = self._solve_qp(
            expiries, breakpoints, knot_vector, M, butterfly_ref=None,
        )

        # Subsequent iterations: linearized butterfly constraints
        # Build mask of flat smiles (skip butterfly for these)
        flat_mask = [self._is_smile_flat(exp) for exp in expiries]

        for _ in range(self.config.max_iterations - 1):
            if all(flat_mask):
                break  # All smiles flat, skip butterfly iteration
            butterfly_ref = weights.copy()
            weights = self._solve_qp(
                expiries, breakpoints, knot_vector, M,
                butterfly_ref=butterfly_ref,
                flat_mask=flat_mask,
            )

        # Compute calibrated vols and errors at market strikes
        calibrated_vols, calibration_errors = self._compute_calibration_diagnostics(
            expiries, weights, breakpoints, knot_vector,
        )

        return CVIResult(
            bspline_weights=weights,
            breakpoints=breakpoints,
            knot_vector=knot_vector,
            expiries=expiries,
            config=self.config,
            calibrated_vols=calibrated_vols,
            calibration_errors=calibration_errors,
        )

    def _calibrate_independent(self, expiries: list[ExpiryData]) -> CVIResult:
        """Calibrate each expiry independently in separate QPs."""
        weights_list = []
        breakpoints_list = []
        knot_vector_list = []

        for exp in expiries:
            # Build knot structure for this expiry only
            breakpoints = self._build_breakpoints([exp])
            knot_vector = build_knot_vector(breakpoints)
            M = build_dual_transform(breakpoints, knot_vector)

            # First iteration
            weights = self._solve_qp(
                [exp], breakpoints, knot_vector, M, butterfly_ref=None,
            )

            # Subsequent iterations: skip if smile is flat
            is_flat = self._is_smile_flat(exp)
            for _ in range(self.config.max_iterations - 1):
                if is_flat:
                    break
                butterfly_ref = weights.copy()
                weights = self._solve_qp(
                    [exp], breakpoints, knot_vector, M,
                    butterfly_ref=butterfly_ref,
                    flat_mask=[False],  # Not flat, apply butterfly
                )

            weights_list.append(weights[0])  # Extract single expiry weights
            breakpoints_list.append(breakpoints)
            knot_vector_list.append(knot_vector)

        # Compute calibrated vols and errors at market strikes
        calibrated_vols, calibration_errors = self._compute_calibration_diagnostics_independent(
            expiries, weights_list, breakpoints_list, knot_vector_list,
        )

        return CVIResult(
            bspline_weights=None,
            breakpoints=None,
            knot_vector=None,
            expiries=expiries,
            config=self.config,
            bspline_weights_list=weights_list,
            breakpoints_list=breakpoints_list,
            knot_vector_list=knot_vector_list,
            calibrated_vols=calibrated_vols,
            calibration_errors=calibration_errors,
        )

    # ------------------------------------------------------------------
    # Knot construction
    # ------------------------------------------------------------------

    def _build_breakpoints(self, expiries: list[ExpiryData]) -> np.ndarray:
        """Build breakpoints based on the configured knot_spacing strategy.

        Strategies:
        - "uniform": Evenly spaced knots from -z_range to +z_range.
        - "atm_dense": Denser knots near ATM (step of 1 for |z|<=2, step of 2
          beyond), following the paper's Appendix C example.
        - "market": Include knots at market z-points for exact interpolation,
          plus evenly spaced base knots for smooth interpolation.
        """
        z_range = self.config.z_range
        spacing = self.config.knot_spacing

        if spacing == "uniform":
            # Evenly spaced knots
            n = self.config.n_knots
            if n % 2 == 0:
                n += 1  # Round up to odd to include z = 0
            return np.linspace(-z_range, z_range, n)

        elif spacing == "atm_dense":
            # ATM-dense pattern from Appendix C (page 29):
            # Step of 1 for |z| <= 2, step of 2 beyond
            # Example for z_range=8: [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
            knots = []
            # Left wing: step of 2 from -z_range to -2
            z = -z_range
            while z < -2.0:
                knots.append(z)
                z += 2.0
            # Near ATM: step of 1 from -2 to 2
            for z in np.arange(-2.0, 2.0 + 0.5, 1.0):
                knots.append(z)
            # Right wing: step of 2 from 2 to z_range
            z = 4.0
            while z <= z_range:
                knots.append(z)
                z += 2.0
            # Ensure edge knots are included
            knots = list(set(knots))
            knots.append(-z_range)
            knots.append(z_range)
            return np.unique(np.sort(np.array(knots)))

        elif spacing in ("market", "market_only"):
            # Collect market z-points for each expiry
            market_z = []
            for exp in expiries:
                sigma_star = exp.anchor_atm_vol
                T = exp.time_to_expiry
                k_arr = np.log(exp.strikes / exp.forward)
                z_arr = k_arr / (sigma_star * np.sqrt(T))
                market_z.extend(z_arr)

            # Clip to z_range
            market_z = np.array(market_z)
            market_z = market_z[(market_z >= -z_range) & (market_z <= z_range)]

            if spacing == "market":
                # Market z-points plus evenly spaced base knots
                n = self.config.n_knots
                if n % 2 == 0:
                    n += 1
                base_knots = np.linspace(-z_range, z_range, n)
                all_knots = np.concatenate([base_knots, market_z, [-z_range, z_range]])
            else:
                # market_only: minimal set with only market z-points and edge knots
                all_knots = np.concatenate([market_z, [-z_range, z_range]])

            return np.unique(np.sort(all_knots))

        else:
            raise ValueError(
                f"Unknown knot_spacing '{spacing}'. "
                "Use 'uniform', 'atm_dense', 'market', or 'market_only'."
            )

    # ------------------------------------------------------------------
    # Flat smile detection
    # ------------------------------------------------------------------

    def _is_smile_flat(
        self,
        exp: ExpiryData,
        vol_range_threshold: float = 0.005,
    ) -> bool:
        """Check if a smile is flat (low vol range).

        Flat smiles can cause numerical instability in butterfly linearization
        because the reference solution has near-zero curvature, making the
        linearized constraints poorly conditioned.

        Args:
            exp: Market data for one expiry.
            vol_range_threshold: Maximum vol range (max - min) to be considered flat.
                Default 0.5% (0.005). Smiles with less vol variation than this are
                effectively flat and butterfly constraints add no value.

        Returns:
            True if the smile is flat and butterfly iteration should be skipped.
        """
        valid = ~np.isnan(exp.bid_vols) & ~np.isnan(exp.ask_vols)
        if not np.any(valid):
            return True  # No valid data, skip butterfly

        mid_vols = (exp.bid_vols[valid] + exp.ask_vols[valid]) / 2.0
        vol_range = mid_vols.max() - mid_vols.min()
        return vol_range < vol_range_threshold

    def _all_smiles_flat(
        self,
        expiries: list[ExpiryData],
        weights: np.ndarray,
        breakpoints: np.ndarray,
        knot_vector: np.ndarray,
    ) -> bool:
        """Check if all smiles are flat.

        Args:
            expiries: Market data per expiry.
            weights: Current B-spline weights (unused, kept for API compatibility).
            breakpoints: Knot breakpoints (unused, kept for API compatibility).
            knot_vector: Augmented B-spline knot vector (unused).

        Returns:
            True if all smiles are flat and butterfly iteration should be skipped.
        """
        return all(self._is_smile_flat(exp) for exp in expiries)

    # ------------------------------------------------------------------
    # Calibration diagnostics
    # ------------------------------------------------------------------

    def _compute_calibration_diagnostics(
        self,
        expiries: list[ExpiryData],
        weights: np.ndarray,
        breakpoints: np.ndarray,
        knot_vector: np.ndarray,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute calibrated vols and errors at market strikes (joint mode).

        Args:
            expiries: Market data per expiry.
            weights: B-spline weights, shape (m, n_basis).
            breakpoints: Knot breakpoints in z-space.
            knot_vector: Augmented B-spline knot vector.

        Returns:
            Tuple of (calibrated_vols, calibration_errors), each a list of arrays.
        """
        calibrated_vols = []
        calibration_errors = []

        for j, exp in enumerate(expiries):
            sigma_star = exp.anchor_atm_vol
            T = exp.time_to_expiry
            F = exp.forward

            # Convert strikes to z-space
            k = np.log(exp.strikes / F)
            z = k / (sigma_star * np.sqrt(T))

            # Evaluate variance
            B = eval_basis_extrap(z, knot_vector, breakpoints)
            variance = B @ weights[j]
            variance = np.maximum(variance, 1e-12)
            cal_vols = np.sqrt(variance)
            calibrated_vols.append(cal_vols)

            # Compute mid vols and errors
            mid_vols = np.where(
                np.isnan(exp.bid_vols) | np.isnan(exp.ask_vols),
                np.nan,
                (exp.bid_vols + exp.ask_vols) / 2.0,
            )
            errors = cal_vols - mid_vols
            calibration_errors.append(errors)

        return calibrated_vols, calibration_errors

    def _compute_calibration_diagnostics_independent(
        self,
        expiries: list[ExpiryData],
        weights_list: list[np.ndarray],
        breakpoints_list: list[np.ndarray],
        knot_vector_list: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Compute calibrated vols and errors at market strikes (independent mode).

        Args:
            expiries: Market data per expiry.
            weights_list: Per-expiry B-spline weights.
            breakpoints_list: Per-expiry breakpoints.
            knot_vector_list: Per-expiry knot vectors.

        Returns:
            Tuple of (calibrated_vols, calibration_errors), each a list of arrays.
        """
        calibrated_vols = []
        calibration_errors = []

        for j, exp in enumerate(expiries):
            sigma_star = exp.anchor_atm_vol
            T = exp.time_to_expiry
            F = exp.forward
            weights = weights_list[j]
            breakpoints = breakpoints_list[j]
            knot_vector = knot_vector_list[j]

            # Convert strikes to z-space
            k = np.log(exp.strikes / F)
            z = k / (sigma_star * np.sqrt(T))

            # Evaluate variance
            B = eval_basis_extrap(z, knot_vector, breakpoints)
            variance = B @ weights
            variance = np.maximum(variance, 1e-12)
            cal_vols = np.sqrt(variance)
            calibrated_vols.append(cal_vols)

            # Compute mid vols and errors
            mid_vols = np.where(
                np.isnan(exp.bid_vols) | np.isnan(exp.ask_vols),
                np.nan,
                (exp.bid_vols + exp.ask_vols) / 2.0,
            )
            errors = cal_vols - mid_vols
            calibration_errors.append(errors)

        return calibrated_vols, calibration_errors

    # ------------------------------------------------------------------
    # QP solve
    # ------------------------------------------------------------------

    def _solve_qp(
        self,
        expiries: list[ExpiryData],
        breakpoints: np.ndarray,
        knot_vector: np.ndarray,
        M: np.ndarray,
        butterfly_ref: np.ndarray | None,
        flat_mask: list[bool] | None = None,
    ) -> np.ndarray:
        """Build and solve the CVXPY quadratic program.

        Args:
            expiries: Market data per expiry.
            breakpoints: Cubic spline knots in z-space.
            knot_vector: Augmented B-spline knot vector.
            M: Dual transformation matrix (cubic_params = M @ bspline_weights).
            butterfly_ref: B-spline weights from previous iteration for
                linearized butterfly constraints, or None for first iteration.
            flat_mask: Per-expiry mask indicating which smiles are flat and should
                skip butterfly constraints (True = flat, skip constraints).

        Returns:
            Calibrated B-spline weights, shape (m, n_basis).
        """
        m = len(expiries)
        n = len(breakpoints)
        n_basis = n + 2

        # CVXPY variables: one vector of B-spline weights per expiry
        alphas = [cp.Variable(n_basis, name=f"alpha_{j}") for j in range(m)]

        objective_terms = []
        constraints = []

        # Precompute basis derivatives at knots (shared across expiries)
        B_d2_knots = eval_basis(breakpoints, knot_vector, 3, deriv=2)
        B_d1_z0 = eval_basis(np.array([breakpoints[0]]), knot_vector, 3, deriv=1)[0]
        B_d1_zn = eval_basis(np.array([breakpoints[-1]]), knot_vector, 3, deriv=1)[0]
        B_d2_z0 = eval_basis(np.array([breakpoints[0]]), knot_vector, 3, deriv=2)[0]
        B_d2_zn = eval_basis(np.array([breakpoints[-1]]), knot_vector, 3, deriv=2)[0]

        # ----------------------------------------------------------
        # Per-expiry objective terms and constraints
        # ----------------------------------------------------------
        for j, exp in enumerate(expiries):
            v_star = exp.anchor_atm_vol**2
            sigma_star = exp.anchor_atm_vol
            T = exp.time_to_expiry
            F = exp.forward

            # Convert strikes to normalized log-moneyness z (Section 2.1)
            k_arr = np.log(exp.strikes / F)
            z_arr = k_arr / (sigma_star * np.sqrt(T))

            # Design matrix at market strikes (with extrapolation)
            B_mkt = eval_basis_extrap(z_arr, knot_vector, breakpoints)

            # Market variances
            bid_var = exp.bid_vols**2
            ask_var = exp.ask_vols**2

            has_bid = ~np.isnan(exp.bid_vols)
            has_ask = ~np.isnan(exp.ask_vols)
            has_mid = has_bid & has_ask

            # === LEAST SQUARES PENALTY (Section 3.2) ===
            idx_mid = np.where(
                has_mid
                & (z_arr >= breakpoints[0])
                & (z_arr <= breakpoints[-1])
            )[0]

            if len(idx_mid) > 0:
                N_mid = len(idx_mid)
                mid_var = ((exp.bid_vols[idx_mid] + exp.ask_vols[idx_mid]) / 2.0) ** 2
                v_spread = ask_var[idx_mid] - bid_var[idx_mid]
                # Weight: 1 / (v_ask - v_bid)^2 — makes it a chi-square
                w_ls = 1.0 / np.maximum(v_spread**2, 1e-20)

                residuals = B_mkt[idx_mid] @ alphas[j] - mid_var
                ls_penalty = (1.0 / N_mid) * cp.sum(cp.multiply(w_ls, cp.square(residuals)))
                objective_terms.append(ls_penalty)

            # Normalization constant q_j for ask/bid penalties
            q_j = 0.0
            if len(idx_mid) > 0:
                v_spread_all = ask_var[np.where(has_mid)[0]] - bid_var[np.where(has_mid)[0]]
                q_j = float(np.sum(1.0 / np.maximum(v_spread_all**2, 1e-20)))
            if q_j == 0.0:
                q_j = 1.0

            # === ABOVE ASK PENALTY (Section 3.2) ===
            idx_ask = np.where(has_ask)[0]
            if len(idx_ask) > 0:
                N_ask = len(idx_ask)
                vega_ask = _bs_vega(F, exp.strikes[idx_ask], exp.ask_vols[idx_ask], T, exp.discount_factor)
                total_vega_ask = np.sum(vega_ask) + 1e-30
                w_ask = q_j * vega_ask / total_vega_ask

                above = B_mkt[idx_ask] @ alphas[j] - ask_var[idx_ask]
                ask_penalty = (1.0 / N_ask) * cp.sum(cp.multiply(w_ask, cp.square(cp.pos(above))))
                objective_terms.append(ask_penalty)

            # === BELOW BID PENALTY (Section 3.2) ===
            idx_bid = np.where(
                has_bid
                & (z_arr >= breakpoints[0])
                & (z_arr <= breakpoints[-1])
            )[0]
            if len(idx_bid) > 0:
                N_bid = len(idx_bid)
                vega_bid = _bs_vega(F, exp.strikes[idx_bid], exp.bid_vols[idx_bid], T, exp.discount_factor)
                total_vega_bid = np.sum(vega_bid) + 1e-30
                w_bid = q_j * vega_bid / total_vega_bid

                below = bid_var[idx_bid] - B_mkt[idx_bid] @ alphas[j]
                bid_penalty = (1.0 / N_bid) * cp.sum(cp.multiply(w_bid, cp.square(cp.pos(below))))
                objective_terms.append(bid_penalty)

            # === STRIKE REGULARIZATION PENALTY (Section 3.2) ===
            # c_i = (1/v*) * d2v/dz2(z_i)
            c_vec = (1.0 / v_star) * (B_d2_knots @ alphas[j])
            # L1 norm of consecutive differences |c[i] - c[i+1]|
            c_diff = c_vec[:-1] - c_vec[1:]
            reg_penalty = self.config.regularization * cp.norm1(c_diff)
            objective_terms.append(reg_penalty)

            # === CONSTRAINTS ===

            # --- Positivity of variance (Section 3.3.1) ---
            # Only needed for the first (shortest) expiry; subsequent ones
            # are bounded below by calendar spread constraints.
            if j == 0:
                z_pos = np.linspace(breakpoints[0], breakpoints[-1],
                                    self.config.n_positivity_points)
                B_pos = eval_basis(z_pos, knot_vector, 3, deriv=0)
                constraints.append(B_pos @ alphas[j] >= self.config.variance_floor)

            # --- Boundary conditions (Section 3.3.2) ---

            # Linear extrapolation: convexity = 0 at edge knots
            constraints.append(B_d2_z0 @ alphas[j] == 0)
            constraints.append(B_d2_zn @ alphas[j] == 0)

            # Upward sloping wings (Eq. 1):
            # dv/dz(z_0) <= 0 and dv/dz(z_{n-1}) >= 0
            constraints.append(B_d1_z0 @ alphas[j] <= 0)
            constraints.append(B_d1_zn @ alphas[j] >= 0)

            # Lee's tail slope bounds (Eq. 2):
            # sqrt(T)/(2*sigma_star) * dv/dz(z_0) > -1
            # sqrt(T)/(2*sigma_star) * dv/dz(z_{n-1}) < 1
            lee_factor = np.sqrt(T) / (2.0 * sigma_star)
            constraints.append(lee_factor * (B_d1_z0 @ alphas[j]) >= -1.0 + 0.01)
            constraints.append(lee_factor * (B_d1_zn @ alphas[j]) <= 1.0 - 0.01)

            # --- Linearized no-butterfly-arbitrage constraints (Section 3.3.4) ---
            # Skip for flat smiles to avoid numerical instability
            skip_butterfly = flat_mask is not None and flat_mask[j]
            if butterfly_ref is not None and not skip_butterfly:
                self._add_butterfly_constraints(
                    constraints, alphas[j], j, exp,
                    breakpoints, knot_vector, butterfly_ref[j],
                )

        # ----------------------------------------------------------
        # No-calendar-spread-arbitrage constraints (Section 3.3.3)
        # ----------------------------------------------------------
        # Skip if calendar_penalty < 0 (disabled) or only one expiry
        if m > 1 and self.config.calendar_penalty >= 0:
            self._add_calendar_constraints(
                constraints, objective_terms, alphas, expiries, breakpoints, knot_vector,
            )

        # ----------------------------------------------------------
        # Solve
        # ----------------------------------------------------------
        objective = cp.Minimize(cp.sum(objective_terms))
        problem = cp.Problem(objective, constraints)

        # Get solver from config with fallback options
        solver_map = {
            "CLARABEL": cp.CLARABEL,
            "OSQP": cp.OSQP,
            "ECOS": cp.ECOS,
            "SCS": cp.SCS,
        }
        # Define fallback order: try primary solver first, then fallbacks
        primary_solver = solver_map.get(self.config.solver, cp.SCS)
        fallback_solvers = [cp.SCS, cp.CLARABEL, cp.ECOS, cp.OSQP]

        solvers_to_try = [primary_solver] + [s for s in fallback_solvers if s != primary_solver]

        last_error = None
        for solver in solvers_to_try:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in ("optimal", "optimal_inaccurate"):
                    break
                last_error = RuntimeError(f"CVI optimization failed: {problem.status}")
            except Exception as e:
                last_error = e
                continue
        else:
            # All solvers failed
            raise RuntimeError(
                f"CVI optimization failed with all solvers. Last error: {last_error}"
            )

        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(
                f"CVI optimization failed: {problem.status}"
            )

        return np.array([alphas[j].value for j in range(m)])

    # ------------------------------------------------------------------
    # Calendar spread constraints (Section 3.3.3)
    # ------------------------------------------------------------------

    def _add_calendar_constraints(
        self,
        constraints: list,
        objective_terms: list,
        alphas: list[cp.Variable],
        expiries: list[ExpiryData],
        breakpoints: np.ndarray,
        knot_vector: np.ndarray,
    ) -> None:
        """Add no-calendar-spread-arbitrage constraints.

        For any fixed strike-to-forward ratio, total variance must increase
        with time (Eq. 3). Also enforces tail constraints (Eq. 4).

        Behavior depends on config.calendar_penalty:
        - If > 0: soft constraints (penalty in objective). This allows the
          solver to find a feasible solution even when input data has
          calendar arbitrage.
        - If = 0: hard constraints (may fail on arbitrageable data).
        - If < 0: this method is not called (calendar constraints disabled).
        """
        m = len(expiries)
        r = self.config.n_calendar_strikes
        z_grid = np.linspace(breakpoints[0], breakpoints[-1], r)
        use_soft = self.config.calendar_penalty > 0

        # First derivative at edge knots (for tail constraints)
        B_d1_z0 = eval_basis(np.array([breakpoints[0]]), knot_vector, 3, deriv=1)[0]
        B_d1_zn = eval_basis(np.array([breakpoints[-1]]), knot_vector, 3, deriv=1)[0]

        for j in range(m - 1):
            exp_j = expiries[j]
            exp_j1 = expiries[j + 1]

            T_j = exp_j.time_to_expiry
            T_j1 = exp_j1.time_to_expiry
            sig_j = exp_j.anchor_atm_vol
            sig_j1 = exp_j1.anchor_atm_vol

            # --- Interior strikes (Eq. 3) ---
            # Convert z_grid to log-moneyness k (same K/F ratio for both expiries)
            k_grid = z_grid * sig_j * np.sqrt(T_j)

            # z-space coordinates for each expiry
            B_j = eval_basis_extrap(z_grid, knot_vector, breakpoints)
            z_j1 = k_grid / (sig_j1 * np.sqrt(T_j1))
            B_j1 = eval_basis_extrap(z_j1, knot_vector, breakpoints)

            # v(K, T_j) * T_j <= v(K * F_{j+1}/F_j, T_{j+1}) * T_{j+1}
            # This is expressed as a vectorized constraint
            w_j = T_j * (B_j @ alphas[j])
            w_j1 = T_j1 * (B_j1 @ alphas[j + 1])

            if use_soft:
                # Soft constraint: penalize violations
                # violation = max(0, w_j - w_j1)
                calendar_violation = cp.sum(cp.square(cp.pos(w_j - w_j1)))
                objective_terms.append(self.config.calendar_penalty * calendar_violation)
            else:
                # Hard constraint
                constraints.append(w_j <= w_j1)

            # --- Tail constraints (Eq. 4) ---
            # Left tail: s * sigma_star * sqrt(T) must not decrease too fast
            # dv/dz(z_0) * sqrt(T) / sigma_star is non-decreasing (magnitudes)
            coeff_j = np.sqrt(T_j) / sig_j
            coeff_j1 = np.sqrt(T_j1) / sig_j1

            # s at z_0 is negative: magnitude decreasing means algebraic value increasing
            left_tail_j = coeff_j * (B_d1_z0 @ alphas[j])
            left_tail_j1 = coeff_j1 * (B_d1_z0 @ alphas[j + 1])

            # s at z_{n-1} is positive: increasing
            right_tail_j = coeff_j * (B_d1_zn @ alphas[j])
            right_tail_j1 = coeff_j1 * (B_d1_zn @ alphas[j + 1])

            if use_soft:
                # Soft tail constraints
                # Left: left_tail_j >= left_tail_j1  =>  penalize pos(left_tail_j1 - left_tail_j)
                # Right: right_tail_j <= right_tail_j1  =>  penalize pos(right_tail_j - right_tail_j1)
                left_violation = cp.square(cp.pos(left_tail_j1 - left_tail_j))
                right_violation = cp.square(cp.pos(right_tail_j - right_tail_j1))
                objective_terms.append(self.config.calendar_penalty * (left_violation + right_violation))
            else:
                # Hard constraints
                constraints.append(left_tail_j >= left_tail_j1)
                constraints.append(right_tail_j <= right_tail_j1)

    # ------------------------------------------------------------------
    # Butterfly arbitrage constraints (Section 3.3.4, Appendix B)
    # ------------------------------------------------------------------

    def _add_butterfly_constraints(
        self,
        constraints: list,
        alpha: cp.Variable,
        expiry_idx: int,
        exp: ExpiryData,
        breakpoints: np.ndarray,
        knot_vector: np.ndarray,
        ref_weights: np.ndarray,
    ) -> None:
        """Add linearized no-butterfly-arbitrage constraints for one expiry.

        These enforce PDF >= 0 by linearizing the non-convex butterfly
        condition at the reference solution from the previous iteration.

        Note on flat smiles:
            For nearly flat smiles (vol range < 0.5%), the linearization can
            become numerically unstable because the reference curvature is
            near zero, making the beta coefficients poorly conditioned. The
            calibrator automatically detects flat smiles and skips butterfly
            constraints for them (see _is_smile_flat).

        Potential improvements for butterfly linearization:
            1. Adaptive beta scaling: Scale beta coefficients by smile curvature
               to prevent numerical instability when curvature is small.
            2. Soft butterfly constraints: Add butterfly violations as a penalty
               term instead of hard constraints (similar to calendar_penalty).
            3. Regularized linearization: Add a small regularization term to the
               beta computation to prevent division by near-zero values.
            4. Trust region: Limit how far the solution can move from the
               reference in each iteration to improve convergence.
        """
        v_star = exp.anchor_atm_vol**2
        sigma_star = exp.anchor_atm_vol
        T = exp.time_to_expiry
        n_pts = self.config.n_butterfly_strikes

        # --- Interior points: linearized PDF >= 0 (Eq. 5) ---
        z_inner = np.linspace(breakpoints[0], breakpoints[-1], n_pts + 2)[1:-1]

        B_val = eval_basis_extrap(z_inner, knot_vector, breakpoints)
        B_d1 = eval_basis(z_inner, knot_vector, 3, deriv=1)
        B_d2 = eval_basis(z_inner, knot_vector, 3, deriv=2)

        # Reference values at interior points
        v_ref = B_val @ ref_weights
        s_ref = (1.0 / v_star) * (B_d1 @ ref_weights)

        # Compute beta coefficients (Eq. 5)
        beta0, beta1, beta2 = _compute_butterfly_betas(
            z_inner, v_ref, s_ref, v_star, sigma_star, T,
        )

        # Constraint: c >= beta0 + beta1*(s - s_ref) + beta2*(v - v_ref)
        # In terms of alpha:
        #   (1/v*) * B_d2 @ alpha >= beta0 + beta1*((1/v*)*B_d1 @ alpha - s_ref)
        #                            + beta2*(B_val @ alpha - v_ref)
        # Rearranged row by row:
        #   A_row @ alpha >= rhs
        for i in range(n_pts):
            # LHS coefficient row
            A_row = (
                (1.0 / v_star) * B_d2[i]
                - beta1[i] * (1.0 / v_star) * B_d1[i]
                - beta2[i] * B_val[i]
            )
            rhs = beta0[i] - beta1[i] * s_ref[i] - beta2[i] * v_ref[i]
            constraints.append(A_row @ alpha >= rhs)

        # --- Edge knot z_0: s >= s_min linearized (Eq. 18, Section 3.3.4) ---
        z0 = breakpoints[0]
        k_z0 = z0 * sigma_star * np.sqrt(T)
        B_val_z0 = eval_basis_extrap(np.array([z0]), knot_vector, breakpoints)[0]
        B_d1_z0 = eval_basis(np.array([z0]), knot_vector, 3, deriv=1)[0]

        v_ref_z0 = float(B_val_z0 @ ref_weights)

        s_min_val = _compute_s_min(v_ref_z0, k_z0, T, sigma_star)
        ds_min_dv = _finite_diff_deriv(_compute_s_min, v_ref_z0, k_z0, T, sigma_star)

        # s >= s_min(v_ref) + ds_min/dv * (v - v_ref)
        # (1/v*) * B_d1_z0 @ alpha >= s_min_val + ds_min_dv * (B_val_z0 @ alpha - v_ref_z0)
        A_row_z0 = (1.0 / v_star) * B_d1_z0 - ds_min_dv * B_val_z0
        rhs_z0 = s_min_val - ds_min_dv * v_ref_z0
        constraints.append(A_row_z0 @ alpha >= rhs_z0)

        # --- Edge knot z_{n-1}: s <= s_max linearized (Eq. 19) ---
        zn = breakpoints[-1]
        k_zn = zn * sigma_star * np.sqrt(T)
        B_val_zn = eval_basis_extrap(np.array([zn]), knot_vector, breakpoints)[0]
        B_d1_zn = eval_basis(np.array([zn]), knot_vector, 3, deriv=1)[0]

        v_ref_zn = float(B_val_zn @ ref_weights)

        s_max_val = _compute_s_max(v_ref_zn, k_zn, T, sigma_star)
        ds_max_dv = _finite_diff_deriv(_compute_s_max, v_ref_zn, k_zn, T, sigma_star)

        # s <= s_max(v_ref) + ds_max/dv * (v - v_ref)
        # (1/v*) * B_d1_zn @ alpha <= s_max_val + ds_max_dv * (B_val_zn @ alpha - v_ref_zn)
        # Negate for >= form:
        A_row_zn = -(1.0 / v_star) * B_d1_zn + ds_max_dv * B_val_zn
        rhs_zn = -(s_max_val - ds_max_dv * v_ref_zn)
        constraints.append(A_row_zn @ alpha >= rhs_zn)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_vol(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Evaluate the calibrated implied volatility at given strikes for one expiry.

    Args:
        result: Calibrated CVI result.
        strikes: Array of strikes at which to evaluate.
        expiry_index: Index of the expiry (0-based) in the calibration.

    Returns:
        Array of implied volatilities (sigma, not variance).
    """
    exp = result.expiries[expiry_index]
    sigma_star = exp.anchor_atm_vol
    T = exp.time_to_expiry
    F = exp.forward

    # Get weights, knot_vector, breakpoints based on calibration mode
    if result.bspline_weights_list is not None:
        # Independent mode: per-expiry structures
        weights = result.bspline_weights_list[expiry_index]
        knot_vector = result.knot_vector_list[expiry_index]
        breakpoints = result.breakpoints_list[expiry_index]
    else:
        # Joint mode: shared structures
        weights = result.bspline_weights[expiry_index]
        knot_vector = result.knot_vector
        breakpoints = result.breakpoints

    # Convert strikes to normalized log-moneyness
    k = np.log(strikes / F)
    z = k / (sigma_star * np.sqrt(T))

    # Evaluate variance via B-spline (with linear extrapolation)
    B = eval_basis_extrap(z, knot_vector, breakpoints)
    variance = B @ weights

    # Variance should be positive; clip for numerical safety
    variance = np.maximum(variance, 1e-12)
    return np.sqrt(variance)


def evaluate_total_variance(
    result: CVIResult,
    strikes: np.ndarray,
    expiry_index: int,
) -> np.ndarray:
    """Evaluate total variance (v * T) at given strikes for one expiry.

    Useful for visualizing calendar spread arbitrage (total variance
    curves should not intersect across expiries).
    """
    exp = result.expiries[expiry_index]
    vols = evaluate_vol(result, strikes, expiry_index)
    return vols**2 * exp.time_to_expiry
