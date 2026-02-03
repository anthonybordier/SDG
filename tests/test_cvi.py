"""Tests for the CVI volatility surface calibrator."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sdg.volatility.types import ExpiryData, CVIConfig
from sdg.volatility.cvi import CVICalibrator, evaluate_vol, evaluate_total_variance

FIXTURES = Path(__file__).parent / "fixtures"
from sdg.volatility.bspline import (
    build_knot_vector,
    eval_basis,
    eval_basis_extrap,
    build_dual_transform,
)


# ---------------------------------------------------------------------------
# Helpers for synthetic market data
# ---------------------------------------------------------------------------

def _make_synthetic_smile(
    T: float = 0.25,
    forward: float = 100.0,
    atm_vol: float = 0.20,
    skew: float = -0.05,
    convexity: float = 0.02,
    n_strikes: int = 21,
    spread: float = 0.005,
    z_range: float = 3.0,
) -> ExpiryData:
    """Generate a synthetic vol smile with bid/ask spread.

    The smile shape is: sigma(z) = atm_vol * sqrt(1 + convexity*z^2 + skew*z)
    where z = log(K/F) / (atm_vol * sqrt(T)).
    """
    sigma_star = atm_vol
    z_grid = np.linspace(-z_range, z_range, n_strikes)
    k_grid = z_grid * sigma_star * np.sqrt(T)
    strikes = forward * np.exp(k_grid)

    # Synthetic vol smile
    shape = 1.0 + convexity * z_grid**2 + skew * z_grid
    shape = np.maximum(shape, 0.1)  # floor to keep positive
    vols = atm_vol * np.sqrt(shape)

    bid_vols = vols - spread / 2.0
    ask_vols = vols + spread / 2.0

    return ExpiryData(
        time_to_expiry=T,
        forward=forward,
        discount_factor=np.exp(-0.05 * T),
        strikes=strikes,
        bid_vols=bid_vols,
        ask_vols=ask_vols,
        anchor_atm_vol=atm_vol,
    )


# ---------------------------------------------------------------------------
# B-spline unit tests
# ---------------------------------------------------------------------------

class TestBSpline:
    """Tests for B-spline basis functions."""

    def test_knot_vector_length(self):
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp, degree=3)
        # Length should be n + 2*degree = 7 + 6 = 13
        assert len(kv) == 13

    def test_n_basis_functions(self):
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp, degree=3)
        n_basis = len(kv) - 3 - 1
        # Should be n + 2 = 9
        assert n_basis == 9

    def test_partition_of_unity(self):
        """B-spline basis functions should sum to 1 inside the knot range."""
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp)
        z = np.linspace(-5.9, 5.9, 50)
        B = eval_basis(z, kv)
        row_sums = B.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_extrapolation_linear(self):
        """Extrapolation should be linear outside breakpoints."""
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp)
        # Set some weights
        weights = np.array([1.0, 0.8, 0.5, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2])

        z_outside = np.array([-8.0, -7.0, 7.0, 8.0])
        B_ext = eval_basis_extrap(z_outside, kv, bp)
        v_outside = B_ext @ weights

        # Check linearity: for the left tail, v(-8) and v(-7) should lie on
        # the line defined by v(-6) and its derivative
        z_inner = np.array([-6.0])
        B_inner = eval_basis(z_inner, kv, deriv=0)
        B_inner_d = eval_basis(z_inner, kv, deriv=1)
        v_boundary = float((B_inner @ weights)[0])
        dv_boundary = float((B_inner_d @ weights)[0])

        expected_m8 = v_boundary + dv_boundary * (-8.0 - (-6.0))
        expected_m7 = v_boundary + dv_boundary * (-7.0 - (-6.0))
        assert abs(v_outside[0] - expected_m8) < 1e-10
        assert abs(v_outside[1] - expected_m7) < 1e-10

    def test_dual_transform_invertible(self):
        """Dual transformation matrix should be invertible."""
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp)
        M = build_dual_transform(bp, kv)
        assert M.shape == (9, 9)
        det = np.linalg.det(M)
        assert abs(det) > 1e-10

    def test_dual_transform_round_trip(self):
        """Converting B-spline -> cubic -> B-spline should be identity."""
        bp = np.linspace(-6, 6, 7)
        kv = build_knot_vector(bp)
        M = build_dual_transform(bp, kv)
        M_inv = np.linalg.inv(M)

        weights = np.array([1.0, 0.8, 0.5, 0.3, 0.4, 0.6, 0.8, 1.0, 1.2])
        cubic_params = M @ weights
        recovered = M_inv @ cubic_params
        np.testing.assert_allclose(recovered, weights, atol=1e-10)


# ---------------------------------------------------------------------------
# CVI calibration tests
# ---------------------------------------------------------------------------

class TestCVISingleExpiry:
    """Test CVI calibration on a single expiry."""

    def test_single_expiry_runs(self):
        """CVI calibration should succeed on synthetic data."""
        exp = _make_synthetic_smile()
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp])

        assert result.bspline_weights.shape[0] == 1
        assert result.bspline_weights.shape[1] == len(result.breakpoints) + 2

    def test_fit_within_bid_ask(self):
        """Fitted vols should mostly be within bid-ask spread."""
        exp = _make_synthetic_smile(spread=0.01)
        config = CVIConfig(n_knots=9, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp])

        fitted_vols = evaluate_vol(result, exp.strikes, 0)

        # Count how many are within bid-ask
        within = (fitted_vols >= exp.bid_vols - 1e-4) & (fitted_vols <= exp.ask_vols + 1e-4)
        pct_within = np.mean(within)
        # Most should be within (allow some tolerance for regularization)
        assert pct_within >= 0.8, f"Only {pct_within:.0%} within bid-ask"

    def test_variance_positive(self):
        """Fitted variance should be positive everywhere."""
        exp = _make_synthetic_smile()
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp])

        # Evaluate at a wide range of strikes
        z_test = np.linspace(-10, 10, 100)
        k_test = z_test * exp.anchor_atm_vol * np.sqrt(exp.time_to_expiry)
        strikes_test = exp.forward * np.exp(k_test)
        vols = evaluate_vol(result, strikes_test, 0)
        assert np.all(vols > 0), "Negative volatility detected"

    def test_with_butterfly_iteration(self):
        """CVI with butterfly linearization should converge."""
        exp = _make_synthetic_smile()
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=2)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp])

        fitted_vols = evaluate_vol(result, exp.strikes, 0)
        assert np.all(np.isfinite(fitted_vols))


class TestCVIMultiExpiry:
    """Test CVI calibration with multiple expiries."""

    def test_two_expiries(self):
        """Calibration with two expiries should succeed."""
        exp1 = _make_synthetic_smile(T=0.25, atm_vol=0.22, skew=-0.06)
        exp2 = _make_synthetic_smile(T=1.0, atm_vol=0.20, skew=-0.04, convexity=0.015)
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp1, exp2])

        assert result.bspline_weights.shape[0] == 2

    def test_no_calendar_spread_arbitrage(self):
        """Total variance should increase with time at fixed K/F."""
        exp1 = _make_synthetic_smile(T=0.25, atm_vol=0.22, skew=-0.06)
        exp2 = _make_synthetic_smile(T=1.0, atm_vol=0.20, skew=-0.04, convexity=0.015)
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([exp1, exp2])

        # Evaluate total variance at the same K/F ratios
        log_moneyness = np.linspace(-0.3, 0.3, 20)
        strikes = exp1.forward * np.exp(log_moneyness)

        tv1 = evaluate_total_variance(result, strikes, 0)
        tv2 = evaluate_total_variance(result, strikes, 1)

        # Total variance for the longer expiry should be >= shorter expiry
        violations = tv2 < tv1 - 1e-6
        assert not np.any(violations), (
            f"Calendar spread arbitrage at {np.sum(violations)} strikes"
        )


# ---------------------------------------------------------------------------
# Full surface calibration tests (Barchart KO fixtures)
# ---------------------------------------------------------------------------

class TestCVIFullSurface:
    """Test CVI calibration on full Barchart KO surface (22 expiries, 880 options)."""

    @pytest.fixture
    def ko_expiries(self):
        """Load KO Palm Oil expiries from Barchart fixtures (first 5 for speed)."""
        from sdg.market_data.barchart.pipeline import load_from_csv

        with open(FIXTURES / "ko_quotes_20260203.json") as f:
            data = json.load(f)
            forwards = data["forwards"]
            valuation_date = date.fromisoformat(data["valuation_date"])

        expiries = load_from_csv(
            FIXTURES / "ko_options_20260203.csv",
            forwards,
            valuation_date=valuation_date,
            rate=0.03,
            min_strikes=3,
        )
        # Use first 5 expiries for faster tests
        return expiries[:5]

    def test_market_knots_exact_interpolation(self, ko_expiries):
        """With market knots and no calendar constraints, error should be ~0%."""
        config = CVIConfig(
            knot_spacing="market",
            calendar_penalty=-1,  # Disable calendar constraints
            max_iterations=2,
        )
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate(ko_expiries)

        # Check that we calibrated all expiries
        assert len(result.expiries) == len(ko_expiries)

        # Check interpolation error at market strikes for each expiry
        max_errors = []
        for i, exp in enumerate(result.expiries):
            cal_vols = evaluate_vol(result, exp.strikes, i)
            mid_vols = np.where(
                np.isnan(exp.bid_vols) | np.isnan(exp.ask_vols),
                np.nan,
                (exp.bid_vols + exp.ask_vols) / 2.0,
            )
            valid = ~np.isnan(mid_vols)
            if np.any(valid):
                errors = np.abs(cal_vols[valid] - mid_vols[valid])
                max_errors.append(errors.max())

        # With market knots, max error should be very small (< 0.5%)
        overall_max = max(max_errors)
        assert overall_max < 0.005, f"Max error {overall_max:.2%} exceeds 0.5%"

    def test_market_only_knots_low_error(self, ko_expiries):
        """With market_only knots, error should still be low."""
        config = CVIConfig(
            knot_spacing="market_only",
            calendar_penalty=-1,
            max_iterations=2,
        )
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate(ko_expiries)

        # Check interpolation error
        max_errors = []
        for i, exp in enumerate(result.expiries):
            cal_vols = evaluate_vol(result, exp.strikes, i)
            mid_vols = np.where(
                np.isnan(exp.bid_vols) | np.isnan(exp.ask_vols),
                np.nan,
                (exp.bid_vols + exp.ask_vols) / 2.0,
            )
            valid = ~np.isnan(mid_vols)
            if np.any(valid):
                errors = np.abs(cal_vols[valid] - mid_vols[valid])
                max_errors.append(errors.max())

        # With market_only knots, max error should be < 1%
        overall_max = max(max_errors)
        assert overall_max < 0.01, f"Max error {overall_max:.2%} exceeds 1%"
