"""Tests for the option pricing module."""

import numpy as np

from sdg.core.black_scholes import call_price as bs_call, put_price as bs_put
from sdg.market_data.converter import build_expiry_data_from_arrays
from sdg.pricing.pricer import (
    OptionKind,
    price_option,
    price_vanilla_call,
    price_vanilla_put,
    price_digital_call,
    price_digital_put,
)
from sdg.volatility.cvi import CVICalibrator
from sdg.volatility.types import CVIConfig, ExpiryData


def _calibrate_flat_surface(
    vol: float = 0.20,
    F: float = 100.0,
    T: float = 0.25,
) -> tuple:
    """Calibrate a CVI surface from a flat vol smile.

    Returns (result, expiry) for testing.
    """
    df = np.exp(-0.05 * T)
    sigma_star = vol
    n_strikes = 21
    z_grid = np.linspace(-3.0, 3.0, n_strikes)
    k_grid = z_grid * sigma_star * np.sqrt(T)
    strikes = F * np.exp(k_grid)

    bid_vols = np.full(n_strikes, vol - 0.002)
    ask_vols = np.full(n_strikes, vol + 0.002)

    expiry = ExpiryData(
        time_to_expiry=T,
        forward=F,
        discount_factor=df,
        strikes=strikes,
        bid_vols=bid_vols,
        ask_vols=ask_vols,
        anchor_atm_vol=vol,
    )

    config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
    calibrator = CVICalibrator(config)
    result = calibrator.calibrate([expiry])
    return result, expiry


class TestPutCallParityFromSurface:
    """C(K) - P(K) = df * (F - K) using surface vols."""

    def test_put_call_parity(self):
        result, expiry = _calibrate_flat_surface()
        strikes = np.linspace(85, 115, 11)
        calls = price_vanilla_call(result, strikes, 0)
        puts = price_vanilla_put(result, strikes, 0)
        expected = expiry.discount_factor * (expiry.forward - strikes)
        np.testing.assert_allclose(calls - puts, expected, atol=1e-6)


class TestDigitalSumFromSurface:
    """DC(K) + DP(K) = df using surface vols."""

    def test_digital_sum_equals_df(self):
        result, expiry = _calibrate_flat_surface()
        strikes = np.linspace(85, 115, 11)
        dc = price_digital_call(result, strikes, 0)
        dp = price_digital_put(result, strikes, 0)
        np.testing.assert_allclose(dc + dp, expiry.discount_factor, atol=1e-6)


class TestPriceOptionDispatch:
    """price_option should dispatch to the correct formula."""

    def test_vanilla_call_matches_convenience(self):
        result, _ = _calibrate_flat_surface()
        strikes = np.array([95.0, 100.0, 105.0])
        p1 = price_option(result, strikes, 0, OptionKind.VANILLA_CALL)
        p2 = price_vanilla_call(result, strikes, 0)
        np.testing.assert_allclose(p1, p2, atol=1e-14)

    def test_digital_put_matches_convenience(self):
        result, _ = _calibrate_flat_surface()
        strikes = np.array([95.0, 100.0, 105.0])
        p1 = price_option(result, strikes, 0, OptionKind.DIGITAL_PUT)
        p2 = price_digital_put(result, strikes, 0)
        np.testing.assert_allclose(p1, p2, atol=1e-14)


class TestEndToEnd:
    """End-to-end: raw prices -> CVI calibration -> reprice."""

    def test_reprice_close_to_input(self):
        F, T = 100.0, 0.25
        df = np.exp(-0.05 * T)
        sigma_star = 0.20
        n_strikes = 15

        z_grid = np.linspace(-2.5, 2.5, n_strikes)
        k_grid = z_grid * sigma_star * np.sqrt(T)
        strikes = F * np.exp(k_grid)

        true_vols = sigma_star * np.sqrt(1.0 + 0.02 * z_grid**2)
        bid_vols = true_vols - 0.003
        ask_vols = true_vols + 0.003

        # Generate prices
        call_bid = bs_call(F, strikes, bid_vols, T, df)
        call_ask = bs_call(F, strikes, ask_vols, T, df)
        put_bid = bs_put(F, strikes, bid_vols, T, df)
        put_ask = bs_put(F, strikes, ask_vols, T, df)

        # Convert to ExpiryData
        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )

        # Calibrate
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([expiry])

        # Reprice calls using surface
        repriced = price_vanilla_call(result, strikes, 0)
        mid_prices = (call_bid + call_ask) / 2.0

        # Repriced should be close to mid-market
        np.testing.assert_allclose(repriced, mid_prices, rtol=0.05)

    def test_call_prices_positive(self):
        result, expiry = _calibrate_flat_surface()
        strikes = np.linspace(70, 130, 21)
        calls = price_vanilla_call(result, strikes, 0)
        assert np.all(calls > 0)
