"""Tests for the implied volatility solver."""

import numpy as np

from sdg.core.black_scholes import call_price, put_price
from sdg.market_data.implied_vol import OptionType, implied_vol


class TestRoundTrip:
    """BS price -> IV -> check vol matches original."""

    def test_call_round_trip(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        true_vols = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        prices = call_price(F, strikes, true_vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-8)

    def test_put_round_trip(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        true_vols = np.array([0.25, 0.22, 0.20, 0.22, 0.25])
        prices = put_price(F, strikes, true_vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.PUT)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-8)

    def test_flat_vol_surface(self):
        F, T, df, vol = 100.0, 1.0, 0.95, 0.20
        strikes = np.linspace(70, 130, 13)
        vols = np.full_like(strikes, vol)
        prices = call_price(F, strikes, vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        np.testing.assert_allclose(recovered, vol, atol=1e-8)

    def test_high_vol_round_trip(self):
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([100.0])
        true_vols = np.array([1.5])
        prices = call_price(F, strikes, true_vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-6)

    def test_short_expiry(self):
        F, T, df = 100.0, 0.01, 0.9999
        strikes = np.array([99.0, 100.0, 101.0])
        true_vols = np.array([0.20, 0.20, 0.20])
        prices = call_price(F, strikes, true_vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-6)


class TestEdgeCases:
    """Edge cases for the IV solver."""

    def test_zero_price_returns_nan(self):
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([100.0])
        prices = np.array([0.0])
        result = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        assert np.isnan(result[0])

    def test_negative_price_returns_nan(self):
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([100.0])
        prices = np.array([-1.0])
        result = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        assert np.isnan(result[0])

    def test_nan_price_returns_nan(self):
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([100.0])
        prices = np.array([np.nan])
        result = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        assert np.isnan(result[0])

    def test_below_intrinsic_returns_nan(self):
        """Price below intrinsic value should return NaN."""
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([80.0])
        # Intrinsic for call = df * (F - K) = 0.95 * 20 = 19
        prices = np.array([10.0])
        result = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        assert np.isnan(result[0])

    def test_deep_otm_call(self):
        """Very deep OTM call should still recover vol if price is valid."""
        F, T, df = 100.0, 0.25, 0.99
        strikes = np.array([150.0])
        true_vols = np.array([0.30])
        prices = call_price(F, strikes, true_vols, T, df)
        recovered = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        np.testing.assert_allclose(recovered, true_vols, atol=1e-6)

    def test_mixed_valid_invalid(self):
        """Array with mix of valid and invalid prices."""
        F, T, df = 100.0, 1.0, 0.95
        strikes = np.array([90.0, 100.0, 110.0])
        true_vols = np.array([0.22, 0.20, 0.22])
        valid_prices = call_price(F, strikes, true_vols, T, df)
        prices = np.array([np.nan, valid_prices[1], -1.0])
        result = implied_vol(prices, F, strikes, T, df, OptionType.CALL)
        assert np.isnan(result[0])
        np.testing.assert_allclose(result[1], true_vols[1], atol=1e-8)
        assert np.isnan(result[2])
