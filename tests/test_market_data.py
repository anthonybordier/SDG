"""Tests for market data conversion (raw prices -> ExpiryData)."""

import numpy as np

from sdg.core.black_scholes import call_price, put_price
from sdg.market_data.converter import (
    RawExpiryData,
    RawOptionQuote,
    build_expiry_data_from_arrays,
    convert_to_expiry_data,
)
from sdg.volatility.cvi import CVICalibrator, evaluate_vol
from sdg.volatility.types import CVIConfig


def _make_prices_from_vols(
    F: float,
    T: float,
    df: float,
    strikes: np.ndarray,
    bid_vols: np.ndarray,
    ask_vols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate call/put bid/ask prices from known vols."""
    call_bid = call_price(F, strikes, bid_vols, T, df)
    call_ask = call_price(F, strikes, ask_vols, T, df)
    put_bid = put_price(F, strikes, bid_vols, T, df)
    put_ask = put_price(F, strikes, ask_vols, T, df)
    return call_bid, call_ask, put_bid, put_ask


class TestCallsOnly:
    """Test conversion with only call prices."""

    def test_calls_only(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([90.0, 100.0, 110.0])
        bid_vols = np.array([0.19, 0.195, 0.21])
        ask_vols = np.array([0.21, 0.205, 0.23])

        call_bid = call_price(F, strikes, bid_vols, T, df)
        call_ask = call_price(F, strikes, ask_vols, T, df)
        put_bid = np.full(3, np.nan)
        put_ask = np.full(3, np.nan)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        np.testing.assert_allclose(expiry.bid_vols, bid_vols, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, ask_vols, atol=1e-6)


class TestPutsOnly:
    """Test conversion with only put prices."""

    def test_puts_only(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([90.0, 100.0, 110.0])
        bid_vols = np.array([0.19, 0.195, 0.21])
        ask_vols = np.array([0.21, 0.205, 0.23])

        call_bid = np.full(3, np.nan)
        call_ask = np.full(3, np.nan)
        put_bid = put_price(F, strikes, bid_vols, T, df)
        put_ask = put_price(F, strikes, ask_vols, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        np.testing.assert_allclose(expiry.bid_vols, bid_vols, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, ask_vols, atol=1e-6)


class TestCombined:
    """Test conversion with both call and put prices."""

    def test_picks_tighter_spread(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([100.0])

        # Call spread: tighter (0.01 wide)
        call_bid_vol = np.array([0.195])
        call_ask_vol = np.array([0.205])
        # Put spread: wider (0.02 wide)
        put_bid_vol = np.array([0.19])
        put_ask_vol = np.array([0.21])

        call_bid = call_price(F, strikes, call_bid_vol, T, df)
        call_ask = call_price(F, strikes, call_ask_vol, T, df)
        put_bid = put_price(F, strikes, put_bid_vol, T, df)
        put_ask = put_price(F, strikes, put_ask_vol, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        # Should pick call vols (tighter spread)
        np.testing.assert_allclose(expiry.bid_vols, call_bid_vol, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, call_ask_vol, atol=1e-6)


class TestNaNHandling:
    """Test NaN handling in converter."""

    def test_all_nan_gives_nan(self):
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([100.0])
        nans = np.full(1, np.nan)
        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, nans, nans, nans, nans,
        )
        assert np.isnan(expiry.bid_vols[0])
        assert np.isnan(expiry.ask_vols[0])

    def test_partial_nan(self):
        """Some strikes have data, others don't."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([90.0, 100.0, 110.0])
        bid_vols = np.array([0.19, 0.195, 0.21])
        ask_vols = np.array([0.21, 0.205, 0.23])

        call_bid = call_price(F, strikes, bid_vols, T, df)
        call_ask = call_price(F, strikes, ask_vols, T, df)
        # Middle strike has no data
        call_bid[1] = np.nan
        call_ask[1] = np.nan
        put_bid = np.full(3, np.nan)
        put_ask = np.full(3, np.nan)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        assert np.isfinite(expiry.bid_vols[0])
        assert np.isnan(expiry.bid_vols[1])
        assert np.isfinite(expiry.bid_vols[2])


class TestOTMSelection:
    """Test that OTM options are preferred by default."""

    def test_otm_puts_for_low_strikes(self):
        """K < F should use put IVs when both are available."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([85.0])

        # Same vol for both sides so we can check which side was chosen
        # Put spread is tighter -> but OTM rule should pick put regardless
        put_bid_vol = np.array([0.22])
        put_ask_vol = np.array([0.24])
        call_bid_vol = np.array([0.21])
        call_ask_vol = np.array([0.23])

        call_bid = call_price(F, strikes, call_bid_vol, T, df)
        call_ask = call_price(F, strikes, call_ask_vol, T, df)
        put_bid = put_price(F, strikes, put_bid_vol, T, df)
        put_ask = put_price(F, strikes, put_ask_vol, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        # Should use put vols (OTM for K < F)
        np.testing.assert_allclose(expiry.bid_vols, put_bid_vol, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, put_ask_vol, atol=1e-6)

    def test_otm_calls_for_high_strikes(self):
        """K > F should use call IVs when both are available."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([115.0])

        put_bid_vol = np.array([0.22])
        put_ask_vol = np.array([0.24])
        call_bid_vol = np.array([0.21])
        call_ask_vol = np.array([0.23])

        call_bid = call_price(F, strikes, call_bid_vol, T, df)
        call_ask = call_price(F, strikes, call_ask_vol, T, df)
        put_bid = put_price(F, strikes, put_bid_vol, T, df)
        put_ask = put_price(F, strikes, put_ask_vol, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        # Should use call vols (OTM for K > F)
        np.testing.assert_allclose(expiry.bid_vols, call_bid_vol, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, call_ask_vol, atol=1e-6)

    def test_fallback_to_itm_when_otm_missing(self):
        """If OTM side is missing, fall back to ITM side."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([85.0])  # OTM put, but only call available

        call_bid_vol = np.array([0.21])
        call_ask_vol = np.array([0.23])

        call_bid = call_price(F, strikes, call_bid_vol, T, df)
        call_ask = call_price(F, strikes, call_ask_vol, T, df)
        put_bid = np.full(1, np.nan)
        put_ask = np.full(1, np.nan)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        # Should fall back to call vols
        np.testing.assert_allclose(expiry.bid_vols, call_bid_vol, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, call_ask_vol, atol=1e-6)

    def test_atm_band_picks_tighter(self):
        """ATM strikes should pick tighter spread when atm_band covers them."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([100.0])

        # Call spread tighter
        call_bid_vol = np.array([0.195])
        call_ask_vol = np.array([0.205])
        # Put spread wider
        put_bid_vol = np.array([0.19])
        put_ask_vol = np.array([0.21])

        call_bid = call_price(F, strikes, call_bid_vol, T, df)
        call_ask = call_price(F, strikes, call_ask_vol, T, df)
        put_bid = put_price(F, strikes, put_bid_vol, T, df)
        put_ask = put_price(F, strikes, put_ask_vol, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
            atm_band=0.01,
        )
        # ATM -> tighter spread -> call
        np.testing.assert_allclose(expiry.bid_vols, call_bid_vol, atol=1e-6)
        np.testing.assert_allclose(expiry.ask_vols, call_ask_vol, atol=1e-6)

    def test_full_smile_otm_selection(self):
        """Full smile with both sides: verify OTM selection across strikes."""
        F, T, df = 100.0, 0.5, 0.97
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        # Different vols for calls vs puts to distinguish which was chosen
        call_vols = np.array([0.30, 0.25, 0.20, 0.22, 0.26])
        put_vols = np.array([0.31, 0.26, 0.21, 0.23, 0.27])

        call_bid = call_price(F, strikes, call_vols - 0.005, T, df)
        call_ask = call_price(F, strikes, call_vols + 0.005, T, df)
        put_bid = put_price(F, strikes, put_vols - 0.005, T, df)
        put_ask = put_price(F, strikes, put_vols + 0.005, T, df)

        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        # K=80,90: OTM puts -> put_vols
        np.testing.assert_allclose(expiry.bid_vols[0], put_vols[0] - 0.005, atol=1e-5)
        np.testing.assert_allclose(expiry.bid_vols[1], put_vols[1] - 0.005, atol=1e-5)
        # K=110,120: OTM calls -> call_vols
        np.testing.assert_allclose(expiry.bid_vols[3], call_vols[3] - 0.005, atol=1e-5)
        np.testing.assert_allclose(expiry.bid_vols[4], call_vols[4] - 0.005, atol=1e-5)


class TestObjectOrientedInput:
    """Test the RawExpiryData / convert_to_expiry_data path."""

    def test_convert_to_expiry_data(self):
        F, T, df = 100.0, 0.5, 0.97
        vol = 0.20
        strikes = [95.0, 100.0, 105.0]
        vols = np.array([vol, vol, vol])
        call_bid_prices = call_price(F, np.array(strikes), vols - 0.005, T, df)
        call_ask_prices = call_price(F, np.array(strikes), vols + 0.005, T, df)

        quotes = [
            RawOptionQuote(
                strike=strikes[i],
                call_bid=float(call_bid_prices[i]),
                call_ask=float(call_ask_prices[i]),
            )
            for i in range(3)
        ]
        raw = RawExpiryData(
            time_to_expiry=T, forward=F, discount_factor=df, quotes=quotes,
        )
        expiry = convert_to_expiry_data(raw)
        assert len(expiry.strikes) == 3
        assert np.all(np.isfinite(expiry.bid_vols))
        assert np.all(np.isfinite(expiry.ask_vols))


class TestRoundTripWithCVI:
    """End-to-end: raw prices -> ExpiryData -> CVI -> evaluate."""

    def test_prices_to_surface_round_trip(self):
        F, T, df = 100.0, 0.25, np.exp(-0.05 * 0.25)
        n_strikes = 15
        sigma_star = 0.20

        # Generate synthetic smile
        z_grid = np.linspace(-2.5, 2.5, n_strikes)
        k_grid = z_grid * sigma_star * np.sqrt(T)
        strikes = F * np.exp(k_grid)

        # Smile: sigma(z) = 0.20 * sqrt(1 + 0.02 * z^2)
        true_vols = sigma_star * np.sqrt(1.0 + 0.02 * z_grid**2)
        bid_vols = true_vols - 0.003
        ask_vols = true_vols + 0.003

        call_bid, call_ask, put_bid, put_ask = _make_prices_from_vols(
            F, T, df, strikes, bid_vols, ask_vols,
        )

        # Convert prices to ExpiryData
        expiry = build_expiry_data_from_arrays(
            T, F, df, strikes, call_bid, call_ask, put_bid, put_ask,
        )
        assert np.all(np.isfinite(expiry.bid_vols))
        assert np.all(np.isfinite(expiry.ask_vols))

        # Calibrate CVI
        config = CVIConfig(n_knots=7, z_range=6.0, max_iterations=1)
        calibrator = CVICalibrator(config)
        result = calibrator.calibrate([expiry])

        # Evaluate fitted vols
        fitted_vols = evaluate_vol(result, strikes, 0)

        # Fitted vols should be close to true vols
        np.testing.assert_allclose(fitted_vols, true_vols, atol=0.01)
