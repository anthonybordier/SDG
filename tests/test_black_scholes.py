"""Tests for the shared Black-Scholes formulas."""

import numpy as np

from sdg.core.black_scholes import (
    call_price,
    put_price,
    digital_call_price,
    digital_put_price,
    vega,
)


class TestPutCallParity:
    """Put-call parity: C - P = df * (F - K)."""

    def test_atm(self):
        F, K, vol, T, df = 100.0, 100.0, 0.20, 1.0, 0.95
        strikes = np.array([K])
        vols = np.array([vol])
        c = call_price(F, strikes, vols, T, df)[0]
        p = put_price(F, strikes, vols, T, df)[0]
        np.testing.assert_allclose(c - p, df * (F - K), atol=1e-12)

    def test_itm_otm(self):
        F, vol, T, df = 100.0, 0.25, 0.5, 0.97
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.full(5, vol)
        c = call_price(F, strikes, vols, T, df)
        p = put_price(F, strikes, vols, T, df)
        np.testing.assert_allclose(c - p, df * (F - strikes), atol=1e-10)

    def test_long_dated(self):
        F, vol, T, df = 100.0, 0.30, 5.0, 0.80
        strikes = np.array([60.0, 80.0, 100.0, 120.0, 140.0])
        vols = np.full(5, vol)
        c = call_price(F, strikes, vols, T, df)
        p = put_price(F, strikes, vols, T, df)
        np.testing.assert_allclose(c - p, df * (F - strikes), atol=1e-10)


class TestDigitalOptions:
    """Digital call + digital put = df (cash-or-nothing)."""

    def test_digital_sum_equals_df(self):
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.full(5, vol)
        dc = digital_call_price(F, strikes, vols, T, df)
        dp = digital_put_price(F, strikes, vols, T, df)
        np.testing.assert_allclose(dc + dp, df, atol=1e-12)

    def test_digital_call_monotone(self):
        """Digital call price should decrease with strike."""
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.full(5, vol)
        dc = digital_call_price(F, strikes, vols, T, df)
        assert np.all(np.diff(dc) < 0)


class TestVega:
    """Vega should be positive for all options."""

    def test_vega_positive(self):
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.full(5, vol)
        v = vega(F, strikes, vols, T, df)
        assert np.all(v > 0)

    def test_vega_peaks_atm(self):
        """Vega should be highest near ATM."""
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.full(5, vol)
        v = vega(F, strikes, vols, T, df)
        assert np.argmax(v) == 2  # ATM strike index


class TestATMApproximation:
    """ATM call ~ 0.4 * F * df * vol * sqrt(T) (Brenner-Subrahmanyam)."""

    def test_atm_call_approximation(self):
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([F])
        vols = np.array([vol])
        c = call_price(F, strikes, vols, T, df)[0]
        approx = 0.3989422804 * F * df * vol * np.sqrt(T)  # N'(0) * F * df * vol * sqrt(T)
        np.testing.assert_allclose(c, approx, rtol=0.01)


class TestBoundaryConditions:
    """Test option prices at extreme strikes."""

    def test_deep_itm_call(self):
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([1.0])
        vols = np.array([vol])
        c = call_price(F, strikes, vols, T, df)[0]
        np.testing.assert_allclose(c, df * (F - strikes[0]), rtol=1e-6)

    def test_deep_otm_call(self):
        F, vol, T, df = 100.0, 0.20, 1.0, 0.95
        strikes = np.array([500.0])
        vols = np.array([vol])
        c = call_price(F, strikes, vols, T, df)[0]
        assert c < 1e-10
