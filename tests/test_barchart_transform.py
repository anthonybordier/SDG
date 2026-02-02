"""Tests for barchart transform module (no network required)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from sdg.market_data.barchart.transform import (
    build_arrays_from_records,
    compute_discount_factor,
    group_by_expiry,
    transform_to_expiry_list,
    _parse_date,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def option_records():
    with open(FIXTURES / "barchart_options.json") as f:
        return json.load(f)["results"]


@pytest.fixture
def quote_records():
    with open(FIXTURES / "barchart_quotes.json") as f:
        return json.load(f)["results"]


class TestGroupByExpiry:
    def test_groups_correctly(self, option_records):
        groups = group_by_expiry(option_records)
        assert set(groups.keys()) == {"2025-11-17", "2025-12-17"}
        assert len(groups["2025-11-17"]) == 10
        assert len(groups["2025-12-17"]) == 10

    def test_empty_input(self):
        assert group_by_expiry([]) == {}

    def test_skips_missing_expiry(self):
        records = [{"strike": 100}, {"expirationDate": "2025-01-01", "strike": 100}]
        groups = group_by_expiry(records)
        assert list(groups.keys()) == ["2025-01-01"]


class TestBuildArraysFromRecords:
    def test_produces_correct_arrays(self, option_records):
        groups = group_by_expiry(option_records)
        records = groups["2025-11-17"]
        strikes, call_bid, call_ask, put_bid, put_ask = build_arrays_from_records(records)

        assert len(strikes) == 5
        np.testing.assert_array_equal(strikes, [65.0, 68.0, 72.0, 76.0, 80.0])
        assert np.isfinite(call_bid).all()
        assert np.isfinite(call_ask).all()
        assert np.isfinite(put_bid).all()
        assert np.isfinite(put_ask).all()

    def test_nan_for_missing_bid(self):
        records = [
            {"strike": 100.0, "type": "Call", "bid": None, "ask": 5.0},
        ]
        strikes, call_bid, call_ask, put_bid, put_ask = build_arrays_from_records(records)
        assert len(strikes) == 1
        assert np.isnan(call_bid[0])
        assert call_ask[0] == 5.0
        assert np.isnan(put_bid[0])

    def test_zero_bid_becomes_nan(self):
        records = [
            {"strike": 100.0, "type": "Call", "bid": 0.0, "ask": 5.0},
        ]
        _, call_bid, _, _, _ = build_arrays_from_records(records)
        assert np.isnan(call_bid[0])

    def test_empty_records(self):
        strikes, cb, ca, pb, pa = build_arrays_from_records([])
        assert len(strikes) == 0


class TestComputeDiscountFactor:
    def test_with_callable(self):
        def fn(t):
            return 0.95

        assert compute_discount_factor(1.0, rate=0.05, discount_factor_fn=fn) == 0.95

    def test_with_rate(self):
        df = compute_discount_factor(1.0, rate=0.05)
        np.testing.assert_almost_equal(df, np.exp(-0.05), decimal=10)

    def test_default(self):
        assert compute_discount_factor(1.0) == 1.0


class TestParseDate:
    def test_api_format(self):
        assert _parse_date("2025-11-17") == date(2025, 11, 17)

    def test_csv_format(self):
        assert _parse_date("11/17/2025") == date(2025, 11, 17)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Cannot parse date"):
            _parse_date("17-Nov-2025")


class TestTransformToExpiryList:
    def test_integration(self, option_records, quote_records):
        expiries = transform_to_expiry_list(
            option_records, quote_records,
            valuation_date=date(2025, 6, 1),
        )
        assert len(expiries) == 2
        # Should be sorted by T
        assert expiries[0].time_to_expiry < expiries[1].time_to_expiry
        # Check forward prices
        np.testing.assert_almost_equal(expiries[0].forward, 74.35)
        np.testing.assert_almost_equal(expiries[1].forward, 73.65)

    def test_with_dict_forwards(self, option_records):
        forwards = {"CLZ25": 74.35, "CLF26": 73.65}
        expiries = transform_to_expiry_list(
            option_records, forwards,
            valuation_date=date(2025, 6, 1),
        )
        assert len(expiries) == 2

    def test_filters_few_strikes(self, option_records, quote_records):
        expiries = transform_to_expiry_list(
            option_records, quote_records,
            valuation_date=date(2025, 6, 1),
            min_strikes=100,
        )
        assert len(expiries) == 0

    def test_filters_expired(self, option_records, quote_records):
        expiries = transform_to_expiry_list(
            option_records, quote_records,
            valuation_date=date(2026, 6, 1),
        )
        assert len(expiries) == 0
