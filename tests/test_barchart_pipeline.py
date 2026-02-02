"""Tests for barchart pipeline (end-to-end with mocked API)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from sdg.market_data.barchart.pipeline import (
    calibrate_surface,
    load_from_api,
    load_from_csv,
)
from sdg.volatility.types import CVIConfig, CVIResult, ExpiryData

FIXTURES = Path(__file__).parent / "fixtures"


def _load_fixture(name):
    with open(FIXTURES / name) as f:
        return json.load(f)


@pytest.fixture
def mock_options():
    return _load_fixture("barchart_options.json")["results"]


@pytest.fixture
def mock_quotes():
    return _load_fixture("barchart_quotes.json")["results"]


class TestLoadFromApi:
    @patch("sdg.market_data.barchart.pipeline.fetch_futures_quote")
    @patch("sdg.market_data.barchart.pipeline.fetch_futures_options")
    def test_returns_expiry_data(self, mock_opts, mock_quotes_fn, mock_options, mock_quotes):
        mock_opts.return_value = mock_options
        mock_quotes_fn.return_value = mock_quotes

        expiries = load_from_api(
            "CL", api_key="test-key", valuation_date=date(2025, 6, 1),
        )
        assert len(expiries) == 2
        assert all(isinstance(e, ExpiryData) for e in expiries)
        assert expiries[0].time_to_expiry < expiries[1].time_to_expiry


class TestLoadFromCsv:
    def test_returns_expiry_data(self):
        forwards = {"CLZ25": 74.35, "CLF26": 73.65}
        expiries = load_from_csv(
            FIXTURES / "barchart_options.csv",
            forwards,
            valuation_date=date(2025, 6, 1),
        )
        assert len(expiries) == 2
        assert all(isinstance(e, ExpiryData) for e in expiries)


class TestCalibrateSurface:
    @patch("sdg.market_data.barchart.pipeline.fetch_futures_quote")
    @patch("sdg.market_data.barchart.pipeline.fetch_futures_options")
    def test_api_mode(self, mock_opts, mock_quotes_fn, mock_options, mock_quotes):
        mock_opts.return_value = mock_options
        mock_quotes_fn.return_value = mock_quotes

        result = calibrate_surface(
            "CL",
            api_key="test-key",
            valuation_date=date(2025, 6, 1),
            cvi_config=CVIConfig(n_knots=7, max_iterations=1),
        )
        assert isinstance(result, CVIResult)
        assert len(result.expiries) == 2

    def test_csv_mode(self):
        forwards = {"CLZ25": 74.35, "CLF26": 73.65}
        result = calibrate_surface(
            csv_path=FIXTURES / "barchart_options.csv",
            forwards=forwards,
            valuation_date=date(2025, 6, 1),
            cvi_config=CVIConfig(n_knots=7, max_iterations=1),
        )
        assert isinstance(result, CVIResult)
        assert len(result.expiries) == 2

    def test_no_source_raises(self):
        with pytest.raises(ValueError, match="Specify either root"):
            calibrate_surface()

    def test_csv_without_forwards_raises(self):
        with pytest.raises(ValueError, match="forwards dict is required"):
            calibrate_surface(csv_path="some.csv")

    @patch("sdg.market_data.barchart.pipeline.fetch_futures_quote")
    @patch("sdg.market_data.barchart.pipeline.fetch_futures_options")
    def test_no_valid_expiries_raises(self, mock_opts, mock_quotes_fn):
        mock_opts.return_value = []
        mock_quotes_fn.return_value = []

        with pytest.raises(ValueError, match="No valid expiries"):
            calibrate_surface(
                "CL",
                api_key="test-key",
                valuation_date=date(2025, 6, 1),
            )
