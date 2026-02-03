"""Tests for barchart API client (mocked HTTP)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sdg.market_data.barchart.client import (
    BarchartAPIError,
    fetch_futures_options,
    fetch_futures_quote,
    _resolve_api_key,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestResolveAPIKey:
    def test_explicit_key(self):
        assert _resolve_api_key("my-key") == "my-key"

    def test_env_var(self, monkeypatch):
        monkeypatch.setenv("BARCHART_API_KEY", "env-key")
        assert _resolve_api_key(None) == "env-key"

    def test_missing_raises(self, monkeypatch):
        monkeypatch.delenv("BARCHART_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            _resolve_api_key(None)


def _mock_urlopen(fixture_name):
    """Create a mock for urllib.request.urlopen that returns fixture data."""
    fixture_path = FIXTURES / fixture_name
    data = fixture_path.read_bytes()
    mock_resp = MagicMock()
    mock_resp.read.return_value = data
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestFetchFuturesOptions:
    @patch("sdg.market_data.barchart.client.urllib.request.urlopen")
    def test_success(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen("barchart_options.json")
        results = fetch_futures_options("KO", api_key="test-key")
        assert len(results) == 20
        assert results[0]["strike"] == 3800.0
        mock_urlopen_fn.assert_called_once()

    @patch("sdg.market_data.barchart.client.urllib.request.urlopen")
    def test_api_key_in_url(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen("barchart_options.json")
        fetch_futures_options("KO", api_key="my-secret-key")
        call_args = mock_urlopen_fn.call_args
        request = call_args[0][0]
        assert "my-secret-key" in request.full_url

    @patch("sdg.market_data.barchart.client.urllib.request.urlopen")
    def test_http_error(self, mock_urlopen_fn):
        import urllib.error
        mock_urlopen_fn.side_effect = urllib.error.HTTPError(
            "http://example.com", 403, "Forbidden", {}, None
        )
        with pytest.raises(BarchartAPIError, match="HTTP 403"):
            fetch_futures_options("KO", api_key="test-key")

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("BARCHART_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key required"):
            fetch_futures_options("KO")


class TestFetchFuturesQuote:
    @patch("sdg.market_data.barchart.client.urllib.request.urlopen")
    def test_success(self, mock_urlopen_fn):
        mock_urlopen_fn.return_value = _mock_urlopen("barchart_quotes.json")
        results = fetch_futures_quote(["KOZ25", "KOF26"], api_key="test-key")
        assert len(results) == 2
        assert results[0]["symbol"] == "KOZ25"
        assert results[0]["settlement"] == 4200.0
