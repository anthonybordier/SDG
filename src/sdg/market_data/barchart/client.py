"""Barchart OnDemand API client using stdlib urllib."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
import urllib.parse


BASE_URL = "https://ondemand.websol.barchart.com"


class BarchartAPIError(Exception):
    """Raised when the Barchart API returns an error."""


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve API key from parameter or environment variable."""
    key = api_key or os.environ.get("BARCHART_API_KEY")
    if not key:
        raise ValueError(
            "Barchart API key required. Pass api_key parameter or set "
            "BARCHART_API_KEY environment variable."
        )
    return key


def _api_request(endpoint: str, params: dict) -> dict:
    """Make a GET request to the Barchart OnDemand API.

    Args:
        endpoint: API endpoint (e.g. "getFuturesOptions.json").
        params: Query parameters.

    Returns:
        Parsed JSON response.

    Raises:
        BarchartAPIError: On HTTP errors or API-level errors.
    """
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise BarchartAPIError(
            f"HTTP {exc.code} from Barchart API: {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise BarchartAPIError(
            f"Connection error: {exc.reason}"
        ) from exc

    if data.get("status", {}).get("code") not in (None, 200):
        msg = data.get("status", {}).get("message", "Unknown API error")
        raise BarchartAPIError(f"Barchart API error: {msg}")

    return data


def fetch_futures_options(
    root: str,
    *,
    api_key: str | None = None,
    exchange: str | None = None,
    contract: str | None = None,
    fields: str = "bid,ask,strike,expirationDate,type,underlyingSymbol",
) -> list[dict]:
    """Fetch futures options data from Barchart OnDemand API.

    Args:
        root: Commodity root symbol (e.g. "CL").
        api_key: Barchart API key. Falls back to BARCHART_API_KEY env var.
        exchange: Exchange filter (e.g. "CME").
        contract: Specific contract month (e.g. "CLZ24").
        fields: Comma-separated fields to request.

    Returns:
        List of option record dicts from the API response.
    """
    key = _resolve_api_key(api_key)
    params: dict[str, str] = {
        "apikey": key,
        "root": root,
        "fields": fields,
    }
    if exchange:
        params["exchange"] = exchange
    if contract:
        params["contract"] = contract

    data = _api_request("getFuturesOptions.json", params)
    return data.get("results", [])


def fetch_futures_quote(
    symbols: list[str],
    *,
    api_key: str | None = None,
    fields: str = "symbol,lastPrice,settlement",
) -> list[dict]:
    """Fetch futures quote data from Barchart OnDemand API.

    Args:
        symbols: List of futures symbols (e.g. ["CLZ24", "CLF25"]).
        api_key: Barchart API key. Falls back to BARCHART_API_KEY env var.
        fields: Comma-separated fields to request.

    Returns:
        List of quote record dicts from the API response.
    """
    key = _resolve_api_key(api_key)
    params: dict[str, str] = {
        "apikey": key,
        "symbols": ",".join(symbols),
        "fields": fields,
    }

    data = _api_request("getQuote.json", params)
    return data.get("results", [])
