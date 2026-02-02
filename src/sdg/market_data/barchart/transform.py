"""Transform raw Barchart records into ExpiryData objects."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
from typing import Callable

import numpy as np

from sdg.market_data.converter import build_expiry_data_from_arrays
from sdg.volatility.types import ExpiryData


def group_by_expiry(records: list[dict]) -> dict[str, list[dict]]:
    """Group option records by expiration date string.

    Args:
        records: Flat list of option record dicts, each containing
            an 'expirationDate' key.

    Returns:
        Dict mapping expiration date string to list of records.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        exp = rec.get("expirationDate")
        if exp is not None:
            groups[str(exp)].append(rec)
    return dict(groups)


def _parse_date(date_str: str) -> date:
    """Parse a date string in either MM/DD/YYYY or YYYY-MM-DD format."""
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: '{date_str}'. Expected MM/DD/YYYY or YYYY-MM-DD.")


def _to_float(val) -> float:
    """Convert a value to float, returning NaN for None or invalid."""
    if val is None:
        return np.nan
    try:
        v = float(val)
        return np.nan if v <= 0 else v
    except (TypeError, ValueError):
        return np.nan


def build_arrays_from_records(
    records: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build strike/price arrays from records for a single expiry.

    Pivots records by strike price, separating calls and puts.

    Args:
        records: Option records for a single expiration date.

    Returns:
        Tuple of (strikes, call_bid, call_ask, put_bid, put_ask) as numpy arrays.
        Missing values are NaN. Zero or negative bids/asks are NaN.
    """
    # Collect data by strike
    strike_data: dict[float, dict] = {}
    for rec in records:
        strike_val = rec.get("strike")
        if strike_val is None:
            continue
        try:
            strike = float(strike_val)
        except (TypeError, ValueError):
            continue

        if strike not in strike_data:
            strike_data[strike] = {
                "call_bid": np.nan, "call_ask": np.nan,
                "put_bid": np.nan, "put_ask": np.nan,
            }

        opt_type = str(rec.get("type", "")).strip().lower()
        if opt_type in ("call", "c"):
            strike_data[strike]["call_bid"] = _to_float(rec.get("bid"))
            strike_data[strike]["call_ask"] = _to_float(rec.get("ask"))
        elif opt_type in ("put", "p"):
            strike_data[strike]["put_bid"] = _to_float(rec.get("bid"))
            strike_data[strike]["put_ask"] = _to_float(rec.get("ask"))

    if not strike_data:
        return (
            np.array([]), np.array([]), np.array([]),
            np.array([]), np.array([]),
        )

    sorted_strikes = sorted(strike_data.keys())
    strikes = np.array(sorted_strikes)
    call_bid = np.array([strike_data[k]["call_bid"] for k in sorted_strikes])
    call_ask = np.array([strike_data[k]["call_ask"] for k in sorted_strikes])
    put_bid = np.array([strike_data[k]["put_bid"] for k in sorted_strikes])
    put_ask = np.array([strike_data[k]["put_ask"] for k in sorted_strikes])

    return strikes, call_bid, call_ask, put_bid, put_ask


def compute_discount_factor(
    T: float,
    rate: float | None = None,
    discount_factor_fn: Callable[[float], float] | None = None,
) -> float:
    """Compute a discount factor for time-to-expiry T.

    Priority:
        1. discount_factor_fn(T) if provided
        2. exp(-rate * T) if rate provided
        3. 1.0 (no discounting)

    Args:
        T: Time to expiry in years.
        rate: Flat continuously compounded rate.
        discount_factor_fn: Callable that maps T to a discount factor.

    Returns:
        Discount factor.
    """
    if discount_factor_fn is not None:
        return discount_factor_fn(T)
    if rate is not None:
        return float(np.exp(-rate * T))
    return 1.0


def transform_to_expiry_list(
    option_records: list[dict],
    futures_quotes: list[dict] | dict[str, float],
    *,
    valuation_date: date,
    rate: float | None = None,
    discount_factor_fn: Callable[[float], float] | None = None,
    atm_band: float = 0.0,
    min_strikes: int = 3,
) -> list[ExpiryData]:
    """Transform raw option records and futures quotes into ExpiryData objects.

    Args:
        option_records: Flat list of option record dicts.
        futures_quotes: Either a list of quote dicts (from API, with 'symbol'
            and 'lastPrice'/'settlement' keys) or a dict mapping underlying
            symbol to forward price (for CSV usage).
        valuation_date: Date for computing time to expiry.
        rate: Flat continuously compounded rate for discounting.
        discount_factor_fn: Callable T -> df (overrides rate).
        atm_band: ATM band for OTM selection in build_expiry_data_from_arrays.
        min_strikes: Minimum number of valid strikes per expiry.

    Returns:
        List of ExpiryData sorted by time_to_expiry.
    """
    # Build forward lookup
    if isinstance(futures_quotes, dict):
        forwards = futures_quotes
    else:
        forwards = _build_forward_lookup(futures_quotes)

    grouped = group_by_expiry(option_records)
    expiries: list[ExpiryData] = []

    for exp_str, records in grouped.items():
        exp_date = _parse_date(exp_str)
        T = (exp_date - valuation_date).days / 365.0
        if T <= 0:
            continue

        # Find forward price for this group
        underlying = _find_underlying(records)
        if underlying is None or underlying not in forwards:
            continue
        forward = forwards[underlying]

        df = compute_discount_factor(T, rate=rate, discount_factor_fn=discount_factor_fn)

        strikes, call_bid, call_ask, put_bid, put_ask = build_arrays_from_records(records)

        # Count valid strikes (at least one side has both bid and ask)
        has_call = np.isfinite(call_bid) & np.isfinite(call_ask)
        has_put = np.isfinite(put_bid) & np.isfinite(put_ask)
        n_valid = int(np.sum(has_call | has_put))
        if n_valid < min_strikes:
            continue

        expiry_data = build_expiry_data_from_arrays(
            T, forward, df,
            strikes, call_bid, call_ask, put_bid, put_ask,
            atm_band=atm_band,
        )
        expiries.append(expiry_data)

    expiries.sort(key=lambda e: e.time_to_expiry)
    return expiries


def _build_forward_lookup(quotes: list[dict]) -> dict[str, float]:
    """Build a symbol -> forward price mapping from API quote records."""
    lookup: dict[str, float] = {}
    for q in quotes:
        sym = q.get("symbol")
        if sym is None:
            continue
        # Prefer settlement, fall back to lastPrice
        price = q.get("settlement") or q.get("lastPrice")
        if price is not None:
            try:
                lookup[sym] = float(price)
            except (TypeError, ValueError):
                continue
    return lookup


def _find_underlying(records: list[dict]) -> str | None:
    """Extract the underlying symbol from a group of option records."""
    for rec in records:
        underlying = rec.get("underlyingSymbol")
        if underlying is not None:
            return str(underlying)
    return None
