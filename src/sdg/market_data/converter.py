"""Convert raw option prices to ExpiryData with implied volatilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdg.market_data.implied_vol import OptionType, implied_vol
from sdg.volatility.types import ExpiryData


@dataclass
class RawOptionQuote:
    """A single option quote with bid/ask prices."""

    strike: float
    call_bid: float | None = None
    call_ask: float | None = None
    put_bid: float | None = None
    put_ask: float | None = None


@dataclass
class RawExpiryData:
    """Raw market data for a single expiry (prices, not vols)."""

    time_to_expiry: float
    forward: float
    discount_factor: float
    quotes: list[RawOptionQuote]


def build_expiry_data_from_arrays(
    T: float,
    forward: float,
    df: float,
    strikes: np.ndarray,
    call_bid: np.ndarray,
    call_ask: np.ndarray,
    put_bid: np.ndarray,
    put_ask: np.ndarray,
    *,
    atm_band: float = 0.0,
) -> ExpiryData:
    """Build ExpiryData from arrays of option prices.

    Uses OTM selection by default: puts for K < F, calls for K > F.
    Near ATM (|log(K/F)| <= atm_band), both sides are considered and the
    one with the tighter IV spread is chosen.

    When only one side (call or put) has valid quotes at a strike, that
    side is used regardless of moneyness.

    Args:
        T: Time to expiry in years.
        forward: Forward price.
        df: Discount factor.
        strikes: Strike prices.
        call_bid: Call bid prices (NaN where missing).
        call_ask: Call ask prices (NaN where missing).
        put_bid: Put bid prices (NaN where missing).
        put_ask: Put ask prices (NaN where missing).
        atm_band: Half-width of the ATM band in log-moneyness.
            Strikes with |log(K/F)| <= atm_band use best-spread selection
            instead of strict OTM preference.  Default 0.0 means only the
            exact ATM strike (K == F) gets best-spread treatment.

    Returns:
        ExpiryData with bid_vols and ask_vols populated.
    """
    strikes = np.asarray(strikes, dtype=float)
    call_bid = np.asarray(call_bid, dtype=float)
    call_ask = np.asarray(call_ask, dtype=float)
    put_bid = np.asarray(put_bid, dtype=float)
    put_ask = np.asarray(put_ask, dtype=float)

    n = len(strikes)
    log_moneyness = np.log(strikes / forward)

    # Compute call IVs
    call_bid_iv = implied_vol(call_bid, forward, strikes, T, df, OptionType.CALL)
    call_ask_iv = implied_vol(call_ask, forward, strikes, T, df, OptionType.CALL)

    # Compute put IVs
    put_bid_iv = implied_vol(put_bid, forward, strikes, T, df, OptionType.PUT)
    put_ask_iv = implied_vol(put_ask, forward, strikes, T, df, OptionType.PUT)

    bid_vols = np.full(n, np.nan)
    ask_vols = np.full(n, np.nan)

    for i in range(n):
        has_call = np.isfinite(call_bid_iv[i]) and np.isfinite(call_ask_iv[i])
        has_put = np.isfinite(put_bid_iv[i]) and np.isfinite(put_ask_iv[i])

        is_atm = abs(log_moneyness[i]) <= atm_band
        is_otm_call = log_moneyness[i] > atm_band   # K > F
        is_otm_put = log_moneyness[i] < -atm_band    # K < F

        if is_atm and has_call and has_put:
            # ATM zone: pick tighter spread
            call_spread = call_ask_iv[i] - call_bid_iv[i]
            put_spread = put_ask_iv[i] - put_bid_iv[i]
            if call_spread <= put_spread:
                bid_vols[i] = call_bid_iv[i]
                ask_vols[i] = call_ask_iv[i]
            else:
                bid_vols[i] = put_bid_iv[i]
                ask_vols[i] = put_ask_iv[i]
        elif is_otm_call or (is_atm and not has_put):
            # OTM call region, or ATM with only call available
            if has_call:
                bid_vols[i] = call_bid_iv[i]
                ask_vols[i] = call_ask_iv[i]
            elif has_put:
                # Fallback: no call data, use put even though ITM
                bid_vols[i] = put_bid_iv[i]
                ask_vols[i] = put_ask_iv[i]
        elif is_otm_put or (is_atm and not has_call):
            # OTM put region, or ATM with only put available
            if has_put:
                bid_vols[i] = put_bid_iv[i]
                ask_vols[i] = put_ask_iv[i]
            elif has_call:
                # Fallback: no put data, use call even though ITM
                bid_vols[i] = call_bid_iv[i]
                ask_vols[i] = call_ask_iv[i]

    return ExpiryData(
        time_to_expiry=T,
        forward=forward,
        discount_factor=df,
        strikes=strikes,
        bid_vols=bid_vols,
        ask_vols=ask_vols,
    )


def convert_to_expiry_data(raw: RawExpiryData) -> ExpiryData:
    """Convert RawExpiryData (object-oriented input) to ExpiryData.

    Args:
        raw: Raw market data with option quotes.

    Returns:
        ExpiryData with implied volatilities.
    """
    strikes = np.array([q.strike for q in raw.quotes])
    call_bid = np.array([q.call_bid if q.call_bid is not None else np.nan for q in raw.quotes])
    call_ask = np.array([q.call_ask if q.call_ask is not None else np.nan for q in raw.quotes])
    put_bid = np.array([q.put_bid if q.put_bid is not None else np.nan for q in raw.quotes])
    put_ask = np.array([q.put_ask if q.put_ask is not None else np.nan for q in raw.quotes])

    return build_expiry_data_from_arrays(
        raw.time_to_expiry, raw.forward, raw.discount_factor,
        strikes, call_bid, call_ask, put_bid, put_ask,
    )
