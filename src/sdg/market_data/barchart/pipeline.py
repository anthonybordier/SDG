"""High-level convenience functions for Barchart data ingestion and calibration."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Callable

from sdg.market_data.barchart.client import fetch_futures_options, fetch_futures_quote
from sdg.market_data.barchart.csv_parser import parse_csv
from sdg.market_data.barchart.transform import transform_to_expiry_list
from sdg.volatility.cvi import CVICalibrator
from sdg.volatility.types import CVIConfig, CVIResult, ExpiryData


def load_from_api(
    root: str,
    *,
    api_key: str | None = None,
    exchange: str | None = None,
    contract: str | None = None,
    valuation_date: date | None = None,
    rate: float | None = None,
    discount_factor_fn: Callable[[float], float] | None = None,
    atm_band: float = 0.0,
    min_strikes: int = 3,
) -> list[ExpiryData]:
    """Fetch option and futures data from Barchart API and transform to ExpiryData.

    Args:
        root: Commodity root symbol (e.g. "KO").
        api_key: Barchart API key.
        exchange: Exchange filter.
        contract: Specific contract month.
        valuation_date: Date for T calculation. Defaults to today.
        rate: Flat rate for discounting.
        discount_factor_fn: Callable T -> df.
        atm_band: ATM band for OTM selection.
        min_strikes: Minimum strikes per expiry.

    Returns:
        List of ExpiryData sorted by time_to_expiry.
    """
    if valuation_date is None:
        valuation_date = date.today()

    options = fetch_futures_options(
        root, api_key=api_key, exchange=exchange, contract=contract,
    )

    # Extract unique underlying symbols for quote lookup
    underlyings = {
        rec["underlyingSymbol"]
        for rec in options
        if rec.get("underlyingSymbol")
    }
    quotes = fetch_futures_quote(sorted(underlyings), api_key=api_key)

    return transform_to_expiry_list(
        options, quotes,
        valuation_date=valuation_date,
        rate=rate,
        discount_factor_fn=discount_factor_fn,
        atm_band=atm_band,
        min_strikes=min_strikes,
    )


def load_from_csv(
    path: str | Path,
    forwards: dict[str, float],
    *,
    valuation_date: date | None = None,
    rate: float | None = None,
    discount_factor_fn: Callable[[float], float] | None = None,
    atm_band: float = 0.0,
    min_strikes: int = 3,
) -> list[ExpiryData]:
    """Parse a Barchart CSV and transform to ExpiryData.

    Args:
        path: Path to Barchart CSV file.
        forwards: Mapping of underlying symbol to forward price.
        valuation_date: Date for T calculation. Defaults to today.
        rate: Flat rate for discounting.
        discount_factor_fn: Callable T -> df.
        atm_band: ATM band for OTM selection.
        min_strikes: Minimum strikes per expiry.

    Returns:
        List of ExpiryData sorted by time_to_expiry.
    """
    if valuation_date is None:
        valuation_date = date.today()

    records = parse_csv(path)

    return transform_to_expiry_list(
        records, forwards,
        valuation_date=valuation_date,
        rate=rate,
        discount_factor_fn=discount_factor_fn,
        atm_band=atm_band,
        min_strikes=min_strikes,
    )


def calibrate_surface(
    root: str | None = None,
    *,
    api_key: str | None = None,
    exchange: str | None = None,
    contract: str | None = None,
    csv_path: str | Path | None = None,
    forwards: dict[str, float] | None = None,
    valuation_date: date | None = None,
    rate: float | None = None,
    discount_factor_fn: Callable[[float], float] | None = None,
    atm_band: float = 0.0,
    min_strikes: int = 3,
    cvi_config: CVIConfig | None = None,
) -> CVIResult:
    """One-call convenience: load data and calibrate a CVI vol surface.

    Detects API vs CSV mode based on which arguments are provided.
    - API mode: root is required, fetches from Barchart OnDemand.
    - CSV mode: csv_path and forwards are required.

    Args:
        root: Commodity root symbol for API mode.
        api_key: Barchart API key for API mode.
        exchange: Exchange filter for API mode.
        contract: Contract filter for API mode.
        csv_path: Path to CSV file for CSV mode.
        forwards: Forward prices for CSV mode.
        valuation_date: Date for T calculation.
        rate: Flat rate for discounting.
        discount_factor_fn: Callable T -> df.
        atm_band: ATM band for OTM selection.
        min_strikes: Minimum strikes per expiry.
        cvi_config: CVI calibration configuration.

    Returns:
        CVIResult from calibration.

    Raises:
        ValueError: If neither API nor CSV source is specified, or if
            no valid expiries are found.
    """
    if csv_path is not None:
        if forwards is None:
            raise ValueError("forwards dict is required for CSV mode")
        expiries = load_from_csv(
            csv_path, forwards,
            valuation_date=valuation_date,
            rate=rate,
            discount_factor_fn=discount_factor_fn,
            atm_band=atm_band,
            min_strikes=min_strikes,
        )
    elif root is not None:
        expiries = load_from_api(
            root,
            api_key=api_key,
            exchange=exchange,
            contract=contract,
            valuation_date=valuation_date,
            rate=rate,
            discount_factor_fn=discount_factor_fn,
            atm_band=atm_band,
            min_strikes=min_strikes,
        )
    else:
        raise ValueError(
            "Specify either root (for API mode) or csv_path (for CSV mode)"
        )

    if not expiries:
        raise ValueError("No valid expiries found after transformation")

    calibrator = CVICalibrator(cvi_config)
    return calibrator.calibrate(expiries)
