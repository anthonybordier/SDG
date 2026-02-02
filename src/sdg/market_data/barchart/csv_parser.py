"""Parse Barchart CSV downloads into normalized option records."""

from __future__ import annotations

import csv
from pathlib import Path

# Mapping from Barchart CSV column headers to normalized keys.
# The normalized keys match the API response format so transform.py
# can handle both sources uniformly.
COLUMN_MAP: dict[str, str] = {
    "Symbol": "symbol",
    "Strike": "strike",
    "Strike Price": "strike",
    "Bid": "bid",
    "Ask": "ask",
    "Type": "type",
    "Option Type": "type",
    "Expiration Date": "expirationDate",
    "Expiration": "expirationDate",
    "Underlying Symbol": "underlyingSymbol",
    "Underlying": "underlyingSymbol",
}

_MISSING_VALUES = {"N/A", "n/a", "-", ""}


def _clean_value(value: str) -> str | float | None:
    """Clean a single CSV cell value.

    Returns None for missing/empty values, float for numeric strings,
    and the original string otherwise.
    """
    value = value.strip()
    if value in _MISSING_VALUES:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def parse_csv(path: str | Path) -> list[dict]:
    """Parse a Barchart CSV download into normalized option records.

    The output dicts use the same keys as the API response
    (strike, bid, ask, type, expirationDate, underlyingSymbol)
    so that transform.py can process both sources identically.

    Args:
        path: Path to the CSV file.

    Returns:
        List of option record dicts.
    """
    path = Path(path)
    records: list[dict] = []

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return records

        # Build header mapping for this file
        header_map: dict[str, str] = {}
        for csv_col in reader.fieldnames:
            normalized = COLUMN_MAP.get(csv_col.strip())
            if normalized:
                header_map[csv_col] = normalized

        for row in reader:
            record: dict = {}
            for csv_col, norm_key in header_map.items():
                raw = row.get(csv_col, "")
                record[norm_key] = _clean_value(raw)
            records.append(record)

    return records
