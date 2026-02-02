"""Commodity symbol registry for Barchart data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommoditySpec:
    """Specification for a commodity futures contract.

    Attributes:
        root: Root symbol (e.g. "CL", "NG").
        exchange: Exchange code (e.g. "CME", "BMD").
        name: Human-readable name.
    """

    root: str
    exchange: str
    name: str


COMMODITY_REGISTRY: dict[str, CommoditySpec] = {
    "CL": CommoditySpec("CL", "CME", "Crude Oil WTI"),
    "KO": CommoditySpec("KO", "BMD", "Palm Oil (FCPO)"),
    "NG": CommoditySpec("NG", "CME", "Natural Gas"),
    "GC": CommoditySpec("GC", "CME", "Gold"),
    "SI": CommoditySpec("SI", "CME", "Silver"),
    "HG": CommoditySpec("HG", "CME", "Copper"),
    "ZC": CommoditySpec("ZC", "CME", "Corn"),
    "ZS": CommoditySpec("ZS", "CME", "Soybeans"),
    "ZW": CommoditySpec("ZW", "CME", "Wheat"),
}


def get_spec(root: str) -> CommoditySpec:
    """Look up a commodity specification by root symbol.

    Args:
        root: Root symbol (e.g. "CL").

    Returns:
        CommoditySpec for the given root.

    Raises:
        KeyError: If the root symbol is not in the registry.
    """
    try:
        return COMMODITY_REGISTRY[root]
    except KeyError:
        raise KeyError(
            f"Unknown commodity root '{root}'. "
            f"Known roots: {sorted(COMMODITY_REGISTRY.keys())}"
        ) from None
