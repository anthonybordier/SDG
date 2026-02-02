"""Tests for barchart CSV parser."""

from __future__ import annotations

from pathlib import Path

from sdg.market_data.barchart.csv_parser import parse_csv

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseCSV:
    def test_parse_fixture(self):
        records = parse_csv(FIXTURES / "barchart_options.csv")
        assert len(records) == 20

        # Check first record
        r = records[0]
        assert r["strike"] == 65.0
        assert r["bid"] == 10.50
        assert r["ask"] == 10.80
        assert r["type"] == "Call"
        assert r["expirationDate"] == "11/17/2025"
        assert r["underlyingSymbol"] == "CLZ25"

    def test_field_values_types(self):
        records = parse_csv(FIXTURES / "barchart_options.csv")
        for r in records:
            assert isinstance(r["strike"], float)
            assert isinstance(r["bid"], float)
            assert isinstance(r["ask"], float)
            assert isinstance(r["type"], str)

    def test_bom_handling(self, tmp_path):
        csv_content = "\ufeffSymbol,Strike,Bid,Ask,Type,Expiration Date,Underlying Symbol\n"
        csv_content += "CLZ25|70C,65.0,10.50,10.80,Call,11/17/2025,CLZ25\n"
        p = tmp_path / "bom_test.csv"
        p.write_text(csv_content, encoding="utf-8-sig")

        records = parse_csv(p)
        assert len(records) == 1
        assert records[0]["strike"] == 65.0

    def test_missing_values(self, tmp_path):
        csv_content = "Symbol,Strike,Bid,Ask,Type,Expiration Date,Underlying Symbol\n"
        csv_content += "CLZ25|70C,65.0,N/A,-,Call,11/17/2025,CLZ25\n"
        csv_content += "CLZ25|70P,65.0,,0.45,Put,11/17/2025,CLZ25\n"
        p = tmp_path / "missing_test.csv"
        p.write_text(csv_content)

        records = parse_csv(p)
        assert len(records) == 2
        assert records[0]["bid"] is None
        assert records[0]["ask"] is None
        assert records[1]["bid"] is None
        assert records[1]["ask"] == 0.45

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("")
        records = parse_csv(p)
        assert records == []
