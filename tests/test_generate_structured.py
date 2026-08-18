"""Tests for bin/generate_structured.py (run as a subprocess)."""
from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).parent.parent / "bin" / "generate_structured.py"


def _run(out: Path, *extra: str) -> None:
    subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), "--num-docs", "30",
         "--seed", "7", *extra],
        check=True, capture_output=True,
    )


class TestGenerateStructured:
    def test_output_valid_and_parseable(self, tmp_path: Path):
        out = tmp_path / "schedules.parquet"
        _run(out)
        df = pl.read_parquet(out)
        assert df.columns == ["text", "format"]
        assert len(df) == 30
        assert set(df["format"].unique()) == {"xml", "json", "kv"}
        for fmt, text in df.select("format", "text").iter_rows():
            body = text.rsplit("\n\n", 1)[0]  # strip the NL summary
            if fmt == "xml":
                root = ET.fromstring(body)
                assert root.tag == "household"
                assert root.find("week") is not None
                assert len(root.find("week").findall("day")) == 7
            elif fmt == "json":
                doc = json.loads(body)
                assert len(doc["week"]) == 7
                assert doc["profile"]["occupants"]

    def test_deterministic_given_seed(self, tmp_path: Path):
        out1, out2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
        _run(out1)
        _run(out2)
        assert (
            pl.read_parquet(out1)["text"].to_list()
            == pl.read_parquet(out2)["text"].to_list()
        )

    def test_format_selection_and_no_summary(self, tmp_path: Path):
        out = tmp_path / "xml_only.parquet"
        _run(out, "--formats", "xml", "--no-summary")
        df = pl.read_parquet(out)
        assert set(df["format"].unique()) == {"xml"}
        # No summary: whole doc is one XML tree
        for text in df["text"]:
            ET.fromstring(text)

    def test_rejects_unknown_format(self, tmp_path: Path):
        with pytest.raises(subprocess.CalledProcessError):
            _run(tmp_path / "x.parquet", "--formats", "yaml")
