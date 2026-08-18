"""Tests for tiny_corpus_prep.io."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from tiny_corpus_prep.io import (
    generate_stats,
    generate_stats_streaming,
    write_parquet_with_stats,
)


class TestGenerateStats:
    def test_basic_text_stats(self):
        df = pl.DataFrame({"text": ["hello", "a longer text here", ""]})
        stats = generate_stats(df, "text")
        assert stats["total_rows"] == 3
        assert stats["text_stats"]["empty_or_null_count"] == 1

    def test_extra_string_column_does_not_crash(self):
        # Regression: value_counts().sort("counts") crashed on Polars >= 0.20
        # (column renamed to "count") — hit whenever the input had e.g. a
        # title/topic column, at the very end of a run.
        df = pl.DataFrame({
            "text": ["doc one", "doc two"],
            "topic": ["science", "science"],
        })
        stats = generate_stats(df, "text")
        topic_stats = stats["column_stats"]["topic"]
        assert topic_stats["unique_values"] == 1
        assert topic_stats["top_values"][0]["topic"] == "science"

    def test_high_cardinality_column_skips_top_values(self):
        df = pl.DataFrame({
            "text": ["doc"] * 1001,
            "title": [f"title {i}" for i in range(1001)],
        })
        stats = generate_stats(df, "text")
        title_stats = stats["column_stats"]["title"]
        assert title_stats["unique_values"] == 1001
        assert "top_values" not in title_stats

    def test_numeric_column_stats(self):
        df = pl.DataFrame({"text": ["a", "b"], "score": [1.0, 3.0]})
        stats = generate_stats(df, "text")
        assert stats["column_stats"]["score"]["mean"] == 2.0


class TestGenerateStatsStreaming:
    def test_matches_in_memory_text_stats(self, tmp_path: Path):
        df = pl.DataFrame({
            "text": ["hello", "a much longer text right here", "mid"],
            "title": ["a", "b", "c"],
        })
        path = tmp_path / "corpus.parquet"
        df.write_parquet(path)
        streamed = generate_stats_streaming(str(path), "text")
        in_memory = generate_stats(df, "text")
        assert streamed["total_rows"] == in_memory["total_rows"]
        for key in ("min_length", "max_length", "mean_length", "total_characters"):
            assert streamed["text_stats"][key] == in_memory["text_stats"][key]


class TestWriteParquetWithStats:
    def test_writes_parquet_and_json(self, tmp_path: Path):
        df = pl.DataFrame({"text": ["hello", "world"], "source": ["wiki", "web"]})
        out = tmp_path / "out.parquet"
        stats = write_parquet_with_stats(df, str(out), "text")

        assert out.exists()
        stats_path = out.with_suffix(".json")
        assert stats_path.exists()
        on_disk = json.loads(stats_path.read_text())
        assert on_disk["total_rows"] == stats["total_rows"] == 2

        round_trip = pl.read_parquet(out)
        assert round_trip["text"].to_list() == ["hello", "world"]
