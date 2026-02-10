"""Tests for tiny_corpus_prep.dedup."""
from __future__ import annotations

import polars as pl
import pytest

from tiny_corpus_prep.dedup import exact_dedup


class TestExactDedup:
    def test_removes_duplicates(self):
        df = pl.DataFrame({"text": ["hello", "world", "hello", "world", "unique"]})
        result = exact_dedup(df, "text")
        assert len(result) == 3

    def test_no_duplicates(self):
        df = pl.DataFrame({"text": ["a", "b", "c"]})
        result = exact_dedup(df, "text")
        assert len(result) == 3

    def test_cross_chunk_hash_set(self):
        seen: set[int] = set()
        df1 = pl.DataFrame({"text": ["hello", "world"]})
        df2 = pl.DataFrame({"text": ["hello", "new"]})

        r1 = exact_dedup(df1, "text", seen_hashes=seen)
        assert len(r1) == 2

        r2 = exact_dedup(df2, "text", seen_hashes=seen)
        # "hello" already seen
        assert len(r2) == 1
        assert r2["text"][0] == "new"

    def test_empty_df(self):
        df = pl.DataFrame({"text": []}).cast({"text": pl.Utf8})
        result = exact_dedup(df, "text")
        assert len(result) == 0


class TestMinHashDedup:
    """MinHash tests are conditional on datasketch being installed."""

    def test_minhash_available(self):
        try:
            from tiny_corpus_prep.dedup import minhash_dedup
        except ImportError:
            pytest.skip("datasketch not installed")

        df = pl.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "The quick brown fox jumps over the lazy dog!",  # near-dup
                "Completely different text about science and math.",
            ]
        })
        result = minhash_dedup(df, "text", threshold=0.8)
        # Should remove the near-duplicate
        assert len(result) <= 2
