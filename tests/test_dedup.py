"""Tests for tiny_corpus_prep.dedup."""
from __future__ import annotations

import polars as pl
import pytest

from tiny_corpus_prep.dedup import exact_dedup, paragraph_dedup


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

    def test_preserves_first_occurrence_order(self):
        # Regression: unique() without maintain_order returned rows in
        # arbitrary order, breaking run-to-run reproducibility.
        texts = [f"doc {i}" for i in range(200)]
        df = pl.DataFrame({"text": texts + texts})  # each doc duplicated
        result = exact_dedup(df, "text")
        assert result["text"].to_list() == texts


class TestMinHashDedup:
    """MinHash tests are conditional on datasketch being installed."""

    def test_minhash_available(self):
        pytest.importorskip("datasketch")
        from tiny_corpus_prep.dedup import minhash_dedup

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

    def test_deduper_catches_near_dups_across_chunks(self):
        pytest.importorskip("datasketch")
        from tiny_corpus_prep.dedup import MinHashDeduper

        deduper = MinHashDeduper(threshold=0.8)
        chunk1 = pl.DataFrame({
            "text": ["The quick brown fox jumps over the lazy dog."]
        })
        chunk2 = pl.DataFrame({
            "text": [
                "The quick brown fox jumps over the lazy dog!",  # near-dup of chunk1
                "Completely different text about science and math.",
            ]
        })
        r1 = deduper.filter_chunk(chunk1)
        r2 = deduper.filter_chunk(chunk2)
        assert len(r1) == 1
        assert r2["text"].to_list() == [
            "Completely different text about science and math."
        ]

    def test_parallel_mapper_matches_serial(self):
        pytest.importorskip("datasketch")
        from tiny_corpus_prep.dedup import MinHashDeduper
        from tiny_corpus_prep.parallel import TextMapper

        texts = [f"A totally unique document number {i} about topic {i}." for i in range(30)]
        texts.append(texts[0] + "!")  # near-dup of the first
        df = pl.DataFrame({"text": texts})

        serial = MinHashDeduper(threshold=0.8).filter_chunk(df)
        with TextMapper(n_workers=2, chunksize=5) as mapper:
            parallel = MinHashDeduper(threshold=0.8).filter_chunk(df, mapper=mapper)
        assert parallel["text"].to_list() == serial["text"].to_list()


class TestParagraphDedup:
    LONG_A = "This is a long boilerplate paragraph about licensing terms that appears in many documents word for word."
    LONG_B = "Another long and unique paragraph with enough characters to be eligible for deduplication checks."

    def test_repeated_long_paragraph_removed_second_time(self):
        df = pl.DataFrame({"text": [
            f"{self.LONG_A}\n\n{self.LONG_B}",
            f"{self.LONG_A}\n\nSomething short.",
        ]})
        result = paragraph_dedup(df, min_chars=50)
        texts = result["text"].to_list()
        assert texts[0] == f"{self.LONG_A}\n\n{self.LONG_B}"
        assert texts[1] == "Something short."

    def test_short_repeated_lines_kept(self):
        df = pl.DataFrame({"text": ["References\n\n" + self.LONG_A,
                                    "References\n\n" + self.LONG_B]})
        result = paragraph_dedup(df, min_chars=50)
        assert all(t.startswith("References") for t in result["text"])

    def test_doc_dropped_when_emptied(self):
        df = pl.DataFrame({"text": [self.LONG_A, self.LONG_A]})
        result = paragraph_dedup(df, min_chars=50)
        assert len(result) == 1

    def test_cross_chunk_state(self):
        seen: set = set()
        chunk1 = pl.DataFrame({"text": [self.LONG_A]})
        chunk2 = pl.DataFrame({"text": [f"{self.LONG_A}\n\n{self.LONG_B}"]})
        paragraph_dedup(chunk1, seen_hashes=seen)
        result = paragraph_dedup(chunk2, seen_hashes=seen)
        assert result["text"].to_list() == [self.LONG_B]
