"""Tests for pipeline-level behavior: parallelism, paragraph dedup, chunked minhash."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tiny_corpus_prep.pipeline import process_corpus, process_corpus_chunked


@pytest.fixture
def prose_parquet(tmp_path: Path) -> Path:
    df = pl.DataFrame({"text": [
        f"The cat number {i} sat on the mat. It watched the birds and was very happy about it."
        for i in range(300)
    ]})
    path = tmp_path / "in.parquet"
    df.write_parquet(path)
    return path


class TestParallelEquivalence:
    def test_n_workers_does_not_change_output(self, prose_parquet: Path, tmp_path: Path):
        out1 = tmp_path / "serial.parquet"
        out2 = tmp_path / "parallel.parquet"
        kwargs = dict(
            normalize_mode="gentle",
            max_grade=12.0,
            synonyms_map={"happy": "glad"},
            dedup_mode="exact",
            generate_stats=False,
        )
        process_corpus(str(prose_parquet), str(out1), n_workers=1, **kwargs)
        process_corpus(str(prose_parquet), str(out2), n_workers=2, **kwargs)
        assert (
            pl.read_parquet(out1)["text"].to_list()
            == pl.read_parquet(out2)["text"].to_list()
        )


class TestParagraphDedupInPipeline:
    def test_boilerplate_removed_across_docs(self, tmp_path: Path):
        boiler = (
            "This content is licensed under the creative commons attribution "
            "license and may be reused freely by anyone."
        )
        df = pl.DataFrame({"text": [
            f"First article body with enough words to pass filters.\n\n{boiler}",
            f"Second article body, different from the first one entirely.\n\n{boiler}",
        ]})
        inp = tmp_path / "in.parquet"
        df.write_parquet(inp)
        out = tmp_path / "out.parquet"
        process_corpus(
            str(inp), str(out),
            max_grade=None, paragraph_dedup=True, paragraph_min_chars=50,
            generate_stats=False,
        )
        texts = pl.read_parquet(out)["text"].to_list()
        assert boiler in texts[0]
        assert boiler not in texts[1]

    def test_paragraph_dedup_state_spans_chunks(self, tmp_path: Path):
        boiler = (
            "This content is licensed under the creative commons attribution "
            "license and may be reused freely by anyone."
        )
        df = pl.DataFrame({"text": [
            f"Doc {i} body which is fully unique and long enough.\n\n{boiler}"
            for i in range(10)
        ]})
        inp = tmp_path / "in.parquet"
        df.write_parquet(inp)
        out = tmp_path / "out.parquet"
        process_corpus_chunked(
            str(inp), str(out), chunk_size=3,
            max_grade=None, paragraph_dedup=True, paragraph_min_chars=50,
        )
        texts = pl.read_parquet(out)["text"].to_list()
        assert sum(boiler in t for t in texts) == 1  # kept only on first occurrence


class TestChunkedMinHash:
    def test_near_dups_removed_across_chunks_without_post_pass(self, tmp_path: Path):
        pytest.importorskip("datasketch")
        base = "The quick brown fox jumps over the lazy dog near the river bank today"
        rows = [f"Unique document {i} about a completely distinct subject matter." for i in range(8)]
        rows.insert(0, base + ".")
        rows.append(base + "!")  # near-dup, lands in a later chunk
        inp = tmp_path / "in.parquet"
        pl.DataFrame({"text": rows}).write_parquet(inp)
        out = tmp_path / "out.parquet"
        stats = process_corpus_chunked(
            str(inp), str(out), chunk_size=4,
            max_grade=None, dedup_mode="both",
        )
        texts = pl.read_parquet(out)["text"].to_list()
        assert base + "." in texts
        assert base + "!" not in texts
        assert stats["total_rows"] == len(texts)

    def test_chunked_stats_are_streaming(self, tmp_path: Path):
        inp = tmp_path / "in.parquet"
        pl.DataFrame({"text": [f"Simple document number {i} here." for i in range(10)]}).write_parquet(inp)
        out = tmp_path / "out.parquet"
        stats = process_corpus_chunked(str(inp), str(out), chunk_size=4, max_grade=None)
        assert stats["total_rows"] == 10
        assert stats["text_stats"]["total_characters"] > 0
        assert (out.with_suffix(".json")).exists()
