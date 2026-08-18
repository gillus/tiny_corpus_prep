"""Tests for tiny_corpus_prep.packing."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("datasets")

from tiny_corpus_prep.packing import pack_shards_to_dataset
from tiny_corpus_prep.tokenization import load_tokenizer, train_tokenizer_from_shards


@pytest.fixture
def packed_setup(tmp_path: Path):
    """Corpus shards + a trained tokenizer."""
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    for i in range(2):
        pl.DataFrame({
            "text": [
                f"The cat sat on mat number {i * 20 + j} and watched the birds."
                for j in range(20)
            ],
            "source": ["prose" if j % 2 == 0 else "structured" for j in range(20)],
        }).write_parquet(shards_dir / f"shard_{i:05d}.parquet")

    tokenizer_path = tmp_path / "tokenizer.json"
    train_tokenizer_from_shards(shards_dir, tokenizer_path, vocab_size=400)
    return shards_dir, tokenizer_path


class TestPackShardsToDataset:
    def test_end_to_end(self, packed_setup, tmp_path: Path):
        from datasets import load_from_disk

        shards_dir, tokenizer_path = packed_setup
        out = tmp_path / "dataset"
        stats = pack_shards_to_dataset(
            shards_dir, tokenizer_path, out,
            seq_length=16, val_fraction=0.25, seed=1,
        )

        ds = load_from_disk(str(out))
        assert set(ds.keys()) == {"train", "validation"}
        assert len(ds["train"]) == stats["splits"]["train"]["sequences"]
        assert all(len(row) == 16 for row in ds["train"]["input_ids"][:5])

        # All 40 docs accounted for, both splits populated at 25%
        split_docs = {s: stats["splits"][s]["documents"] for s in ("train", "validation")}
        assert sum(split_docs.values()) == 40
        assert split_docs["validation"] > 0
        assert split_docs["train"] > split_docs["validation"]

        # Internal consistency of token accounting
        for s in ("train", "validation"):
            sp = stats["splits"][s]
            assert sp["packed_tokens"] == sp["sequences"] * 16
            assert sp["raw_tokens"] == sp["packed_tokens"] + sp["dropped_tail_tokens"]
            assert sum(sp["tokens_per_source"].values()) == sp["raw_tokens"]

        assert stats["dtype"] == "uint16"
        assert 0 < stats["vocab_coverage"] <= 1
        assert stats["chars_per_token"] > 0

    def test_eos_separates_documents(self, packed_setup, tmp_path: Path):
        from datasets import load_from_disk

        shards_dir, tokenizer_path = packed_setup
        out = tmp_path / "dataset"
        stats = pack_shards_to_dataset(
            shards_dir, tokenizer_path, out,
            seq_length=16, val_fraction=0.0, seed=1,
        )
        ds = load_from_disk(str(out))
        flat = [t for row in ds["train"]["input_ids"] for t in row]
        eos_count = sum(1 for t in flat if t == stats["eos_token_id"])
        # One EOS per doc, minus at most those lost in the dropped tail
        assert eos_count >= stats["splits"]["train"]["documents"] - 1

        # Decoding the packed stream reproduces the original text
        tok = load_tokenizer(tokenizer_path)
        decoded = tok.decode(flat, skip_special_tokens=True)
        assert "The cat sat on mat number 0" in decoded

    def test_val_fraction_zero(self, packed_setup, tmp_path: Path):
        shards_dir, tokenizer_path = packed_setup
        stats = pack_shards_to_dataset(
            shards_dir, tokenizer_path, tmp_path / "ds0",
            seq_length=16, val_fraction=0.0, seed=1,
        )
        assert stats["splits"]["validation"]["documents"] == 0
        assert stats["splits"]["validation"]["sequences"] == 0

    def test_deterministic(self, packed_setup, tmp_path: Path):
        from datasets import load_from_disk

        shards_dir, tokenizer_path = packed_setup

        def run(out):
            stats = pack_shards_to_dataset(
                shards_dir, tokenizer_path, out,
                seq_length=16, val_fraction=0.25, seed=7,
            )
            ds = load_from_disk(str(out))
            return stats, ds["train"]["input_ids"][0]

        stats1, first1 = run(tmp_path / "ds1")
        stats2, first2 = run(tmp_path / "ds2")
        assert stats1 == stats2
        assert first1 == first2

    def test_small_flush_threshold_gives_identical_result(self, packed_setup, tmp_path: Path):
        from datasets import load_from_disk

        shards_dir, tokenizer_path = packed_setup

        def run(out, flush_tokens):
            stats = pack_shards_to_dataset(
                shards_dir, tokenizer_path, out,
                seq_length=16, val_fraction=0.25, seed=7,
                flush_tokens=flush_tokens,
            )
            ds = load_from_disk(str(out))
            return stats, ds["train"]["input_ids"]

        stats_big, rows_big = run(tmp_path / "big", flush_tokens=8_000_000)
        stats_small, rows_small = run(tmp_path / "small", flush_tokens=16)  # flush every sequence
        assert stats_big == stats_small
        assert rows_big == rows_small

    def test_missing_eos_token_raises(self, packed_setup, tmp_path: Path):
        shards_dir, tokenizer_path = packed_setup
        with pytest.raises(ValueError, match="EOS"):
            pack_shards_to_dataset(
                shards_dir, tokenizer_path, tmp_path / "ds",
                eos_token="<|missing|>",
            )
