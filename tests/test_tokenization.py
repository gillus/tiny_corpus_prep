"""Tests for tiny_corpus_prep.tokenization."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

pytest.importorskip("tokenizers")

from tiny_corpus_prep.tokenization import (
    check_single_tokens,
    iter_shard_text_batches,
    list_shard_files,
    load_tokenizer,
    train_bpe_tokenizer,
    train_tokenizer_from_shards,
)

STRUCTURAL = ["{", "}", "[", "]", ":", '"', ",", "-", "(", ")"]


def _text_batches():
    prose = [
        f"The cat sat on the mat and watched bird number {i} fly over the town."
        for i in range(50)
    ]
    structured = [
        f'{{"day": "monday", "activity": "shower", "time": "07:{i:02d}"}}'
        for i in range(50)
    ]
    return [prose, structured]


class TestTrainBpeTokenizer:
    def test_special_tokens_get_lowest_ids(self):
        tok = train_bpe_tokenizer(_text_batches(), vocab_size=400)
        assert tok.token_to_id("<|endoftext|>") == 0
        assert tok.token_to_id("<|pad|>") == 1

    def test_extra_special_tokens(self):
        tok = train_bpe_tokenizer(
            _text_batches(), vocab_size=400, extra_special_tokens=["<schedule>"]
        )
        assert tok.token_to_id("<schedule>") == 2

    def test_duplicate_special_tokens_raise(self):
        with pytest.raises(ValueError, match="Duplicate"):
            train_bpe_tokenizer(
                _text_batches(), vocab_size=400,
                extra_special_tokens=["<|endoftext|>"],
            )

    def test_structural_chars_are_single_tokens(self):
        tok = train_bpe_tokenizer(_text_batches(), vocab_size=400)
        assert check_single_tokens(tok, STRUCTURAL) == {}

    def test_byte_level_roundtrip_is_lossless(self):
        tok = train_bpe_tokenizer(_text_batches(), vocab_size=400)
        for text in [
            '{"key": [1, 2], "name": "café"}',
            "The cat's mat — with unicode…",
            "line one\nline two",
        ]:
            assert tok.decode(tok.encode(text).ids) == text


class TestShardHelpers:
    @pytest.fixture
    def shards(self, tmp_path: Path) -> Path:
        shards_dir = tmp_path / "shards"
        shards_dir.mkdir()
        for i in range(2):
            pl.DataFrame(
                {"text": [f"shard {i} doc {j} with words" for j in range(5)]}
            ).write_parquet(shards_dir / f"shard_{i:05d}.parquet")
        return shards_dir

    def test_list_shard_files_sorted(self, shards: Path):
        files = list_shard_files(shards)
        assert [f.name for f in files] == ["shard_00000.parquet", "shard_00001.parquet"]

    def test_missing_shards_raise(self, tmp_path: Path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="build_corpus"):
            list_shard_files(tmp_path / "empty")

    def test_iter_batches_in_order(self, shards: Path):
        batches = list(iter_shard_text_batches(shards, batch_size=3))
        texts = [t for b in batches for t in b]
        assert len(texts) == 10
        assert texts[0] == "shard 0 doc 0 with words"
        assert texts[-1] == "shard 1 doc 4 with words"

    def test_train_from_shards_saves_and_reloads(self, shards: Path, tmp_path: Path):
        out = tmp_path / "tok" / "tokenizer.json"
        path = train_tokenizer_from_shards(
            shards, out, vocab_size=400, require_single_tokens=STRUCTURAL
        )
        assert path.exists()
        tok = load_tokenizer(path)
        assert tok.token_to_id("<|endoftext|>") == 0
        text = "shard 0 doc 0 with words"
        assert tok.decode(tok.encode(text).ids) == text
