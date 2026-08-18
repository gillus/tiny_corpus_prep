"""End-to-end tests for config-driven corpus building."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from tiny_corpus_prep.build import build_corpus
from tiny_corpus_prep.config import load_config


@pytest.fixture
def corpus_setup(tmp_path: Path) -> Path:
    """Two raw sources (prose + structured) and a config file."""
    raw = tmp_path / "raw"
    raw.mkdir()

    prose = pl.DataFrame({
        "text": [
            f"The cat sat on the mat and looked at bird number {i}. "
            "It was a sunny day and the cat was happy."
            for i in range(30)
        ],
        "title": [f"doc {i}" for i in range(30)],
    })
    prose.write_parquet(raw / "prose.parquet")

    structured = pl.DataFrame({
        "text": [
            f'{{"day": "monday", "activity": "shower", "time": "07:{i:02d}"}}'
            for i in range(30)
        ],
    })
    structured.write_parquet(raw / "structured.parquet")

    config_path = tmp_path / "corpus.yaml"
    config_path.write_text(f"""
seed: 42
output:
  dir: {tmp_path / "out"}
  shard_max_docs: 10
profiles:
  prose:
    normalize_mode: gentle
    min_words: 5
    dedup_mode: exact
  structured:
    normalize_mode: none
    max_grade: null
    dedup_mode: exact
sources:
  prose:
    input: {raw / "prose.parquet"}
    profile: prose
    weight: 0.6
  structured:
    input: {raw / "structured.parquet"}
    profile: structured
    weight: 0.4
    allow_upsample: true
mix:
  target_total_words: 300
tokenizer:
  vocab_size: 400
dataset:
  seq_length: 16
  val_fraction: 0.2
""")
    return config_path


class TestBuildCorpus:
    def test_end_to_end(self, corpus_setup: Path):
        manifest = build_corpus(corpus_setup)
        out_dir = Path(load_config(corpus_setup).output.dir)

        # Per-source processed outputs exist
        assert (out_dir / "processed" / "prose.parquet").exists()
        assert (out_dir / "processed" / "structured.parquet").exists()

        # Shards exist, respect shard_max_docs, and carry the source column
        shard_files = sorted((out_dir / "shards").glob("shard_*.parquet"))
        assert shard_files
        total_docs = 0
        for shard in shard_files:
            df = pl.read_parquet(shard)
            assert df.columns == ["text", "source"]
            assert len(df) <= 10
            total_docs += len(df)
        assert total_docs == manifest["mix"]["total_docs"]

        # Structured text survived byte-for-byte (profile: no normalization)
        all_docs = pl.concat([pl.read_parquet(s) for s in shard_files])
        structured_texts = all_docs.filter(pl.col("source") == "structured")["text"]
        assert len(structured_texts) > 0
        assert all(t.startswith('{"day": "monday"') for t in structured_texts)

        # Manifest on disk matches the returned one and captures the config
        on_disk = json.loads((out_dir / "manifest.json").read_text())
        assert on_disk["mix"]["total_docs"] == manifest["mix"]["total_docs"]
        assert on_disk["config"]["seed"] == 42
        assert on_disk["config"]["sources"]["prose"]["weight"] == 0.6
        assert "tiny_corpus_prep" in on_disk["versions"]

    def test_deterministic_across_runs(self, corpus_setup: Path, tmp_path: Path):
        build_corpus(corpus_setup)
        out_dir = Path(load_config(corpus_setup).output.dir)
        first = pl.concat([
            pl.read_parquet(s) for s in sorted((out_dir / "shards").glob("*.parquet"))
        ])

        # Rebuild into a second directory with the same seed/config
        second_config = tmp_path / "corpus2.yaml"
        second_config.write_text(
            corpus_setup.read_text().replace(str(out_dir), str(tmp_path / "out2"))
        )
        build_corpus(second_config)
        second = pl.concat([
            pl.read_parquet(s)
            for s in sorted((tmp_path / "out2" / "shards").glob("*.parquet"))
        ])

        assert first["text"].to_list() == second["text"].to_list()
        assert first["source"].to_list() == second["source"].to_list()


class TestTokenizeCorpus:
    def test_build_then_tokenize(self, corpus_setup: Path):
        pytest.importorskip("tokenizers")
        pytest.importorskip("datasets")
        from datasets import load_from_disk

        from tiny_corpus_prep.build import tokenize_corpus
        from tiny_corpus_prep.tokenization import load_tokenizer

        build_corpus(corpus_setup)
        manifest = tokenize_corpus(corpus_setup)

        out_dir = Path(load_config(corpus_setup).output.dir)
        assert (out_dir / "tokenizer" / "tokenizer.json").exists()
        assert (out_dir / "tokenization_manifest.json").exists()

        ds = load_from_disk(str(out_dir / "dataset"))
        assert set(ds.keys()) == {"train", "validation"}
        assert len(ds["train"]) > 0
        assert len(ds["train"]["input_ids"][0]) == 16

        # Structured docs survive the whole pipeline byte-for-byte:
        # decode the packed stream and find an exact JSON schedule line.
        tok = load_tokenizer(out_dir / "tokenizer" / "tokenizer.json")
        flat = [t for row in ds["train"]["input_ids"] for t in row]
        decoded = tok.decode(flat, skip_special_tokens=True)
        assert '{"day": "monday", "activity": "shower"' in decoded

        stats = manifest["stats"]
        assert stats["vocab_size"] > 256
        assert stats["dtype"] == "uint16"
        assert stats["total_documents"] == manifest_docs_from_stats(stats)

    def test_tokenizer_reused_on_second_run(self, corpus_setup: Path):
        pytest.importorskip("tokenizers")
        pytest.importorskip("datasets")

        from tiny_corpus_prep.build import tokenize_corpus

        build_corpus(corpus_setup)
        tokenize_corpus(corpus_setup)
        out_dir = Path(load_config(corpus_setup).output.dir)
        tok_path = out_dir / "tokenizer" / "tokenizer.json"
        mtime = tok_path.stat().st_mtime_ns

        tokenize_corpus(corpus_setup)  # second run must not retrain
        assert tok_path.stat().st_mtime_ns == mtime


def manifest_docs_from_stats(stats: dict) -> int:
    return sum(s["documents"] for s in stats["splits"].values())
