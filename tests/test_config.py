"""Tests for tiny_corpus_prep.config."""
from __future__ import annotations

from pathlib import Path

import pytest

from tiny_corpus_prep.config import CorpusConfig, load_config


VALID_YAML = """
seed: 7
output:
  dir: out/corpus
  shard_max_docs: 500
processing:
  text_column: text
profiles:
  prose:
    normalize_mode: gentle
    max_grade: 10.0
    min_words: 5
    dedup_mode: exact
  structured:
    normalize_mode: none
    max_grade: null
    dedup_mode: exact
sources:
  wiki:
    input: data/raw/wiki.parquet
    profile: prose
    weight: 0.6
  schedules:
    input: data/raw/schedules.parquet
    profile: structured
    weight: 0.4
    allow_upsample: true
mix:
  cross_source_dedup: true
"""


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content)
    return path


class TestLoadConfig:
    def test_valid_config(self, tmp_path: Path):
        config = load_config(_write(tmp_path, VALID_YAML))
        assert config.seed == 7
        assert config.output.shard_max_docs == 500
        assert config.profiles["structured"].normalize_mode is None
        assert config.profiles["structured"].max_grade is None
        assert config.sources["schedules"].allow_upsample is True
        assert list(config.sources) == ["wiki", "schedules"]  # order preserved

    def test_none_string_coerced(self, tmp_path: Path):
        yaml_text = VALID_YAML.replace("normalize_mode: none", 'normalize_mode: "none"')
        config = load_config(_write(tmp_path, yaml_text))
        assert config.profiles["structured"].normalize_mode is None

    def test_defaults(self, tmp_path: Path):
        minimal = """
output:
  dir: out
profiles:
  p: {}
sources:
  s:
    input: in.parquet
    profile: p
"""
        config = load_config(_write(tmp_path, minimal))
        assert config.seed == 42
        assert config.processing.text_column == "text"
        assert config.profiles["p"].normalize_mode == "gentle"
        assert config.profiles["p"].dedup_mode == "exact"
        assert config.sources["s"].weight == 1.0
        assert config.mix.target_total_words is None

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nope.yaml")

    def test_unknown_profile_reference(self, tmp_path: Path):
        bad = VALID_YAML.replace("profile: structured", "profile: nonexistent")
        with pytest.raises(ValueError, match="nonexistent"):
            load_config(_write(tmp_path, bad))

    def test_unknown_profile_key_rejected(self, tmp_path: Path):
        bad = VALID_YAML.replace("max_grade: 10.0", "max_grde: 10.0")  # typo
        with pytest.raises(ValueError, match="max_grde"):
            load_config(_write(tmp_path, bad))

    def test_invalid_normalize_mode(self, tmp_path: Path):
        bad = VALID_YAML.replace("normalize_mode: gentle", "normalize_mode: extreme")
        with pytest.raises(ValueError, match="normalize_mode"):
            load_config(_write(tmp_path, bad))

    def test_invalid_rare_level(self, tmp_path: Path):
        bad = VALID_YAML.replace(
            "max_grade: 10.0", "rare_levels: [B2, Z9]"
        )
        with pytest.raises(ValueError, match="Z9"):
            load_config(_write(tmp_path, bad))

    def test_nonpositive_weight(self, tmp_path: Path):
        bad = VALID_YAML.replace("weight: 0.6", "weight: 0")
        with pytest.raises(ValueError, match="weight"):
            load_config(_write(tmp_path, bad))

    def test_source_requires_input_and_profile(self, tmp_path: Path):
        bad = """
output:
  dir: out
profiles:
  p: {}
sources:
  s:
    profile: p
"""
        with pytest.raises(ValueError, match="input"):
            load_config(_write(tmp_path, bad))

    def test_tokenizer_and_dataset_sections(self, tmp_path: Path):
        extended = VALID_YAML + """
tokenizer:
  vocab_size: 24000
  extra_special_tokens: ["<schedule>"]
dataset:
  seq_length: 2048
  val_fraction: 0.01
"""
        config = load_config(_write(tmp_path, extended))
        assert config.tokenizer.vocab_size == 24000
        assert config.tokenizer.extra_special_tokens == ["<schedule>"]
        assert config.dataset.seq_length == 2048

    def test_tokenizer_dataset_defaults(self, tmp_path: Path):
        config = load_config(_write(tmp_path, VALID_YAML))
        assert config.tokenizer.vocab_size == 32000
        assert config.tokenizer.eos_token == "<|endoftext|>"
        assert config.dataset.seq_length == 1024
        assert config.dataset.val_fraction == 0.005

    def test_vocab_size_too_small(self, tmp_path: Path):
        bad = VALID_YAML + "\ntokenizer:\n  vocab_size: 100\n"
        with pytest.raises(ValueError, match="vocab_size"):
            load_config(_write(tmp_path, bad))

    def test_val_fraction_out_of_range(self, tmp_path: Path):
        bad = VALID_YAML + "\ndataset:\n  val_fraction: 1.5\n"
        with pytest.raises(ValueError, match="val_fraction"):
            load_config(_write(tmp_path, bad))

    def test_unknown_tokenizer_key_rejected(self, tmp_path: Path):
        bad = VALID_YAML + "\ntokenizer:\n  vocab_sze: 24000\n"
        with pytest.raises(ValueError, match="vocab_sze"):
            load_config(_write(tmp_path, bad))

    def test_quality_profile_fields(self, tmp_path: Path):
        extended = VALID_YAML.replace(
            "max_grade: 10.0",
            "max_grade: 10.0\n"
            "    strip_boilerplate: true\n"
            "    repetition_filter: true\n"
            "    repetition_overrides: {max_duplicate_line_ratio: 0.5}\n"
            "    language: en\n",
        )
        config = load_config(_write(tmp_path, extended))
        prose = config.profiles["prose"]
        assert prose.strip_boilerplate is True
        assert prose.repetition_filter is True
        assert prose.repetition_overrides == {"max_duplicate_line_ratio": 0.5}
        assert prose.language == "en"
        assert config.profiles["structured"].language is None

    def test_unknown_repetition_override_rejected(self, tmp_path: Path):
        bad = VALID_YAML.replace(
            "max_grade: 10.0",
            "max_grade: 10.0\n    repetition_overrides: {max_dup_lines: 0.5}\n",
        )
        with pytest.raises(ValueError, match="max_dup_lines"):
            load_config(_write(tmp_path, bad))

    def test_language_min_prob_out_of_range(self, tmp_path: Path):
        bad = VALID_YAML.replace(
            "max_grade: 10.0", "max_grade: 10.0\n    language_min_prob: 1.5\n"
        )
        with pytest.raises(ValueError, match="language_min_prob"):
            load_config(_write(tmp_path, bad))

    def test_to_dict_round_trips_for_manifest(self, tmp_path: Path):
        config = load_config(_write(tmp_path, VALID_YAML))
        d = config.to_dict()
        assert d["seed"] == 7
        assert d["profiles"]["prose"]["max_grade"] == 10.0
        assert d["sources"]["wiki"]["weight"] == 0.6
