"""Tests for tiny_corpus_prep.common."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from tiny_corpus_prep.common import (
    preserve_case_like,
    tokenize_basic,
    detokenize_basic,
    simple_lemma_like,
    CEFRIndex,
    CEFR_ORDER,
)


class TestPreserveCaseLike:
    def test_all_upper(self):
        assert preserve_case_like("HELLO", "world") == "WORLD"

    def test_title_case(self):
        assert preserve_case_like("Hello", "world") == "World"

    def test_lower(self):
        assert preserve_case_like("hello", "world") == "world"

    def test_mixed(self):
        # Mixed case → return as-is
        assert preserve_case_like("hELLo", "world") == "world"

    def test_single_char_upper(self):
        assert preserve_case_like("A", "be") == "BE"


class TestTokenizeBasic:
    def test_simple(self):
        tokens = tokenize_basic("Hello, world!")
        assert "Hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_contractions(self):
        tokens = tokenize_basic("don't it's")
        assert "don't" in tokens
        assert "it's" in tokens

    def test_numbers(self):
        tokens = tokenize_basic("There are 42 cats.")
        assert "42" in tokens

    def test_empty(self):
        assert tokenize_basic("") == []


class TestDetokenizeBasic:
    def test_simple(self):
        result = detokenize_basic(["Hello", ",", "world", "!"])
        assert result == "Hello, world!"

    def test_contractions(self):
        result = detokenize_basic(["don't", "worry"])
        assert "don't" in result


class TestSimpleLemmaLike:
    def test_plural_s(self):
        assert simple_lemma_like("cats") == "cat"

    def test_plural_ies(self):
        assert simple_lemma_like("stories") == "story"

    def test_plural_es(self):
        assert simple_lemma_like("boxes") == "box"

    def test_ing(self):
        assert simple_lemma_like("running") == "run"

    def test_ed(self):
        assert simple_lemma_like("walked") == "walk"

    def test_ly(self):
        assert simple_lemma_like("quickly") == "quick"

    def test_short_word_unchanged(self):
        # Too short for any rule to fire
        assert simple_lemma_like("cat") == "cat"
        assert simple_lemma_like("go") == "go"


class TestCEFRIndex:
    @pytest.fixture
    def cefr_csv(self, tmp_path: Path) -> Path:
        csv_path = tmp_path / "cefr.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["headword", "CEFR"])
            writer.writerow(["the", "A1"])
            writer.writerow(["cat", "A1"])
            writer.writerow(["observe", "B2"])
            writer.writerow(["quantum", "C2"])
        return csv_path

    def test_from_csv(self, cefr_csv: Path):
        idx = CEFRIndex.from_csv(cefr_csv)
        assert idx.rank("the") == CEFR_ORDER["A1"]
        assert idx.rank("quantum") == CEFR_ORDER["C2"]

    def test_is_easy(self, cefr_csv: Path):
        idx = CEFRIndex.from_csv(cefr_csv)
        assert idx.is_easy("the") is True
        assert idx.is_easy("quantum") is False

    def test_is_difficult(self, cefr_csv: Path):
        idx = CEFRIndex.from_csv(cefr_csv)
        assert idx.is_difficult("quantum") is True
        assert idx.is_difficult("the") is False

    def test_unknown_word(self, cefr_csv: Path):
        idx = CEFRIndex.from_csv(cefr_csv)
        assert idx.rank("xyzzyx") is None
        assert idx.is_easy("xyzzyx") is False
        assert idx.is_difficult("xyzzyx") is False

    @pytest.mark.parametrize("header", [
        ("headword", "CEFR"),
        ("Headword", "CEFR"),
        ("HEADWORD", "cefr"),
        ("headword", "Cefr"),
    ])
    def test_from_csv_header_case_insensitive(self, tmp_path: Path, header):
        # Regression: mismatched header case used to silently produce an
        # empty index (every row skipped), making the vocab filter a no-op.
        csv_path = tmp_path / "cefr_case.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(header))
            writer.writerow(["the", "A1"])
            writer.writerow(["quantum", "C2"])
        idx = CEFRIndex.from_csv(csv_path)
        assert idx.rank("the") == CEFR_ORDER["A1"]
        assert idx.rank("quantum") == CEFR_ORDER["C2"]

    def test_from_csv_missing_columns_raises(self, tmp_path: Path):
        csv_path = tmp_path / "bad.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "level"])
            writer.writerow(["the", "A1"])
        with pytest.raises(ValueError):
            CEFRIndex.from_csv(csv_path)
