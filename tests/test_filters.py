"""Tests for tiny_corpus_prep.filters."""
from __future__ import annotations

import polars as pl
import pytest

from tiny_corpus_prep.filters import (
    calculate_readability_grade,
    filter_by_readability,
    filter_by_keywords,
    filter_by_length,
    filter_by_vocabulary_complexity,
    is_middle_school_level,
)
from tiny_corpus_prep.common import CEFRIndex


# ── readability ──────────────────────────────────────────────────

class TestCalculateReadabilityGrade:
    def test_empty_text(self):
        assert calculate_readability_grade("") is None
        assert calculate_readability_grade("   ") is None

    def test_returns_float(self):
        grade = calculate_readability_grade(
            "The cat sat on the mat. The dog ran in the park."
        )
        assert isinstance(grade, float)


class TestIsMiddleSchoolLevel:
    def test_empty(self):
        assert is_middle_school_level("") is False

    def test_simple_text(self):
        # Very simple text should be middle-school level
        assert is_middle_school_level("The cat sat on the mat.") is True


class TestFilterByReadability:
    def test_filters_rows(self, sample_df: pl.DataFrame):
        result = filter_by_readability(sample_df, max_grade=12.0)
        # Should have fewer or equal rows (empty string removed by None grade)
        assert len(result) <= len(sample_df)

    def test_no_readability_grade_column_leaked(self, sample_df: pl.DataFrame):
        result = filter_by_readability(sample_df, max_grade=12.0)
        assert "_readability_grade" not in result.columns


# ── keywords ─────────────────────────────────────────────────────

class TestFilterByKeywords:
    def test_empty_keywords(self, sample_df: pl.DataFrame):
        result = filter_by_keywords(sample_df, [])
        assert len(result) == len(sample_df)

    def test_filters_by_keyword(self):
        df = pl.DataFrame({"text": ["cat sat", "dog ran", "fish swam"]})
        result = filter_by_keywords(df, ["cat"])
        assert len(result) == 1
        assert "cat" in result["text"][0]

    def test_case_insensitive(self):
        df = pl.DataFrame({"text": ["The CAT sat"]})
        result = filter_by_keywords(df, ["cat"])
        assert len(result) == 1


# ── length ───────────────────────────────────────────────────────

class TestFilterByLength:
    def test_no_filters(self):
        df = pl.DataFrame({"text": ["short", "a bit longer text here"]})
        result = filter_by_length(df)
        assert len(result) == 2

    def test_min_chars(self):
        df = pl.DataFrame({"text": ["hi", "hello world"]})
        result = filter_by_length(df, min_chars=5)
        assert len(result) == 1

    def test_max_chars(self):
        df = pl.DataFrame({"text": ["hi", "hello world and more"]})
        result = filter_by_length(df, max_chars=5)
        assert len(result) == 1

    def test_min_words(self):
        df = pl.DataFrame({"text": ["one", "one two three"]})
        result = filter_by_length(df, min_words=2)
        assert len(result) == 1

    def test_max_words(self):
        df = pl.DataFrame({"text": ["one", "one two three"]})
        result = filter_by_length(df, max_words=2)
        assert len(result) == 1

    def test_combined(self):
        df = pl.DataFrame({"text": ["a", "hello world", "a very long text " * 10]})
        result = filter_by_length(df, min_chars=5, max_chars=50)
        assert len(result) == 1

    def test_word_count_handles_newlines_and_extra_spaces(self):
        # Regression: split(" ") counted "one\ntwo\nthree" as 1 word and
        # "a  b" (double space) as 3 words.
        df = pl.DataFrame({"text": ["one\ntwo\nthree", "a  b"]})
        assert len(filter_by_length(df, min_words=3)) == 1
        assert len(filter_by_length(df, max_words=2)) == 1


# ── vocabulary complexity ────────────────────────────────────────

@pytest.fixture
def simple_cefr_index() -> CEFRIndex:
    """A tiny CEFR index for testing."""
    return CEFRIndex(headword_to_rank={
        "the": 0,  # A1
        "cat": 0,   # A1
        "sat": 0,   # A1
        "on": 0,    # A1
        "mat": 1,   # A2
        "quantum": 5,  # C2
        "entanglement": 5,  # C2
        "particle": 4,  # C1
    })


class TestFilterByVocabularyComplexity:
    def test_keeps_simple(self, simple_cefr_index: CEFRIndex):
        df = pl.DataFrame({"text": ["the cat sat on the mat"]})
        result = filter_by_vocabulary_complexity(df, simple_cefr_index, max_rare_ratio=0.3)
        assert len(result) == 1

    def test_drops_complex(self, simple_cefr_index: CEFRIndex):
        df = pl.DataFrame({"text": ["quantum entanglement particle"]})
        result = filter_by_vocabulary_complexity(df, simple_cefr_index, max_rare_ratio=0.1)
        assert len(result) == 0

    def test_unknown_not_rare_by_default(self, simple_cefr_index: CEFRIndex):
        df = pl.DataFrame({"text": ["the unknownword sat"]})
        result = filter_by_vocabulary_complexity(
            df, simple_cefr_index, max_rare_ratio=0.1, count_unknown_as_rare=False
        )
        assert len(result) == 1

    def test_unknown_as_rare(self, simple_cefr_index: CEFRIndex):
        # "unknownword" is not in CEFR, counted as rare
        df = pl.DataFrame({"text": ["unknownword unknownword unknownword"]})
        result = filter_by_vocabulary_complexity(
            df, simple_cefr_index, max_rare_ratio=0.1, count_unknown_as_rare=True
        )
        assert len(result) == 0
