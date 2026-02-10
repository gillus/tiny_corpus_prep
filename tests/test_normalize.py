"""Tests for tiny_corpus_prep.normalize."""
from __future__ import annotations

import pytest

from tiny_corpus_prep.normalize import (
    normalize_text,
    normalize_text_aggressive,
    _normalize_text_gentle,
    normalize_iter,
)


class TestNormalizeTextAggressive:
    def test_empty(self):
        assert normalize_text_aggressive("") == ""

    def test_none_like(self):
        assert normalize_text_aggressive("") == ""

    def test_lowercases(self):
        result = normalize_text_aggressive("HELLO World")
        assert result == result.lower()

    def test_strips_disallowed_chars(self):
        result = normalize_text_aggressive("hello@world#test")
        assert "@" not in result
        assert "#" not in result

    def test_collapses_whitespace(self):
        result = normalize_text_aggressive("hello    world")
        assert "    " not in result

    def test_preserves_allowed_punct(self):
        result = normalize_text_aggressive("hello, world! how? yes.")
        assert "," in result
        assert "!" in result
        assert "?" in result
        assert "." in result

    def test_unicode_smart_quotes(self):
        result = normalize_text_aggressive("\u201CHello\u201D")
        # Smart quotes should be removed (not in allowed set after lowering)
        assert "\u201C" not in result
        assert "\u201D" not in result


class TestNormalizeTextGentle:
    def test_empty(self):
        assert _normalize_text_gentle("") == ""

    def test_preserves_case(self):
        result = _normalize_text_gentle("Hello World")
        assert result == "Hello World"

    def test_nfc_normalization(self):
        # e + combining accent vs precomposed
        composed = "\u00e9"  # é precomposed
        decomposed = "e\u0301"  # e + combining acute
        assert _normalize_text_gentle(decomposed) == _normalize_text_gentle(composed)

    def test_smart_quotes_to_ascii(self):
        result = _normalize_text_gentle("\u201CHello\u201D")
        assert result == '"Hello"'

    def test_collapses_horizontal_space(self):
        result = _normalize_text_gentle("hello   \t  world")
        assert result == "hello world"

    def test_preserves_paragraph_breaks(self):
        result = _normalize_text_gentle("para1\n\npara2")
        assert result == "para1\n\npara2"

    def test_collapses_excessive_newlines(self):
        result = _normalize_text_gentle("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_strips_leading_trailing(self):
        result = _normalize_text_gentle("  hello  ")
        assert result == "hello"

    def test_preserves_structural_chars(self):
        """Gentle mode should NOT strip brackets, colons, etc."""
        input_text = '{"key": "value"}'
        result = _normalize_text_gentle(input_text)
        assert "{" in result
        assert "}" in result
        assert ":" in result


class TestNormalizeTextDispatch:
    def test_gentle_default(self):
        result = normalize_text("Hello World")
        assert result == "Hello World"

    def test_aggressive_mode(self):
        result = normalize_text("Hello World", mode="aggressive")
        assert "hello" in result.lower()

    def test_none_passthrough(self):
        text = "  Hello  World  "
        assert normalize_text(text, mode=None) == text

    def test_none_string_passthrough(self):
        text = "  Hello  World  "
        assert normalize_text(text, mode="none") == text

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization mode"):
            normalize_text("test", mode="bogus")


class TestNormalizeIter:
    def test_filters_empty(self):
        lines = ["hello", "", "world", ""]
        result = list(normalize_iter(lines, mode="gentle"))
        assert result == ["hello", "world"]

    def test_aggressive_mode(self):
        lines = ["Hello", "World"]
        result = list(normalize_iter(lines, mode="aggressive"))
        assert all(r == r.lower() for r in result)
