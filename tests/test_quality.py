"""Tests for tiny_corpus_prep.quality."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from tiny_corpus_prep.quality import (
    filter_by_repetition,
    is_degenerate,
    repetition_signals,
    strip_boilerplate,
    strip_boilerplate_lines,
)

NORMAL = (
    "The old lighthouse keeper walked along the rocky shore every morning. "
    "He checked the lamp, cleaned the glass, and wrote notes about the weather. "
    "Ships passed far away on the horizon while gulls circled above the cliffs. "
    "In the evening he read books about distant countries and drank warm tea. "
    "His daughter visited on weekends and brought fresh bread from the village."
)


class TestRepetitionSignals:
    def test_normal_text_passes(self):
        assert is_degenerate(NORMAL) is False

    def test_repeated_line_caught(self):
        text = "\n".join(["Buy cheap watches online now!"] * 10)
        signals = repetition_signals(text)
        assert signals["duplicate_line_ratio"] > 0.8
        assert is_degenerate(text) is True

    def test_looping_ngram_caught(self):
        text = "click here to win " * 40  # one 4-gram covers everything
        assert is_degenerate(text) is True

    def test_char_run_caught(self):
        text = NORMAL + " " + "a" * 100
        assert repetition_signals(text)["char_run"] >= 100
        assert is_degenerate(text) is True

    def test_short_text_not_falsely_flagged_by_ngrams(self):
        # 6 words, top 2-gram ratio would be huge — but min_words guards it
        text = "the cat sat on the mat"
        assert is_degenerate(text) is False

    def test_threshold_override(self):
        text = "\n".join(["Repeated line here."] * 3 + ["A unique closing line."])
        assert is_degenerate(text) is True  # dup line ratio 0.5 > 0.3
        assert is_degenerate(text, {"max_duplicate_line_ratio": 0.9}) is False

    def test_filter_dataframe(self):
        df = pl.DataFrame({"text": [NORMAL, "spam spam spam " * 30]})
        result = filter_by_repetition(df)
        assert result["text"].to_list() == [NORMAL]


class TestStripBoilerplate:
    def test_boilerplate_lines_removed(self):
        text = (
            "The article begins with a real paragraph of content.\n"
            "This website uses cookies to improve your experience. Accept our policy.\n"
            "Another genuine sentence about the topic at hand.\n"
            "Subscribe to our newsletter for weekly updates!\n"
            "Share this on Facebook and Twitter\n"
            "© 2024 Example Corp. All rights reserved."
        )
        cleaned = strip_boilerplate_lines(text)
        assert "real paragraph" in cleaned
        assert "genuine sentence" in cleaned
        assert "cookies" not in cleaned
        assert "newsletter" not in cleaned
        assert "Facebook" not in cleaned
        assert "rights reserved" not in cleaned

    def test_heading_only_boilerplate_removed(self):
        text = "Real content about lighthouses.\nReferences\nExternal links"
        cleaned = strip_boilerplate_lines(text)
        assert cleaned == "Real content about lighthouses."

    def test_extra_patterns(self):
        text = "Keep this line.\nCUSTOM-FOOTER-MARKER trailing junk"
        cleaned = strip_boilerplate_lines(text, extra_patterns=[r"CUSTOM-FOOTER-MARKER"])
        assert cleaned == "Keep this line."

    def test_dataframe_drops_emptied_docs(self):
        df = pl.DataFrame({"text": [
            "Genuine content stays here.",
            "Click here\nAdvertisement",  # everything is boilerplate
        ]})
        result = strip_boilerplate(df)
        assert result["text"].to_list() == ["Genuine content stays here."]


class TestLanguageFilter:
    def test_detect_and_filter(self):
        pytest.importorskip("langdetect")
        from tiny_corpus_prep.quality import detect_language, filter_by_language

        lang, prob = detect_language(NORMAL)
        assert lang == "en"
        assert prob > 0.9

        df = pl.DataFrame({"text": [
            NORMAL,
            "Il vecchio guardiano del faro camminava lungo la costa rocciosa "
            "ogni mattina e controllava la lampada con grande attenzione.",
        ]})
        result = filter_by_language(df, "en", min_prob=0.8)
        assert result["text"].to_list() == [NORMAL]

    def test_detection_is_deterministic(self):
        pytest.importorskip("langdetect")
        from tiny_corpus_prep.quality import detect_language

        ambiguous = "Piano solo concert in la villa grande"
        assert detect_language(ambiguous) == detect_language(ambiguous)

    def test_empty_text_unknown(self):
        pytest.importorskip("langdetect")
        from tiny_corpus_prep.quality import detect_language

        assert detect_language("   ") == ("unknown", 0.0)


class TestPipelineIntegration:
    def test_quality_options_via_process_corpus(self, tmp_path: Path):
        pytest.importorskip("langdetect")
        from tiny_corpus_prep.pipeline import process_corpus

        docs = [
            NORMAL,                                       # good English prose
            "spam spam spam eggs " * 40,                  # degenerate
            "Questo documento è scritto interamente in lingua italiana "
            "e parla del tempo e delle stagioni dell'anno in campagna.",  # not English
            "Accept our cookie policy to continue.\n" + NORMAL,  # boilerplate line
        ]
        inp = tmp_path / "in.parquet"
        pl.DataFrame({"text": docs}).write_parquet(inp)
        out = tmp_path / "out.parquet"
        process_corpus(
            str(inp), str(out),
            max_grade=None,
            strip_boilerplate=True,
            repetition_filter=True,
            language="en",
            generate_stats=False,
        )
        texts = pl.read_parquet(out)["text"].to_list()
        # spam dropped (repetition), Italian dropped (language), doc 4's
        # boilerplate line stripped leaving an exact copy of NORMAL which
        # exact-dedup then removes — one doc survives.
        assert texts == [NORMAL]
