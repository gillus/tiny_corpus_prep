"""
Quality heuristics (step 5): degenerate-repetition detection, boilerplate
line stripping, and language filtering.

All per-doc functions are module-level and picklable, so they parallelize
through parallel.TextMapper like the other filters.

Repetition signals follow the Gopher-style recipe: within-document duplicate
line/paragraph fractions, the word share of the most frequent 2/3/4-gram, and
the longest single-character run. Thresholds are configurable; n-gram signals
are skipped below `min_words` so short documents aren't trivially flagged.
"""
from __future__ import annotations

import functools
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import polars as pl

logger = logging.getLogger(__name__)

# ── degenerate repetition ────────────────────────────────────────

DEFAULT_REPETITION_THRESHOLDS: Dict[str, float] = {
    "max_duplicate_line_ratio": 0.30,
    "max_duplicate_paragraph_ratio": 0.30,
    "max_top_2gram_ratio": 0.20,
    "max_top_3gram_ratio": 0.18,
    "max_top_4gram_ratio": 0.16,
    "max_char_run": 30,
    # n-gram signals only apply to docs with at least this many words
    "min_words": 50,
}

_CHAR_RUN_RE = re.compile(r"(.)\1{9,}")  # runs of 10+; exact length checked below


def _top_ngram_ratio(words: List[str], n: int) -> float:
    """Fraction of words covered by the single most frequent word n-gram."""
    if len(words) < n * 2:
        return 0.0
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    top = counts.most_common(1)[0][1]
    if top < 2:
        return 0.0
    return top * n / len(words)


def repetition_signals(text: str, min_words: int = 50) -> Dict[str, float]:
    """Compute within-document repetition signals for one text."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    words = text.split()

    signals: Dict[str, float] = {
        "duplicate_line_ratio": (
            1.0 - len(set(lines)) / len(lines) if len(lines) > 1 else 0.0
        ),
        "duplicate_paragraph_ratio": (
            1.0 - len(set(paragraphs)) / len(paragraphs)
            if len(paragraphs) > 1 else 0.0
        ),
    }

    if len(words) >= min_words:
        for n in (2, 3, 4):
            signals[f"top_{n}gram_ratio"] = _top_ngram_ratio(words, n)
    else:
        for n in (2, 3, 4):
            signals[f"top_{n}gram_ratio"] = 0.0

    char_run = 0
    for match in _CHAR_RUN_RE.finditer(text):
        char_run = max(char_run, len(match.group(0)))
    signals["char_run"] = float(char_run)
    return signals


def is_degenerate(text: str, thresholds: Optional[Dict[str, float]] = None) -> bool:
    """True if any repetition signal exceeds its threshold."""
    t = dict(DEFAULT_REPETITION_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    s = repetition_signals(text, min_words=int(t["min_words"]))
    return (
        s["duplicate_line_ratio"] > t["max_duplicate_line_ratio"]
        or s["duplicate_paragraph_ratio"] > t["max_duplicate_paragraph_ratio"]
        or s["top_2gram_ratio"] > t["max_top_2gram_ratio"]
        or s["top_3gram_ratio"] > t["max_top_3gram_ratio"]
        or s["top_4gram_ratio"] > t["max_top_4gram_ratio"]
        or s["char_run"] > t["max_char_run"]
    )


def filter_by_repetition(
    df: pl.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    text_column: str = "text",
    mapper=None,
) -> pl.DataFrame:
    """Drop documents with degenerate repetition."""
    fn = functools.partial(is_degenerate, thresholds=thresholds)
    texts = df[text_column].to_list()
    degenerate = mapper.map(fn, texts) if mapper is not None else [fn(t) for t in texts]
    return df.filter(pl.Series([not d for d in degenerate]))


# ── boilerplate line stripping ───────────────────────────────────

DEFAULT_BOILERPLATE_PATTERNS: List[str] = [
    r"\bcookies?\b.*\b(policy|accept|consent|enable|settings)\b",
    r"\ball rights reserved\b",
    r"^\s*copyright\b",
    r"©\s*\d{4}",
    r"\b(subscribe|sign\s*up)\b.*\bnewsletter\b",
    r"\bclick here\b",
    r"^\s*read more\s*$",
    r"\bshare (this|on)\b.*\b(facebook|twitter|linkedin|pinterest|whatsapp)\b",
    r"\bfollow us on\b",
    r"\bprivacy policy\b",
    r"\bterms of (use|service)\b",
    r"\bjavascript\b.*\b(disabled|enabled|enable)\b",
    r"^\s*(external links|see also|references|further reading)\s*$",
    r"\badvertisement\b",
]


def compile_boilerplate_patterns(
    extra_patterns: Optional[Sequence[str]] = None,
) -> List[re.Pattern]:
    """Compile the default patterns plus any extras (case-insensitive)."""
    patterns = list(DEFAULT_BOILERPLATE_PATTERNS) + list(extra_patterns or [])
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def strip_boilerplate_lines(
    text: str,
    extra_patterns: Optional[Sequence[str]] = None,
) -> str:
    """
    Remove lines matching boilerplate patterns; preserves the remaining
    line structure. Returns "" if nothing survives.

    Note: module-level and picklable — the pattern list is compiled per call
    when used standalone; the pipeline pre-binds a compiled list via
    _strip_with_patterns for speed.
    """
    return _strip_with_compiled(text, compile_boilerplate_patterns(extra_patterns))


def _strip_with_compiled(text: str, compiled: List[re.Pattern]) -> str:
    kept = [
        line for line in text.split("\n")
        if not any(p.search(line) for p in compiled)
    ]
    return "\n".join(kept).strip()


def _strip_for_pool(text: str, extra_patterns: Optional[Tuple[str, ...]]) -> str:
    """Picklable per-text strip function; compiles once per process (cached)."""
    compiled = _compiled_cache(extra_patterns)
    return _strip_with_compiled(text, compiled)


@functools.lru_cache(maxsize=8)
def _compiled_cache(extra_patterns: Optional[Tuple[str, ...]]) -> List[re.Pattern]:
    return compile_boilerplate_patterns(extra_patterns)


def strip_boilerplate(
    df: pl.DataFrame,
    extra_patterns: Optional[Sequence[str]] = None,
    text_column: str = "text",
    mapper=None,
) -> pl.DataFrame:
    """Strip boilerplate lines from every doc; drop docs left empty."""
    extras = tuple(extra_patterns) if extra_patterns else None
    fn = functools.partial(_strip_for_pool, extra_patterns=extras)
    texts = df[text_column].to_list()
    cleaned = mapper.map(fn, texts) if mapper is not None else [fn(t) for t in texts]
    return (
        df.with_columns(pl.Series(text_column, cleaned, dtype=pl.Utf8))
        .filter(pl.Series([bool(t) for t in cleaned]))
    )


# ── language filtering ───────────────────────────────────────────

_LANG_SAMPLE_CHARS = 2000  # detection on a prefix; plenty for reliable LID


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the dominant language of a text. Returns (iso_code, probability),
    or ("unknown", 0.0) when detection fails.

    Requires the optional `langdetect` package. Detection is made
    deterministic by seeding langdetect's DetectorFactory.
    """
    try:
        from langdetect import DetectorFactory, detect_langs
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        raise ImportError(
            "Language filtering requires langdetect. "
            "Install with: pip install 'tiny_corpus_prep[quality]'"
        )
    DetectorFactory.seed = 0
    if not text or not text.strip():
        return ("unknown", 0.0)
    try:
        candidates = detect_langs(text[:_LANG_SAMPLE_CHARS])
    except LangDetectException:
        return ("unknown", 0.0)
    if not candidates:
        return ("unknown", 0.0)
    top = candidates[0]
    return (top.lang, float(top.prob))


def _language_ok(text: str, language: str, min_prob: float) -> bool:
    lang, prob = detect_language(text)
    return lang == language and prob >= min_prob


def filter_by_language(
    df: pl.DataFrame,
    language: str,
    min_prob: float = 0.8,
    text_column: str = "text",
    mapper=None,
) -> pl.DataFrame:
    """Keep only documents detected as *language* with at least *min_prob*."""
    fn = functools.partial(_language_ok, language=language, min_prob=min_prob)
    texts = df[text_column].to_list()
    keep = mapper.map(fn, texts) if mapper is not None else [fn(t) for t in texts]
    return df.filter(pl.Series(keep))
