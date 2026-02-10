"""
Text normalization primitives.

Modes:
- "gentle": NFC unicode normalization, smart quotes→ASCII, collapse whitespace
            (preserves paragraph breaks), strip. Does NOT lowercase or remove
            structural characters.
- "aggressive": lowercase, keep only letters/digits/space and , . ? !,
               collapse whitespace and punctuation runs, strip.
- None / "none": passthrough (no-op).
"""
from __future__ import annotations
import re
import unicodedata
from typing import Iterable, Optional

# ── shared constants ──────────────────────────────────────────────

# Map unicode punctuation to ASCII equivalents
_PUNCT_NORMALIZE = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # smart single quotes
    "\u201C": '"', "\u201D": '"',   # smart double quotes
    "\u2013": "-", "\u2014": "-",   # en/em dash
    "\u00A0": " ",                  # non-breaking space
    "\u2026": "...",                # ellipsis
})

# ── aggressive-mode regexes ───────────────────────────────────────

_DISALLOWED_RE = re.compile(r"[^a-z0-9 ,\.!\?\n]+")
_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_PUNCT_RE = re.compile(r"([,\.!\?]){2,}")

# ── gentle-mode regexes ──────────────────────────────────────────

# Collapse runs of spaces/tabs (but NOT newlines) into a single space
_GENTLE_HSPACE_RE = re.compile(r"[^\S\n]+")
# Collapse 3+ consecutive newlines into exactly 2 (preserve paragraph breaks)
_GENTLE_VSPACE_RE = re.compile(r"\n{3,}")


def normalize_text_aggressive(s: str) -> str:
    """
    Aggressive normalization:
    - lowercase
    - only allowed punctuation: , . ? !
    - collapse whitespace and punctuation runs
    - trim
    """
    if not s:
        return ""
    # Lowercase and normalize some unicode punctuation
    s = s.lower().translate(_PUNCT_NORMALIZE)
    # Replace disallowed chars with space
    s = _DISALLOWED_RE.sub(" ", s)
    # Remove apostrophes and quotes explicitly (not allowed)
    s = s.replace("'", " ").replace('"', " ").replace("-", " ")
    # Collapse multiple punctuation to a single char
    s = _MULTI_PUNCT_RE.sub(r"\1", s)
    # Collapse whitespace
    s = _MULTI_SPACE_RE.sub(" ", s)
    # Space around punctuation to avoid word join
    s = re.sub(r"\s*([,\.!\?])\s*", r" \1 ", s)
    # Collapse again and strip
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def _normalize_text_gentle(s: str) -> str:
    """
    Gentle normalization:
    - NFC unicode normalization
    - smart quotes / dashes → ASCII equivalents
    - collapse horizontal whitespace runs into single space
    - collapse 3+ newlines into 2 (preserve paragraph breaks)
    - strip leading/trailing whitespace
    Does NOT lowercase or remove structural characters.
    """
    if not s:
        return ""
    # NFC normalization
    s = unicodedata.normalize("NFC", s)
    # Map smart quotes, dashes, NBSP to ASCII
    s = s.translate(_PUNCT_NORMALIZE)
    # Collapse horizontal whitespace (preserve newlines)
    s = _GENTLE_HSPACE_RE.sub(" ", s)
    # Collapse excessive blank lines
    s = _GENTLE_VSPACE_RE.sub("\n\n", s)
    return s.strip()


def normalize_text(s: str, mode: Optional[str] = "gentle") -> str:
    """
    Normalize text with the given mode.

    Args:
        s: Input string.
        mode: "gentle" (default), "aggressive", or None/"none" for passthrough.

    Returns:
        Normalized string.
    """
    if mode is None or mode == "none":
        return s
    if mode == "aggressive":
        return normalize_text_aggressive(s)
    if mode == "gentle":
        return _normalize_text_gentle(s)
    raise ValueError(f"Unknown normalization mode: {mode!r}")


def normalize_iter(
    lines: Iterable[str],
    mode: Optional[str] = "gentle",
) -> Iterable[str]:
    """Iterate over lines, yielding normalized non-empty results."""
    for line in lines:
        out = normalize_text(line, mode=mode)
        if out:
            yield out
