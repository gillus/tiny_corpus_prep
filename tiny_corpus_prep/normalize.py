"""
Text normalization primitives.
- Lowercase
- Keep only letters/digits/space and , . ? !
- Collapse repeated spaces
- Sentence splitting optional
"""
from __future__ import annotations
import re
from typing import Iterable

# Precompile for speed
# Allow: lowercase letters, digits, whitespace, and , . ? !
_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 ,.?!")
# Map unicode punctuation to ASCII or space
_PUNCT_NORMALIZE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201C": '"', "\u201D": '"',
    "\u2013": "-", "\u2014": "-", "\u00A0": " ",
})

# regex that removes any char not in allowed set
# We'll do this by replacing with space, then restrip
_DISALLOWED_RE = re.compile(r"[^a-z0-9 ,\.!\?\n]+")
_MULTI_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_PUNCT_RE = re.compile(r"([,\.!\?]){2,}")

def normalize_text(s: str) -> str:
    """
    Normalize a string to:
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

def normalize_iter(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        out = normalize_text(line)
        if out:
            yield out
