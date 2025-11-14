#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for CEFR-based text simplification.
No external dependencies beyond Python stdlib and (optionally) NLTK in the builder.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

CEFR_ORDER = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}

ALPHA_RE = re.compile(r"^[A-Za-z]+$")


def normalize_token(tok: str) -> str:
    """Lower-case normalization for dictionary keys."""
    return tok.strip().lower()


def _best_titlecase(repl: str) -> str:
    """Titlecase each token in a multi-word replacement."""
    return " ".join([w[:1].upper() + w[1:] if w else w for w in repl.split()])


def preserve_case_like(source_token: str, replacement: str) -> str:
    """
    Return `replacement` adjusted to mimic the case pattern of `source_token`.
    - If source is ALLCAPS -> replacement upper.
    - If Titlecase -> titlecase each word in replacement.
    - Else -> replacement lower (as provided).
    """
    if source_token.isupper():
        return replacement.upper()
    if len(source_token) > 1 and source_token[0].isupper() and source_token[1:].islower():
        return _best_titlecase(replacement)
    # For mixed or lower, return replacement as-is (assumed lowercase coming from dict)
    return replacement


def tokenize_basic(text: str) -> List[str]:
    """
    Simple, whitespace-preserving tokenizer:
    - Words with apostrophes are kept as single tokens: don't, it's
    - Numbers are tokens
    - Punctuation and symbols are separate tokens
    """
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]", text, flags=re.UNICODE)
    return tokens


def detokenize_basic(tokens: Sequence[str]) -> str:
    """
    Basic detokenizer that joins tokens with a single space, then fixes common spacing around punctuation.
    Not perfect; adjust or replace with more sophisticated detokenizer if needed.
    """
    text = " ".join(tokens)
    # Remove space before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Remove space after opening bracket/quote
    text = re.sub(r"([(\[{\"'])\s+", r"\1", text)
    # Remove space before closing bracket/quote
    text = re.sub(r"\s+([)\]}\"'])", r"\1", text)
    # Fix contractions (common English patterns)
    text = text.replace(" n't", "n't").replace(" 's", "'s").replace(" 're", "'re")
    text = text.replace(" 'm", "'m").replace(" 've", "'ve").replace(" 'd", "'d")
    return text


def simple_lemma_like(word: str) -> str:
    """
    Very lightweight heuristic "lemmatizer" without external libs.
    Returns a candidate base form, not necessarily correct.
    """
    w = word.lower()
    # Common plural/singular and verb suffix patterns
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 5:
        stem = w[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if w.endswith("ed") and len(w) > 4:
        stem = w[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if w.endswith("ly") and len(w) > 4:
        return w[:-2]
    return w


@dataclass(frozen=True)
class CEFREntry:
    headword: str
    level: str


@dataclass
class CEFRIndex:
    """Index of headwords to their *minimum* CEFR difficulty rank."""
    headword_to_rank: Dict[str, int]

    @classmethod
    def from_csv(cls, csv_path: Path) -> "CEFRIndex":
        """
        Load CEFR index from CSV with columns: headword, CEFR
        """
        ranks: Dict[str, int] = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.strip().lower() for c in reader.fieldnames or []]
            try:
                cols.index("headword")
                cols.index("cefr")
            except ValueError:
                raise ValueError(f"CSV must have 'headword' and 'CEFR' columns; found {reader.fieldnames}")
            
            f.seek(0)
            reader = csv.DictReader(f)
            for row in reader:
                head = (row.get("headword") or "").strip()
                cefr = (row.get("CEFR") or row.get("cefr") or "").strip().upper()
                if not head or cefr not in CEFR_ORDER:
                    continue
                head_l = head.lower()
                rank = CEFR_ORDER[cefr]
                prev = ranks.get(head_l)
                if prev is None or rank < prev:
                    ranks[head_l] = rank
        return cls(headword_to_rank=ranks)

    def rank(self, word: str) -> Optional[int]:
        """Get the CEFR rank for a word (0=A1, 5=C2)"""
        return self.headword_to_rank.get(word.lower())

    def is_easy(self, word: str, easy_levels: Sequence[str] = ("A1", "A2")) -> bool:
        """Check if a word is in the easy CEFR levels"""
        r = self.rank(word)
        if r is None:
            return False
        return r <= max(CEFR_ORDER[l] for l in easy_levels)

    def is_difficult(self, word: str, difficult_levels: Sequence[str] = ("B2", "C1", "C2")) -> bool:
        """Check if a word is in the difficult CEFR levels"""
        r = self.rank(word)
        if r is None:
            return False
        return r >= min(CEFR_ORDER[l] for l in difficult_levels)


def load_mapping(mapping_path: Path) -> Dict[str, str]:
    """Load a mapping from JSON or 2-column CSV."""
    mapping: Dict[str, str] = {}
    suffix = mapping_path.suffix.lower()
    
    if suffix == ".json":
        with open(mapping_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            mapping = {normalize_token(k): normalize_token(v) for k, v in data.items()}
    elif suffix == ".csv":
        with open(mapping_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header is None:
                return mapping
            
            header_l = [h.strip().lower() for h in header]
            try:
                i_from = header_l.index("from")
                i_to = header_l.index("to")
            except ValueError:
                # Fallback to first two columns
                i_from, i_to = 0, 1
                f.seek(0)
                reader = csv.reader(f)
                next(reader, None)  # skip header
            
            for row in reader:
                if not row or len(row) < max(i_from, i_to) + 1:
                    continue
                src = normalize_token(row[i_from])
                dst = normalize_token(row[i_to])
                if src and dst and src != dst:
                    mapping[src] = dst
    else:
        raise ValueError(f"Unsupported mapping format: {mapping_path}")
    
    return mapping

