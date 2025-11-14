"""
Vocabulary simplification.
Two strategies:
1) Rule-based mapping using a user-provided JSON map { "synonym": "canonical", ... }.
2) Optional WordNet-based collapse if nltk + wordnet are available.
"""
from __future__ import annotations
from typing import Dict, Iterable, Tuple
import json
import re
import os

try:
    from .common import preserve_case_like
except ImportError:
    # Fallback if common module is not available
    def preserve_case_like(source_token: str, replacement: str) -> str:
        if source_token.isupper():
            return replacement.upper()
        if len(source_token) > 1 and source_token[0].isupper() and source_token[1:].islower():
            return replacement.title()
        return replacement

_WORD_RE = re.compile(r"\b\w+\b")

class SynonymMapper:
    def __init__(self, mapping: Dict[str, str] | None = None, preserve_case: bool = True):
        """
        Initialize SynonymMapper.
        
        Args:
            mapping: Dictionary mapping synonyms to canonical forms
            preserve_case: If True, preserve the case pattern of the original word
        """
        # Expect lowercased keys and values
        self.mapping = {k.lower(): v.lower() for k, v in (mapping or {}).items()}
        self.preserve_case = preserve_case

    @classmethod
    def from_json(cls, path: str, preserve_case: bool = True):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, preserve_case=preserve_case)

    def add(self, synonym: str, canonical: str):
        self.mapping[synonym.lower()] = canonical.lower()

    def simplify_line(self, s: str) -> str:
        if not self.mapping:
            return s
        # replace whole words using regex
        def repl(m):
            w = m.group(0)
            lw = w.lower()
            replacement = self.mapping.get(lw)
            if replacement:
                if self.preserve_case:
                    return preserve_case_like(w, replacement)
                return replacement
            return w
        return _WORD_RE.sub(repl, s)

    def simplify_iter(self, lines: Iterable[str]) -> Iterable[str]:
        for line in lines:
            yield self.simplify_line(line)

def build_wordnet_mapping(target_pos: Tuple[str, ...] = ("n","v","a","r")) -> Dict[str, str]:
    """
    Build a synonym->canonical mapping using WordNet if available.
    Canonical is the first lemma name of each synset.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except Exception:
        return {}

    # ensure wordnet is downloaded
    try:
        wn.ensure_loaded()
    except Exception:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            return {}
    mapping: Dict[str, str] = {}
    for pos in target_pos:
        for syn in wn.all_synsets(pos=pos):
            lemmas = syn.lemma_names()
            if not lemmas:
                continue
            canonical = lemmas[0].replace("_"," ").lower()
            for lemma in lemmas:
                w = lemma.replace("_"," ").lower()
                if w != canonical and w not in mapping:
                    mapping[w] = canonical
    return mapping
