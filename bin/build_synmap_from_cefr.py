#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a synonym-replacement dictionary from a CEFR wordlist.

Default behavior:
  - Replace *difficult* words (B2/C1/C2; configurable) with *easy* synonyms (A1/A2; configurable).
  - Synonyms are sourced from WordNet (if available) and filtered to easy CEFR words.
  - Fallbacks include crude lemmatization to an easier form and a small built-in manual map.

Outputs:
  - JSON mapping file (word -> simpler synonym)
  - CSV with provenance and CEFR ranks
  - TXT listing unmapped difficult words
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tiny_corpus_prep.common import (
    CEFR_ORDER,
    CEFRIndex,
    normalize_token,
    simple_lemma_like,
)


def _load_wordnet():
    """Try to load WordNet, return (wn, success_bool)"""
    try:
        import nltk
        from nltk.corpus import wordnet as wn  # type: ignore
        try:
            wn.synsets("dog")
            return wn, True
        except LookupError:
            try:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)
                wn.synsets("dog")
                return wn, True
            except Exception:
                return None, False
    except Exception:
        return None, False


# Manual seed mapping for common formal-to-simple replacements
MANUAL_SEED: Dict[str, str] = {
    "utilize": "use",
    "commence": "start",
    "terminate": "end",
    "endeavor": "try",
    "reside": "live",
    "assist": "help",
    "purchase": "buy",
    "endeavour": "try",
    "subsequent": "next",
    "prior": "before",
    "obtain": "get",
    "require": "need",
    "demonstrate": "show",
    "approximately": "about",
    "sufficient": "enough",
    "numerous": "many",
    "optimal": "best",
    "diminish": "reduce",
    "increase": "raise",
    "decrease": "lower",
    "inform": "tell",
    "attempt": "try",
    "assistive": "helping",
    "assistance": "help",
    "comprehend": "understand",
    "indicate": "show",
    "modify": "change",
    "nevertheless": "still",
    "therefore": "so",
    "consequently": "so",
    "subsequently": "later",
    "commonly": "often",
    "frequently": "often",
    "illustrate": "show",
    "observe": "see",
    "perceive": "see",
    "possess": "have",
    "select": "choose",
    "inquire": "ask",
    "facilitate": "help",
}


def best_easy_synonym_wordnet(
    wn,
    word: str,
    easy_rank_max: int,
    easy_words: Dict[str, int],
    allow_multiword: bool = False,
) -> Optional[str]:
    """
    Find the best easy synonym for a word using WordNet.
    Returns the synonym with the lowest CEFR rank, highest frequency, and shortest length.
    """
    cands: List[Tuple[int, int, int, str]] = []
    
    try:
        synsets = wn.synsets(word)
    except Exception:
        synsets = []
    
    if not synsets:
        return None
    
    seen = set()
    for ss in synsets:
        for lemma in ss.lemmas():
            name = lemma.name().replace("_", " ").lower()
            
            if not allow_multiword and " " in name:
                continue
            if name == word.lower():
                continue
            if name in seen:
                continue
            
            seen.add(name)
            rank = easy_words.get(name)
            if rank is None or rank > easy_rank_max:
                continue
            
            # Get WordNet frequency count
            freq = getattr(lemma, "count", lambda: 0)()
            cands.append((rank, -int(freq), len(name), name))
    
    if not cands:
        return None
    
    cands.sort()
    return cands[0][3]


def main():
    ap = argparse.ArgumentParser(description="Build easy-synonym mapping from CEFR list.")
    ap.add_argument("--cefr_csv", type=Path, required=True, 
                    help="Path to CEFR CSV with columns: headword, CEFR")
    ap.add_argument("--out_dir", type=Path, required=True, 
                    help="Output directory")
    ap.add_argument("--easy_levels", default="A1,A2", 
                    help="Comma-separated levels treated as EASY (default: A1,A2)")
    ap.add_argument("--difficult_levels", default="B2,C1,C2", 
                    help="Comma-separated levels treated as DIFFICULT (default: B2,C1,C2)")
    ap.add_argument("--include_b1", action="store_true", 
                    help="Also treat B1 words as difficult")
    ap.add_argument("--allow_multiword", action="store_true", 
                    help="Allow multiword replacements from WordNet (default: False)")
    ap.add_argument("--no_wordnet", action="store_true", 
                    help="Disable WordNet even if available")
    ap.add_argument("--min_alpha_only", action="store_true", 
                    help="Only build mappings for purely alphabetic headwords (skip entries like 'a.m.')")
    
    args = ap.parse_args()
    
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    easy_levels = [s.strip().upper() for s in args.easy_levels.split(",") if s.strip()]
    difficult_levels = [s.strip().upper() for s in args.difficult_levels.split(",") if s.strip()]
    if args.include_b1 and "B1" not in difficult_levels:
        difficult_levels.append("B1")
    
    # Load CEFR index
    print(f"Loading CEFR index from {args.cefr_csv}")
    cefr = CEFRIndex.from_csv(args.cefr_csv)
    
    # Determine easy words
    easy_max_rank = max(CEFR_ORDER[l] for l in easy_levels)
    easy_words: Dict[str, int] = {
        w: r for w, r in cefr.headword_to_rank.items() if r <= easy_max_rank
    }
    print(f"Found {len(easy_words)} easy words ({', '.join(easy_levels)})")
    
    # Load WordNet if requested
    wn = None
    if not args.no_wordnet:
        print("Loading WordNet...")
        wn, ok = _load_wordnet()
        if not ok:
            print("  WordNet not available")
            wn = None
        else:
            print("  WordNet loaded successfully")
    
    # Collect difficult target words
    diff_targets: List[str] = []
    for w, r in cefr.headword_to_rank.items():
        if args.min_alpha_only and not re.fullmatch(r"[A-Za-z]+", w):
            continue
        # Skip single letter 'a' as it's an article
        if w == "a":
            continue
        if cefr.is_difficult(w, difficult_levels=difficult_levels):
            diff_targets.append(w)
    
    print(f"Found {len(diff_targets)} difficult words to map ({', '.join(difficult_levels)})")
    
    # Build mapping
    mapping: Dict[str, str] = {}
    provenance: Dict[str, str] = {}
    
    # 1. Apply manual seed mappings
    print("Applying manual seed mappings...")
    for src, dst in MANUAL_SEED.items():
        s, d = src.lower(), dst.lower()
        if s in cefr.headword_to_rank and d in easy_words:
            mapping[s] = d
            provenance[s] = "manual"
    print(f"  Added {len(mapping)} manual mappings")
    
    # 2. Try WordNet or lemma fallback for remaining words
    print("Finding synonyms from WordNet and lemmatization...")
    wordnet_count = 0
    lemma_count = 0
    
    for w in diff_targets:
        if w in mapping:
            continue
        
        best: Optional[str] = None
        
        # Try WordNet first
        if wn is not None:
            best = best_easy_synonym_wordnet(
                wn, w, 
                easy_rank_max=easy_max_rank, 
                easy_words=easy_words, 
                allow_multiword=args.allow_multiword
            )
            if best:
                wordnet_count += 1
        
        # Fallback to simple lemmatization
        if not best:
            lemma = simple_lemma_like(w)
            if lemma != w and lemma in easy_words:
                best = lemma
                lemma_count += 1
        
        if best:
            mapping[w] = best
            provenance[w] = "wordnet" if (best and wn and w not in mapping) else "lemma_fallback"
    
    print(f"  Added {wordnet_count} from WordNet")
    print(f"  Added {lemma_count} from lemmatization")
    
    # Write outputs
    json_path = out_dir / "synonyms.json"
    csv_path = out_dir / "synonyms.csv"
    unmapped_path = out_dir / "unmapped.txt"
    stats_path = out_dir / "build_stats.txt"
    
    print(f"\nWriting outputs to {out_dir}")
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["from", "to", "source", "from_cefr", "to_cefr"])
        for src, dst in sorted(mapping.items()):
            src_r = cefr.headword_to_rank.get(src, None)
            dst_r = cefr.headword_to_rank.get(dst, None)
            
            # Convert rank back to level name
            src_level = next((k for k, v in CEFR_ORDER.items() if v == src_r), "") if src_r is not None else ""
            dst_level = next((k for k, v in CEFR_ORDER.items() if v == dst_r), "") if dst_r is not None else ""
            
            writer.writerow([src, dst, provenance.get(src, "unknown"), src_level, dst_level])
    
    unmapped = sorted(set(diff_targets) - set(mapping.keys()))
    with open(unmapped_path, "w", encoding="utf-8") as f:
        for w in unmapped:
            f.write(w + "\n")
    
    total_diff = len(diff_targets)
    mapped = len(mapping)
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(f"Total difficult targets: {total_diff}\n")
        f.write(f"Mapped: {mapped} ({(mapped/total_diff*100 if total_diff else 0):.2f}%)\n")
        f.write(f"Unmapped: {total_diff - mapped}\n")
        f.write(f"WordNet used: {wn is not None}\n")
        f.write(f"Easy levels: {', '.join(easy_levels)}\n")
        f.write(f"Difficult levels: {', '.join(difficult_levels)}\n")
    
    print(f"  JSON mapping: {json_path}")
    print(f"  CSV mapping : {csv_path}")
    print(f"  Unmapped    : {unmapped_path}")
    print(f"  Stats       : {stats_path}")
    print(f"\nSummary: {mapped}/{total_diff} ({(mapped/total_diff*100 if total_diff else 0):.1f}%) words mapped")


if __name__ == "__main__":
    main()

