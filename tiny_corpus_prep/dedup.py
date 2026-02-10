"""
Deduplication utilities.

- exact_dedup: hash-based exact duplicate removal (supports cross-chunk hash set).
- minhash_dedup: near-duplicate removal via MinHash LSH (requires datasketch).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional, Set

import polars as pl

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> int:
    """Return a 64-bit hash for a text string (for exact dedup)."""
    return int.from_bytes(
        hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(),
        byteorder="big",
    )


def exact_dedup(
    df: pl.DataFrame,
    text_column: str = "text",
    seen_hashes: Optional[Set[int]] = None,
) -> pl.DataFrame:
    """
    Remove exact duplicates by text hash.

    Args:
        df: Input Polars DataFrame
        text_column: Name of the text column
        seen_hashes: Optional mutable set of hashes already seen
                     (for cross-chunk dedup). Will be updated in-place.

    Returns:
        DataFrame with duplicates removed
    """
    if seen_hashes is None:
        # Simple in-frame dedup
        return df.unique(subset=[text_column], keep="first")

    # Cross-chunk dedup: compute hash per row, keep only new ones
    # Use Python-level hashing to avoid Int64 overflow (values > 2^63 become null in pl.Int64)
    hashes = [_text_hash(t) for t in df[text_column].to_list()]
    keep_mask = []
    for h in hashes:
        if h in seen_hashes:
            keep_mask.append(False)
        else:
            seen_hashes.add(h)
            keep_mask.append(True)

    return df.filter(pl.Series(keep_mask))


def minhash_dedup(
    df: pl.DataFrame,
    text_column: str = "text",
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
) -> pl.DataFrame:
    """
    Remove near-duplicates using MinHash LSH.

    Requires the ``datasketch`` package (optional dependency).

    Args:
        df: Input Polars DataFrame
        text_column: Name of the text column
        threshold: Jaccard similarity threshold (0.0–1.0)
        num_perm: Number of permutations for MinHash
        ngram_size: Character n-gram size for shingling

    Returns:
        DataFrame with near-duplicates removed
    """
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        raise ImportError(
            "minhash_dedup requires datasketch. "
            "Install with: pip install 'tiny_corpus_prep[dedup]'"
        )

    texts = df[text_column].to_list()
    logger.info("Building MinHash signatures for %d documents...", len(texts))

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = []
    for i, text in enumerate(texts):
        mh = MinHash(num_perm=num_perm)
        if text:
            for j in range(max(len(text) - ngram_size + 1, 1)):
                mh.update(text[j : j + ngram_size].encode("utf-8"))
        minhashes.append(mh)

    # Insert into LSH, tracking which indices to keep
    keep_indices: list[int] = []
    for i, mh in enumerate(minhashes):
        key = str(i)
        # Query first — if there's already a near-dup, skip this doc
        result = lsh.query(mh)
        if result:
            continue  # near-duplicate found, skip
        try:
            lsh.insert(key, mh)
        except ValueError:
            # Duplicate key — shouldn't happen, but handle gracefully
            continue
        keep_indices.append(i)

    logger.info(
        "MinHash dedup: kept %d / %d documents (removed %d near-duplicates)",
        len(keep_indices), len(texts), len(texts) - len(keep_indices),
    )
    return df[keep_indices]
