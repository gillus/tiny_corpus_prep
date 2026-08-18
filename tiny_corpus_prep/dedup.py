"""
Deduplication utilities.

- exact_dedup: hash-based exact duplicate removal (supports cross-chunk hash set).
- MinHashDeduper / minhash_dedup: near-duplicate removal via MinHash LSH,
  streamable chunk by chunk (requires datasketch).
- paragraph_dedup: drop repeated paragraphs (boilerplate) across documents.
"""
from __future__ import annotations

import functools
import hashlib
import logging
from typing import List, Optional, Set

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
        # Simple in-frame dedup. maintain_order keeps output row order
        # deterministic, which downstream order-sensitive steps (MinHash LSH)
        # rely on for reproducibility.
        return df.unique(subset=[text_column], keep="first", maintain_order=True)

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


def _build_minhash(text: str, num_perm: int, ngram_size: int):
    """Build one MinHash signature (module-level so it can run in a pool)."""
    from datasketch import MinHash

    mh = MinHash(num_perm=num_perm)
    if text:
        for j in range(max(len(text) - ngram_size + 1, 1)):
            mh.update(text[j : j + ngram_size].encode("utf-8"))
    return mh


class MinHashDeduper:
    """
    Stateful near-duplicate filter using MinHash LSH.

    Call filter_chunk() repeatedly over corpus chunks: the LSH index persists
    across calls, so near-duplicates are caught across chunk boundaries without
    ever holding the corpus text in memory (only the index grows, ~O(docs kept)).
    Greedy keep-first semantics: results depend on document order, so keep
    input order deterministic for reproducible runs.

    Requires the ``datasketch`` package (optional dependency).
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        ngram_size: int = 5,
    ):
        try:
            from datasketch import MinHashLSH
        except ImportError:
            raise ImportError(
                "MinHash dedup requires datasketch. "
                "Install with: pip install 'tiny_corpus_prep[dedup]'"
            )
        self.threshold = threshold
        self.num_perm = num_perm
        self.ngram_size = ngram_size
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._next_key = 0

    def filter_chunk(
        self,
        df: pl.DataFrame,
        text_column: str = "text",
        mapper=None,
    ) -> pl.DataFrame:
        """
        Remove docs that near-duplicate anything seen so far (this chunk or
        earlier ones). Signature building can be parallelized by passing a
        parallel.TextMapper; querying/inserting stays serial (ordered).
        """
        texts = df[text_column].to_list()
        if not texts:
            return df

        build = functools.partial(
            _build_minhash, num_perm=self.num_perm, ngram_size=self.ngram_size
        )
        if mapper is not None:
            minhashes = mapper.map(build, texts)
        else:
            minhashes = [build(t) for t in texts]

        keep_mask: List[bool] = []
        kept = 0
        for mh in minhashes:
            if self._lsh.query(mh):
                keep_mask.append(False)
                continue
            self._lsh.insert(str(self._next_key), mh)
            self._next_key += 1
            kept += 1
            keep_mask.append(True)

        logger.info(
            "MinHash dedup: kept %d / %d documents in chunk (index size: %d)",
            kept, len(texts), self._next_key,
        )
        return df.filter(pl.Series(keep_mask))


def minhash_dedup(
    df: pl.DataFrame,
    text_column: str = "text",
    threshold: float = 0.8,
    num_perm: int = 128,
    ngram_size: int = 5,
) -> pl.DataFrame:
    """
    Remove near-duplicates within one DataFrame using MinHash LSH.

    One-shot wrapper around MinHashDeduper; use the class directly for
    cross-chunk streaming dedup.
    """
    deduper = MinHashDeduper(
        threshold=threshold, num_perm=num_perm, ngram_size=ngram_size
    )
    return deduper.filter_chunk(df, text_column)


def paragraph_dedup(
    df: pl.DataFrame,
    text_column: str = "text",
    seen_hashes: Optional[Set[int]] = None,
    min_chars: int = 50,
) -> pl.DataFrame:
    """
    Remove paragraphs that already appeared in an earlier document
    (repeated boilerplate: navigation blocks, license footers,
    "References" sections, …).

    Paragraphs are blank-line-separated blocks. Only paragraphs with at least
    *min_chars* characters are eligible for removal — short repeated lines
    (headings, list items) are legitimate text and always kept. Documents left
    empty after removal are dropped.

    Args:
        df: Input Polars DataFrame
        text_column: Name of the text column
        seen_hashes: Optional mutable set of paragraph hashes for
                     cross-chunk state. Updated in place.
        min_chars: Minimum paragraph length (in chars) to consider for dedup.

    Returns:
        DataFrame with deduplicated paragraph content.
    """
    seen = seen_hashes if seen_hashes is not None else set()

    new_texts: List[str] = []
    keep_mask: List[bool] = []
    removed_paragraphs = 0
    for text in df[text_column]:
        kept_parts: List[str] = []
        for paragraph in text.split("\n\n"):
            stripped = paragraph.strip()
            if len(stripped) >= min_chars:
                h = _text_hash(stripped)
                if h in seen:
                    removed_paragraphs += 1
                    continue
                seen.add(h)
            kept_parts.append(paragraph)
        new_text = "\n\n".join(kept_parts).strip()
        keep_mask.append(bool(new_text))
        new_texts.append(new_text)

    if removed_paragraphs:
        logger.info(
            "Paragraph dedup: removed %d repeated paragraphs", removed_paragraphs
        )

    return (
        df.with_columns(pl.Series(text_column, new_texts, dtype=pl.Utf8))
        .filter(pl.Series(keep_mask))
    )
