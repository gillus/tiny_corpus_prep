"""
Seeded mixing of processed sources into a single training corpus.

Weights are interpreted as target *word* shares (words ≈ proxy for tokens
until the tokenizer stage exists). Selection and the final shuffle are driven
by a single seeded RNG, so the same inputs + config + seed reproduce the
exact same corpus.

Two implementations with identical selection semantics:
- mix_sources: in-memory DataFrames (small corpora, tests, notebooks);
- mix_sources_streaming: file-to-file, never holds the corpus text in memory —
  use this at scale (build_corpus uses it).
"""
from __future__ import annotations

import logging
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import polars as pl
import pyarrow.parquet as pq

from .dedup import _text_hash

logger = logging.getLogger(__name__)

_WORD_RE = r"\S+"


@dataclass
class MixSource:
    """One processed source ready for mixing."""
    name: str
    df: pl.DataFrame
    weight: float
    allow_upsample: bool = False


def _select_rows(
    n_rows: int,
    words: Sequence[int],
    target_words: float,
    rng: random.Random,
    allow_upsample: bool,
) -> Tuple[List[int], int, int]:
    """
    Greedily take shuffled rows until *target_words* is reached.

    Returns (selected_indices, selected_words, epochs). With upsampling, rows
    repeat across reshuffled epochs; the last epoch may be partial.
    """
    if n_rows == 0 or sum(words) == 0:
        return [], 0, 0

    order = list(range(n_rows))
    rng.shuffle(order)
    selected: List[int] = []
    total = 0
    epochs = 1
    pos = 0
    while total < target_words:
        if pos >= len(order):
            if not allow_upsample:
                break
            rng.shuffle(order)
            pos = 0
            epochs += 1
        i = order[pos]
        pos += 1
        selected.append(i)
        total += words[i]
    return selected, total, epochs


def _resolve_total_target(
    sources_meta: List[Tuple[str, float, bool]],  # (name, weight, allow_upsample)
    available: Dict[str, int],
    weight_sum: float,
    target_total_words: Optional[int],
) -> int:
    """Resolve the total word target and validate feasibility."""
    if target_total_words is None:
        caps = [
            available[name] / (weight / weight_sum)
            for name, weight, allow_upsample in sources_meta
            if not allow_upsample
        ]
        if not caps:
            raise ValueError(
                "mix_sources: all sources allow upsampling; "
                "set target_total_words explicitly"
            )
        return int(min(caps))

    for name, weight, allow_upsample in sources_meta:
        needed = (weight / weight_sum) * target_total_words
        if not allow_upsample and needed > available[name]:
            raise ValueError(
                f"mix_sources: source '{name}' has {available[name]} words "
                f"but needs {int(needed)} for the target; "
                f"set allow_upsample or lower target_total_words"
            )
    return target_total_words


def mix_sources(
    sources: List[MixSource],
    *,
    seed: int,
    text_column: str = "text",
    target_total_words: Optional[int] = None,
    cross_source_dedup: bool = True,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Combine processed sources into one shuffled corpus at the given weights.

    Args:
        sources: Processed sources with weights (any positive scale; normalized
                 internally). List order sets dedup priority: a doc appearing in
                 several sources is kept in the earliest-listed one.
        seed: RNG seed driving dedup-independent selection and the final shuffle.
        text_column: Name of the text column (all other columns are dropped;
                     a 'source' column is added).
        target_total_words: Total corpus size in words. None = the largest total
                 achievable without upsampling any non-upsample source.
        cross_source_dedup: Remove exact duplicates across sources before
                 selection (upsample repeats are intentional and exempt).

    Returns:
        (mixed DataFrame with columns [text_column, 'source'], mix statistics)
    """
    if not sources:
        raise ValueError("mix_sources: at least one source is required")

    rng = random.Random(seed)
    weight_sum = sum(s.weight for s in sources)
    if weight_sum <= 0:
        raise ValueError("mix_sources: weights must sum to a positive value")

    # Prepare each source: keep only the text column, count words,
    # drop cross-source duplicates (earlier-listed source wins).
    prepared: List[Tuple[MixSource, pl.DataFrame]] = []
    seen_hashes: set = set()
    for src in sources:
        if text_column not in src.df.columns:
            raise ValueError(
                f"mix_sources: source '{src.name}' has no '{text_column}' column"
            )
        df = src.df.select(
            pl.col(text_column),
            pl.col(text_column).str.count_matches(_WORD_RE).cast(pl.Int64).alias("_words"),
        )
        if cross_source_dedup:
            keep = []
            removed = 0
            for text in df[text_column]:
                h = _text_hash(text)
                if h in seen_hashes:
                    keep.append(False)
                    removed += 1
                else:
                    seen_hashes.add(h)
                    keep.append(True)
            if removed:
                logger.info(
                    "Mix: dropped %d cross-source duplicates from '%s'", removed, src.name
                )
                df = df.filter(pl.Series(keep))
        prepared.append((src, df))

    available = {src.name: int(df["_words"].sum()) for src, df in prepared}

    total_target = _resolve_total_target(
        [(src.name, src.weight, src.allow_upsample) for src, _ in prepared],
        available, weight_sum, target_total_words,
    )

    # Seeded per-source selection.
    parts: List[pl.DataFrame] = []
    per_source: Dict[str, Any] = {}
    for src, df in prepared:
        target = (src.weight / weight_sum) * total_target
        words = df["_words"].to_list()
        selected, sel_words, epochs = _select_rows(
            len(df), words, target, rng, src.allow_upsample
        )
        logger.info(
            "Mix: '%s' → %d docs, %d words (target %d, %d epoch(s))",
            src.name, len(selected), sel_words, int(target), epochs,
        )
        per_source[src.name] = {
            "available_words": available[src.name],
            "target_words": int(target),
            "selected_words": sel_words,
            "selected_docs": len(selected),
            "unique_docs": len(set(selected)),
            "epochs": epochs,
        }
        if selected:
            parts.append(
                df[selected]
                .select(pl.col(text_column))
                .with_columns(pl.lit(src.name).alias("source"))
            )

    if not parts:
        raise ValueError("mix_sources: no documents selected from any source")

    mixed = pl.concat(parts)

    # Seeded global shuffle.
    order = list(range(len(mixed)))
    rng.shuffle(order)
    mixed = mixed[order]

    stats = {
        "seed": seed,
        "target_total_words": total_target,
        "total_docs": len(mixed),
        "total_words": sum(s["selected_words"] for s in per_source.values()),
        "cross_source_dedup": cross_source_dedup,
        "per_source": per_source,
    }
    return mixed, stats


# ── streaming mixing ─────────────────────────────────────────────


@dataclass
class MixSourceFile:
    """One processed source, referenced by file path (streaming mixing)."""
    name: str
    path: Union[str, Path]
    weight: float
    allow_upsample: bool = False


def _iter_parquet_batches(
    path: Union[str, Path],
    columns: List[str],
    batch_size: int,
) -> Iterator[pl.DataFrame]:
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        df = pl.from_arrow(batch)
        if len(df) > 0:
            yield df


class _ShardRouter:
    """Routes (position, text, source) rows into per-shard temp parquet files."""

    def __init__(self, tmp_dir: Path, n_shards: int, buffer_rows: int = 2048):
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.n_shards = n_shards
        self.buffer_rows = buffer_rows
        self._buffers: Dict[int, Dict[str, list]] = {}
        self._writers: Dict[int, pq.ParquetWriter] = {}

    def add(self, shard_id: int, pos: int, text: str, source: str) -> None:
        buf = self._buffers.setdefault(
            shard_id, {"__pos": [], "text": [], "source": []}
        )
        buf["__pos"].append(pos)
        buf["text"].append(text)
        buf["source"].append(source)
        if len(buf["__pos"]) >= self.buffer_rows:
            self._flush(shard_id)

    def _flush(self, shard_id: int) -> None:
        buf = self._buffers.pop(shard_id, None)
        if not buf or not buf["__pos"]:
            return
        table = pl.DataFrame({
            "__pos": pl.Series(buf["__pos"], dtype=pl.Int64),
            "text": pl.Series(buf["text"], dtype=pl.Utf8),
            "source": pl.Series(buf["source"], dtype=pl.Utf8),
        }).to_arrow()
        writer = self._writers.get(shard_id)
        if writer is None:
            writer = pq.ParquetWriter(
                str(self.tmp_dir / f"tmp_{shard_id:05d}.parquet"), table.schema
            )
            self._writers[shard_id] = writer
        writer.write_table(table)

    def close(self) -> None:
        for shard_id in list(self._buffers):
            self._flush(shard_id)
        for writer in self._writers.values():
            writer.close()
        self._writers = {}

    def tmp_path(self, shard_id: int) -> Path:
        return self.tmp_dir / f"tmp_{shard_id:05d}.parquet"


def mix_sources_streaming(
    sources: List[MixSourceFile],
    *,
    seed: int,
    output_dir: Union[str, Path],
    shard_max_docs: int,
    text_column: str = "text",
    target_total_words: Optional[int] = None,
    cross_source_dedup: bool = True,
    batch_size: int = 10_000,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Streaming version of mix_sources: reads sources from parquet files and
    writes shuffled shards directly, never holding the corpus in memory.

    Memory footprint is O(documents) for per-doc word counts / hashes /
    selected positions (a few dozen bytes per doc), plus one shard of text at
    a time — not O(corpus text).

    Same selection semantics as mix_sources: word-share weights, seeded
    selection, upsampling epochs, cross-source exact dedup with earlier-listed
    sources taking priority. The global shuffle assigns every selected doc a
    random position, so shard membership and within-shard order are seeded.

    Returns:
        (mix statistics, [{"file": shard_name, "docs": n}, ...])
    """
    if not sources:
        raise ValueError("mix_sources: at least one source is required")

    rng = random.Random(seed)
    weight_sum = sum(s.weight for s in sources)
    if weight_sum <= 0:
        raise ValueError("mix_sources: weights must sum to a positive value")

    # Pass 1 — stream each source once: per-doc word counts + cross-source
    # exact dedup (earlier-listed source keeps shared docs).
    per_source_words: List[List[int]] = []
    per_source_eligible: List[List[int]] = []
    seen_hashes: set = set()
    for src in sources:
        words: List[int] = []
        eligible: List[int] = []
        removed = 0
        row_idx = 0
        for batch in _iter_parquet_batches(src.path, [text_column], batch_size):
            batch_words = (
                batch[text_column].str.count_matches(_WORD_RE).cast(pl.Int64).to_list()
            )
            texts = batch[text_column].to_list() if cross_source_dedup else None
            for i, w in enumerate(batch_words):
                words.append(w)
                if cross_source_dedup:
                    h = _text_hash(texts[i])
                    if h in seen_hashes:
                        removed += 1
                        row_idx += 1
                        continue
                    seen_hashes.add(h)
                eligible.append(row_idx)
                row_idx += 1
        if removed:
            logger.info(
                "Mix: dropped %d cross-source duplicates from '%s'", removed, src.name
            )
        per_source_words.append(words)
        per_source_eligible.append(eligible)

    available = {
        src.name: int(sum(per_source_words[i][j] for j in per_source_eligible[i]))
        for i, src in enumerate(sources)
    }
    total_target = _resolve_total_target(
        [(s.name, s.weight, s.allow_upsample) for s in sources],
        available, weight_sum, target_total_words,
    )

    # Seeded per-source selection (same RNG consumption order as sources list).
    per_source_selected: List[List[int]] = []
    per_source_stats: Dict[str, Any] = {}
    for i, src in enumerate(sources):
        target = (src.weight / weight_sum) * total_target
        eligible = per_source_eligible[i]
        eligible_words = [per_source_words[i][j] for j in eligible]
        positions, sel_words, epochs = _select_rows(
            len(eligible), eligible_words, target, rng, src.allow_upsample
        )
        selected = [eligible[p] for p in positions]
        logger.info(
            "Mix: '%s' → %d docs, %d words (target %d, %d epoch(s))",
            src.name, len(selected), sel_words, int(target), epochs,
        )
        per_source_selected.append(selected)
        per_source_stats[src.name] = {
            "available_words": available[src.name],
            "target_words": int(target),
            "selected_words": sel_words,
            "selected_docs": len(selected),
            "unique_docs": len(set(selected)),
            "epochs": epochs,
        }

    total_docs = sum(len(s) for s in per_source_selected)
    if total_docs == 0:
        raise ValueError("mix_sources: no documents selected from any source")

    # Global shuffle: assign every selected doc a random output position.
    positions = list(range(total_docs))
    rng.shuffle(positions)
    n_shards = (total_docs + shard_max_docs - 1) // shard_max_docs

    # Per source: row index → list of output positions (repeats = upsampling).
    occurrence_maps: List[Dict[int, List[int]]] = []
    offset = 0
    for selected in per_source_selected:
        occ: Dict[int, List[int]] = {}
        for j, row_idx in enumerate(selected):
            occ.setdefault(row_idx, []).append(positions[offset + j])
        occurrence_maps.append(occ)
        offset += len(selected)

    # Pass 2 — stream each source again, routing selected rows to shard files.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router = _ShardRouter(output_dir / "_tmp_mix", n_shards)
    try:
        for src, occ in zip(sources, occurrence_maps):
            row_idx = 0
            for batch in _iter_parquet_batches(src.path, [text_column], batch_size):
                texts = batch[text_column].to_list()
                for text in texts:
                    for pos in occ.get(row_idx, ()):
                        router.add(pos // shard_max_docs, pos, text, src.name)
                    row_idx += 1
    finally:
        router.close()

    # Finalize: order each shard by position and write the final file.
    shard_files: List[Dict[str, Any]] = []
    for shard_id in range(n_shards):
        shard_df = (
            pl.read_parquet(router.tmp_path(shard_id))
            .sort("__pos")
            .select("text", "source")
        )
        shard_name = f"shard_{shard_id:05d}.parquet"
        shard_df.write_parquet(output_dir / shard_name)
        shard_files.append({"file": shard_name, "docs": len(shard_df)})
    shutil.rmtree(router.tmp_dir)
    logger.info("Mix: wrote %d shard(s), %d docs total", n_shards, total_docs)

    stats = {
        "seed": seed,
        "target_total_words": total_target,
        "total_docs": total_docs,
        "total_words": sum(s["selected_words"] for s in per_source_stats.values()),
        "cross_source_dedup": cross_source_dedup,
        "n_shards": n_shards,
        "per_source": per_source_stats,
    }
    return stats, shard_files
