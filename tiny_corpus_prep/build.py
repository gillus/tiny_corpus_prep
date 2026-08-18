"""
Config-driven corpus building: process each source with its profile,
mix with seeded sampling, write shards + stats + a run manifest.

Output layout under output.dir:
    processed/<source>.parquet   per-source filtered data (+ .json stats)
    shards/shard_00000.parquet   final mixed corpus, shard_max_docs docs each
    manifest.json                resolved config, versions, per-stage stats
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import polars as pl

from .common import CEFRIndex
from .config import CorpusConfig, load_config
from .mixing import MixSourceFile, mix_sources_streaming
from .pipeline import process_corpus, process_corpus_chunked

logger = logging.getLogger(__name__)


def _process_sources(config: CorpusConfig, processed_dir: Path) -> Dict[str, Any]:
    """Run each source through its profile; return per-source stats."""
    cefr_cache: Dict[str, CEFRIndex] = {}
    per_source_stats: Dict[str, Any] = {}

    for name, source in config.sources.items():
        profile = config.profiles[source.profile]
        logger.info("=== Processing source '%s' (profile '%s') ===", name, profile.name)

        cefr_index = None
        if profile.cefr_csv:
            if profile.cefr_csv not in cefr_cache:
                cefr_cache[profile.cefr_csv] = CEFRIndex.from_csv(Path(profile.cefr_csv))
            cefr_index = cefr_cache[profile.cefr_csv]

        kwargs = dict(
            input_path=source.input,
            output_path=str(processed_dir / f"{name}.parquet"),
            text_column=config.processing.text_column,
            normalize_mode=profile.normalize_mode,
            keywords=profile.keywords,
            max_grade=profile.max_grade,
            min_chars=profile.min_chars,
            max_chars=profile.max_chars,
            min_words=profile.min_words,
            max_words=profile.max_words,
            cefr_index=cefr_index,
            max_rare_ratio=profile.max_rare_ratio,
            rare_levels=tuple(profile.rare_levels),
            count_unknown_as_rare=profile.count_unknown_as_rare,
            synonyms_map_path=profile.synonyms_map_path,
            dedup_mode=profile.dedup_mode,
            n_workers=config.processing.n_workers,
            paragraph_dedup=profile.paragraph_dedup,
            paragraph_min_chars=profile.paragraph_min_chars,
            minhash_threshold=profile.minhash_threshold,
            minhash_num_perm=profile.minhash_num_perm,
            strip_boilerplate=profile.strip_boilerplate,
            boilerplate_patterns=profile.boilerplate_patterns,
            repetition_filter=profile.repetition_filter,
            repetition_overrides=profile.repetition_overrides,
            language=profile.language,
            language_min_prob=profile.language_min_prob,
        )
        if config.processing.chunk_size:
            stats = process_corpus_chunked(**kwargs, chunk_size=config.processing.chunk_size)
        else:
            stats = process_corpus(**kwargs)
        per_source_stats[name] = {"profile": profile.name, "stats": stats}

    return per_source_stats


def build_corpus(
    config: Union[CorpusConfig, str, Path],
) -> Dict[str, Any]:
    """
    Build a mixed corpus from a config (CorpusConfig or path to a YAML file).

    Returns the run manifest (also written to output.dir/manifest.json).
    """
    if not isinstance(config, CorpusConfig):
        config = load_config(config)

    out_dir = Path(config.output.dir)
    processed_dir = out_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    per_source_stats = _process_sources(config, processed_dir)

    # Mix (streaming: corpus text is never fully held in memory)
    logger.info("=== Mixing %d sources (seed=%d) ===", len(config.sources), config.seed)
    mix_inputs = [
        MixSourceFile(
            name=name,
            path=processed_dir / f"{name}.parquet",
            weight=source.weight,
            allow_upsample=source.allow_upsample,
        )
        for name, source in config.sources.items()
    ]
    mix_stats, shard_files = mix_sources_streaming(
        mix_inputs,
        seed=config.seed,
        output_dir=out_dir / "shards",
        shard_max_docs=config.output.shard_max_docs,
        text_column=config.processing.text_column,
        target_total_words=config.mix.target_total_words,
        cross_source_dedup=config.mix.cross_source_dedup,
    )

    from . import __version__

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "tiny_corpus_prep": __version__,
            "polars": pl.__version__,
            "python": sys.version.split()[0],
        },
        "config": config.to_dict(),
        "sources": per_source_stats,
        "mix": mix_stats,
        "shards": shard_files,
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Manifest written to %s", manifest_path)

    return manifest


def tokenize_corpus(
    config: Union[CorpusConfig, str, Path],
    *,
    retrain_tokenizer: bool = False,
) -> Dict[str, Any]:
    """
    Step 3: train the BPE tokenizer on the mixed shards (or reuse a previously
    trained one), pack the corpus into an HF Arrow DatasetDict, and write a
    tokenization manifest.

    Layout under output.dir:
        tokenizer/tokenizer.json       trained tokenizer (kept with the data)
        dataset/                       DatasetDict with train/validation splits
        tokenization_manifest.json     settings, versions, token-level stats

    Returns the tokenization manifest.
    """
    from .packing import pack_shards_to_dataset
    from .tokenization import train_tokenizer_from_shards

    if not isinstance(config, CorpusConfig):
        config = load_config(config)

    out_dir = Path(config.output.dir)
    shards_dir = out_dir / "shards"
    tokenizer_path = out_dir / "tokenizer" / "tokenizer.json"
    tok_cfg = config.tokenizer
    ds_cfg = config.dataset

    if tokenizer_path.exists() and not retrain_tokenizer:
        logger.info("Reusing existing tokenizer: %s", tokenizer_path)
    else:
        train_tokenizer_from_shards(
            shards_dir,
            tokenizer_path,
            vocab_size=tok_cfg.vocab_size,
            eos_token=tok_cfg.eos_token,
            pad_token=tok_cfg.pad_token,
            extra_special_tokens=tok_cfg.extra_special_tokens,
            require_single_tokens=tok_cfg.require_single_tokens,
            min_frequency=tok_cfg.min_frequency,
            text_column=config.processing.text_column,
        )

    stats = pack_shards_to_dataset(
        shards_dir,
        tokenizer_path,
        out_dir / "dataset",
        seq_length=ds_cfg.seq_length,
        val_fraction=ds_cfg.val_fraction,
        seed=config.seed,
        eos_token=tok_cfg.eos_token,
        text_column=config.processing.text_column,
    )

    import datasets as hf_datasets
    import tokenizers as hf_tokenizers

    from . import __version__

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "tiny_corpus_prep": __version__,
            "polars": pl.__version__,
            "tokenizers": hf_tokenizers.__version__,
            "datasets": hf_datasets.__version__,
            "python": sys.version.split()[0],
        },
        "tokenizer_path": str(tokenizer_path),
        "dataset_path": str(out_dir / "dataset"),
        "config": {
            "seed": config.seed,
            "tokenizer": config.to_dict()["tokenizer"],
            "dataset": config.to_dict()["dataset"],
        },
        "stats": stats,
    }
    manifest_path = out_dir / "tokenization_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Tokenization manifest written to %s", manifest_path)

    return manifest
