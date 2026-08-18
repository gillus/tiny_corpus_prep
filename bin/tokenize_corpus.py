#!/usr/bin/env python3
"""
Tokenizer training + packing (Step 3).

Reads the mixed shards produced by bin/build_corpus.py, trains a byte-level
BPE tokenizer (unless one already exists), packs the corpus into a
HuggingFace Arrow DatasetDict with EOS-joined fixed-length sequences, and
writes token-level statistics.

Example:
    python bin/tokenize_corpus.py --config examples/corpus_config.yaml

Loading the result for training:
    from datasets import load_from_disk
    ds = load_from_disk("<output.dir>/dataset")   # ds["train"]["input_ids"]
"""
import argparse
import logging
import sys

from tiny_corpus_prep.build import tokenize_corpus
from tiny_corpus_prep.config import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train tokenizer and pack corpus into an HF Arrow dataset"
    )
    ap.add_argument("--config", required=True, help="Path to corpus config YAML")
    ap.add_argument(
        "--retrain",
        action="store_true",
        help="Retrain the tokenizer even if tokenizer.json already exists",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid config: %s", e)
        return 1

    try:
        manifest = tokenize_corpus(config, retrain_tokenizer=args.retrain)
    except Exception as e:
        logger.error("Tokenization failed: %s", e, exc_info=True)
        return 1

    stats = manifest["stats"]
    logger.info("")
    logger.info("Tokenization complete!")
    logger.info("  Tokenizer: %s (vocab %d)", manifest["tokenizer_path"], stats["vocab_size"])
    logger.info("  Dataset: %s", manifest["dataset_path"])
    logger.info("  Vocab coverage: %.1f%%", 100 * stats["vocab_coverage"])
    logger.info("  Chars/token: %s", stats["chars_per_token"])
    for split, s in stats["splits"].items():
        logger.info(
            "  %s: %s sequences × %d tokens (%s docs)",
            split, s["sequences"], stats["seq_length"], s["documents"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
