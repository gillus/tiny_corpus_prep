#!/usr/bin/env python3
"""
Config-driven corpus building (Step 1+2 combined).

Processes each source with its profile, mixes them at the configured weights
with a seeded RNG, and writes shards + stats + a run manifest.

Example:
    python bin/build_corpus.py --config examples/corpus_config.yaml
"""
import argparse
import logging
import sys

from tiny_corpus_prep.build import build_corpus
from tiny_corpus_prep.config import load_config

logger = logging.getLogger(__name__)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build a mixed training corpus from a YAML config"
    )
    ap.add_argument("--config", required=True, help="Path to corpus config YAML")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        config = load_config(args.config)
    except (ValueError, FileNotFoundError) as e:
        logger.error("Invalid config: %s", e)
        return 1

    try:
        manifest = build_corpus(config)
    except Exception as e:
        logger.error("Build failed: %s", e, exc_info=True)
        return 1

    logger.info("")
    logger.info("Corpus build complete!")
    logger.info("  Output dir: %s", config.output.dir)
    logger.info("  Total docs: %s", manifest["mix"]["total_docs"])
    logger.info("  Total words: %s", manifest["mix"]["total_words"])
    for name, stats in manifest["mix"]["per_source"].items():
        logger.info(
            "  %s: %d docs, %d words (%d epoch(s))",
            name, stats["selected_docs"], stats["selected_words"], stats["epochs"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
