#!/usr/bin/env python3
"""
CLI for corpus preparation using Polars.

Installation:
    uv pip install -e ".[annotators]"

Example:
python bin/prepare_corpus.py \
  --input data.parquet \
  --output out.parquet \
  --synonyms example_synonyms.json \
  --keywords math,science \
  --max-grade 8 \
  --annotate gemini \
  --api-key YOUR_API_KEY
"""
import argparse
import logging
from pathlib import Path
from typing import List, Optional

from tiny_corpus_prep.pipeline import process_corpus, process_corpus_chunked
from tiny_corpus_prep.annotators import GeminiAnnotator
from tiny_corpus_prep.common import CEFRIndex

logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Process text corpus with filtering and annotation"
    )
    ap.add_argument(
        "--input", 
        required=True, 
        help="Input parquet file with 'text' column"
    )
    ap.add_argument(
        "--output", 
        required=True, 
        help="Output parquet file"
    )
    ap.add_argument(
        "--text-column", 
        default="text",
        help="Name of text column (default: text)"
    )
    ap.add_argument(
        "--synonyms", 
        help="Path to synonyms map JSON {synonym: canonical}"
    )
    ap.add_argument(
        "--keywords", 
        help="Comma-separated keywords for topic filter"
    )
    ap.add_argument(
        "--max-grade",
        type=float,
        default=10.0,
        help="Flesch-Kincaid max grade level (default: 10.0)"
    )
    ap.add_argument(
        "--no-max-grade",
        action="store_true",
        help="Disable readability filtering"
    )
    # Length filters
    ap.add_argument("--min-chars", type=int, default=None, help="Minimum document character count")
    ap.add_argument("--max-chars", type=int, default=None, help="Maximum document character count")
    ap.add_argument("--min-words", type=int, default=None, help="Minimum document word count")
    ap.add_argument("--max-words", type=int, default=None, help="Maximum document word count")
    # Vocabulary complexity filters
    ap.add_argument("--cefr-csv", default=None, help="Path to CEFR CSV for vocabulary complexity filter")
    ap.add_argument("--max-rare-ratio", type=float, default=0.3, help="Max rare-word ratio (default: 0.3)")
    ap.add_argument("--count-unknown-as-rare", action="store_true", help="Count words not in CEFR index as rare")
    ap.add_argument(
        "--normalize-mode",
        choices=["gentle", "aggressive", "none"],
        default="gentle",
        help="Text normalization mode (default: gentle)"
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization (alias for --normalize-mode none)"
    )
    # Dedup options
    ap.add_argument(
        "--dedup-mode",
        choices=["exact", "minhash", "both", "none"],
        default="exact",
        help="Deduplication mode (default: exact)"
    )
    ap.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip deduplication (alias for --dedup-mode none)"
    )
    ap.add_argument("--minhash-threshold", type=float, default=0.8, help="MinHash Jaccard threshold (default: 0.8)")
    ap.add_argument("--minhash-num-perm", type=int, default=128, help="MinHash permutations (default: 128)")
    # Paragraph dedup
    ap.add_argument("--paragraph-dedup", action="store_true", help="Remove paragraphs repeated across documents")
    ap.add_argument("--paragraph-min-chars", type=int, default=50, help="Min paragraph length eligible for dedup (default: 50)")
    # Quality heuristics
    ap.add_argument("--strip-boilerplate", action="store_true", help="Remove boilerplate lines (cookie banners, share buttons, ...)")
    ap.add_argument("--repetition-filter", action="store_true", help="Drop docs with degenerate repetition (Gopher-style signals)")
    ap.add_argument("--language", default=None, help="Keep only docs in this language, e.g. 'en' (requires langdetect)")
    ap.add_argument("--language-min-prob", type=float, default=0.8, help="Min language detection probability (default: 0.8)")
    # Parallelism / chunked processing
    ap.add_argument("--n-workers", type=int, default=1, help="Worker processes for per-doc operations (default: 1)")
    ap.add_argument("--chunk-size", type=int, default=None, help="Process in chunks of N rows (enables chunked mode)")
    ap.add_argument(
        "--no-stats",
        action="store_true",
        help="Don't generate statistics JSON"
    )
    ap.add_argument(
        "--annotate",
        choices=["gemini"],
        help="Add annotation using specified annotator"
    )
    ap.add_argument(
        "--api-key",
        help="API key for annotator (e.g., Google API key for Gemini)"
    )
    ap.add_argument(
        "--gemini-model",
        default="gemini-2.5-flash-lite",
        help="Gemini model name (default: gemini-2.5-flash-lite)"
    )
    return ap.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    # Parse keywords
    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    
    # Determine normalize_mode
    normalize_mode = None if args.no_normalize else args.normalize_mode
    if normalize_mode == "none":
        normalize_mode = None

    # Determine max_grade
    max_grade = None if args.no_max_grade else args.max_grade

    # Determine dedup_mode
    dedup_mode = None if args.no_dedup else args.dedup_mode
    if dedup_mode == "none":
        dedup_mode = None

    # Load CEFR index if provided
    cefr_index = None
    if args.cefr_csv:
        from pathlib import Path as _P
        cefr_index = CEFRIndex.from_csv(_P(args.cefr_csv))
        logger.info("Loaded CEFR index from %s", args.cefr_csv)

    # Setup annotators
    annotators = []
    if args.annotate == "gemini":
        if not args.api_key:
            logger.warning("--api-key not provided for Gemini. Will try to load from environment.")
        try:
            annotators.append(
                GeminiAnnotator(
                    api_key=args.api_key,
                    model_name=args.gemini_model
                )
            )
            logger.info("Added Gemini annotator with model: %s", args.gemini_model)
        except Exception as e:
            logger.error("Error initializing Gemini annotator: %s", e)
            return 1
    
    # Process corpus
    try:
        common_kwargs = dict(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            normalize_mode=normalize_mode,
            keywords=keywords,
            max_grade=max_grade,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            min_words=args.min_words,
            max_words=args.max_words,
            cefr_index=cefr_index,
            max_rare_ratio=args.max_rare_ratio,
            count_unknown_as_rare=args.count_unknown_as_rare,
            synonyms_map_path=args.synonyms,
            dedup_mode=dedup_mode,
            n_workers=args.n_workers,
            paragraph_dedup=args.paragraph_dedup,
            paragraph_min_chars=args.paragraph_min_chars,
            minhash_threshold=args.minhash_threshold,
            minhash_num_perm=args.minhash_num_perm,
            strip_boilerplate=args.strip_boilerplate,
            repetition_filter=args.repetition_filter,
            language=args.language,
            language_min_prob=args.language_min_prob,
        )
        if args.chunk_size:
            stats = process_corpus_chunked(
                **common_kwargs,
                chunk_size=args.chunk_size,
            )
        else:
            stats = process_corpus(
                **common_kwargs,
                annotators=annotators if annotators else None,
                generate_stats=not args.no_stats,
            )
        
        logger.info("Processing complete!")
        if stats:
            logger.info("  Final rows: %s", stats.get('total_rows', 'N/A'))
            logger.info("  Output: %s", args.output)
            logger.info("  Stats: %s", Path(args.output).with_suffix('.json'))
        
        return 0
    
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
