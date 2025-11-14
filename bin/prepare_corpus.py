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
from pathlib import Path
from typing import List, Optional

from tiny_corpus_prep.pipeline import process_corpus
from tiny_corpus_prep.annotators import GeminiAnnotator


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
        default=8.0,
        help="Flesch-Kincaid max grade level (default: 8.0)"
    )
    ap.add_argument(
        "--no-max-grade",
        action="store_true",
        help="Disable readability filtering"
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip text normalization"
    )
    ap.add_argument(
        "--no-dedup", 
        action="store_true",
        help="Skip deduplication"
    )
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
    args = parse_args()
    
    # Parse keywords
    keywords = None
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    
    # Determine max_grade
    max_grade = None if args.no_max_grade else args.max_grade
    
    # Setup annotators
    annotators = []
    if args.annotate == "gemini":
        if not args.api_key:
            print("Warning: --api-key not provided for Gemini. Will try to load from environment.")
        try:
            annotators.append(
                GeminiAnnotator(
                    api_key=args.api_key,
                    model_name=args.gemini_model
                )
            )
            print(f"Added Gemini annotator with model: {args.gemini_model}")
        except Exception as e:
            print(f"Error initializing Gemini annotator: {e}")
            return 1
    
    # Process corpus
    try:
        stats = process_corpus(
            input_path=args.input,
            output_path=args.output,
            text_column=args.text_column,
            normalize=not args.no_normalize,
            keywords=keywords,
            max_grade=max_grade,
            synonyms_map_path=args.synonyms,
            annotators=annotators if annotators else None,
            dedup=not args.no_dedup,
            generate_stats=not args.no_stats,
        )
        
        print("\n✓ Processing complete!")
        if stats:
            print(f"  Final rows: {stats.get('total_rows', 'N/A')}")
            print(f"  Output: {args.output}")
            print(f"  Stats: {Path(args.output).with_suffix('.json')}")
        
        return 0
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
