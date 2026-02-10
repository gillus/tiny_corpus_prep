\
"""
End-to-end pipeline using Polars DataFrames.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import logging
import polars as pl

logger = logging.getLogger(__name__)

from .normalize import normalize_text
from .synonyms import SynonymMapper
from .filters import (
    filter_by_keywords,
    filter_by_readability,
    filter_by_length,
    filter_by_vocabulary_complexity,
)
from .common import CEFRIndex
from .annotators import BaseAnnotator
from .dedup import exact_dedup, minhash_dedup
from . import io as io_mod


class DataPipeline:
    """
    Main pipeline for processing text data with Polars.
    
    Supports:
    - Normalization
    - Keyword filtering
    - Readability filtering
    - Synonym mapping
    - Custom annotations
    - Deduplication
    """
    
    def __init__(
        self,
        text_column: str = "text",
        normalize: Union[bool, str, None] = True,
        dedup: Union[bool, str, None] = True,
        normalize_mode: Optional[str] = None,
        dedup_mode: Optional[str] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            text_column: Name of the text column
            normalize: Backwards-compat flag. True → "gentle", False → None.
                       Ignored when normalize_mode is explicitly set.
            dedup: Backwards-compat. True → "exact", False → None.
            normalize_mode: "gentle", "aggressive", or None for no normalization.
            dedup_mode: "exact", "minhash", "both", or None. Overrides dedup.
        """
        self.text_column = text_column
        # Resolve normalize_mode: explicit param wins, then coerce bool
        if normalize_mode is not None:
            self.normalize_mode = normalize_mode
        elif isinstance(normalize, str):
            self.normalize_mode = normalize
        elif normalize:
            self.normalize_mode = "gentle"
        else:
            self.normalize_mode = None
        # Resolve dedup_mode
        if dedup_mode is not None:
            self.dedup_mode = dedup_mode
        elif isinstance(dedup, str):
            self.dedup_mode = dedup
        elif dedup:
            self.dedup_mode = "exact"
        else:
            self.dedup_mode = None
        self.filters = []
        self.annotators = []
        self.synonym_mapper: Optional[SynonymMapper] = None
    
    def add_keyword_filter(self, keywords: List[str]) -> "DataPipeline":
        """
        Add keyword filter to pipeline.
        
        Args:
            keywords: List of keywords to filter by
            
        Returns:
            Self for chaining
        """
        if keywords:
            self.filters.append(("keywords", keywords))
        return self
    
    def add_readability_filter(self, max_grade: float = 10.0) -> "DataPipeline":
        """
        Add readability filter to pipeline.
        
        Args:
            max_grade: Maximum Flesch-Kincaid grade level
            
        Returns:
            Self for chaining
        """
        self.filters.append(("readability", max_grade))
        return self

    def add_length_filter(
        self,
        *,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        min_words: Optional[int] = None,
        max_words: Optional[int] = None,
    ) -> "DataPipeline":
        """Add document-length filter to pipeline."""
        self.filters.append(("length", {
            "min_chars": min_chars,
            "max_chars": max_chars,
            "min_words": min_words,
            "max_words": max_words,
        }))
        return self

    def add_vocabulary_filter(
        self,
        cefr_index: CEFRIndex,
        *,
        max_rare_ratio: float = 0.3,
        rare_levels: tuple = ("B2", "C1", "C2"),
        count_unknown_as_rare: bool = False,
    ) -> "DataPipeline":
        """Add vocabulary complexity filter to pipeline."""
        self.filters.append(("vocabulary", {
            "cefr_index": cefr_index,
            "max_rare_ratio": max_rare_ratio,
            "rare_levels": rare_levels,
            "count_unknown_as_rare": count_unknown_as_rare,
        }))
        return self

    def add_synonym_mapper(
        self, 
        mapping: Optional[Dict[str, str]] = None,
        mapping_path: Optional[str] = None
    ) -> "DataPipeline":
        """
        Add synonym mapping to pipeline.
        
        Args:
            mapping: Dictionary of synonym -> canonical mappings
            mapping_path: Path to JSON file with mappings
            
        Returns:
            Self for chaining
        """
        if mapping:
            self.synonym_mapper = SynonymMapper(mapping)
        elif mapping_path:
            self.synonym_mapper = SynonymMapper.from_json(mapping_path)
        return self
    
    def add_annotator(self, annotator: BaseAnnotator) -> "DataPipeline":
        """
        Add custom annotator to pipeline.
        
        Args:
            annotator: Instance of BaseAnnotator
            
        Returns:
            Self for chaining
        """
        self.annotators.append(annotator)
        return self
    
    def process(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process a DataFrame through the pipeline.
        
        Args:
            df: Input Polars DataFrame with text column
            
        Returns:
            Processed DataFrame
        """
        # Validate text column exists
        if self.text_column not in df.columns:
            raise ValueError(
                f"Text column '{self.text_column}' not found in DataFrame. "
                f"Available columns: {df.columns}"
            )
        
        logger.info("Starting pipeline with %d rows...", len(df))
        
        # Step 1: Normalize text if enabled
        if self.normalize_mode:
            logger.info("Normalizing text (mode=%s)...", self.normalize_mode)
            mode = self.normalize_mode
            df = df.with_columns([
                pl.col(self.text_column)
                .map_elements(lambda s: normalize_text(s, mode=mode), return_dtype=pl.Utf8)
                .alias(self.text_column)
            ])
            # Remove empty rows after normalization
            df = df.filter(
                pl.col(self.text_column).is_not_null() & 
                (pl.col(self.text_column).str.strip_chars() != "")
            )
            logger.info("After normalization: %d rows", len(df))
        
        # Step 2: Apply filters
        for filter_type, filter_arg in self.filters:
            if filter_type == "keywords":
                logger.info("Applying keyword filter (%d keywords)...", len(filter_arg))
                df = filter_by_keywords(df, filter_arg, self.text_column)
                logger.info("After keyword filter: %d rows", len(df))
            
            elif filter_type == "readability":
                logger.info("Applying readability filter (max grade: %s)...", filter_arg)
                df = filter_by_readability(df, filter_arg, self.text_column)
                logger.info("After readability filter: %d rows", len(df))

            elif filter_type == "length":
                logger.info("Applying length filter...")
                df = filter_by_length(df, text_column=self.text_column, **filter_arg)
                logger.info("After length filter: %d rows", len(df))

            elif filter_type == "vocabulary":
                logger.info("Applying vocabulary complexity filter (max_rare_ratio=%s)...", filter_arg["max_rare_ratio"])
                df = filter_by_vocabulary_complexity(df, text_column=self.text_column, **filter_arg)
                logger.info("After vocabulary filter: %d rows", len(df))

        # Step 3: Apply synonym mapping
        if self.synonym_mapper:
            logger.info("Applying synonym mapping...")
            df = df.with_columns([
                pl.col(self.text_column)
                .map_elements(self.synonym_mapper.simplify_line, return_dtype=pl.Utf8)
                .alias(self.text_column)
            ])
        
        # Step 4: Deduplicate
        if self.dedup_mode:
            original_len = len(df)
            if self.dedup_mode in ("exact", "both"):
                logger.info("Running exact dedup...")
                df = exact_dedup(df, self.text_column)
            if self.dedup_mode in ("minhash", "both"):
                logger.info("Running MinHash dedup...")
                df = minhash_dedup(df, self.text_column)
            removed = original_len - len(df)
            logger.info("Dedup (%s): removed %d rows, %d remaining", self.dedup_mode, removed, len(df))
        
        # Step 5: Apply annotators
        for i, annotator in enumerate(self.annotators):
            logger.info("Applying annotator %d/%d...", i + 1, len(self.annotators))
            df = annotator.annotate_dataframe(df, self.text_column)
        
        logger.info("Pipeline complete! Final: %d rows", len(df))
        return df


def process_corpus(
    input_path: str,
    output_path: str,
    *,
    text_column: str = "text",
    normalize: Union[bool, str, None] = True,
    normalize_mode: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    max_grade: Optional[float] = 10.0,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    cefr_index: Optional[CEFRIndex] = None,
    max_rare_ratio: float = 0.3,
    count_unknown_as_rare: bool = False,
    synonyms_map: Optional[Dict[str, str]] = None,
    synonyms_map_path: Optional[str] = None,
    annotators: Optional[List[BaseAnnotator]] = None,
    dedup: Union[bool, str, None] = True,
    dedup_mode: Optional[str] = None,
    generate_stats: bool = True,
) -> Dict[str, Any]:
    """
    Process a parquet corpus through the pipeline.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        text_column: Name of text column (default: "text")
        normalize: Backwards-compat. True → "gentle", False → None.
        normalize_mode: "gentle", "aggressive", None. Overrides normalize.
        keywords: List of keywords for filtering (optional)
        max_grade: Maximum readability grade (optional)
        synonyms_map: Synonym mapping dictionary (optional)
        synonyms_map_path: Path to synonym mapping JSON (optional)
        annotators: List of custom annotators (optional)
        dedup: Backwards-compat. True → "exact", False → None.
        dedup_mode: "exact", "minhash", "both", None. Overrides dedup.
        generate_stats: Whether to generate statistics JSON

    Returns:
        Statistics dictionary if generate_stats=True, else empty dict
    """
    # Read input
    logger.info("Reading input from: %s", input_path)
    df = io_mod.read_parquet(input_path, text_column=text_column)
    logger.info("Loaded %d rows with columns: %s", len(df), df.columns)

    # Build pipeline
    pipeline = DataPipeline(
        text_column=text_column,
        normalize=normalize,
        normalize_mode=normalize_mode,
        dedup=dedup,
        dedup_mode=dedup_mode,
    )
    
    # Add filters
    if keywords:
        pipeline.add_keyword_filter(keywords)
    
    if max_grade is not None:
        pipeline.add_readability_filter(max_grade)

    if any(v is not None for v in (min_chars, max_chars, min_words, max_words)):
        pipeline.add_length_filter(
            min_chars=min_chars, max_chars=max_chars,
            min_words=min_words, max_words=max_words,
        )

    if cefr_index is not None:
        pipeline.add_vocabulary_filter(
            cefr_index,
            max_rare_ratio=max_rare_ratio,
            count_unknown_as_rare=count_unknown_as_rare,
        )

    # Add synonym mapping
    if synonyms_map or synonyms_map_path:
        pipeline.add_synonym_mapper(synonyms_map, synonyms_map_path)
    
    # Add annotators
    if annotators:
        for annotator in annotators:
            pipeline.add_annotator(annotator)
    
    # Process
    df_processed = pipeline.process(df)
    
    # Write output
    logger.info("Writing output to: %s", output_path)
    if generate_stats:
        stats = io_mod.write_parquet_with_stats(df_processed, output_path, text_column)
        logger.info("Statistics written to: %s", Path(output_path).with_suffix('.json'))
        return stats
    else:
        io_mod.write_parquet(df_processed, output_path)
        return {}


def process_corpus_chunked(
    input_path: str,
    output_path: str,
    *,
    chunk_size: int = 50_000,
    text_column: str = "text",
    normalize: Union[bool, str, None] = True,
    normalize_mode: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    max_grade: Optional[float] = 10.0,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    cefr_index: Optional[CEFRIndex] = None,
    max_rare_ratio: float = 0.3,
    count_unknown_as_rare: bool = False,
    synonyms_map: Optional[Dict[str, str]] = None,
    synonyms_map_path: Optional[str] = None,
    dedup: Union[bool, str, None] = True,
    dedup_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a parquet corpus in chunks for memory efficiency.

    Exact dedup uses a global hash set across chunks. MinHash dedup
    (when dedup_mode is "minhash" or "both") runs as a post-pass on the
    output file after chunked processing.

    Returns:
        Statistics dictionary for the final output.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq_mod

    # Build a pipeline with dedup disabled per-chunk (we handle it separately)
    # Resolve effective dedup_mode
    if dedup_mode is not None:
        effective_dedup = dedup_mode
    elif isinstance(dedup, str):
        effective_dedup = dedup
    elif dedup:
        effective_dedup = "exact"
    else:
        effective_dedup = None

    pipeline = DataPipeline(
        text_column=text_column,
        normalize=normalize,
        normalize_mode=normalize_mode,
        dedup=False,      # we handle dedup ourselves
        dedup_mode=None,
    )

    if keywords:
        pipeline.add_keyword_filter(keywords)
    if max_grade is not None:
        pipeline.add_readability_filter(max_grade)
    if any(v is not None for v in (min_chars, max_chars, min_words, max_words)):
        pipeline.add_length_filter(
            min_chars=min_chars, max_chars=max_chars,
            min_words=min_words, max_words=max_words,
        )
    if cefr_index is not None:
        pipeline.add_vocabulary_filter(
            cefr_index,
            max_rare_ratio=max_rare_ratio,
            count_unknown_as_rare=count_unknown_as_rare,
        )
    if synonyms_map or synonyms_map_path:
        pipeline.add_synonym_mapper(synonyms_map, synonyms_map_path)

    # Global hash set for exact dedup across chunks
    seen_hashes: set[int] = set()
    use_exact = effective_dedup in ("exact", "both")
    use_minhash = effective_dedup in ("minhash", "both")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[pq_mod.ParquetWriter] = None
    total_rows = 0
    chunk_num = 0

    try:
        for chunk_df in io_mod.read_parquet_chunked(input_path, text_column, chunk_size):
            chunk_num += 1
            logger.info("Processing chunk %d (%d rows)...", chunk_num, len(chunk_df))

            chunk_df = pipeline.process(chunk_df)

            if use_exact and len(chunk_df) > 0:
                chunk_df = exact_dedup(chunk_df, text_column, seen_hashes=seen_hashes)

            if len(chunk_df) == 0:
                continue

            arrow_table = chunk_df.to_arrow()
            if writer is None:
                writer = pq_mod.ParquetWriter(str(out_path), arrow_table.schema)
            writer.write_table(arrow_table)
            total_rows += len(chunk_df)
    finally:
        if writer is not None:
            writer.close()

    logger.info("Chunked processing complete: %d chunks, %d rows written", chunk_num, total_rows)

    # MinHash post-pass if requested
    if use_minhash and total_rows > 0:
        logger.info("Running MinHash post-pass on output...")
        df_out = pl.read_parquet(str(out_path))
        df_out = minhash_dedup(df_out, text_column)
        df_out.write_parquet(str(out_path))
        total_rows = len(df_out)

    # Generate stats
    if total_rows > 0:
        df_final = pl.read_parquet(str(out_path))
        stats = io_mod.write_parquet_with_stats(df_final, output_path, text_column)
    else:
        stats = {"total_rows": 0}

    return stats
