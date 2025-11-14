\
"""
End-to-end pipeline using Polars DataFrames.
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any
from pathlib import Path
import hashlib
import polars as pl

from .normalize import normalize_text
from .synonyms import SynonymMapper
from .filters import filter_by_keywords, filter_by_readability
from .annotators import BaseAnnotator
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
        normalize: bool = True,
        dedup: bool = True,
    ):
        """
        Initialize the pipeline.
        
        Args:
            text_column: Name of the text column
            normalize: Whether to normalize text (lowercase, punctuation)
            dedup: Whether to deduplicate rows by text content
        """
        self.text_column = text_column
        self.normalize = normalize
        self.dedup = dedup
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
    
    def add_readability_filter(self, max_grade: float = 8.0) -> "DataPipeline":
        """
        Add readability filter to pipeline.
        
        Args:
            max_grade: Maximum Flesch-Kincaid grade level
            
        Returns:
            Self for chaining
        """
        self.filters.append(("readability", max_grade))
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
        
        print(f"Starting pipeline with {len(df)} rows...")
        
        # Step 1: Normalize text if enabled
        if self.normalize:
            print("Normalizing text...")
            df = df.with_columns([
                pl.col(self.text_column)
                .map_elements(normalize_text, return_dtype=pl.Utf8)
                .alias(self.text_column)
            ])
            # Remove empty rows after normalization
            df = df.filter(
                pl.col(self.text_column).is_not_null() & 
                (pl.col(self.text_column).str.strip_chars() != "")
            )
            print(f"After normalization: {len(df)} rows")
        
        # Step 2: Apply filters
        for filter_type, filter_arg in self.filters:
            if filter_type == "keywords":
                print(f"Applying keyword filter ({len(filter_arg)} keywords)...")
                df = filter_by_keywords(df, filter_arg, self.text_column)
                print(f"After keyword filter: {len(df)} rows")
            
            elif filter_type == "readability":
                print(f"Applying readability filter (max grade: {filter_arg})...")
                df = filter_by_readability(df, filter_arg, self.text_column)
                print(f"After readability filter: {len(df)} rows")
        
        # Step 3: Apply synonym mapping
        if self.synonym_mapper:
            print("Applying synonym mapping...")
            df = df.with_columns([
                pl.col(self.text_column)
                .map_elements(self.synonym_mapper.simplify_line, return_dtype=pl.Utf8)
                .alias(self.text_column)
            ])
        
        # Step 4: Deduplicate
        if self.dedup:
            print("Deduplicating...")
            original_len = len(df)
            df = df.unique(subset=[self.text_column], keep="first")
            removed = original_len - len(df)
            print(f"Removed {removed} duplicate rows, {len(df)} remaining")
        
        # Step 5: Apply annotators
        for i, annotator in enumerate(self.annotators):
            print(f"Applying annotator {i+1}/{len(self.annotators)}...")
            df = annotator.annotate_dataframe(df, self.text_column)
        
        print(f"Pipeline complete! Final: {len(df)} rows")
        return df


def process_corpus(
    input_path: str,
    output_path: str,
    *,
    text_column: str = "text",
    normalize: bool = True,
    keywords: Optional[List[str]] = None,
    max_grade: Optional[float] = 8.0,
    synonyms_map: Optional[Dict[str, str]] = None,
    synonyms_map_path: Optional[str] = None,
    annotators: Optional[List[BaseAnnotator]] = None,
    dedup: bool = True,
    generate_stats: bool = True,
) -> Dict[str, Any]:
    """
    Process a parquet corpus through the pipeline.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        text_column: Name of text column (default: "text")
        normalize: Whether to normalize text
        keywords: List of keywords for filtering (optional)
        max_grade: Maximum readability grade (optional)
        synonyms_map: Synonym mapping dictionary (optional)
        synonyms_map_path: Path to synonym mapping JSON (optional)
        annotators: List of custom annotators (optional)
        dedup: Whether to deduplicate
        generate_stats: Whether to generate statistics JSON
        
    Returns:
        Statistics dictionary if generate_stats=True, else empty dict
    """
    # Read input
    print(f"Reading input from: {input_path}")
    df = io_mod.read_parquet(input_path, text_column=text_column)
    print(f"Loaded {len(df)} rows with columns: {df.columns}")
    
    # Build pipeline
    pipeline = DataPipeline(
        text_column=text_column,
        normalize=normalize,
        dedup=dedup
    )
    
    # Add filters
    if keywords:
        pipeline.add_keyword_filter(keywords)
    
    if max_grade is not None:
        pipeline.add_readability_filter(max_grade)
    
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
    print(f"\nWriting output to: {output_path}")
    if generate_stats:
        stats = io_mod.write_parquet_with_stats(df_processed, output_path, text_column)
        print(f"Statistics written to: {Path(output_path).with_suffix('.json')}")
        return stats
    else:
        io_mod.write_parquet(df_processed, output_path)
        return {}
