"""Minimal, fast data prep for small GPT pretraining using Polars.

Modules:
- normalize: lowercase + punctuation filter.
- synonyms: vocabulary simplification by mapping synonyms -> canonical form.
- filters: readability and keyword filters using textstat.
- annotators: custom annotation framework with Gemini example.
- io: Polars parquet I/O and statistics generation.
- pipeline: end-to-end processing with Polars DataFrames.
- common: CEFR utilities for text simplification and tokenization.
"""

from .pipeline import DataPipeline, process_corpus
from .annotators import BaseAnnotator, GeminiAnnotator, CustomFunctionAnnotator
from .filters import filter_by_readability, filter_by_keywords, is_middle_school_level
from .io import read_parquet, write_parquet, write_parquet_with_stats, generate_stats
from .common import (
    CEFRIndex, 
    load_mapping, 
    preserve_case_like, 
    tokenize_basic, 
    detokenize_basic,
    simple_lemma_like,
)

__version__ = "0.2.0"

__all__ = [
    # Pipeline
    "DataPipeline",
    "process_corpus",
    # Annotators
    "BaseAnnotator",
    "GeminiAnnotator",
    "CustomFunctionAnnotator",
    # Filters
    "filter_by_readability",
    "filter_by_keywords",
    "is_middle_school_level",
    # IO
    "read_parquet",
    "write_parquet",
    "write_parquet_with_stats",
    "generate_stats",
    # Common / CEFR utilities
    "CEFRIndex",
    "load_mapping",
    "preserve_case_like",
    "tokenize_basic",
    "detokenize_basic",
    "simple_lemma_like",
]
