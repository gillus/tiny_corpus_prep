"""
Filtering: readability, length, vocabulary complexity.
- Readability filter: uses textstat Flesch-Kincaid grade.
- Length filter: min/max character and word counts.
- Vocabulary complexity filter: CEFR-based rare-word ratio.
- Polars-compatible filtering functions.
"""
from __future__ import annotations
from typing import Optional, Sequence
import polars as pl
import textstat

from .common import CEFRIndex, tokenize_basic, simple_lemma_like, CEFR_ORDER


def calculate_readability_grade(text: str) -> Optional[float]:
    """
    Calculate Flesch-Kincaid grade level using textstat.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Grade level as float, or None if calculation fails
    """
    if not text or not text.strip():
        return None
    try:
        grade = textstat.flesch_kincaid_grade(text)
        return float(grade)
    except Exception:
        return None


def is_middle_school_level(text: str) -> bool:
    """
    Check if text is at middle school reading level (grade <= 8).
    
    Args:
        text: Input text to analyze
        
    Returns:
        True if text is at or below 8th grade level
    """
    if not text:
        return False
    try:
        grade = textstat.flesch_kincaid_grade(text)
        return grade <= 8
    except Exception:
        return False


def filter_by_readability(
    df: pl.DataFrame, 
    max_grade: float = 10.0,
    text_column: str = "text"
) -> pl.DataFrame:
    """
    Filter DataFrame by readability level.
    
    Args:
        df: Input Polars DataFrame
        max_grade: Maximum Flesch-Kincaid grade level to keep
        text_column: Name of the text column
        
    Returns:
        Filtered DataFrame
    """
    # Add readability grade column
    df = df.with_columns([
        pl.col(text_column)
        .map_elements(calculate_readability_grade, return_dtype=pl.Float64)
        .alias("_readability_grade")
    ])
    
    # Filter and remove temporary column
    return df.filter(
        (pl.col("_readability_grade").is_not_null()) & 
        (pl.col("_readability_grade") <= max_grade)
    ).drop("_readability_grade")


def filter_by_keywords(
    df: pl.DataFrame, 
    keywords: list[str], 
    text_column: str = "text"
) -> pl.DataFrame:
    """
    Filter DataFrame by keywords presence.
    
    Args:
        df: Input Polars DataFrame
        keywords: List of keywords (case-insensitive)
        text_column: Name of the text column
        
    Returns:
        Filtered DataFrame
    """
    if not keywords:
        return df
    
    # Create filter expression: check if any keyword is in text (case-insensitive)
    filter_expr = None
    for keyword in keywords:
        keyword_lower = keyword.lower()
        expr = pl.col(text_column).str.to_lowercase().str.contains(keyword_lower)
        filter_expr = expr if filter_expr is None else (filter_expr | expr)
    
    return df.filter(filter_expr) if filter_expr is not None else df


def filter_by_length(
    df: pl.DataFrame,
    *,
    min_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    min_words: Optional[int] = None,
    max_words: Optional[int] = None,
    text_column: str = "text",
) -> pl.DataFrame:
    """
    Filter DataFrame by document length (characters and/or words).

    Uses Polars native expressions — no row-level Python calls.

    Args:
        df: Input Polars DataFrame
        min_chars: Minimum character count (inclusive)
        max_chars: Maximum character count (inclusive)
        min_words: Minimum word count (inclusive)
        max_words: Maximum word count (inclusive)
        text_column: Name of the text column

    Returns:
        Filtered DataFrame
    """
    conditions = []
    if min_chars is not None:
        conditions.append(pl.col(text_column).str.len_chars() >= min_chars)
    if max_chars is not None:
        conditions.append(pl.col(text_column).str.len_chars() <= max_chars)
    if min_words is not None or max_words is not None:
        # Split on whitespace for word count
        word_count = pl.col(text_column).str.split(" ").list.len()
        if min_words is not None:
            conditions.append(word_count >= min_words)
        if max_words is not None:
            conditions.append(word_count <= max_words)

    if not conditions:
        return df

    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined & cond
    return df.filter(combined)


def _compute_rare_ratio(
    text: str,
    cefr_index: CEFRIndex,
    rare_min_rank: int,
    count_unknown_as_rare: bool,
) -> float:
    """Compute the fraction of word tokens that are 'rare' per CEFR levels."""
    tokens = tokenize_basic(text)
    alpha_tokens = [t for t in tokens if t.isalpha()]
    if not alpha_tokens:
        return 0.0
    rare_count = 0
    for tok in alpha_tokens:
        word = tok.lower()
        rank = cefr_index.rank(word)
        if rank is None:
            # Try lemmatised form
            rank = cefr_index.rank(simple_lemma_like(word))
        if rank is None:
            if count_unknown_as_rare:
                rare_count += 1
        elif rank >= rare_min_rank:
            rare_count += 1
    return rare_count / len(alpha_tokens)


def filter_by_vocabulary_complexity(
    df: pl.DataFrame,
    cefr_index: CEFRIndex,
    *,
    max_rare_ratio: float = 0.3,
    rare_levels: Sequence[str] = ("B2", "C1", "C2"),
    count_unknown_as_rare: bool = False,
    text_column: str = "text",
) -> pl.DataFrame:
    """
    Filter DataFrame by vocabulary complexity using CEFR word levels.

    Drops documents where the ratio of rare words exceeds *max_rare_ratio*.

    Args:
        df: Input Polars DataFrame
        cefr_index: CEFRIndex loaded from CSV
        max_rare_ratio: Maximum allowed fraction of rare words (0.0–1.0)
        rare_levels: CEFR levels considered "rare" (default B2, C1, C2)
        count_unknown_as_rare: Whether words not in CEFR index count as rare
        text_column: Name of the text column

    Returns:
        Filtered DataFrame
    """
    rare_min_rank = min(CEFR_ORDER[lv] for lv in rare_levels)

    df = df.with_columns(
        pl.col(text_column)
        .map_elements(
            lambda s: _compute_rare_ratio(s, cefr_index, rare_min_rank, count_unknown_as_rare),
            return_dtype=pl.Float64,
        )
        .alias("_rare_ratio")
    )
    return df.filter(pl.col("_rare_ratio") <= max_rare_ratio).drop("_rare_ratio")
