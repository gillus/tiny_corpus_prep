"""
Filtering: readability using textstat.
- Readability filter: uses textstat Flesch-Kincaid grade.
- Polars-compatible filtering functions.
"""
from __future__ import annotations
from typing import Optional
import polars as pl
import textstat


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
    max_grade: float = 8.0, 
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
