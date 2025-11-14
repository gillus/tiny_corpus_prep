"""
IO helpers for Polars DataFrame operations with parquet.
Includes reading, writing, and statistics generation.
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from pathlib import Path
import json
import polars as pl


def read_parquet(path: str, text_column: str = "text") -> pl.DataFrame:
    """
    Read a parquet file into a Polars DataFrame.
    
    Args:
        path: Path to parquet file
        text_column: Name of required text column
        
    Returns:
        Polars DataFrame
        
    Raises:
        ValueError: If text column is missing
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    
    df = pl.read_parquet(path)
    
    if text_column not in df.columns:
        raise ValueError(
            f"Required column '{text_column}' not found in parquet file. "
            f"Available columns: {df.columns}"
        )
    
    return df


def write_parquet(df: pl.DataFrame, path: str) -> None:
    """
    Write a Polars DataFrame to parquet file.
    
    Args:
        df: Polars DataFrame to write
        path: Output path for parquet file
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def generate_stats(df: pl.DataFrame, text_column: str = "text") -> Dict[str, Any]:
    """
    Generate statistics about a DataFrame.
    
    Args:
        df: Polars DataFrame
        text_column: Name of text column
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": df.columns,
    }
    
    # Text column statistics
    if text_column in df.columns:
        text_lengths = df[text_column].str.len_chars()
        stats["text_stats"] = {
            "min_length": text_lengths.min(),
            "max_length": text_lengths.max(),
            "mean_length": text_lengths.mean(),
            "median_length": text_lengths.median(),
            "total_characters": text_lengths.sum(),
        }
        
        # Count empty/null texts
        empty_count = df.filter(
            pl.col(text_column).is_null() | 
            (pl.col(text_column).str.strip_chars() == "")
        ).height
        stats["text_stats"]["empty_or_null_count"] = empty_count
    
    # Column-specific statistics for other columns
    column_stats = {}
    for col in df.columns:
        if col == text_column:
            continue
        
        col_type = str(df[col].dtype)
        col_info = {
            "dtype": col_type,
            "null_count": df[col].null_count(),
        }
        
        # Add value counts for categorical/string columns
        if df[col].dtype in [pl.Utf8, pl.Categorical]:
            value_counts = df[col].value_counts().sort("counts", descending=True)
            col_info["unique_values"] = len(value_counts)
            col_info["top_values"] = value_counts.head(10).to_dicts()
        
        # Add numeric statistics for numeric columns
        elif df[col].dtype in [pl.Int64, pl.Int32, pl.Float64, pl.Float32]:
            col_info["min"] = df[col].min()
            col_info["max"] = df[col].max()
            col_info["mean"] = df[col].mean()
            col_info["median"] = df[col].median()
        
        column_stats[col] = col_info
    
    stats["column_stats"] = column_stats
    
    return stats


def write_stats(stats: Dict[str, Any], output_path: str) -> None:
    """
    Write statistics to a JSON file.
    
    Args:
        stats: Statistics dictionary
        output_path: Path for output JSON file
    """
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)


def write_parquet_with_stats(
    df: pl.DataFrame, 
    output_path: str, 
    text_column: str = "text"
) -> Dict[str, Any]:
    """
    Write parquet file and generate statistics JSON.
    
    Args:
        df: Polars DataFrame to write
        output_path: Output path for parquet file
        text_column: Name of text column for statistics
        
    Returns:
        Statistics dictionary
    """
    # Write parquet
    write_parquet(df, output_path)
    
    # Generate stats
    stats = generate_stats(df, text_column)
    
    # Write stats to JSON with same name as parquet
    stats_path = Path(output_path).with_suffix(".json")
    write_stats(stats, str(stats_path))
    
    return stats
