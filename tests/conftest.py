"""Shared fixtures for tiny_corpus_prep tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def sample_df() -> pl.DataFrame:
    """A small DataFrame with a 'text' column for testing filters."""
    return pl.DataFrame({
        "text": [
            "The cat sat on the mat.",
            "Quantum entanglement describes correlations between particles.",
            "Hello world!",
            "This is a simple sentence for testing purposes.",
            "",
            "A very short text.",
        ]
    })


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_parquet(sample_df: pl.DataFrame, tmp_dir: Path) -> Path:
    """Write sample_df to a parquet file and return its path."""
    path = tmp_dir / "sample.parquet"
    sample_df.write_parquet(path)
    return path
