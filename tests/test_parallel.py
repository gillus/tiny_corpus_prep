"""Tests for tiny_corpus_prep.parallel."""
from __future__ import annotations

import functools

import pytest

from tiny_corpus_prep.parallel import TextMapper
from tiny_corpus_prep.filters import calculate_readability_grade
from tiny_corpus_prep.normalize import normalize_text


def _upper(s: str) -> str:
    return s.upper()


class TestTextMapper:
    def test_serial_map(self):
        with TextMapper(n_workers=1) as mapper:
            assert mapper.map(_upper, ["a", "b"]) == ["A", "B"]

    def test_invalid_workers(self):
        with pytest.raises(ValueError):
            TextMapper(n_workers=0)

    def test_parallel_matches_serial_and_preserves_order(self):
        texts = [f"The cat number {i} sat on the mat. It was happy." for i in range(600)]
        with TextMapper(n_workers=1) as serial, TextMapper(n_workers=2, chunksize=50) as par:
            fns = [
                _upper,
                calculate_readability_grade,
                functools.partial(normalize_text, mode="aggressive"),
            ]
            for fn in fns:
                assert par.map(fn, texts) == serial.map(fn, texts)

    def test_small_batches_stay_serial(self):
        # Below chunksize the pool is never started (no fork overhead)
        mapper = TextMapper(n_workers=4, chunksize=256)
        assert mapper.map(_upper, ["x"] * 10) == ["X"] * 10
        assert mapper._pool is None
        mapper.close()

    def test_pool_reused_across_calls(self):
        with TextMapper(n_workers=2, chunksize=10) as mapper:
            mapper.map(_upper, ["a"] * 50)
            pool = mapper._pool
            mapper.map(_upper, ["b"] * 50)
            assert mapper._pool is pool
