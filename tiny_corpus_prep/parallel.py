"""
Process-pool helper for per-document text functions.

The per-doc operations (readability scoring, rare-word ratios, normalization,
synonym rewriting) are pure-Python and GIL-bound; at corpus scale they dominate
runtime. TextMapper applies a picklable function over a list of texts either
serially (n_workers=1, the default) or via a multiprocessing pool, preserving
input order in both cases so results are deterministic.

Functions passed to `map` must be picklable: module-level functions,
functools.partial of them, or bound methods of picklable objects.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Callable, List, Optional, Sequence, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class TextMapper:
    """
    Ordered map over texts, serial or process-parallel.

    The pool is created lazily on first parallel use and reused across calls;
    call close() (or use as a context manager) when done.
    """

    def __init__(self, n_workers: int = 1, chunksize: int = 256):
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.n_workers = n_workers
        self.chunksize = chunksize
        self._pool: Optional[mp.pool.Pool] = None

    def _ensure_pool(self) -> mp.pool.Pool:
        if self._pool is None:
            # spawn, not fork: polars and tokenizers start native thread pools
            # in the parent, and fork()ing a multithreaded process can deadlock
            # the children. Spawn startup cost is paid once per pipeline run.
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(self.n_workers)
            logger.info("Started process pool with %d workers", self.n_workers)
        return self._pool

    def map(self, fn: Callable[[T], R], items: Sequence[T]) -> List[R]:
        """Apply fn to every item, preserving order."""
        if self.n_workers == 1 or len(items) <= self.chunksize:
            return [fn(item) for item in items]
        return self._ensure_pool().map(fn, items, chunksize=self.chunksize)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> "TextMapper":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
