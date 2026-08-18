"""Tests for tiny_corpus_prep.mixing."""
from __future__ import annotations

import polars as pl
import pytest

from tiny_corpus_prep.mixing import MixSource, mix_sources


def _df(prefix: str, n: int, words_per_doc: int = 10) -> pl.DataFrame:
    body = " ".join(["word"] * (words_per_doc - 1))
    return pl.DataFrame({"text": [f"{prefix}{i} {body}" for i in range(n)]})


class TestMixSources:
    def test_weights_respected_by_words(self):
        sources = [
            MixSource("a", _df("a", 100), weight=0.7),
            MixSource("b", _df("b", 100), weight=0.3),
        ]
        mixed, stats = mix_sources(sources, seed=1, target_total_words=500)
        a_words = stats["per_source"]["a"]["selected_words"]
        b_words = stats["per_source"]["b"]["selected_words"]
        # Greedy selection overshoots by at most one doc (10 words) per source
        assert abs(a_words - 350) <= 10
        assert abs(b_words - 150) <= 10
        assert set(mixed["source"].unique()) == {"a", "b"}

    def test_same_seed_reproduces_exactly(self):
        def run():
            sources = [
                MixSource("a", _df("a", 50), weight=0.5),
                MixSource("b", _df("b", 50), weight=0.5),
            ]
            mixed, _ = mix_sources(sources, seed=123, target_total_words=300)
            return mixed["text"].to_list()

        assert run() == run()

    def test_different_seed_differs(self):
        sources1 = [MixSource("a", _df("a", 100), weight=1.0)]
        sources2 = [MixSource("a", _df("a", 100), weight=1.0)]
        m1, _ = mix_sources(sources1, seed=1, target_total_words=500)
        m2, _ = mix_sources(sources2, seed=2, target_total_words=500)
        assert m1["text"].to_list() != m2["text"].to_list()

    def test_auto_target_uses_binding_source(self):
        # a has 1000 words at weight 0.5 → caps total at 2000;
        # b has 5000 words so a is binding.
        sources = [
            MixSource("a", _df("a", 100), weight=0.5),
            MixSource("b", _df("b", 500), weight=0.5),
        ]
        _, stats = mix_sources(sources, seed=1)
        assert stats["target_total_words"] == 2000

    def test_upsampling_repeats_small_source(self):
        sources = [
            MixSource("big", _df("big", 200), weight=0.5),
            MixSource("small", _df("small", 10), weight=0.5, allow_upsample=True),
        ]
        _, stats = mix_sources(sources, seed=1, target_total_words=1000)
        small = stats["per_source"]["small"]
        assert small["epochs"] > 1
        assert small["selected_docs"] > small["unique_docs"]
        assert abs(small["selected_words"] - 500) <= 10

    def test_infeasible_target_raises(self):
        sources = [MixSource("a", _df("a", 10), weight=1.0)]  # only 100 words
        with pytest.raises(ValueError, match="allow_upsample"):
            mix_sources(sources, seed=1, target_total_words=1000)

    def test_all_upsample_requires_explicit_target(self):
        sources = [MixSource("a", _df("a", 10), weight=1.0, allow_upsample=True)]
        with pytest.raises(ValueError, match="target_total_words"):
            mix_sources(sources, seed=1)

    def test_cross_source_dedup_earlier_source_wins(self):
        shared = _df("shared", 20)
        sources = [
            MixSource("first", shared, weight=0.5),
            MixSource("second", shared.clone(), weight=0.5),
        ]
        with pytest.raises(ValueError, match="no documents selected|second"):
            # second source becomes empty after dedup → its target is
            # unreachable (0 available words at weight 0.5)
            mix_sources(sources, seed=1, target_total_words=100)

    def test_cross_source_dedup_disabled(self):
        shared = _df("shared", 20)
        sources = [
            MixSource("first", shared, weight=0.5),
            MixSource("second", shared.clone(), weight=0.5),
        ]
        mixed, _ = mix_sources(
            sources, seed=1, target_total_words=100, cross_source_dedup=False
        )
        assert set(mixed["source"].unique()) == {"first", "second"}

    def test_metadata_columns_dropped_source_added(self):
        df = _df("a", 20).with_columns(pl.lit("t").alias("title"))
        mixed, _ = mix_sources(
            [MixSource("a", df, weight=1.0)], seed=1, target_total_words=100
        )
        assert mixed.columns == ["text", "source"]

    def test_empty_sources_raise(self):
        with pytest.raises(ValueError):
            mix_sources([], seed=1)


class TestMixSourcesStreaming:
    @pytest.fixture
    def source_files(self, tmp_path):
        paths = {}
        for name, n in (("a", 100), ("b", 100)):
            path = tmp_path / f"{name}.parquet"
            _df(name, n).write_parquet(path)
            paths[name] = path
        return paths

    def _sources(self, paths, wa=0.7, wb=0.3, upsample_b=False):
        from tiny_corpus_prep.mixing import MixSourceFile
        return [
            MixSourceFile("a", paths["a"], weight=wa),
            MixSourceFile("b", paths["b"], weight=wb, allow_upsample=upsample_b),
        ]

    def test_matches_in_memory_selection(self, source_files, tmp_path):
        from tiny_corpus_prep.mixing import mix_sources_streaming

        sources_mem = [
            MixSource("a", _df("a", 100), weight=0.7),
            MixSource("b", _df("b", 100), weight=0.3),
        ]
        _, stats_mem = mix_sources(sources_mem, seed=5, target_total_words=500)
        stats_stream, _ = mix_sources_streaming(
            self._sources(source_files), seed=5,
            output_dir=tmp_path / "shards", shard_max_docs=20,
            target_total_words=500,
        )
        # Same seed + same selection semantics ⇒ identical per-source stats
        assert stats_stream["per_source"] == stats_mem["per_source"]
        assert stats_stream["total_docs"] == stats_mem["total_docs"]

    def test_shards_written_and_deterministic(self, source_files, tmp_path):
        from tiny_corpus_prep.mixing import mix_sources_streaming

        def run(out):
            stats, shard_files = mix_sources_streaming(
                self._sources(source_files), seed=9,
                output_dir=out, shard_max_docs=20, target_total_words=600,
            )
            dfs = [pl.read_parquet(out / s["file"]) for s in shard_files]
            return stats, shard_files, pl.concat(dfs)

        stats1, files1, all1 = run(tmp_path / "s1")
        stats2, files2, all2 = run(tmp_path / "s2")
        assert stats1 == stats2
        assert all1["text"].to_list() == all2["text"].to_list()
        assert all1.columns == ["text", "source"]
        assert all(f["docs"] <= 20 for f in files1)
        assert sum(f["docs"] for f in files1) == stats1["total_docs"]
        assert not (tmp_path / "s1" / "_tmp_mix").exists()  # tmp cleaned up

    def test_upsampling_and_weights(self, source_files, tmp_path):
        from tiny_corpus_prep.mixing import MixSourceFile, mix_sources_streaming

        small = tmp_path / "small.parquet"
        _df("small", 5).write_parquet(small)
        stats, _ = mix_sources_streaming(
            [
                MixSourceFile("a", source_files["a"], weight=0.5),
                MixSourceFile("small", small, weight=0.5, allow_upsample=True),
            ],
            seed=3, output_dir=tmp_path / "shards",
            shard_max_docs=50, target_total_words=400,
        )
        assert stats["per_source"]["small"]["epochs"] > 1
        assert abs(stats["per_source"]["a"]["selected_words"] - 200) <= 10
        assert abs(stats["per_source"]["small"]["selected_words"] - 200) <= 10

    def test_cross_source_dedup_streaming(self, tmp_path):
        from tiny_corpus_prep.mixing import MixSourceFile, mix_sources_streaming

        shared = _df("shared", 20)
        p1, p2 = tmp_path / "s1.parquet", tmp_path / "s2.parquet"
        shared.write_parquet(p1)
        shared.write_parquet(p2)
        with pytest.raises(ValueError, match="second"):
            mix_sources_streaming(
                [
                    MixSourceFile("first", p1, weight=0.5),
                    MixSourceFile("second", p2, weight=0.5),
                ],
                seed=1, output_dir=tmp_path / "shards",
                shard_max_docs=50, target_total_words=100,
            )
