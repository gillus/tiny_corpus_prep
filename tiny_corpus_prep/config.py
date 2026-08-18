"""
YAML-driven configuration for corpus building.

A corpus config describes:
- profiles:  named per-source processing settings (normalize/filter/dedup),
             so prose and structured sources can be treated differently;
- sources:   input files, each pointing at a profile, with a mix weight;
- mix:       how sources are combined (seeded, weight-proportional by words);
- output:    where shards, stats, and the run manifest are written.

Unknown keys anywhere raise ValueError — a typo in a config must fail loudly,
not silently change the corpus.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from .common import CEFR_ORDER

VALID_NORMALIZE_MODES = {None, "gentle", "aggressive"}
VALID_DEDUP_MODES = {None, "exact", "minhash", "both"}


def _coerce_none(value: Any) -> Any:
    """Treat YAML strings 'none'/'None' the same as null."""
    if value in ("none", "None"):
        return None
    return value


def _check_keys(section: str, data: Dict[str, Any], allowed: set) -> None:
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(
            f"{section}: unknown keys {sorted(unknown)}. Allowed: {sorted(allowed)}"
        )


@dataclass
class ProfileConfig:
    """Processing settings applied to every source that references this profile."""
    name: str
    normalize_mode: Optional[str] = "gentle"
    keywords: Optional[List[str]] = None
    max_grade: Optional[float] = None
    min_chars: Optional[int] = None
    max_chars: Optional[int] = None
    min_words: Optional[int] = None
    max_words: Optional[int] = None
    cefr_csv: Optional[str] = None
    max_rare_ratio: float = 0.3
    rare_levels: List[str] = field(default_factory=lambda: ["B2", "C1", "C2"])
    count_unknown_as_rare: bool = False
    synonyms_map_path: Optional[str] = None
    dedup_mode: Optional[str] = "exact"
    paragraph_dedup: bool = False
    paragraph_min_chars: int = 50
    minhash_threshold: float = 0.8
    minhash_num_perm: int = 128
    strip_boilerplate: bool = False
    boilerplate_patterns: Optional[List[str]] = None  # extras added to defaults
    repetition_filter: bool = False
    repetition_overrides: Optional[Dict[str, float]] = None
    language: Optional[str] = None      # e.g. "en"; None = no language gate
    language_min_prob: float = 0.8

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ProfileConfig":
        allowed = {f.name for f in fields(cls)} - {"name"}
        _check_keys(f"profiles.{name}", data, allowed)
        d = dict(data)
        if "normalize_mode" in d:
            d["normalize_mode"] = _coerce_none(d["normalize_mode"])
        if "dedup_mode" in d:
            d["dedup_mode"] = _coerce_none(d["dedup_mode"])
        profile = cls(name=name, **d)
        if profile.normalize_mode not in VALID_NORMALIZE_MODES:
            raise ValueError(
                f"profiles.{name}: invalid normalize_mode {profile.normalize_mode!r}"
            )
        if profile.dedup_mode not in VALID_DEDUP_MODES:
            raise ValueError(
                f"profiles.{name}: invalid dedup_mode {profile.dedup_mode!r}"
            )
        bad_levels = [lv for lv in profile.rare_levels if lv not in CEFR_ORDER]
        if bad_levels:
            raise ValueError(
                f"profiles.{name}: invalid CEFR levels {bad_levels} in rare_levels"
            )
        if not (0.0 < profile.minhash_threshold <= 1.0):
            raise ValueError(
                f"profiles.{name}: minhash_threshold must be in (0, 1], "
                f"got {profile.minhash_threshold}"
            )
        if profile.repetition_overrides:
            from .quality import DEFAULT_REPETITION_THRESHOLDS

            unknown = set(profile.repetition_overrides) - set(
                DEFAULT_REPETITION_THRESHOLDS
            )
            if unknown:
                raise ValueError(
                    f"profiles.{name}: unknown repetition_overrides keys "
                    f"{sorted(unknown)}. Allowed: "
                    f"{sorted(DEFAULT_REPETITION_THRESHOLDS)}"
                )
        if not (0.0 < profile.language_min_prob <= 1.0):
            raise ValueError(
                f"profiles.{name}: language_min_prob must be in (0, 1], "
                f"got {profile.language_min_prob}"
            )
        return profile


@dataclass
class SourceConfig:
    """One input corpus and how it participates in the final mix."""
    name: str
    input: str
    profile: str
    weight: float = 1.0
    allow_upsample: bool = False

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "SourceConfig":
        allowed = {f.name for f in fields(cls)} - {"name"}
        _check_keys(f"sources.{name}", data, allowed)
        if "input" not in data:
            raise ValueError(f"sources.{name}: 'input' is required")
        if "profile" not in data:
            raise ValueError(f"sources.{name}: 'profile' is required")
        source = cls(name=name, **data)
        if source.weight <= 0:
            raise ValueError(f"sources.{name}: weight must be > 0, got {source.weight}")
        return source


@dataclass
class MixConfig:
    # None = the largest total achievable without upsampling any
    # non-upsample source at the given weights.
    target_total_words: Optional[int] = None
    # Drop docs that already appeared in an earlier-listed source
    # (intentional upsample repeats are exempt).
    cross_source_dedup: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MixConfig":
        _check_keys("mix", data, {f.name for f in fields(cls)})
        return cls(**data)


@dataclass
class OutputConfig:
    dir: str
    shard_max_docs: int = 100_000

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        _check_keys("output", data, {f.name for f in fields(cls)})
        if "dir" not in data:
            raise ValueError("output: 'dir' is required")
        out = cls(**data)
        if out.shard_max_docs <= 0:
            raise ValueError(f"output.shard_max_docs must be > 0, got {out.shard_max_docs}")
        return out


@dataclass
class ProcessingConfig:
    text_column: str = "text"
    chunk_size: Optional[int] = None  # rows per chunk; None = whole file at once
    n_workers: int = 1                # worker processes for per-doc operations

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        _check_keys("processing", data, {f.name for f in fields(cls)})
        cfg = cls(**data)
        if cfg.n_workers < 1:
            raise ValueError(f"processing.n_workers must be >= 1, got {cfg.n_workers}")
        return cfg


@dataclass
class TokenizerConfig:
    """Byte-level BPE tokenizer training settings (step 3)."""
    vocab_size: int = 32000
    min_frequency: int = 2
    eos_token: str = "<|endoftext|>"
    pad_token: str = "<|pad|>"
    extra_special_tokens: List[str] = field(default_factory=list)
    # Characters that must encode as exactly one token (guaranteed by the
    # byte-level alphabet; verified after training as a safety net).
    require_single_tokens: List[str] = field(
        default_factory=lambda: ["{", "}", "[", "]", ":", '"', ",", "-", "(", ")"]
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenizerConfig":
        _check_keys("tokenizer", data, {f.name for f in fields(cls)})
        cfg = cls(**data)
        n_special = 2 + len(cfg.extra_special_tokens)
        if cfg.vocab_size < 256 + n_special + 32:
            raise ValueError(
                f"tokenizer.vocab_size={cfg.vocab_size} is too small for a "
                f"byte-level BPE (256 bytes + {n_special} special tokens + merges)"
            )
        return cfg


@dataclass
class DatasetConfig:
    """Packing settings for the HF Arrow training dataset (step 3)."""
    seq_length: int = 1024
    val_fraction: float = 0.005

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetConfig":
        _check_keys("dataset", data, {f.name for f in fields(cls)})
        cfg = cls(**data)
        if cfg.seq_length <= 0:
            raise ValueError(f"dataset.seq_length must be > 0, got {cfg.seq_length}")
        if not (0.0 <= cfg.val_fraction < 1.0):
            raise ValueError(
                f"dataset.val_fraction must be in [0, 1), got {cfg.val_fraction}"
            )
        return cfg


@dataclass
class CorpusConfig:
    output: OutputConfig
    profiles: Dict[str, ProfileConfig]
    sources: Dict[str, SourceConfig]  # insertion order matters for dedup priority
    seed: int = 42
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    mix: MixConfig = field(default_factory=MixConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CorpusConfig":
        _check_keys(
            "config", data,
            {"seed", "output", "processing", "profiles", "sources", "mix",
             "tokenizer", "dataset"},
        )
        for required in ("output", "profiles", "sources"):
            if required not in data:
                raise ValueError(f"config: '{required}' section is required")

        profiles = {
            name: ProfileConfig.from_dict(name, d or {})
            for name, d in data["profiles"].items()
        }
        sources = {
            name: SourceConfig.from_dict(name, d or {})
            for name, d in data["sources"].items()
        }
        if not sources:
            raise ValueError("config: at least one source is required")
        for source in sources.values():
            if source.profile not in profiles:
                raise ValueError(
                    f"sources.{source.name}: profile {source.profile!r} not defined. "
                    f"Available: {sorted(profiles)}"
                )

        return cls(
            seed=data.get("seed", 42),
            output=OutputConfig.from_dict(data["output"]),
            processing=ProcessingConfig.from_dict(data.get("processing") or {}),
            profiles=profiles,
            sources=sources,
            mix=MixConfig.from_dict(data.get("mix") or {}),
            tokenizer=TokenizerConfig.from_dict(data.get("tokenizer") or {}),
            dataset=DatasetConfig.from_dict(data.get("dataset") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Fully-resolved config (defaults included) for the run manifest."""
        return dataclasses.asdict(self)


def load_config(path: Union[str, Path]) -> CorpusConfig:
    """Load and validate a corpus config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return CorpusConfig.from_dict(data)
