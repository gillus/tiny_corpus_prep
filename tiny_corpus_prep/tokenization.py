"""
Custom BPE tokenizer training (step 3a).

Byte-level BPE (GPT-2 style) trained on the mixed corpus shards:
- the full 256-byte alphabet is seeded into the vocab, so every character —
  including structural ones like { } [ ] : " — is representable and encodes
  losslessly (no UNK token needed);
- special tokens (EOS, PAD, plus any extras such as schedule/XML tags) get the
  lowest ids;
- after training, `check_single_tokens` verifies required structural characters
  encode as exactly one token.

Requires the optional `tokenizers` dependency:
    pip install 'tiny_corpus_prep[tokenizer]'
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Union

import polars as pl

logger = logging.getLogger(__name__)


def _require_tokenizers():
    try:
        import tokenizers  # noqa: F401
        return tokenizers
    except ImportError:
        raise ImportError(
            "Tokenizer training requires the 'tokenizers' package. "
            "Install with: pip install 'tiny_corpus_prep[tokenizer]'"
        )


def list_shard_files(shards_dir: Union[str, Path]) -> List[Path]:
    """Sorted shard files from a build_corpus output; raises if none exist."""
    shards_dir = Path(shards_dir)
    files = sorted(shards_dir.glob("shard_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No shard_*.parquet files in {shards_dir}. Run build_corpus first."
        )
    return files


def iter_shard_text_batches(
    shards_dir: Union[str, Path],
    text_column: str = "text",
    batch_size: int = 1000,
) -> Iterator[List[str]]:
    """Yield batches of document texts from corpus shards (streaming, in order)."""
    for shard in list_shard_files(shards_dir):
        df = pl.read_parquet(shard, columns=[text_column])
        for offset in range(0, len(df), batch_size):
            yield df[text_column].slice(offset, batch_size).to_list()


def train_bpe_tokenizer(
    text_batches: Iterable[List[str]],
    *,
    vocab_size: int,
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<|pad|>",
    extra_special_tokens: Sequence[str] = (),
    min_frequency: int = 2,
    show_progress: bool = False,
):
    """
    Train a byte-level BPE tokenizer on an iterable of text batches.

    Returns a tokenizers.Tokenizer. Special-token ids start at 0
    (eos=0, pad=1, then extras).
    """
    tk = _require_tokenizers()
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    special_tokens = [eos_token, pad_token, *extra_special_tokens]
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError(f"Duplicate special tokens: {special_tokens}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        # Seed all 256 byte tokens so any character is representable and
        # single-character structural tokens are guaranteed to exist.
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )

    def _flat() -> Iterator[str]:
        for batch in text_batches:
            yield from batch

    logger.info("Training byte-level BPE (vocab_size=%d)...", vocab_size)
    tokenizer.train_from_iterator(_flat(), trainer=trainer)
    logger.info("Trained tokenizer: %d tokens in vocab", tokenizer.get_vocab_size())
    return tokenizer


def check_single_tokens(tokenizer, chars: Sequence[str]) -> Dict[str, int]:
    """
    Return {char: token_count} for every char that does NOT encode as a
    single token. Empty dict = all good.
    """
    bad: Dict[str, int] = {}
    for ch in chars:
        n = len(tokenizer.encode(ch).ids)
        if n != 1:
            bad[ch] = n
    return bad


def train_tokenizer_from_shards(
    shards_dir: Union[str, Path],
    output_path: Union[str, Path],
    *,
    vocab_size: int,
    eos_token: str = "<|endoftext|>",
    pad_token: str = "<|pad|>",
    extra_special_tokens: Sequence[str] = (),
    require_single_tokens: Sequence[str] = (),
    min_frequency: int = 2,
    text_column: str = "text",
) -> Path:
    """
    Train a tokenizer on corpus shards, verify structural tokens, and save
    tokenizer.json to *output_path*. Returns the saved path.
    """
    tokenizer = train_bpe_tokenizer(
        iter_shard_text_batches(shards_dir, text_column=text_column),
        vocab_size=vocab_size,
        eos_token=eos_token,
        pad_token=pad_token,
        extra_special_tokens=extra_special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
    )

    bad = check_single_tokens(tokenizer, require_single_tokens)
    if bad:
        raise ValueError(
            f"Structural characters not encoded as single tokens: {bad}. "
            "This should not happen with the byte-level alphabet — "
            "check the tokenizer configuration."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    logger.info("Tokenizer saved to %s", output_path)
    return output_path


def load_tokenizer(path: Union[str, Path]):
    """Load a saved tokenizer.json."""
    _require_tokenizers()
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))
