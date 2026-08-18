"""
Token packing into an HF Arrow training dataset (step 3b).

Documents are tokenized shard by shard, joined with an EOS token after every
document, split train/validation at *document* level (deterministic hash of
seed + global doc index, so no document straddles the split), then chunked
into fixed-length sequences and saved as a datasets.DatasetDict.

Streaming: packed sequences are flushed to parquet part files every
~flush_tokens tokens, so memory stays bounded regardless of corpus size; the
final DatasetDict is built from the part files via Arrow memory-mapping.

Requires the optional `datasets` dependency:
    pip install 'tiny_corpus_prep[tokenizer]'
"""
from __future__ import annotations

import hashlib
import logging
import shutil
from array import array
from pathlib import Path
from typing import Any, Dict, Union

import polars as pl

from .tokenization import list_shard_files, load_tokenizer

logger = logging.getLogger(__name__)

_SPLITS = ("train", "validation")


class _SplitPacker:
    """
    Accumulates token ids for one split and flushes complete fixed-length
    sequences to a parquet part file, keeping memory bounded.
    """

    def __init__(self, part_path: Path, seq_length: int, array_code: str,
                 np_dtype, pa_dtype, flush_tokens: int):
        import numpy as np  # noqa: F401 (kept for the frombuffer below)

        self.part_path = part_path
        self.seq_length = seq_length
        self.array_code = array_code
        self.np_dtype = np_dtype
        self.pa_dtype = pa_dtype
        self.flush_tokens = max(flush_tokens, seq_length)
        self.buf = array(array_code)
        self.raw_tokens = 0
        self.sequences = 0
        self._writer = None

    def add(self, ids, eos_id: int) -> None:
        self.buf.extend(ids)
        self.buf.append(eos_id)
        self.raw_tokens += len(ids) + 1
        if len(self.buf) >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq

        n = len(self.buf) // self.seq_length
        if n == 0:
            return
        n_tokens = n * self.seq_length
        flat = np.frombuffer(self.buf, dtype=self.np_dtype)[:n_tokens]
        offsets = np.arange(0, n_tokens + 1, self.seq_length, dtype=np.int32)
        list_array = pa.ListArray.from_arrays(
            pa.array(offsets), pa.array(flat, type=self.pa_dtype)
        )
        table = pa.Table.from_arrays([list_array], names=["input_ids"])
        if self._writer is None:
            self.part_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = pq.ParquetWriter(str(self.part_path), table.schema)
        self._writer.write_table(table)
        self.sequences += n
        self.buf = array(self.array_code, self.buf[n_tokens:])

    def close(self) -> int:
        """Final flush; returns the dropped tail length in tokens."""
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return len(self.buf)


def _is_validation_doc(seed: int, doc_index: int, val_fraction: float) -> bool:
    """Deterministic, order-independent doc-level split assignment."""
    if val_fraction <= 0.0:
        return False
    digest = hashlib.blake2b(
        f"{seed}:{doc_index}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64 < val_fraction


def pack_shards_to_dataset(
    shards_dir: Union[str, Path],
    tokenizer_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    seq_length: int = 1024,
    val_fraction: float = 0.005,
    seed: int = 42,
    eos_token: str = "<|endoftext|>",
    text_column: str = "text",
    source_column: str = "source",
    batch_size: int = 1000,
    flush_tokens: int = 8_000_000,
) -> Dict[str, Any]:
    """
    Tokenize corpus shards and save a packed DatasetDict to *output_dir*.

    Each row of the output is a fixed-length `input_ids` sequence
    (uint16 when the vocab fits, else uint32). Sequences are flushed to disk
    every ~flush_tokens tokens, so memory stays bounded at any corpus size.
    Returns token-level statistics.
    """
    try:
        import numpy as np
        import pyarrow as pa
        from datasets import Dataset, DatasetDict, Features, Sequence, Value
    except ImportError:
        raise ImportError(
            "Packing requires 'datasets' and 'numpy'. "
            "Install with: pip install 'tiny_corpus_prep[tokenizer]'"
        )

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id(eos_token)
    if eos_id is None:
        raise ValueError(f"EOS token {eos_token!r} not found in tokenizer vocab")

    if vocab_size <= 65536:
        array_code, np_dtype, pa_dtype, hf_dtype = "H", np.uint16, pa.uint16(), "uint16"
    else:
        array_code, np_dtype, pa_dtype, hf_dtype = "I", np.uint32, pa.uint32(), "uint32"

    output_dir = Path(output_dir)
    tmp_dir = output_dir.with_name(output_dir.name + "_tmp")
    packers = {
        split: _SplitPacker(
            tmp_dir / f"{split}.parquet", seq_length,
            array_code, np_dtype, pa_dtype, flush_tokens,
        )
        for split in _SPLITS
    }
    docs = {split: 0 for split in _SPLITS}
    chars = {split: 0 for split in _SPLITS}
    tokens_per_source: Dict[str, Dict[str, int]] = {split: {} for split in _SPLITS}
    vocab_seen = np.zeros(vocab_size, dtype=bool)
    vocab_seen[eos_id] = True

    doc_index = 0
    for shard in list_shard_files(shards_dir):
        columns = [text_column]
        has_source = source_column in pl.read_parquet_schema(shard)
        if has_source:
            columns.append(source_column)
        df = pl.read_parquet(shard, columns=columns)

        for offset in range(0, len(df), batch_size):
            batch = df.slice(offset, batch_size)
            texts = batch[text_column].to_list()
            sources = (
                batch[source_column].to_list() if has_source else ["unknown"] * len(batch)
            )
            encodings = tokenizer.encode_batch(texts)

            for text, source, encoding in zip(texts, sources, encodings):
                split = (
                    "validation"
                    if _is_validation_doc(seed, doc_index, val_fraction)
                    else "train"
                )
                doc_index += 1
                packers[split].add(encoding.ids, eos_id)
                docs[split] += 1
                chars[split] += len(text)
                if encoding.ids:
                    vocab_seen[np.asarray(encoding.ids)] = True
                per_source = tokens_per_source[split]
                per_source[source] = per_source.get(source, 0) + len(encoding.ids) + 1

    dropped_tails = {split: packers[split].close() for split in _SPLITS}
    logger.info(
        "Tokenized %d docs: %d train tokens, %d validation tokens",
        doc_index, packers["train"].raw_tokens, packers["validation"].raw_tokens,
    )

    # Build the DatasetDict from the flushed part files (Arrow memory-mapped —
    # the packed corpus is never fully loaded into RAM).
    empty_features = Features({"input_ids": Sequence(Value(hf_dtype))})
    dataset_splits = {}
    split_stats: Dict[str, Any] = {}
    for split in _SPLITS:
        packer = packers[split]
        if packer.sequences > 0:
            dataset_splits[split] = Dataset.from_parquet(str(packer.part_path))
        else:
            dataset_splits[split] = Dataset.from_dict(
                {"input_ids": []}, features=empty_features
            )
        split_stats[split] = {
            "documents": docs[split],
            "raw_tokens": int(packer.raw_tokens),
            "sequences": int(packer.sequences),
            "packed_tokens": int(packer.sequences * seq_length),
            "dropped_tail_tokens": int(dropped_tails[split]),
            "tokens_per_source": tokens_per_source[split],
        }

    DatasetDict(dataset_splits).save_to_disk(str(output_dir))
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Packed dataset saved to %s", output_dir)

    total_raw = sum(s["raw_tokens"] for s in split_stats.values())
    total_chars = sum(chars.values())
    stats = {
        "seq_length": seq_length,
        "val_fraction": val_fraction,
        "seed": seed,
        "vocab_size": vocab_size,
        "eos_token": eos_token,
        "eos_token_id": int(eos_id),
        "dtype": hf_dtype,
        "total_documents": doc_index,
        "total_raw_tokens": total_raw,
        "vocab_coverage": round(int(vocab_seen.sum()) / vocab_size, 4) if vocab_size else 0.0,
        "chars_per_token": round(total_chars / total_raw, 3) if total_raw else 0.0,
        "splits": split_stats,
    }
    return stats
