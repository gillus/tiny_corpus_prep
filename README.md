\
# tiny_corpus_prep

Minimal, fast corpus preparation for training a tiny GPT-like model using **Polars**

## Features

- **Data downloading**: Built-in tools to download Wikipedia or FineWeb-edu datasets
- **Polars-based processing**: Fast, efficient DataFrame operations
- **Text normalization**: Lowercase, punctuation cleanup
- **Readability filtering**: Uses `textstat` for Flesch-Kincaid grade level filtering
- **Keyword filtering**: Filter by topic keywords
- **Vocabulary simplification**: Map synonyms to canonical forms
- **CEFR-based synonym mapping**: Intelligent word simplification using CEFR language levels (see [examples/cefr_synonym_example.py](examples/cefr_synonym_example.py))
- **Custom annotations**: Extend with your own annotation logic
- **Gemini integration**: Built-in support for Google Gemini API annotations
- **Statistics generation**: Automatic JSON stats for output datasets
- **Deduplication**: Remove duplicate text entries

## Installation

### Quick Install (Recommended)

Run the installation script:

```bash
./install.sh
```

This will:
1. Install UV if needed
2. Install tiny_corpus_prep with your choice of features

### Manual Install with UV

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with all features (recommended)
uv pip install -e ".[annotators,dedup]"

# Or install with annotation support only
uv pip install -e ".[annotators]"

# Or just core dependencies
uv pip install -e .
```

### Alternative: pip

```bash
# With all features
pip install -e ".[annotators,dedup]"

# Or with annotations only
pip install -e ".[annotators]"
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

## Quick Start

### Step 0 (Optional): Download Data

If you need training data, use the download script to get Wikipedia or FineWeb-edu data:

```bash
# Download Simple Wikipedia (use -u for real-time progress bars)
python -u bin/download_data.py --source wikipedia --output-dir data/raw

# Or download FineWeb-edu (HuggingFace)
python -u bin/download_data.py --source fineweb --output-dir data/raw --num-files 5
```

**Tip:** Use `python -u` for unbuffered output to see progress bars in real-time.

See [QUICKSTART.md](QUICKSTART.md) for detailed instructions.

### Config-driven corpus building (recommended)

Describe sources, per-source processing profiles, and mix weights in one YAML
file, then build the full corpus in a single reproducible run:

```bash
python bin/build_corpus.py --config examples/corpus_config.yaml
```

Each source is processed with its own *profile* (so prose can be
readability/vocabulary-filtered while structured text like JSON/XML passes
through untouched), then all sources are mixed at the configured word-share
weights with a seeded RNG. Small sources can be upsampled to hit their share.

Profiles also expose quality heuristics: boilerplate line stripping
(`strip_boilerplate`), a Gopher-style degenerate-repetition filter
(`repetition_filter`), paragraph-level dedup across documents
(`paragraph_dedup`), MinHash near-dedup (`dedup_mode: both`), and an optional
language gate (`language: en`, requires `pip install 'tiny_corpus_prep[quality]'`).
Set `processing.n_workers` to parallelize the per-document filters across CPU
cores. The output directory contains:

- `processed/<source>.parquet` — per-source filtered data (+ stats JSON)
- `shards/shard_00000.parquet` … — the final shuffled corpus (`text`, `source`)
- `manifest.json` — resolved config, library versions, and per-stage stats,
  so every corpus is traceable to the exact settings that produced it

The same config + seed reproduces the corpus byte-for-byte. See
[examples/corpus_config.yaml](examples/corpus_config.yaml) for a commented example.

### Tokenizer training & packing

After building the mixed corpus, train a custom byte-level BPE tokenizer on it
and pack everything into a training-ready HuggingFace dataset:

```bash
# Requires: pip install 'tiny_corpus_prep[tokenizer]'
python bin/tokenize_corpus.py --config examples/corpus_config.yaml
```

This adds to the output directory:

- `tokenizer/tokenizer.json` — byte-level BPE (configurable vocab size,
  EOS/PAD + custom special tokens, structural characters like `{ } [ ] : "`
  verified to be single tokens), saved alongside the data for reproducibility
- `dataset/` — a `DatasetDict` with `train`/`validation` splits of fixed-length
  `input_ids` sequences (documents EOS-joined then chunked; the split is at
  document level, so no document straddles train/val)
- `tokenization_manifest.json` — token-level stats: total tokens, per-source
  token distribution, vocab coverage, chars/token

Load it for training:

```python
from datasets import load_from_disk
from tokenizers import Tokenizer

ds = load_from_disk("data/corpus_v1/dataset")        # ds["train"]["input_ids"]
tok = Tokenizer.from_file("data/corpus_v1/tokenizer/tokenizer.json")
```

An existing tokenizer is reused on re-runs (pass `--retrain` to force
retraining, e.g. after changing the corpus mix).

### Pilot training

Two helper scripts close the loop from corpus to model:

```bash
# Generate the structured/schedule source (persona-conditioned 7-day
# household schedules in XML/JSON/key-value; deterministic per seed)
python bin/generate_structured.py --output data/raw/schedules.parquet --num-docs 30000

# Pretrain a ~50M Llama-architecture model from scratch on the built corpus
# (requires: pip install 'tiny_corpus_prep[train]')
python bin/train_pilot.py --corpus-dir data/corpus_v1 --output-dir runs/pilot50m
```

`train_pilot.py` reads `dataset/` + `tokenizer/` from the corpus dir,
auto-selects precision (bf16 on Ampere+, fp16 on Turing cards like T4/RTX 20xx),
supports `--resume`, and prints sample generations at the end. Model size is
fully configurable (`--hidden-size`, `--num-layers`, ...) for the
50M → 150M → 600M ladder.

### CLI Usage

Process a parquet file with basic filtering:

```bash
python bin/prepare_corpus.py \
  --input data.parquet \
  --output out.parquet \
  --max-grade 8 \
  --keywords science,math
```

With Gemini annotation:

```bash
python bin/prepare_corpus.py \
  --input data.parquet \
  --output out.parquet \
  --max-grade 8 \
  --annotate gemini \
  --api-key YOUR_GOOGLE_API_KEY
```

All CLI options:

```bash
python bin/prepare_corpus.py \
  --input data.parquet \
  --output out.parquet \
  --text-column text \
  --synonyms synonyms.json \
  --keywords science,math,physics \
  --max-grade 8.0 \
  --annotate gemini \
  --api-key YOUR_API_KEY \
  --gemini-model gemini-2.5-flash-lite
```

### Python API

**Simple processing:**

```python
from tiny_corpus_prep import process_corpus

# Process with filtering and stats
stats = process_corpus(
    input_path="data.parquet",
    output_path="out.parquet",
    keywords=["science", "math"],
    max_grade=8.0,
    dedup=True,
)

print(f"Processed {stats['total_rows']} rows")
```

**Advanced pipeline with custom annotator:**

```python
import polars as pl
from tiny_corpus_prep import DataPipeline, GeminiAnnotator, read_parquet, write_parquet_with_stats

# Read input
df = read_parquet("data.parquet")

# Build pipeline
pipeline = (
    DataPipeline(text_column="text", normalize=True, dedup=True)
    .add_keyword_filter(["science", "math"])
    .add_readability_filter(max_grade=8.0)
    .add_synonym_mapper(mapping_path="synonyms.json")
    .add_annotator(GeminiAnnotator(api_key="YOUR_KEY"))
)

# Process
df_processed = pipeline.process(df)

# Save with stats
stats = write_parquet_with_stats(df_processed, "out.parquet")
```

**Custom annotation function:**

```python
from tiny_corpus_prep import CustomFunctionAnnotator

def my_annotator(text: str) -> dict:
    """Add custom metadata."""
    return {
        "word_count": len(text.split()),
        "has_numbers": any(c.isdigit() for c in text),
    }

pipeline = DataPipeline().add_annotator(
    CustomFunctionAnnotator(my_annotator)
)
```

**Readability filtering:**

```python
from tiny_corpus_prep import filter_by_readability, is_middle_school_level
import polars as pl

df = pl.DataFrame({"text": ["Simple text.", "Complex terminology."]})

# Filter by grade level
df_filtered = filter_by_readability(df, max_grade=8.0)

# Or use the helper function
texts = ["Some text here", "Another text"]
readable = [t for t in texts if is_middle_school_level(t)]
```

## API Reference

### Pipeline

- `process_corpus(input_path, output_path, **options)`: Main function for processing
- `DataPipeline`: Fluent API for building custom pipelines

### Annotators

- `BaseAnnotator`: Base class for custom annotators
- `GeminiAnnotator`: Google Gemini API annotator for topic/education classification
- `CustomFunctionAnnotator`: Wrap a function as an annotator

### Filters

- `filter_by_readability(df, max_grade, text_column)`: Filter by Flesch-Kincaid grade
- `filter_by_keywords(df, keywords, text_column)`: Filter by keyword presence
- `is_middle_school_level(text)`: Check if text is ≤ 8th grade level

### I/O

- `read_parquet(path, text_column)`: Read parquet with validation
- `write_parquet(df, path)`: Write parquet file
- `write_parquet_with_stats(df, path, text_column)`: Write parquet + JSON stats
- `generate_stats(df, text_column)`: Generate statistics dictionary

## Input/Output Format

**Input:** Parquet file with required `text` column (configurable name)

**Output:** Parquet file with:
- Original `text` column
- Any annotation columns added by annotators
- Filtered/processed rows

**Statistics JSON** (same name as output parquet):
```json
{
  "total_rows": 1000,
  "total_columns": 3,
  "columns": ["text", "topic", "education"],
  "text_stats": {
    "min_length": 10,
    "max_length": 500,
    "mean_length": 120.5,
    "median_length": 115.0,
    "total_characters": 120500,
    "empty_or_null_count": 0
  },
  "column_stats": {
    "topic": {
      "dtype": "String",
      "null_count": 5,
      "unique_values": 15,
      "top_values": [...]
    }
  }
}
```

## Example: Gemini Annotator

The built-in Gemini annotator classifies text into topics and education levels:

```python
from tiny_corpus_prep import GeminiAnnotator

annotator = GeminiAnnotator(
    api_key="YOUR_KEY",
    model_name="gemini-2.5-flash-lite"
)

# Annotates with 'topic' and 'education' columns
result = annotator.annotate("The Copy Number Variation analysis...")
# {'topic': 'Life Sciences', 'education': 'PhD degree'}
```

Topics include: Arts & Humanities, Mathematics, Computer Science, Health & Medicine, and more.

Education levels: primary school, middle school, high school, university degree, PhD degree.

## Notes

- All processing uses Polars for efficiency
- Input must be parquet with a `text` column
- Readability uses `textstat` library (Flesch-Kincaid grade)
- Synonym mapping is word-level replacement
- Deduplication is based on exact text matches
- Statistics JSON automatically generated alongside output

## Building Synonym Maps

### CEFR-Based Intelligent Mapping

Build a smarter synonym mapping that replaces difficult words with easier alternatives based on CEFR language levels:

```bash
python bin/build_synmap_from_cefr.py \
  --cefr_csv data/cefr_wordlist.csv \
  --out_dir output/synonyms \
  --easy_levels A1,A2 \
  --difficult_levels B2,C1,C2
```

This generates:
- `synonyms.json` - JSON mapping for use in the pipeline
- `synonyms.csv` - Detailed mapping with CEFR levels and sources
- `unmapped.txt` - Words that couldn't be simplified
- `build_stats.txt` - Statistics about the mapping process

See [examples/cefr_synonym_example.py](examples/cefr_synonym_example.py) and [bin/build_synmap_from_cefr.py](bin/build_synmap_from_cefr.py) for complete usage examples.

## Downloading Data (Step 0)

The `download_data.py` script helps you get training data from public sources:

### Wikipedia

Download and process Simple English Wikipedia:

```bash
# Install optional dependency
uv pip install -e .

# Download and convert to parquet
python bin/download_data.py --source wikipedia --output-dir data/raw

# Specify a different date
python bin/download_data.py --source wikipedia --output-dir data/raw --date 20240101
```

This will create a parquet file with columns: `filename`, `title`, `text`, `number_of_characters`, `number_of_words`, `topic`, `text_quality`.

Downloads show a progress bar using Python's `requests` library.

**Requirements**: `bzip2` (system dependency) and `wikiextractor` (installed with the core package)

### FineWeb-edu (HuggingFace)

Download pre-processed educational web content:

```bash
# Download 1 file
python bin/download_data.py --source fineweb --output-dir data/raw

# Download multiple files
python bin/download_data.py --source fineweb --output-dir data/raw --num-files 10 --start-index 0
```

Downloads show a progress bar for each file.

The downloaded parquet files can be directly processed with `prepare_corpus.py`.

## License

MIT
