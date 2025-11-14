# Quick Start Guide

Get started with tiny_corpus_prep in 5 minutes!

## Installation

### Quick Method

```bash
cd tiny_corpus_prep_lib
./install.sh
```

The script will guide you through installing UV and tiny_corpus_prep.

### Manual Method

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to the repository
cd tiny_corpus_prep_lib

# Install with annotation support (recommended)
uv pip install -e ".[annotators]"

# Or just core dependencies
uv pip install -e .
```

### Using pip

```bash
pip install -e ".[annotators]"
```

## Basic Usage

### Step 0 (Optional): Download Data

If you don't have data yet, you can download from these sources:

#### Option A: Simple Wikipedia

Download and extract Simple Wikipedia (good for general knowledge):

```bash
# Download, extract, and convert to parquet
python bin/download_data.py --source wikipedia --output-dir data/raw

# For real-time progress bars, use -u flag (unbuffered output)
python -u bin/download_data.py --source wikipedia --output-dir data/raw

# This will create: data/raw/wikipedia.parquet
```

The script will:
1. Download the Simple Wikipedia XML dump
2. Extract with `bzip2`
3. Use WikiExtractor to convert to JSON
4. Create a parquet file with columns: `filename`, `title`, `text`, `number_of_characters`, `number_of_words`, `topic`, `text_quality`

**Note:** Requires `bzip2` and `wikiextractor`:
```bash
# Install WikiExtractor
uv pip install wikiextractor
# or: pip install wikiextractor

# System dependency (if needed)
sudo apt-get install bzip2  # Debian/Ubuntu
brew install bzip2          # macOS
```

#### Option B: FineWeb-edu (HuggingFace)

Download pre-processed educational web content:

```bash
# Download 1 file (default)
python bin/download_data.py --source fineweb --output-dir data/raw

# Download multiple files
python bin/download_data.py --source fineweb --output-dir data/raw --num-files 5

# This will create: data/raw/000_00000.parquet (or fineweb_combined.parquet for multiple files)
```

**Note:** Downloads automatically with progress bars using Python's `requests` library (included in core dependencies).

#### Manual Data Preparation

If you have your own data, convert it to parquet format (see Step 1 below).

---

### Step 1: Prepare Your Data

If you already have data or used Step 0, ensure it's a **parquet file with a `text` column**.

```python
import polars as pl

# Create from list of texts
texts = [
    "The cat sat on the mat.",
    "Quantum mechanics is fascinating.",
    "I love programming!",
]

df = pl.DataFrame({"text": texts})
df.write_parquet("my_data.parquet")
```

Or convert from existing formats:

```python
# From TXT file
with open("data.txt") as f:
    texts = [line.strip() for line in f if line.strip()]
pl.DataFrame({"text": texts}).write_parquet("data.parquet")

# From JSONL
df = pl.read_ndjson("data.jsonl")
df.select("text").write_parquet("data.parquet")
```

### Step 2: Process Your Data

```python
from tiny_corpus_prep import process_corpus

stats = process_corpus(
    input_path="my_data.parquet",
    output_path="processed.parquet",
    max_grade=8.0,  # Keep only middle-school level text
    normalize=True,  # Clean up text
    dedup=True,  # Remove duplicates
)

print(f"Processed {stats['total_rows']} rows!")
```

### Step 3: Check Your Results

Two files are created:
- `processed.parquet` - Your processed data
- `processed.json` - Statistics about the data

```python
import polars as pl
import json

# View processed data
df = pl.read_parquet("processed.parquet")
print(df.head())

# View statistics
with open("processed.json") as f:
    stats = json.load(f)
    print(f"Total rows: {stats['total_rows']}")
    print(f"Mean text length: {stats['text_stats']['mean_length']:.1f}")
```

## Common Tasks

### Filter by Keywords

```python
process_corpus(
    input_path="data.parquet",
    output_path="science_data.parquet",
    keywords=["science", "physics", "chemistry"],
)
```

### Use Synonym Mapping

```python
import json

# Create synonym map
synonyms = {
    "automobile": "car",
    "doctor": "physician",
}

with open("synonyms.json", "w") as f:
    json.dump(synonyms, f)

# Process with synonyms
process_corpus(
    input_path="data.parquet",
    output_path="simplified.parquet",
    synonyms_map_path="synonyms.json",
)
```

### Add Custom Annotations

```python
from tiny_corpus_prep import DataPipeline, CustomFunctionAnnotator, read_parquet, write_parquet_with_stats

def count_features(text: str) -> dict:
    return {
        "word_count": len(text.split()),
        "sentence_count": text.count(".") + text.count("!") + text.count("?"),
    }

df = read_parquet("data.parquet")

pipeline = (
    DataPipeline()
    .add_readability_filter(max_grade=8.0)
    .add_annotator(CustomFunctionAnnotator(count_features))
)

df_out = pipeline.process(df)
write_parquet_with_stats(df_out, "annotated.parquet")
```

### Use Gemini for Classification

```bash
# Set your API key
export GOOGLE_API_KEY="your_key_here"
```

```python
from tiny_corpus_prep import GeminiAnnotator, DataPipeline, read_parquet, write_parquet_with_stats

df = read_parquet("data.parquet")

pipeline = (
    DataPipeline()
    .add_annotator(GeminiAnnotator())
)

df_out = pipeline.process(df)
write_parquet_with_stats(df_out, "classified.parquet")

# Output will have 'topic' and 'education' columns
print(df_out.select(["text", "topic", "education"]).head())
```

## CLI Usage

Process from command line:

```bash
# Basic processing
python bin/prepare_corpus.py \
  --input data.parquet \
  --output processed.parquet \
  --max-grade 8

# With keyword filter
python bin/prepare_corpus.py \
  --input data.parquet \
  --output filtered.parquet \
  --keywords science,math,physics \
  --max-grade 10

# With Gemini annotation
python bin/prepare_corpus.py \
  --input data.parquet \
  --output classified.parquet \
  --annotate gemini \
  --api-key YOUR_KEY
```

## Complete Workflow Example

Here's a full example using Step 0 to download data and process it:

```bash
# Step 0: Download Wikipedia data (use -u for real-time progress)
python -u bin/download_data.py --source wikipedia --output-dir data/raw

# Step 1 & 2: Process the downloaded data
python bin/prepare_corpus.py \
  --input data/raw/wikipedia.parquet \
  --output data/processed/wikipedia_filtered.parquet \
  --max-grade 8 \
  --keywords science,math,physics,biology

# Step 3: Examine the results
python -c "
import polars as pl
import json

df = pl.read_parquet('data/processed/wikipedia_filtered.parquet')
print(f'Total rows: {len(df)}')
print(df.head())

with open('data/processed/wikipedia_filtered.json') as f:
    stats = json.load(f)
    print(f\"\\nMean text length: {stats['text_stats']['mean_length']:.1f} chars\")
"
```

Or with Python API:

```python
# Step 0: Download FineWeb-edu data
import subprocess
subprocess.run([
    "python", "bin/download_data.py",
    "--source", "fineweb",
    "--output-dir", "data/raw",
    "--num-files", "3"
])

# Steps 1-3: Process and examine
from tiny_corpus_prep import process_corpus
import polars as pl

stats = process_corpus(
    input_path="data/raw/fineweb_combined.parquet",
    output_path="data/processed/fineweb_filtered.parquet",
    max_grade=10.0,
    normalize=True,
    dedup=True,
)

print(f"Processed {stats['total_rows']} rows")
print(f"Mean text length: {stats['text_stats']['mean_length']:.1f}")

# View some results
df = pl.read_parquet("data/processed/fineweb_filtered.parquet")
print(df.select(["text"]).head(3))
```

## What's Next?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 🔄 Migrating from v0.1? See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)
- 💡 Check [examples/](examples/) directory for more examples
- 🎯 Create your own annotator by subclassing `BaseAnnotator`

## Troubleshooting

**"Required column 'text' not found"**
- Make sure your parquet file has a column named `text`
- Or specify a different column: `text_column="your_column_name"`

**"Google API Key not found"**
- Set environment variable: `export GOOGLE_API_KEY="your_key"`
- Or pass directly: `GeminiAnnotator(api_key="your_key")`

**Import errors for Gemini**
- Install optional dependencies: `uv pip install -e ".[annotators]"`

**"bzip2 not found"**
- Install system dependency:
  - Debian/Ubuntu: `sudo apt-get install bzip2`
  - macOS: `brew install bzip2`

**"WikiExtractor not found"**
- Install the downloaders extra: `uv pip install -e ".[downloaders]"`
- Or install directly: `pip install wikiextractor`

**Download script is slow**
- Wikipedia dumps are large (several GB), download may take time
- Use `--date` to select smaller/older dumps if needed
- FineWeb-edu files are smaller but you may want to download multiple files

**Progress bars not showing / output buffered**
- Use `python -u` for unbuffered output: `python -u bin/download_data.py ...`
- Or set environment variable: `PYTHONUNBUFFERED=1 python bin/download_data.py ...`
- Avoid `uv run` if you want real-time progress (it buffers output)

## Need Help?

- Check the [README.md](README.md) for full documentation
- Look at [examples/basic_usage.py](examples/basic_usage.py) for working code
- Review [examples/gemini_example.py](examples/gemini_example.py) for Gemini usage

