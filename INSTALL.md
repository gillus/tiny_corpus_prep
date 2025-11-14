# Installation Guide

## Recommended: Install with UV

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, written in Rust. It's 10-100x faster than pip.

### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or on Windows:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Install tiny_corpus_prep

```bash
# Navigate to the repository
cd /home/gillus/tiny_corpus_prep_lib

# Install with all features (recommended)
uv pip install -e ".[annotators]"

# Or just core dependencies
uv pip install -e .
```

### 3. Verify Installation

```bash
python test_refactored.py
```

## Alternative: Install with pip

If you prefer to use pip:

```bash
cd /home/gillus/tiny_corpus_prep_lib
pip install -e ".[annotators]"
```

## What Gets Installed

### Core Dependencies (always installed)
- **polars** - Fast DataFrame library
- **textstat** - Readability calculations
- **pyarrow** - Parquet file support
- **tqdm** - Progress bars

### Optional Dependencies (with `[annotators]`)
- **google-generativeai** - Gemini API client
- **python-dotenv** - Environment variable management

## Quick Test

After installation, run a quick test:

```python
python -c "from tiny_corpus_prep import process_corpus, GeminiAnnotator; print('✓ Installation successful!')"
```

## Troubleshooting

**UV not found after installation**
- Restart your terminal or run: `source ~/.bashrc` (Linux/Mac)
- On Windows, restart your terminal

**Import errors**
- Make sure you installed with `[annotators]` for Gemini support
- Check you're in the right directory: `cd /home/gillus/tiny_corpus_prep_lib`

**Permission errors**
- UV doesn't need sudo/admin rights
- If using pip, you may need `--user` flag

## Development Installation

For development work:

```bash
# Clone and install in development mode
git clone <repo-url>
cd tiny_corpus_prep_lib

# Install with UV
uv pip install -e ".[annotators]"

# Run tests
python test_refactored.py

# Try examples
python examples/basic_usage.py
```

## Uninstalling

```bash
# With UV
uv pip uninstall mini-gpt-prep

# With pip
pip uninstall mini-gpt-prep
```

## More Information

- **Quick Start**: See [QUICKSTART.md](QUICKSTART.md)
- **Full Documentation**: See [README.md](README.md)
- **Examples**: Check the `examples/` directory

