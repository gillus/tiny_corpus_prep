#!/usr/bin/env python3
"""
Download and prepare data from Wikipedia or FineWeb-edu for corpus preparation.

This is Step 0 (optional) - downloads and prepares raw data that can be fed into prepare_corpus.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Force unbuffered output for progress bars
try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
except Exception:
    # Ignore reconfigure errors (e.g., in some WSL environments)
    pass

# Set environment variable as fallback
os.environ['PYTHONUNBUFFERED'] = '1'

try:
    import polars as pl
    from tqdm import tqdm
    import requests
except ImportError as e:
    print(f"Error: Missing required dependency: {e}", flush=True)
    print("Please install tiny_corpus_prep first:", flush=True)
    print("  uv pip install -e .", flush=True)
    sys.exit(1)

def _log_banner():
    logger.info("=" * 70)
    logger.info("tiny_corpus_prep - Data Download Tool (Step 0)")
    logger.info("=" * 70)


def _get_session() -> requests.Session:
    """Create a requests session with retry logic."""
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file_with_progress(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with a progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for the progress bar
    """
    session = _get_session()
    response = session.get(url, stream=True, timeout=60)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    logger.info("=" * 60)
    logger.info("Starting download: %s", desc)
    logger.info("File size: %.1f MB", total_size / (1024*1024))
    logger.info("=" * 60)

    # Check if we're in an interactive terminal
    use_progressbar = sys.stderr.isatty()

    if not use_progressbar:
        logger.info("Note: Running in non-interactive mode, showing periodic updates...")
    
    downloaded = 0
    last_print = 0
    print_interval = 10 * 1024 * 1024  # Print every 10MB
    
    with open(output_path, 'wb') as f:
        if use_progressbar:
            # Use tqdm progress bar
            with tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True, 
                unit_divisor=1024, 
                desc=desc,
                disable=False,
                file=sys.stderr,
                miniters=1,
                mininterval=0.5
            ) as pbar:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
                        downloaded += len(chunk)
        else:
            # Fallback: simple text updates
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 10MB
                    if downloaded - last_print >= print_interval:
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        logger.info("Downloaded: %.1f MB / %.1f MB (%.1f%%)", downloaded / (1024*1024), total_size / (1024*1024), percent)
                        last_print = downloaded

    logger.info("=" * 60)
    logger.info("Download complete: %s", desc)
    logger.info("Total downloaded: %.1f MB", downloaded / (1024*1024))
    logger.info("=" * 60)


def download_wikipedia(output_dir: str, date: str = "20251020") -> str:
    """
    Download and extract Simple Wikipedia data.
    
    Args:
        output_dir: Directory to save extracted data
        date: Wikipedia dump date (YYYYMMDD format)
    
    Returns:
        Path to output parquet file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dump_filename = f"simplewiki-{date}-pages-articles-multistream.xml.bz2"
    xml_filename = f"simplewiki-{date}-pages-articles-multistream.xml"
    dump_url = f"https://dumps.wikimedia.org/simplewiki/{date}/{dump_filename}"
    
    logger.info("=== Downloading Simple Wikipedia (%s) ===", date)

    # Check if files already exist
    dump_path = output_path / dump_filename
    xml_path = output_path / xml_filename

    if xml_path.exists():
        logger.info("XML file already exists: %s", xml_path)
    elif dump_path.exists():
        logger.info("Compressed file already exists: %s", dump_path)

        # Check if file is complete/valid by testing extraction
        logger.info("Verifying file integrity...")
        test_result = subprocess.run(
            ["bzip2", "-t", str(dump_path)],
            capture_output=True
        )

        if test_result.returncode != 0:
            logger.warning("File is corrupted or incomplete. Deleting and re-downloading...")
            dump_path.unlink()
            logger.info("Downloading from: %s", dump_url)
            download_file_with_progress(dump_url, dump_path, desc=f"Wikipedia {date}")
        else:
            logger.info("File is valid. Extracting...")

        logger.info("Extracting...")
        subprocess.run(["bzip2", "-dk", str(dump_path)], check=True)
    else:
        logger.info("Downloading from: %s", dump_url)
        download_file_with_progress(dump_url, dump_path, desc=f"Wikipedia {date}")

        logger.info("Extracting...")
        subprocess.run(["bzip2", "-d", str(dump_path)], check=True)

    # Check for WikiExtractor
    logger.info("=== Extracting Wikipedia content ===")
    try:
        subprocess.run(["python", "-m", "wikiextractor.WikiExtractor", "--help"],
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("WikiExtractor not found! Please install it: uv pip install wikiextractor")
        sys.exit(1)

    extracted_dir = output_path / "extracted"
    if extracted_dir.exists() and any(extracted_dir.rglob("wiki_*")):
        logger.info("Extracted files already exist in: %s", extracted_dir)
    else:
        logger.info("Running WikiExtractor (output to %s)...", extracted_dir)
        subprocess.run([
            "python", "-m", "wikiextractor.WikiExtractor",
            str(xml_path),
            "--json",
            "--output", str(extracted_dir)
        ], check=True)

    # Convert to parquet
    logger.info("=== Converting to Parquet ===")
    parquet_path = output_path / "wikipedia.parquet"

    json_files = list(extracted_dir.rglob("wiki_*"))
    if not json_files:
        logger.error("No extracted wiki files found!")
        sys.exit(1)

    logger.info("Found %d JSON files to process", len(json_files))
    
    data_rows = []
    for file_path in tqdm(json_files, desc="Processing files"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    title = record.get("title", "")
                    text = record.get("text", "")
                    
                    data_rows.append({
                        "filename": str(file_path),
                        "title": title,
                        "text": text,
                        "number_of_characters": len(text),
                        "number_of_words": len(text.split()),
                        "topic": "N-A",
                        "text_quality": 0,
                    })
                except json.JSONDecodeError as e:
                    logger.warning("Error decoding JSON in %s: %s", file_path, e)

    df = pl.DataFrame(data_rows)
    logger.info("Created DataFrame with %d rows", len(df))
    logger.info("%s", df.head())

    df.write_parquet(parquet_path)
    logger.info("Saved to: %s", parquet_path)
    
    return str(parquet_path)


def download_fineweb(output_dir: str, num_files: int = 1, start_index: int = 0) -> str:
    """
    Download FineWeb-edu data from HuggingFace.
    
    Args:
        output_dir: Directory to save downloaded files
        num_files: Number of files to download
        start_index: Starting file index
    
    Returns:
        Path to output parquet file (or directory if multiple files)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== Downloading FineWeb-edu ===")
    logger.info("Downloading %d file(s) starting from index %d", num_files, start_index)

    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"

    downloaded_files = []
    for i in range(start_index, start_index + num_files):
        filename = f"{i:03d}_00000.parquet"
        file_url = f"{base_url}/{filename}"
        output_file = output_path / filename

        if output_file.exists():
            logger.info("File already exists: %s", filename)
        else:
            download_file_with_progress(file_url, output_file, desc=f"FineWeb {filename}")

        downloaded_files.append(output_file)

    # If multiple files, combine them into one parquet
    if num_files == 1:
        result_path = downloaded_files[0]
        logger.info("Downloaded to: %s", result_path)
    else:
        logger.info("=== Combining %d files ===", num_files)
        dfs = []
        for file_path in tqdm(downloaded_files, desc="Reading files"):
            df = pl.read_parquet(file_path)
            dfs.append(df)

        combined_df = pl.concat(dfs)
        result_path = output_path / "fineweb_combined.parquet"
        combined_df.write_parquet(result_path)
        logger.info("Combined %d rows into: %s", len(combined_df), result_path)

    # Show preview
    preview_df = pl.read_parquet(result_path, use_pyarrow=True)
    logger.info("DataFrame preview (%d rows, %d columns):", len(preview_df), len(preview_df.columns))
    logger.info("%s", preview_df.head())
    logger.info("Columns: %s", preview_df.columns)
    
    return str(result_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare data for corpus preparation (Step 0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download Simple Wikipedia
  python bin/download_data.py --source wikipedia --output-dir data/raw
  
  # Download 5 FineWeb-edu files
  python bin/download_data.py --source fineweb --output-dir data/raw --num-files 5
  
  # Download specific Wikipedia dump date
  python bin/download_data.py --source wikipedia --output-dir data/raw --date 20240101

The output parquet file can then be processed with prepare_corpus.py (Step 1).
        """
    )
    
    parser.add_argument(
        "--source",
        choices=["wikipedia", "fineweb"],
        required=True,
        help="Data source to download"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to save downloaded data (default: data/raw)"
    )
    
    # Wikipedia-specific options
    parser.add_argument(
        "--date",
        default="20251020",
        help="Wikipedia dump date in YYYYMMDD format (default: 20251020)"
    )
    
    # FineWeb-specific options
    parser.add_argument(
        "--num-files",
        type=int,
        default=1,
        help="Number of FineWeb files to download (default: 1)"
    )
    
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting file index for FineWeb (default: 0)"
    )
    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _log_banner()

    logger.info("Configuration:")
    logger.info("  Source: %s", args.source)
    logger.info("  Output directory: %s", args.output_dir)
    if args.source == "wikipedia":
        logger.info("  Date: %s", args.date)
    else:
        logger.info("  Number of files: %d", args.num_files)
        logger.info("  Start index: %d", args.start_index)

    # Download based on source
    try:
        if args.source == "wikipedia":
            output_file = download_wikipedia(args.output_dir, args.date)
        else:  # fineweb
            output_file = download_fineweb(args.output_dir, args.num_files, args.start_index)

        logger.info("=" * 60)
        logger.info("Download complete!")
        logger.info("=" * 60)
        logger.info("Output file: %s", output_file)
        logger.info("Next step: Process with prepare_corpus.py")
        logger.info("Example:")
        logger.info("  python bin/prepare_corpus.py \\")
        logger.info("    --input %s \\", output_file)
        logger.info("    --output data/processed.parquet \\")
        logger.info("    --max-grade 10")

    except KeyboardInterrupt:
        logger.warning("Download interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

