#!/usr/bin/env python3
"""
Download and prepare data from Wikipedia or FineWeb-edu for corpus preparation.

This is Step 0 (optional) - downloads and prepares raw data that can be fed into prepare_corpus.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

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

# Immediate output to verify script is running
print("=" * 70, flush=True)
print("tiny_corpus_prep - Data Download Tool (Step 0)", flush=True)
print("=" * 70, flush=True)


def download_file_with_progress(url: str, output_path: Path, desc: str = "Downloading") -> None:
    """
    Download a file with a progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        desc: Description for the progress bar
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    print(f"\n{'='*60}", flush=True)
    print(f"Starting download: {desc}", flush=True)
    print(f"File size: {total_size / (1024*1024):.1f} MB", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Check if we're in an interactive terminal
    use_progressbar = sys.stderr.isatty()
    
    if not use_progressbar:
        print("Note: Running in non-interactive mode, showing periodic updates...", flush=True)
    
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
                    
                    # Print progress every 10MB
                    if downloaded - last_print >= print_interval:
                        percent = (downloaded / total_size * 100) if total_size > 0 else 0
                        print(f"Downloaded: {downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", flush=True)
                        last_print = downloaded
    
    print(f"\n{'='*60}", flush=True)
    print(f"✓ Download complete: {desc}", flush=True)
    print(f"Total downloaded: {downloaded / (1024*1024):.1f} MB", flush=True)
    print(f"{'='*60}\n", flush=True)


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
    
    print(f"\n=== Downloading Simple Wikipedia ({date}) ===", flush=True)
    
    # Check if files already exist
    dump_path = output_path / dump_filename
    xml_path = output_path / xml_filename
    
    if xml_path.exists():
        print(f"✓ XML file already exists: {xml_path}", flush=True)
    elif dump_path.exists():
        print(f"✓ Compressed file already exists: {dump_path}", flush=True)
        
        # Check if file is complete/valid by testing extraction
        print("Verifying file integrity...", flush=True)
        test_result = subprocess.run(
            ["bzip2", "-t", str(dump_path)], 
            capture_output=True
        )
        
        if test_result.returncode != 0:
            print("⚠️  File is corrupted or incomplete. Deleting and re-downloading...", flush=True)
            dump_path.unlink()
            print(f"Downloading from: {dump_url}", flush=True)
            download_file_with_progress(dump_url, dump_path, desc=f"Wikipedia {date}")
        else:
            print("✓ File is valid. Extracting...", flush=True)
        
        print("Extracting...", flush=True)
        subprocess.run(["bzip2", "-dk", str(dump_path)], check=True)
    else:
        print(f"Downloading from: {dump_url}", flush=True)
        download_file_with_progress(dump_url, dump_path, desc=f"Wikipedia {date}")
        
        print("\nExtracting...", flush=True)
        subprocess.run(["bzip2", "-d", str(dump_path)], check=True)
    
    # Check for WikiExtractor
    print("\n=== Extracting Wikipedia content ===", flush=True)
    try:
        subprocess.run(["python", "-m", "wikiextractor.WikiExtractor", "--help"], 
                      capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: WikiExtractor not found!")
        print("Please install it:")
        print("  uv pip install wikiextractor")
        print("  # or")
        print("  pip install wikiextractor")
        sys.exit(1)
    
    extracted_dir = output_path / "extracted"
    if extracted_dir.exists() and any(extracted_dir.rglob("wiki_*")):
        print(f"✓ Extracted files already exist in: {extracted_dir}")
    else:
        print(f"Running WikiExtractor (output to {extracted_dir})...")
        subprocess.run([
            "python", "-m", "wikiextractor.WikiExtractor",
            str(xml_path),
            "--json",
            "--output", str(extracted_dir)
        ], check=True)
    
    # Convert to parquet
    print("\n=== Converting to Parquet ===")
    parquet_path = output_path / "wikipedia.parquet"
    
    json_files = list(extracted_dir.rglob("wiki_*"))
    if not json_files:
        print("Error: No extracted wiki files found!")
        sys.exit(1)
    
    print(f"Found {len(json_files)} JSON files to process")
    
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
                    print(f"Warning: Error decoding JSON in {file_path}: {e}")
    
    df = pl.DataFrame(data_rows)
    print(f"\nCreated DataFrame with {len(df)} rows")
    print(df.head())
    
    df.write_parquet(parquet_path)
    print(f"\n✓ Saved to: {parquet_path}")
    
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
    
    print(f"\n=== Downloading FineWeb-edu ===", flush=True)
    print(f"Downloading {num_files} file(s) starting from index {start_index}", flush=True)
    
    base_url = "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"
    
    downloaded_files = []
    for i in range(start_index, start_index + num_files):
        filename = f"{i:03d}_00000.parquet"
        file_url = f"{base_url}/{filename}"
        output_file = output_path / filename
        
        if output_file.exists():
            print(f"✓ File already exists: {filename}", flush=True)
        else:
            download_file_with_progress(file_url, output_file, desc=f"FineWeb {filename}")
        
        downloaded_files.append(output_file)
    
    # If multiple files, combine them into one parquet
    if num_files == 1:
        result_path = downloaded_files[0]
        print(f"\n✓ Downloaded to: {result_path}")
    else:
        print(f"\n=== Combining {num_files} files ===")
        dfs = []
        for file_path in tqdm(downloaded_files, desc="Reading files"):
            df = pl.read_parquet(file_path)
            dfs.append(df)
        
        combined_df = pl.concat(dfs)
        result_path = output_path / "fineweb_combined.parquet"
        combined_df.write_parquet(result_path)
        print(f"✓ Combined {len(combined_df)} rows into: {result_path}")
    
    # Show preview
    preview_df = pl.read_parquet(result_path, use_pyarrow=True)
    print(f"\nDataFrame preview ({len(preview_df)} rows, {len(preview_df.columns)} columns):")
    print(preview_df.head())
    print(f"\nColumns: {preview_df.columns}")
    
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
    
    print(f"\nConfiguration:", flush=True)
    print(f"  Source: {args.source}", flush=True)
    print(f"  Output directory: {args.output_dir}", flush=True)
    if args.source == "wikipedia":
        print(f"  Date: {args.date}", flush=True)
    else:
        print(f"  Number of files: {args.num_files}", flush=True)
        print(f"  Start index: {args.start_index}", flush=True)
    print(flush=True)
    
    # Download based on source
    try:
        if args.source == "wikipedia":
            output_file = download_wikipedia(args.output_dir, args.date)
        else:  # fineweb
            output_file = download_fineweb(args.output_dir, args.num_files, args.start_index)
        
        print(f"\n{'='*60}")
        print("✓ Download complete!")
        print(f"{'='*60}")
        print(f"\nOutput file: {output_file}")
        print("\nNext step: Process with prepare_corpus.py")
        print("Example:")
        print(f"  python bin/prepare_corpus.py \\")
        print(f"    --input {output_file} \\")
        print(f"    --output data/processed.parquet \\")
        print(f"    --max-grade 8")
        
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Download interrupted by user", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)
        import traceback
        print("\nFull traceback:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

