"""
Example using Gemini API for text classification.

Requirements:
    uv pip install -e ".[annotators]"
    
Set your API key:
    export GOOGLE_API_KEY="your_key_here"
    
Or create a .env file with:
    GOOGLE_API_KEY=your_key_here
"""
import polars as pl
from tiny_corpus_prep import DataPipeline, GeminiAnnotator, write_parquet_with_stats


def create_science_data():
    """Create sample scientific texts."""
    texts = [
        "The cat played with the ball in the garden.",
        "Quantum entanglement is a physical phenomenon that occurs when pairs of particles interact.",
        "DNA replication is the biological process of producing two identical replicas of DNA.",
        "I went to the store to buy milk and cookies.",
        "The Krebs cycle is a series of chemical reactions used by aerobic organisms to generate energy.",
        "Machine learning algorithms can learn from and make predictions on data.",
        "The French Revolution was a period of radical political and societal change in France.",
    ]
    
    df = pl.DataFrame({"text": texts})
    df.write_parquet("science_texts.parquet")
    print(f"Created science_texts.parquet with {len(df)} texts")
    return df


def process_with_gemini():
    """Process texts with Gemini annotation."""
    try:
        # Initialize Gemini annotator
        gemini = GeminiAnnotator(
            model_name="gemini-2.5-flash-lite",
            # api_key="your_key"  # Optional: if not in environment
        )
        
        print("\n✓ Gemini annotator initialized")
        
        # Create pipeline
        pipeline = (
            DataPipeline(normalize=False, dedup=True)
            .add_readability_filter(max_grade=15.0)  # Allow complex texts
            .add_annotator(gemini)
        )
        
        # Read and process
        df = pl.read_parquet("science_texts.parquet")
        print(f"\nProcessing {len(df)} texts with Gemini...")
        
        df_processed = pipeline.process(df)
        
        # Save results
        stats = write_parquet_with_stats(df_processed, "output_gemini.parquet")
        
        print(f"\n✓ Processing complete!")
        print(f"  Rows: {stats['total_rows']}")
        print(f"  Columns: {stats['columns']}")
        
        # Show sample results
        print("\nSample results:")
        print(df_processed.head())
        
        # Show topic distribution
        if "topic" in df_processed.columns:
            print("\nTopic distribution:")
            topic_counts = df_processed["topic"].value_counts().sort("counts", descending=True)
            print(topic_counts)
        
        # Show education distribution
        if "education" in df_processed.columns:
            print("\nEducation level distribution:")
            edu_counts = df_processed["education"].value_counts().sort("counts", descending=True)
            print(edu_counts)
        
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease install annotation dependencies:")
        print("  uv pip install -e '.[annotators]'")
    
    except ValueError as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease set your Google API key:")
        print("  export GOOGLE_API_KEY='your_key_here'")
        print("Or create a .env file with GOOGLE_API_KEY=your_key_here")


if __name__ == "__main__":
    print("Gemini Annotation Example")
    print("=" * 60)
    
    # Create sample data
    create_science_data()
    
    # Process with Gemini
    process_with_gemini()
    
    print("\n" + "=" * 60)
    print("Example complete!")

