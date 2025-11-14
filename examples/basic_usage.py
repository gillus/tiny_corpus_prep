"""
Basic usage examples for tiny_corpus_prep library.
"""
import polars as pl
from pathlib import Path

# Example 1: Create a sample dataset
def create_sample_data():
    """Create a sample parquet file for testing."""
    data = {
        "text": [
            "The cat sat on the mat. It was a sunny day.",
            "Quantum mechanics is a fundamental theory in physics.",
            "I love pizza and ice cream!",
            "The mitochondria is the powerhouse of the cell.",
            "Simple sentence here.",
            "Photosynthesis converts light energy into chemical energy.",
            "Hello world! This is a test.",
            "The Copy Number Variation analysis of DNA sequences helps characterize cells.",
        ]
    }
    df = pl.DataFrame(data)
    df.write_parquet("sample_data.parquet")
    print("Created sample_data.parquet")


# Example 2: Basic processing without annotation
def example_basic_processing():
    """Simple processing with filtering only."""
    from tiny_corpus_prep import process_corpus
    
    stats = process_corpus(
        input_path="sample_data.parquet",
        output_path="output_basic.parquet",
        max_grade=8.0,  # Middle school level
        normalize=True,
        dedup=True,
    )
    
    print(f"\nBasic processing complete!")
    print(f"Output rows: {stats['total_rows']}")


# Example 3: Processing with keyword filter
def example_keyword_filtering():
    """Filter by keywords."""
    from tiny_corpus_prep import process_corpus
    
    stats = process_corpus(
        input_path="sample_data.parquet",
        output_path="output_keywords.parquet",
        keywords=["cell", "DNA", "energy"],
        max_grade=None,  # No readability filter
    )
    
    print(f"\nKeyword filtering complete!")
    print(f"Output rows: {stats['total_rows']}")


# Example 4: Using DataPipeline with custom annotator
def example_custom_annotator():
    """Use custom annotator."""
    from tiny_corpus_prep import DataPipeline, CustomFunctionAnnotator, read_parquet, write_parquet_with_stats
    
    # Custom annotation function
    def my_annotator(text: str) -> dict:
        return {
            "word_count": len(text.split()),
            "char_count": len(text),
            "has_question": "?" in text,
        }
    
    # Read data
    df = read_parquet("sample_data.parquet")
    
    # Build and run pipeline
    pipeline = (
        DataPipeline(normalize=True, dedup=True)
        .add_readability_filter(max_grade=10.0)
        .add_annotator(CustomFunctionAnnotator(my_annotator))
    )
    
    df_processed = pipeline.process(df)
    
    # Save with stats
    stats = write_parquet_with_stats(df_processed, "output_annotated.parquet")
    
    print(f"\nCustom annotation complete!")
    print(f"Output rows: {stats['total_rows']}")
    print(f"Columns: {stats['columns']}")


# Example 5: Direct DataFrame operations
def example_direct_operations():
    """Direct filtering on DataFrame."""
    from tiny_corpus_prep import filter_by_readability, is_middle_school_level
    import polars as pl
    
    # Read data
    df = pl.read_parquet("sample_data.parquet")
    print(f"\nOriginal: {len(df)} rows")
    
    # Filter by readability
    df_filtered = filter_by_readability(df, max_grade=8.0)
    print(f"After readability filter: {len(df_filtered)} rows")
    
    # Check individual texts
    text = "This is a simple sentence."
    if is_middle_school_level(text):
        print(f"✓ Text is middle school level")


# Example 6: Synonym mapping
def example_synonym_mapping():
    """Use synonym mapping."""
    import json
    from tiny_corpus_prep import process_corpus
    
    # Create simple synonym map
    synonyms = {
        "automobile": "car",
        "physician": "doctor",
        "utilize": "use",
    }
    
    with open("synonyms.json", "w") as f:
        json.dump(synonyms, f)
    
    stats = process_corpus(
        input_path="sample_data.parquet",
        output_path="output_synonyms.parquet",
        synonyms_map_path="synonyms.json",
    )
    
    print(f"\nSynonym mapping complete!")


if __name__ == "__main__":
    print("Mini GPT Prep - Usage Examples")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Run examples
    print("\n1. Basic Processing")
    print("-" * 50)
    example_basic_processing()
    
    print("\n2. Keyword Filtering")
    print("-" * 50)
    example_keyword_filtering()
    
    print("\n3. Custom Annotator")
    print("-" * 50)
    example_custom_annotator()
    
    print("\n4. Direct Operations")
    print("-" * 50)
    example_direct_operations()
    
    print("\n5. Synonym Mapping")
    print("-" * 50)
    example_synonym_mapping()
    
    print("\n" + "=" * 50)
    print("All examples complete!")
    print("\nGenerated files:")
    print("  - sample_data.parquet (input)")
    print("  - output_basic.parquet + .json")
    print("  - output_keywords.parquet + .json")
    print("  - output_annotated.parquet + .json")
    print("  - output_synonyms.parquet + .json")

