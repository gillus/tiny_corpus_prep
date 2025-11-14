#!/usr/bin/env python3
"""
Example: Using CEFR-based synonym mapping to simplify text.

This example demonstrates:
1. Creating a simple synonym mapping
2. Using the SynonymMapper with case preservation
3. Processing text with difficult words -> easy alternatives
"""
from tiny_corpus_prep.synonyms import SynonymMapper
from tiny_corpus_prep.common import preserve_case_like, tokenize_basic, detokenize_basic

def example_basic_mapping():
    """Example 1: Basic synonym mapping"""
    print("=" * 60)
    print("Example 1: Basic Synonym Mapping")
    print("=" * 60)
    
    # Create a mapping from difficult words to easier alternatives
    mapping = {
        "utilize": "use",
        "commence": "start",
        "terminate": "end",
        "physician": "doctor",
        "demonstrate": "show",
        "sufficient": "enough",
        "approximately": "about",
    }
    
    # Create mapper with case preservation
    mapper = SynonymMapper(mapping, preserve_case=True)
    
    # Test sentences
    sentences = [
        "The physician will demonstrate the procedure.",
        "We should UTILIZE all available resources.",
        "Commence the experiment when ready.",
        "Approximately five participants will suffice.",
    ]
    
    print("\nOriginal -> Simplified:")
    for sentence in sentences:
        simplified = mapper.simplify_line(sentence)
        print(f"  {sentence}")
        print(f"  → {simplified}")
        print()


def example_case_preservation():
    """Example 2: Case preservation in action"""
    print("=" * 60)
    print("Example 2: Case Preservation")
    print("=" * 60)
    
    mapping = {"utilize": "use"}
    mapper = SynonymMapper(mapping, preserve_case=True)
    
    test_cases = [
        "utilize the tool",       # lowercase -> lowercase
        "Utilize the tool",       # Titlecase -> Titlecase
        "UTILIZE THE TOOL",       # UPPERCASE -> UPPERCASE
    ]
    
    print("\nCase patterns are preserved:")
    for text in test_cases:
        result = mapper.simplify_line(text)
        print(f"  '{text}' -> '{result}'")


def example_tokenization():
    """Example 3: Tokenization utilities"""
    print("\n" + "=" * 60)
    print("Example 3: Tokenization and Detokenization")
    print("=" * 60)
    
    text = "Don't worry, it's fine! The doctor's here."
    
    print(f"\nOriginal: {text}")
    
    # Tokenize
    tokens = tokenize_basic(text)
    print(f"Tokens: {tokens}")
    
    # Detokenize
    reconstructed = detokenize_basic(tokens)
    print(f"Reconstructed: {reconstructed}")


def example_manual_case_handling():
    """Example 4: Manual case preservation"""
    print("\n" + "=" * 60)
    print("Example 4: Manual Case Preservation")
    print("=" * 60)
    
    examples = [
        ("DIFFICULT", "easy"),
        ("Difficult", "easy"),
        ("difficult", "easy"),
    ]
    
    print("\nManual case preservation:")
    for source, replacement in examples:
        result = preserve_case_like(source, replacement)
        print(f"  {source} + '{replacement}' = {result}")


def example_batch_processing():
    """Example 5: Batch text processing"""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)
    
    mapping = {
        "utilize": "use",
        "physician": "doctor",
        "demonstrate": "show",
        "commence": "start",
    }
    
    mapper = SynonymMapper(mapping, preserve_case=True)
    
    # Simulate processing multiple lines
    lines = [
        "The physician will demonstrate the technique.\n",
        "Utilize the equipment carefully.\n",
        "Commence when ready.\n",
        "All physicians must demonstrate competency.\n",
    ]
    
    print("\nProcessing batch of lines:")
    print("Original lines:")
    for line in lines:
        print(f"  {line.rstrip()}")
    
    print("\nSimplified lines:")
    for simplified_line in mapper.simplify_iter(lines):
        print(f"  {simplified_line.rstrip()}")


def example_real_world_text():
    """Example 6: Real-world text simplification"""
    print("\n" + "=" * 60)
    print("Example 6: Real-World Text Simplification")
    print("=" * 60)
    
    # Extended mapping
    mapping = {
        "utilize": "use",
        "physician": "doctor",
        "demonstrate": "show",
        "commence": "start",
        "terminate": "end",
        "approximately": "about",
        "sufficient": "enough",
        "numerous": "many",
        "assist": "help",
        "obtain": "get",
        "require": "need",
        "subsequently": "later",
    }
    
    mapper = SynonymMapper(mapping, preserve_case=True)
    
    # Sample academic text
    text = """
    The physician will demonstrate numerous techniques to assist patients.
    Utilize the equipment carefully and obtain sufficient measurements.
    Approximately ten participants will be required for the study.
    Commence the procedure and subsequently terminate after recording.
    """
    
    print("\nOriginal text:")
    print(text)
    
    print("\nSimplified text:")
    print(mapper.simplify_line(text))


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("CEFR Synonym Mapping Examples")
    print("=" * 60)
    print()
    
    example_basic_mapping()
    example_case_preservation()
    example_tokenization()
    example_manual_case_handling()
    example_batch_processing()
    example_real_world_text()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Build a synonym map from a CEFR wordlist:")
    print("   python bin/build_synmap_from_cefr.py --cefr_csv data.csv --out_dir output")
    print()
    print("2. Use the generated synonyms.json in your pipeline:")
    print("   mapper = SynonymMapper.from_json('output/synonyms/synonyms.json')")
    print()
    print("See CEFR_SYNONYMS.md for complete documentation.")
    print()


if __name__ == "__main__":
    main()

