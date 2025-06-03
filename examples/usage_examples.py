#!/usr/bin/env python3
"""
Text Vectorify Usage Examples

This file demonstrates how to use various features of the text-vectorify tool.
"""

import os
import sys
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify import TextVectorify, EmbedderFactory


def example_1_sentence_bert():
    """Example 1: Using SentenceBERT multilingual model"""
    print("=" * 60)
    print("Example 1: SentenceBERT Multilingual Model")
    print("=" * 60)
    
    try:
        # Create SentenceBERT embedder
        embedder = EmbedderFactory.create_embedder(
            "SentenceBertEmbedder",
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="./cache"
        )
        
        # Create vectorizer
        vectorizer = TextVectorify(embedder)
        
        # Process sample file
        input_file = Path(__file__).parent / "sample_input.jsonl"
        output_file = Path(__file__).parent / "output_sbert.jsonl"
        
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        vectorizer.process_jsonl(
            input_path=str(input_file),
            output_path=str(output_file),
            input_field_main=["title"],
            input_field_subtitle=["content"],
            output_field="sbert_vector"
        )
        
        print("✅ SentenceBERT processing complete!")
        
    except Exception as e:
        print(f"❌ SentenceBERT example failed: {e}")
        print("Tip: Please install sentence-transformers: pip install sentence-transformers")


def example_2_openai():
    """Example 2: Using OpenAI API (requires API Key)"""
    print("\n" + "=" * 60)
    print("Example 2: OpenAI Embedding Model")
    print("=" * 60)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("Example: export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        # Create OpenAI embedder
        embedder = EmbedderFactory.create_embedder(
            "OpenAIEmbedder",
            "text-embedding-3-small",
            cache_dir="./cache",
            api_key=api_key
        )
        
        # Create vectorizer
        vectorizer = TextVectorify(embedder)
        
        # Process sample file
        input_file = Path(__file__).parent / "sample_input.jsonl"
        output_file = Path(__file__).parent / "output_openai.jsonl"
        
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        vectorizer.process_jsonl(
            input_path=str(input_file),
            output_path=str(output_file),
            input_field_main=["title"],
            input_field_subtitle=["content"],
            output_field="openai_vector"
        )
        
        print("✅ OpenAI processing completed!")
        
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}")
        print("Hint: Please install openai: pip install openai")


def example_3_bge():
    """Example 3: Using BGE Chinese model"""
    print("\n" + "=" * 60)
    print("Example 3: BGE Chinese-specific model")
    print("=" * 60)
    
    try:
        # Create BGE embedder
        embedder = EmbedderFactory.create_embedder(
            "BGEEmbedder",
            "BAAI/bge-small-zh-v1.5",
            cache_dir="./cache"
        )
        
        # Create vectorizer
        vectorizer = TextVectorify(embedder)
        
        # Process example file
        input_file = Path(__file__).parent / "sample_input.jsonl"
        output_file = Path(__file__).parent / "output_bge.jsonl"
        
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        vectorizer.process_jsonl(
            input_path=str(input_file),
            output_path=str(output_file),
            input_field_main=["title"],
            input_field_subtitle=["content"],
            output_field="bge_vector"
        )
        
        print("✅ BGE processing completed!")
        
    except Exception as e:
        print(f"❌ BGE example failed: {e}")
        print("Hint: Please install sentence-transformers: pip install sentence-transformers")


def example_4_multiple_fields():
    """Example 4: Multiple fields combination example"""
    print("\n" + "=" * 60)
    print("Example 4: Multiple fields combination processing")
    print("=" * 60)
    
    try:
        # Create SentenceBERT embedder (lighter weight)
        embedder = EmbedderFactory.create_embedder(
            "SentenceBertEmbedder",
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="./cache"
        )
        
        # Create vectorizer
        vectorizer = TextVectorify(embedder)
        
        # Process example file, combining multiple fields
        input_file = Path(__file__).parent / "sample_input.jsonl"
        output_file = Path(__file__).parent / "output_multi_fields.jsonl"
        
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        print("Combined fields: title + content + author + category")
        
        vectorizer.process_jsonl(
            input_path=str(input_file),
            output_path=str(output_file),
            input_field_main=["title", "author"],  # Main fields
            input_field_subtitle=["content", "category"],  # Subtitle fields
            output_field="combined_vector"
        )
        
        print("✅ Multiple fields combination processing completed!")
        
    except Exception as e:
        print(f"❌ Multiple fields example failed: {e}")


def example_5_command_line_usage():
    """Example 5: Command line usage demonstration (Updated CLI)"""
    print("\n" + "=" * 60)
    print("Example 5: Modern CLI usage (Unified Interface)")
    print("=" * 60)
    
    print("🚀 Using the unified CLI interface:")
    print()
    
    print("1. Quick BGE testing with cache:")
    print("python -m text_vectorify.main \\")
    print("  --input examples/sample_input.jsonl \\")
    print("  --input-field-main title \\")
    print("  --input-field-subtitle content \\")
    print("  --process-method BGEEmbedder \\")
    print("  --cache-dir ./quick_test_cache")
    print()
    
    print("2. OpenAI with custom cache location:")
    print("python -m text_vectorify.main \\")
    print("  --input examples/sample_input.jsonl \\")
    print("  --input-field-main title \\")
    print("  --input-field-subtitle content \\")
    print("  --process-method OpenAIEmbedder \\")
    print("  --model-name text-embedding-3-small \\")
    print("  --extra-data $OPENAI_API_KEY \\")
    print("  --cache-dir ./openai_cache")
    print()
    
    print("3. SentenceBERT multilingual:")
    print("python -m text_vectorify.main \\")
    print("  --input examples/sample_input.jsonl \\")
    print("  --input-field-main title \\")
    print("  --input-field-subtitle content \\")
    print("  --process-method SentenceBertEmbedder \\")
    print("  --model-name paraphrase-multilingual-MiniLM-L12-v2 \\")
    print("  --cache-dir ./sbert_cache")
    print()
    
    print("4. 🛠️  Utility commands:")
    print("# Demo with sample data")
    print("python -m text_vectorify.main --demo")
    print()
    print("# Check cache statistics")
    print("python -m text_vectorify.main --show-cache-stats")
    print()
    print("# List cache files")
    print("python -m text_vectorify.main --list-cache-files")
    print()
    print("# Clear all caches")
    print("python -m text_vectorify.main --clear-all-caches")
    
    print("\n💡 CLI-First Development Benefits:")
    print("  ✅ Quick experimentation with small datasets")
    print("  ✅ Cache building for library reuse")
    print("  ✅ Easy parameter tuning")
    print("  ✅ Immediate result verification")


def list_available_embedders():
    """List all available embedders"""
    print("\n" + "=" * 60)
    print("Available Embedders")
    print("=" * 60)
    
    embedders = EmbedderFactory.list_embedders()
    for i, embedder in enumerate(embedders, 1):
        print(f"{i}. {embedder}")
    
    print("\nModel recommendations:")
    print("- OpenAIEmbedder: Commercial API, best quality, requires API Key")
    print("- SentenceBertEmbedder: Open source, multilingual support, lightweight")
    print("- BGEEmbedder: Open source, Chinese optimized, excellent performance")
    print("- M3EEmbedder: Open source, Chinese-specific, good performance")
    print("- HuggingFaceEmbedder: Open source, can use any HF model")


def example_6_cli_library_workflow():
    """Example 6: CLI-first → Library integration workflow"""
    print("\n" + "=" * 60)
    print("Example 6: CLI-first → Library Integration Workflow")
    print("=" * 60)
    
    print("🔄 Recommended development workflow:")
    print()
    
    print("STEP 1: 🔬 Start with CLI for quick testing")
    print("-" * 50)
    print("# Create small test dataset and run CLI")
    print("python -m text_vectorify.main \\")
    print("  --input examples/sample_input.jsonl \\")
    print("  --input-field-main title \\")
    print("  --process-method BGEEmbedder \\")
    print("  --cache-dir ./workflow_cache \\")
    print("  --output ./test_output.jsonl")
    print()
    
    print("STEP 2: 🔍 Verify results and iterate")
    print("-" * 50)
    print("# Check cache stats")
    print("python -m text_vectorify.main --show-cache-stats")
    print("# Inspect output file")
    print("head -3 test_output.jsonl")
    print()
    
    print("STEP 3: 🔗 Switch to library with cache reuse")
    print("-" * 50)
    
    # Demonstrate library integration
    try:
        # Sample data for demonstration
        sample_data = [
            {"title": "Test Document", "content": "Sample content for testing"},
            {"title": "Another Doc", "content": "More test content"}
        ]
        
        print("# Python library code:")
        print("""
from text_vectorify import EmbedderFactory

# Reuse cache from CLI experiments
embedder = EmbedderFactory.create_embedder(
    "BGEEmbedder",
    cache_dir="./workflow_cache"  # Same as CLI!
)

# Process data in memory (no files needed)
for item in your_data:
    text = f"{item['title']} {item['content']}"
    vector = embedder.encode(text)  # Uses cache - no recomputation!
    item['embedding'] = vector
        """)
        
        print("\n💡 Key Benefits:")
        print("  ✅ CLI: Fast iteration, easy debugging")
        print("  ✅ Cache: No duplicate computations")
        print("  ✅ Library: Seamless integration")
        print("  ✅ Workflow: Develop → Test → Scale")
        
    except Exception as e:
        print(f"Demo note: {e}")


def main():
    """Main program"""
    print("🚀 Text Vectorify Usage Examples")
    print("This program demonstrates various usage methods of text-vectorify")
    
    # List available embedders
    list_available_embedders()
    
    # Run examples (in order, from lightweight to heavyweight)
    example_1_sentence_bert()
    example_4_multiple_fields()
    example_3_bge()
    example_2_openai()  # Requires API Key
    example_5_command_line_usage()
    example_6_cli_library_workflow()  # New workflow example
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("Please check the output files in the examples/ directory")
    print("\n💡 Next steps:")
    print("  1. Try: python examples/development_workflow_guide.py")
    print("  2. For CLI-first workflow demonstration")
    print("=" * 60)


if __name__ == "__main__":
    main()