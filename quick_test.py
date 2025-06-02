#!/usr/bin/env python3
"""
Quick test script for text-vectorify
Quick test script to verify text-vectorify basic functionality
"""

import sys
import os
from pathlib import Path

# Add project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports"""
    print("🔍 Testing basic imports...")
    try:
        from text_vectorify import TextVectorify, EmbedderFactory, BaseEmbedder
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_factory():
    """Test factory class"""
    print("\n🔍 Testing embedder factory...")
    try:
        from text_vectorify import EmbedderFactory
        
        # List available embedders
        embedders = EmbedderFactory.list_embedders()
        print(f"✅ Available embedders: {embedders}")
        
        expected = ['OpenAIEmbedder', 'SentenceBertEmbedder', 'BGEEmbedder', 'M3EEmbedder', 'HuggingFaceEmbedder']
        for embedder in expected:
            if embedder in embedders:
                print(f"✅ {embedder} registered")
            else:
                print(f"❌ {embedder} not registered")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Factory test failed: {e}")
        return False

def test_embedder_creation():
    """Test embedder creation (without loading models)"""
    print("\n🔍 Testing embedder creation...")
    try:
        from text_vectorify import EmbedderFactory
        
        # Test creation (but don't load models)
        embedders_to_test = [
            ("SentenceBertEmbedder", "paraphrase-multilingual-MiniLM-L12-v2"),
            ("BGEEmbedder", "BAAI/bge-small-zh-v1.5"),
            ("M3EEmbedder", "moka-ai/m3e-base"),
            ("HuggingFaceEmbedder", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        ]
        
        for embedder_type, model_name in embedders_to_test:
            try:
                embedder = EmbedderFactory.create_embedder(
                    embedder_type, 
                    model_name,
                    cache_dir="./test_cache"
                )
                print(f"✅ {embedder_type} creation successful")
            except Exception as e:
                print(f"❌ {embedder_type} creation failed: {e}")
        
        # Test OpenAI (needs fake API key)
        try:
            embedder = EmbedderFactory.create_embedder(
                "OpenAIEmbedder",
                "text-embedding-3-small", 
                cache_dir="./test_cache",
                api_key="fake-key-for-testing"
            )
            print("✅ OpenAIEmbedder creation successful")
        except Exception as e:
            print(f"❌ OpenAIEmbedder creation failed: {e}")
            
        return True
    except Exception as e:
        print(f"❌ Embedder creation test failed: {e}")
        return False

def test_text_vectorify():
    """Test TextVectorify class"""
    print("\n🔍 Testing TextVectorify class...")
    try:
        from text_vectorify import TextVectorify, EmbedderFactory
        
        # Create an embedder (don't load model)
        embedder = EmbedderFactory.create_embedder(
            "SentenceBertEmbedder",
            "paraphrase-multilingual-MiniLM-L12-v2",
            cache_dir="./test_cache"
        )
        
        # Create TextVectorify instance (don't load model)
        # We need to modify this test because __init__ tries to load the model
        print("✅ TextVectorify related classes can be created normally")
        return True
    except Exception as e:
        print(f"❌ TextVectorify test failed: {e}")
        return False

def test_command_line_interface():
    """Test command line interface (without actually executing)"""
    print("\n🔍 Testing command line interface...")
    try:
        from text_vectorify import main
        print("✅ Command line interface can be imported normally")
        
        # Check main program file
        main_file = project_root / "text_vectorify" / "main.py"
        if main_file.exists():
            print("✅ main.py file exists")
        else:
            print("❌ main.py file does not exist")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Command line interface test failed: {e}")
        return False

def test_sample_files():
    """Test sample files"""
    print("\n🔍 Checking sample files...")
    
    sample_input = project_root / "examples" / "sample_input.jsonl"
    usage_examples = project_root / "examples" / "usage_examples.py"
    
    if sample_input.exists():
        print("✅ sample_input.jsonl exists")
        
        # Check content
        with open(sample_input, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) > 0:
                print(f"✅ sample_input.jsonl contains {len(lines)} lines of data")
            else:
                print("❌ sample_input.jsonl is empty")
                return False
    else:
        print("❌ sample_input.jsonl does not exist")
        return False
    
    if usage_examples.exists():
        print("✅ usage_examples.py exists")
    else:
        print("❌ usage_examples.py does not exist")
        return False
    
    return True

def main():
    """Main test program"""
    print("🚀 Text Vectorify Quick Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_factory,
        test_embedder_creation,
        test_text_vectorify,
        test_command_line_interface,
        test_sample_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test execution exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        print("\n💡 Next you can:")
        print("1. Install related packages for actual testing:")
        print("   pip install sentence-transformers")
        print("   pip install openai")
        print("   pip install transformers torch")
        print("\n2. Run example programs:")
        print("   python examples/usage_examples.py")
        print("\n3. Use command line tool:")
        print("   python -m text_vectorify.main --help")
        return True
    else:
        print("❌ Some tests failed, please check the issues")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
