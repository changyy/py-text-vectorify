#!/usr/bin/env python3
"""
Complete script for testing all embedding models
"""

import json
import subprocess
import sys
from pathlib import Path

def run_model_test(model_name, model_path, output_suffix):
    """Run single model test"""
    print(f"\n🧪 Testing {model_name} model...")
    
    output_file = f"examples/test_{output_suffix}_output.jsonl"
    
    cmd = [
        "text-vectorify",
        "--input", "examples/sample_input.jsonl",
        "--output", output_file,
        "--input-field-main", "title",
        "--input-field-subtitle", "content",
        "--process-method", model_name,
        "--process-model-name", model_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {model_name} test successful")
        
        # Check output file
        if Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                first_line = json.loads(f.readline())
                vector_dim = len(first_line['vector'])
                print(f"   Vector dimension: {vector_dim}")
                return True, vector_dim
        else:
            print(f"❌ Output file does not exist: {output_file}")
            return False, 0
            
    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} test failed:")
        print(f"   Error: {e.stderr}")
        return False, 0

def main():
    """Main test function"""
    print("🚀 Starting tests for all embedding models...")
    
    # Test configuration
    models_config = [
        ("SentenceBertEmbedder", "paraphrase-multilingual-MiniLM-L12-v2", "sbert"),
        ("BGEEmbedder", "BAAI/bge-small-en-v1.5", "bge"),
        ("M3EEmbedder", "moka-ai/m3e-base", "m3e"),
        ("HuggingFaceEmbedder", "distilbert-base-uncased", "hf"),
    ]
    
    results = []
    
    for model_name, model_path, output_suffix in models_config:
        success, dimensions = run_model_test(model_name, model_path, output_suffix)
        results.append({
            'model': model_name,
            'path': model_path,
            'success': success,
            'dimensions': dimensions
        })
    
    # Summary results
    print("\n📊 Test Results Summary:")
    print("=" * 60)
    print(f"{'Model Name':<20} {'Model Path':<35} {'Status':<8} {'Dims':<6}")
    print("-" * 60)
    
    successful_tests = 0
    for result in results:
        status = "✅ Success" if result['success'] else "❌ Failed"
        dims = str(result['dimensions']) if result['success'] else "N/A"
        print(f"{result['model']:<20} {result['path']:<35} {status:<8} {dims:<6}")
        if result['success']:
            successful_tests += 1
    
    print("-" * 60)
    print(f"Successful tests: {successful_tests}/{len(results)}")
    
    if successful_tests == len(results):
        print("\n🎉 All model tests passed! TextVectorify tool is ready.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - successful_tests} model tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
