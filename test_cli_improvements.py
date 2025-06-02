#!/usr/bin/env python3
"""
Test script for CLI improvements:
1. Test default model names
2. Test stdin input support
"""

import subprocess
import json
import tempfile
import os
from pathlib import Path

# Test data
test_data = [
    {"title": "First Article", "content": "This is the first article content"},
    {"title": "Second Article", "content": "This is the second article content"}
]

def test_default_models():
    """Test using default models without specifying --process-model-name"""
    print("🧪 Testing default model usage...")
    
    # Create test input file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for item in test_data:
            f.write(json.dumps(item) + '\n')
        input_file = f.name
    
    try:
        # Test with BGE embedder (should use default model)
        cmd = [
            'python', '-m', 'text_vectorify.main',
            '--input', input_file,
            '--input-field-main', 'title',
            '--input-field-subtitle', 'content',
            '--process-method', 'BGEEmbedder',
            '--verbose'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Volumes/Data/UserData/py-text-vectorify')
        
        if result.returncode == 0:
            print("✅ Default model test passed!")
            print("STDOUT:", result.stdout)
        else:
            print("❌ Default model test failed!")
            print("STDERR:", result.stderr)
            
    finally:
        os.unlink(input_file)

def test_stdin_input():
    """Test reading from stdin"""
    print("\n🧪 Testing stdin input...")
    
    # Prepare input data
    input_data = '\n'.join(json.dumps(item) for item in test_data)
    
    try:
        # Test stdin input
        cmd = [
            'python', '-m', 'text_vectorify.main',
            '--input-field-main', 'title',
            '--process-method', 'BGEEmbedder',
            '--verbose'
        ]
        
        print(f"Running: echo '{input_data}' | {' '.join(cmd)}")
        
        # Use shell=True to enable pipe
        full_cmd = f"echo '{input_data}' | python -m text_vectorify.main --input-field-main 'title' --process-method 'BGEEmbedder' --verbose"
        
        result = subprocess.run(full_cmd, capture_output=True, text=True, shell=True, cwd='/Volumes/Data/UserData/py-text-vectorify')
        
        if result.returncode == 0:
            print("✅ Stdin input test passed!")
            print("STDOUT:", result.stdout)
        else:
            print("❌ Stdin input test failed!")
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error during stdin test: {e}")

def test_explicit_stdin():
    """Test explicit stdin marker"""
    print("\n🧪 Testing explicit stdin marker...")
    
    # Prepare input data
    input_data = '\n'.join(json.dumps(item) for item in test_data)
    
    try:
        # Test explicit stdin marker
        full_cmd = f"echo '{input_data}' | python -m text_vectorify.main --input - --input-field-main 'title' --process-method 'BGEEmbedder' --verbose"
        
        print(f"Running: {full_cmd}")
        result = subprocess.run(full_cmd, capture_output=True, text=True, shell=True, cwd='/Volumes/Data/UserData/py-text-vectorify')
        
        if result.returncode == 0:
            print("✅ Explicit stdin test passed!")
            print("STDOUT:", result.stdout)
        else:
            print("❌ Explicit stdin test failed!")
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error during explicit stdin test: {e}")

def test_help_output():
    """Test that help shows updated information"""
    print("\n🧪 Testing help output...")
    
    cmd = ['python', '-m', 'text_vectorify.main', '--help']
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/Volumes/Data/UserData/py-text-vectorify')
    
    help_text = result.stdout
    
    # Check for expected improvements in help text
    checks = [
        "default models" in help_text.lower(),
        "stdin" in help_text.lower(),
        "optional" in help_text.lower(),
        "text-embedding-3-small" in help_text
    ]
    
    if all(checks):
        print("✅ Help text includes expected improvements!")
    else:
        print("❌ Help text missing expected improvements")
        print("Help output:", help_text)

if __name__ == "__main__":
    print("🚀 Testing CLI improvements...")
    print("=" * 50)
    
    test_help_output()
    test_default_models()
    test_stdin_input()
    test_explicit_stdin()
    
    print("\n" + "=" * 50)
    print("🏁 Testing complete!")
