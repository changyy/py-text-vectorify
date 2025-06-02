#!/usr/bin/env python3
"""
Simple test script to verify basic functionality of various embedders
"""

import tempfile
import os
from text_vectorify import EmbedderFactory

def test_sentence_bert():
    """Test SentenceBERT embedder"""
    print("Testing SentenceBERT embedder...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            embedder = EmbedderFactory.create_embedder(
                "SentenceBertEmbedder",
                "paraphrase-multilingual-MiniLM-L12-v2",
                cache_dir=temp_dir
            )
            
            test_text = "This is a test text"
            vector = embedder.encode(test_text)
            
            print(f"✅ SentenceBERT encoding successful, vector dimension: {len(vector)}")
            print(f"   First 5 values: {vector[:5]}")
            
    except Exception as e:
        print(f"❌ SentenceBERT test failed: {e}")

def test_factory():
    """Test factory pattern"""
    print("\nTesting embedder factory...")
    
    try:
        embedders = EmbedderFactory.list_embedders()
        print(f"✅ Available embedders: {', '.join(embedders)}")
        
    except Exception as e:
        print(f"❌ Factory test failed: {e}")

if __name__ == "__main__":
    test_factory()
    test_sentence_bert()
    print("\nTesting completed!")
