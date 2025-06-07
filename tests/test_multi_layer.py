#!/usr/bin/env python3
"""
Test multi-layer embeddings functionality
"""

import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
import pytest
import numpy as np

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify import EmbedderFactory
from text_vectorify.embedders.multi_layer import MultiLayerEmbedder


@pytest.mark.integration
@pytest.mark.multilayer
class TestMultiLayerEmbeddings(unittest.TestCase):
    """Test multi-layer embedding functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.test_texts = [
            "机器学习是人工智能的重要分支",
            "深度学习在图像识别中表现出色",
            "自然语言处理帮助理解文本含义",
            "数据科学结合统计学和计算机科学"
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass
    
    def test_tfidf_embedder_chinese(self):
        """Test TF-IDF embedder with Chinese text support"""
        embedder = EmbedderFactory.create_embedder(
            "TFIDFEmbedder",
            cache_dir=self.cache_dir,
            max_features=100,
            chinese_tokenizer='jieba',
            ngram_range=(1, 2)
        )
        
        # Test single text
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertTrue(len(vector) > 0)
        
        # Test multiple texts
        vectors = embedder.encode(self.test_texts)
        self.assertEqual(len(vectors), len(self.test_texts))
        self.assertTrue(all(isinstance(v, list) for v in vectors))
    
    def test_topic_embedder_lda(self):
        """Test Topic embedder with LDA"""
        embedder = EmbedderFactory.create_embedder(
            "TopicEmbedder",
            cache_dir=self.cache_dir,
            n_topics=3,
            method="lda",
            language="chinese"
        )
        
        # Test encoding
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 3)  # n_topics = 3
        
        # Test multiple texts
        vectors = embedder.encode(self.test_texts)
        self.assertEqual(len(vectors), len(self.test_texts))
        
        # Test topic information
        topics = embedder.get_topics()
        self.assertIsInstance(topics, list)
    
    @pytest.mark.skipif(True, reason="BERTopic optional dependency")
    def test_topic_embedder_bertopic(self):
        """Test Topic embedder with BERTopic"""
        try:
            import bertopic
            
            embedder = EmbedderFactory.create_embedder(
                "TopicEmbedder",
                cache_dir=self.cache_dir,
                n_topics=3,
                method="bertopic",
                language="chinese"
            )
            
            # Test encoding
            vector = embedder.encode(self.test_texts[0])
            self.assertIsInstance(vector, list)
            self.assertEqual(len(vector), 3)
            
        except ImportError:
            self.skipTest("BERTopic not available")
    
    def test_multi_layer_concat_fusion(self):
        """Test multi-layer embedder with concatenation fusion"""
        embedder_configs = [
            {
                'name': 'tfidf_layer',
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 50,
                    'chinese_tokenizer': 'jieba',
                    'ngram_range': (1, 2)
                }
            },
            {
                'name': 'topic_layer',
                'type': 'TopicEmbedder',
                'config': {
                    'n_topics': 5,
                    'method': 'lda',
                    'language': 'chinese'
                }
            }
        ]
        
        embedder = EmbedderFactory.create_embedder(
            "MultiLayerEmbedder",
            cache_dir=self.cache_dir,
            embedder_configs=embedder_configs,
            fusion_method="concat",
            normalize=True
        )
        
        # Test single text
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertTrue(len(vector) > 5)  # Should be sum of both layers
        
        # Test multiple texts
        vectors = embedder.encode(self.test_texts)
        self.assertEqual(len(vectors), len(self.test_texts))
        self.assertTrue(all(len(v) == len(vector) for v in vectors))
    
    def test_multi_layer_weighted_fusion(self):
        """Test multi-layer embedder with weighted fusion"""
        embedder_configs = [
            {
                'name': 'tfidf_layer',
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 30,
                    'chinese_tokenizer': 'jieba',
                    'ngram_range': (1, 1)
                }
            },
            {
                'name': 'topic_layer',
                'type': 'TopicEmbedder',
                'config': {
                    'n_topics': 3,
                    'method': 'lda',
                    'language': 'chinese'
                }
            }
        ]
        
        embedder = EmbedderFactory.create_embedder(
            "MultiLayerEmbedder",
            cache_dir=self.cache_dir,
            embedder_configs=embedder_configs,
            fusion_method="weighted",
            weights=[0.6, 0.4],
            normalize=True
        )
        
        # Test encoding
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertTrue(len(vector) > 0)
    
    def test_multi_layer_attention_fusion(self):
        """Test multi-layer embedder with attention fusion"""
        embedder_configs = [
            {
                'name': 'tfidf_layer',
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 20,
                    'chinese_tokenizer': 'jieba'
                }
            },
            {
                'name': 'topic_layer',
                'type': 'TopicEmbedder',
                'config': {
                    'n_topics': 3,
                    'method': 'lda',
                    'language': 'chinese'
                }
            }
        ]
        
        embedder = EmbedderFactory.create_embedder(
            "MultiLayerEmbedder",
            cache_dir=self.cache_dir,
            embedder_configs=embedder_configs,
            fusion_method="attention",
            normalize=True
        )
        
        # Test encoding
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertTrue(len(vector) > 0)
    
    def test_config_file_loading(self):
        """Test loading multi-layer embedder from config file"""
        config_data = {
            "embedder_type": "MultiLayerEmbedder",
            "config": {
                "embedder_configs": [
                    {
                        "name": "tfidf",
                        "type": "TFIDFEmbedder",
                        "config": {
                            "max_features": 20,
                            "ngram_range": (1, 2),
                            "chinese_tokenizer": "jieba"
                        }
                    },
                    {
                        "name": "topic",
                        "type": "TopicEmbedder",
                        "config": {
                            "n_topics": 3,
                            "method": "lda",
                            "language": "chinese"
                        }
                    }
                ],
                "fusion_method": "concat",
                "normalize": True
            }
        }
        
        # Create embedder from config
        embedder = EmbedderFactory.create_embedder(
            config_data['embedder_type'],
            cache_dir=self.cache_dir,
            **config_data['config']
        )
        
        # Test encoding
        vector = embedder.encode(self.test_texts[0])
        self.assertIsInstance(vector, list)
        self.assertTrue(len(vector) > 0)
    
    def test_caching_functionality(self):
        """Test that caching works correctly for multi-layer embedder"""
        embedder_configs = [
            {
                'name': 'tfidf_layer',
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 20,
                    'chinese_tokenizer': 'jieba'
                }
            }
        ]
        
        embedder = EmbedderFactory.create_embedder(
            "MultiLayerEmbedder",
            cache_dir=self.cache_dir,
            embedder_configs=embedder_configs,
            fusion_method="concat"
        )
        
        test_text = self.test_texts[0]
        
        # First encoding
        vector1 = embedder.encode(test_text)
        
        # Second encoding (should use cache)
        vector2 = embedder.encode(test_text)
        
        # Vectors should be identical
        self.assertEqual(vector1, vector2)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        embedder_configs = [
            {
                'name': 'tfidf_layer',
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 30,
                    'chinese_tokenizer': 'jieba'
                }
            },
            {
                'name': 'topic_layer',
                'type': 'TopicEmbedder',
                'config': {
                    'n_topics': 3,
                    'method': 'lda'
                }
            }
        ]
        
        embedder = EmbedderFactory.create_embedder(
            "MultiLayerEmbedder",
            cache_dir=self.cache_dir,
            embedder_configs=embedder_configs,
            fusion_method="concat"
        )
        
        # Test batch processing
        vectors = embedder.encode(self.test_texts)
        
        self.assertEqual(len(vectors), len(self.test_texts))
        self.assertTrue(all(isinstance(v, list) for v in vectors))
        
        # All vectors should have the same dimension
        vector_dims = [len(v) for v in vectors]
        self.assertTrue(all(dim == vector_dims[0] for dim in vector_dims))


if __name__ == '__main__':
    unittest.main()
