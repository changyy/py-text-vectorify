#!/usr/bin/env python3
"""
Test all configuration files in configs/ directory
"""

import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
import pytest

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify import EmbedderFactory
from text_vectorify.embedders import TFIDFEmbedder, MultiLayerEmbedder


@pytest.mark.integration
@pytest.mark.config
class TestConfigFiles(unittest.TestCase):
    """Test all configuration files in configs/ directory"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, "cache")
        self.configs_dir = project_root / "configs"
        
        # Test data for Chinese text processing
        self.chinese_test_texts = [
            "這是第一個測試文本，包含機器學習的內容。",
            "人工智能技術在自然語言處理領域發展迅速。",
            "深度學習模型在文本分析中表現出色。",
            "中文分詞是自然語言處理的基礎任務。",
            "向量化技術幫助計算機理解文本語義。"
        ]
        
        # Test data for English text
        self.english_test_texts = [
            "This is a test document about machine learning.",
            "Natural language processing is advancing rapidly.",
            "Deep learning models excel at text analysis.",
            "Tokenization is fundamental for NLP tasks.",
            "Vectorization helps computers understand semantics."
        ]
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_files_exist(self):
        """Test that expected config files exist"""
        expected_configs = [
            "tfidf_spacy_example.json",
            "tfidf_jieba_example.json", 
            "tfidf_pkuseg_example.json",
            "multi_layer_simple.json",
            "multi_layer_spacy_recommended.json",
            "multi_layer_jieba_lightweight.json",
            "multi_layer_advanced_research.json"
        ]
        
        for config_file in expected_configs:
            config_path = self.configs_dir / config_file
            self.assertTrue(config_path.exists(), f"Config file {config_file} does not exist")
    
    def test_config_files_valid_json(self):
        """Test that all JSON config files are valid"""
        json_files = list(self.configs_dir.glob("*.json"))
        self.assertGreater(len(json_files), 0, "No JSON config files found")
        
        for config_file in json_files:
            with self.subTest(config_file=config_file.name):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    self.assertIsInstance(config, dict, f"Config {config_file.name} is not a valid dictionary")
                except json.JSONDecodeError as e:
                    self.fail(f"Config file {config_file.name} contains invalid JSON: {e}")
    
    def test_tfidf_spacy_config(self):
        """Test TF-IDF with spaCy configuration"""
        config_path = self.configs_dir / "tfidf_spacy_example.json"
        self.assertTrue(config_path.exists(), "tfidf_spacy_example.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'TFIDFEmbedder')
        self.assertIn('config', config)
        self.assertEqual(config['config']['chinese_tokenizer'], 'spacy')
        
        # Test functionality
        try:
            embedder = TFIDFEmbedder(**config['config'])
            self.assertEqual(embedder.chinese_tokenizer, 'spacy')
            
            # Test with Chinese text
            embedder.fit(self.chinese_test_texts)
            vector = embedder.encode(self.chinese_test_texts[0])
            
            self.assertIsInstance(vector, list)
            self.assertGreater(len(vector), 0)
            self.assertTrue(all(isinstance(x, float) for x in vector))
            
        except Exception as e:
            self.fail(f"TF-IDF spaCy config test failed: {e}")
    
    def test_tfidf_jieba_config(self):
        """Test TF-IDF with jieba configuration"""
        config_path = self.configs_dir / "tfidf_jieba_example.json"
        self.assertTrue(config_path.exists(), "tfidf_jieba_example.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'TFIDFEmbedder')
        self.assertEqual(config['config']['chinese_tokenizer'], 'jieba')
        
        # Test functionality
        try:
            embedder = TFIDFEmbedder(**config['config'])
            self.assertEqual(embedder.chinese_tokenizer, 'jieba')
            
            # Test with Chinese text
            embedder.fit(self.chinese_test_texts)
            vector = embedder.encode(self.chinese_test_texts[0])
            
            self.assertIsInstance(vector, list)
            self.assertGreater(len(vector), 0)
            
        except Exception as e:
            self.fail(f"TF-IDF jieba config test failed: {e}")
    
    def test_tfidf_pkuseg_config(self):
        """Test TF-IDF with pkuseg configuration (may skip if not available)"""
        config_path = self.configs_dir / "tfidf_pkuseg_example.json"
        self.assertTrue(config_path.exists(), "tfidf_pkuseg_example.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'TFIDFEmbedder')
        self.assertEqual(config['config']['chinese_tokenizer'], 'pkuseg')
        
        # Test functionality (may fallback to other tokenizers)
        try:
            embedder = TFIDFEmbedder(**config['config'])
            # Note: pkuseg might not be available, so it may fallback
            
            embedder.fit(self.chinese_test_texts)
            vector = embedder.encode(self.chinese_test_texts[0])
            
            self.assertIsInstance(vector, list)
            self.assertGreater(len(vector), 0)
            
        except Exception as e:
            self.fail(f"TF-IDF pkuseg config test failed: {e}")
    
    def test_multi_layer_simple_config(self):
        """Test simple multi-layer configuration"""
        config_path = self.configs_dir / "multi_layer_simple.json"
        self.assertTrue(config_path.exists(), "multi_layer_simple.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'MultiLayerEmbedder')
        self.assertIn('embedder_configs', config['config'])
        
        # Check TF-IDF layer uses spaCy
        tfidf_layer = next(
            (layer for layer in config['config']['embedder_configs'] 
             if layer['type'] == 'TFIDFEmbedder'), None
        )
        self.assertIsNotNone(tfidf_layer, "TFIDFEmbedder layer not found in config")
        self.assertEqual(tfidf_layer['config']['chinese_tokenizer'], 'spacy')
        
        # Test basic config validation (without full execution for speed)
        self.assertIn('fusion_method', config['config'])
        self.assertIn('normalize', config['config'])
    
    def test_multi_layer_spacy_recommended_config(self):
        """Test spaCy recommended multi-layer configuration"""
        config_path = self.configs_dir / "multi_layer_spacy_recommended.json"
        self.assertTrue(config_path.exists(), "multi_layer_spacy_recommended.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'MultiLayerEmbedder')
        
        # Check TF-IDF layer uses spaCy
        tfidf_layer = next(
            (layer for layer in config['config']['embedder_configs'] 
             if layer['type'] == 'TFIDFEmbedder'), None
        )
        if tfidf_layer:  # TF-IDF layer is optional in some configs
            self.assertEqual(tfidf_layer['config']['chinese_tokenizer'], 'spacy')
    
    def test_multi_layer_jieba_lightweight_config(self):
        """Test jieba lightweight multi-layer configuration"""
        config_path = self.configs_dir / "multi_layer_jieba_lightweight.json"
        self.assertTrue(config_path.exists(), "multi_layer_jieba_lightweight.json not found")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Validate config structure
        self.assertEqual(config['embedder_type'], 'MultiLayerEmbedder')
        
        # Check TF-IDF layer uses jieba
        tfidf_layer = next(
            (layer for layer in config['config']['embedder_configs'] 
             if layer['type'] == 'TFIDFEmbedder'), None
        )
        if tfidf_layer:
            self.assertEqual(tfidf_layer['config']['chinese_tokenizer'], 'jieba')
    
    def test_config_parameter_types(self):
        """Test that config parameters have correct types"""
        json_files = list(self.configs_dir.glob("*.json"))
        
        for config_file in json_files:
            with self.subTest(config_file=config_file.name):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if config.get('embedder_type') == 'TFIDFEmbedder':
                    config_data = config['config']
                    
                    # Check ngram_range is a list (will be converted to tuple)
                    if 'ngram_range' in config_data:
                        self.assertIsInstance(config_data['ngram_range'], list)
                        self.assertEqual(len(config_data['ngram_range']), 2)
                    
                    # Check max_features is integer
                    if 'max_features' in config_data:
                        self.assertIsInstance(config_data['max_features'], int)
                        self.assertGreater(config_data['max_features'], 0)
                    
                    # Check chinese_tokenizer is valid
                    if 'chinese_tokenizer' in config_data:
                        valid_tokenizers = ['spacy', 'jieba', 'pkuseg']
                        self.assertIn(config_data['chinese_tokenizer'], valid_tokenizers)
    
    def test_config_consistency_with_defaults(self):
        """Test that config files are consistent with current defaults"""
        # Check that new configs use spaCy as default
        spacy_configs = [
            "tfidf_spacy_example.json",
            "multi_layer_simple.json", 
            "multi_layer_spacy_recommended.json",
            "multi_layer_advanced_research.json"
        ]
        
        for config_name in spacy_configs:
            config_path = self.configs_dir / config_name
            if config_path.exists():
                with self.subTest(config_file=config_name):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if config.get('embedder_type') == 'TFIDFEmbedder':
                        tokenizer = config['config'].get('chinese_tokenizer')
                        self.assertEqual(tokenizer, 'spacy', 
                                       f"Config {config_name} should use spaCy as default")
                    
                    elif config.get('embedder_type') == 'MultiLayerEmbedder':
                        # Check TF-IDF layers in multi-layer configs
                        for layer in config['config'].get('embedder_configs', []):
                            if layer['type'] == 'TFIDFEmbedder':
                                tokenizer = layer['config'].get('chinese_tokenizer')
                                if config_name in ["multi_layer_simple.json", 
                                                 "multi_layer_spacy_recommended.json",
                                                 "multi_layer_advanced_research.json"]:
                                    self.assertEqual(tokenizer, 'spacy',
                                                   f"TF-IDF layer in {config_name} should use spaCy")
    
    @pytest.mark.slow
    def test_config_functional_validation(self):
        """Functional test of key config files (slow test)"""
        # Test a subset of configs with actual processing
        test_configs = [
            "tfidf_spacy_example.json",
            "tfidf_jieba_example.json"
        ]
        
        for config_name in test_configs:
            config_path = self.configs_dir / config_name
            if config_path.exists():
                with self.subTest(config_file=config_name):
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        if config['embedder_type'] == 'TFIDFEmbedder':
                            embedder = TFIDFEmbedder(**config['config'])
                            embedder.fit(self.chinese_test_texts[:3])  # Use fewer texts for speed
                            vector = embedder.encode(self.chinese_test_texts[0])
                            
                            self.assertIsInstance(vector, list)
                            self.assertGreater(len(vector), 0)
                            self.assertGreater(embedder.get_vocabulary_size(), 0)
                            
                    except Exception as e:
                        self.fail(f"Functional test failed for {config_name}: {e}")
    
    def test_config_readme_exists(self):
        """Test that configuration documentation exists"""
        readme_path = self.configs_dir / "README.md"
        self.assertTrue(readme_path.exists(), "configs/README.md documentation not found")
        
        # Check that README contains key sections
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        expected_sections = [
            "配置文件說明",
            "中文分詞器選擇指南", 
            "使用方法",
            "性能對比"
        ]
        
        for section in expected_sections:
            self.assertIn(section, readme_content, 
                         f"README.md missing section: {section}")
    
    def test_legacy_config_validation(self):
        """Test validation of legacy configuration files"""
        legacy_configs = ["multi_layer_example.json", "multi_layer_1500_articles.json"]
        
        for config_name in legacy_configs:
            config_path = self.configs_dir / config_name
            if config_path.exists():
                with self.subTest(config_file=config_name):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # Legacy configs use "layers" format instead of "embedder_type"
                    if "layers" in config:
                        # Check basic structure of legacy format
                        self.assertIn('layers', config)
                        self.assertIsInstance(config['layers'], list)
                        self.assertGreater(len(config['layers']), 0)
                        
                        # Check each layer has required fields
                        for layer in config['layers']:
                            self.assertIn('type', layer)
                            self.assertIn('config', layer)
                            # Legacy configs may have 'weight' field
                            
                    else:
                        # Standard format
                        self.assertEqual(config['embedder_type'], 'MultiLayerEmbedder')
                        self.assertIn('embedder_configs', config['config'])
    
    def test_all_configs_tokenizer_consistency(self):
        """Test that all config files have consistent tokenizer settings"""
        json_files = list(self.configs_dir.glob("*.json"))
        tokenizer_inconsistencies = []
        
        for config_file in json_files:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            if config.get('embedder_type') == 'TFIDFEmbedder':
                tokenizer = config['config'].get('chinese_tokenizer')
                if tokenizer not in ['spacy', 'jieba', 'pkuseg']:
                    tokenizer_inconsistencies.append(f"{config_file.name}: invalid tokenizer '{tokenizer}'")
                    
            elif config.get('embedder_type') == 'MultiLayerEmbedder':
                for layer in config['config'].get('embedder_configs', []):
                    if layer['type'] == 'TFIDFEmbedder':
                        tokenizer = layer['config'].get('chinese_tokenizer')
                        if tokenizer and tokenizer not in ['spacy', 'jieba', 'pkuseg']:
                            tokenizer_inconsistencies.append(
                                f"{config_file.name}: invalid tokenizer '{tokenizer}' in TF-IDF layer"
                            )
        
        if tokenizer_inconsistencies:
            self.fail("Tokenizer inconsistencies found: " + "; ".join(tokenizer_inconsistencies))
    
    def test_config_performance_parameters(self):
        """Test that performance-critical parameters are reasonable"""
        json_files = list(self.configs_dir.glob("*.json"))
        
        for config_file in json_files:
            with self.subTest(config_file=config_file.name):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if config.get('embedder_type') == 'TFIDFEmbedder':
                    self._check_tfidf_performance_params(config['config'], config_file.name)
                    
                elif config.get('embedder_type') == 'MultiLayerEmbedder':
                    for layer in config['config'].get('embedder_configs', []):
                        if layer['type'] == 'TFIDFEmbedder':
                            self._check_tfidf_performance_params(layer['config'], 
                                                               f"{config_file.name}:TF-IDF layer")
    
    def _check_tfidf_performance_params(self, config, config_name):
        """Helper method to check TF-IDF performance parameters"""
        # Check max_features is reasonable
        max_features = config.get('max_features')
        if max_features is not None:
            self.assertIsInstance(max_features, int, f"{config_name}: max_features must be int")
            self.assertGreater(max_features, 0, f"{config_name}: max_features must be positive")
            self.assertLessEqual(max_features, 100000, f"{config_name}: max_features too large")
        
        # Check min_df and max_df
        min_df = config.get('min_df', 1)
        max_df = config.get('max_df', 1.0)
        
        if isinstance(min_df, int):
            self.assertGreaterEqual(min_df, 1, f"{config_name}: min_df integer must be >= 1")
        elif isinstance(min_df, float):
            self.assertGreater(min_df, 0.0, f"{config_name}: min_df float must be > 0.0")
            self.assertLess(min_df, 1.0, f"{config_name}: min_df float must be < 1.0")
        
        if isinstance(max_df, int):
            self.assertGreater(max_df, min_df if isinstance(min_df, int) else 1,
                             f"{config_name}: max_df integer inconsistent with min_df")
        elif isinstance(max_df, float):
            self.assertGreater(max_df, 0.0, f"{config_name}: max_df float must be > 0.0")
            self.assertLessEqual(max_df, 1.0, f"{config_name}: max_df float must be <= 1.0")
    
    def test_config_encoding_validation(self):
        """Test that all config files are properly UTF-8 encoded"""
        json_files = list(self.configs_dir.glob("*.json"))
        
        for config_file in json_files:
            with self.subTest(config_file=config_file.name):
                try:
                    # Try to read with UTF-8 encoding
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Ensure content can be decoded and re-encoded
                    content.encode('utf-8').decode('utf-8')
                    
                except UnicodeDecodeError:
                    self.fail(f"Config file {config_file.name} is not properly UTF-8 encoded")
                except UnicodeEncodeError:
                    self.fail(f"Config file {config_file.name} contains invalid UTF-8 characters")
    
    @pytest.mark.slow
    def test_complete_workflow_validation(self):
        """Complete workflow test for critical config files"""
        critical_configs = [
            "tfidf_spacy_example.json",
            "multi_layer_simple.json"
        ]
        
        for config_name in critical_configs:
            config_path = self.configs_dir / config_name
            if config_path.exists():
                with self.subTest(config_file=config_name):
                    try:
                        # Load config
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        
                        # Create embedder based on config type
                        if config['embedder_type'] == 'TFIDFEmbedder':
                            embedder = TFIDFEmbedder(**config['config'])
                        elif config['embedder_type'] == 'MultiLayerEmbedder':
                            embedder = MultiLayerEmbedder(**config['config'])
                        else:
                            # Use factory for other types
                            embedder = EmbedderFactory.create_embedder(
                                config['embedder_type'],
                                cache_dir=self.cache_dir,
                                **config['config']
                            )
                        
                        # Test complete workflow: fit -> encode -> get_vocabulary_size
                        embedder.fit(self.chinese_test_texts[:3])
                        
                        # Test encoding
                        for text in self.chinese_test_texts[:2]:
                            vector = embedder.encode(text)
                            self.assertIsInstance(vector, list)
                            self.assertGreater(len(vector), 0)
                            self.assertTrue(all(isinstance(x, float) for x in vector))
                        
                        # Test vocabulary size (if available)
                        if hasattr(embedder, 'get_vocabulary_size'):
                            vocab_size = embedder.get_vocabulary_size()
                            self.assertIsInstance(vocab_size, int)
                            self.assertGreater(vocab_size, 0)
                            
                    except Exception as e:
                        self.fail(f"Complete workflow test failed for {config_name}: {e}")


@pytest.mark.unit
@pytest.mark.config
class TestConfigHelpers(unittest.TestCase):
    """Test configuration helper functions and utilities"""
    
    def test_ngram_range_conversion(self):
        """Test ngram_range type conversion from list to tuple"""
        from text_vectorify.embedders.tfidf import TFIDFEmbedder
        
        # Test list conversion
        embedder = TFIDFEmbedder(ngram_range=[1, 3])
        self.assertIsInstance(embedder.ngram_range, tuple)
        self.assertEqual(embedder.ngram_range, (1, 3))
        
        # Test tuple preservation
        embedder = TFIDFEmbedder(ngram_range=(2, 4))
        self.assertIsInstance(embedder.ngram_range, tuple)
        self.assertEqual(embedder.ngram_range, (2, 4))
    
    def test_multilayer_config_conversion(self):
        """Test MultiLayerEmbedder config type conversion"""
        from text_vectorify.embedders.multi_layer import MultiLayerEmbedder
        
        # Test that _convert_config_types works properly with dummy config
        dummy_embedders = [
            {
                'type': 'TFIDFEmbedder',
                'config': {
                    'max_features': 100,
                    'ngram_range': [1, 2],
                    'chinese_tokenizer': 'spacy'
                }
            }
        ]
        
        embedder = MultiLayerEmbedder(embedder_configs=dummy_embedders)
        
        test_config = {
            'ngram_range': [1, 2],
            'max_features': 1000,
            'chinese_tokenizer': 'spacy'
        }
        
        converted_config = embedder._convert_config_types(test_config)
        
        self.assertIsInstance(converted_config['ngram_range'], tuple)
        self.assertEqual(converted_config['ngram_range'], (1, 2))
        self.assertEqual(converted_config['max_features'], 1000)
        self.assertEqual(converted_config['chinese_tokenizer'], 'spacy')


if __name__ == "__main__":
    unittest.main(verbosity=2)
