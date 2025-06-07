#!/usr/bin/env python3
"""
Multi-layer embeddings usage examples demonstrating the new multi-layer vectorization capabilities
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for importing
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from text_vectorify import MultiLayerEmbedder, ClusteringPipeline, EmbedderFactory

def example_1_basic_multi_layer():
    """Basic multi-layer embedding example"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Multi-Layer Embedding")
    print("=" * 60)
    
    # Sample texts for demonstration
    texts = [
        "人工智慧技術正在快速發展，機器學習模型越來越強大",
        "Machine learning algorithms are becoming more sophisticated",
        "深度學習在電腦視覺領域取得了突破性進展",
        "Natural language processing has improved significantly",
        "大數據分析幫助企業做出更好的商業決策",
        "Cloud computing provides scalable infrastructure solutions",
        "區塊鏈技術正在改變金融行業的運作方式",
        "Cybersecurity threats are becoming more complex and frequent"
    ]
    
    # Create multi-layer embedder with manual configuration
    embedder_configs = [
        {
            "name": "tfidf",
            "type": "TFIDFEmbedder",
            "config": {
                "max_features": 500,
                "ngram_range": [1, 2]
            },
            "weight": 0.3
        },
        {
            "name": "semantic",
            "type": "BGEEmbedder",
            "config": {
                "model_name": "BAAI/bge-base-zh-v1.5"
            },
            "weight": 0.5
        },
        {
            "name": "topic",
            "type": "TopicEmbedder", 
            "config": {
                "n_topics": 8,
                "method": "bertopic",
                "language": "chinese"
            },
            "weight": 0.2
        }
    ]
    
    try:
        multi_embedder = MultiLayerEmbedder(
            embedder_configs=embedder_configs,
            fusion_method="weighted",
            cache_dir="./cache/multi_layer_demo"
        )
        
        # Fit embedders that need fitting (TF-IDF, Topic models)
        print("Fitting embedders on sample texts...")
        multi_embedder.fit(texts)
        
        # Encode texts with individual layers
        print("Encoding with individual layers...")
        layer_results = multi_embedder.encode_layers(texts[:3])
        
        for layer_name, vectors in layer_results.items():
            print(f"Layer '{layer_name}': {len(vectors)} vectors, dim={len(vectors[0])}")
        
        # Encode with fusion
        print("Encoding with multi-layer fusion...")
        fused_vectors = multi_embedder.encode(texts[:3])
        print(f"Fused vectors: {len(fused_vectors)} vectors, dim={len(fused_vectors[0])}")
        
        # Get layer information
        layer_info = multi_embedder.get_layer_info()
        print("\nLayer Information:")
        for name, info in layer_info.items():
            print(f"  {name}: {info['type']} - {info.get('max_features', info.get('n_topics', 'N/A'))}")
        
    except ImportError as e:
        print(f"Skipping example due to missing dependency: {e}")
    except Exception as e:
        print(f"Error in example: {e}")

def example_2_config_file_approach():
    """Multi-layer embedding using configuration file"""
    print("\n" + "=" * 60)
    print("Example 2: Config File Approach")
    print("=" * 60)
    
    # Sample data for 1500 articles simulation
    sample_articles = [
        {
            "title": "人工智慧在醫療診斷中的應用",
            "content": "深度學習技術正在革命性地改變醫療診斷的準確性和效率，特別是在影像識別和疾病預測方面。"
        },
        {
            "title": "區塊鏈技術的金融創新",
            "content": "去中心化金融(DeFi)平台利用智能合約提供傳統金融服務的替代方案，降低了交易成本。"
        },
        {
            "title": "5G網路對物聯網發展的影響",
            "content": "超低延遲和高頻寬的5G技術為智慧城市和工業4.0應用提供了強大的基礎設施支持。"
        },
        {
            "title": "可持續能源技術的突破",
            "content": "太陽能和風能技術的成本持續下降，使得再生能源成為更具競爭力的能源選擇。"
        },
        {
            "title": "量子計算的實際應用前景",
            "content": "量子演算法在密碼學、藥物發現和金融建模等領域展現出巨大的潛力和應用價值。"
        }
    ]
    
    # Combine title and content
    texts = [f"{article['title']} {article['content']}" for article in sample_articles]
    
    config_path = project_root / "configs" / "multi_layer_example.json"
    
    try:
        # Load multi-layer embedder from config
        multi_embedder = MultiLayerEmbedder.from_config_file(
            str(config_path),
            cache_dir="./cache/config_demo"
        )
        
        # Fit and process
        print("Fitting multi-layer embedder on sample articles...")
        multi_embedder.fit(texts)
        
        # Process articles
        print("Processing articles with multi-layer embedding...")
        results = {}
        
        # Get individual layer results
        layer_results = multi_embedder.encode_layers(texts)
        results['layers'] = layer_results
        
        # Get fused results
        fused_vectors = multi_embedder.encode(texts)
        results['fused'] = fused_vectors
        
        print(f"Processed {len(texts)} articles")
        print("Layer dimensions:")
        for layer_name, vectors in layer_results.items():
            print(f"  {layer_name}: {len(vectors[0])} dimensions")
        print(f"Fused: {len(fused_vectors[0])} dimensions")
        
        # Save configuration
        config_save_path = "./cache/saved_config.json"
        multi_embedder.save_config(config_save_path)
        print(f"Configuration saved to {config_save_path}")
        
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except ImportError as e:
        print(f"Skipping example due to missing dependency: {e}")
    except Exception as e:
        print(f"Error in example: {e}")

def example_3_clustering_pipeline():
    """Multi-layer embedding with intelligent clustering"""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Layer Embedding + Clustering Pipeline")
    print("=" * 60)
    
    # Extended sample for clustering
    sample_texts = [
        # AI & ML cluster
        "人工智慧技術在各個領域的應用越來越廣泛",
        "機器學習模型的準確性不斷提升",
        "深度學習網路架構越來越複雜",
        
        # Blockchain cluster  
        "區塊鏈技術正在改變金融業",
        "加密貨幣的市場波動性很大",
        "智能合約提供了新的商業模式",
        
        # Healthcare cluster
        "醫療人工智慧幫助診斷疾病",
        "遠程醫療服務越來越普及",
        "基因治療技術取得突破",
        
        # Technology cluster
        "5G網路提供超高速連接",
        "物聯網設備數量急速增長",
        "雲端計算降低了IT成本"
    ]
    
    try:
        # Create simple multi-layer embedder for clustering
        embedder_configs = [
            {
                "name": "tfidf",
                "type": "TFIDFEmbedder",
                "config": {"max_features": 300, "ngram_range": [1, 2]},
                "weight": 0.4
            },
            {
                "name": "semantic",
                "type": "BGEEmbedder",
                "config": {"model_name": "BAAI/bge-base-zh-v1.5"},
                "weight": 0.4
            },
            {
                "name": "topic",
                "type": "TopicEmbedder",
                "config": {"n_topics": 6, "method": "bertopic"},
                "weight": 0.2
            }
        ]
        
        multi_embedder = MultiLayerEmbedder(
            embedder_configs=embedder_configs,
            fusion_method="weighted",
            cache_dir="./cache/clustering_demo"
        )
        
        print("Fitting multi-layer embedder...")
        multi_embedder.fit(sample_texts)
        
        print("Generating embeddings...")
        # Get layer results for clustering
        layer_results = multi_embedder.encode_layers(sample_texts)
        fused_vectors = multi_embedder.encode(sample_texts)
        
        # Use topic vectors for guidance
        topic_vectors = layer_results.get('topic', None)
        
        print("Starting clustering pipeline...")
        clustering_pipeline = ClusteringPipeline(
            primary_method="DBSCAN",
            secondary_method="KMeans",
            target_clusters=4,
            eps=0.3,
            min_samples=2
        )
        
        # Perform clustering
        clustering_results = clustering_pipeline.fit_predict(
            fused_vectors, 
            topic_vectors=topic_vectors
        )
        
        print("Clustering Results:")
        print(json.dumps(clustering_results['final_analysis'], indent=2, ensure_ascii=False))
        
        # Get cluster summaries
        cluster_summaries = clustering_pipeline.get_cluster_summaries(sample_texts)
        print("\nCluster Summaries:")
        for cluster_id, summary in cluster_summaries.items():
            print(f"\nCluster {cluster_id} ({summary['size']} texts, {summary['percentage']:.1f}%):")
            for text in summary['sample_texts']:
                print(f"  - {text[:50]}...")
        
    except ImportError as e:
        print(f"Skipping example due to missing dependency: {e}")
    except Exception as e:
        print(f"Error in example: {e}")

def example_4_individual_embedders():
    """Test individual new embedders"""
    print("\n" + "=" * 60)
    print("Example 4: Individual New Embedders")
    print("=" * 60)
    
    test_texts = [
        "這是一個測試文本，用於展示TF-IDF向量化",
        "機器學習和人工智慧技術發展迅速"
    ]
    
    # Test TF-IDF Embedder
    print("Testing TF-IDF Embedder...")
    try:
        tfidf_embedder = EmbedderFactory.create_embedder(
            "TFIDFEmbedder",
            cache_dir="./cache/tfidf_test"
        )
        tfidf_embedder.fit(test_texts)
        tfidf_vectors = tfidf_embedder.encode(test_texts[0])
        print(f"TF-IDF vector dim: {len(tfidf_vectors)}")
        print(f"Vocabulary size: {tfidf_embedder.get_vocabulary_size()}")
    except Exception as e:
        print(f"TF-IDF test failed: {e}")
    
    # Test Topic Embedder
    print("\nTesting Topic Embedder...")
    try:
        topic_embedder = EmbedderFactory.create_embedder(
            "TopicEmbedder",
            cache_dir="./cache/topic_test",
            n_topics=3,
            method="bertopic"
        )
        topic_embedder.fit(test_texts)
        topic_vectors = topic_embedder.encode(test_texts[0])
        print(f"Topic vector dim: {len(topic_vectors)}")
        topics = topic_embedder.get_topics()
        print(f"Found {len(topics)} topics")
    except Exception as e:
        print(f"Topic test failed: {e}")

def main():
    """Run all examples"""
    print("Multi-Layer Embeddings Usage Examples")
    print("=====================================")
    
    example_1_basic_multi_layer()
    example_2_config_file_approach()
    example_3_clustering_pipeline()
    example_4_individual_embedders()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
