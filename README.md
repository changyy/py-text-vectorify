# text-vectorify

[![PyPI version](https://img.shields.io/pypi/v/text-vectorify.svg)](https://pypi.org/project/text-vectorify)
[![PyPI Downloads](https://static.pepy.tech/badge/text-vectorify)](https://pepy.tech/projects/text-vectorify)

A powerful and flexible Python tool for text vectorization with multiple embedding models and intelligent caching.

## 📋 Simple Description

**text-vectorify** is a command-line tool that converts text data in JSONL format into vector embeddings using various state-of-the-art models including OpenAI, SentenceBERT, BGE, M3E, and HuggingFace transformers. It features intelligent caching, multi-field text combination, and seamless JSONL processing for efficient text analysis pipelines.

## 🚀 Quick Start

```bash
pip install text-vectorify

# Basic usage with default model
text-vectorify \
  --input data.jsonl \
  --input-field-main "title" \
  --input-field-subtitle "content" \
  --process-method "OpenAIEmbedder" \
  --process-extra-data "your-openai-api-key"

# Using stdin input
cat data.jsonl | text-vectorify \
  --input-field-main "title" \
  --process-method "BGEEmbedder"
```

## ✨ Features

- **🎯 Multiple Embedding Models**: OpenAI, SentenceBERT, BGE, M3E, HuggingFace, TF-IDF, Topic Models
- **🌟 Multi-layer Vectorization**: Combine different embedding types with 3 fusion methods
- **🔤 Chinese Text Support**: Intelligent Chinese tokenization (jieba, spaCy, pkuseg)
- **📊 Topic Modeling**: LDA and BERTopic support for semantic topic analysis
- **🚄 Intelligent Caching**: Avoid recomputing embeddings for duplicate texts
- **📊 Flexible Field Combination**: Combine multiple JSON fields for embedding
- **📁 JSONL Processing**: Seamless input/output in JSONL format
- **⚡ Batch Processing**: Efficient processing of large datasets
- **🛡️ Error Resilience**: Continue processing even if individual records fail
- **📥 Stdin Support**: Read input from pipes or stdin for flexible data processing
- **🎛️ Smart Defaults**: Default model names for quick start without configuration
- **🔧 Flexible Input**: Support file input, stdin, or explicit stdin markers

## 📖 Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Supported Models](#supported-models)
- [Examples](#examples)
- [Library Usage](#library-usage)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Method 1: pip install (Recommended)

```bash
# Install core package (includes TF-IDF, Topic, and Multi-layer embedders)
pip install text-vectorify

# Install with specific advanced embedder support
pip install text-vectorify[openai]        # OpenAI support
pip install text-vectorify[transformers]  # SentenceBERT, BGE, M3E support
pip install text-vectorify[advanced]      # BERTopic, advanced tokenizers
pip install text-vectorify[all]           # All embedding models

# Install with development dependencies
pip install text-vectorify[dev]
```

**🆕 Core Features Included by Default:**
- ✅ **TF-IDF Embedder** with Chinese tokenization (spacy)
- ✅ **Topic Embedder** with LDA support
- ✅ **Multi-layer Embedder** with 3 fusion methods
- ✅ **Smart caching** and **CLI tools**

### Method 2: From source

```bash
# Clone repository
git clone https://github.com/changyy/py-text-vectorify.git
cd py-text-vectorify

# Automated development setup (recommended for contributors)
./setup.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package  
pip install -e .                # Core package only
# or
pip install -e ".[dev]"         # With development dependencies
# or  
pip install -e ".[all,dev]"     # With all optional dependencies
```

### Method 3: Development setup

```bash
# Automated setup for developers (recommended)
git clone https://github.com/changyy/py-text-vectorify.git
cd py-text-vectorify
./setup.sh

# Manual development setup
python3 -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Run tests to verify installation
python test_runner.py --quick
```

Install additional packages based on the embedding models you plan to use:

```bash
# For Chinese text processing (recommended - included by default)
python -m spacy download zh_core_web_sm

# For OpenAI embeddings
pip install openai

# For SentenceBERT, BGE, M3E models
pip install sentence-transformers

# For HuggingFace models
pip install transformers torch

# For alternative Chinese tokenizers (optional)
pip install text-vectorify[advanced]  # Includes jieba and pkuseg
```

### 🔤 Chinese Tokenization Setup

This package uses **spaCy as the default and recommended Chinese tokenizer** for superior accuracy and modern NLP features. Alternative tokenizers are available for specific use cases.

#### spaCy (Recommended - Default) 🌟
```bash
# Install spaCy Chinese model (required for Chinese text processing)
python -m spacy download zh_core_web_sm

# Usage in CLI (default - no need to specify tokenizer)
text-vectorify --process-method "TFIDFEmbedder" --input data.jsonl

# Explicit usage (optional)
text-vectorify --process-method "TFIDFEmbedder" --chinese-tokenizer spacy --input data.jsonl
```

**Why spaCy?**
- 🔄 Actively maintained with regular updates
- 🎯 Superior accuracy for modern Chinese text
- 🛠️ Rich linguistic features (POS tagging, NER, etc.)
- 📚 Comprehensive language model support

#### Alternative Tokenizers

**jieba** (Lightweight option)
```bash
# Install alternative tokenizers
pip install text-vectorify[advanced]

# Usage
text-vectorify --process-method "TFIDFEmbedder" --chinese-tokenizer jieba --input data.jsonl
```

**pkuseg** (Academic option)
```bash
# Already included in advanced extras
text-vectorify --process-method "TFIDFEmbedder" --chinese-tokenizer pkuseg --input data.jsonl
```

**Automatic Selection Priority:** spaCy → jieba → pkuseg

## 📝 Usage

### Command Line Interface

```bash
text-vectorify [OPTIONS]
```

#### Required Arguments

- `--input-field-main`: Main text fields (comma-separated)
- `--process-method`: Embedding method to use

#### Optional Arguments

- `--input`: Path to input JSONL file (use "-" for stdin, or omit to read from stdin)
- `--process-model-name`: Model name to use (optional, will use defaults if not specified)
- `--input-field-subtitle`: Additional text fields (comma-separated)
- `--process-extra-data`: Extra data like API keys
- `--output-field`: Output embedding field name (default: "embedding")
- `--output-cache-dir`: Cache directory (default: "./cache")
- `--output`: Output file path (default: auto-generated)

### Quick Start Features

The tool now supports smart defaults and flexible input methods for easier usage:

#### Default Models
Each embedder has intelligent default models, so you don't need to specify `--process-model-name`:
- **OpenAI**: `text-embedding-3-small`
- **BGE**: `BAAI/bge-small-en-v1.5` 
- **SentenceBERT**: `paraphrase-multilingual-MiniLM-L12-v2`
- **M3E**: `moka-ai/m3e-base`
- **HuggingFace**: `sentence-transformers/all-MiniLM-L6-v2`

#### Flexible Input Methods
- **File input**: `--input data.jsonl`
- **Stdin (auto-detect)**: `cat data.jsonl | text-vectorify ...`
- **Explicit stdin**: `--input -`

#### Minimal Example
```bash
# The simplest possible usage
cat data.jsonl | text-vectorify --input-field-main "title" --process-method "BGEEmbedder"
```

### Input Format

JSONL file with text data:

```json
{"title": "Sample Article", "content": "This is the content...", "author": "John Doe"}
{"title": "Another Article", "content": "More content here...", "author": "Jane Smith"}
```

### Output Format

JSONL file with added vector embeddings:

```json
{"title": "Sample Article", "content": "This is the content...", "author": "John Doe", "embedding": [0.1, 0.2, 0.3, ...]}
{"title": "Another Article", "content": "More content here...", "author": "Jane Smith", "embedding": [0.4, 0.5, 0.6, ...]}
```

## 🤖 Supported Models

### Core Embedders (Included by Default)

#### TF-IDF Embedder
- **Method**: `TFIDFEmbedder`
- **Default Features**: 50 dimensions
- **Chinese Support**: Uses spaCy tokenization by default (alternative: jieba, pkuseg)
- **Customizable**: `max_features`, `min_df`, `max_df` parameters
- **Cache**: Full sklearn vectorizer caching

#### Topic Embedder  
- **Method**: `TopicEmbedder`
- **Default Topics**: 10 topics (LDA)
- **Algorithms**: LDA (default), BERTopic (with advanced install)
- **Dimensions**: n_topics + 6 (topic distribution + metadata)
- **Custom Config**: `--process-extra-data "n_topics=20,method=lda"`

#### Multi-layer Embedder
- **Method**: `MultiLayerEmbedder`
- **Fusion Methods**: concatenate (default), average, weighted
- **Layer Combinations**: Any combination of available embedders
- **Config Support**: JSON config files for complex setups
- **Dimensions**: Varies by fusion method and layers

### Advanced Embedders (Optional Install)

#### OpenAI Embeddings
- **Default Model**: `text-embedding-3-small`
- **Other Models**: `text-embedding-3-large`
- **API Key**: Required via `--process-extra-data`
- **Dimensions**: 1536 (small), 3072 (large)
- **Install**: `pip install text-vectorify[openai]`

#### SentenceBERT
- **Default Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Language**: Multilingual support
- **Dimensions**: 384
- **Install**: `pip install text-vectorify[transformers]`

#### BGE (Beijing Academy of AI)
- **Default Model**: `BAAI/bge-small-en-v1.5`
- **Other Models**: `BAAI/bge-base-zh-v1.5`, `BAAI/bge-small-zh-v1.5`
- **Language**: Optimized for Chinese and English
- **Dimensions**: 512 (small), 768 (base)
- **Install**: `pip install text-vectorify[transformers]`

#### M3E (Moka Massive Mixed Embedding)
- **Default Model**: `moka-ai/m3e-base`
- **Other Models**: `moka-ai/m3e-small`
- **Language**: Chinese specialized
- **Dimensions**: 768 (base), 512 (small)
- **Install**: `pip install text-vectorify[transformers]`

#### HuggingFace Transformers
- **Default Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Flexibility**: Custom model selection
- **Dimensions**: Model-dependent
- **Install**: `pip install text-vectorify[transformers]`

### 🔧 Quick Model Selection Guide

```bash
# Core models (no extra install needed)
--process-method "TFIDFEmbedder"           # Fast, traditional, good baseline
--process-method "TopicEmbedder"           # Semantic topics, interpretable  
--process-method "MultiLayerEmbedder"      # Rich, combined representations

# Advanced models (require optional dependencies)
--process-method "BGEEmbedder"             # Best for Chinese/English
--process-method "SentenceBertEmbedder"    # Multilingual, balanced
--process-method "OpenAIEmbedder"          # High quality, requires API key
--process-method "HuggingFaceEmbedder"     # Most flexible, custom models
```

## 📚 Examples

### Example 1: OpenAI Embeddings (with default model)

```bash
text-vectorify \
  --input articles.jsonl \
  --input-field-main "title" \
  --input-field-subtitle "content,summary" \
  --process-method "OpenAIEmbedder" \
  --process-extra-data "sk-your-openai-api-key" \
  --output-field "embedding" \
  --output processed_articles.jsonl
```

### Example 2: Using stdin input with default BGE model

```bash
cat chinese_news.jsonl | text-vectorify \
  --input-field-main "title,content" \
  --process-method "BGEEmbedder" \
  --output-cache-dir ./models_cache
```

### Example 3: Multilingual with SentenceBERT (default model)

```bash
text-vectorify \
  --input multilingual_docs.jsonl \
  --input-field-main "title" \
  --input-field-subtitle "description,tags" \
  --process-method "SentenceBertEmbedder"
```

### Example 4: Custom model specification

```bash
text-vectorify \
  --input products.jsonl \
  --input-field-main "name,brand" \
  --input-field-subtitle "description,category,tags" \
  --process-method "BGEEmbedder" \
  --process-model-name "BAAI/bge-base-zh-v1.5" \
  --output-field "product_vector"
```

### Example 5: Explicit stdin marker

```bash
echo '{"title": "Sample", "content": "Text content"}' | text-vectorify \
  --input - \
  --input-field-main "title" \
  --process-method "M3EEmbedder"
```

### Example 6: Quick start with minimal arguments

```bash
# Most minimal usage - using defaults
cat data.jsonl | text-vectorify \
  --input-field-main "title" \
  --process-method "BGEEmbedder"
```

### Example 7: TF-IDF Embeddings (included by default)

```bash
# TF-IDF with Chinese tokenization (spacy is the default)
text-vectorify \
  --input chinese_articles.jsonl \
  --input-field-main "title,content" \
  --process-method "TFIDFEmbedder" \
  --max-features 1500

# Use alternative tokenizer if needed (default is spacy)
text-vectorify \
  --input chinese_articles.jsonl \
  --input-field-main "title,content" \
  --process-method "TFIDFEmbedder" \
  --chinese-tokenizer jieba \
  --max-features 1500
```

### Example 8: Topic Embeddings with LDA

```bash
# Generate topic-based embeddings
text-vectorify \
  --input documents.jsonl \
  --input-field-main "content" \
  --process-method "TopicEmbedder" \
  --n-topics 20
```

### Example 9: Multi-layer Vectorization with Config File

First, create a config file `multi_config.json`:
```json
{
  "fusion_method": "concatenate",
  "layers": [
    {
      "embedder_type": "TFIDFEmbedder",
      "config": {
        "tokenizer": "chinese",
        "max_features": 1000
      }
    },
    {
      "embedder_type": "TopicEmbedder", 
      "config": {
        "n_topics": 10,
        "method": "lda"
      }
    }
  ]
}
```

Then run:
```bash
# Use config file for complex multi-layer setup
text-vectorify \
  --input documents.jsonl \
  --input-field-main "content" \
  --process-method "MultiLayerEmbedder" \
  --config-file multi_config.json
```

### Example 10: Multi-layer with Direct Parameters

```bash
# Multi-layer with weighted fusion (uses default layers)
text-vectorify \
  --input mixed_docs.jsonl \
  --input-field-main "title,content" \
  --process-method "MultiLayerEmbedder" \
  --fusion-method weighted
```

### Example 11: Cache Management

```bash
# Show cache statistics
text-vectorify --show-cache-stats

# List cached files
text-vectorify --list-cache-files

# Use custom cache directory
text-vectorify \
  --input data.jsonl \
  --input-field-main "title" \
  --process-method "TFIDFEmbedder" \
  --cache-dir "./my_project_cache"

# Clear specific cache before processing
text-vectorify \
  --input data.jsonl \
  --input-field-main "title" \
  --process-method "TFIDFEmbedder" \
  --clear-cache

# Clear all caches (with confirmation)
text-vectorify --clear-all-caches
```

### 🎯 CLI-First Development Workflow

For the most efficient development experience, we recommend starting with CLI experimentation:

```bash
# Step 1: Try the interactive workflow demo
python examples/cli_first_workflow_demo.py

# Step 2: Try the comprehensive development guide  
python examples/development_workflow_guide.py

# Step 3: Use your own data with the CLI-first approach
```

**Benefits**: Fast iteration → Cache building → Library integration → Zero recomputation

## 📚 Library Usage

For programmatic integration, `text-vectorify` provides a powerful Python API that allows you to process data in-memory using List[Dict] format instead of file-based operations. **We recommend a CLI-first development workflow** for maximum efficiency and cache optimization.

### 🔄 Recommended Development Workflow

#### Step 1: CLI Experimentation (Cache Building)
Start with CLI commands on small datasets to build cache and verify results:

```bash
# Quick test with small dataset
python -m text_vectorify.main \
  --input small_sample.jsonl \
  --input-field-main "title" \
  --input-field-subtitle "content" \
  --process-method "BGEEmbedder" \
  --cache-dir "./my_project_cache" \
  --output test_results.jsonl

# Verify results
head -3 test_results.jsonl
python -m text_vectorify.main --show-cache-stats
```

#### Step 2: Library Integration (Cache Reuse)
Switch to library usage while reusing the CLI-built cache:

```python
from text_vectorify import EmbedderFactory

# Reuse cache from CLI experiments - no recomputation!
embedder = EmbedderFactory.create_embedder(
    "BGEEmbedder",
    cache_dir="./my_project_cache"  # Same as CLI
)

# Process data in memory
data = [
    {"title": "Python Programming", "content": "Learn Python basics"},
    {"title": "Machine Learning", "content": "Introduction to ML concepts"}
]

for item in data:
    text = f"{item['title']} {item['content']}"
    vector = embedder.encode(text)  # Uses cache - instant results!
    item['embedding'] = vector
```

### 🚀 Quick Start with In-Memory Processing

```python
from text_vectorify import TextVectorify, EmbedderFactory

# Process data directly in memory
data = [
    {"title": "Python Programming", "content": "Learn Python basics"},
    {"title": "Machine Learning", "content": "Introduction to ML concepts"},
    {"title": "Data Science", "content": "Working with data in Python"}
]

# Create embedder with custom cache location
embedder = EmbedderFactory.create_embedder(
    "BGEEmbedder",
    cache_dir="./my_custom_cache"  # Customize cache location
)

# Initialize vectorizer
vectorizer = TextVectorify(embedder)

# Process list of dictionaries directly
for item in data:
    # Combine text fields
    text = f"{item['title']} {item['content']}"
    
    # Generate embedding
    vector = embedder.encode(text)
    
    # Add to your data structure
    item['embedding'] = vector

print(f"Processed {len(data)} items with embeddings")
```

### 🔧 Advanced Library Integration

#### Batch Processing with Custom Cache Management

```python
from text_vectorify import TextVectorify, EmbedderFactory
from pathlib import Path
import tempfile

def process_documents_batch(documents: List[Dict], 
                          embedder_type: str = "BGEEmbedder",
                          cache_dir: str = None) -> List[Dict]:
    """
    Process a batch of documents with intelligent caching.
    
    Args:
        documents: List of dictionaries with text data
        embedder_type: Type of embedder to use
        cache_dir: Custom cache directory (optional)
    
    Returns:
        List of documents with added embeddings
    """
    
    # Use temporary cache if not specified
    if cache_dir is None:
        cache_dir = tempfile.mkdtemp(prefix="text_vectorify_")
    
    # Create embedder with custom cache
    embedder = EmbedderFactory.create_embedder(
        embedder_type,
        cache_dir=cache_dir
    )
    
    vectorizer = TextVectorify(embedder)
    
    # Process each document
    results = []
    for doc in documents:
        # Extract and combine text fields
        text = " ".join(str(doc[field]) for field in ['title', 'content', 'description'] if field in doc)
        
        # Generate embedding with automatic caching
        vector = embedder.encode(text)
        
        # Create result with original data + embedding
        result = doc.copy()
        result['embedding'] = vector
        results.append(result)
    
    return results

# Usage example
documents = [
    {"title": "AI Research", "content": "Latest developments in AI"},
    {"title": "Data Analysis", "content": "Tools for data processing"},
]

# Process with custom cache location
processed = process_documents_batch(
    documents, 
    embedder_type="BGEEmbedder",
    cache_dir="./persistent_cache"
)
```

#### Multi-Model Processing Pipeline

```python
from text_vectorify import EmbedderFactory
from typing import Dict, List, Optional
import logging

class MultiModelEmbeddingPipeline:
    """
    Pipeline for processing text with multiple embedding models.
    Useful for comparing embeddings or ensemble approaches.
    """
    
    def __init__(self, cache_base_dir: str = "./embeddings_cache"):
        self.cache_base_dir = Path(cache_base_dir)
        self.embedders = {}
        
    def add_embedder(self, name: str, embedder_type: str, model_name: str = None, **kwargs):
        """Add an embedder to the pipeline."""
        cache_dir = self.cache_base_dir / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        embedder = EmbedderFactory.create_embedder(
            embedder_type,
            model_name=model_name,
            cache_dir=str(cache_dir),
            **kwargs
        )
        
        self.embedders[name] = embedder
        logging.info(f"Added embedder '{name}' with cache at {cache_dir}")
    
    def process_texts(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        """Process texts with all configured embedders."""
        results = {}
        
        for embedder_name, embedder in self.embedders.items():
            embeddings = []
            for text in texts:
                vector = embedder.encode(text)
                embeddings.append(vector)
            
            results[embedder_name] = embeddings
            logging.info(f"Processed {len(texts)} texts with {embedder_name}")
        
        return results

# Usage example
pipeline = MultiModelEmbeddingPipeline("./multi_model_cache")

# Add different embedders with custom cache locations
pipeline.add_embedder("bge_small", "BGEEmbedder", "BAAI/bge-small-en-v1.5")
pipeline.add_embedder("sentence_bert", "SentenceBertEmbedder")

# For OpenAI (requires API key)
if openai_api_key:
    pipeline.add_embedder("openai", "OpenAIEmbedder", api_key=openai_api_key)

# Process texts
texts = ["Machine learning concepts", "Natural language processing"]
all_embeddings = pipeline.process_texts(texts)

# Results contain embeddings from all models
for model_name, embeddings in all_embeddings.items():
    print(f"{model_name}: {len(embeddings)} embeddings, dim={len(embeddings[0])}")
```

### 🎯 Integration Patterns

#### Django/Flask Web Application Integration

```python
# views.py or similar
from text_vectorify import EmbedderFactory
from django.conf import settings

class DocumentEmbeddingService:
    """Service for generating embeddings in web applications."""
    
    def __init__(self):
        # Use application cache directory
        cache_dir = getattr(settings, 'EMBEDDINGS_CACHE_DIR', './embeddings_cache')
        
        self.embedder = EmbedderFactory.create_embedder(
            "BGEEmbedder",  # Fast and good quality
            cache_dir=cache_dir
        )
    
    def embed_document(self, title: str, content: str) -> List[float]:
        """Generate embedding for a document."""
        text = f"{title} {content}"
        return self.embedder.encode(text)
    
    def search_similar(self, query: str, document_embeddings: List[List[float]], 
                      threshold: float = 0.7) -> List[int]:
        """Find similar documents using cosine similarity."""
        query_embedding = self.embedder.encode(query)
        
        # Simple similarity calculation (use numpy/scipy for better performance)
        similar_indices = []
        for i, doc_embedding in enumerate(document_embeddings):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                similar_indices.append(i)
        
        return similar_indices
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot_product / (norm_a * norm_b)

# Usage in views
embedding_service = DocumentEmbeddingService()

def create_document(request):
    # ... extract title and content from request ...
    
    # Generate embedding
    embedding = embedding_service.embed_document(title, content)
    
    # Save to database with embedding
    Document.objects.create(
        title=title,
        content=content,
        embedding=embedding  # Store as JSON or use vector database
    )
```

#### Data Pipeline Integration

```python
import pandas as pd
from text_vectorify import EmbedderFactory
from typing import Iterator

def embedding_pipeline(data_source: Iterator[Dict], 
                      output_path: str,
                      batch_size: int = 100,
                      cache_dir: str = "./pipeline_cache") -> None:
    """
    Process large datasets in batches with persistent caching.
    
    This approach avoids memory issues and provides resume capability.
    """
    
    # Create embedder with persistent cache
    embedder = EmbedderFactory.create_embedder(
        "BGEEmbedder",
        cache_dir=cache_dir
    )
    
    batch = []
    processed_count = 0
    
    for item in data_source:
        batch.append(item)
        
        if len(batch) >= batch_size:
            # Process batch
            processed_batch = process_batch(batch, embedder)
            
            # Save batch results
            save_batch_results(processed_batch, output_path, processed_count)
            
            processed_count += len(batch)
            batch = []
            
            print(f"Processed {processed_count} items")
    
    # Process remaining items
    if batch:
        processed_batch = process_batch(batch, embedder)
        save_batch_results(processed_batch, output_path, processed_count)

def process_batch(batch: List[Dict], embedder) -> List[Dict]:
    """Process a single batch of items."""
    results = []
    
    for item in batch:
        # Extract text
        text = f"{item.get('title', '')} {item.get('content', '')}"
        
        # Generate embedding (cached automatically)
        embedding = embedder.encode(text)
        
        # Add embedding to item
        result = item.copy()
        result['embedding'] = embedding
        results.append(result)
    
    return results

# Usage with pandas
def process_dataframe_with_embeddings(df: pd.DataFrame, 
                                    text_columns: List[str],
                                    cache_dir: str = None) -> pd.DataFrame:
    """Add embeddings to a pandas DataFrame."""
    
    embedder = EmbedderFactory.create_embedder(
        "BGEEmbedder",
        cache_dir=cache_dir or "./dataframe_cache"
    )
    
    # Combine text columns
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Generate embeddings
    embeddings = []
    for text in df['combined_text']:
        embedding = embedder.encode(text)
        embeddings.append(embedding)
    
    df['embedding'] = embeddings
    return df
```

### 🌟 Multi-layer Vectorization Examples

The multi-layer embedder allows you to combine different embedding techniques for richer text representations. Here are comprehensive examples:

#### Basic Multi-layer Usage

```python
from text_vectorify import EmbedderFactory

# Create multi-layer embedder with concatenation fusion
multi_embedder = EmbedderFactory.create_embedder(
    "MultiLayerEmbedder",
    extra_data="fusion_method=concatenate,layer1=TFIDFEmbedder,layer2=TopicEmbedder"
)

# Process text
text = "Machine learning algorithms in natural language processing applications"
fused_vector = multi_embedder.encode(text)

print(f"Fused vector dimensions: {len(fused_vector)}")
# Output: Fused vector dimensions: 66 (TF-IDF: 50 + Topic: 16)
```

#### Advanced Multi-layer with Configuration

```python
import json
from text_vectorify import EmbedderFactory

# Create configuration for complex multi-layer setup
config = {
    "fusion_method": "concatenate",
    "layers": [
        {
            "embedder_type": "TFIDFEmbedder",
            "config": {
                "tokenizer": "chinese",
                "max_features": 1000,
                "min_df": 2,
                "max_df": 0.8
            }
        },
        {
            "embedder_type": "TopicEmbedder",
            "config": {
                "n_topics": 20,
                "method": "lda",
                "random_state": 42
            }
        }
    ]
}

# Save config to file
with open("advanced_multi_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Create embedder using config file
multi_embedder = EmbedderFactory.create_embedder(
    "MultiLayerEmbedder",
    model_name="advanced_multi_config.json"
)

# Process documents
documents = [
    "人工智能技术在自然语言处理中的应用",
    "机器学习算法的最新发展趋势",
    "深度学习模型在文本分析中的表现"
]

embeddings = []
for doc in documents:
    vector = multi_embedder.encode(doc)
    embeddings.append(vector)
    print(f"Document: {doc[:30]}... -> Vector dim: {len(vector)}")
```

#### Multi-layer with Different Fusion Methods

```python
from text_vectorify import EmbedderFactory

# Test different fusion methods
fusion_methods = ["concatenate", "average", "weighted"]
text = "Advanced machine learning techniques for text analysis"

for method in fusion_methods:
    embedder = EmbedderFactory.create_embedder(
        "MultiLayerEmbedder",
        extra_data=f"fusion_method={method},layer1=TFIDFEmbedder,layer2=TopicEmbedder"
    )
    
    vector = embedder.encode(text)
    print(f"{method.capitalize()} fusion: {len(vector)} dimensions")

# Output:
# Concatenate fusion: 66 dimensions
# Average fusion: 50 dimensions (max of layer dimensions)
# Weighted fusion: 50 dimensions (max of layer dimensions)
```

#### Multi-layer Processing Pipeline

```python
from text_vectorify import EmbedderFactory
from typing import List, Dict
import numpy as np

class MultiLayerDocumentProcessor:
    """Advanced document processor using multi-layer embeddings."""
    
    def __init__(self, config_path: str = None, cache_dir: str = "./multi_cache"):
        if config_path:
            self.embedder = EmbedderFactory.create_embedder(
                "MultiLayerEmbedder",
                model_name=config_path,
                cache_dir=cache_dir
            )
        else:
            # Default configuration
            self.embedder = EmbedderFactory.create_embedder(
                "MultiLayerEmbedder",
                extra_data="fusion_method=concatenate,layer1=TFIDFEmbedder,layer2=TopicEmbedder",
                cache_dir=cache_dir
            )
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process documents with multi-layer embeddings."""
        results = []
        
        for doc in documents:
            # Combine text fields
            text_parts = []
            for field in ['title', 'content', 'description', 'tags']:
                if field in doc and doc[field]:
                    text_parts.append(str(doc[field]))
            
            combined_text = " ".join(text_parts)
            
            # Generate multi-layer embedding
            embedding = self.embedder.encode(combined_text)
            
            # Add to result
            result = doc.copy()
            result['embedding'] = embedding
            result['embedding_dim'] = len(embedding)
            result['text_combined'] = combined_text
            results.append(result)
        
        return results
    
    def calculate_similarities(self, documents: List[Dict], 
                             query_text: str) -> List[tuple]:
        """Calculate similarities between query and documents."""
        query_embedding = self.embedder.encode(query_text)
        
        similarities = []
        for i, doc in enumerate(documents):
            if 'embedding' in doc:
                similarity = self._cosine_similarity(query_embedding, doc['embedding'])
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        a_np = np.array(a)
        b_np = np.array(b)
        return np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np))

# Usage example
processor = MultiLayerDocumentProcessor()

documents = [
    {
        "title": "Machine Learning Basics", 
        "content": "Introduction to ML algorithms and concepts",
        "tags": "AI, ML, algorithms"
    },
    {
        "title": "Deep Learning Networks",
        "content": "Neural networks and deep learning architectures", 
        "tags": "DL, neural networks, AI"
    }
]

# Process documents
processed_docs = processor.process_documents(documents)

# Find similar documents
query = "artificial intelligence algorithms"
similarities = processor.calculate_similarities(processed_docs, query)

print("Document similarities:")
for doc_idx, similarity in similarities:
    title = processed_docs[doc_idx]['title']
    print(f"  {title}: {similarity:.4f}")
```

#### Batch Multi-layer Processing

```python
from text_vectorify import EmbedderFactory
import json
from pathlib import Path

def batch_multi_layer_processing(input_file: str, 
                                config_file: str,
                                output_file: str,
                                batch_size: int = 100):
    """
    Process large datasets with multi-layer embeddings in batches.
    
    Args:
        input_file: Path to input JSONL file
        config_file: Path to configuration file
        output_file: Path to output JSONL file
        batch_size: Number of records per batch
    """
    
    # Create multi-layer embedder
    embedder = EmbedderFactory.create_embedder(
        "MultiLayerEmbedder",
        model_name=config_file,
        cache_dir="./batch_multi_cache"
    )
    
    # Process in batches
    batch = []
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                item = json.loads(line.strip())
                batch.append(item)
                
                if len(batch) >= batch_size:
                    # Process batch
                    for item in batch:
                        # Combine text fields
                        text = f"{item.get('title', '')} {item.get('content', '')}"
                        
                        # Generate multi-layer embedding
                        embedding = embedder.encode(text)
                        
                        # Add embedding to item
                        item['embedding'] = embedding
                        item['embedding_dim'] = len(embedding)
                        
                        # Write to output
                        outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                    
                    processed_count += len(batch)
                    batch = []
                    print(f"Processed {processed_count} documents")
            
            # Process remaining items
            if batch:
                for item in batch:
                    text = f"{item.get('title', '')} {item.get('content', '')}"
                    embedding = embedder.encode(text)
                    item['embedding'] = embedding
                    item['embedding_dim'] = len(embedding)
                    outfile.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                processed_count += len(batch)
    
    print(f"Total processed: {processed_count} documents")
    print(f"Output saved to: {output_file}")

# Usage
batch_multi_layer_processing(
    input_file="large_dataset.jsonl",
    config_file="configs/multi_layer_simple.json", 
    output_file="processed_with_multilayer.jsonl",
    batch_size=50
)
```

### 🔍 Performance Optimization

#### Cache Management Best Practices

```python
from text_vectorify import EmbedderFactory
from pathlib import Path
import tempfile
import shutil

class OptimizedEmbeddingManager:
    """
    Optimized embedding manager with advanced cache control.
    """
    
    def __init__(self, base_cache_dir: str = None, 
                 max_cache_size_gb: float = 5.0):
        self.base_cache_dir = Path(base_cache_dir or "./optimized_cache")
        self.max_cache_size_gb = max_cache_size_gb
        self.embedders = {}
    
    def get_embedder(self, embedder_type: str, model_name: str = None, **kwargs):
        """Get or create embedder with optimized caching."""
        
        # Create unique cache directory for this configuration
        cache_key = f"{embedder_type}_{model_name or 'default'}"
        cache_dir = self.base_cache_dir / cache_key
        
        if cache_key not in self.embedders:
            # Ensure cache directory exists
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create embedder
            embedder = EmbedderFactory.create_embedder(
                embedder_type,
                model_name=model_name,
                cache_dir=str(cache_dir),
                **kwargs
            )
            
            self.embedders[cache_key] = embedder
            
            # Clean old caches if needed
            self._cleanup_cache_if_needed()
        
        return self.embedders[cache_key]
    
    def _cleanup_cache_if_needed(self):
        """Clean up cache if it exceeds size limit."""
        total_size = self._get_cache_size_gb()
        
        if total_size > self.max_cache_size_gb:
            print(f"Cache size ({total_size:.2f}GB) exceeds limit ({self.max_cache_size_gb}GB)")
            self._cleanup_oldest_caches()
    
    def _get_cache_size_gb(self) -> float:
        """Calculate total cache size in GB."""
        total_bytes = 0
        for cache_dir in self.base_cache_dir.iterdir():
            if cache_dir.is_dir():
                total_bytes += sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def _cleanup_oldest_caches(self):
        """Remove oldest cache directories to free space."""
        cache_dirs = [(d, d.stat().st_mtime) for d in self.base_cache_dir.iterdir() if d.is_dir()]
        cache_dirs.sort(key=lambda x: x[1])  # Sort by modification time
        
        # Remove oldest 30% of caches
        to_remove = cache_dirs[:len(cache_dirs) // 3]
        
        for cache_dir, _ in to_remove:
            print(f"Removing old cache: {cache_dir}")
            shutil.rmtree(cache_dir)

# Usage
manager = OptimizedEmbeddingManager(max_cache_size_gb=2.0)

# Get embedders with automatic cache management
bge_embedder = manager.get_embedder("BGEEmbedder")
openai_embedder = manager.get_embedder("OpenAIEmbedder", api_key=api_key)
```

### 💡 Key Benefits of CLI-First + Library Workflow

1. **🔬 Fast Development Cycle**: CLI for quick iteration and testing
2. **💾 Smart Cache Reuse**: No duplicate computations between CLI and library usage
3. **🔍 Enhanced Observability**: Easy result inspection with CLI output files
4. **🧪 Improved Testability**: Start small with CLI, scale with library
5. **⚡ Zero Recomputation**: Library leverages CLI-built cache automatically
6. **🎯 Custom Cache Control**: Configure cache location per project or use case
7. **🔄 Seamless Transition**: Move from CLI prototyping to library integration
8. **🛡️ Production Ready**: Robust error handling and resource management

### 🎯 CLI-First Workflow Benefits

- **Development Phase**: Use CLI with small datasets (5-10 samples)
- **Verification Phase**: Inspect output files and tune parameters
- **Integration Phase**: Switch to library with instant cache hits
- **Production Phase**: Scale up processing with zero wasted computation

### 🔗 Next Steps

- Try the workflow: `python examples/development_workflow_guide.py`
- See [API Reference](#-api-reference) for detailed method documentation
- Check [Examples](#-examples) for more CLI usage patterns
- Review [Configuration](#️-configuration) for cache and performance tuning

## 🔧 API Reference

### Python API Usage

```python
from text_vectorify import TextVectorify, EmbedderFactory

# Create embedder
embedder = EmbedderFactory.create_embedder(
    "OpenAIEmbedder",
    "text-embedding-3-small",
    api_key="your-api-key"
)

# Initialize vectorizer
vectorizer = TextVectorify(embedder)

# Process JSONL file
vectorizer.process_jsonl(
    input_path="input.jsonl",
    output_path="output.jsonl",
    input_field_main=["title"],
    input_field_subtitle=["content"],
    output_field="embedding"
)
```

### Available Embedders

```python
from text_vectorify import EmbedderFactory

# List all available embedders
embedders = EmbedderFactory.list_embedders()
print(embedders)
# ['OpenAIEmbedder', 'SentenceBertEmbedder', 'BGEEmbedder', 'M3EEmbedder', 'HuggingFaceEmbedder']
```

## ⚙️ Configuration

### 📁 Pre-built Configuration Files

The `configs/` directory provides ready-to-use configuration files for various scenarios:

#### Single Embedder Configurations
- **`tfidf_spacy_example.json`** - TF-IDF with spaCy tokenization (recommended for accuracy)
- **`tfidf_jieba_example.json`** - TF-IDF with jieba tokenization (lightweight option)
- **`tfidf_pkuseg_example.json`** - TF-IDF with pkuseg tokenization (domain-specific)

#### Multi-Layer Configurations
- **`multi_layer_spacy_recommended.json`** - Recommended multi-layer setup (TF-IDF + BGE + Topic)
- **`multi_layer_jieba_lightweight.json`** - Lightweight multi-layer (TF-IDF + M3E)
- **`multi_layer_simple.json`** - Basic multi-layer configuration
- **`multi_layer_advanced_research.json`** - Advanced research configuration
- **`multi_layer_1500_articles.json`** - Large-scale processing configuration

#### Usage with Config Files
```bash
# Use pre-built configuration
python -m text_vectorify --config configs/multi_layer_spacy_recommended.json --input data.jsonl

# List available configurations
ls configs/*.json
```

💡 **Tip: See [`configs/README.md`](configs/README.md) for detailed descriptions of each configuration file.**

### Multi-layer Configuration Files

Multi-layer embedder supports JSON configuration files for complex setups:

#### Basic Configuration
```json
{
  "fusion_method": "concatenate",
  "layers": [
    {
      "embedder_type": "TFIDFEmbedder",
      "config": {
        "tokenizer": "chinese",
        "max_features": 1000
      }
    },
    {
      "embedder_type": "TopicEmbedder",
      "config": {
        "n_topics": 10,
        "method": "lda"
      }
    }
  ]
}
```

#### Advanced Configuration
```json
{
  "fusion_method": "weighted",
  "fusion_weights": [0.7, 0.3],
  "layers": [
    {
      "embedder_type": "TFIDFEmbedder",
      "config": {
        "tokenizer": "chinese",
        "max_features": 2000,
        "min_df": 2,
        "max_df": 0.8,
        "use_idf": true,
        "smooth_idf": false,
        "sublinear_tf": true
      }
    },
    {
      "embedder_type": "TopicEmbedder",
      "config": {
        "n_topics": 20,
        "method": "lda",
        "random_state": 42,
        "max_iter": 100,
        "doc_topic_prior": 0.1,
        "topic_word_prior": 0.01
      }
    }
  ]
}
```

#### Available Configuration Options

**Fusion Methods:**
- `concatenate`: Join vectors end-to-end (default)
- `average`: Element-wise average of vectors
- `weighted`: Weighted average (requires `fusion_weights`)

**TFIDFEmbedder Config:**
- `chinese_tokenizer`: "spacy", "jieba", "pkuseg" (default: "spacy")
- `max_features`: Maximum number of features (default: 50)
- `min_df`: Minimum document frequency (default: 1)
- `max_df`: Maximum document frequency (default: 1.0)
- `use_idf`: Use inverse document frequency (default: true)
- `smooth_idf`: Smooth IDF weights (default: true)
- `sublinear_tf`: Apply sublinear tf scaling (default: false)

**TopicEmbedder Config:**
- `n_topics`: Number of topics (default: 10)
- `method`: "lda" or "bertopic" (default: "lda")
- `random_state`: Random seed for reproducibility
- `max_iter`: Maximum iterations for LDA (default: 20)
- `doc_topic_prior`: Document-topic prior (alpha)
- `topic_word_prior`: Topic-word prior (beta)

### Cache Management

The tool automatically caches:
- **Text embeddings**: Avoid recomputing identical texts
- **Model files**: Download models once and reuse
- **Layer embeddings**: Cache individual layer outputs
- **Fused vectors**: Cache final multi-layer results
- **Cache location**: Configurable via `--output-cache-dir`

### Cache Structure

```
cache/
├── models/                          # Downloaded models
│   ├── sentence_transformers/
│   ├── huggingface/
│   └── bge/
├── TFIDFEmbedder_*.json           # TF-IDF cache
├── TopicEmbedder_*.json           # Topic embedder cache  
├── MultiLayerEmbedder_*.json      # Multi-layer cache
└── layers/                         # Layer-specific caches
    ├── tfidf/
    ├── topic/
    └── multi_layer/
```

### Cache Management Commands

```bash
# Show cache statistics
text-vectorify --show-cache-stats

# List cached files
text-vectorify --list-cache-files

# Use custom cache directory
text-vectorify \
  --input data.jsonl \
  --input-field-main "title" \
  --process-method "MultiLayerEmbedder" \
  --output-cache-dir "./my_project_cache"
```

### Environment Variables

```bash
export OPENAI_API_KEY="your-openai-api-key"        # For OpenAI embeddings
export HUGGINGFACE_HUB_TOKEN="your-hf-token"       # For private HF models
export TEXT_VECTORIFY_CACHE_DIR="./custom_cache"   # Default cache directory
```

## 🔍 Performance Tips

1. **Use caching**: Enable caching to avoid recomputing embeddings
2. **Batch processing**: Process large files in chunks
3. **Model selection**: Choose appropriate model for your language and use case
4. **Field combination**: Combine relevant fields for better semantic representation
5. **Stdin processing**: Use stdin for pipeline integration and memory efficiency
6. **Default models**: Start with default models for quick prototyping, then customize as needed

## 🐛 Troubleshooting

### Common Issues

**Import Error**: Missing dependencies
```bash
pip install sentence-transformers transformers torch openai
```

**API Key Error**: Invalid or missing OpenAI API key
```bash
export OPENAI_API_KEY="your-valid-api-key"
```

**Memory Error**: Large models on limited RAM
- Use smaller models like `bge-small-zh-v1.5`
- Process files in smaller batches

**Cache Permission Error**: Insufficient cache directory permissions
```bash
chmod 755 ./cache
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Automated setup (recommended)
git clone https://github.com/changyy/py-text-vectorify.git
cd py-text-vectorify
./setup.sh

# Manual setup
git clone https://github.com/changyy/py-text-vectorify.git
cd py-text-vectorify
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Running Tests

**Using the test runner (recommended):**
```bash
# Quick validation (fastest)
python test_runner.py --quick

# Core functionality tests (no external dependencies)
python test_runner.py --core

# Integration tests (requires external models)
python test_runner.py --integration

# All tests
python test_runner.py --all

# Tests with coverage report
python test_runner.py --coverage

# Check dependencies
python test_runner.py --deps
```

**Direct pytest commands:**
```bash
# Quick smoke tests
pytest -m "quick or smoke" -v

# Core functionality
pytest -m "core" -v

# Integration tests
pytest -m "integration" -v

# Cache-related tests
pytest -m "cache" -v

# Embedder-specific tests
pytest -m "embedder" -v

# All tests
pytest -v

# With coverage
pytest --cov=text_vectorify --cov-report=html -v
```

### Development Tools

```bash
# Cache management
python tools/cache_tool.py --stats
python tools/cache_tool.py --clear-all

# Feature demonstration
python tools/demo_features.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📊 Benchmarks

| Model | Language | Dimension | Speed | Quality |
|-------|----------|-----------|-------|---------|
| OpenAI text-embedding-3-small | Multi | 1536 | Fast | Excellent |
| BGE-base-zh | Chinese | 768 | Medium | Excellent |
| SentenceBERT | Multi | 384 | Fast | Good |
| M3E-base | Chinese | 768 | Medium | Excellent |

## 🔗 Related Projects

- [sentence-transformers](https://github.com/UKPLab/sentence-transformers)
- [BGE](https://github.com/FlagOpen/FlagEmbedding)
- [M3E](https://github.com/wangyuxinwhy/uniem)

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/changyy/py-text-vectorify/issues)
- **Documentation**: [Full documentation](https://github.com/changyy/py-text-vectorify/wiki)

---

Made with ❤️ for the text analysis community
