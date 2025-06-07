# Configuration Guide

This directory contains various configuration file examples to help you choose the appropriate configuration based on different usage scenarios.

## Chinese Tokenizer Selection Guide

### Recommended (Default)
- **spaCy** (Default) - Most accurate Chinese tokenization, suitable for most application scenarios
- Installation: `pip install spacy && python -m spacy download zh_core_web_sm`

### Lightweight Option  
- **jieba** - Lightweight, simple installation, suitable for resource-constrained environments
- Installation: `pip install jieba`

### Domain-Specific
- **pkuseg** - Suitable for domain-specific text, customizable dictionary
- Installation: `pip install pkuseg`

## Configuration Files

### Single Embedder Configurations

1. **`tfidf_spacy_example.json`** - TF-IDF configuration using spaCy tokenization
   - Use case: General Chinese text analysis
   - Features: High-precision tokenization + stopwords filtering
   - Recommended for: News articles, academic papers, formal documents

2. **`tfidf_jieba_example.json`** - TF-IDF configuration using jieba tokenization  
   - Use case: Rapid deployment, resource-constrained environments
   - Features: Lightweight + fast processing
   - Recommended for: Social media text, comment data

3. **`tfidf_pkuseg_example.json`** - TF-IDF configuration using pkuseg tokenization
   - Use case: Domain-specific text processing
   - Features: Domain-adaptive tokenization
   - Recommended for: Professional text in medical, legal, financial domains

### Multi-Layer Embedder Configurations

4. **`multi_layer_simple.json`** - Basic multi-layer configuration
   - Combination: TF-IDF (spaCy) + Topic Modeling
   - Use case: Entry-level multi-dimensional text analysis
   - Computational cost: Low

5. **`multi_layer_spacy_recommended.json`** - Recommended multi-layer configuration
   - Combination: TF-IDF (spaCy) + BGE semantic embedding + Topic modeling
   - Use case: High-quality Chinese text analysis
   - Computational cost: Medium

6. **`multi_layer_jieba_lightweight.json`** - Lightweight multi-layer configuration
   - Combination: TF-IDF (jieba) + M3E semantic embedding
   - Use case: Rapid prototyping, resource-constrained environments
   - Computational cost: Low

7. **`multi_layer_advanced_research.json`** - Advanced research configuration
   - Combination: Advanced TF-IDF + Large semantic models + BERTopic
   - Use case: Academic research, business intelligence analysis
   - Computational cost: High

### Existing Configurations

8. **`multi_layer_example.json`** - Composite embedder example
   - Combination: TF-IDF + BGE + Topic (weighted fusion)
   - Use case: Applications balancing precision and efficiency

9. **`multi_layer_1500_articles.json`** - Large-scale article processing configuration
   - Combination: Statistical + Multilingual semantic + Topic modeling
   - Use case: Large-scale document collection analysis

## Usage

### Command Line Usage

```bash
# Use specific configuration file
python -m text_vectorify --config configs/tfidf_spacy_example.json --input data.jsonl --output vectors.pkl

# Or specify tokenizer directly
python -m text_vectorify --chinese-tokenizer spacy --input data.jsonl --output vectors.pkl
```

### Programmatic Usage

```python
import json
from text_vectorify.embedders import TFIDFEmbedder

# Load configuration
with open('configs/tfidf_spacy_example.json', 'r') as f:
    config = json.load(f)

# Create embedder
embedder = TFIDFEmbedder(**config['config'])
```

## Performance Comparison

| Tokenizer | Accuracy | Speed | Memory Usage | Installation Complexity |
|-----------|----------|-------|--------------|-------------------------|
| spaCy     | Highest  | Medium| Higher       | Medium                  |
| jieba     | Good     | Fastest| Lowest      | Simplest                |
| pkuseg    | Good     | Slower | Medium      | Simple                  |

## Custom Configuration

You can create your own configuration files based on these examples:

1. Copy the configuration file closest to your needs
2. Adjust parameters according to your data characteristics
3. Test different tokenizer options
4. Adjust model size according to performance requirements

## Troubleshooting

If you encounter tokenizer-related issues:

1. **spaCy model not found**: Run `python -m spacy download zh_core_web_sm`
2. **Poor jieba tokenization**: Consider adding custom dictionary
3. **pkuseg installation issues**: Ensure sufficient memory and disk space

## Recommended Workflow

1. **Development phase**: Use `tfidf_jieba_example.json` for quick testing
2. **Testing phase**: Use `multi_layer_spacy_recommended.json` to evaluate effectiveness
3. **Production phase**: Choose the most suitable configuration based on resource and accuracy requirements
4. **Research phase**: Use `multi_layer_advanced_research.json` for best results
