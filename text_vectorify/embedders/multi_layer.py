import logging
import json
import numpy as np
from typing import Union, List, Dict, Any, Optional
from pathlib import Path
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

class MultiLayerEmbedder(BaseEmbedder):
    """Multi-layer embedder that combines different types of embeddings"""
    
    def __init__(self, model_name: str = "multi-layer-default",
                 embedder_configs: Optional[List[Dict]] = None,
                 fusion_method: str = "concat",
                 weights: Optional[List[float]] = None,
                 normalize: bool = True,
                 **kwargs):
        self.embedder_configs = embedder_configs or []
        self.fusion_method = fusion_method.lower()
        self.weights = weights
        self.normalize = normalize
        self.embedders = {}
        self.layer_names = []
        self.fitted_embedders = set()  # Track which embedders need fitting
        super().__init__(model_name, **kwargs)
    
    def _convert_config_types(self, config: Dict) -> Dict:
        """Convert config types from JSON (e.g., list to tuple for ngram_range)"""
        converted_config = config.copy()
        
        # Convert ngram_range from list to tuple if needed
        if 'ngram_range' in converted_config and isinstance(converted_config['ngram_range'], list):
            converted_config['ngram_range'] = tuple(converted_config['ngram_range'])
        
        return converted_config
    
    def load_model(self):
        """Load all configured embedders"""
        logger.info(f"Loading multi-layer embedder with {len(self.embedder_configs)} layers")
        
        for config in self.embedder_configs:
            layer_name = config.get('name', f"layer_{len(self.embedders)}")
            embedder_type = config['type']
            embedder_config = config.get('config', {})
            
            # Convert config types (e.g., list to tuple)
            embedder_config = self._convert_config_types(embedder_config)
            
            # Create cache directory for this layer
            layer_cache_dir = self.cache_dir / "layers" / layer_name
            layer_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create embedder instance
            try:
                # 使用延遲導入避免循環導入
                from ..factory import EmbedderFactory
                embedder = EmbedderFactory.create_embedder(
                    embedder_type,
                    cache_dir=str(layer_cache_dir),
                    **embedder_config
                )
                self.embedders[layer_name] = embedder
                self.layer_names.append(layer_name)
                
                # Check if this embedder needs fitting (TF-IDF, Topic models)
                if hasattr(embedder, 'fit'):
                    self.fitted_embedders.add(layer_name)
                
                logger.info(f"Loaded layer '{layer_name}' with type '{embedder_type}'")
            except Exception as e:
                logger.error(f"Failed to load layer '{layer_name}': {e}")
                raise
        
        # Set up fusion weights
        if self.weights is None:
            # Equal weights for all layers
            self.weights = [1.0 / len(self.embedders)] * len(self.embedders)
        elif len(self.weights) != len(self.embedders):
            logger.warning(f"Weight count ({len(self.weights)}) doesn't match layer count ({len(self.embedders)}), using equal weights")
            self.weights = [1.0 / len(self.embedders)] * len(self.embedders)
        
        logger.info(f"Multi-layer embedder loaded with fusion method: {self.fusion_method}")
    
    @classmethod
    def from_config_file(cls, config_path: str, **kwargs) -> 'MultiLayerEmbedder':
        """Create MultiLayerEmbedder from JSON config file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        embedder_configs = config.get('layers', [])
        fusion_config = config.get('fusion', {})
        
        return cls(
            embedder_configs=embedder_configs,
            fusion_method=fusion_config.get('method', 'concat'),
            weights=[layer.get('weight', 1.0) for layer in embedder_configs],
            normalize=fusion_config.get('normalize', True),
            **kwargs
        )
    
    def fit(self, texts: List[str]) -> 'MultiLayerEmbedder':
        """Fit embedders that require fitting (TF-IDF, Topic models)"""
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Fitting multi-layer embedder on {len(texts)} documents")
        
        for layer_name in self.fitted_embedders:
            embedder = self.embedders[layer_name]
            if hasattr(embedder, 'fit'):
                logger.info(f"Fitting layer '{layer_name}'")
                embedder.fit(texts)
        
        return self
    
    def encode_layers(self, texts: Union[str, List[str]]) -> Dict[str, Union[List[float], List[List[float]]]]:
        """Encode texts with all layers separately, returning individual layer results"""
        if isinstance(texts, str):
            texts = [texts]
        
        layer_results = {}
        
        for layer_name, embedder in self.embedders.items():
            try:
                vectors = embedder.encode(texts)
                layer_results[layer_name] = vectors
                logger.debug(f"Layer '{layer_name}' encoded {len(texts)} texts")
            except Exception as e:
                logger.error(f"Error encoding with layer '{layer_name}': {e}")
                # Create zero vectors as fallback
                if hasattr(embedder, 'max_features'):
                    dim = embedder.max_features
                elif hasattr(embedder, 'n_topics'):
                    dim = embedder.n_topics
                else:
                    dim = 384  # Default dimension
                
                zero_vector = [0.0] * dim
                layer_results[layer_name] = [zero_vector] * len(texts)
        
        return layer_results
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text using multi-layer fusion"""
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        # Check cache first
        results = []
        texts_to_process = []
        
        for text in texts:
            cached_vector = self.get_from_cache(text)
            if cached_vector:
                results.append(cached_vector)
            else:
                texts_to_process.append(text)
                results.append(None)
        
        # Process uncached texts
        if texts_to_process:
            # Get layer embeddings
            layer_results = self.encode_layers(texts_to_process)
            
            # Fuse the embeddings
            fused_vectors = self._fuse_vectors(layer_results, texts_to_process)
            
            # Fill results and save to cache
            process_idx = 0
            for i, result in enumerate(results):
                if result is None:
                    vector = fused_vectors[process_idx]
                    results[i] = vector
                    self.save_to_cache(texts[i], vector)
                    process_idx += 1
        
        return results[0] if single_input else results
    
    def _fuse_vectors(self, layer_results: Dict[str, List[List[float]]], texts: List[str]) -> List[List[float]]:
        """Fuse vectors from different layers"""
        if not layer_results:
            return [[0.0] for _ in texts]
        
        num_texts = len(texts)
        
        if self.fusion_method == "concat":
            return self._fuse_concat(layer_results, num_texts)
        elif self.fusion_method == "weighted":
            return self._fuse_weighted(layer_results, num_texts)
        elif self.fusion_method == "attention":
            return self._fuse_attention(layer_results, num_texts)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using concatenation")
            return self._fuse_concat(layer_results, num_texts)
    
    def _fuse_concat(self, layer_results: Dict[str, List[List[float]]], num_texts: int) -> List[List[float]]:
        """Concatenate all layer vectors"""
        fused_vectors = []
        
        for text_idx in range(num_texts):
            concatenated = []
            for layer_name in self.layer_names:
                if layer_name in layer_results:
                    layer_vector = layer_results[layer_name][text_idx]
                    concatenated.extend(layer_vector)
            fused_vectors.append(concatenated)
        
        return fused_vectors
    
    def _fuse_weighted(self, layer_results: Dict[str, List[List[float]]], num_texts: int) -> List[List[float]]:
        """Weighted combination of layer vectors (requires same dimensions)"""
        # First, standardize all vectors to the same dimension
        standardized_results = self._standardize_dimensions(layer_results, num_texts)
        
        fused_vectors = []
        for text_idx in range(num_texts):
            weighted_sum = None
            total_weight = 0.0
            
            for i, layer_name in enumerate(self.layer_names):
                if layer_name in standardized_results:
                    layer_vector = np.array(standardized_results[layer_name][text_idx])
                    weight = self.weights[i] if i < len(self.weights) else 1.0
                    
                    if weighted_sum is None:
                        weighted_sum = weight * layer_vector
                    else:
                        weighted_sum += weight * layer_vector
                    total_weight += weight
            
            if weighted_sum is not None and total_weight > 0:
                weighted_sum = weighted_sum / total_weight
                fused_vectors.append(weighted_sum.tolist())
            else:
                fused_vectors.append([0.0] * 384)  # Default dimension
        
        return fused_vectors
    
    def _fuse_attention(self, layer_results: Dict[str, List[List[float]]], num_texts: int) -> List[List[float]]:
        """Attention-based fusion (simplified version)"""
        # For now, implement as weighted fusion with learned attention weights
        # In a full implementation, this would include attention mechanisms
        standardized_results = self._standardize_dimensions(layer_results, num_texts)
        
        # Simple attention: compute similarity between layers for each text
        fused_vectors = []
        for text_idx in range(num_texts):
            # Collect all layer vectors for this text
            layer_vectors = []
            for layer_name in self.layer_names:
                if layer_name in standardized_results:
                    layer_vectors.append(np.array(standardized_results[layer_name][text_idx]))
            
            if not layer_vectors:
                fused_vectors.append([0.0] * 384)
                continue
            
            # Simple attention mechanism: average with dynamic weights
            layer_matrix = np.array(layer_vectors)
            # Compute attention weights based on vector norms
            norms = np.linalg.norm(layer_matrix, axis=1)
            attention_weights = norms / np.sum(norms) if np.sum(norms) > 0 else np.ones(len(norms)) / len(norms)
            
            # Weighted combination
            fused_vector = np.average(layer_matrix, axis=0, weights=attention_weights)
            fused_vectors.append(fused_vector.tolist())
        
        return fused_vectors
    
    def _standardize_dimensions(self, layer_results: Dict[str, List[List[float]]], num_texts: int) -> Dict[str, List[List[float]]]:
        """Standardize all vectors to the same dimension"""
        if not layer_results:
            return layer_results
        
        # Find the maximum dimension across all layers
        max_dim = 0
        for vectors in layer_results.values():
            if vectors:
                max_dim = max(max_dim, len(vectors[0]))
        
        if max_dim == 0:
            return layer_results
        
        standardized_results = {}
        
        for layer_name, vectors in layer_results.items():
            standardized_vectors = []
            
            for vector in vectors:
                current_dim = len(vector)
                
                if current_dim == max_dim:
                    # No change needed
                    standardized_vector = vector[:]
                elif current_dim < max_dim:
                    # Pad with zeros
                    standardized_vector = vector + [0.0] * (max_dim - current_dim)
                else:
                    # Truncate (should not happen with our design)
                    standardized_vector = vector[:max_dim]
                
                # Apply normalization if requested
                if self.normalize:
                    norm = np.linalg.norm(standardized_vector)
                    if norm > 0:
                        standardized_vector = (np.array(standardized_vector) / norm).tolist()
                
                standardized_vectors.append(standardized_vector)
            
            standardized_results[layer_name] = standardized_vectors
        
        return standardized_results
    
    def get_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all layers"""
        layer_info = {}
        for layer_name, embedder in self.embedders.items():
            info = {
                'type': embedder.__class__.__name__,
                'model_name': embedder.model_name,
            }
            
            # Add specific info for different embedder types
            if hasattr(embedder, 'max_features'):
                info['max_features'] = embedder.max_features
            if hasattr(embedder, 'n_topics'):
                info['n_topics'] = embedder.n_topics
                if hasattr(embedder, 'get_topics'):
                    info['topics'] = embedder.get_topics()
            
            layer_info[layer_name] = info
        
        return layer_info
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        config = {
            'layers': self.embedder_configs,
            'fusion': {
                'method': self.fusion_method,
                'weights': self.weights,
                'normalize': self.normalize
            }
        }
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Configuration saved to {config_path}")
