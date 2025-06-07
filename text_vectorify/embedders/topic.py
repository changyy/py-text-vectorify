import logging
import numpy as np
from typing import Union, List, Dict, Any, Optional
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

class TopicEmbedder(BaseEmbedder):
    """Topic modeling embedder using BERTopic or LDA"""
    
    def __init__(self, model_name: str = "bertopic-default",
                 n_topics: int = 30,
                 method: str = "bertopic",
                 language: str = "chinese",
                 **kwargs):
        self.n_topics = n_topics
        self.method = method.lower()
        self.language = language
        self.fitted_model = None
        super().__init__(model_name, **kwargs)
    
    def load_model(self):
        """Load topic modeling library"""
        if self.method == "bertopic":
            try:
                from bertopic import BERTopic
                self.model_class = BERTopic
                logger.info(f"BERTopic loaded for topic modeling with {self.n_topics} topics")
            except ImportError:
                raise ImportError("Please install BERTopic package: pip install bertopic")
        elif self.method == "lda":
            try:
                from sklearn.decomposition import LatentDirichletAllocation
                from sklearn.feature_extraction.text import CountVectorizer
                self.model_class = LatentDirichletAllocation
                self.vectorizer_class = CountVectorizer
                logger.info(f"LDA loaded for topic modeling with {self.n_topics} topics")
            except ImportError:
                raise ImportError("Please install scikit-learn package: pip install scikit-learn")
        else:
            raise ValueError(f"Unsupported topic modeling method: {self.method}")
    
    def fit(self, texts: List[str]) -> 'TopicEmbedder':
        """Fit the topic model on a corpus"""
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"Fitting {self.method} topic model on {len(texts)} documents")
        
        if self.method == "bertopic":
            self.fitted_model = self.model_class(
                language=self.language,
                nr_topics=self.n_topics,
                verbose=False
            )
            # Fit and get topics
            topics, probabilities = self.fitted_model.fit_transform(texts)
            self.topic_probabilities = probabilities
            
        elif self.method == "lda":
            # Create vectorizer for LDA
            self.count_vectorizer = self.vectorizer_class(
                max_features=1000,
                stop_words='english' if self.language == 'english' else None,
                token_pattern=r'(?u)\b\w\w+\b'
            )
            
            # Transform texts to document-term matrix
            doc_term_matrix = self.count_vectorizer.fit_transform(texts)
            
            # Fit LDA model
            self.fitted_model = self.model_class(
                n_components=self.n_topics,
                random_state=42,
                max_iter=100
            )
            self.fitted_model.fit(doc_term_matrix)
            
            # Get topic probabilities for training documents
            self.topic_probabilities = self.fitted_model.transform(doc_term_matrix)
        
        logger.info(f"Topic model fitted successfully")
        return self
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text to topic probability vectors"""
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
            if self.fitted_model is None:
                # If not fitted, fit on the input texts
                logger.info("Topic model not fitted, fitting on input texts")
                self.fit(texts_to_process)
                # Use the probabilities from fitting
                vectors = self.topic_probabilities[-len(texts_to_process):]
            else:
                # Transform new texts
                try:
                    if self.method == "bertopic":
                        topics, probabilities = self.fitted_model.transform(texts_to_process)
                        vectors = probabilities
                    elif self.method == "lda":
                        doc_term_matrix = self.count_vectorizer.transform(texts_to_process)
                        vectors = self.fitted_model.transform(doc_term_matrix)
                except Exception as e:
                    logger.error(f"Error in topic modeling: {e}")
                    # Return uniform distribution as fallback
                    uniform_prob = 1.0 / self.n_topics
                    vectors = np.full((len(texts_to_process), self.n_topics), uniform_prob)
            
            # Fill results and save to cache
            process_idx = 0
            for i, result in enumerate(results):
                if result is None:
                    vector = vectors[process_idx].tolist()
                    results[i] = vector
                    self.save_to_cache(texts[i], vector)
                    process_idx += 1
        
        return results[0] if single_input else results
    
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get topic information"""
        if self.fitted_model is None:
            return []
        
        topics_info = []
        if self.method == "bertopic":
            topics = self.fitted_model.get_topics()
            for topic_id in topics:
                if topic_id != -1:  # Skip outlier topic
                    words = self.fitted_model.get_topic(topic_id)
                    topics_info.append({
                        'topic_id': topic_id,
                        'words': words,
                        'representation': [word for word, _ in words[:5]]
                    })
        elif self.method == "lda":
            try:
                feature_names = self.count_vectorizer.get_feature_names_out()
            except AttributeError:
                feature_names = self.count_vectorizer.get_feature_names()
            
            for topic_idx, topic in enumerate(self.fitted_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics_info.append({
                    'topic_id': topic_idx,
                    'words': [(word, topic[feature_names.tolist().index(word)]) for word in top_words],
                    'representation': top_words[:5]
                })
        
        return topics_info
    
    def get_topic_distribution_dim(self) -> int:
        """Get the dimension of topic distribution vectors"""
        return self.n_topics
