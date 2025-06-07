import logging
import numpy as np
import re
from typing import Union, List, Dict, Any, Optional, Callable
from .base import BaseEmbedder

logger = logging.getLogger(__name__)

class TFIDFEmbedder(BaseEmbedder):
    """TF-IDF embedder for text vectorization using scikit-learn with Chinese support"""
    
    def __init__(self, model_name: str = "tfidf-sklearn", 
                 max_features: int = 1000,
                 ngram_range: tuple = (1, 2),
                 stop_words: Optional[Union[str, List[str]]] = None,
                 chinese_tokenizer: str = "spacy",  # 預設使用 spacy（主動維護且功能強大）
                 custom_tokenizer: Optional[Callable] = None,
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0,
                 **kwargs):
        self.max_features = max_features
        # Convert ngram_range from list to tuple if needed (for JSON config compatibility)
        self.ngram_range = tuple(ngram_range) if isinstance(ngram_range, list) else ngram_range
        self.stop_words = stop_words
        self.chinese_tokenizer = chinese_tokenizer
        self.custom_tokenizer = custom_tokenizer
        self.min_df = min_df
        self.max_df = max_df
        self.fitted_vectorizer = None
        self._tokenizer_cache = {}  # 緩存已初始化的分詞器
        super().__init__(model_name, **kwargs)
    
    def _check_tokenizer_availability(self, tokenizer_name: str) -> bool:
        """檢查指定分詞器是否可用"""
        if tokenizer_name == "spacy":
            try:
                import spacy
                # 嘗試加載中文模型
                spacy.load("zh_core_web_sm")
                return True
            except (ImportError, OSError):
                return False
        elif tokenizer_name == "jieba":
            try:
                import jieba
                return True
            except ImportError:
                return False
        elif tokenizer_name == "pkuseg":
            try:
                import pkuseg
                return True
            except ImportError:
                return False
        return False
    
    def _get_best_available_tokenizer(self) -> str:
        """獲取最佳可用分詞器，優先使用用戶指定的分詞器"""
        # 如果用戶明確指定了分詞器，優先檢查它是否可用
        if hasattr(self, 'chinese_tokenizer') and self.chinese_tokenizer:
            if self._check_tokenizer_availability(self.chinese_tokenizer):
                return self.chinese_tokenizer
            else:
                logger.warning(f"Specified tokenizer '{self.chinese_tokenizer}' not available, falling back to alternatives")
        
        # 按默認優先級順序查找：spacy > jieba > pkuseg（spacy 主動維護且功能更豐富）
        preferred_order = ["spacy", "jieba", "pkuseg"]
        
        for tokenizer in preferred_order:
            if self._check_tokenizer_availability(tokenizer):
                if tokenizer != getattr(self, 'chinese_tokenizer', None):
                    logger.info(f"Using {tokenizer} tokenizer")
                return tokenizer
        
        logger.warning("No Chinese tokenizers available. Install one of: jieba, spacy, or pkuseg")
        return "none"
    
    def _detect_chinese(self, text: str) -> bool:
        """Detect if text contains Chinese characters"""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(chinese_pattern.search(text))
    
    def _preprocess_texts(self, texts: List[str]) -> List[str]:
        """使用智能分詞器預處理文本"""
        processed_texts = []
        
        # 獲取最佳可用分詞器
        best_tokenizer = self._get_best_available_tokenizer()
        if best_tokenizer == "none":
            return texts  # 沒有分詞器可用，返回原文本
        
        tokenizer_func = self._get_tokenizer_function(best_tokenizer)
        
        for text in texts:
            if self._detect_chinese(text):
                processed_texts.append(tokenizer_func(text))
            else:
                processed_texts.append(text)
        
        return processed_texts
    
    def _get_tokenizer_function(self, tokenizer_name: str):
        """獲取分詞器函數"""
        if tokenizer_name in self._tokenizer_cache:
            return self._tokenizer_cache[tokenizer_name]
        
        if tokenizer_name == "spacy":
            func = self._spacy_tokenizer
        elif tokenizer_name == "pkuseg":
            func = self._pkuseg_tokenizer  
        elif tokenizer_name == "jieba":
            func = self._jieba_tokenizer
        else:
            func = lambda text: text
        
        self._tokenizer_cache[tokenizer_name] = func
        return func
    
    def load_model(self):
        """Load TF-IDF vectorizer with intelligent tokenizer selection"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 設置分詞器
            tokenizer = self.custom_tokenizer
            if tokenizer is None:
                # 使用智能分詞器選擇，無需額外設置
                tokenizer = None
            
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words=self.stop_words,
                tokenizer=tokenizer,
                token_pattern=r'(?u)\b\w+\b',  # Support Chinese characters
                min_df=self.min_df,
                max_df=self.max_df,
                lowercase=True,
                analyzer='word'
            )
            
            best_tokenizer = self._get_best_available_tokenizer()
            logger.info(f"TF-IDF vectorizer created with max_features={self.max_features}, "
                       f"ngram_range={self.ngram_range}, using {best_tokenizer} tokenizer")
                       
        except ImportError:
            raise ImportError("Please install scikit-learn package: pip install scikit-learn")
    
    def fit(self, texts: List[str]) -> 'TFIDFEmbedder':
        """Fit the TF-IDF vectorizer on a corpus"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts for Chinese if needed
        processed_texts = self._preprocess_texts(texts)
        
        logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
        self.fitted_vectorizer = self.vectorizer.fit(processed_texts)
        return self
    
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Encode text to TF-IDF vectors"""
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
            # Preprocess texts for Chinese if needed
            processed_texts = self._preprocess_texts(texts_to_process)
            
            if self.fitted_vectorizer is None:
                # If not fitted, fit on the input texts
                logger.info("TF-IDF vectorizer not fitted, fitting on input texts")
                self.fitted_vectorizer = self.vectorizer.fit(processed_texts)
            
            try:
                # Transform texts to TF-IDF vectors
                tfidf_matrix = self.fitted_vectorizer.transform(processed_texts)
                vectors = tfidf_matrix.toarray()  # Convert sparse matrix to dense
                
                # Fill results and save to cache
                process_idx = 0
                for i, result in enumerate(results):
                    if result is None:
                        vector = vectors[process_idx].tolist()
                        results[i] = vector
                        self.save_to_cache(texts[i], vector)
                        process_idx += 1
                        
            except Exception as e:
                logger.error(f"Error in TF-IDF encoding: {e}")
                # Return zero vectors as fallback
                vector_dim = self.max_features
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = [0.0] * vector_dim
        
        return results[0] if single_input else results
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from fitted vectorizer"""
        if self.fitted_vectorizer is None:
            return []
        try:
            return self.fitted_vectorizer.get_feature_names_out().tolist()
        except AttributeError:
            # For older sklearn versions
            return self.fitted_vectorizer.get_feature_names()
    
    def get_vocabulary_size(self) -> int:
        """Get vocabulary size"""
        if self.fitted_vectorizer is None:
            return 0
        return len(self.fitted_vectorizer.vocabulary_)
    
    def _get_chinese_tokenizer(self):
        """獲取中文分詞器"""
        if self.chinese_tokenizer == "jieba":
            return self._jieba_tokenizer
        elif self.chinese_tokenizer == "spacy":
            return self._spacy_tokenizer
        elif self.chinese_tokenizer == "pkuseg":
            return self._pkuseg_tokenizer
        else:
            logger.warning(f"Unknown tokenizer: {self.chinese_tokenizer}, fallback to jieba")
            return self._jieba_tokenizer
    
    def _jieba_tokenizer(self, text: str) -> str:
        """使用 jieba 分詞"""
        try:
            import jieba
            words = jieba.cut(text, cut_all=False)
            return ' '.join(words)
        except ImportError:
            logger.warning("jieba not available, install with: pip install jieba")
            return text
        except Exception as e:
            logger.warning(f"Error in jieba tokenization: {e}")
            return text
    
    def _spacy_tokenizer(self, text: str) -> str:
        """使用 spaCy 分詞 (推薦選項)"""
        try:
            import spacy
            if not hasattr(self, '_spacy_nlp'):
                try:
                    self._spacy_nlp = spacy.load("zh_core_web_sm")
                except OSError:
                    # 嘗試其他中文模型
                    for model_name in ["zh_core_web_md", "zh_core_web_lg", "zh_core_web_trf"]:
                        try:
                            self._spacy_nlp = spacy.load(model_name)
                            logger.info(f"Using spaCy model: {model_name}")
                            break
                        except OSError:
                            continue
                    else:
                        raise OSError("No Chinese spaCy model found")
            
            doc = self._spacy_nlp(text)
            return ' '.join([token.text for token in doc])
            
        except ImportError:
            logger.warning("spaCy not available. Install with: pip install spacy && python -m spacy download zh_core_web_sm")
            return self._jieba_tokenizer(text)
        except OSError:
            logger.warning("No Chinese spaCy model found. Download with: python -m spacy download zh_core_web_sm")
            return self._jieba_tokenizer(text)
        except Exception as e:
            logger.warning(f"Error in spaCy tokenization: {e}")
            return self._jieba_tokenizer(text)
    
    def _pkuseg_tokenizer(self, text: str) -> str:
        """使用 pkuseg 分詞 (學術級選項)"""
        try:
            import pkuseg
            if not hasattr(self, '_pkuseg_seg'):
                self._pkuseg_seg = pkuseg.pkuseg()
            words = self._pkuseg_seg.cut(text)
            return ' '.join(words)
        except ImportError:
            logger.warning("pkuseg not available, install with: pip install pkuseg")
            return self._jieba_tokenizer(text)
        except Exception as e:
            logger.warning(f"Error in pkuseg tokenization: {e}")
            return self._jieba_tokenizer(text)
