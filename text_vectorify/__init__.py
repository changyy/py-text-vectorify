# text_vectorify package
from .vectorify import TextVectorify
from .factory import EmbedderFactory
from .embedders.base import BaseEmbedder
from .embedders.multi_layer import MultiLayerEmbedder

__version__ = "1.2.0"
__all__ = [
    "TextVectorify", 
    "EmbedderFactory", 
    "BaseEmbedder", 
    "MultiLayerEmbedder"
]
