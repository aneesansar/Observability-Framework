"""
Embedding Service.

Manages sentence embeddings for semantic search.
"""

import logging
from typing import List, Optional, Union

import numpy as np

from ..config import rag_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses sentence-transformers for embedding generation.
    Implements singleton pattern for model reuse.
    """
    
    _instance: Optional["EmbeddingService"] = None
    _model = None
    
    def __new__(cls) -> "EmbeddingService":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service."""
        if self._initialized:
            return
        
        self._model_name = rag_settings.embed_model
        self._initialized = True
    
    def _get_model(self):
        """Lazy load the embedding model."""
        if EmbeddingService._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                EmbeddingService._model = SentenceTransformer(self._model_name)
                logger.info(f"Loaded embedding model: {self._model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return EmbeddingService._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text or list of texts to encode
            normalize: Whether to L2-normalize embeddings
            convert_to_numpy: Convert to numpy array
            
        Returns:
            Numpy array of embeddings
        """
        model = self._get_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = model.encode(
            texts,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
        )
        
        return embeddings.astype(np.float32)
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a query for retrieval.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding as numpy array
        """
        return self.encode(query, normalize=True)[0]
    
    def compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        try:
            from sentence_transformers import util
            
            model = self._get_model()
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            
            return float(util.cos_sim(emb1, emb2).item())
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0
    
    def compute_max_similarity(
        self,
        text: str,
        candidates: List[str],
    ) -> float:
        """
        Compute maximum similarity between text and candidates.
        
        Args:
            text: Text to compare
            candidates: List of candidate texts
            
        Returns:
            Maximum similarity score
        """
        if not candidates:
            return 0.0
        
        try:
            from sentence_transformers import util
            
            model = self._get_model()
            text_emb = model.encode(text, convert_to_tensor=True)
            
            max_sim = 0.0
            for candidate in candidates:
                cand_emb = model.encode(str(candidate), convert_to_tensor=True)
                sim = util.cos_sim(text_emb, cand_emb).item()
                max_sim = max(max_sim, sim)
            
            return float(max_sim)
        except Exception as e:
            logger.warning(f"Max similarity computation failed: {e}")
            return 0.0
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        model = self._get_model()
        return model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name


def get_embedding_service() -> EmbeddingService:
    """Get the embedding service instance."""
    return EmbeddingService()


# Convenience alias
embedding_service = get_embedding_service()
