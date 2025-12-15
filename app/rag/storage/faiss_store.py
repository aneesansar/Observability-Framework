"""
FAISS Vector Store.

Local file-based vector storage using FAISS.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from ..config import rag_settings
from ..preprocessing.document_loader import DocumentLoader

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    Features:
    - Persistent storage to disk
    - In-memory caching
    - Automatic reindexing
    """
    
    _instance: Optional["FAISSVectorStore"] = None
    
    def __new__(cls) -> "FAISSVectorStore":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store."""
        if self._initialized:
            return
        
        self._index: Optional[faiss.Index] = None
        self._metadata: List[Dict[str, Any]] = []
        self._initialized = True
    
    @property
    def is_loaded(self) -> bool:
        """Check if index is loaded."""
        return self._index is not None and len(self._metadata) > 0
    
    @property
    def document_count(self) -> int:
        """Get number of indexed documents."""
        return len(self._metadata)
    
    def load(self) -> bool:
        """
        Load index from disk.
        
        Returns:
            True if loaded successfully
        """
        index_path = rag_settings.index_path
        meta_path = rag_settings.meta_path
        
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            logger.debug("No persisted index found")
            return False
        
        try:
            self._index = faiss.read_index(index_path)
            
            self._metadata = []
            with open(meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self._metadata.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Loaded FAISS index with {len(self._metadata)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self._index = None
            self._metadata = []
            return False
    
    def save(self) -> bool:
        """
        Save index to disk.
        
        Returns:
            True if saved successfully
        """
        if not self.is_loaded:
            return False
        
        try:
            # Ensure directories exist
            os.makedirs(rag_settings.data_dir, exist_ok=True)
            
            # Save index
            faiss.write_index(self._index, rag_settings.index_path)
            
            # Save metadata
            with open(rag_settings.meta_path, "w", encoding="utf-8") as f:
                for meta in self._metadata:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            
            logger.info(f"Saved FAISS index with {len(self._metadata)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    def index_documents(
        self,
        texts: List[str],
        metadata: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """
        Index documents with their embeddings.
        
        Args:
            texts: Document texts
            metadata: Document metadata
            embeddings: Pre-computed embeddings
            
        Returns:
            Number of documents indexed
        """
        if len(texts) == 0:
            return 0
        
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._metadata = metadata
        
        return len(texts)
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[List[float], List[int], List[Dict[str, Any]]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            Tuple of (scores, indices, metadata)
        """
        if not self.is_loaded:
            return [], [], []
        
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        k = min(top_k, len(self._metadata))
        scores, indices = self._index.search(query_vector.astype(np.float32), k)
        
        result_scores = []
        result_indices = []
        result_metadata = []
        
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx == -1:
                continue
            result_scores.append(score)
            result_indices.append(idx)
            result_metadata.append(self._metadata[idx])
        
        return result_scores, result_indices, result_metadata
    
    def clear(self) -> None:
        """Clear the index."""
        self._index = None
        self._metadata = []
        
        # Remove persisted files
        for path in [rag_settings.index_path, rag_settings.meta_path]:
            if os.path.exists(path):
                os.remove(path)


def get_faiss_store() -> FAISSVectorStore:
    """Get the FAISS store instance."""
    return FAISSVectorStore()


def reindex(documents_dir: str = None) -> Dict[str, int]:
    """
    Reindex all documents.
    
    Args:
        documents_dir: Directory containing documents
        
    Returns:
        Dict with indexing stats
    """
    from ..core.embeddings import get_embedding_service
    
    documents_dir = documents_dir or rag_settings.documents_dir
    
    # Ensure directories exist
    os.makedirs(documents_dir, exist_ok=True)
    os.makedirs(rag_settings.data_dir, exist_ok=True)
    
    # Load documents
    loader = DocumentLoader()
    texts, metadata = loader.load_directory(documents_dir)
    
    if not texts:
        store = get_faiss_store()
        store.clear()
        return {"documents": 0, "dim": 0}
    
    # Generate embeddings
    embedding_service = get_embedding_service()
    embeddings = embedding_service.encode(texts, normalize=True)
    
    # Index
    store = get_faiss_store()
    count = store.index_documents(texts, metadata, embeddings)
    store.save()
    
    return {
        "documents": count,
        "dim": int(embeddings.shape[1]),
    }
