"""
Document Retriever.

Handles document retrieval from vector stores.
"""

import logging
from typing import Dict, List, Optional

from ..config import rag_settings
from ..models.documents import DocumentMetadata, RetrievalResult
from .embeddings import get_embedding_service

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Retrieves relevant documents using vector similarity search.
    
    Supports multiple backends:
    - FAISS (local file-based)
    - MongoDB (with vector embeddings)
    """
    
    _instance: Optional["DocumentRetriever"] = None
    
    def __new__(cls) -> "DocumentRetriever":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the retriever."""
        if self._initialized:
            return
        
        self._embedding_service = get_embedding_service()
        self._faiss_store = None
        self._mongo_store = None
        self._initialized = True
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        min_score: float = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or rag_settings.default_top_k
        min_score = min_score or rag_settings.similarity_threshold
        
        # Try MongoDB first if configured
        if rag_settings.mongo_uri:
            try:
                results = self._retrieve_from_mongo(query, top_k)
                if results:
                    return [r for r in results if r.score >= min_score]
            except Exception as e:
                logger.debug(f"MongoDB retrieval failed, falling back to FAISS: {e}")
        
        # Fall back to FAISS
        results = self._retrieve_from_faiss(query, top_k)
        return [r for r in results if r.score >= min_score]
    
    def _retrieve_from_faiss(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Retrieve from FAISS vector store.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of RetrievalResult
        """
        from ..storage.faiss_store import get_faiss_store
        
        store = get_faiss_store()
        if not store.is_loaded:
            store.load()
        
        if not store.is_loaded:
            return []
        
        # Encode query
        query_vec = self._embedding_service.encode_query(query)
        
        # Search
        scores, indices, metadata_list = store.search(query_vec, top_k)
        
        results = []
        for score, idx, meta in zip(scores, indices, metadata_list):
            if idx == -1:
                continue
            results.append(RetrievalResult(
                metadata=DocumentMetadata.from_dict(meta),
                score=float(score),
            ))
        
        return results
    
    def _retrieve_from_mongo(
        self,
        query: str,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Retrieve from MongoDB vector store.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of RetrievalResult
        """
        from ..storage.mongo_store import get_mongo_store
        
        store = get_mongo_store()
        if not store.is_available:
            return []
        
        # Encode query
        query_vec = self._embedding_service.encode_query(query)
        
        # Search
        results = store.search(query_vec, top_k)
        
        return [
            RetrievalResult(
                metadata=DocumentMetadata.from_dict(r),
                score=r.get("score", 0.0),
            )
            for r in results
        ]
    
    def build_context(
        self,
        results: List[RetrievalResult],
    ) -> str:
        """
        Build context string from retrieval results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        lines = []
        
        for result in results:
            meta = result.metadata
            
            # Build structured block
            block = []
            
            title = meta.title or meta.trip_name or "Unknown"
            block.append(f"Title: {title}")
            
            if meta.provider:
                block.append(f"Provider: {meta.provider}")
            if meta.trip_name:
                block.append(f"Trip: {meta.trip_name}")
            if meta.country:
                block.append(f"Country: {meta.country}")
            if meta.duration:
                block.append(f"Duration: {meta.duration}")
            if meta.trip_type:
                block.append(f"Type: {meta.trip_type}")
            if meta.date_summary:
                block.append(f"Dates (sample): {meta.date_summary}")
            
            if not block[1:]:  # Only title
                block.append(meta.preview or "")
            
            lines.append("\n".join(block))
        
        return "\n\n".join(lines)
    
    def get_context_texts(
        self,
        results: List[RetrievalResult],
    ) -> List[str]:
        """
        Get list of context texts from results.
        
        Args:
            results: List of retrieval results
            
        Returns:
            List of text strings
        """
        texts = []
        for result in results:
            meta = result.metadata
            text = meta.preview or ""
            if meta.date_summary:
                text += f"\n{meta.date_summary}"
            texts.append(text)
        return texts


def get_retriever() -> DocumentRetriever:
    """Get the document retriever instance."""
    return DocumentRetriever()


def retrieve_documents(
    query: str,
    top_k: int = None,
) -> List[RetrievalResult]:
    """
    Convenience function to retrieve documents.
    
    Args:
        query: Search query
        top_k: Number of results
        
    Returns:
        List of RetrievalResult
    """
    retriever = get_retriever()
    return retriever.retrieve(query, top_k)
