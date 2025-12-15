"""
MongoDB Vector Store.

MongoDB-based vector storage with embedding support.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..config import rag_settings

logger = logging.getLogger(__name__)


class MongoVectorStore:
    """
    MongoDB-based vector store for document embeddings.
    
    Features:
    - Persistent storage in MongoDB
    - In-memory caching with TTL
    - Vector similarity search
    """
    
    _instance: Optional["MongoVectorStore"] = None
    
    def __new__(cls) -> "MongoVectorStore":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the vector store."""
        if self._initialized:
            return
        
        self._client = None
        self._vectors: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._last_load: float = 0.0
        self._initialized = True
    
    @property
    def is_available(self) -> bool:
        """Check if MongoDB is available."""
        if not rag_settings.mongo_uri:
            return False
        
        try:
            from pymongo import MongoClient
            return True
        except ImportError:
            return False
    
    def _get_client(self):
        """Get MongoDB client."""
        if self._client is None:
            try:
                from pymongo import MongoClient
                self._client = MongoClient(rag_settings.mongo_uri)
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {e}")
                return None
        return self._client
    
    def _get_collection(self):
        """Get the trips collection."""
        client = self._get_client()
        if client is None:
            return None
        return client[rag_settings.mongo_db][rag_settings.mongo_collection]
    
    def _load_cache(self) -> bool:
        """
        Load vectors and metadata into cache.
        
        Returns:
            True if loaded successfully
        """
        now = time.time()
        
        # Check if cache is still valid
        if self._vectors is not None:
            if (now - self._last_load) < rag_settings.mongo_cache_ttl:
                return True
        
        collection = self._get_collection()
        if collection is None:
            return False
        
        try:
            projection = {
                "embedding": 1,
                "path": 1,
                "provider": 1,
                "website": 1,
                "trip_index": 1,
                "trip_name": 1,
                "country": 1,
                "duration": 1,
                "trip_type": 1,
                "date_summary": 1,
                "title": 1,
                "preview": 1,
            }
            
            docs = list(collection.find({}, projection))
            
            vectors = []
            metadata = []
            
            for doc in docs:
                emb = doc.get("embedding")
                if not isinstance(emb, list):
                    continue
                
                vectors.append(emb)
                metadata.append({
                    "path": doc.get("path"),
                    "provider": doc.get("provider"),
                    "website": doc.get("website"),
                    "trip_index": doc.get("trip_index"),
                    "trip_name": doc.get("trip_name"),
                    "country": doc.get("country"),
                    "duration": doc.get("duration"),
                    "trip_type": doc.get("trip_type"),
                    "date_summary": doc.get("date_summary"),
                    "title": doc.get("title"),
                    "preview": doc.get("preview"),
                })
            
            if not vectors:
                return False
            
            self._vectors = np.array(vectors, dtype=np.float32)
            self._metadata = metadata
            self._last_load = now
            
            logger.debug(f"Loaded {len(vectors)} vectors from MongoDB")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load from MongoDB: {e}")
            return False
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query embedding
            top_k: Number of results
            
        Returns:
            List of metadata dicts with scores
        """
        if not self._load_cache():
            return []
        
        # Compute similarities
        similarities = self._vectors @ query_vector
        
        # Get top-k indices
        k = min(top_k, len(similarities))
        top_indices = np.argpartition(-similarities, k - 1)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        results = []
        for idx in top_indices.tolist():
            meta = self._metadata[idx].copy()
            meta["score"] = float(similarities[idx])
            results.append(meta)
        
        return results
    
    def insert_documents(
        self,
        metadata: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """
        Insert documents with embeddings.
        
        Args:
            metadata: Document metadata
            embeddings: Document embeddings
            
        Returns:
            Number of documents inserted
        """
        collection = self._get_collection()
        if collection is None:
            return 0
        
        try:
            docs = []
            for meta, emb in zip(metadata, embeddings):
                doc = meta.copy()
                doc["embedding"] = emb.tolist()
                docs.append(doc)
            
            if docs:
                collection.insert_many(docs)
            
            # Invalidate cache
            self._vectors = None
            self._metadata = []
            
            return len(docs)
            
        except Exception as e:
            logger.error(f"Failed to insert documents: {e}")
            return 0
    
    def clear(self) -> None:
        """Clear all documents."""
        collection = self._get_collection()
        if collection is not None:
            try:
                collection.delete_many({})
            except Exception as e:
                logger.error(f"Failed to clear collection: {e}")
        
        self._vectors = None
        self._metadata = []


def get_mongo_store() -> MongoVectorStore:
    """Get the MongoDB store instance."""
    return MongoVectorStore()
