"""
RAG Configuration.

Centralized settings for the RAG pipeline.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGSettings(BaseSettings):
    """
    Configuration settings for the RAG pipeline.
    """
    
    # Embedding Model
    embed_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBED_MODEL",
        description="Sentence transformer model for embeddings"
    )
    
    # LLM Settings
    ollama_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_URL",
        description="Ollama API base URL"
    )
    llm_model: str = Field(
        default="gemma3:latest",
        alias="LLM_MODEL",
        description="Default LLM model for generation"
    )
    llm_timeout: int = Field(
        default=90,
        alias="LLM_TIMEOUT",
        description="LLM request timeout in seconds"
    )
    
    # MongoDB Settings
    mongo_uri: str = Field(
        default="mongodb://localhost:27017",
        alias="MONGO_URI",
        description="MongoDB connection URI"
    )
    mongo_db: str = Field(
        default="rag",
        alias="MONGO_DB",
        description="MongoDB database name"
    )
    mongo_collection: str = Field(
        default="trips",
        alias="MONGO_COLL",
        description="MongoDB collection name"
    )
    
    # Retrieval Settings
    default_top_k: int = Field(
        default=5,
        alias="RAG_TOP_K",
        description="Default number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.3,
        alias="RAG_SIMILARITY_THRESHOLD",
        description="Minimum similarity score for retrieval"
    )
    
    # Cache Settings
    mongo_cache_ttl: float = Field(
        default=60.0,
        alias="MONGO_CACHE_TTL",
        description="MongoDB cache TTL in seconds"
    )
    
    @property
    def ollama_generate_url(self) -> str:
        """Get the Ollama generate API endpoint."""
        return f"{self.ollama_url.rstrip('/')}/api/generate"
    
    @property
    def documents_dir(self) -> str:
        """Get the documents directory path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(base_dir, "documents")
    
    @property
    def data_dir(self) -> str:
        """Get the data directory path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return os.path.join(base_dir, "data")
    
    @property
    def index_path(self) -> str:
        """Get the FAISS index file path."""
        return os.path.join(self.data_dir, "index.faiss")
    
    @property
    def meta_path(self) -> str:
        """Get the metadata file path."""
        return os.path.join(self.data_dir, "meta.jsonl")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache()
def get_rag_settings() -> RAGSettings:
    """Get cached RAG settings instance."""
    return RAGSettings()


# Convenience alias
rag_settings = get_rag_settings()
