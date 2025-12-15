"""
RAG (Retrieval-Augmented Generation) Module.

Provides the complete RAG pipeline for travel query processing.

Architecture:
    rag/
    ├── __init__.py              # Public API
    ├── config.py                # RAG settings
    ├── core/
    │   ├── embeddings.py        # Embedding generation
    │   ├── retriever.py         # Document retrieval
    │   ├── generator.py         # LLM response generation
    │   └── pipeline.py          # RAG pipeline orchestration
    ├── preprocessing/
    │   ├── document_loader.py   # JSON document loading
    │   └── text_processor.py    # Text normalization
    ├── storage/
    │   ├── faiss_store.py       # FAISS vector store
    │   └── mongo_store.py       # MongoDB vector store
    ├── models/
    │   ├── documents.py         # Document models
    │   └── queries.py           # Query models
    └── utils/
        ├── country_utils.py     # Country normalization
        └── query_parser.py      # Query parsing utilities

Usage:
    from app.rag import RAGPipeline, query_trips
    
    pipeline = RAGPipeline()
    result = pipeline.query("Find trips to Italy")
"""

# Core components
from .core.pipeline import (
    RAGPipeline,
    get_rag_pipeline,
)
from .core.generator import (
    LLMGenerator,
    generate_response,
)
from .core.retriever import (
    DocumentRetriever,
    retrieve_documents,
)
from .core.embeddings import (
    EmbeddingService,
    get_embedding_service,
)

# Storage
from .storage.faiss_store import (
    FAISSVectorStore,
    reindex,
)

# Models
from .models.documents import (
    TripDocument,
    TripDate,
    RetrievalResult,
)
from .models.queries import (
    QueryTerms,
    QueryIntent,
)

# Configuration
from .config import (
    rag_settings,
    RAGSettings,
)

__all__ = [
    # Pipeline
    "RAGPipeline",
    "get_rag_pipeline",
    # Generator
    "LLMGenerator",
    "generate_response",
    # Retriever
    "DocumentRetriever",
    "retrieve_documents",
    # Embeddings
    "EmbeddingService",
    "get_embedding_service",
    # Storage
    "FAISSVectorStore",
    "reindex",
    # Models
    "TripDocument",
    "TripDate",
    "RetrievalResult",
    "QueryTerms",
    "QueryIntent",
    # Config
    "rag_settings",
    "RAGSettings",
]
