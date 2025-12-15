"""
RAG Models.

Data models for documents, queries, and results.
"""

from .documents import (
    TripDate,
    TripDocument,
    RetrievalResult,
    DocumentMetadata,
)
from .queries import (
    QueryTerms,
    QueryIntent,
    SortOrder,
)

__all__ = [
    "TripDate",
    "TripDocument",
    "RetrievalResult",
    "DocumentMetadata",
    "QueryTerms",
    "QueryIntent",
    "SortOrder",
]
