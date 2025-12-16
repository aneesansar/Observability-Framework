"""
Feedback Module.

Provides feedback storage and retrieval for RAG responses.
Feedback is linked to traces via conversation_id.
"""

from .store import (
    FeedbackStore,
    get_feedback_store,
    FeedbackEntry,
)

__all__ = [
    "FeedbackStore",
    "get_feedback_store",
    "FeedbackEntry",
]
