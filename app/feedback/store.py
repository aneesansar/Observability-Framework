"""
Feedback Store.

Hybrid storage for feedback entries with MongoDB persistence and in-memory cache.
Provides methods to store, retrieve, and query feedback by conversation_id.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy import to avoid circular dependencies
_feedback_repo = None
_evaluation_repo = None


def _get_feedback_repo():
    """Get feedback repository lazily."""
    global _feedback_repo
    if _feedback_repo is None:
        try:
            from app.db.repositories import get_feedback_repository
            _feedback_repo = get_feedback_repository()
        except Exception as e:
            logger.warning(f"Failed to initialize feedback repository: {e}")
    return _feedback_repo


def _get_evaluation_repo():
    """Get evaluation repository lazily."""
    global _evaluation_repo
    if _evaluation_repo is None:
        try:
            from app.db.repositories import get_evaluation_repository
            _evaluation_repo = get_evaluation_repository()
        except Exception as e:
            logger.warning(f"Failed to initialize evaluation repository: {e}")
    return _evaluation_repo


@dataclass
class FeedbackEntry:
    """A single feedback entry."""
    
    id: str
    conversation_id: str
    tenant_id: str
    user_id: str
    feedback: str
    reason: str  # 'correct' or 'incorrect'
    suggested_answer: Optional[str] = None
    rating: Optional[bool] = None
    message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "feedback": self.feedback,
            "reason": self.reason,
            "suggested_answer": self.suggested_answer,
            "rating": self.rating,
            "message_id": self.message_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "created_by": self.created_by,
        }
    
    def get_feedback_attributes(self) -> Dict[str, Any]:
        """Get attributes suitable for trace enrichment."""
        return {
            "feedback.id": self.id,
            "feedback.status": "given",
            "feedback.reason": self.reason,
            "feedback.rating": "positive" if self.rating else "negative",
            "feedback.has_suggested_answer": self.suggested_answer is not None,
            "feedback.timestamp": self.timestamp,
        }


class FeedbackStore:
    """
    Hybrid feedback storage with MongoDB persistence and in-memory cache.
    
    Provides methods to store and retrieve feedback entries,
    with lookup by conversation_id for trace enrichment.
    Falls back to in-memory storage if MongoDB is unavailable.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize the feedback store.
        
        Args:
            max_entries: Maximum number of entries to store in memory
        """
        self._entries: Dict[str, FeedbackEntry] = {}
        self._by_conversation: Dict[str, str] = {}  # conversation_id -> feedback_id
        self._lock = Lock()
        self._max_entries = max_entries
        self._use_mongodb = True
        logger.info(f"FeedbackStore initialized (max_entries={max_entries}, mongodb=True)")
    
    def add(self, entry: FeedbackEntry) -> None:
        """
        Add a feedback entry.
        
        Persists to MongoDB and caches in memory.
        
        Args:
            entry: FeedbackEntry to store
        """
        # Try to persist to MongoDB first
        mongodb_saved = False
        if self._use_mongodb:
            mongodb_saved = self._save_to_mongodb(entry)
        
        # Always cache in memory
        with self._lock:
            # Trim if over capacity
            if len(self._entries) >= self._max_entries:
                # Remove oldest entries (first 10%)
                to_remove = list(self._entries.keys())[:self._max_entries // 10]
                for key in to_remove:
                    old_entry = self._entries.pop(key, None)
                    if old_entry:
                        self._by_conversation.pop(old_entry.conversation_id, None)
                logger.debug(f"Trimmed {len(to_remove)} old feedback entries")
            
            self._entries[entry.id] = entry
            self._by_conversation[entry.conversation_id] = entry.id
            
            logger.info(
                f"Feedback stored: {entry.id} for conversation {entry.conversation_id} (mongodb={mongodb_saved})",
                extra={"tenant_id": entry.tenant_id, "mongodb_saved": mongodb_saved}
            )
        
        # Update evaluation with feedback if exists
        if mongodb_saved:
            self._update_evaluation_with_feedback(entry)
    
    def _save_to_mongodb(self, entry: FeedbackEntry) -> bool:
        """Save feedback to MongoDB."""
        try:
            repo = _get_feedback_repo()
            if repo is None:
                return False
            
            from app.db.models import FeedbackDocument
            
            doc = FeedbackDocument(
                id=entry.id,
                conversation_id=entry.conversation_id,
                tenant_id=entry.tenant_id,
                user_id=entry.user_id,
                feedback=entry.feedback,
                reason=entry.reason,
                suggested_answer=entry.suggested_answer,
                rating=entry.rating,
                message_id=entry.message_id,
                metadata=entry.metadata,
                timestamp=entry.timestamp,
                created_by=entry.created_by,
            )
            return repo.save(doc)
        except Exception as e:
            logger.warning(f"Failed to save feedback to MongoDB: {e}")
            return False
    
    def _update_evaluation_with_feedback(self, entry: FeedbackEntry) -> None:
        """Update evaluation record with feedback data."""
        try:
            repo = _get_evaluation_repo()
            if repo is not None:
                repo.update_with_feedback(
                    conversation_id=entry.conversation_id,
                    feedback_id=entry.id,
                    feedback_rating=entry.rating or False,
                )
        except Exception as e:
            logger.debug(f"Failed to update evaluation with feedback: {e}")
    
    def get(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """
        Get a feedback entry by ID.
        
        Checks memory cache first, then MongoDB.
        
        Args:
            feedback_id: ID of the feedback entry
            
        Returns:
            FeedbackEntry or None if not found
        """
        # Check memory cache first
        with self._lock:
            entry = self._entries.get(feedback_id)
            if entry:
                return entry
        
        # Try MongoDB
        if self._use_mongodb:
            return self._get_from_mongodb(feedback_id)
        
        return None
    
    def _get_from_mongodb(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """Get feedback from MongoDB."""
        try:
            repo = _get_feedback_repo()
            if repo is None:
                return None
            
            doc = repo.get_by_id(feedback_id)
            if doc:
                return self._doc_to_entry(doc)
        except Exception as e:
            logger.debug(f"Failed to get feedback from MongoDB: {e}")
        return None
    
    def _doc_to_entry(self, doc) -> FeedbackEntry:
        """Convert FeedbackDocument to FeedbackEntry."""
        return FeedbackEntry(
            id=doc.id,
            conversation_id=doc.conversation_id,
            tenant_id=doc.tenant_id,
            user_id=doc.user_id,
            feedback=doc.feedback,
            reason=doc.reason,
            suggested_answer=doc.suggested_answer,
            rating=doc.rating,
            message_id=doc.message_id,
            metadata=doc.metadata,
            timestamp=doc.timestamp,
            created_by=doc.created_by,
        )
    
    def get_by_conversation(self, conversation_id: str) -> Optional[FeedbackEntry]:
        """
        Get feedback for a specific conversation.
        
        Checks memory cache first, then MongoDB.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            FeedbackEntry or None if no feedback exists
        """
        # Check memory cache first
        with self._lock:
            feedback_id = self._by_conversation.get(conversation_id)
            if feedback_id:
                entry = self._entries.get(feedback_id)
                if entry:
                    return entry
        
        # Try MongoDB
        if self._use_mongodb:
            try:
                repo = _get_feedback_repo()
                if repo is not None:
                    doc = repo.get_by_conversation_id(conversation_id)
                    if doc:
                        return self._doc_to_entry(doc)
            except Exception as e:
                logger.debug(f"Failed to get feedback from MongoDB: {e}")
        
        return None
    
    def get_feedback_attributes_for_conversation(
        self, 
        conversation_id: str
    ) -> Dict[str, Any]:
        """
        Get feedback attributes for trace enrichment.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            Dict of feedback attributes for the trace
        """
        entry = self.get_by_conversation(conversation_id)
        if entry:
            return entry.get_feedback_attributes()
        
        # No feedback given
        return {
            "feedback.status": "not_given",
            "feedback.rating": "pending",
        }
    
    def list_by_tenant(
        self, 
        tenant_id: str, 
        limit: int = 50
    ) -> List[FeedbackEntry]:
        """
        List feedback entries for a tenant.
        
        Fetches from MongoDB if available, falls back to memory.
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum entries to return
            
        Returns:
            List of FeedbackEntry objects
        """
        # Try MongoDB first
        if self._use_mongodb:
            try:
                repo = _get_feedback_repo()
                if repo is not None:
                    docs = repo.list_by_tenant(tenant_id, limit)
                    if docs:
                        return [self._doc_to_entry(doc) for doc in docs]
            except Exception as e:
                logger.debug(f"Failed to list feedback from MongoDB: {e}")
        
        # Fall back to memory
        with self._lock:
            entries = [
                e for e in self._entries.values()
                if e.tenant_id == tenant_id
            ]
            # Sort by timestamp descending
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            return entries[:limit]
    
    def list_all(self, limit: int = 100) -> List[FeedbackEntry]:
        """
        List all feedback entries (admin only).
        
        Fetches from MongoDB if available, falls back to memory.
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of FeedbackEntry objects
        """
        # Try MongoDB first
        if self._use_mongodb:
            try:
                repo = _get_feedback_repo()
                if repo is not None:
                    docs = repo.list_all(limit)
                    if docs:
                        return [self._doc_to_entry(doc) for doc in docs]
            except Exception as e:
                logger.debug(f"Failed to list all feedback from MongoDB: {e}")
        
        # Fall back to memory
        with self._lock:
            entries = list(self._entries.values())
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            return entries[:limit]
    
    def get_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get feedback statistics.
        
        Args:
            tenant_id: Optional tenant filter
            
        Returns:
            Dict with feedback statistics
        """
        if self._use_mongodb:
            try:
                repo = _get_feedback_repo()
                if repo is not None:
                    return repo.get_feedback_stats(tenant_id)
            except Exception as e:
                logger.debug(f"Failed to get feedback stats from MongoDB: {e}")
        
        # Fall back to memory calculation
        with self._lock:
            entries = list(self._entries.values())
            if tenant_id:
                entries = [e for e in entries if e.tenant_id == tenant_id]
            
            return {
                "total": len(entries),
                "positive": sum(1 for e in entries if e.rating is True),
                "negative": sum(1 for e in entries if e.rating is False),
                "correct": sum(1 for e in entries if e.reason == "correct"),
                "incorrect": sum(1 for e in entries if e.reason == "incorrect"),
            }
    
    def clear(self) -> int:
        """
        Clear all feedback entries.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._by_conversation.clear()
            logger.info(f"Cleared {count} feedback entries")
            return count


# Global singleton instance
_feedback_store: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """
    Get the global feedback store instance.
    
    Returns:
        FeedbackStore singleton
    """
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store
