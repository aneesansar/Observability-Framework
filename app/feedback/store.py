"""
Feedback Store.

Thread-safe in-memory storage for feedback entries.
Provides methods to store, retrieve, and query feedback by conversation_id.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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
    Thread-safe in-memory feedback storage.
    
    Provides methods to store and retrieve feedback entries,
    with lookup by conversation_id for trace enrichment.
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize the feedback store.
        
        Args:
            max_entries: Maximum number of entries to store
        """
        self._entries: Dict[str, FeedbackEntry] = {}
        self._by_conversation: Dict[str, str] = {}  # conversation_id -> feedback_id
        self._lock = Lock()
        self._max_entries = max_entries
        logger.info(f"FeedbackStore initialized (max_entries={max_entries})")
    
    def add(self, entry: FeedbackEntry) -> None:
        """
        Add a feedback entry.
        
        Args:
            entry: FeedbackEntry to store
        """
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
                f"Feedback stored: {entry.id} for conversation {entry.conversation_id}",
                extra={"tenant_id": entry.tenant_id}
            )
    
    def get(self, feedback_id: str) -> Optional[FeedbackEntry]:
        """
        Get a feedback entry by ID.
        
        Args:
            feedback_id: ID of the feedback entry
            
        Returns:
            FeedbackEntry or None if not found
        """
        with self._lock:
            return self._entries.get(feedback_id)
    
    def get_by_conversation(self, conversation_id: str) -> Optional[FeedbackEntry]:
        """
        Get feedback for a specific conversation.
        
        Args:
            conversation_id: ID of the conversation
            
        Returns:
            FeedbackEntry or None if no feedback exists
        """
        with self._lock:
            feedback_id = self._by_conversation.get(conversation_id)
            if feedback_id:
                return self._entries.get(feedback_id)
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
        
        Args:
            tenant_id: Tenant identifier
            limit: Maximum entries to return
            
        Returns:
            List of FeedbackEntry objects
        """
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
        
        Args:
            limit: Maximum entries to return
            
        Returns:
            List of FeedbackEntry objects
        """
        with self._lock:
            entries = list(self._entries.values())
            entries.sort(key=lambda x: x.timestamp, reverse=True)
            return entries[:limit]
    
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
