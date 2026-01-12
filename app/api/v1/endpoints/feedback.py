"""
Feedback Endpoints.

User feedback collection for RAG responses with trace integration.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from app.auth import get_current_user, UserIdentity
from app.feedback import get_feedback_store, FeedbackEntry

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# === Request/Response Models ===
# ============================================================

class FeedbackRequest(BaseModel):
    """
    Request body for creating feedback.
    
    When reason is 'correct', suggested_answer is not required.
    When reason is 'incorrect', suggested_answer is required.
    """
    
    conversation_id: str = Field(..., description="ID of the conversation to provide feedback for")
    feedback: str = Field(..., description="Feedback text from user")
    reason: str = Field(..., description="Reason for feedback: 'correct' or 'incorrect'")
    suggested_answer: Optional[str] = Field(None, description="Suggested correct answer (required if reason is 'incorrect')")
    rating: Optional[bool] = Field(None, description="Positive (true) or negative (false) rating")
    user_id: Optional[str] = Field(None, description="User identifier")
    message_id: Optional[str] = Field(None, description="Specific message ID if applicable")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    @model_validator(mode='after')
    def validate_suggested_answer(self):
        """Validate that suggested_answer is provided when reason is 'incorrect'."""
        if self.reason.lower() == 'incorrect' and not self.suggested_answer:
            raise ValueError("suggested_answer is required when reason is 'incorrect'")
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "feedback": "The answer was helpful",
                "reason": "correct",
                "rating": True,
                "metadata": {}
            }
        }


class FeedbackResponse(BaseModel):
    """Response after creating feedback."""
    
    id: str = Field(..., description="Generated feedback ID")
    conversation_id: str = Field(..., description="Associated conversation ID")
    tenant_id: str = Field(..., description="Tenant identifier")
    feedback: str = Field(..., description="Feedback text")
    reason: str = Field(..., description="Feedback reason")
    suggested_answer: Optional[str] = Field(None, description="Suggested answer if provided")
    rating: Optional[bool] = Field(None, description="Rating")
    timestamp: datetime = Field(..., description="Feedback creation timestamp")
    trace_updated: bool = Field(..., description="Whether the trace was updated with feedback")
    message: str = Field(..., description="Status message")


class FeedbackListResponse(BaseModel):
    """Response for listing feedback."""
    
    feedbacks: List[Dict[str, Any]]
    total: int


# ============================================================
# === Helper Functions ===
# ============================================================

def check_trace_exists(conversation_id: str) -> bool:
    """
    Check if a trace exists for the given conversation_id.
    
    Args:
        conversation_id: The conversation ID to find
        
    Returns:
        True if trace was found, False otherwise
    """
    try:
        from app.observability import get_inmemory_exporter
        
        exporter = get_inmemory_exporter()
        if not exporter:
            return False
        
        # Get all spans and find ones matching the conversation_id
        spans = exporter.get_finished_spans()
        
        for span in spans:
            attrs = dict(span.attributes) if span.attributes else {}
            span_conversation_id = attrs.get("conversation_id") or attrs.get("rag.conversation_id")
            
            if span_conversation_id == conversation_id:
                logger.info(
                    f"Found trace for conversation {conversation_id}",
                    extra={"trace_id": format(span.context.trace_id, "032x")}
                )
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to check trace: {e}")
        return False


# ============================================================
# === Endpoints ===
# ============================================================

@router.post("", response_model=FeedbackResponse)
async def create_feedback(
    body: FeedbackRequest,
    user: UserIdentity = Depends(get_current_user),
    x_tenant_id: Optional[str] = Header(None, alias="x-tenant-id"),
) -> FeedbackResponse:
    """
    Create new feedback.
    
    Creates a new feedback entry for the specified conversation.
    When reason is 'correct', suggested_answer is optional.
    When reason is 'incorrect', suggested_answer is required.
    
    The feedback is linked to the RAG trace via conversation_id.
    
    Args:
        body: Feedback data to create
        user: Current user (from auth)
        x_tenant_id: Tenant identifier (from header, optional)
        
    Returns:
        Created feedback with generated ID
        
    Raises:
        HTTPException: If creation fails or validation fails
    """
    # Use tenant from header or from user (normalize to lowercase for consistency)
    tenant_id = str(x_tenant_id or user.tenant_id).lower()
    
    try:
        # Generate feedback ID
        feedback_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # Determine rating based on reason if not explicitly provided
        rating = body.rating
        if rating is None:
            rating = body.reason.lower() == 'correct'
        
        # Create feedback entry
        feedback_entry = FeedbackEntry(
            id=feedback_id,
            conversation_id=body.conversation_id,
            tenant_id=tenant_id,
            user_id=body.user_id or user.username,
            feedback=body.feedback,
            reason=body.reason,
            suggested_answer=body.suggested_answer,
            rating=rating,
            message_id=body.message_id,
            metadata=body.metadata or {},
            timestamp=timestamp.isoformat(),
            created_by=user.username,
        )
        
        # Store feedback in the global store
        store = get_feedback_store()
        store.add(feedback_entry)
        
        # Check if trace exists for this conversation
        trace_exists = check_trace_exists(body.conversation_id)
        
        # Log the feedback creation
        logger.info(
            f"Feedback created: {feedback_id} for conversation {body.conversation_id}",
            extra={
                "tenant_id": tenant_id,
                "feedback_id": feedback_id,
                "conversation_id": body.conversation_id,
                "reason": body.reason,
                "rating": rating,
                "trace_exists": trace_exists,
            }
        )
        
        # Determine message based on trace existence
        if trace_exists:
            message = "Feedback created and linked to trace successfully"
        else:
            message = "Feedback created successfully. Trace not found for linking."
        
        return FeedbackResponse(
            id=feedback_id,
            conversation_id=body.conversation_id,
            tenant_id=tenant_id,
            feedback=body.feedback,
            reason=body.reason,
            suggested_answer=body.suggested_answer,
            rating=rating,
            timestamp=timestamp,
            trace_updated=trace_exists,
            message=message,
        )
        
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create feedback")


@router.get("", response_model=FeedbackListResponse)
async def list_feedback(
    user: UserIdentity = Depends(get_current_user),
    conversation_id: Optional[str] = None,
    limit: int = 50,
) -> FeedbackListResponse:
    """
    List feedback entries.
    
    Returns feedback entries for the current tenant.
    Optionally filter by conversation_id.
    
    Args:
        user: Current user (from auth)
        conversation_id: Optional filter by conversation ID
        limit: Maximum number of entries to return
        
    Returns:
        List of feedback entries
    """
    store = get_feedback_store()
    
    # Get feedback entries
    if user.is_admin:
        entries = store.list_all(limit=limit)
    else:
        entries = store.list_by_tenant(user.tenant_id, limit=limit)
    
    # Filter by conversation_id if provided
    if conversation_id:
        entries = [e for e in entries if e.conversation_id == conversation_id]
    
    feedbacks = [e.to_dict() for e in entries]
    
    return FeedbackListResponse(
        feedbacks=feedbacks,
        total=len(feedbacks),
    )


@router.get("/{feedback_id}")
async def get_feedback(
    feedback_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a specific feedback entry.
    
    Args:
        feedback_id: ID of the feedback to retrieve
        user: Current user (from auth)
        
    Returns:
        Feedback entry
        
    Raises:
        HTTPException: If feedback not found or access denied
    """
    store = get_feedback_store()
    entry = store.get(feedback_id)
    
    if not entry:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    # Check tenant access
    if entry.tenant_id != user.tenant_id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return entry.to_dict()


@router.get("/conversation/{conversation_id}")
async def get_feedback_by_conversation(
    conversation_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get feedback for a specific conversation.
    
    Args:
        conversation_id: ID of the conversation
        user: Current user (from auth)
        
    Returns:
        Feedback entry or indication that no feedback was given
    """
    store = get_feedback_store()
    entry = store.get_by_conversation(conversation_id)
    
    if not entry:
        return {
            "conversation_id": conversation_id,
            "feedback_status": "not_given",
            "message": "No feedback has been provided for this conversation"
        }
    
    # Check tenant access
    if entry.tenant_id != user.tenant_id and not user.is_admin:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        **entry.to_dict(),
        "feedback_status": "given",
    }
