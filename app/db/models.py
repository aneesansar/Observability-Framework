"""
MongoDB Document Models.

Defines document structures for traces, feedback, evaluations, and prompts.
"""

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class TraceDocument:
    """MongoDB document for trace data."""
    
    trace_id: str
    tenant_id: str
    conversation_id: Optional[str] = None
    spans: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Aggregated metrics from spans
    total_spans: int = 0
    total_duration_ms: float = 0.0
    has_errors: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceDocument":
        """Create from dictionary."""
        return cls(
            trace_id=data.get("trace_id", ""),
            tenant_id=data.get("tenant_id", ""),
            conversation_id=data.get("conversation_id"),
            spans=data.get("spans", []),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            metadata=data.get("metadata", {}),
            total_spans=data.get("total_spans", 0),
            total_duration_ms=data.get("total_duration_ms", 0.0),
            has_errors=data.get("has_errors", False),
        )


@dataclass
class SpanDocument:
    """MongoDB document for individual span data."""
    
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str = ""
    start_time: int = 0
    end_time: int = 0
    duration_ms: float = 0.0
    status: str = "UNSET"
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanDocument":
        """Create from dictionary."""
        return cls(
            span_id=data.get("span_id", ""),
            trace_id=data.get("trace_id", ""),
            parent_span_id=data.get("parent_span_id"),
            name=data.get("name", ""),
            start_time=data.get("start_time", 0),
            end_time=data.get("end_time", 0),
            duration_ms=data.get("duration_ms", 0.0),
            status=data.get("status", "UNSET"),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
        )


@dataclass
class FeedbackDocument:
    """MongoDB document for feedback data."""
    
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
    
    # Link to trace
    trace_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackDocument":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            conversation_id=data.get("conversation_id", ""),
            tenant_id=data.get("tenant_id", ""),
            user_id=data.get("user_id", ""),
            feedback=data.get("feedback", ""),
            reason=data.get("reason", ""),
            suggested_answer=data.get("suggested_answer"),
            rating=data.get("rating"),
            message_id=data.get("message_id"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            created_by=data.get("created_by", ""),
            trace_id=data.get("trace_id"),
        )
    
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


@dataclass
class EvaluationDocument:
    """MongoDB document for LLM evaluation data."""
    
    id: str
    trace_id: str
    conversation_id: Optional[str] = None
    tenant_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Input data
    prompt: str = ""
    prompt_hash: str = ""
    response: str = ""
    context_texts: List[str] = field(default_factory=list)
    
    # LLM self-reported metrics
    self_confidence: float = 0.0
    hallucination_risk: float = 0.0
    
    # Calculated metrics
    semantic_similarity: float = 0.0
    prompt_consistency: float = 0.0
    max_context_similarity: float = 0.0
    calculated_hallucination: float = 0.0
    
    # DeepEval metrics
    answer_relevancy: float = 0.0
    faithfulness: float = 0.0
    consistency: bool = False
    hallucination_detected: bool = False
    
    # Prompt similarity metrics (computed from DB)
    similar_prompt_count: int = 0
    avg_similarity_to_past_prompts: float = 0.0
    most_similar_prompt_id: Optional[str] = None
    most_similar_prompt_score: float = 0.0
    
    # Performance metrics
    latency_seconds: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    
    # Feedback link
    feedback_id: Optional[str] = None
    feedback_rating: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDocument":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            trace_id=data.get("trace_id", ""),
            conversation_id=data.get("conversation_id"),
            tenant_id=data.get("tenant_id", ""),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            prompt=data.get("prompt", ""),
            prompt_hash=data.get("prompt_hash", ""),
            response=data.get("response", ""),
            context_texts=data.get("context_texts", []),
            self_confidence=data.get("self_confidence", 0.0),
            hallucination_risk=data.get("hallucination_risk", 0.0),
            semantic_similarity=data.get("semantic_similarity", 0.0),
            prompt_consistency=data.get("prompt_consistency", 0.0),
            max_context_similarity=data.get("max_context_similarity", 0.0),
            calculated_hallucination=data.get("calculated_hallucination", 0.0),
            answer_relevancy=data.get("answer_relevancy", 0.0),
            faithfulness=data.get("faithfulness", 0.0),
            consistency=data.get("consistency", False),
            hallucination_detected=data.get("hallucination_detected", False),
            similar_prompt_count=data.get("similar_prompt_count", 0),
            avg_similarity_to_past_prompts=data.get("avg_similarity_to_past_prompts", 0.0),
            most_similar_prompt_id=data.get("most_similar_prompt_id"),
            most_similar_prompt_score=data.get("most_similar_prompt_score", 0.0),
            latency_seconds=data.get("latency_seconds", 0.0),
            tokens_input=data.get("tokens_input", 0),
            tokens_output=data.get("tokens_output", 0),
            feedback_id=data.get("feedback_id"),
            feedback_rating=data.get("feedback_rating"),
        )


@dataclass
class PromptDocument:
    """MongoDB document for storing prompts with embeddings for similarity computation."""
    
    id: str
    prompt_hash: str
    prompt_text: str
    tenant_id: str
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Embedding vector for similarity computation
    embedding: List[float] = field(default_factory=list)
    
    # Associated response quality
    response_quality_score: float = 0.0
    feedback_rating: Optional[bool] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MongoDB insertion."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptDocument":
        """Create from dictionary."""
        return cls(
            id=data.get("id", ""),
            prompt_hash=data.get("prompt_hash", ""),
            prompt_text=data.get("prompt_text", ""),
            tenant_id=data.get("tenant_id", ""),
            conversation_id=data.get("conversation_id"),
            trace_id=data.get("trace_id"),
            timestamp=data.get("timestamp", datetime.utcnow().isoformat()),
            embedding=data.get("embedding", []),
            response_quality_score=data.get("response_quality_score", 0.0),
            feedback_rating=data.get("feedback_rating"),
            metadata=data.get("metadata", {}),
        )
    
    @staticmethod
    def compute_hash(prompt_text: str) -> str:
        """Compute hash of prompt text for deduplication."""
        return hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
