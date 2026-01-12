"""
MongoDB Span Exporter.

Persists OpenTelemetry spans to MongoDB for long-term storage and analysis.
Works alongside the in-memory exporter for real-time access.
"""

import logging
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_trace_repo = None
_evaluation_repo = None
_prompt_repo = None


def _get_trace_repo():
    """Get trace repository lazily."""
    global _trace_repo
    if _trace_repo is None:
        try:
            from app.db.repositories import get_trace_repository
            _trace_repo = get_trace_repository()
        except Exception as e:
            logger.warning(f"Failed to initialize trace repository: {e}")
    return _trace_repo


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


def _get_prompt_repo():
    """Get prompt repository lazily."""
    global _prompt_repo
    if _prompt_repo is None:
        try:
            from app.db.repositories import get_prompt_repository
            _prompt_repo = get_prompt_repository()
        except Exception as e:
            logger.warning(f"Failed to initialize prompt repository: {e}")
    return _prompt_repo


class MongoDBSpanExporter(SpanExporter):
    """
    MongoDB span exporter for persistent trace storage.
    
    Exports spans to MongoDB, grouping them by trace_id.
    Also extracts evaluation data from LLM spans for analysis.
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize the exporter.
        
        Args:
            batch_size: Number of spans to batch before writing
        """
        self._lock = Lock()
        self._batch_size = batch_size
        self._pending_spans: Dict[str, List[Dict[str, Any]]] = {}  # trace_id -> spans
        self._export_count = 0
        logger.info(f"MongoDBSpanExporter initialized (batch_size={batch_size})")
    
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """
        Export spans to MongoDB.
        
        Groups spans by trace_id and persists them.
        
        Args:
            spans: Sequence of spans to export
            
        Returns:
            SpanExportResult.SUCCESS
        """
        if not spans:
            return SpanExportResult.SUCCESS
        
        try:
            with self._lock:
                for span in spans:
                    self._process_span(span)
                
                # Flush pending traces that are complete
                self._flush_complete_traces()
            
            return SpanExportResult.SUCCESS
            
        except Exception as e:
            logger.error(f"Failed to export spans to MongoDB: {e}")
            return SpanExportResult.FAILURE
    
    def _process_span(self, span: ReadableSpan) -> None:
        """Process a single span."""
        trace_id = format(span.context.trace_id, "032x")
        span_data = self._span_to_dict(span)
        
        if trace_id not in self._pending_spans:
            self._pending_spans[trace_id] = []
        
        self._pending_spans[trace_id].append(span_data)
        
        # Extract evaluation data from LLM spans
        self._extract_evaluation_data(span, trace_id)
    
    def _span_to_dict(self, span: ReadableSpan) -> Dict[str, Any]:
        """Convert span to dictionary."""
        parent_span_id = None
        if span.parent and hasattr(span.parent, "span_id"):
            parent_span_id = format(span.parent.span_id, "016x")
        
        # Extract events
        events = []
        if hasattr(span, "events") and span.events:
            for event in span.events:
                events.append({
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": dict(event.attributes) if event.attributes else {},
                })
        
        # Calculate duration
        duration_ms = 0.0
        if span.start_time and span.end_time:
            duration_ms = (span.end_time - span.start_time) / 1_000_000  # ns to ms
        
        return {
            "span_id": format(span.context.span_id, "016x"),
            "trace_id": format(span.context.trace_id, "032x"),
            "parent_span_id": parent_span_id,
            "name": span.name,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": duration_ms,
            "status": span.status.status_code.name if span.status else "UNSET",
            "attributes": dict(span.attributes) if span.attributes else {},
            "events": events,
        }
    
    def _extract_evaluation_data(self, span: ReadableSpan, trace_id: str) -> None:
        """Extract evaluation data from LLM-related spans."""
        attrs = dict(span.attributes) if span.attributes else {}
        
        # Check if this is an LLM generation span
        if not attrs.get("LLM.response") and not attrs.get("generate_response.model"):
            return
        
        try:
            import uuid
            from app.db.models import EvaluationDocument, PromptDocument
            
            conversation_id = attrs.get("conversation_id") or attrs.get("rag.conversation_id")
            tenant_id = attrs.get("tenant_id", "")
            
            # Create evaluation document
            evaluation = EvaluationDocument(
                id=str(uuid.uuid4()),
                trace_id=trace_id,
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                timestamp=datetime.utcnow().isoformat(),
                
                # LLM self-reported metrics
                self_confidence=float(attrs.get("LLM.self_confidence", 0.0)),
                hallucination_risk=float(attrs.get("LLM.estimated_hallucination_risk", 0.0)),
                
                # Calculated metrics
                semantic_similarity=float(attrs.get("Calculated.semantic_similarity_to_prompt", 0.0)),
                prompt_consistency=float(attrs.get("Calculated.prompt_consistency", 0.0)),
                max_context_similarity=float(attrs.get("Calculated.max_similarity_to_any_snippet", 0.0)),
                calculated_hallucination=float(attrs.get("Calculated.hallucination_score", 0.0)),
                
                # DeepEval metrics
                answer_relevancy=float(attrs.get("DeepEval.answer_relevancy", 0.0)),
                faithfulness=float(attrs.get("DeepEval.faithfulness", 0.0)),
                consistency=bool(attrs.get("DeepEval.consistency", False)),
                hallucination_detected=bool(attrs.get("DeepEval.hallucination", False)),
                
                # Performance metrics
                latency_seconds=float(attrs.get("LLM.latency_seconds", 0.0)),
                tokens_input=int(attrs.get("Inference.tokens_input", 0)),
                tokens_output=int(attrs.get("Inference.tokens_output", 0)),
                
                # Response data
                response=str(attrs.get("LLM.response", ""))[:1000],  # Truncate for storage
            )
            
            # Save evaluation
            eval_repo = _get_evaluation_repo()
            if eval_repo:
                eval_repo.save(evaluation)
                logger.debug(f"Saved evaluation for trace {trace_id}")
            
            # Store prompt for similarity computation
            prompt_text = str(attrs.get("rag.query", ""))
            if prompt_text:
                self._store_prompt(
                    prompt_text=prompt_text,
                    tenant_id=tenant_id,
                    conversation_id=conversation_id,
                    trace_id=trace_id,
                    quality_score=evaluation.answer_relevancy,
                )
                
        except Exception as e:
            logger.debug(f"Failed to extract evaluation data: {e}")
    
    def _store_prompt(
        self,
        prompt_text: str,
        tenant_id: str,
        conversation_id: Optional[str],
        trace_id: str,
        quality_score: float,
    ) -> None:
        """Store prompt with embedding for similarity computation."""
        try:
            import uuid
            from app.db.models import PromptDocument
            
            prompt_hash = PromptDocument.compute_hash(prompt_text)
            
            # Get embedding for the prompt
            embedding = self._get_prompt_embedding(prompt_text)
            
            prompt_doc = PromptDocument(
                id=str(uuid.uuid4()),
                prompt_hash=prompt_hash,
                prompt_text=prompt_text[:500],  # Truncate for storage
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                trace_id=trace_id,
                embedding=embedding,
                response_quality_score=quality_score,
            )
            
            prompt_repo = _get_prompt_repo()
            if prompt_repo:
                prompt_repo.save(prompt_doc)
                logger.debug(f"Saved prompt for trace {trace_id}")
                
        except Exception as e:
            logger.debug(f"Failed to store prompt: {e}")
    
    def _get_prompt_embedding(self, prompt_text: str) -> List[float]:
        """Get embedding for prompt text."""
        try:
            from app.rag.embeddings import get_embedding_service
            
            service = get_embedding_service()
            embedding = service.embed_text(prompt_text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.debug(f"Failed to get prompt embedding: {e}")
            return []
    
    def _flush_complete_traces(self, force: bool = False) -> None:
        """Flush traces that appear complete to MongoDB.
        
        Args:
            force: If True, flush all pending traces regardless of batch size
        """
        if not self._pending_spans:
            return
        
        trace_repo = _get_trace_repo()
        if trace_repo is None:
            logger.warning("Trace repository not available, cannot flush traces")
            return
        
        try:
            from app.db.models import TraceDocument
            
            # Flush pending traces based on batch size or force flag
            for trace_id, spans in list(self._pending_spans.items()):
                should_flush = force or len(spans) >= self._batch_size or self._export_count % 5 == 0
                if should_flush:
                    # Extract tenant_id and conversation_id from any span that has them
                    tenant_id = ""
                    conversation_id = None
                    query = ""
                    model = ""
                    
                    for span in spans:
                        attrs = span.get("attributes", {})
                        if not tenant_id and attrs.get("tenant_id"):
                            tenant_id = str(attrs.get("tenant_id"))
                        if not conversation_id:
                            conversation_id = attrs.get("conversation_id") or attrs.get("rag.conversation_id")
                        if not query and attrs.get("rag.query"):
                            query = str(attrs.get("rag.query"))
                        if not model and attrs.get("LLM.model"):
                            model = str(attrs.get("LLM.model"))
                    
                    # Calculate total duration
                    total_duration = sum(s.get("duration_ms", 0) for s in spans)
                    has_errors = any(s.get("status") == "ERROR" for s in spans)
                    
                    # Log for debugging
                    if tenant_id:
                        logger.debug(f"Saving trace {trace_id} for tenant {tenant_id}")
                    else:
                        logger.warning(f"Trace {trace_id} has no tenant_id, spans: {len(spans)}")
                    
                    trace_doc = TraceDocument(
                        trace_id=trace_id,
                        tenant_id=tenant_id,
                        conversation_id=conversation_id,
                        spans=spans,
                        total_spans=len(spans),
                        total_duration_ms=total_duration,
                        has_errors=has_errors,
                        metadata={
                            "query": query,
                            "model": model,
                        },
                    )
                    
                    if trace_repo.save(trace_doc):
                        del self._pending_spans[trace_id]
                        self._export_count += 1
                        logger.debug(f"Flushed trace {trace_id} to MongoDB")
                        
        except Exception as e:
            logger.error(f"Failed to flush traces: {e}")
    
    def _find_root_span(self, spans: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the root span (no parent) in a list of spans."""
        for span in spans:
            if not span.get("parent_span_id"):
                return span
        return spans[0] if spans else None
    
    def shutdown(self) -> None:
        """Shutdown the exporter, flushing any pending spans."""
        with self._lock:
            # Force flush all remaining spans
            pending_count = len(self._pending_spans)
            if pending_count > 0:
                logger.info(f"Flushing {pending_count} pending traces to MongoDB on shutdown")
            self._flush_complete_traces(force=True)
        logger.info("MongoDBSpanExporter shutdown complete")
    
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush all pending spans."""
        with self._lock:
            self._flush_complete_traces(force=True)
        return True


# Singleton instance
_mongodb_exporter: Optional[MongoDBSpanExporter] = None


def get_mongodb_exporter() -> MongoDBSpanExporter:
    """Get the singleton MongoDB exporter instance."""
    global _mongodb_exporter
    if _mongodb_exporter is None:
        _mongodb_exporter = MongoDBSpanExporter()
    return _mongodb_exporter
