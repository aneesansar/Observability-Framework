"""
RAG Endpoints.

Query and retrieval endpoints for the RAG pipeline.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.auth import get_current_user, UserIdentity
from app.rag import get_rag_pipeline, reindex
from app.observability import get_current_span

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================
# === Request/Response Models ===
# ============================================================

class QueryRequest(BaseModel):
    """Request body for RAG query."""
    
    query: str
    top_k: Optional[int] = 5


class TripSchedule(BaseModel):
    """A single trip schedule/date option."""
    
    start: Optional[str] = Field(None, alias="start_date")
    end: Optional[str] = Field(None, alias="end_date")
    price: Optional[str] = None
    departure: Optional[str] = Field(None, alias="airport")
    
    class Config:
        populate_by_name = True


class TripResult(BaseModel):
    """A single trip result."""
    
    title: Optional[str] = Field(None, alias="name")
    url: Optional[str] = None
    country: Optional[str] = None
    trip_id: Optional[str] = None
    duration: Optional[int] = Field(None, alias="trip_duration")
    provider: Optional[str] = None
    description: Optional[str] = None
    score: Optional[float] = None
    schedules: List[TripSchedule] = Field(default_factory=list)
    
    class Config:
        populate_by_name = True


class QueryResultMeta(BaseModel):
    """Metadata about the query result."""
    
    tenant_id: str
    query: str
    retrieved_count: int = 0
    has_filters: bool = False


class RAGQueryResponse(BaseModel):
    """
    Response from RAG query endpoint.
    
    Supports two formats:
    1. answer contains fenced JSON with results array
    2. results array directly at root level
    
    Also includes a natural language text_explanation for chat-style responses.
    """
    
    # Conversation ID for feedback tracking
    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Natural language explanation of the results
    text_explanation: Optional[str] = None
    
    # Option 1: Answer as fenced JSON string
    answer: Optional[str] = None
    
    # Option 2: Direct results array
    results: Optional[List[TripResult]] = None
    
    # Additional metadata
    reasoning: List[str] = Field(default_factory=list)
    query_result: Optional[QueryResultMeta] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)


class ReindexResponse(BaseModel):
    """Response from reindex operation."""
    
    documents: int
    dim: int


# ============================================================
# === Helper Functions ===
# ============================================================

def format_results_as_json(retrieved_docs: List[Any]) -> List[Dict[str, Any]]:
    """
    Format retrieved documents as structured trip results.
    
    Args:
        retrieved_docs: List of RetrievalResult objects
        
    Returns:
        List of trip result dictionaries
    """
    results = []
    
    for doc in retrieved_docs:
        meta = doc.metadata if hasattr(doc, 'metadata') else doc
        
        # Parse schedules from date_summary
        schedules = []
        date_summary = getattr(meta, 'date_summary', None) if not isinstance(meta, dict) else meta.get('date_summary', '')
        
        if date_summary:
            for part in str(date_summary).split(";"):
                part = part.strip()
                if not part:
                    continue
                try:
                    # Format: "start→end | price | airport"
                    dates_part, price_part, airport_part = [x.strip() for x in part.split("|")]
                    start, end = [x.strip() for x in dates_part.split("→")]
                    
                    schedules.append({
                        "start": start or None,
                        "end": end or None,
                        "price": price_part or None,
                        "departure": airport_part or None,
                    })
                except (ValueError, IndexError):
                    continue
        
        # Get duration as int
        duration_str = getattr(meta, 'duration', None) if not isinstance(meta, dict) else meta.get('duration', '')
        duration = None
        if duration_str:
            try:
                # Extract number from string like "14 days"
                import re
                match = re.search(r'\d+', str(duration_str))
                if match:
                    duration = int(match.group())
            except (ValueError, TypeError):
                pass
        
        # Build result with all details
        title = getattr(meta, 'trip_name', None) or getattr(meta, 'title', None) if not isinstance(meta, dict) else meta.get('trip_name') or meta.get('title')
        url = getattr(meta, 'url', None) if not isinstance(meta, dict) else meta.get('url')
        country = getattr(meta, 'country', None) if not isinstance(meta, dict) else meta.get('country')
        trip_id = getattr(meta, 'trip_id', None) if not isinstance(meta, dict) else meta.get('trip_id')
        provider = getattr(meta, 'provider', None) if not isinstance(meta, dict) else meta.get('provider')
        description = getattr(meta, 'description', None) if not isinstance(meta, dict) else meta.get('description')
        
        result = {
            "title": title,
            "url": url,
            "country": country,
            "trip_id": trip_id,
            "duration": duration,
            "provider": provider,
            "description": description,
            "score": doc.score if hasattr(doc, 'score') else None,
            "schedules": schedules,
        }
        
        # Try to extract URL from preview if not present
        if not result["url"]:
            preview = getattr(meta, 'preview', None) if not isinstance(meta, dict) else meta.get('preview', '')
            if preview:
                for line in str(preview).splitlines():
                    if line.lower().startswith("url:"):
                        result["url"] = line.split("url:", 1)[1].strip()
                        break
        
        results.append(result)
    
    return results


def format_as_fenced_json(results: List[Dict[str, Any]]) -> str:
    """
    Format results as fenced JSON string.
    
    Args:
        results: List of trip result dictionaries
        
    Returns:
        Fenced JSON string
    """
    json_obj = {"results": results}
    json_str = json.dumps(json_obj, indent=2, ensure_ascii=False)
    return f"```json\n{json_str}\n```"


def generate_text_explanation(query: str, results: List[Dict[str, Any]]) -> str:
    """
    Generate a natural language explanation of the trip results.
    
    Args:
        query: The user's original query
        results: List of trip result dictionaries
        
    Returns:
        Natural language explanation string (plain text, no markdown)
    """
    if not results:
        return "I couldn't find any trips matching your search criteria. Please try adjusting your query or filters."
    
    count = len(results)
    
    # Find earliest trip and cheapest trip for comparison
    earliest_trip = None
    earliest_date = None
    cheapest_trip = None
    cheapest_price = float('inf')
    
    for trip in results:
        schedules = trip.get("schedules", [])
        for sched in schedules:
            # Check for earliest date
            start_date = sched.get("start")
            if start_date and (earliest_date is None or start_date < earliest_date):
                earliest_date = start_date
                earliest_trip = trip
            
            # Check for cheapest price
            price_str = sched.get("price")
            if price_str:
                try:
                    price = float(str(price_str).replace(",", "."))
                    if price < cheapest_price:
                        cheapest_price = price
                        cheapest_trip = trip
                except (ValueError, TypeError):
                    pass
    
    # Build plain text response
    parts = []
    
    # Opening
    parts.append(f"I have found {count} trip{'s' if count > 1 else ''} matching your search.")
    
    # Add comparison metrics
    if earliest_trip and earliest_date:
        parts.append(f"The earliest starting trip is on {earliest_date} by {earliest_trip.get('provider', 'Unknown')} - \"{earliest_trip.get('title', 'Unknown Trip')}\".")
    
    if cheapest_trip and cheapest_price < float('inf'):
        # Find the schedule with the cheapest price
        cheapest_start = None
        for sched in cheapest_trip.get("schedules", []):
            price_str = sched.get("price")
            if price_str:
                try:
                    price = float(str(price_str).replace(",", "."))
                    if price == cheapest_price:
                        cheapest_start = sched.get("start")
                        break
                except (ValueError, TypeError):
                    pass
        
        price_display = f"€{cheapest_price:.0f}" if cheapest_price == int(cheapest_price) else f"€{cheapest_price:.2f}"
        cheapest_info = f"The trip with the lowest price is \"{cheapest_trip.get('title', 'Unknown Trip')}\" at {price_display}"
        if cheapest_start:
            cheapest_info += f" starting on {cheapest_start}"
        cheapest_info += "."
        parts.append(cheapest_info)
    
    # List the trips
    parts.append("")
    parts.append("Here are the trips:")
    
    for i, trip in enumerate(results, 1):
        title = trip.get("title") or "Unnamed Trip"
        country = trip.get("country") or "Unknown destination"
        duration = trip.get("duration")
        provider = trip.get("provider")
        schedules = trip.get("schedules", [])
        
        trip_info = f"{i}. {title} ({country})"
        if duration:
            trip_info += f" - {duration} days"
        if provider:
            trip_info += f" by {provider}"
        
        # Add first available schedule
        if schedules:
            first_sched = schedules[0]
            start = first_sched.get("start", "")
            price = first_sched.get("price", "")
            if start and price:
                trip_info += f". Starting {start} from €{price}"
            elif start:
                trip_info += f". Starting {start}"
            elif price:
                trip_info += f". From €{price}"
        
        parts.append(trip_info)
    
    parts.append("")
    parts.append("Would you like more details about any of these trips, or would you like me to refine the search?")
    
    return " ".join(parts)


# ============================================================
# === Endpoints ===
# ============================================================

@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request: Request,
    body: QueryRequest,
    user: UserIdentity = Depends(get_current_user),
) -> RAGQueryResponse:
    """
    Query the RAG pipeline.
    
    Retrieves relevant documents and generates an answer.
    
    Response format includes:
    - `text_explanation`: Natural language chat-style explanation of the results
    - `answer`: Fenced JSON containing results (for backward compatibility)
    - `results`: Structured array of trip results
    """
    tenant_id = user.tenant_id
    
    # Generate conversation ID early for feedback tracking across all spans
    conversation_id = str(uuid.uuid4())
    
    try:
        # Set conversation_id on the current span immediately
        try:
            span = get_current_span()
            if span:
                span.set_attribute("conversation_id", conversation_id)
                span.set_attribute("rag.conversation_id", conversation_id)
                span.set_attribute("feedback.status", "not_given")
                span.set_attribute("feedback.rating", "pending")
        except Exception as e:
            logger.debug(f"Could not set initial span attributes: {e}")
        
        logger.info(
            f"Processing RAG query for tenant {tenant_id}",
            extra={"tenant_id": tenant_id, "conversation_id": conversation_id}
        )
        
        pipeline = get_rag_pipeline()
        response = pipeline.query(
            question=body.query,
            tenant_id=tenant_id,
            top_k=body.top_k,
            conversation_id=conversation_id,  # Pass conversation_id to pipeline
        )
        
        # Format retrieved docs as structured results
        results = format_results_as_json(response.retrieved_docs)
        
        # Create fenced JSON answer
        answer = format_as_fenced_json(results)
        
        # Build sources from retrieved docs
        sources = []
        for doc in response.retrieved_docs:
            meta = doc.metadata if hasattr(doc, 'metadata') else doc
            sources.append({
                "title": getattr(meta, 'title', None) or meta.get('title'),
                "provider": getattr(meta, 'provider', None) or meta.get('provider'),
                "score": doc.score if hasattr(doc, 'score') else 0.0,
            })
        
        # Build query metadata
        query_meta = QueryResultMeta(
            tenant_id=tenant_id,
            query=body.query,
            retrieved_count=len(response.retrieved_docs),
            has_filters=response.query_terms.has_filters() if response.query_terms else False,
        )
        
        # Get reasoning trace if available
        reasoning = []
        if response.generation_result and hasattr(response.generation_result, 'reasoning_trace'):
            reasoning = response.generation_result.reasoning_trace or []
        
        # Generate natural language explanation
        text_explanation = generate_text_explanation(body.query, results)
        
        logger.info(
            f"RAG query completed for tenant {tenant_id}: {len(results)} results",
            extra={"tenant_id": tenant_id, "conversation_id": conversation_id}
        )
        
        return RAGQueryResponse(
            conversation_id=conversation_id,
            text_explanation=text_explanation,
            answer=answer,
            results=[TripResult(**r) for r in results],
            reasoning=reasoning,
            query_result=query_meta,
            sources=sources,
        )
        
    except Exception as e:
        logger.error(
            f"RAG query failed: {e}",
            exc_info=True,
            extra={"tenant_id": tenant_id}
        )
        raise HTTPException(status_code=500, detail="Query processing failed")


@router.post("/reindex", response_model=ReindexResponse)
async def reindex_documents(
    user: UserIdentity = Depends(get_current_user),
) -> ReindexResponse:
    """
    Reindex all documents.
    
    Rebuilds the vector index from source documents.
    """
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        logger.info(
            f"Reindexing documents (requested by {user.username})",
            extra={"tenant_id": user.tenant_id}
        )
        
        result = reindex()
        
        logger.info(
            f"Reindex complete: {result['documents']} documents",
            extra={"tenant_id": user.tenant_id}
        )
        
        return ReindexResponse(**result)
        
    except Exception as e:
        logger.error(f"Reindex failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reindex failed")


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    
    Returns the status of the RAG service.
    """
    return {"status": "healthy", "service": "rag"}
