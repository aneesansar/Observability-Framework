"""
Observability Endpoints.

Endpoints for accessing traces, logs, and metrics.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import PlainTextResponse

from app.auth import get_current_user, UserIdentity, auth_service
from app.core.config import get_settings
from app.observability import (
    get_recent_spans,
    get_log_buffer,
    clear_spans,
    clear_log_buffer,
)
from app.feedback import get_feedback_store
from app.observability.metrics import (
    get_function_metrics,
    get_llm_metrics,
    get_rag_metrics,
)
from app.db.repositories import (
    get_trace_repository,
    get_feedback_repository,
    get_evaluation_repository,
    get_prompt_repository,
)
from app.core import get_resource_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory cache for system performance history (last 15 readings per tenant)
_system_performance_history: Dict[str, Dict[str, List[Any]]] = {}
MAX_HISTORY_POINTS = 15


def _get_system_history(tenant_id: str) -> Dict[str, List[Any]]:
    """Get or initialize system performance history for a tenant."""
    if tenant_id not in _system_performance_history:
        _system_performance_history[tenant_id] = {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io_mb": [],
            "network_mb": [],
        }
    return _system_performance_history[tenant_id]


def _add_system_reading(tenant_id: str, cpu: float, memory: float, disk_mb: float, network_mb: float) -> None:
    """Add a system performance reading to history."""
    from datetime import datetime
    
    history = _get_system_history(tenant_id)
    timestamp = datetime.utcnow().strftime("%H:%M")
    
    # Add new readings
    history["timestamps"].append(timestamp)
    history["cpu_usage"].append(round(cpu, 1))
    history["memory_usage"].append(round(memory, 1))
    history["disk_io_mb"].append(round(disk_mb, 2))
    history["network_mb"].append(round(network_mb, 2))
    
    # Keep only last MAX_HISTORY_POINTS
    for key in history:
        if len(history[key]) > MAX_HISTORY_POINTS:
            history[key] = history[key][-MAX_HISTORY_POINTS:]


@router.get("/traces")
async def get_traces(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    window_seconds: int = Query(default=600, ge=0, description="Only include spans ending within the last N seconds (0 = no limit)"),
    max_spans: int = Query(default=5000, ge=100, le=20000, description="Cap the total spans considered before pagination"),
    include_db: bool = Query(default=True, description="Include traces from MongoDB for persistence across restarts"),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get recent traces with hierarchical structure.
    
    Returns traces grouped by trace_id with parent-child span relationships.
    Filtered by tenant_id for non-admin users.
    
    When include_db=True (default), also fetches persisted traces from MongoDB
    to ensure traces are available after server restarts or user re-login.
    """
    from collections import defaultdict
    import time
    
    EXCLUDED_PATH_KEYWORDS = [
        "/observability/traces",
        "/observability/stats",
        "/observability/reset",
        "/metrics",
        "/prometheus",
        "/metrics/summary",
        "/docs",
        "/openapi.json",
    ]
    
    def is_excluded(span_data):
        name = span_data.name.lower()
        target = str(span_data.attributes.get("http.target", "")).lower()
        route = str(span_data.attributes.get("http.route", "")).lower()
        return any(
            keyword in name or keyword in target or keyword in route
            for keyword in EXCLUDED_PATH_KEYWORDS
        )
    
    # Get all spans from in-memory storage
    raw_spans = get_recent_spans(limit=max_spans * 10, tenant_id=None)
    
    # Also fetch from MongoDB if enabled for persistence across restarts
    db_traces = []
    if include_db:
        try:
            trace_repo = get_trace_repository()
            db_traces = trace_repo.list_by_tenant(str(user.tenant_id), limit=limit * 2)
            logger.info(f"Fetched {len(db_traces)} traces from MongoDB for tenant '{user.tenant_id}'")
        except Exception as e:
            logger.warning(f"Failed to fetch traces from MongoDB: {e}")
    
    # Apply time window filter
    if window_seconds and window_seconds > 0:
        now_ns = int(time.time() * 1e9)
        cutoff = now_ns - window_seconds * 1_000_000_000
        raw_spans = [s for s in raw_spans if s.end_time and s.end_time >= cutoff]
    
    # Exclude observability/system routes
    filtered = [s for s in raw_spans if not is_excluded(s)]
    
    # Tenant isolation - strict filtering
    tenant_id = user.tenant_id
    tenant_filtered = []
    for span in filtered:
        span_tenant = str(span.attributes.get("tenant_id", ""))
        if not span_tenant:
            continue  # Skip spans without tenant_id
        if span_tenant != str(tenant_id):
            continue  # Skip other tenants' spans
        tenant_filtered.append(span)
    
    # Keep only most recent max_spans
    if len(tenant_filtered) > max_spans:
        tenant_filtered.sort(key=lambda s: s.end_time or 0, reverse=True)
        tenant_filtered = tenant_filtered[:max_spans]
    
    # Group spans by trace_id
    traces_dict = defaultdict(list)
    for span in tenant_filtered:
        traces_dict[span.trace_id].append(span)
    
    # Get allowed attribute prefixes for role
    allowed_prefixes = auth_service.get_allowed_attr_prefixes(user.role_group)
    
    # Get settings for trace configuration
    settings = get_settings()
    span_depth = settings.trace_span_depth.lower()
    enabled_categories = settings.enabled_telemetry_categories
    
    # Get feedback store for enriching traces with feedback data
    feedback_store = get_feedback_store()
    
    # Build hierarchical traces from in-memory spans
    in_memory_trace_ids = set(traces_dict.keys())
    traces = []
    
    for trace_id in traces_dict.keys():
        spans = traces_dict[trace_id]
        span_map = {}
        
        for span in spans:
            # Format events with LLM reasoning data
            formatted_events = []
            for event in span.events:
                event_data = {
                    "name": event.get("name"),
                    "time": event.get("timestamp"),
                }
                # Include LLM reasoning attributes if present
                attrs = event.get("attributes", {})
                if attrs:
                    event_data.update({
                        "tenant_id": attrs.get("tenant_id"),
                        "LLM.model": attrs.get("LLM.model"),
                        "LLM.reasoning_step": attrs.get("LLM.reasoning_step"),
                        "LLM.confidence": attrs.get("LLM.confidence"),
                        "LLM.hallucination_risk": attrs.get("LLM.hallucination_risk"),
                        "LLM.semantic_similarity": attrs.get("LLM.semantic_similarity"),
                        "LLM.prompt_consistency": attrs.get("LLM.prompt_consistency"),
                        "LLM.justification": attrs.get("LLM.justification"),
                        "LLM.answer_snippet": attrs.get("LLM.answer_snippet"),
                    })
                formatted_events.append(event_data)
            
            # Filter attributes by role and category
            filtered_attrs = _filter_attributes_by_role_and_category(
                span.attributes, 
                allowed_prefixes,
                enabled_categories
            )
            
            # Enrich with feedback data if conversation_id exists
            conversation_id = span.attributes.get("conversation_id") or span.attributes.get("rag.conversation_id")
            if conversation_id:
                feedback_attrs = feedback_store.get_feedback_attributes_for_conversation(conversation_id)
                filtered_attrs.update(feedback_attrs)
            
            span_dict = {
                "trace_id": span.trace_id,
                "span_id": span.span_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "kind": "INTERNAL",  # Default kind
                "status": span.status,
                "attributes": filtered_attrs,
                "events": formatted_events,
                "children": [],
                "source": "memory",
            }
            span_map[span_dict["span_id"]] = span_dict
        
        # Build parent-child hierarchy
        roots = []
        for span_dict in span_map.values():
            parent_id = span_dict["parent_span_id"]
            if parent_id and parent_id in span_map:
                span_map[parent_id]["children"].append(span_dict)
            else:
                roots.append(span_dict)
        
        # Apply span depth filtering
        filtered_roots = _apply_span_depth_filter(roots, span_depth)
        
        traces.append({
            "trace_id": trace_id,
            "spans": filtered_roots,
            "source": "memory",
        })
    
    # Add MongoDB traces that are not in memory (for persistence across restarts)
    if db_traces:
        for db_trace in db_traces:
            # Skip if already in memory
            if db_trace.trace_id in in_memory_trace_ids:
                continue
            
            # Convert MongoDB trace to response format
            db_spans = []
            for span_data in db_trace.spans:
                # Filter attributes
                filtered_attrs = _filter_attributes_by_role_and_category(
                    span_data.get("attributes", {}),
                    allowed_prefixes,
                    enabled_categories
                )
                
                # Enrich with feedback data
                conversation_id = span_data.get("attributes", {}).get("conversation_id") or \
                                  span_data.get("attributes", {}).get("rag.conversation_id")
                if conversation_id:
                    feedback_attrs = feedback_store.get_feedback_attributes_for_conversation(conversation_id)
                    filtered_attrs.update(feedback_attrs)
                
                db_spans.append({
                    "trace_id": db_trace.trace_id,
                    "span_id": span_data.get("span_id"),
                    "parent_span_id": span_data.get("parent_span_id"),
                    "name": span_data.get("name"),
                    "kind": "INTERNAL",
                    "status": span_data.get("status", "OK"),
                    "attributes": filtered_attrs,
                    "events": span_data.get("events", []),
                    "children": [],
                    "source": "mongodb",
                })
            
            # Build hierarchy for DB spans
            span_map = {s["span_id"]: s for s in db_spans}
            roots = []
            for span_dict in span_map.values():
                parent_id = span_dict["parent_span_id"]
                if parent_id and parent_id in span_map:
                    span_map[parent_id]["children"].append(span_dict)
                else:
                    roots.append(span_dict)
            
            filtered_roots = _apply_span_depth_filter(roots, span_depth)
            
            traces.append({
                "trace_id": db_trace.trace_id,
                "spans": filtered_roots,
                "source": "mongodb",
                "timestamp": db_trace.timestamp,
            })
    
    # Sort all traces by timestamp (most recent first)
    # In-memory traces come first (source_priority=0), then MongoDB traces sorted by timestamp
    def get_trace_sort_key(t):
        source_priority = 0 if t.get("source") == "memory" else 1
        timestamp = t.get("timestamp", "")
        # Convert timestamp string to sortable format (ISO format strings sort correctly)
        if isinstance(timestamp, str) and timestamp:
            return (source_priority, timestamp)
        return (source_priority, "9999-99-99")  # Put traces without timestamp at end
    
    traces.sort(key=get_trace_sort_key, reverse=False)  # Memory first, then DB by timestamp desc
    
    # Pagination
    total = len(traces)
    start, end = (page - 1) * limit, (page - 1) * limit + limit
    paginated_traces = traces[start:end]
    
    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": (total + limit - 1) // limit,
        "data": paginated_traces,
        "sources": {
            "memory": len([t for t in traces if t.get("source") == "memory"]),
            "mongodb": len([t for t in traces if t.get("source") == "mongodb"]),
        },
    }


@router.get("/logs")
async def get_logs(
    level: Optional[str] = Query(default=None),
    limit: int = Query(default=100, le=1000),
    user: UserIdentity = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """
    Get recent logs.
    
    Returns logs from the in-memory buffer.
    Filtered by tenant_id for non-admin users.
    """
    # Non-admins can only see their own tenant's logs
    tenant_filter = None if user.is_admin else user.tenant_id
    
    logs = get_log_buffer(
        level=level,
        tenant_id=tenant_filter,
        limit=limit,
    )
    
    return [log.to_dict() for log in logs]


@router.delete("/traces")
async def clear_traces(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Clear all stored traces.
    
    Admin only.
    """
    if not user.is_admin:
        return {"error": "Admin access required", "cleared": 0}
    
    count = clear_spans()
    logger.info(f"Cleared {count} traces (by {user.username})")
    
    return {"cleared": count}


@router.delete("/logs")
async def clear_logs(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Clear all stored logs.
    
    Admin only.
    """
    if not user.is_admin:
        return {"error": "Admin access required", "cleared": 0}
    
    count = clear_log_buffer()
    logger.info(f"Cleared {count} logs (by {user.username})")
    
    return {"cleared": count}


@router.get("/stats")
async def observability_stats(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Return all observability metrics in chart-friendly format.
    All values are returned as arrays (last 15 data points) for line chart visualization.
    Also includes current totals and averages.
    """
    import psutil
    
    tenant_id = str(user.tenant_id).lower()
    
    # Initialize default values
    total_results_generated = 0
    total_tokens = 0
    total_tokens_input = 0
    total_tokens_output = 0
    today_tokens = 0
    today_results = 0
    total_feedback = 0
    positive_feedback = 0
    negative_feedback = 0
    trace_count = 0
    
    # Chart data defaults
    labels = []
    relevancy_data = []
    faithfulness_data = []
    confidence_data = []
    hallucination_data = []
    latency_data = []
    tokens_input_data = []
    tokens_output_data = []
    
    try:
        eval_repo = get_evaluation_repository()
        trace_repo = get_trace_repository()
        feedback_repo = get_feedback_repository()
        
        # Get comprehensive evaluation stats for this tenant from MongoDB
        eval_stats = eval_repo.get_evaluation_stats(tenant_id)
        if eval_stats:
            total_results_generated = eval_stats.get("total", 0)
            total_tokens_input = eval_stats.get("total_tokens_input", 0)
            total_tokens_output = eval_stats.get("total_tokens_output", 0)
            total_tokens = eval_stats.get("total_tokens", 0)
        
        # Get today's stats
        today_stats = eval_repo.get_today_stats(tenant_id)
        if today_stats:
            today_results = today_stats.get("total_results", 0)
            today_tokens = today_stats.get("total_tokens", 0)
        
        # Get feedback stats from MongoDB
        feedback_stats = feedback_repo.get_feedback_stats(tenant_id)
        logger.debug(f"Feedback stats for tenant {tenant_id}: {feedback_stats}")
        total_feedback = feedback_stats.get("total", 0)
        positive_feedback = feedback_stats.get("positive", 0)
        negative_feedback = feedback_stats.get("negative", 0)
        
        # Get trace count for this tenant
        trace_count = trace_repo.count_by_tenant(tenant_id)
        
        # Get historical chart data (last 15 evaluations)
        chart_data = eval_repo.get_historical_chart_data(tenant_id, limit=15)
        labels = chart_data.get("timestamps", [])
        relevancy_data = chart_data.get("relevancy", [])
        faithfulness_data = chart_data.get("faithfulness", [])
        confidence_data = chart_data.get("confidence", [])
        hallucination_data = chart_data.get("hallucination", [])
        latency_data = chart_data.get("latency", [])
        tokens_input_data = chart_data.get("tokens_input", [])
        tokens_output_data = chart_data.get("tokens_output", [])
        
    except Exception as e:
        logger.warning(f"Failed to get DB stats: {e}")
    
    # --- System metrics ---
    cpu_usage = psutil.cpu_percent(interval=0.2)
    memory_usage = psutil.virtual_memory().percent
    disk_io = psutil.disk_io_counters()
    net_io = psutil.net_io_counters()
    disk_io_mb = round(disk_io.write_bytes / 1024 / 1024, 2)
    network_mb = round(net_io.bytes_sent / 1024 / 1024, 2)
    
    # Add current system reading to history
    _add_system_reading(tenant_id, cpu_usage, memory_usage, disk_io_mb, network_mb)
    
    # Get system performance history for charts
    system_history = _get_system_history(tenant_id)
    
    # Calculate feedback ratio
    feedback_ratio = round(positive_feedback / total_feedback, 3) if total_feedback > 0 else 0.0
    
    # Calculate current averages from chart data
    avg_relevancy = round(sum(relevancy_data) / len(relevancy_data), 3) if relevancy_data else 0.0
    avg_faithfulness = round(sum(faithfulness_data) / len(faithfulness_data), 3) if faithfulness_data else 0.0
    avg_confidence = round(sum(confidence_data) / len(confidence_data), 3) if confidence_data else 0.0
    avg_hallucination = round(sum(hallucination_data) / len(hallucination_data), 3) if hallucination_data else 0.0
    avg_latency = round(sum(latency_data) / len(latency_data), 3) if latency_data else 0.0
    
    # Calculate system averages
    cpu_data = system_history.get("cpu_usage", [])
    memory_data = system_history.get("memory_usage", [])
    disk_data = system_history.get("disk_io_mb", [])
    network_data = system_history.get("network_mb", [])
    
    avg_cpu = round(sum(cpu_data) / len(cpu_data), 1) if cpu_data else 0.0
    avg_memory = round(sum(memory_data) / len(memory_data), 1) if memory_data else 0.0
    
    # --- Response in chart-friendly format ---
    return {
        "tenant_id": tenant_id,
        "labels": labels,  # X-axis labels (timestamps HH:MM)
        
        # Quality metrics as chart arrays
        "relevancy": {
            "data": relevancy_data,
            "current": relevancy_data[-1] if relevancy_data else 0,
            "average": avg_relevancy,
        },
        "faithfulness": {
            "data": faithfulness_data,
            "current": faithfulness_data[-1] if faithfulness_data else 0,
            "average": avg_faithfulness,
        },
        "confidence": {
            "data": confidence_data,
            "current": confidence_data[-1] if confidence_data else 0,
            "average": avg_confidence,
        },
        "hallucination": {
            "data": hallucination_data,
            "current": hallucination_data[-1] if hallucination_data else 0,
            "average": avg_hallucination,
        },
        "latency": {
            "data": latency_data,
            "current": latency_data[-1] if latency_data else 0,
            "average": avg_latency,
        },
        "tokens_input": {
            "data": tokens_input_data,
            "current": tokens_input_data[-1] if tokens_input_data else 0,
            "total": total_tokens_input,
        },
        "tokens_output": {
            "data": tokens_output_data,
            "current": tokens_output_data[-1] if tokens_output_data else 0,
            "total": total_tokens_output,
        },
        
        # Summary totals
        "totals": {
            "results_generated": total_results_generated,
            "tokens": total_tokens,
            "tokens_input": total_tokens_input,
            "tokens_output": total_tokens_output,
            "today_results": today_results,
            "today_tokens": today_tokens,
            "traces": trace_count,
            "feedback": total_feedback,
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "feedback_ratio": feedback_ratio,
        },
        
        # System performance in chart format
        "cpu_usage": {
            "data": cpu_data,
            "current": cpu_usage,
            "average": avg_cpu,
        },
        "memory_usage": {
            "data": memory_data,
            "current": memory_usage,
            "average": avg_memory,
        },
        "disk_io": {
            "data": disk_data,
            "current": disk_io_mb,
        },
        "network": {
            "data": network_data,
            "current": network_mb,
        },
        
        # System labels (separate from quality labels since they're collected at different times)
        "system_labels": system_history.get("timestamps", []),
        
        # Data point counts
        "data_points": len(labels),
        "system_data_points": len(cpu_data),
    }


@router.post("/reset")
async def reset_observability(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, str]:
    """Clear all in-memory telemetry data. Admin only."""
    if not user.is_admin:
        return {"message": "Admin access required"}
    
    clear_spans()
    clear_log_buffer()
    return {"message": "All in-memory telemetry data cleared."}


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check for observability service.
    """
    return {"status": "healthy", "service": "observability"}


@router.get("/prometheus", response_class=PlainTextResponse)
async def prometheus_metrics(
    user: UserIdentity = Depends(get_current_user),
) -> str:
    """
    Export metrics in Prometheus format.
    
    Returns tenant-specific metrics including function calls, HTTP status, and request details.
    Excludes generic requests like login, health checks, and observability endpoints.
    """
    try:
        from prometheus_client import generate_latest, CollectorRegistry
        from prometheus_client import Counter, Gauge, Histogram
        
        # Get metrics provider to check if metrics are enabled
        from app.observability.metrics import get_metrics_provider
        metrics_provider = get_metrics_provider()
        
        if not metrics_provider or not metrics_provider.enabled:
            return "# Metrics are disabled\n"
        
        # Create tenant-specific registry
        tenant_registry = CollectorRegistry()
        
        # Get tenant-specific spans, excluding generic requests
        all_tenant_spans = get_recent_spans(limit=1000, tenant_id=user.tenant_id)
        
        # Filter out generic requests
        EXCLUDED_PATTERNS = ['login', 'health', '/metrics', '/prometheus', '/docs', '/openapi']
        tenant_spans = [
            span for span in all_tenant_spans
            if not any(pattern in span.name.lower() or 
                      pattern in str(span.attributes.get('http.target', '')).lower()
                      for pattern in EXCLUDED_PATTERNS)
        ]
        
        # Metrics by request type and tenant
        requests_by_type = Counter(
            'tenant_requests_total',
            'Total requests by type for tenant',
            ['tenant_id', 'request_type', 'http_method', 'http_status'],
            registry=tenant_registry
        )
        
        # Function call metrics
        function_calls = Counter(
            'tenant_function_calls_total',
            'Total function calls for tenant',
            ['tenant_id', 'function_name', 'status'],
            registry=tenant_registry
        )
        
        # Response time by request type
        response_time_histogram = Histogram(
            'tenant_response_time_seconds',
            'Response time distribution by request type',
            ['tenant_id', 'request_type'],
            registry=tenant_registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # HTTP status code distribution
        http_status_counter = Counter(
            'tenant_http_status_total',
            'HTTP status code distribution for tenant',
            ['tenant_id', 'status_code', 'http_method', 'http_route'],
            registry=tenant_registry
        )
        
        # Error counter
        errors_counter = Counter(
            'tenant_errors_total',
            'Total errors for tenant',
            ['tenant_id', 'error_type', 'function_name'],
            registry=tenant_registry
        )
        
        # HTTP transfer size metrics
        http_request_size_bytes = Histogram(
            'tenant_http_request_size_bytes',
            'HTTP request size distribution for tenant',
            ['tenant_id', 'http_method', 'http_route'],
            registry=tenant_registry,
            buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000)
        )
        
        http_response_size_bytes = Histogram(
            'tenant_http_response_size_bytes',
            'HTTP response size distribution for tenant',
            ['tenant_id', 'http_method', 'http_route', 'http_status'],
            registry=tenant_registry,
            buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000)
        )
        
        # Total transfer size counter
        http_transfer_bytes_total = Counter(
            'tenant_http_transfer_bytes_total',
            'Total HTTP bytes transferred for tenant',
            ['tenant_id', 'direction', 'http_method', 'http_route'],
            registry=tenant_registry
        )
        
        # Populate metrics from spans
        for span in tenant_spans:
            # Determine request type
            request_type = 'other'
            if 'LLM' in span.name or 'LLM.prompt' in span.attributes:
                request_type = 'llm'
            elif 'RAG' in span.name or 'RAG.query' in span.name:
                request_type = 'rag'
            elif span.attributes.get('http.method'):
                request_type = 'http'
            
            # Get HTTP details with better handler detection
            http_method = str(span.attributes.get('http.method', 'UNKNOWN'))
            http_status = str(span.attributes.get('http.status_code', '0'))
            
            # Try multiple sources for the route/handler name
            http_route = (
                span.attributes.get('http.route') or 
                span.attributes.get('http.target') or
                span.attributes.get('http.url') or
                span.attributes.get('http.path') or
                span.name  # Fallback to span name which often contains the handler
            )
            
            # Clean up the route name to show only the last segment
            if http_route:
                http_route = str(http_route)
                # Extract handler name from common patterns
                if 'GET' in http_route or 'POST' in http_route:
                    # Remove HTTP method prefix if present
                    http_route = http_route.split(' ', 1)[-1] if ' ' in http_route else http_route
                # Remove query parameters
                http_route = http_route.split('?')[0]
                # Get only the last segment of the path
                http_route = http_route.rstrip('/')
                if '/' in http_route:
                    http_route = http_route.split('/')[-1]
                if not http_route:
                    http_route = '/'
                # Ensure it starts with / for root
                if http_route == '/':
                    http_route = '/'
                elif not http_route.startswith('/'):
                    http_route = '/' + http_route
            else:
                http_route = '/unknown'
            
            # Count requests by type
            requests_by_type.labels(
                tenant_id=user.tenant_id,
                request_type=request_type,
                http_method=http_method,
                http_status=http_status
            ).inc()
            
            # Count HTTP status codes
            if http_status != '0':
                http_status_counter.labels(
                    tenant_id=user.tenant_id,
                    status_code=http_status,
                    http_method=http_method,
                    http_route=http_route
                ).inc()
            
            # Function call metrics
            function_name = span.attributes.get('function.name', span.name)
            status = 'success' if span.status != 'ERROR' else 'error'
            function_calls.labels(
                tenant_id=user.tenant_id,
                function_name=function_name,
                status=status
            ).inc()
            
            # Response time
            if span.start_time and span.end_time:
                duration = (span.end_time - span.start_time) / 1_000_000_000
                response_time_histogram.labels(
                    tenant_id=user.tenant_id,
                    request_type=request_type
                ).observe(duration)
            
            # Error tracking
            if span.status == 'ERROR':
                error_type = span.attributes.get('error.type', 'unknown')
                errors_counter.labels(
                    tenant_id=user.tenant_id,
                    error_type=error_type,
                    function_name=function_name
                ).inc()
            
            # HTTP transfer size tracking
            if span.attributes.get('http.method'):
                # Get request and response sizes from span attributes
                request_size = span.attributes.get('http.request_content_length', 0)
                response_size = span.attributes.get('http.response_content_length', 0)
                
                # Record request size
                if request_size and request_size > 0:
                    http_request_size_bytes.labels(
                        tenant_id=user.tenant_id,
                        http_method=http_method,
                        http_route=http_route
                    ).observe(request_size)
                    
                    http_transfer_bytes_total.labels(
                        tenant_id=user.tenant_id,
                        direction='request',
                        http_method=http_method,
                        http_route=http_route
                    )._value._value += request_size
                
                # Record response size
                if response_size and response_size > 0:
                    http_response_size_bytes.labels(
                        tenant_id=user.tenant_id,
                        http_method=http_method,
                        http_route=http_route,
                        http_status=http_status
                    ).observe(response_size)
                    
                    http_transfer_bytes_total.labels(
                        tenant_id=user.tenant_id,
                        direction='response',
                        http_method=http_method,
                        http_route=http_route
                    )._value._value += response_size
        
        # Add system metrics
        import psutil
        cpu_gauge = Gauge('system_cpu_percent', 'System CPU usage', registry=tenant_registry)
        memory_gauge = Gauge('system_memory_percent', 'System memory usage', registry=tenant_registry)
        
        cpu_gauge.set(psutil.cpu_percent())
        memory_gauge.set(psutil.virtual_memory().percent)
        
        # Generate metrics
        metrics_data = generate_latest(tenant_registry)
        
        # Log access
        logger.info(f"Prometheus metrics accessed by {user.username} (tenant: {user.tenant_id})")
        
        return metrics_data.decode('utf-8')
        
    except ImportError:
        return "# Prometheus client not installed\n"
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        return f"# Error generating metrics: {str(e)}\n"


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint(
    user: UserIdentity = Depends(get_current_user),
) -> str:
    """
    Standard Prometheus metrics endpoint.
    
    This is the conventional /metrics endpoint that Prometheus servers expect.
    Returns all collected metrics in the Prometheus text format.
    """
    # Delegate to the prometheus endpoint for consistency
    return await prometheus_metrics(user)


@router.get("/metrics/summary")
async def metrics_summary(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a summary of key metrics for the current tenant.
    
    Returns tenant-specific metrics plus generic system metrics in JSON format.
    """
    import psutil
    import time
    from collections import defaultdict
    
    tenant_id = user.tenant_id
    
    # Get recent spans for tenant-specific metrics only
    all_tenant_spans = get_recent_spans(limit=1000, tenant_id=tenant_id)
    
    # Filter out generic requests (login, health, metrics endpoints)
    EXCLUDED_PATTERNS = ['login', 'health', '/metrics', '/prometheus', '/docs', '/openapi']
    tenant_spans = [
        span for span in all_tenant_spans
        if not any(pattern in span.name.lower() or 
                  pattern in str(span.attributes.get('http.target', '')).lower()
                  for pattern in EXCLUDED_PATTERNS)
    ]
    
    # Calculate tenant-specific metrics
    total_requests = len(tenant_spans)
    llm_requests = len([
        span for span in tenant_spans 
        if "LLM.prompt" in span.attributes or 'LLM' in span.name
    ])
    rag_requests = len([
        span for span in tenant_spans 
        if "RAG.query" in span.name or 'RAG' in span.name
    ])
    http_requests = len([
        span for span in tenant_spans
        if span.attributes.get('http.method')
    ])
    
    # HTTP status code breakdown
    http_status_breakdown = {}
    for span in tenant_spans:
        status_code = span.attributes.get('http.status_code')
        if status_code:
            status_code = str(status_code)
            http_status_breakdown[status_code] = http_status_breakdown.get(status_code, 0) + 1
    
    # Better handler/endpoint breakdown
    handler_breakdown = {}
    for span in tenant_spans:
        # Get handler name with multiple fallbacks
        handler_name = (
            span.attributes.get('http.route') or 
            span.attributes.get('http.target') or
            span.attributes.get('http.url') or
            span.attributes.get('http.path') or
            span.name
        )
        
        # Clean up handler name to show only the last segment
        if handler_name:
            handler_name = str(handler_name)
            # Remove HTTP method prefix if present
            if 'GET' in handler_name or 'POST' in handler_name:
                handler_name = handler_name.split(' ', 1)[-1] if ' ' in handler_name else handler_name
            # Remove query parameters
            handler_name = handler_name.split('?')[0]
            # Get only the last segment of the path
            handler_name = handler_name.rstrip('/')
            if '/' in handler_name:
                handler_name = handler_name.split('/')[-1]
            if not handler_name:
                handler_name = '/'
            # Ensure it starts with / for root
            if handler_name == '/':
                handler_name = '/'
            elif not handler_name.startswith('/'):
                handler_name = '/' + handler_name
            
            handler_breakdown[handler_name] = handler_breakdown.get(handler_name, 0) + 1
    
    # Function call breakdown
    function_calls_breakdown = {}
    for span in tenant_spans:
        func_name = span.attributes.get('function.name', span.name)
        if func_name:
            function_calls_breakdown[func_name] = function_calls_breakdown.get(func_name, 0) + 1
    
    # HTTP transfer size metrics
    total_request_bytes = 0
    total_response_bytes = 0
    transfer_size_by_route = {}
    
    for span in tenant_spans:
        if span.attributes.get('http.method'):
            request_size = span.attributes.get('http.request_content_length', 0) or 0
            response_size = span.attributes.get('http.response_content_length', 0) or 0
            
            # Use the same improved route detection
            route = (
                span.attributes.get('http.route') or 
                span.attributes.get('http.target') or
                span.attributes.get('http.url') or
                span.attributes.get('http.path') or
                span.name
            )
            
            if route:
                route = str(route)
                # Remove HTTP method prefix if present
                if 'GET' in route or 'POST' in route:
                    route = route.split(' ', 1)[-1] if ' ' in route else route
                # Remove query parameters
                route = route.split('?')[0]
                # Get only the last segment of the path
                route = route.rstrip('/')
                if '/' in route:
                    route = route.split('/')[-1]
                if not route:
                    route = '/'
                # Ensure it starts with / for root
                if route == '/':
                    route = '/'
                elif not route.startswith('/'):
                    route = '/' + route
            else:
                route = '/unknown'
            
            total_request_bytes += request_size
            total_response_bytes += response_size
            
            if route not in transfer_size_by_route:
                transfer_size_by_route[route] = {'request_bytes': 0, 'response_bytes': 0, 'total_bytes': 0}
            
            transfer_size_by_route[route]['request_bytes'] += request_size
            transfer_size_by_route[route]['response_bytes'] += response_size
            transfer_size_by_route[route]['total_bytes'] += request_size + response_size
    
    # Calculate average response times for tenant
    response_times = []
    for span in tenant_spans:
        if span.start_time and span.end_time:
            duration = (span.end_time - span.start_time) / 1_000_000_000  # Convert to seconds
            response_times.append(duration)
    
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Generic system metrics (not tenant-specific)
    cpu_usage = psutil.cpu_percent(interval=0.1)
    memory_usage = psutil.virtual_memory().percent
    
    # Tenant-specific error rate
    error_spans = len([
        span for span in tenant_spans 
        if span.status and span.status == "ERROR"
    ])
    error_rate = (error_spans / total_requests * 100) if total_requests > 0 else 0
    
    # Additional system metrics
    import sys
    import os
    import datetime
    
    # Memory details
    memory_info = psutil.virtual_memory()
    swap_info = psutil.swap_memory()
    
    # Process information
    process = psutil.Process()
    process_memory = process.memory_info()
    process_cpu_times = process.cpu_times()
    
    # System information
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    
    # File descriptors
    try:
        file_descriptors = process.num_fds()
        max_file_descriptors = psutil.os.getrlimit(psutil.os.RLIMIT_NOFILE)[1]
        fd_percentage = (file_descriptors / max_file_descriptors * 100) if max_file_descriptors > 0 else 0
    except (AttributeError, OSError):
        file_descriptors = 0
        max_file_descriptors = 0
        fd_percentage = 0
    
    return {
        "tenant_id": tenant_id,
        "timestamp": int(time.time()),
        "runtime_environment": {
            "python_version": sys.version,
            "python_implementation": sys.implementation.name if hasattr(sys, 'implementation') else "Unknown",
            "python_version_info": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "system_health": {
            "status": "Healthy" if cpu_usage < 90 and memory_usage < 90 else "Warning",
            "uptime_seconds": round(uptime_seconds, 2),
            "uptime_formatted": str(datetime.timedelta(seconds=int(uptime_seconds))),
        },
        "memory_usage": {
            "virtual": {
                "total": memory_info.total,
                "available": memory_info.available,
                "used": memory_info.used,
                "percentage": memory_info.percent,
                "formatted": {
                    "total": f"{memory_info.total / (1024**3):.2f} GB",
                    "used": f"{memory_info.used / (1024**3):.2f} GB",
                    "available": f"{memory_info.available / (1024**3):.2f} GB",
                }
            },
            "resident": {
                "rss": process_memory.rss,
                "vms": process_memory.vms,
                "formatted": {
                    "rss": f"{process_memory.rss / (1024**2):.2f} MB",
                    "vms": f"{process_memory.vms / (1024**2):.2f} MB",
                }
            },
            "memory_efficiency": round((memory_info.available / memory_info.total) * 100, 2),
            "swap": {
                "total": swap_info.total,
                "used": swap_info.used,
                "percentage": swap_info.percent,
                "formatted": {
                    "total": f"{swap_info.total / (1024**3):.2f} GB",
                    "used": f"{swap_info.used / (1024**3):.2f} GB",
                }
            }
        },
        "file_descriptors": {
            "open": file_descriptors,
            "max": max_file_descriptors,
            "percentage": round(fd_percentage, 2),
        },
        "cpu_time": {
            "user_time": round(process_cpu_times.user, 2),
            "system_time": round(process_cpu_times.system, 2),
            "total_cpu_time": round(process_cpu_times.user + process_cpu_times.system, 2),
            "cpu_percentage": cpu_usage,
        },
        "request_metrics": {
            "total_requests": total_requests,
            "llm_requests": llm_requests,
            "rag_requests": rag_requests,
            "http_requests": http_requests,
            "avg_response_time_seconds": round(avg_response_time, 3),
            "error_rate_percent": round(error_rate, 2),
            "http_status_breakdown": http_status_breakdown,
            "handler_breakdown": dict(sorted(handler_breakdown.items(), key=lambda x: x[1], reverse=True)[:20]),  # Top 20 handlers
            "function_calls_breakdown": dict(sorted(function_calls_breakdown.items(), key=lambda x: x[1], reverse=True)[:20]),  # Top 20 functions
            "transfer_metrics": {
                "total_request_bytes": total_request_bytes,
                "total_response_bytes": total_response_bytes,
                "total_transfer_bytes": total_request_bytes + total_response_bytes,
                "transfer_size_by_route": dict(sorted(transfer_size_by_route.items(), key=lambda x: x[1]['total_bytes'], reverse=True)[:10]),  # Top 10 routes by transfer
            },
        },
        "status": "success",
    }


def _get_attribute_category(key: str) -> str:
    """
    Determine the category of an attribute based on its prefix.
    
    Categories:
    - basic: tenant_id, http.*, service.*, etc.
    - llm: LLM.* attributes (model, prompt, response, etc.)
    - deepeval: DeepEval.* attributes (metrics from deepeval library)
    - rag: RAG.* attributes (retrieval, context, etc.)
    - quality: Quality metrics (hallucination, semantic_similarity, etc.)
    - system: System-level attributes
    """
    key_lower = key.lower()
    
    if key_lower.startswith("llm."):
        return "llm"
    elif key_lower.startswith("deepeval."):
        return "deepeval"
    elif key_lower.startswith("rag."):
        return "rag"
    elif "hallucination" in key_lower or "semantic_similarity" in key_lower or "confidence" in key_lower:
        return "quality"
    elif key_lower.startswith("http.") or key_lower.startswith("service.") or key in ["tenant_id", "trace_id", "span_id"]:
        return "basic"
    elif key_lower.startswith("system.") or key_lower.startswith("process."):
        return "system"
    else:
        return "basic"  # Default to basic


def _filter_attributes_by_role_and_category(
    attributes: Dict[str, Any],
    allowed_prefixes: List[str],
    enabled_categories: List[str],
) -> Dict[str, Any]:
    """
    Filter span attributes based on role group AND enabled categories.
    
    Role-based filtering (legacy logic):
    - admin: Only Admin.* attributes
    - manager: Only Manager.* attributes  
    - developer: All attributes EXCEPT Admin.* and Manager.*
    
    Category-based filtering:
    - Only include attributes from enabled categories
    
    Args:
        attributes: Original attributes
        allowed_prefixes: List of allowed prefixes (e.g., ["Admin", "Manager"])
        enabled_categories: List of enabled categories (e.g., ["basic", "llm"])
        
    Returns:
        Filtered attributes as strings
    """
    if not attributes:
        return {}
    
    result = {}
    
    # Determine role group from allowed_prefixes
    is_admin = "Admin" in allowed_prefixes
    is_manager = "Manager" in allowed_prefixes and not is_admin
    is_developer = not is_admin and not is_manager
    
    for key, value in attributes.items():
        key_str = str(key)
        
        # First apply role-based filtering
        role_allowed = False
        if is_admin:
            # Admin sees ONLY Admin.* prefixed attributes
            if key_str.startswith("Admin.") or key_str.startswith("Admin_"):
                role_allowed = True
        elif is_manager:
            # Manager sees ONLY Manager.* prefixed attributes
            if key_str.startswith("Manager.") or key_str.startswith("Manager_"):
                role_allowed = True
        else:  # developer
            # Developer sees everything EXCEPT Admin.* and Manager.* prefixed attributes
            if not (key_str.startswith("Admin.") or key_str.startswith("Admin_") or 
                    key_str.startswith("Manager.") or key_str.startswith("Manager_")):
                role_allowed = True
        
        if not role_allowed:
            continue
        
        # Then apply category filtering
        category = _get_attribute_category(key_str)
        if category in enabled_categories:
            result[key_str] = str(value)
    
    return result


def _apply_span_depth_filter(spans: List[Dict[str, Any]], depth_mode: str) -> List[Dict[str, Any]]:
    """
    Filter spans based on depth mode.
    
    Modes:
    - 'parent': Only root spans (spans with no parent_span_id), no children shown
    - 'has_child': Only spans that have children at any depth level, keeps their children
    - 'complete': All spans with full hierarchy (default)
    
    Args:
        spans: List of span dictionaries with children (hierarchical structure)
        depth_mode: Depth filtering mode
        
    Returns:
        Filtered spans
    """
    if depth_mode == "complete":
        return spans
    
    if depth_mode == "parent":
        # Only show root spans (those without parent_span_id), strip all children
        result = []
        for span in spans:
            # Root spans are those with no parent_span_id or parent_span_id is None
            if not span.get("parent_span_id"):
                result.append({**span, "children": []})
        return result
    
    elif depth_mode == "has_child":
        # Only show spans that have children at any depth
        # Keep the children of those spans so we can see the hierarchy
        def filter_span(span: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            children = span.get("children", [])
            
            # Recursively filter children first
            filtered_children = []
            for child in children:
                filtered_child = filter_span(child)
                if filtered_child:
                    filtered_children.append(filtered_child)
            
            # Check if this span originally had children (before filtering)
            has_children = len(children) > 0
            
            if not has_children:
                # This is a leaf node - exclude it
                return None
            
            # This span has children, so include it with its filtered children
            return {**span, "children": filtered_children}
        
        result = []
        for span in spans:
            filtered = filter_span(span)
            if filtered:
                result.append(filtered)
        
        return result
    
    return spans


def _filter_attributes(
    attributes: Dict[str, Any],
    allowed_prefixes: List[str],
) -> Dict[str, Any]:
    """
    Filter span attributes based on role group (legacy compatibility).
    
    Matches legacy logic:
    - admin: Only Admin.* attributes
    - manager: Only Manager.* attributes  
    - developer: All attributes EXCEPT Admin.* and Manager.*
    
    Args:
        attributes: Original attributes
        allowed_prefixes: List of allowed prefixes (e.g., ["Admin", "Manager"])
        
    Returns:
        Filtered attributes as strings
    """
    if not attributes:
        return {}
    
    result = {}
    
    # Determine role group from allowed_prefixes
    is_admin = "Admin" in allowed_prefixes
    is_manager = "Manager" in allowed_prefixes and not is_admin
    is_developer = not is_admin and not is_manager
    
    for key, value in attributes.items():
        key_str = str(key)
        
        if is_admin:
            # Admin sees ONLY Admin.* prefixed attributes
            if key_str.startswith("Admin.") or key_str.startswith("Admin_"):
                result[key_str] = str(value)
        elif is_manager:
            # Manager sees ONLY Manager.* prefixed attributes
            if key_str.startswith("Manager.") or key_str.startswith("Manager_"):
                result[key_str] = str(value)
        else:  # developer
            # Developer sees everything EXCEPT Admin.* and Manager.* prefixed attributes
            if not (key_str.startswith("Admin.") or key_str.startswith("Admin_") or 
                    key_str.startswith("Manager.") or key_str.startswith("Manager_")):
                result[key_str] = str(value)
    
    return result


# ============================================================
# === MongoDB-based Endpoints ===
# ============================================================

@router.get("/db/traces")
async def get_db_traces(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get traces from MongoDB.
    
    Returns persisted traces with full span data.
    Filtered by tenant_id for non-admin users.
    """
    try:
        trace_repo = get_trace_repository()
        
        skip = (page - 1) * limit
        
        if user.is_admin:
            traces = trace_repo.list_recent(limit=limit)
            total = trace_repo.count_by_tenant(user.tenant_id)  # Approximate
        else:
            traces = trace_repo.list_by_tenant(user.tenant_id, limit=limit, skip=skip)
            total = trace_repo.count_by_tenant(user.tenant_id)
        
        return {
            "traces": [t.to_dict() for t in traces],
            "total": total,
            "page": page,
            "limit": limit,
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get traces from MongoDB: {e}")
        return {
            "traces": [],
            "total": 0,
            "page": page,
            "limit": limit,
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/traces/{trace_id}")
async def get_db_trace_by_id(
    trace_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get a specific trace from MongoDB by trace_id.
    """
    try:
        trace_repo = get_trace_repository()
        trace = trace_repo.get_by_id(trace_id)
        
        if trace is None:
            return {"error": "Trace not found", "trace_id": trace_id}
        
        # Check tenant access
        if not user.is_admin and trace.tenant_id != user.tenant_id:
            return {"error": "Access denied", "trace_id": trace_id}
        
        return {
            "trace": trace.to_dict(),
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get trace from MongoDB: {e}")
        return {"error": str(e), "trace_id": trace_id}


@router.get("/db/evaluations")
async def get_db_evaluations(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get LLM evaluations from MongoDB.
    
    Returns evaluation metrics for LLM responses including:
    - Self-reported metrics (confidence, hallucination risk)
    - Calculated metrics (semantic similarity, faithfulness)
    - Prompt similarity metrics
    """
    try:
        eval_repo = get_evaluation_repository()
        
        evaluations = eval_repo.list_by_tenant(user.tenant_id, limit=limit)
        stats = eval_repo.get_evaluation_stats(user.tenant_id)
        
        return {
            "evaluations": [e.to_dict() for e in evaluations],
            "stats": stats,
            "total": len(evaluations),
            "page": page,
            "limit": limit,
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get evaluations from MongoDB: {e}")
        return {
            "evaluations": [],
            "stats": {},
            "total": 0,
            "page": page,
            "limit": limit,
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/evaluations/{conversation_id}")
async def get_db_evaluation_by_conversation(
    conversation_id: str,
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get evaluation for a specific conversation from MongoDB.
    """
    try:
        eval_repo = get_evaluation_repository()
        evaluation = eval_repo.get_by_conversation_id(conversation_id)
        
        if evaluation is None:
            return {"error": "Evaluation not found", "conversation_id": conversation_id}
        
        # Check tenant access
        if not user.is_admin and evaluation.tenant_id != user.tenant_id:
            return {"error": "Access denied", "conversation_id": conversation_id}
        
        return {
            "evaluation": evaluation.to_dict(),
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get evaluation from MongoDB: {e}")
        return {"error": str(e), "conversation_id": conversation_id}


@router.get("/db/feedback")
async def get_db_feedback(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=50, ge=1, le=500),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get feedback from MongoDB.
    
    Returns user feedback with statistics.
    """
    try:
        feedback_repo = get_feedback_repository()
        
        if user.is_admin:
            feedbacks = feedback_repo.list_all(limit=limit)
        else:
            feedbacks = feedback_repo.list_by_tenant(user.tenant_id, limit=limit)
        
        stats = feedback_repo.get_feedback_stats(user.tenant_id if not user.is_admin else None)
        
        return {
            "feedbacks": [f.to_dict() for f in feedbacks],
            "stats": stats,
            "total": len(feedbacks),
            "page": page,
            "limit": limit,
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get feedback from MongoDB: {e}")
        return {
            "feedbacks": [],
            "stats": {},
            "total": 0,
            "page": page,
            "limit": limit,
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/prompts")
async def get_db_prompts(
    limit: int = Query(default=50, ge=1, le=500),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get stored prompts from MongoDB.
    
    Returns prompts with their quality scores and feedback.
    Used for prompt similarity analysis.
    """
    try:
        prompt_repo = get_prompt_repository()
        
        prompts = prompt_repo.get_recent_prompts(
            tenant_id=user.tenant_id if not user.is_admin else None,
            limit=limit,
        )
        
        total = prompt_repo.count_prompts(user.tenant_id if not user.is_admin else None)
        
        # Don't include embeddings in response (too large)
        prompts_data = []
        for p in prompts:
            data = p.to_dict()
            data.pop("embedding", None)  # Remove embedding from response
            prompts_data.append(data)
        
        return {
            "prompts": prompts_data,
            "total": total,
            "limit": limit,
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get prompts from MongoDB: {e}")
        return {
            "prompts": [],
            "total": 0,
            "limit": limit,
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/prompt-similarity")
async def compute_prompt_similarity_endpoint(
    prompt: str = Query(..., description="Prompt text to find similar prompts for"),
    top_k: int = Query(default=5, ge=1, le=20),
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Compute similarity between a prompt and stored prompts.
    
    Returns similar past prompts with their quality scores.
    Useful for predicting response quality based on historical data.
    """
    try:
        from app.rag.utils.prompt_similarity import get_prompt_similarity_service
        
        service = get_prompt_similarity_service()
        result = service.compute_similarity(
            prompt_text=prompt,
            tenant_id=user.tenant_id,
            top_k=top_k,
        )
        
        # Convert similar prompts to dicts
        similar_prompts = []
        for sp in result.similar_prompts:
            similar_prompts.append({
                "prompt_id": sp.prompt_id,
                "prompt_text": sp.prompt_text,
                "similarity_score": sp.similarity_score,
                "quality_score": sp.quality_score,
                "feedback_rating": sp.feedback_rating,
                "conversation_id": sp.conversation_id,
            })
        
        # Get quality prediction
        predicted_quality, confidence = service.get_quality_prediction(
            prompt_text=prompt,
            tenant_id=user.tenant_id,
        )
        
        return {
            "query_prompt": prompt[:200],  # Truncate for response
            "similar_prompts": similar_prompts,
            "metrics": {
                "total_prompts_searched": result.total_prompts_searched,
                "avg_similarity": result.avg_similarity,
                "max_similarity": result.max_similarity,
                "avg_quality_of_similar": result.avg_quality_of_similar,
                "positive_feedback_ratio": result.positive_feedback_ratio,
            },
            "quality_prediction": {
                "predicted_quality": predicted_quality,
                "confidence": confidence,
            },
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to compute prompt similarity: {e}")
        return {
            "query_prompt": prompt[:200],
            "similar_prompts": [],
            "metrics": {},
            "quality_prediction": {},
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/stats")
async def get_db_stats(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get aggregated statistics from MongoDB.
    
    Returns combined stats for traces, evaluations, feedback, and prompts.
    """
    try:
        trace_repo = get_trace_repository()
        eval_repo = get_evaluation_repository()
        feedback_repo = get_feedback_repository()
        prompt_repo = get_prompt_repository()
        
        tenant_id = user.tenant_id if not user.is_admin else None
        
        return {
            "traces": {
                "count": trace_repo.count_by_tenant(user.tenant_id),
            },
            "evaluations": eval_repo.get_evaluation_stats(tenant_id),
            "feedback": feedback_repo.get_feedback_stats(tenant_id),
            "prompts": {
                "count": prompt_repo.count_prompts(tenant_id),
            },
            "source": "mongodb",
        }
    except Exception as e:
        logger.error(f"Failed to get DB stats: {e}")
        return {
            "traces": {"count": 0},
            "evaluations": {},
            "feedback": {},
            "prompts": {"count": 0},
            "source": "mongodb",
            "error": str(e),
        }


@router.get("/db/debug")
async def get_db_debug(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Debug endpoint to check MongoDB contents.
    
    Shows all tenant_ids in the traces collection and counts.
    """
    try:
        trace_repo = get_trace_repository()
        coll = trace_repo.collection
        
        if coll is None:
            return {"error": "MongoDB collection not available"}
        
        # Get all unique tenant_ids
        tenant_ids = coll.distinct("tenant_id")
        
        # Count traces per tenant
        tenant_counts = {}
        for tid in tenant_ids:
            count = coll.count_documents({"tenant_id": tid})
            tenant_counts[str(tid)] = count
        
        # Get total count
        total = coll.count_documents({})
        
        # Get sample traces
        sample_traces = []
        for doc in coll.find().limit(5):
            sample_traces.append({
                "trace_id": doc.get("trace_id"),
                "tenant_id": doc.get("tenant_id"),
                "timestamp": doc.get("timestamp"),
                "total_spans": doc.get("total_spans"),
            })
        
        # Check for current user's traces
        user_traces_count = coll.count_documents({"tenant_id": str(user.tenant_id)})
        
        return {
            "current_user_tenant_id": str(user.tenant_id),
            "current_user_traces_count": user_traces_count,
            "total_traces": total,
            "unique_tenant_ids": tenant_ids,
            "traces_per_tenant": tenant_counts,
            "sample_traces": sample_traces,
        }
    except Exception as e:
        logger.error(f"Failed to get DB debug info: {e}")
        return {"error": str(e)}


@router.get("/resource-quotas")
async def get_resource_quota_stats(
    user: UserIdentity = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Get resource quota statistics for all tenants.
    
    Returns current resource usage, limits, and fairness metrics.
    Useful for monitoring fair resource distribution across tenants.
    """
    try:
        manager = get_resource_manager()
        stats = manager.get_stats()
        fairness = manager.get_fairness_metrics()
        
        # Get current tenant's quota info
        quota = manager.get_tenant_quota(user.tenant_id)
        state = manager.get_tenant_state(user.tenant_id)
        
        return {
            "current_tenant": {
                "tenant_id": user.tenant_id,
                "concurrent_requests": state.concurrent_requests,
                "concurrent_limit": quota.max_concurrent_requests,
                "requests_this_minute": state.requests_this_minute,
                "rate_limit": quota.requests_per_minute,
                "tokens_this_minute": state.tokens_this_minute,
                "token_limit": quota.tokens_per_minute,
                "total_requests": state.total_requests,
                "rejected_requests": state.rejected_requests,
            },
            "global": {
                "concurrent": stats["global_concurrent"],
                "max_concurrent": stats["global_max_concurrent"],
            },
            "all_tenants": stats["tenants"],
            "fairness": fairness,
        }
    except Exception as e:
        logger.error(f"Failed to get resource quota stats: {e}")
        return {"error": str(e)}
