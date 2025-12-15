"""
Instrumentation Decorators.

Provides the @observe decorator for automatic function tracing and metrics.
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, TypeVar, Union

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span

from ..config import observability_settings
from ..context import (
    get_current_request,
    get_current_tenant_id,
    get_function_stack,
    push_function,
    pop_function,
    should_skip_path,
)
from ..metrics import get_function_metrics
from ..semantic_conventions import (
    SpanAttributes,
    EventNames,
    TracerNames,
    RoleBasedAttributes,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class ObserveConfig:
    """Configuration for the @observe decorator."""
    
    label: Optional[str] = None
    """Custom label for the span (defaults to function qualname)."""
    
    skip_span: bool = False
    """Skip creating a span (useful for utility functions)."""
    
    record_args: bool = True
    """Record function arguments as span attributes."""
    
    record_result: bool = True
    """Record function result as span attribute."""
    
    max_arg_length: int = 500
    """Maximum length for argument string representation."""
    
    max_result_length: int = 500
    """Maximum length for result string representation."""
    
    reasoning_keywords: List[str] = field(default_factory=lambda: [
        "search", "query", "rerank", "retrieve", "prepare", "apply", "generate"
    ])
    """Keywords that trigger reasoning step events."""


def observe(
    label: Optional[str] = None,
    skip_span: bool = False,
    config: Optional[ObserveConfig] = None,
) -> Callable[[F], F]:
    """
    Decorator that instruments functions with OpenTelemetry tracing and Prometheus metrics.
    
    Features:
    - Creates spans with function metadata
    - Records arguments and results (configurable)
    - Tracks function call counts and duration
    - Adds tenant/user context to spans
    - Marks reasoning steps for relevant functions
    - Handles errors with proper span status
    
    Args:
        label: Custom span label (defaults to function qualname)
        skip_span: Skip span creation for utility functions
        config: Full configuration object (overrides other args)
        
    Returns:
        Decorated function
        
    Example:
        @observe()
        def my_function(arg1, arg2):
            return result
            
        @observe(label="CustomName", skip_span=False)
        def another_function():
            pass
    """
    # Build config
    if config is None:
        config = ObserveConfig(label=label, skip_span=skip_span)
    
    def decorator(func: F) -> F:
        # CRITICAL: Never instrument observability code itself to prevent infinite recursion
        if func.__module__ and func.__module__.startswith("app.observability"):
            return func  # type: ignore
        
        func_label = config.label or func.__qualname__
        func_name = func.__name__
        
        # Auto-skip private functions unless explicitly configured
        should_skip = config.skip_span or (
            func_name.startswith("_") and not func_name.startswith("__")
        )
        
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Check if we're in a request context
            request = get_current_request()
            if request is None:
                # No request context - skip tracing
                return func(*args, **kwargs)
            
            # Check if path should be skipped
            try:
                path = str(request.url.path).lower()
                if should_skip_path(path, observability_settings.skip_paths):
                    return func(*args, **kwargs)
            except Exception:
                pass
            
            # Skip span creation if configured
            if should_skip:
                return func(*args, **kwargs)
            
            # Get tracer and metrics
            tracer = trace.get_tracer(TracerNames.CUSTOM_FUNCTIONS)
            metrics = get_function_metrics()
            
            # Track function stack for hierarchy
            stack = get_function_stack()
            token = push_function(func_label)
            
            parent = stack[-1] if stack else None
            depth = len(stack) + 1
            is_root = parent is None
            
            start_time = time.time()
            
            with tracer.start_as_current_span(func_label) as span:
                try:
                    # Set basic attributes
                    _set_span_attributes(
                        span=span,
                        func_label=func_label,
                        depth=depth,
                        is_root=is_root,
                        parent=parent,
                        stack=stack + [func_label],
                        args=args,
                        kwargs=kwargs,
                        config=config,
                    )
                    
                    # Add reasoning step event if applicable
                    if any(kw in func_label.lower() for kw in config.reasoning_keywords):
                        span.add_event(
                            EventNames.REASONING_STEP,
                            {"detail": f"Entered {func_label}"}
                        )
                    
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Record result
                    if config.record_result and is_root:
                        result_str = str(result)[:config.max_result_length]
                        span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.FUNCTION_RESULT), result_str)
                        span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_RESULT), result_str)
                        span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_RESULT), result_str)
                    elif config.record_result:
                        span.set_attribute(
                            RoleBasedAttributes.developer(SpanAttributes.FUNCTION_RESULT_LENGTH),
                            len(str(result))
                        )
                    
                    # Record success metrics
                    duration = time.time() - start_time
                    metrics.record_call(func_label, duration, success=True)
                    
                    return result
                    
                except Exception as e:
                    # Record error
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.ERROR), str(e))
                    
                    # Record error metrics
                    duration = time.time() - start_time
                    metrics.record_call(
                        func_label,
                        duration,
                        success=False,
                        error_type=type(e).__name__,
                    )
                    
                    raise
                    
                finally:
                    # Record duration
                    duration = time.time() - start_time
                    span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_DURATION), duration)
                    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_DURATION), duration)
                    
                    # Pop function from stack
                    pop_function(token)
        
        return wrapper  # type: ignore
    
    return decorator


def observe_async(
    label: Optional[str] = None,
    skip_span: bool = False,
    config: Optional[ObserveConfig] = None,
) -> Callable[[F], F]:
    """
    Async version of the @observe decorator.
    
    Same features as @observe but for async functions.
    
    Args:
        label: Custom span label
        skip_span: Skip span creation
        config: Full configuration object
        
    Returns:
        Decorated async function
    """
    if config is None:
        config = ObserveConfig(label=label, skip_span=skip_span)
    
    def decorator(func: F) -> F:
        # CRITICAL: Never instrument observability code itself to prevent infinite recursion
        if func.__module__ and func.__module__.startswith("app.observability"):
            return func  # type: ignore
        
        func_label = config.label or func.__qualname__
        func_name = func.__name__
        
        should_skip = config.skip_span or (
            func_name.startswith("_") and not func_name.startswith("__")
        )
        
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request = get_current_request()
            if request is None:
                return await func(*args, **kwargs)
            
            try:
                path = str(request.url.path).lower()
                if should_skip_path(path, observability_settings.skip_paths):
                    return await func(*args, **kwargs)
            except Exception:
                pass
            
            if should_skip:
                return await func(*args, **kwargs)
            
            tracer = trace.get_tracer(TracerNames.CUSTOM_FUNCTIONS)
            metrics = get_function_metrics()
            
            stack = get_function_stack()
            token = push_function(func_label)
            
            parent = stack[-1] if stack else None
            depth = len(stack) + 1
            is_root = parent is None
            
            start_time = time.time()
            
            with tracer.start_as_current_span(func_label) as span:
                try:
                    _set_span_attributes(
                        span=span,
                        func_label=func_label,
                        depth=depth,
                        is_root=is_root,
                        parent=parent,
                        stack=stack + [func_label],
                        args=args,
                        kwargs=kwargs,
                        config=config,
                    )
                    
                    if any(kw in func_label.lower() for kw in config.reasoning_keywords):
                        span.add_event(
                            EventNames.REASONING_STEP,
                            {"detail": f"Entered {func_label}"}
                        )
                    
                    result = await func(*args, **kwargs)
                    
                    if config.record_result and is_root:
                        result_str = str(result)[:config.max_result_length]
                        span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.FUNCTION_RESULT), result_str)
                        span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_RESULT), result_str)
                        span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_RESULT), result_str)
                    elif config.record_result:
                        span.set_attribute(
                            RoleBasedAttributes.developer(SpanAttributes.FUNCTION_RESULT_LENGTH),
                            len(str(result))
                        )
                    
                    duration = time.time() - start_time
                    metrics.record_call(func_label, duration, success=True)
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.ERROR), str(e))
                    
                    duration = time.time() - start_time
                    metrics.record_call(
                        func_label,
                        duration,
                        success=False,
                        error_type=type(e).__name__,
                    )
                    
                    raise
                    
                finally:
                    duration = time.time() - start_time
                    span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_DURATION), duration)
                    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_DURATION), duration)
                    pop_function(token)
        
        return wrapper  # type: ignore
    
    return decorator


def _set_span_attributes(
    span: Span,
    func_label: str,
    depth: int,
    is_root: bool,
    parent: Optional[str],
    stack: List[str],
    args: tuple,
    kwargs: dict,
    config: ObserveConfig,
) -> None:
    """
    Set span attributes based on function context.
    
    Args:
        span: The span to set attributes on
        func_label: Function label/name
        depth: Call stack depth
        is_root: Whether this is a root span
        parent: Parent function name
        stack: Current function stack
        args: Function arguments
        kwargs: Function keyword arguments
        config: Observe configuration
    """
    # Add tenant_id (critical for isolation)
    tenant_id = get_current_tenant_id()
    if tenant_id:
        span.set_attribute(SpanAttributes.TENANT_ID, tenant_id)
        span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.TENANT_ID), tenant_id)
    
    # Generic attributes (all spans)
    span.set_attribute(SpanAttributes.FUNCTION_NAME, func_label)
    span.set_attribute(SpanAttributes.FUNCTION_DEPTH, depth)
    span.set_attribute(SpanAttributes.FUNCTION_IS_ROOT, is_root)
    
    if is_root:
        # Root span - full details
        span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.FUNCTION_NAME), func_label)
        span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_NAME), func_label)
        span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_HIERARCHY), func_label)
        
        if config.record_args:
            args_str = str(args)[:config.max_arg_length]
            kwargs_str = str(kwargs)[:config.max_arg_length]
            span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.FUNCTION_ARGS), args_str)
            span.set_attribute(RoleBasedAttributes.admin(SpanAttributes.FUNCTION_KWARGS), kwargs_str)
            span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_ARGS), args_str)
            span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_KWARGS), kwargs_str)
    else:
        # Child span - minimal attributes
        span.set_attribute(SpanAttributes.FUNCTION_PARENT, parent or "ROOT")
        span.set_attribute(RoleBasedAttributes.manager(SpanAttributes.FUNCTION_HIERARCHY), " > ".join(stack))
    
    # Developer attributes (all spans)
    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_NAME), func_label)
    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_PARENT), parent or "ROOT")
    span.set_attribute(RoleBasedAttributes.developer(SpanAttributes.FUNCTION_DEPTH), depth)
