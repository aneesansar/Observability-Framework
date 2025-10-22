import time
import functools
import inspect
import os
import importlib
from types import FunctionType, MethodType, ModuleType
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from prometheus_client import Counter, Histogram
from contextvars import ContextVar
import re

# Thread-safe storage
request_context: ContextVar = ContextVar("request_context", default=None)
function_stack: ContextVar = ContextVar("function_stack", default=[])

# Prometheus metrics
function_counter = Counter(
    "function_calls_total", "Total function calls", ["function"]
)
function_duration = Histogram(
    "function_duration_seconds", "Function duration", ["function"]
)

# ----------------------------
# Simple check helpers for LLM observability
# ----------------------------
def check_hallucination(prompt, response, context=None):
    """Naive hallucination detector: check if any context tokens appear in response."""
    if not context:
        return False
    return not any(str(c).lower() in str(response).lower() for c in context)


def check_consistency(prompt, response):
    """Naive consistency: does first word of prompt appear in response?"""
    if not prompt or not response:
        return True
    return prompt.split()[0].lower() in response.lower()


def extract_reasoning_steps(response: str):
    """Extract heuristic reasoning steps from response text."""
    if not response:
        return []
    lines = str(response).split("\n")
    # Match "Step 1", "1.", "Firstly", "-", "Because"
    step_pattern = re.compile(r"(step\s*\d+|^\d+\.)|(^\-\s)|\b(because|therefore|first|next|finally)\b", re.IGNORECASE)
    steps = []
    for line in lines:
        if step_pattern.search(line.strip()):
            steps.append(line.strip())
    return steps


# ----------------------------
# Main decorator
# ----------------------------
def observe(label=None):
    """
    Decorator that instruments functions and adds LLM + reasoning attributes.
    - Captures args, kwargs, results, errors
    - Adds hallucination/consistency checks for LLM calls
    - Marks reasoning steps automatically for relevant backend functions
    """
    def decorator(func):
        func_label = label or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer("custom.functions")
            function_counter.labels(func_label).inc()
            start_time = time.time()

            # Manage stack for parent-child relationships
            stack = function_stack.get()
            new_stack = stack + [func_label]
            token = function_stack.set(new_stack)

            parent = stack[-1] if stack else None
            depth = len(new_stack)

            with tracer.start_as_current_span(func_label) as span:
                # === HIGH LEVEL TAGS ===
                span.set_attribute("High_Level.function.name", func_label)
                span.set_attribute("High_Level.function.parent", parent or "ROOT")
                if parent is None:
                    span.set_attribute("High_Level.function.args", str(args))
                    span.set_attribute("High_Level.function.kwargs", str(kwargs))

                # === MEDIUM LEVEL TAGS ===
                span.set_attribute("Medium_Level.function.name", func_label)
                span.set_attribute("Medium_Level.function.parent", parent or "ROOT")
                span.set_attribute("Medium_Level.function.depth", depth)
                span.set_attribute("Medium_Level.function.hierarchy", " > ".join(new_stack))

                # === LOW LEVEL TAGS ===
                ontology_value = None
                try:
                    request = request_context.get()
                    if request:
                        ontology_value = request.headers.get("ontology")
                except Exception:
                    pass

                span.set_attribute("Low_Level.function.name", func_label)
                span.set_attribute("Low_Level.function.parent", parent or "ROOT")
                span.set_attribute("Low_Level.function.depth", depth)
                span.set_attribute("Low_Level.tenant.ontology", ontology_value or "unknown")
                span.set_attribute("Low_Level.function.args", str(args))
                span.set_attribute("Low_Level.function.kwargs", str(kwargs))

                # === Reasoning Step marker for backend methods ===
                if any(keyword in func_label.lower() for keyword in ["search", "query", "rerank", "retrieve", "prepare", "apply"]):
                    span.add_event("Reasoning Step", {"detail": f"Entered {func_label}"})

                try:
                    result = func(*args, **kwargs)

                    # Record results
                    if parent is None:
                        span.set_attribute("High_Level.function.result", str(result)[:500])
                    span.set_attribute("Medium_Level.function.result", str(result)[:500])
                    span.set_attribute("Low_Level.function.result", str(result)[:500])

                    # ----------------------------
                    # LLM-Specific Instrumentation
                    # ----------------------------
                    prompt = kwargs.get("prompt")
                    if not prompt and args and isinstance(args[0], str):
                        prompt = args[0]

                    if prompt:
                        response = str(result)
                        context = kwargs.get("context")

                        hallucination = check_hallucination(prompt, response, context)
                        consistency = check_consistency(prompt, response)
                        steps = extract_reasoning_steps(response)

                        # Attach LLM attributes
                        span.set_attribute("LLM.prompt", str(prompt)[:500])
                        span.set_attribute("LLM.response", response[:500])
                        span.set_attribute("LLM.hallucination", hallucination)
                        span.set_attribute("LLM.prompt_consistency", consistency)

                        for i, step in enumerate(steps):
                            span.add_event(f"Reasoning Step {i+1}", {"detail": step})

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.set_attribute("Low_Level.error", str(e))
                    raise
                finally:
                    duration = time.time() - start_time
                    function_duration.labels(func_label).observe(duration)
                    span.set_attribute("Medium_Level.function.duration", duration)
                    span.set_attribute("Low_Level.function.duration", duration)
                    function_stack.reset(token)

        return wrapper
    return decorator


# ----------------------------
# Utilities to instrument everything
# ----------------------------
def observe_all_methods(cls):
    for attr_name in dir(cls):
        if attr_name.startswith("__"):
            continue
        attr = getattr(cls, attr_name)
        if isinstance(attr, (FunctionType, MethodType)) and callable(attr):
            wrapped = observe(attr_name)(attr)
            setattr(cls, attr_name, wrapped)
    return cls


def observe_all_functions_in_module(module: ModuleType):
    for name in dir(module):
        attr = getattr(module, name)
        if isinstance(attr, FunctionType) and attr.__module__ == module.__name__:
            setattr(module, name, observe(name)(attr))


def observe_all_in_file(module: ModuleType):
    observe_all_functions_in_module(module)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            observe_all_methods(obj)


def instrument_all_in_package(package_path: str, package_root: str):
    for dirpath, _, filenames in os.walk(package_path):
        for filename in filenames:
            if filename.endswith(".py") and not filename.startswith("__"):
                module_rel_path = os.path.relpath(
                    os.path.join(dirpath, filename), package_path
                )
                module_name = module_rel_path.replace(os.sep, ".").rsplit(".py", 1)[0]
                full_import_path = f"{package_root}.{module_name}"
                try:
                    module = importlib.import_module(full_import_path)
                    observe_all_in_file(module)
                except Exception as e:
                    print(f"[WARN] Could not instrument {full_import_path}: {e}")
