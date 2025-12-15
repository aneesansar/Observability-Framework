"""
Instrumentation Module.

Provides decorators and utilities for automatic function instrumentation.
"""

from .decorators import (
    observe,
    observe_async,
    ObserveConfig,
)
from .auto_instrument import (
    observe_all_methods,
    observe_all_functions_in_module,
    observe_all_in_file,
    instrument_all_in_package,
)

__all__ = [
    "observe",
    "observe_async",
    "ObserveConfig",
    "observe_all_methods",
    "observe_all_functions_in_module",
    "observe_all_in_file",
    "instrument_all_in_package",
]
