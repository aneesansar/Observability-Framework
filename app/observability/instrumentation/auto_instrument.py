"""
Auto-Instrumentation Utilities.

Provides utilities for automatically instrumenting modules, classes, and packages.
"""

import importlib
import inspect
import logging
import os
import re
from types import FunctionType, MethodType, ModuleType
from typing import List, Optional, Pattern, Set

from .decorators import observe
from ..config import observability_settings

logger = logging.getLogger(__name__)


def observe_all_methods(cls: type) -> type:
    """
    Instrument all methods of a class with @observe.
    
    Skips dunder methods (__init__, __str__, etc.).
    
    Args:
        cls: Class to instrument
        
    Returns:
        Instrumented class
        
    Example:
        @observe_all_methods
        class MyService:
            def method1(self):
                pass
            def method2(self):
                pass
    """
    for attr_name in dir(cls):
        # Skip dunder methods
        if attr_name.startswith("__"):
            continue
        
        attr = getattr(cls, attr_name)
        
        # Only instrument callable methods
        if isinstance(attr, (FunctionType, MethodType)) and callable(attr):
            wrapped = observe(label=f"{cls.__name__}.{attr_name}")(attr)
            setattr(cls, attr_name, wrapped)
    
    return cls


def observe_all_functions_in_module(module: ModuleType) -> None:
    """
    Instrument all functions defined in a module.
    
    Only instruments functions that are defined in the module itself,
    not imported functions.
    
    Args:
        module: Module to instrument
    """
    for name in dir(module):
        attr = getattr(module, name)
        
        # Only instrument functions defined in this module
        if isinstance(attr, FunctionType) and attr.__module__ == module.__name__:
            setattr(module, name, observe(label=name)(attr))


def observe_all_in_file(module: ModuleType) -> None:
    """
    Instrument all functions and classes in a module.
    
    Combines observe_all_functions_in_module and observe_all_methods
    for comprehensive instrumentation.
    
    Args:
        module: Module to instrument
    """
    # Instrument standalone functions
    observe_all_functions_in_module(module)
    
    # Instrument class methods
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            observe_all_methods(obj)


def instrument_all_in_package(
    package_path: str,
    package_root: str,
    exclude_patterns: Optional[List[str]] = None,
) -> None:
    """
    Recursively instrument all modules in a package.
    
    Walks through all Python files in the package directory and
    instruments functions and classes.
    
    Args:
        package_path: Absolute path to the package directory
        package_root: Python import path root (e.g., "app")
        exclude_patterns: Regex patterns for modules to exclude
        
    Example:
        instrument_all_in_package(
            "/path/to/app",
            "app",
            exclude_patterns=[".*test.*", ".*migrations.*"]
        )
    """
    # Compile exclude patterns
    compiled_excludes = _compile_exclude_patterns(exclude_patterns)
    
    # Track instrumented modules to avoid duplicates
    instrumented: Set[str] = set()
    
    for dirpath, _, filenames in os.walk(package_path):
        for filename in filenames:
            # Only process Python files, skip __init__ and __pycache__
            if not filename.endswith(".py") or filename.startswith("__"):
                continue
            
            # Build import path
            module_rel_path = os.path.relpath(
                os.path.join(dirpath, filename),
                package_path
            )
            module_name = module_rel_path.replace(os.sep, ".").rsplit(".py", 1)[0]
            full_import_path = f"{package_root}.{module_name}"
            
            # Skip if already instrumented
            if full_import_path in instrumented:
                continue
            
            # Check exclude patterns
            if _should_exclude(full_import_path, compiled_excludes):
                logger.debug(f"Skipping excluded module: {full_import_path}")
                continue
            
            # Try to instrument
            try:
                module = importlib.import_module(full_import_path)
                observe_all_in_file(module)
                instrumented.add(full_import_path)
                logger.debug(f"Instrumented module: {full_import_path}")
            except Exception as e:
                logger.warning(f"Could not instrument {full_import_path}: {e}")


def _compile_exclude_patterns(
    patterns: Optional[List[str]] = None,
) -> List[Pattern]:
    """
    Compile exclude patterns from settings and arguments.
    
    Args:
        patterns: Additional patterns to compile
        
    Returns:
        List of compiled regex patterns
    """
    all_patterns = list(observability_settings.observe_exclude_patterns)
    
    if patterns:
        all_patterns.extend(patterns)
    
    compiled = []
    for pattern in all_patterns:
        try:
            compiled.append(re.compile(pattern))
        except re.error as e:
            logger.warning(f"Invalid exclude pattern '{pattern}': {e}")
    
    return compiled


def _should_exclude(
    import_path: str,
    patterns: List[Pattern],
) -> bool:
    """
    Check if an import path should be excluded.
    
    Args:
        import_path: Full import path to check
        patterns: Compiled exclude patterns
        
    Returns:
        True if the path should be excluded
    """
    return any(pattern.search(import_path) for pattern in patterns)


class ModuleInstrumentor:
    """
    Class-based module instrumentor for more control.
    
    Provides fine-grained control over which modules and functions
    to instrument.
    
    Example:
        instrumentor = ModuleInstrumentor(
            package_root="app",
            exclude_patterns=[".*test.*"],
        )
        instrumentor.instrument_module("app.services.llm_service")
        instrumentor.instrument_package("/path/to/app/services")
    """
    
    def __init__(
        self,
        package_root: str,
        exclude_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize the instrumentor.
        
        Args:
            package_root: Python import path root
            exclude_patterns: Regex patterns for modules to exclude
        """
        self._package_root = package_root
        self._compiled_excludes = _compile_exclude_patterns(exclude_patterns)
        self._instrumented: Set[str] = set()
    
    def instrument_module(self, import_path: str) -> bool:
        """
        Instrument a single module by import path.
        
        Args:
            import_path: Full import path (e.g., "app.services.llm_service")
            
        Returns:
            True if instrumentation succeeded
        """
        if import_path in self._instrumented:
            return True
        
        if _should_exclude(import_path, self._compiled_excludes):
            logger.debug(f"Skipping excluded module: {import_path}")
            return False
        
        try:
            module = importlib.import_module(import_path)
            observe_all_in_file(module)
            self._instrumented.add(import_path)
            logger.debug(f"Instrumented module: {import_path}")
            return True
        except Exception as e:
            logger.warning(f"Could not instrument {import_path}: {e}")
            return False
    
    def instrument_package(self, package_path: str) -> int:
        """
        Instrument all modules in a package directory.
        
        Args:
            package_path: Absolute path to package directory
            
        Returns:
            Number of modules instrumented
        """
        count = 0
        
        for dirpath, _, filenames in os.walk(package_path):
            for filename in filenames:
                if not filename.endswith(".py") or filename.startswith("__"):
                    continue
                
                module_rel_path = os.path.relpath(
                    os.path.join(dirpath, filename),
                    os.path.dirname(package_path)
                )
                module_name = module_rel_path.replace(os.sep, ".").rsplit(".py", 1)[0]
                
                if self.instrument_module(module_name):
                    count += 1
        
        return count
    
    @property
    def instrumented_modules(self) -> Set[str]:
        """Get set of instrumented module paths."""
        return self._instrumented.copy()
