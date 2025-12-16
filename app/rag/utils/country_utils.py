"""
Country Utilities.

Country name normalization and extraction.
"""

import json
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache for country aliases
_ALIASES_CACHE: Optional[Dict[str, str]] = None


def _load_country_aliases() -> Dict[str, str]:
    """
    Load country aliases from JSON file.
    
    Returns:
        Dict mapping alias -> display name
    """
    global _ALIASES_CACHE
    
    if _ALIASES_CACHE is not None:
        return _ALIASES_CACHE
    
    try:
        # Find the aliases file (in the same directory as this module)
        path = os.path.join(os.path.dirname(__file__), "country_aliases.json")
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Normalize keys to lowercase
        _ALIASES_CACHE = {str(k).lower(): str(v) for k, v in data.items()}
        return _ALIASES_CACHE
        
    except Exception as e:
        logger.debug(f"Failed to load country aliases: {e}")
        _ALIASES_CACHE = {}
        return _ALIASES_CACHE


def normalize_country(value: Optional[str]) -> str:
    """
    Normalize a country name to its display form.
    
    Args:
        value: Country name or alias
        
    Returns:
        Normalized display name
    """
    if not value:
        return ""
    
    aliases = _load_country_aliases()
    v = str(value).strip()
    
    # Check alias map
    normalized = aliases.get(v.lower())
    if normalized:
        return normalized
    
    # Return as-is if not found
    return v


def extract_countries_from_query(query: str) -> List[str]:
    """
    Extract country names from a query string.
    
    Args:
        query: User query
        
    Returns:
        List of detected country names
    """
    if not query:
        return []
    
    query_lower = query.lower()
    aliases = _load_country_aliases()
    
    found = []
    for alias_key, display in aliases.items():
        if alias_key and alias_key in query_lower:
            if display not in found:
                found.append(display)
    
    return found


def detect_provider_from_query(query: str) -> str:
    """
    Detect travel provider from query.
    
    Args:
        query: User query
        
    Returns:
        Provider name or empty string
    """
    if not query:
        return ""
    
    query_lower = query.lower()
    
    # Provider name mappings
    providers = {
        "trendtours": "TrendTours",
        "studiosus": "Studiosus",
        "marco polo": "Marco Polo",
        "marcopolo": "Marco Polo",
        "berge & meer": "Berge & Meer",
        "berge und meer": "Berge & Meer",
        "berge_meer": "Berge & Meer",
        "chamaeleon": "Chamaeleon",
        "chamäleon": "Chamaeleon",
    }
    
    for key, name in providers.items():
        if key in query_lower:
            return name
    
    return ""
