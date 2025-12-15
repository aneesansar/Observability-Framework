"""
Query Parser.

Extracts structured query terms from natural language.
"""

import logging
import re
from typing import Any, Dict, Optional

from ..models.queries import QueryTerms, SortOrder
from .country_utils import extract_countries_from_query, normalize_country

logger = logging.getLogger(__name__)


class QueryParser:
    """
    Parses natural language queries into structured terms.
    
    Supports:
    - Country extraction
    - Airport/departure detection
    - Sorting intent detection
    - Provider detection
    """
    
    # Keywords for sort detection
    PRICE_KEYWORDS = ["price", "cheapest", "cheap", "günstig", "guenstig", "preis", "lowest"]
    DATE_KEYWORDS = ["start", "earliest", "soonest", "first", "früh", "frueh", "beginn"]
    
    def parse(self, query: str, tenant_id: str = "anonymous") -> QueryTerms:
        """
        Parse a query into structured terms.
        
        Args:
            query: Natural language query
            tenant_id: Tenant identifier
            
        Returns:
            QueryTerms with extracted filters
        """
        if not query:
            return QueryTerms.empty()
        
        query_lower = query.lower()
        
        # Extract countries
        countries = extract_countries_from_query(query)
        
        # Detect airport/departure
        airport = self._extract_airport(query)
        
        # Remove departure city from countries if present
        if airport:
            countries = [c for c in countries if airport.lower() not in c.lower()]
        
        # Detect sorting intent
        sort_by = self._detect_sort(query_lower)
        
        # Detect provider
        provider = self._detect_provider(query_lower)
        
        return QueryTerms(
            countries=countries,
            airport=airport,
            provider=provider,
            sort_by=sort_by,
            top_n=5,
            raw_query=query,
        )
    
    def _extract_airport(self, query: str) -> Optional[str]:
        """
        Extract departure airport/city from query.
        
        Args:
            query: User query
            
        Returns:
            Airport/city name or None
        """
        # Match "from <City>" pattern
        match = re.search(r'from\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', query)
        if match:
            return match.group(1).strip()
        
        return None
    
    def _detect_sort(self, query_lower: str) -> SortOrder:
        """
        Detect sorting intent from query.
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            SortOrder enum
        """
        if any(kw in query_lower for kw in self.PRICE_KEYWORDS):
            return SortOrder.PRICE
        
        if any(kw in query_lower for kw in self.DATE_KEYWORDS):
            return SortOrder.START_DATE
        
        return SortOrder.NONE
    
    def _detect_provider(self, query_lower: str) -> Optional[str]:
        """
        Detect provider from query.
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            Provider name or None
        """
        providers = {
            "trendtours": "TrendTours",
            "studiosus": "Studiosus",
            "marco polo": "Marco Polo",
            "marcopolo": "Marco Polo",
            "berge": "Berge & Meer",
            "chamaeleon": "Chamaeleon",
        }
        
        for key, name in providers.items():
            if key in query_lower:
                return name
        
        return None


def parse_query_terms(query: str, tenant_id: str = "anonymous") -> QueryTerms:
    """
    Parse query into structured terms.
    
    This is a simplified rule-based parser. For more complex queries,
    consider using LLM-based parsing.
    
    Args:
        query: Natural language query
        tenant_id: Tenant identifier
        
    Returns:
        QueryTerms with extracted filters
    """
    parser = QueryParser()
    return parser.parse(query, tenant_id)
