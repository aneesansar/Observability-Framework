"""
Query Models.

Data models for query parsing and intent detection.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SortOrder(str, Enum):
    """Sort order for query results."""
    
    PRICE = "price"
    START_DATE = "start"
    RELEVANCE = "relevance"
    NONE = ""


@dataclass
class QueryIntent:
    """Detected intent from a query."""
    
    wants_sorting: bool = False
    sort_by_price: bool = False
    sort_by_date: bool = False
    filter_country: Optional[str] = None
    filter_provider: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "wants_sorting": self.wants_sorting,
            "sort_by_price": self.sort_by_price,
            "sort_by_date": self.sort_by_date,
            "filter_country": self.filter_country,
            "filter_provider": self.filter_provider,
        }
    
    @classmethod
    def from_query(cls, query: str) -> "QueryIntent":
        """Detect intent from query string."""
        ql = (query or "").lower()
        
        return cls(
            wants_sorting=any(t in ql for t in ["sort", "sorted", "order"]),
            sort_by_price=any(t in ql for t in ["price", "cheapest", "least", "günstig"]),
            sort_by_date=any(t in ql for t in ["start", "earliest", "soonest", "first"]),
        )


@dataclass
class QueryTerms:
    """Structured query terms extracted from natural language."""
    
    countries: List[str] = field(default_factory=list)
    airport: Optional[str] = None
    provider: Optional[str] = None
    sort_by: SortOrder = SortOrder.NONE
    top_n: int = 5
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    raw_query: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "countries": self.countries,
            "airport": self.airport,
            "provider": self.provider,
            "sort_by": self.sort_by.value if isinstance(self.sort_by, SortOrder) else self.sort_by,
            "top_n": self.top_n,
            "date_start": self.date_start,
            "date_end": self.date_end,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], raw_query: Optional[str] = None) -> "QueryTerms":
        """Create from dictionary."""
        sort_by = data.get("sort_by", "")
        if isinstance(sort_by, str):
            try:
                sort_by = SortOrder(sort_by.lower()) if sort_by else SortOrder.NONE
            except ValueError:
                sort_by = SortOrder.NONE
        
        return cls(
            countries=data.get("countries", []),
            airport=data.get("airport"),
            provider=data.get("provider"),
            sort_by=sort_by,
            top_n=data.get("top_n", 5),
            date_start=data.get("date_start"),
            date_end=data.get("date_end"),
            raw_query=raw_query,
        )
    
    @classmethod
    def empty(cls) -> "QueryTerms":
        """Create empty query terms."""
        return cls()
    
    def has_filters(self) -> bool:
        """Check if any filters are set."""
        return bool(
            self.countries
            or self.airport
            or self.provider
            or self.date_start
            or self.date_end
        )
    
    def has_sorting(self) -> bool:
        """Check if sorting is requested."""
        return self.sort_by != SortOrder.NONE
