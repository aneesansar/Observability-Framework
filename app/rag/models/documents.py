"""
Document Models.

Data models for travel documents and retrieval results.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional


@dataclass
class TripDate:
    """Represents a single trip date option."""
    
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    price: Optional[float] = None
    airport: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "price": self.price,
            "airport": self.airport,
        }
    
    def format_display(self) -> str:
        """Format for display."""
        parts = []
        if self.start_date or self.end_date:
            parts.append(f"{self.start_date or '?'}→{self.end_date or '?'}")
        if self.price is not None:
            parts.append(f"{self.price}")
        if self.airport:
            parts.append(self.airport)
        return " | ".join(parts)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripDate":
        """Create from dictionary."""
        price = data.get("price")
        if price is not None:
            try:
                price = float(price)
            except (ValueError, TypeError):
                price = None
        
        return cls(
            start_date=data.get("start_date"),
            end_date=data.get("end_date"),
            price=price,
            airport=data.get("airport"),
        )


@dataclass
class DocumentMetadata:
    """Metadata for a document in the vector store."""
    
    path: Optional[str] = None
    provider: Optional[str] = None
    website: Optional[str] = None
    trip_index: Optional[int] = None
    trip_name: Optional[str] = None
    country: Optional[str] = None
    duration: Optional[str] = None
    trip_type: Optional[str] = None
    date_summary: Optional[str] = None
    title: Optional[str] = None
    preview: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "provider": self.provider,
            "website": self.website,
            "trip_index": self.trip_index,
            "trip_name": self.trip_name,
            "country": self.country,
            "duration": self.duration,
            "trip_type": self.trip_type,
            "date_summary": self.date_summary,
            "title": self.title,
            "preview": self.preview,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """Create from dictionary."""
        return cls(
            path=data.get("path"),
            provider=data.get("provider"),
            website=data.get("website"),
            trip_index=data.get("trip_index"),
            trip_name=data.get("trip_name"),
            country=data.get("country"),
            duration=data.get("duration"),
            trip_type=data.get("trip_type"),
            date_summary=data.get("date_summary"),
            title=data.get("title"),
            preview=data.get("preview"),
        )


@dataclass
class TripDocument:
    """Represents a complete trip document."""
    
    provider: str
    trip_name: str
    country: Optional[str] = None
    trip_id: Optional[str] = None
    duration: Optional[str] = None
    trip_type: Optional[str] = None
    url: Optional[str] = None
    website: Optional[str] = None
    dates: List[TripDate] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider,
            "trip_name": self.trip_name,
            "country": self.country,
            "trip_id": self.trip_id,
            "duration": self.duration,
            "trip_type": self.trip_type,
            "url": self.url,
            "website": self.website,
            "dates": [d.to_dict() for d in self.dates],
        }
    
    def format_display(self) -> str:
        """Format for display."""
        lines = [f"{self.provider} – {self.trip_name}"]
        if self.country:
            lines[0] += f" ({self.country})"
        if self.url:
            lines.append(f"  URL: {self.url}")
        for d in self.dates[:3]:  # Show first 3 dates
            lines.append(f"  - {d.format_display()}")
        return "\n".join(lines)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TripDocument":
        """Create from dictionary."""
        dates = [
            TripDate.from_dict(d) if isinstance(d, dict) else d
            for d in data.get("dates", [])
        ]
        return cls(
            provider=data.get("provider", ""),
            trip_name=data.get("trip_name", ""),
            country=data.get("country"),
            trip_id=data.get("trip_id"),
            duration=data.get("duration"),
            trip_type=data.get("trip_type"),
            url=data.get("url"),
            website=data.get("website"),
            dates=dates,
        )


@dataclass
class RetrievalResult:
    """Result from document retrieval."""
    
    metadata: DocumentMetadata
    score: float
    text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = self.metadata.to_dict()
        result["score"] = self.score
        if self.text:
            result["text"] = self.text
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalResult":
        """Create from dictionary."""
        score = data.pop("score", 0.0)
        text = data.pop("text", None)
        return cls(
            metadata=DocumentMetadata.from_dict(data),
            score=score,
            text=text,
        )
