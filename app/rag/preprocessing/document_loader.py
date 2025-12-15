"""
Document Loader.

Loads and processes JSON travel documents for indexing.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads travel documents from JSON files.
    
    Handles:
    - Directory traversal
    - JSON parsing
    - Trip extraction
    - Text flattening for embeddings
    """
    
    # Known country name mappings
    COUNTRY_MAP = {
        "indien": "Indien",
        "india": "India",
        "nepal": "Nepal",
        "sri-lanka": "Sri Lanka",
        "sri_lanka": "Sri Lanka",
        "sri lanka": "Sri Lanka",
        "bhutan": "Bhutan",
        "pakistan": "Pakistan",
        "bangladesh": "Bangladesh",
        "bangladesch": "Bangladesch",
        "chile": "Chile",
        "argentinien": "Argentinien",
        "argentina": "Argentina",
        "china": "China",
        "japan": "Japan",
        "italien": "Italien",
        "italy": "Italy",
        "deutschland": "Deutschland",
        "germany": "Germany",
    }
    
    def load_directory(
        self,
        directory: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load all JSON documents from a directory.
        
        Args:
            directory: Path to documents directory
            
        Returns:
            Tuple of (texts, metadata)
        """
        texts: List[str] = []
        metadata: List[Dict[str, Any]] = []
        
        # Find all JSON files
        pattern = os.path.join(directory, "**", "*.json")
        paths = glob.glob(pattern, recursive=True)
        
        for path in paths:
            try:
                doc_texts, doc_meta = self._load_file(path)
                texts.extend(doc_texts)
                metadata.extend(doc_meta)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
                continue
        
        logger.info(f"Loaded {len(texts)} documents from {directory}")
        return texts, metadata
    
    def _load_file(
        self,
        path: str,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Load a single JSON file.
        
        Args:
            path: Path to JSON file
            
        Returns:
            Tuple of (texts, metadata)
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not isinstance(data, dict):
            return [], []
        
        provider = str(data.get("name", "")).strip()
        website = str(data.get("website", "")).strip()
        trips = data.get("trips", [])
        
        texts = []
        metadata = []
        
        # Index per-trip for better recall
        if isinstance(trips, list) and trips:
            for idx, trip in enumerate(trips):
                try:
                    text, meta = self._process_trip(
                        trip, idx, path, provider, website
                    )
                    texts.append(text)
                    metadata.append(meta)
                except Exception:
                    # Fallback to file-level
                    text = self._flatten_json(data)
                    texts.append(text)
                    metadata.append({
                        "path": path,
                        "title": os.path.basename(path),
                        "preview": text[:500],
                    })
        else:
            # No trips - index whole document
            text = self._flatten_json(data)
            texts.append(text)
            metadata.append({
                "path": path,
                "title": os.path.basename(path),
                "preview": text[:500],
            })
        
        return texts, metadata
    
    def _process_trip(
        self,
        trip: Dict[str, Any],
        index: int,
        path: str,
        provider: str,
        website: str,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a single trip entry.
        
        Args:
            trip: Trip data
            index: Trip index in file
            path: Source file path
            provider: Provider name
            website: Provider website
            
        Returns:
            Tuple of (text, metadata)
        """
        trip_name = str(trip.get("name", "")).strip()
        country = self._infer_country(trip)
        duration = str(trip.get("trip_duration", "")).strip()
        trip_type = str(trip.get("trip_type", "")).strip()
        
        # Summarize dates
        dates = trip.get("trip_dates", []) or []
        date_summary = self._summarize_dates(dates[:3])
        
        # Build focused text for embeddings
        focused = {
            "provider": provider,
            "website": website,
            "trip": {
                "name": trip_name,
                "country": country,
                "trip_id": trip.get("trip_id"),
                "duration": duration,
                "type": trip_type,
                "url": trip.get("url"),
                "dates": dates,
            }
        }
        text = self._flatten_json(focused)
        
        # Build metadata
        title = f"{provider} – {trip_name}"
        if country:
            title += f" ({country})"
        
        meta = {
            "path": path,
            "provider": provider,
            "website": website,
            "trip_index": index,
            "trip_name": trip_name,
            "country": country,
            "duration": duration,
            "trip_type": trip_type,
            "date_summary": date_summary,
            "title": title.strip(" – ()"),
            "preview": text[:500],
        }
        
        return text, meta
    
    def _infer_country(self, trip: Dict[str, Any]) -> str:
        """
        Infer country from trip data.
        
        Args:
            trip: Trip data
            
        Returns:
            Country name
        """
        # Check explicit country field
        country = trip.get("country")
        if isinstance(country, str) and country.strip():
            return country
        
        # Try URL path
        url = str(trip.get("url") or "")
        try:
            path = urlparse(url).path or ""
            parts = [p for p in path.split("/") if p]
            
            for part in parts:
                key = part.lower()
                if key in self.COUNTRY_MAP:
                    return self.COUNTRY_MAP[key]
            
            # Check for /reise/<country>/
            if "reise" in parts:
                idx = parts.index("reise")
                if idx + 1 < len(parts):
                    key = parts[idx + 1].lower().replace("_", " ")
                    if key in self.COUNTRY_MAP:
                        return self.COUNTRY_MAP[key]
                    return self._normalize_country_display(key)
        except Exception:
            pass
        
        # Try trip name
        name = str(trip.get("name") or "").lower()
        for key, value in self.COUNTRY_MAP.items():
            if key in name:
                return value
        
        return ""
    
    def _normalize_country_display(self, raw: str) -> str:
        """Normalize country name for display."""
        if not raw:
            return ""
        r = str(raw).strip()
        if r.isupper():
            return r
        return r.title()
    
    def _summarize_dates(self, dates: List[Dict[str, Any]]) -> str:
        """
        Create date summary string.
        
        Args:
            dates: List of date entries
            
        Returns:
            Summary string
        """
        parts = []
        for d in dates:
            sd = d.get("start_date", "")
            ed = d.get("end_date", "")
            pr = d.get("price", "")
            ap = d.get("airport", "")
            parts.append(f"{sd}→{ed} | {pr} | {ap}")
        return "; ".join(parts)
    
    def _flatten_json(self, obj: Any) -> str:
        """
        Flatten JSON object to text.
        
        Args:
            obj: JSON object
            
        Returns:
            Flattened text
        """
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(f"{k}: {self._flatten_json(v)}")
            return "\n".join(parts)
        
        if isinstance(obj, list):
            return "\n".join(self._flatten_json(v) for v in obj)
        
        return str(obj)


def load_documents(directory: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Convenience function to load documents.
    
    Args:
        directory: Path to documents directory
        
    Returns:
        Tuple of (texts, metadata)
    """
    loader = DocumentLoader()
    return loader.load_directory(directory)
