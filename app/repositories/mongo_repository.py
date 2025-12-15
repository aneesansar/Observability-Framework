from __future__ import annotations
from typing import Any, Dict, List
from app.core import settings

try:
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover
    MongoClient = None  # type: ignore


def query_candidates(terms: Dict[str, Any], top_k: int) -> List[Dict[str, Any]]:
    """Fetch candidate rows from Mongo based on extracted terms.

    Falls back to empty list if Mongo is not configured or unavailable.
    """
    mongo_uri = (settings.mongo_uri or "").strip()
    if not mongo_uri or MongoClient is None:
        return []

    try:
        client = MongoClient(mongo_uri)
        coll = client[settings.mongo_db][settings.mongo_coll]
        filt: Dict[str, Any] = {}
        provider = (terms.get("provider") or "").strip()
        if provider:
            filt["provider"] = {"$regex": provider, "$options": "i"}
        countries = terms.get("countries") or []
        if countries:
            ors = []
            for c in countries:
                try:
                    import re as _re
                    pattern = f"^{_re.escape(str(c))}$"
                except Exception:
                    pattern = str(c)
                ors.append({"country": {"$regex": pattern, "$options": "i"}})
            filt["$or"] = ors
        proj = {
            "path": 1,
            "provider": 1,
            "website": 1,
            "trip_index": 1,
            "trip_name": 1,
            "country": 1,
            "duration": 1,
            "trip_type": 1,
            "date_summary": 1,
            "title": 1,
            "preview": 1,
        }
        cur = coll.find(filt, proj).limit(max(200, top_k))
        return [dict(x) for x in cur]
    except Exception:
        return []
