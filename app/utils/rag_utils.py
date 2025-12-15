from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import re
from datetime import datetime
import os


_ALIASES_CACHE: Dict[str, str] | None = None


def _load_country_aliases() -> Dict[str, str]:
    global _ALIASES_CACHE
    if _ALIASES_CACHE is not None:
        return _ALIASES_CACHE
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # app/
        path = os.path.join(base_dir, "data", "country_aliases.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # normalize keys to lowercase, values as-is (display form from file)
        _ALIASES_CACHE = {str(k).lower(): str(v) for k, v in data.items()}
        return _ALIASES_CACHE
    except Exception:
        _ALIASES_CACHE = {}
        return _ALIASES_CACHE


def normalize_country(value: str | None) -> str:
    if not value:
        return ""
    aliases = _load_country_aliases()
    v = str(value).strip()
    # direct alias key
    m = aliases.get(v.lower())
    if m:
        return m
    # if the value already equals a display value, keep it
    # or try simple title-case normalization
    return v


def extract_countries_from_query(q: str) -> List[str]:
    ql = (q or "").lower()
    aliases = _load_country_aliases()
    out: List[str] = []
    # scan for any alias-key substring occurrences
    for alias_key, display in aliases.items():
        if alias_key and alias_key in ql and display not in out:
            out.append(display)
    return out


def detect_provider_from_query(q: str) -> str:
    ql = (q or "").lower()
    # Extend this map as new providers are added
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
        if key in ql:
            return name
    return ""


def ensure_plain_text_answer(ans: str) -> str:
    s = (ans or "").strip()
    if not s:
        return ""

    def try_parse_any(text: str):
        try:
            return json.loads(text)
        except Exception:
            pass
        opens = []
        for i, ch in enumerate(text):
            if ch in "[{":
                opens.append((i, ch))
            elif ch in "]}":
                if not opens:
                    continue
                start, stype = opens.pop()
                if (stype == "[" and ch != "]") or (stype == "{" and ch != "}"):
                    continue
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    return obj
                except Exception:
                    continue
        return None

    obj = try_parse_any(s)
    if obj is None:
        return s

    def fmt_date(d: Dict[str, Any]) -> str:
        sd = d.get("start_date")
        ed = d.get("end_date")
        pr = d.get("price")
        ap = d.get("airport")
        core = f"{sd}→{ed}" if sd or ed else ""
        if isinstance(pr, (int, float)):
            core += f", {pr}"
        if ap:
            core += f", {ap}"
        return core.strip(", ")

    lines: List[str] = []
    if isinstance(obj, list):
        for r in obj:
            if not isinstance(r, dict):
                lines.append(str(r))
                continue
            head = " - " + " – ".join([x for x in [r.get("provider"), r.get("trip_name")] if x])
            tail_parts: List[str] = []
            if r.get("country"):
                tail_parts.append(f"({r.get('country')})")
            if r.get("url"):
                tail_parts.append(str(r.get("url")))
            if tail_parts:
                head += " " + " ".join(tail_parts)
            lines.append(head)
            dates = r.get("dates") or []
            for d in dates:
                if isinstance(d, dict):
                    lines.append("    - " + fmt_date(d))
        return "\n".join(lines)

    if isinstance(obj, dict):
        pieces = [p for p in [obj.get("provider") or "", obj.get("trip_name") or ""] if p]
        line = " – ".join(pieces) or (obj.get("title") or "")
        if obj.get("country"):
            line += f" ({obj.get('country')})"
        lines.append(line)
        for d in obj.get("dates") or []:
            if isinstance(d, dict):
                lines.append("    - " + fmt_date(d))
        return "\n".join(lines)

    return s


def keyword_score(query: str, m: Dict[str, Any]) -> int:
    ql = (query or "").lower().split()
    corpus = " ".join(
        [
            str(m.get("title") or ""),
            str(m.get("provider") or ""),
            str(m.get("trip_name") or ""),
            str(m.get("country") or ""),
            str(m.get("trip_type") or ""),
            str(m.get("date_summary") or ""),
            str(m.get("preview") or ""),
        ]
    ).lower()
    s = 0
    for tok in ql:
        if len(tok) >= 3 and tok in corpus:
            s += 1
    return s


def parse_date_summary(date_summary: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not date_summary:
        return items
    for part in str(date_summary).split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            dates, price, airport = [x.strip() for x in part.split("|")]
            start, end = [x.strip() for x in dates.split("→")]
            try:
                price_f = float(price)
            except Exception:
                price_f = None
            items.append(
                {
                    "start_date": start or None,
                    "end_date": end or None,
                    "price": price_f,
                    "airport": airport or None,
                }
            )
        except Exception:
            continue
    return items


def extract_url_from_preview(preview: str) -> Optional[str]:
    if not preview:
        return None
    for line in str(preview).splitlines():
        line = line.strip()
        if line.lower().startswith("url:"):
            url = line.split("url:", 1)[1].strip()
            return url or None
    return None


def hydrate_from_source(hit: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    try:
        path = hit.get("path")
        idx = hit.get("trip_index")
        if not path or idx is None:
            return [], None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        trips = data.get("trips") or []
        if not isinstance(trips, list) or idx < 0 or idx >= len(trips):
            return [], None
        trip = trips[idx] or {}
        url = trip.get("url")
        dates_src = trip.get("trip_dates") or []
        items: List[Dict[str, Any]] = []
        for d in dates_src:
            try:
                items.append(
                    {
                        "start_date": d.get("start_date"),
                        "end_date": d.get("end_date"),
                        "price": float(d.get("price")) if d.get("price") is not None else None,
                        "airport": d.get("airport"),
                    }
                )
            except Exception:
                continue
        return items, url
    except Exception:
        return [], None


def intent_flags(q: str) -> Dict[str, bool]:
    ql = (q or "").lower()
    return {
        "sort": ("sort" in ql) or ("sorted" in ql) or ("order" in ql),
        "by_price": ("price" in ql) or ("cheapest" in ql) or ("least" in ql),
        "by_start": ("start date" in ql) or ("start" in ql) or ("earliest" in ql),
        "filter_india": ("india" in ql) or ("indien" in ql),
    }
