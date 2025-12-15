"""
Rule-based query parser for travel queries.
Extracts countries, airports, sorting, and filters without relying on LLM.
"""
import re
from typing import Dict, Any, List
from app.utils.rag_utils import normalize_country, extract_countries_from_query


def parse_query_robust(question: str) -> Dict[str, Any]:
    """
    Robust rule-based query parser.
    Extracts: countries, airport, provider, sort_by, top_n, date_start, date_end
    """
    ql = question.lower()
    
    result = {
        "countries": [],
        "airport": "",
        "provider": "",
        "sort_by": "",
        "top_n": 5,
        "date_start": "",
        "date_end": ""
    }
    
    # 1. Extract sort intent
    if any(kw in ql for kw in ["cheapest", "cheap", "günstig", "guenstig", "lowest price", "preis"]):
        result["sort_by"] = "price"
    elif any(kw in ql for kw in ["earliest", "soonest", "first", "früh", "frueh", "soon"]):
        result["sort_by"] = "start"
    
    # 2. Extract airport/departure city (after "from")
    from_match = re.search(r'\bfrom\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', question)
    if from_match:
        result["airport"] = from_match.group(1).strip()
    
    # 3. Extract countries (destination)
    # Pattern 1: "to <Country>"
    to_match = re.search(r'\bto\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', question)
    if to_match:
        country = to_match.group(1).strip()
        normalized = normalize_country(country)
        if normalized:
            result["countries"] = [normalized]
    
    # Pattern 2: Use existing country extraction
    if not result["countries"]:
        result["countries"] = extract_countries_from_query(question)
    
    # 4. Remove departure city from countries if it appears in airport
    if result["airport"]:
        airport_lower = result["airport"].lower()
        result["countries"] = [
            c for c in result["countries"] 
            if airport_lower not in c.lower()
        ]
    
    # 5. Extract provider (common travel companies)
    providers = ["studiosus", "skr", "berge", "meer", "marco polo", "trendtours", "chamäleon", "chamaeleon"]
    for provider in providers:
        if provider in ql:
            result["provider"] = provider.title()
            break
    
    # 6. Extract top_n
    # Pattern: "top 3", "3 trips", "show 5"
    top_match = re.search(r'\b(?:top|show|give)\s+(\d+)\b', ql)
    if top_match:
        result["top_n"] = int(top_match.group(1))
    else:
        num_match = re.search(r'\b(\d+)\s+(?:trip|reise)', ql)
        if num_match:
            result["top_n"] = int(num_match.group(1))
    
    return result
