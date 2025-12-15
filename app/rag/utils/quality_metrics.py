"""
Quality Metrics.

DeepEval-inspired metrics for LLM response quality assessment.
"""

import re
from typing import List, Union


def compute_answer_relevancy(prompt: str, response: str) -> float:
    """
    Compute answer relevancy score.
    
    Measures how well the response addresses the prompt's intent.
    
    Args:
        prompt: User's question
        response: LLM response
        
    Returns:
        Relevancy score (0.0 to 1.0)
    """
    if not prompt or not response:
        return 0.0
    
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    
    # Stop words to filter
    stop_words = {
        "the", "a", "an", "to", "for", "of", "in", "on", "and", "or", "please",
        "show", "me", "find", "get", "give", "tell", "what", "where", "when", "how",
        "provide", "looking", "search", "suche", "möchte", "moechte"
    }
    
    # Intent keywords with multilingual support
    intent_map = {
        "cheapest": ["cheapest", "cheap", "price", "günstig", "preis"],
        "italy": ["italy", "italien", "italia"],
        "germany": ["germany", "deutschland"],
        "france": ["france", "frankreich"],
        "trip": ["trip", "trips", "reise", "reisen"],
    }
    
    # Extract meaningful words from prompt
    prompt_words = [w.lower() for w in prompt.split() if len(w) >= 3]
    meaningful_words = [w for w in prompt_words if w not in stop_words]
    
    if not meaningful_words:
        return 0.5
    
    # Check direct matches
    direct_matches = sum(1 for word in meaningful_words if word in response_lower)
    
    # Check intent matches
    intent_matches = 0
    for key, aliases in intent_map.items():
        if key in prompt_lower or any(alias in prompt_lower for alias in aliases[:2]):
            if any(alias in response_lower for alias in aliases):
                intent_matches += 1
    
    # Structure boost
    structure_indicators = ["→", "–", "http", ".com", ".de"]
    has_structure = any(ind in response for ind in structure_indicators)
    structure_boost = 0.15 if has_structure else 0.0
    
    # Price sorting boost
    if any(kw in prompt_lower for kw in ["cheapest", "cheap", "günstig"]):
        prices = re.findall(r'\b\d{3,4}\.0\b', response)
        if len(prices) >= 2:
            price_values = [float(p) for p in prices]
            is_sorted = all(
                price_values[i] <= price_values[i+1]
                for i in range(len(price_values)-1)
            )
            if is_sorted:
                intent_matches += 3
                structure_boost += 0.1
    
    # Multiple results boost
    if response.count('–') >= 3 or response.count('http') >= 3:
        structure_boost += 0.05
    
    base_score = (direct_matches + intent_matches) / max(1, len(meaningful_words) + 2)
    return min(1.0, base_score + structure_boost)


def compute_faithfulness(
    response: str,
    context: Union[str, List[str]],
) -> float:
    """
    Compute faithfulness score.
    
    Measures how well the response is grounded in the provided context.
    
    Args:
        response: LLM response
        context: Context string or list of context strings
        
    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not context or not response:
        return 0.0
    
    # Normalize context
    if isinstance(context, list):
        context_str = " ".join(str(c) for c in context).lower()
    else:
        context_str = str(context).lower()
    
    response_lower = response.lower()
    
    # Extract meaningful words from response
    response_words = re.findall(r'\b\w{4,}\b', response_lower)
    
    # Filter common words
    common = {
        "http", "https", "www", "com", "reisen", "reise", "trips", "trip",
        "italien", "italy", "deutschland", "germany", "from", "with", "this",
        "that", "these", "those", "about", "more", "some", "such", "which"
    }
    meaningful_words = [w for w in response_words if w not in common]
    
    if not meaningful_words:
        return 0.0
    
    # Count grounded words
    grounded = sum(1 for word in meaningful_words if word in context_str)
    
    # Partial credit for compound words
    for word in meaningful_words:
        if word not in context_str and len(word) >= 5:
            if any(word in ctx_word for ctx_word in context_str.split()):
                grounded += 0.5
    
    # Extract and verify key entities
    providers = re.findall(r'\b(studiosus|skr|trendtours|berge|meer)\b', response_lower)
    prices = re.findall(r'\b\d{3,4}\.0\b', response)
    dates = re.findall(r'\d{4}-\d{2}-\d{2}', response)
    
    entity_grounded = 0
    entity_total = 0
    
    for provider in set(providers):
        entity_total += 1
        if provider in context_str:
            entity_grounded += 1
    
    for price in set(prices):
        entity_total += 1
        if price in context_str:
            entity_grounded += 1
    
    # Structure boost
    has_structure = bool(dates) and bool(prices)
    structure_boost = 0.1 if has_structure else 0.0
    
    # Combine scores
    word_score = grounded / max(1, len(meaningful_words))
    entity_score = entity_grounded / max(1, entity_total) if entity_total > 0 else 0.0
    
    base_score = (word_score * 0.6) + (entity_score * 0.4)
    return min(1.0, base_score + structure_boost)


def check_consistency(prompt: str, response: str) -> bool:
    """
    Check if response is consistent with prompt intent.
    
    Args:
        prompt: User's question
        response: LLM response
        
    Returns:
        True if consistent
    """
    if not prompt or not response:
        return False
    
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    
    # Country aliases
    country_aliases = {
        "italy": ["italy", "italien", "italia"],
        "germany": ["germany", "deutschland", "germania"],
        "france": ["france", "frankreich", "francia"],
        "spain": ["spain", "spanien", "españa"],
        "japan": ["japan", "japón"],
    }
    
    # Check if country from prompt appears in response
    entity_found = False
    for key, aliases in country_aliases.items():
        if key in prompt_lower:
            if any(alias in response_lower for alias in aliases):
                entity_found = True
                break
    
    # Fallback: extract capitalized words
    if not entity_found:
        entities = re.findall(r'\b[A-ZÄÖÜ][a-zäöüß]+\b', prompt)
        entity_found = any(entity.lower() in response_lower for entity in entities)
    
    # Check for proper formatting
    has_format = "→" in response and ("–" in response or "http" in response)
    
    return entity_found and has_format


def check_hallucination(
    prompt: str,
    response: str,
    context: Union[str, List[str]] = None,
) -> bool:
    """
    Check if response contains hallucinated content.
    
    Args:
        prompt: User's question
        response: LLM response
        context: Optional context for grounding check
        
    Returns:
        True if hallucination detected
    """
    if not context:
        return False
    
    # Normalize context
    if isinstance(context, list):
        context_str = " ".join(str(c).lower() for c in context)
    else:
        context_str = str(context).lower()
    
    response_lower = response.lower()
    
    # Extract meaningful words from response
    response_words = re.findall(r'\b[A-Za-zäöüÄÖÜß]{4,}\b', response_lower)
    
    # Filter common words
    common = {
        "http", "https", "www", "com", "reisen", "reise", "trips", "trip",
        "italien", "italy", "from", "with"
    }
    meaningful_words = [w for w in response_words if w not in common and len(w) >= 4]
    
    if not meaningful_words:
        return False
    
    # Count grounded words
    grounded_count = sum(1 for word in meaningful_words if word in context_str)
    grounded_ratio = grounded_count / len(meaningful_words)
    
    # Threshold: 60% grounded is acceptable
    return grounded_ratio < 0.6
