"""
Prompt Similarity Service.

Computes similarity between prompts using stored embeddings from MongoDB.
Used for LLM evaluation to detect similar past queries and their outcomes.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports to avoid circular dependencies
_prompt_repo = None
_embedding_service = None


def _get_prompt_repo():
    """Get prompt repository lazily."""
    global _prompt_repo
    if _prompt_repo is None:
        try:
            from app.db.repositories import get_prompt_repository
            _prompt_repo = get_prompt_repository()
        except Exception as e:
            logger.warning(f"Failed to initialize prompt repository: {e}")
    return _prompt_repo


def _get_embedding_service():
    """Get embedding service lazily."""
    global _embedding_service
    if _embedding_service is None:
        try:
            from app.rag.embeddings import get_embedding_service
            _embedding_service = get_embedding_service()
        except Exception as e:
            logger.warning(f"Failed to initialize embedding service: {e}")
    return _embedding_service


@dataclass
class SimilarPrompt:
    """Represents a similar prompt found in the database."""
    
    prompt_id: str
    prompt_text: str
    similarity_score: float
    quality_score: float
    feedback_rating: Optional[bool]
    conversation_id: Optional[str]


@dataclass
class PromptSimilarityResult:
    """Result of prompt similarity computation."""
    
    similar_prompts: List[SimilarPrompt]
    total_prompts_searched: int
    avg_similarity: float
    max_similarity: float
    most_similar: Optional[SimilarPrompt]
    
    # Quality insights from similar prompts
    avg_quality_of_similar: float
    positive_feedback_ratio: float


class PromptSimilarityService:
    """
    Service for computing prompt similarity using MongoDB-stored embeddings.
    
    Provides:
    - Finding similar past prompts
    - Computing similarity metrics
    - Quality insights from similar queries
    """
    
    _instance: Optional["PromptSimilarityService"] = None
    
    def __new__(cls) -> "PromptSimilarityService":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service."""
        if self._initialized:
            return
        
        self._cache: Dict[str, PromptSimilarityResult] = {}
        self._max_cache_size = 100
        self._initialized = True
        logger.info("PromptSimilarityService initialized")
    
    def compute_similarity(
        self,
        prompt_text: str,
        tenant_id: Optional[str] = None,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> PromptSimilarityResult:
        """
        Compute similarity between a prompt and stored prompts.
        
        Args:
            prompt_text: The prompt to compare
            tenant_id: Optional tenant filter
            top_k: Number of similar prompts to return
            min_similarity: Minimum similarity threshold
            
        Returns:
            PromptSimilarityResult with similar prompts and metrics
        """
        # Check cache
        cache_key = f"{hash(prompt_text)}:{tenant_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        try:
            # Get embedding for the input prompt
            embedding_service = _get_embedding_service()
            if embedding_service is None:
                return self._empty_result()
            
            prompt_embedding = embedding_service.embed_text(prompt_text)
            if prompt_embedding is None:
                return self._empty_result()
            
            # Convert to numpy array
            if hasattr(prompt_embedding, 'tolist'):
                prompt_vec = np.array(prompt_embedding)
            else:
                prompt_vec = np.array(list(prompt_embedding))
            
            # Get stored prompts with embeddings
            prompt_repo = _get_prompt_repo()
            if prompt_repo is None:
                return self._empty_result()
            
            stored_prompts = prompt_repo.get_prompts_with_embeddings(
                tenant_id=tenant_id,
                limit=100,  # Search through recent prompts
            )
            
            if not stored_prompts:
                return self._empty_result()
            
            # Compute similarities
            similar_prompts = []
            for stored in stored_prompts:
                if not stored.embedding:
                    continue
                
                stored_vec = np.array(stored.embedding)
                similarity = self._cosine_similarity(prompt_vec, stored_vec)
                
                if similarity >= min_similarity:
                    similar_prompts.append(SimilarPrompt(
                        prompt_id=stored.id,
                        prompt_text=stored.prompt_text,
                        similarity_score=float(similarity),
                        quality_score=stored.response_quality_score,
                        feedback_rating=stored.feedback_rating,
                        conversation_id=stored.conversation_id,
                    ))
            
            # Sort by similarity
            similar_prompts.sort(key=lambda x: x.similarity_score, reverse=True)
            similar_prompts = similar_prompts[:top_k]
            
            # Compute metrics
            result = self._compute_metrics(similar_prompts, len(stored_prompts))
            
            # Cache result
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute prompt similarity: {e}")
            return self._empty_result()
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _compute_metrics(
        self,
        similar_prompts: List[SimilarPrompt],
        total_searched: int,
    ) -> PromptSimilarityResult:
        """Compute aggregate metrics from similar prompts."""
        if not similar_prompts:
            return self._empty_result()
        
        similarities = [p.similarity_score for p in similar_prompts]
        qualities = [p.quality_score for p in similar_prompts if p.quality_score > 0]
        feedbacks = [p.feedback_rating for p in similar_prompts if p.feedback_rating is not None]
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        max_similarity = max(similarities) if similarities else 0.0
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        positive_ratio = sum(1 for f in feedbacks if f) / len(feedbacks) if feedbacks else 0.0
        
        return PromptSimilarityResult(
            similar_prompts=similar_prompts,
            total_prompts_searched=total_searched,
            avg_similarity=round(avg_similarity, 3),
            max_similarity=round(max_similarity, 3),
            most_similar=similar_prompts[0] if similar_prompts else None,
            avg_quality_of_similar=round(avg_quality, 3),
            positive_feedback_ratio=round(positive_ratio, 3),
        )
    
    def _empty_result(self) -> PromptSimilarityResult:
        """Return an empty result."""
        return PromptSimilarityResult(
            similar_prompts=[],
            total_prompts_searched=0,
            avg_similarity=0.0,
            max_similarity=0.0,
            most_similar=None,
            avg_quality_of_similar=0.0,
            positive_feedback_ratio=0.0,
        )
    
    def _cache_result(self, key: str, result: PromptSimilarityResult) -> None:
        """Cache a result with LRU eviction."""
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = result
    
    def get_quality_prediction(
        self,
        prompt_text: str,
        tenant_id: Optional[str] = None,
    ) -> Tuple[float, float]:
        """
        Predict quality score based on similar past prompts.
        
        Args:
            prompt_text: The prompt to predict quality for
            tenant_id: Optional tenant filter
            
        Returns:
            Tuple of (predicted_quality, confidence)
        """
        result = self.compute_similarity(prompt_text, tenant_id)
        
        if not result.similar_prompts:
            return (0.5, 0.0)  # Default with no confidence
        
        # Weight quality by similarity
        weighted_sum = 0.0
        weight_total = 0.0
        
        for prompt in result.similar_prompts:
            if prompt.quality_score > 0:
                weighted_sum += prompt.quality_score * prompt.similarity_score
                weight_total += prompt.similarity_score
        
        if weight_total == 0:
            return (0.5, 0.0)
        
        predicted_quality = weighted_sum / weight_total
        confidence = min(1.0, result.max_similarity * len(result.similar_prompts) / 5)
        
        return (round(predicted_quality, 3), round(confidence, 3))
    
    def clear_cache(self) -> None:
        """Clear the similarity cache."""
        self._cache.clear()


# Singleton instance
_similarity_service: Optional[PromptSimilarityService] = None


def get_prompt_similarity_service() -> PromptSimilarityService:
    """Get the prompt similarity service singleton."""
    global _similarity_service
    if _similarity_service is None:
        _similarity_service = PromptSimilarityService()
    return _similarity_service


def compute_prompt_similarity(
    prompt_text: str,
    tenant_id: Optional[str] = None,
    top_k: int = 10,
) -> PromptSimilarityResult:
    """
    Convenience function to compute prompt similarity.
    
    Args:
        prompt_text: The prompt to compare
        tenant_id: Optional tenant filter
        top_k: Number of similar prompts to return
        
    Returns:
        PromptSimilarityResult
    """
    service = get_prompt_similarity_service()
    return service.compute_similarity(prompt_text, tenant_id, top_k)
