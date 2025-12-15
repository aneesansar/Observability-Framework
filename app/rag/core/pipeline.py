"""
RAG Pipeline.

Orchestrates the complete RAG workflow.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.observability import observe, get_current_span

from ..config import rag_settings
from ..models.documents import RetrievalResult
from ..models.queries import QueryTerms, SortOrder
from .embeddings import get_embedding_service
from .generator import get_generator, GenerationResult
from .retriever import get_retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG pipeline response."""
    
    answer: str
    query_terms: QueryTerms
    retrieved_docs: List[RetrievalResult]
    generation_result: Optional[GenerationResult] = None
    tenant_id: str = "anonymous"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "query_terms": self.query_terms.to_dict(),
            "retrieved_count": len(self.retrieved_docs),
            "tenant_id": self.tenant_id,
        }


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    
    Coordinates:
    1. Query parsing and term extraction
    2. Document retrieval
    3. Context building
    4. LLM response generation
    """
    
    _instance: Optional["RAGPipeline"] = None
    
    def __new__(cls) -> "RAGPipeline":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the pipeline."""
        if self._initialized:
            return
        
        self._retriever = get_retriever()
        self._generator = get_generator()
        self._embedding_service = get_embedding_service()
        self._initialized = True
    
    @observe(label="RAGPipeline.query")
    def query(
        self,
        question: str,
        tenant_id: str = "anonymous",
        top_k: int = None,
        model: str = None,
    ) -> RAGResponse:
        """
        Execute the complete RAG pipeline.
        
        Args:
            question: User's question
            tenant_id: Tenant identifier
            top_k: Number of documents to retrieve
            model: LLM model to use
            
        Returns:
            RAGResponse with answer and metadata
        """
        top_k = top_k or rag_settings.default_top_k
        
        span = get_current_span()
        span.set_attribute("rag.query", question)
        span.set_attribute("rag.tenant_id", tenant_id)
        
        # Step 1: Parse query terms
        query_terms = self._parse_query(question, tenant_id)
        span.set_attribute("rag.has_filters", query_terms.has_filters())
        
        # Step 2: Retrieve documents
        results = self._retrieve(question, query_terms, top_k)
        span.set_attribute("rag.retrieved_count", len(results))
        
        if not results:
            return RAGResponse(
                answer="No relevant documents found for your query.",
                query_terms=query_terms,
                retrieved_docs=[],
                tenant_id=tenant_id,
            )
        
        # Step 3: Build context
        context = self._retriever.build_context(results)
        context_texts = self._retriever.get_context_texts(results)
        
        # Step 4: Build prompt
        prompt = self._build_prompt(question, context)
        
        # Step 5: Generate response
        gen_result = self._generator.generate(
            prompt=prompt,
            tenant_id=tenant_id,
            model=model,
            user_question=question,
            context_texts=context_texts,
        )
        
        return RAGResponse(
            answer=gen_result.answer,
            query_terms=query_terms,
            retrieved_docs=results,
            generation_result=gen_result,
            tenant_id=tenant_id,
        )
    
    def _parse_query(
        self,
        question: str,
        tenant_id: str,
    ) -> QueryTerms:
        """
        Parse query to extract structured terms.
        
        Args:
            question: User's question
            tenant_id: Tenant identifier
            
        Returns:
            QueryTerms with extracted filters
        """
        try:
            from ..utils.query_parser import parse_query_terms
            return parse_query_terms(question, tenant_id)
        except Exception as e:
            logger.warning(f"Query parsing failed: {e}")
            return QueryTerms(raw_query=question)
    
    def _retrieve(
        self,
        question: str,
        query_terms: QueryTerms,
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents.
        
        Args:
            question: User's question
            query_terms: Parsed query terms
            top_k: Number of documents
            
        Returns:
            List of retrieval results
        """
        # Basic retrieval
        results = self._retriever.retrieve(question, top_k=top_k * 2)
        
        # Apply filters from query terms
        if query_terms.countries:
            results = self._filter_by_countries(results, query_terms.countries)
        
        if query_terms.provider:
            results = self._filter_by_provider(results, query_terms.provider)
        
        # Apply sorting
        if query_terms.sort_by == SortOrder.PRICE:
            results = self._sort_by_price(results)
        elif query_terms.sort_by == SortOrder.START_DATE:
            results = self._sort_by_date(results)
        
        return results[:top_k]
    
    def _filter_by_countries(
        self,
        results: List[RetrievalResult],
        countries: List[str],
    ) -> List[RetrievalResult]:
        """Filter results by country."""
        countries_lower = [c.lower() for c in countries]
        return [
            r for r in results
            if r.metadata.country and r.metadata.country.lower() in countries_lower
        ]
    
    def _filter_by_provider(
        self,
        results: List[RetrievalResult],
        provider: str,
    ) -> List[RetrievalResult]:
        """Filter results by provider."""
        provider_lower = provider.lower()
        return [
            r for r in results
            if r.metadata.provider and provider_lower in r.metadata.provider.lower()
        ]
    
    def _sort_by_price(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Sort results by price (ascending)."""
        def get_min_price(r: RetrievalResult) -> float:
            # Extract price from date_summary
            summary = r.metadata.date_summary or ""
            try:
                # Format: "date→date | price | airport"
                for part in summary.split(";"):
                    if "|" in part:
                        price_str = part.split("|")[1].strip()
                        return float(price_str)
            except (ValueError, IndexError):
                pass
            return float("inf")
        
        return sorted(results, key=get_min_price)
    
    def _sort_by_date(
        self,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """Sort results by start date (ascending)."""
        def get_first_date(r: RetrievalResult) -> str:
            summary = r.metadata.date_summary or ""
            try:
                # Format: "date→date | price | airport"
                for part in summary.split(";"):
                    if "→" in part:
                        return part.split("→")[0].strip()
            except (ValueError, IndexError):
                pass
            return "9999-99-99"
        
        return sorted(results, key=get_first_date)
    
    def _build_prompt(self, question: str, context: str) -> str:
        """Build the full prompt with context."""
        return f"""Based on the following travel data, answer the user's question.

CONTEXT:
{context}

QUESTION: {question}

Provide a concise, factual answer based only on the context above."""


def get_rag_pipeline() -> RAGPipeline:
    """Get the RAG pipeline instance."""
    return RAGPipeline()


def query_trips(
    question: str,
    tenant_id: str = "anonymous",
    top_k: int = None,
) -> str:
    """
    Convenience function to query trips.
    
    Args:
        question: User's question
        tenant_id: Tenant identifier
        top_k: Number of documents
        
    Returns:
        Answer string
    """
    pipeline = get_rag_pipeline()
    response = pipeline.query(question, tenant_id, top_k)
    return response.answer
