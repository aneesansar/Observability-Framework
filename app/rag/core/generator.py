"""
LLM Generator.

Handles LLM-based response generation with observability.
"""

import json
import logging
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

import requests

from app.observability import (
    get_current_span,
    get_tracer,
    observe,
    SpanAttributes,
)
from opentelemetry.trace import Status, StatusCode

from ..config import rag_settings
from .embeddings import get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    
    answer: str
    reasoning_trace: List[str]
    self_confidence: float
    hallucination_risk: float
    justification: str
    raw_response: Optional[str] = None
    latency_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "reasoning_trace": self.reasoning_trace,
            "self_confidence": self.self_confidence,
            "hallucination_risk": self.hallucination_risk,
            "justification": self.justification,
            "latency_seconds": self.latency_seconds,
        }


class LLMGenerator:
    """
    LLM-based response generator with structured output.
    
    Features:
    - Structured JSON output with reasoning trace
    - Automatic JSON repair for malformed responses
    - Comprehensive observability integration
    - Quality metrics computation
    """
    
    _instance: Optional["LLMGenerator"] = None
    
    def __new__(cls) -> "LLMGenerator":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the generator."""
        if self._initialized:
            return
        
        self._embedding_service = get_embedding_service()
        self._last_question_emb = None
        self._initialized = True
    
    @observe(label="LLMGenerator.generate")
    def generate(
        self,
        prompt: str,
        tenant_id: str,
        model: str = None,
        user_question: str = None,
        context_texts: List[str] = None,
    ) -> GenerationResult:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: Full prompt with context
            tenant_id: Tenant identifier
            model: LLM model to use
            user_question: Original user question
            context_texts: Retrieved context texts
            
        Returns:
            GenerationResult with answer and metadata
        """
        model = model or rag_settings.llm_model
        context_texts = context_texts or []
        
        span = get_current_span()
        start_time = time.time()
        
        # Set basic span attributes
        self._set_span_attributes(span, tenant_id, model, prompt)
        
        try:
            # Build system prompt
            system_prompt = self._build_prompt(prompt)
            
            # Call LLM
            logger.info(f"Calling Ollama with model={model}", extra={"tenant_id": tenant_id})
            response = self._call_llm(system_prompt, model)
            
            latency = time.time() - start_time
            
            # Parse response
            result = self._parse_response(response, latency)
            
            # Record LLM metrics
            self._record_llm_metrics(span, response, latency, model)
            
            # Record quality metrics
            self._record_quality_metrics(
                span, result, prompt, user_question, context_texts, tenant_id
            )
            
            # Add reasoning events
            self._add_reasoning_events(span, result, tenant_id, model, prompt)
            
            span.set_status(Status(StatusCode.OK))
            return result
            
        except Exception as e:
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            logger.error(f"Generation failed: {e}", extra={"tenant_id": tenant_id})
            
            return GenerationResult(
                answer=f"Error: {e}",
                reasoning_trace=["Generation failed"],
                self_confidence=0.0,
                hallucination_risk=1.0,
                justification="Exception occurred",
                latency_seconds=time.time() - start_time,
            )
    
    def _build_prompt(self, user_prompt: str) -> str:
        """Build the system prompt with schema."""
        return f"""
You are a deterministic data-retrieval assistant.
You MUST answer STRICTLY from the provided context data ONLY.

STRICT RULES:
1. Do NOT hallucinate. If a value does not exist in the context, you cannot generate it.
2. Do NOT invent trips, dates, airports, prices, or any fields.
3. Use ONLY the JSON fields exactly as they appear in the context.
4. If no matching data exists, state: "No results found based on the dataset."
5. NEVER add commentary or explanations.

You MUST respond ONLY with a valid JSON object matching this schema exactly:

{{
  "reasoning_trace": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "answer": "short, factual answer only from context",
  "self_confidence": float (0.0–1.0),
  "estimated_hallucination_risk": float (0.0–1.0),
  "justification": "brief justification based on context"
}}

Rules:
- No markdown or text outside the JSON.
- Use double quotes for all strings.
- JSON must start with '{{' and end with '}}'.
- Answer must be deterministic and reproducible.
- Never guess or invent data not in context.

User query: {user_prompt}
"""
    
    def _call_llm(self, prompt: str, model: str) -> Dict[str, Any]:
        """Call the Ollama API."""
        response = requests.post(
            rag_settings.ollama_generate_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "parameters": {},
            },
            timeout=rag_settings.llm_timeout,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.text}")
        
        return response.json()
    
    def _parse_response(
        self,
        response: Dict[str, Any],
        latency: float,
    ) -> GenerationResult:
        """Parse LLM response into GenerationResult."""
        raw_content = response.get("response", response)
        
        # Parse and repair JSON
        parsed = self._parse_and_repair_json(raw_content)
        
        # Handle double-encoded JSON
        parsed = self._unwrap_nested_json(parsed)
        
        return GenerationResult(
            answer=str(parsed.get("answer", "")).strip(),
            reasoning_trace=parsed.get("reasoning_trace", []),
            self_confidence=float(parsed.get("self_confidence", 0.0)),
            hallucination_risk=float(parsed.get("estimated_hallucination_risk", 0.0)),
            justification=parsed.get("justification", ""),
            raw_response=str(raw_content),
            latency_seconds=latency,
        )
    
    def _parse_and_repair_json(self, raw: Any) -> Dict[str, Any]:
        """Parse and repair potentially malformed JSON."""
        if isinstance(raw, dict):
            return raw
        
        if not isinstance(raw, str):
            return self._fallback_response(str(raw))
        
        raw = raw.strip()
        
        # Strip code fences
        if raw.startswith("```"):
            raw = raw.strip("`\n ")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip("\n ")
        
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        
        # Try to extract valid JSON object
        obj = self._extract_json_object(raw)
        if obj is not None:
            return obj
        
        return self._fallback_response(raw)
    
    def _extract_json_object(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract a valid JSON object from text."""
        opens = []
        best = None
        
        for i, ch in enumerate(text):
            if ch == '{':
                opens.append(i)
            elif ch == '}' and opens:
                start = opens.pop()
                candidate = text[start:i+1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        if "reasoning_trace" in obj and "answer" in obj:
                            return obj
                        best = best or obj
                except Exception:
                    continue
        
        return best
    
    def _unwrap_nested_json(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """Unwrap double-encoded JSON in answer field."""
        try:
            ans_field = parsed.get("answer")
            if isinstance(ans_field, str) and "reasoning_trace" in ans_field:
                if ans_field.strip().startswith("{"):
                    nested = json.loads(ans_field)
                    if isinstance(nested, dict) and "answer" in nested:
                        return nested
        except Exception:
            pass
        return parsed
    
    def _fallback_response(self, raw: str) -> Dict[str, Any]:
        """Create fallback response for unparseable output."""
        return {
            "reasoning_trace": ["Model did not follow JSON schema."],
            "answer": raw,
            "self_confidence": 0.5,
            "estimated_hallucination_risk": 0.8,
            "justification": "Auto-repaired malformed response",
        }
    
    def _set_span_attributes(
        self,
        span,
        tenant_id: str,
        model: str,
        prompt: str,
    ) -> None:
        """Set basic span attributes."""
        span.set_attribute(SpanAttributes.TENANT_ID, tenant_id)
        span.set_attribute("function", "generate_response")
        span.set_attribute("generate_response.model", model)
        span.set_attribute("generate_response.prompt_present", True)
    
    def _record_llm_metrics(
        self,
        span,
        response: Dict[str, Any],
        latency: float,
        model: str,
    ) -> None:
        """Record LLM-specific metrics on span."""
        span.set_attribute("LLM.model", model)
        span.set_attribute("LLM.source", "local_ollama")
        span.set_attribute("LLM.latency_seconds", round(latency, 3))
        
        # Token counts from Ollama
        if isinstance(response, dict):
            if "prompt_eval_count" in response:
                span.set_attribute(
                    "Inference.tokens_input",
                    int(response.get("prompt_eval_count", 0))
                )
            if "eval_count" in response:
                span.set_attribute(
                    "Inference.tokens_output",
                    int(response.get("eval_count", 0))
                )
            
            # Duration metrics
            for key, attr in [
                ("total_duration", "Inference.total_duration_ms"),
                ("load_duration", "Inference.load_duration_ms"),
                ("eval_duration", "Inference.eval_duration_ms"),
            ]:
                val = response.get(key)
                if isinstance(val, (int, float)):
                    span.set_attribute(attr, int(val / 1_000_000))
    
    def _record_quality_metrics(
        self,
        span,
        result: GenerationResult,
        prompt: str,
        user_question: str,
        context_texts: List[str],
        tenant_id: str,
    ) -> None:
        """Record quality metrics on span."""
        answer = result.answer
        
        # Record basic result attributes
        span.set_attribute("LLM.response", answer)
        span.set_attribute("LLM.reasoning_trace_length", len(result.reasoning_trace))
        span.set_attribute("LLM.self_confidence", result.self_confidence)
        span.set_attribute("LLM.estimated_hallucination_risk", result.hallucination_risk)
        
        # Compute analytical metrics
        max_sim = 0.0
        if context_texts:
            max_sim = self._embedding_service.compute_max_similarity(answer, context_texts)
        
        hallucination_score = self._compute_hallucination(answer, context_texts, max_sim)
        semantic_similarity = self._embedding_service.compute_similarity(prompt, answer)
        consistency = self._compute_consistency(prompt, answer, semantic_similarity)
        
        span.set_attribute("Calculated.hallucination_score", hallucination_score)
        span.set_attribute("Calculated.max_similarity_to_any_snippet", round(max_sim, 3))
        span.set_attribute("Calculated.semantic_similarity_to_prompt", semantic_similarity)
        span.set_attribute("Calculated.prompt_consistency", consistency)
        
        # DeepEval-style metrics
        self._record_deepeval_metrics(
            span, answer, user_question or prompt, context_texts, tenant_id
        )
    
    def _record_deepeval_metrics(
        self,
        span,
        answer: str,
        question: str,
        context_texts: List[str],
        tenant_id: str,
    ) -> None:
        """Record DeepEval-style quality metrics."""
        try:
            from app.rag.utils.quality_metrics import (
                compute_answer_relevancy,
                compute_faithfulness,
                check_consistency,
                check_hallucination,
            )
            
            relevancy = compute_answer_relevancy(question, answer)
            consistency = check_consistency(question, answer)
            
            span.set_attribute("DeepEval.answer_relevancy", relevancy)
            span.set_attribute("DeepEval.consistency", consistency)
            span.set_attribute("DeepEval.prompt_length", len(question))
            span.set_attribute("DeepEval.response_length", len(answer))
            
            if context_texts:
                faithfulness = compute_faithfulness(answer, context_texts)
                hallucination = check_hallucination(question, answer, context_texts)
                
                span.set_attribute("DeepEval.faithfulness", faithfulness)
                span.set_attribute("DeepEval.hallucination", hallucination)
                span.set_attribute("DeepEval.has_context", True)
                span.set_attribute(
                    "DeepEval.context_length",
                    len(" ".join(str(c) for c in context_texts))
                )
            else:
                span.set_attribute("DeepEval.faithfulness", 0.0)
                span.set_attribute("DeepEval.hallucination", False)
                span.set_attribute("DeepEval.has_context", False)
                
        except Exception as e:
            logger.warning(f"Failed to compute DeepEval metrics: {e}")
    
    def _add_reasoning_events(
        self,
        span,
        result: GenerationResult,
        tenant_id: str,
        model: str,
        prompt: str,
    ) -> None:
        """Add reasoning step events to span."""
        for i, step in enumerate(result.reasoning_trace, 1):
            span.add_event(
                f"Reasoning Step {i}",
                {
                    "tenant_id": tenant_id,
                    "LLM.model": model,
                    "LLM.reasoning_step": step,
                    "LLM.prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                }
            )
        
        span.add_event(
            "Reasoning Summary",
            {
                "tenant_id": tenant_id,
                "LLM.model": model,
                "LLM.confidence": result.self_confidence,
                "LLM.hallucination_risk": result.hallucination_risk,
                "LLM.answer_snippet": result.answer[:100] + ("..." if len(result.answer) > 100 else ""),
            }
        )
    
    def _compute_hallucination(
        self,
        answer: str,
        context_texts: List[str],
        max_sim: float,
    ) -> float:
        """Compute hallucination score."""
        uncertainty_terms = [
            "i think", "maybe", "could be", "possibly", "likely", "i assume",
            "vielleicht", "könnte", "möglicherweise", "wahrscheinlich",
        ]
        
        ans_lower = (answer or "").lower()
        uncertainty = sum(term in ans_lower for term in uncertainty_terms)
        uncertainty_score = uncertainty / max(len(uncertainty_terms), 1)
        
        base = 1.0 - max(0.0, min(1.0, max_sim))
        return round(base * 0.8 + uncertainty_score * 0.2, 3)
    
    def _compute_consistency(
        self,
        prompt: str,
        response: str,
        semantic_sim: float,
    ) -> float:
        """Compute consistency score."""
        seq_ratio = SequenceMatcher(
            None, prompt.lower(), response.lower()
        ).ratio()
        return round(seq_ratio * 0.4 + semantic_sim * 0.6, 3)


def get_generator() -> LLMGenerator:
    """Get the LLM generator instance."""
    return LLMGenerator()


@observe(label="generate_response")
def generate_response(
    prompt: str,
    tenant_id: str,
    model: str = None,
    user_question: str = None,
    context_texts: List[str] = None,
    **kwargs,
) -> str:
    """
    Generate a response using the LLM.
    
    This is the main entry point for LLM generation.
    
    Args:
        prompt: Full prompt with context
        tenant_id: Tenant identifier
        model: LLM model to use
        user_question: Original user question
        context_texts: Retrieved context texts
        
    Returns:
        Generated answer string
    """
    generator = get_generator()
    result = generator.generate(
        prompt=prompt,
        tenant_id=tenant_id,
        model=model,
        user_question=user_question,
        context_texts=context_texts,
    )
    return result.answer or "No answer generated."
