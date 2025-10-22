import requests
import time
import json
from observe.tracing import get_current_span
from opentelemetry.trace import Status, StatusCode
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

# Ollama API settings
OLLAMA_URL = "https://g9xd91n6gg4npr5q.eu-west-1.aws.endpoints.huggingface.cloud/"
DEFAULT_MODEL = "tgi"

# Embedding model for semantic similarity checks
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def generate_response(prompt: str, tenant_id: str, model: str = DEFAULT_MODEL) -> str:
    """
    Generate a consistent JSON-structured reasoning response across all Ollama models.
    Returns only the concise final answer string in the API response,
    while all structured reasoning is captured via observability spans.
    """
    span = get_current_span()
    start_time = time.time()

    # Base tracing attributes
    span.set_attribute("tenant_id", tenant_id)
    span.set_attribute("LLM.model", model)
    span.set_attribute("LLM.prompt", prompt)
    span.set_attribute("LLM.source", "local_ollama")

    try:
        # ==========================================================
        # Strict schema-enforced prompt for consistent JSON structure
        # ==========================================================
        system_prompt = f"""
You are an observability-enabled AI assistant.
You MUST respond ONLY with a valid JSON object matching this schema exactly:

{{
  "reasoning_trace": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
  "answer": "short, factual answer only",
  "self_confidence": float (0.0–1.0),
  "estimated_hallucination_risk": float (0.0–1.0),
  "justification": "brief justification or confidence reason"
}}

Rules:
- No markdown or text outside the JSON.
- Use double quotes for all strings.
- JSON must start with '{{' and end with '}}'.

User query: {prompt}
"""

        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "inputs": system_prompt, "stream": False, "parameters": {}},
            #payload= { "inputs": system_prompt, 	"parameters": {} },
            #timeout=90,
        )

        latency = time.time() - start_time
        span.set_attribute("LLM.latency_seconds", round(latency, 3))

        # ==========================================================
        # Error handling
        # ==========================================================
        if response.status_code != 200:
            span.set_status(Status(StatusCode.ERROR))
            span.add_event("Error", {"detail": response.text})
            return f"Error: {response.text}"

        data = response.json()
        raw_content = data.get("response", data)

        # ==========================================================
        # Parse JSON or auto-repair malformed output
        # ==========================================================
        reasoning = _parse_and_repair_json(raw_content)

        reasoning_trace = reasoning.get("reasoning_trace", [])
        answer = str(reasoning.get("answer", "")).strip()
        self_conf = float(reasoning.get("self_confidence", 0.0))
        hallu_risk = float(reasoning.get("estimated_hallucination_risk", 0.0))
        justification = reasoning.get("justification", "")

        # ==========================================================
        # Observability attributes and events
        # ==========================================================
        span.set_attribute("LLM.response", answer)
        span.set_attribute("LLM.reasoning_trace_length", len(reasoning_trace))
        span.set_attribute("LLM.reasoning_trace_preview", str(reasoning_trace[:3]))
        span.set_attribute("LLM.self_confidence", self_conf)
        span.set_attribute("LLM.estimated_hallucination_risk", hallu_risk)
        span.set_attribute("LLM.justification", justification)

        for i, step in enumerate(reasoning_trace, 1):
            span.add_event(
                f"Reasoning Step {i}",
                {
                    "tenant_id": tenant_id,
                    "LLM.model": model,
                    "LLM.reasoning_step": step,
                    "LLM.prompt_snippet": prompt[:80] + ("..." if len(prompt) > 80 else ""),
                }
            )

        # ==========================================================
        # Analytical metrics (semantic, hallucination, consistency)
        # ==========================================================
        hallucination_score = _compute_hallucination(prompt, answer)
        semantic_similarity = _compute_similarity(prompt, answer)
        prompt_consistency = _compute_consistency(prompt, answer)

        span.set_attribute("LLM.hallucination_score", hallucination_score)
        span.set_attribute("LLM.semantic_similarity_to_prompt", semantic_similarity)
        span.set_attribute("LLM.prompt_consistency", prompt_consistency)

        span.add_event(
            "Reasoning Summary",
            {
                "tenant_id": tenant_id,
                "LLM.model": model,
                "LLM.confidence": self_conf,
                "LLM.hallucination_risk": hallu_risk,
                "LLM.semantic_similarity": semantic_similarity,
                "LLM.prompt_consistency": prompt_consistency,
                "LLM.answer_snippet": answer[:100] + ("..." if len(answer) > 100 else ""),
            }
        )

        span.set_status(Status(StatusCode.OK))

        # ==========================================================
        # Return only the concise final answer (no reasoning JSON)
        # ==========================================================
        return answer or "No answer generated."

    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        return f"Exception: {e}"


# ===============================================================
# Helper: JSON Repair / Normalization
# ===============================================================
def _parse_and_repair_json(raw):
    """Ensure Ollama output is valid JSON, auto-fix if malformed."""
    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        raw = raw.strip()

        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Extract substring between first { and last }
        if "{" in raw and "}" in raw:
            try:
                candidate = raw[raw.index("{"): raw.rindex("}") + 1]
                return json.loads(candidate)
            except Exception:
                pass

        # Fallback
        return {
            "reasoning_trace": ["Model did not follow JSON schema."],
            "answer": raw,
            "self_confidence": 0.5,
            "estimated_hallucination_risk": 0.8,
            "justification": "Auto-repaired malformed response",
        }

    # Non-string fallback
    return {
        "reasoning_trace": [],
        "answer": str(raw),
        "self_confidence": 0.0,
        "estimated_hallucination_risk": 1.0,
        "justification": "Non-string response type",
    }


# ===============================================================
# Helper: Analytical Metrics
# ===============================================================
def _compute_hallucination(prompt: str, response: str) -> float:
    ratio = SequenceMatcher(None, prompt.lower(), response.lower()).ratio()
    hallucination_keywords = ["i think", "maybe", "could be", "possibly", "likely", "i assume"]
    uncertainty_score = sum(k in response.lower() for k in hallucination_keywords) / len(hallucination_keywords)
    return round((1 - ratio) * 0.7 + uncertainty_score * 0.3, 3)


def _compute_similarity(prompt: str, response: str) -> float:
    try:
        emb1 = embedder.encode(prompt, convert_to_tensor=True)
        emb2 = embedder.encode(response, convert_to_tensor=True)
        return round(util.cos_sim(emb1, emb2).item(), 3)
    except Exception:
        return 0.0


def _compute_consistency(prompt: str, response: str) -> float:
    seq_ratio = SequenceMatcher(None, prompt.lower(), response.lower()).ratio()
    semantic_ratio = _compute_similarity(prompt, response)
    return round(seq_ratio * 0.4 + semantic_ratio * 0.6, 3)
