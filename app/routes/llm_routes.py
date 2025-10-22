from fastapi import APIRouter
from observe.inmemory_exporter import in_memory_exporter

router = APIRouter(prefix="/llm-metrics", tags=["LLM Metrics"])


@router.get("/")
def llm_metrics():
    spans = [
        s for s in in_memory_exporter.get_finished_spans()
        if s.attributes.get("LLM.source") == "local_ollama"
    ]
    return [
        {
            "tenant_id": s.attributes.get("tenant_id"),
            "model": s.attributes.get("LLM.model"),
            "latency": s.attributes.get("LLM.latency_seconds"),
            "hallucination_score": s.attributes.get("LLM.hallucination_score"),
            "semantic_similarity": s.attributes.get("LLM.semantic_similarity_to_prompt"),
            "prompt_consistency": s.attributes.get("LLM.prompt_consistency"),
            "confidence": s.attributes.get("LLM.self_confidence"),
            "hallucination_risk": s.attributes.get("LLM.estimated_hallucination_risk"),
            "reasoning_trace": s.attributes.get("LLM.reasoning_trace_preview"),
            "justification": s.attributes.get("LLM.justification"),
            "prompt": s.attributes.get("LLM.prompt"),
            "response": s.attributes.get("LLM.response"),
        }
        for s in spans
    ]


@router.get("/summary")
def llm_summary():
    spans = [
        s for s in in_memory_exporter.get_finished_spans()
        if s.attributes.get("LLM.source") == "local_ollama"
    ]
    total = len(spans)
    if total == 0:
        return {"message": "No LLM data available"}

    avg_latency = sum(float(s.attributes.get("LLM.latency_seconds", 0)) for s in spans) / total
    avg_hallucination = sum(float(s.attributes.get("LLM.hallucination_score", 0)) for s in spans) / total
    avg_similarity = sum(float(s.attributes.get("LLM.semantic_similarity_to_prompt", 0)) for s in spans) / total
    avg_confidence = sum(float(s.attributes.get("LLM.self_confidence", 0)) for s in spans) / total

    return {
        "total_llm_calls": total,
        "avg_latency_seconds": round(avg_latency, 3),
        "avg_hallucination_score": round(avg_hallucination, 3),
        "avg_semantic_similarity": round(avg_similarity, 3),
        "avg_self_confidence": round(avg_confidence, 3),
    }
