import os

# Simulated documents for different tenants
TENANT_DOCUMENTS = {
    "tenant_a": [
        "OpenTelemetry helps you collect traces, metrics, and logs from your application.",
        "You can use Jaeger or Zipkin to visualize traces.",
        "Prometheus is commonly used for metric collection and monitoring."
    ],
    "tenant_b": [
        "LangChain provides tools to build applications with LLMs.",
        "FAISS is a library for efficient similarity search and clustering of dense vectors.",
        "LLMs like GPT-4 can be used for summarization and question answering."
    ]
}

def get_documents_for_tenant(tenant_id: str) -> list[str]:
    return TENANT_DOCUMENTS.get(tenant_id, [
        "No tenant-specific data found. Please check the tenant ID or provide custom documents."
    ])
