"""Согласованный профиль Parser → RAG; финальная нарезка принадлежит RAG."""

PROCESSING_PROFILE_VERSION = "structural-rag-v1"
DRAWING_MIN_COMPLETENESS = 0.90
DRAWING_MIN_FACT_CONFIDENCE = 0.95

# Эти параметры описывают потребителя, Parser не токенизирует и не усекает
# структурные единицы под его окно.
RAG_EMBEDDING_PROFILE = {
    "model": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb18",
    "target_tokens": 512,
    "max_tokens": 768,
    "overlap_ratio": 0.15,
    "owner": "rag",
}
