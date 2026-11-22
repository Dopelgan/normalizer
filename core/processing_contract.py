"""Согласованный профиль Parser → RAG; финальная нарезка принадлежит RAG."""

from core.config import settings

# Действующий профиль по умолчанию. Профиль v2 включается флагом
# STRUCTURAL_RAG_V2; брать версию нужно через processing_profile_version().
PROCESSING_PROFILE_VERSION = "structural-rag-v1"
PROCESSING_PROFILE_VERSION_V2 = "structural-rag-v2"
DRAWING_MIN_COMPLETENESS = 0.90
DRAWING_MIN_FACT_CONFIDENCE = 0.95


def structural_v2() -> bool:
    """Включён ли профиль structural-rag-v2 (читается при каждом вызове)."""
    return bool(settings.STRUCTURAL_RAG_V2)


def processing_profile_version() -> str:
    """Версия профиля, под которой сейчас выдаётся и кэшируется результат."""
    return PROCESSING_PROFILE_VERSION_V2 if structural_v2() else PROCESSING_PROFILE_VERSION


# Эти параметры описывают потребителя, Parser не токенизирует и не усекает
# структурные единицы под его окно.
RAG_EMBEDDING_PROFILE = {
    "model": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "target_tokens": 512,
    "max_tokens": 768,
    "overlap_ratio": 0.15,
    "owner": "rag",
}
