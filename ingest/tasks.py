"""
Celery-задача обработки одного документа.

Конвейер: поиск файла по s3_fileid -> парсинг -> нормализация ->
обработка чертежей -> сохранение в БД -> публикация результата в состояние
запроса. Когда готов последний документ пакета, срабатывает опциональная
push-доставка.

Идемпотентность: doc_id детерминированно выводится из s3_fileid, строка
документа блокируется через SELECT ... FOR UPDATE, а при operation=create
уже проиндексированный документ отдаётся из БД без повторной обработки.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery.exceptions import SoftTimeLimitExceeded

from core.celery_app import app
from core.config import settings
from core.db.session import SessionLocal
from core.drawing_processor import DrawingProcessor
from core.mappers import document_to_metadata, fragment_to_row, row_to_fragment
from core.models.contract import DocumentMetadata, DocumentResult, ParseResponse
from core.normalizer import TextNormalizer
from core.providers.document_parser_factory import DocumentParserFactory
from core.providers.file_locator import FileLocator, LocatedFile, SourceFileNotFound
from core.providers.storage import StorageProviderFactory
from core.repositories import ChunkRepository, DocumentRepository
from core.result_aggregator import (
    claim_publication,
    get_request_state,
    mark_started,
    put_document_result,
)
from core.result_publisher import publish_result

logger = logging.getLogger(__name__)


# ===========================================================================
# Вспомогательные функции
# ===========================================================================

def generate_doc_id(s3_fileid: str) -> str:
    """Стабильный идентификатор документа: один и тот же файл — один doc_id."""
    digest = hashlib.sha256(s3_fileid.encode("utf-8")).hexdigest()
    return f"doc-{digest[:32]}"


def compute_file_hash(storage, uri: str) -> Optional[str]:
    """SHA-256 содержимого. Для S3 пропускаем — это лишняя полная выкачка."""
    if uri.startswith("s3://"):
        return None
    try:
        digest = hashlib.sha256()
        with storage.get_stream(uri) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Не удалось посчитать хеш %s: %s", uri, exc)
        return None


def build_error_document(s3_fileid: str, doc_id: str, source_path: str, error: str) -> Dict[str, Any]:
    return DocumentResult(
        s3_fileid=s3_fileid,
        document_metadata=DocumentMetadata(
            doc_id=doc_id,
            language=settings.DEFAULT_LANGUAGE,
            source_path=source_path,
            updated_at=datetime.now(timezone.utc),
            status="error",
        ),
        fragments=[],
        error=error,
    ).model_dump(mode="json")


def finalize(request_id: str, dialog_id: str, s3_fileid: str, document: Dict[str, Any]) -> None:
    """Кладёт документ в состояние запроса и, если пакет готов, публикует его."""
    processed = put_document_result(request_id, s3_fileid, document)
    state = get_request_state(request_id)
    if not state:
        logger.warning("Состояние запроса %s не найдено (истёк TTL?)", request_id)
        return
    if not (state["total"] > 0 and processed >= state["total"]):
        return
    # Задачи пакета заканчиваются параллельно, и «обработаны все» могут
    # увидеть сразу несколько. Публикует та, что первой заняла флаг.
    if not claim_publication(request_id):
        logger.info("Результат запроса %s уже опубликован — повтор пропущен", request_id)
        return

    logger.info(
        "Запрос %s завершён: %d/%d, статус %s",
        request_id, state["processed"], state["total"], state["status"],
    )
    publish_result(
        ParseResponse(
            request_id=request_id,
            dialog_id=dialog_id or state.get("dialog_id", ""),
            documents=[DocumentResult(**d) for d in state["documents"]],
        )
    )


# ===========================================================================
# Задача
# ===========================================================================

@app.task(
    bind=True,
    name="ingest.tasks.process_document",
    queue=settings.CELERY_ML_QUEUE,
    max_retries=settings.CELERY_TASK_MAX_RETRIES,
    default_retry_delay=60,
    acks_late=True,
)
def process_document(
    self,
    request_id: str,
    dialog_id: str,
    s3_fileid: str,
    operation: str = "create",
) -> Dict[str, Any]:
    """Обрабатывает один файл пакета. Возвращает краткий статус для логов."""
    mark_started(request_id)
    doc_id = generate_doc_id(s3_fileid)
    storage = StorageProviderFactory.default()
    session = SessionLocal()
    located: Optional[LocatedFile] = None

    try:
        # ------------------------------------------------------ 1. Поиск файла
        try:
            located = FileLocator(storage).locate(s3_fileid)
        except SourceFileNotFound as exc:
            logger.error("Файл не найден: %s", exc)
            finalize(request_id, dialog_id, s3_fileid,
                     build_error_document(s3_fileid, doc_id, "", str(exc)))
            return {"status": "error", "doc_id": doc_id, "reason": "source_not_found"}

        # ------------------------------- 2. Идемпотентность и запись документа
        with session.begin():
            doc_repo = DocumentRepository(session)
            document = doc_repo.get(doc_id, for_update=True)

            if document is not None and document.status == "indexed" and operation != "update":
                logger.info("Документ %s уже проиндексирован, отдаём из БД", doc_id)
                fragments = [row_to_fragment(r) for r in ChunkRepository(session).get_by_doc_id(doc_id)]
                cached = DocumentResult(
                    s3_fileid=s3_fileid,
                    document_metadata=document_to_metadata(document),
                    fragments=fragments,
                ).model_dump(mode="json")
                finalize(request_id, dialog_id, s3_fileid, cached)
                return {"status": "already_processed", "doc_id": doc_id}

            if document is not None and document.status == "pending":
                if doc_repo.is_stale(document, settings.PROCESSING_TIMEOUT_SECONDS):
                    logger.warning("Документ %s завис в pending, перезапускаем", doc_id)
                else:
                    logger.warning(
                        "Документ %s уже обрабатывается другим запросом — "
                        "обрабатываем повторно, чтобы ответить на свой request_id", doc_id
                    )

            if document is None:
                doc_repo.create({
                    "id": doc_id,
                    "s3_fileid": s3_fileid,
                    "source_path": located.uri,
                    "language": settings.DEFAULT_LANGUAGE,
                    "status": "pending",
                    "extra_data": {},
                })
            else:
                doc_repo.update_metadata(
                    doc_id, status="pending", source_path=located.uri, s3_fileid=s3_fileid
                )

        # ------------------------------------- 3. Долгие операции (без сессии)
        metadata: Dict[str, Any] = {
            "doc_id": doc_id,
            "s3_fileid": s3_fileid,
            "source_path": located.uri,
            "file_type": located.file_type,
            "language": settings.DEFAULT_LANGUAGE,
            "asset_prefix": f"{settings.ASSETS_PREFIX.rstrip('/')}/{doc_id}/",
        }

        parser = DocumentParserFactory.get_parser(storage)
        parse_result = parser.parse(located.uri, located.file_type, metadata)
        metadata["source_kind"] = parse_result.source_kind
        # Классификация до разбора нужна обработчику чертежей: по ней он
        # понимает, что сам документ — растровый лист, а не текст с
        # картинками. Без этого скан чертежа проходил мимо ветки чертежей.
        metadata["classification"] = (parse_result.ladder or {}).get("classification") or {}

        normalizer = TextNormalizer()
        fragments, flags = normalizer.normalize_with_flags(parse_result, metadata)
        # Пометки привязываются к фрагменту по его идентификатору, а не по
        # месту в списке: обработчик чертежей вставляет разбор листа целиком
        # в начало, и позиционное сопоставление уезжает на один фрагмент.
        flags_by_id = {
            fragment.fragment_id: extra for fragment, extra in zip(fragments, flags)
        }

        fragments = DrawingProcessor(storage).enrich_fragments(fragments, metadata)

        # ----------------------------------------- 4. Сохранение (транзакция)
        file_hash = compute_file_hash(storage, located.uri)
        raw_uri = f"{settings.RAW_PARSE_S3_PREFIX.rstrip('/')}/{doc_id}.json"

        with session.begin():
            chunk_repo = ChunkRepository(session)
            doc_repo = DocumentRepository(session)

            chunk_repo.delete_by_doc_id(doc_id)
            rows = [
                fragment_to_row(fragment, doc_id, flags_by_id.get(fragment.fragment_id, {}))
                for fragment in fragments
            ]
            if rows:
                chunk_repo.create_chunks_bulk(rows)

            doc_repo.update_metadata(
                doc_id,
                status="indexed",
                file_hash=file_hash,
                language=metadata["language"],
            )
            ladder = parse_result.ladder or {}
            doc_repo.update_extra(doc_id, {
                "parsed_pages_count": parse_result.page_count or len(parse_result.pages()),
                "total_fragments": len(fragments),
                "parser": parse_result.parser_name,
                "source_kind": parse_result.source_kind,
                "is_fallback": parse_result.is_fallback,
                "needs_review": sum(1 for f in flags if f.get("needs_review")),
                "raw_parse_uri": raw_uri,
                "last_operation": operation,
                # Как выбирался уровень — рядом с документом, а не только в
                # логе воркера и в сыром JSON. Без этого на вопрос «почему в
                # ответе OCR, а не разбор» нельзя ответить, не имея доступа
                # к хранилищу.
                "chosen_level": ladder.get("chosen_level"),
                "chosen_strategy": ladder.get("chosen_strategy"),
                "ladder_attempts": ladder.get("attempts") or [],
                "fallback_won": bool(ladder.get("fallback_won")),
                "degraded": list(parse_result.degraded),
                "text_layer_stats": parse_result.text_layer_stats or {},
            })
            document = doc_repo.get(doc_id)
            doc_metadata = document_to_metadata(document)

        # ------------------------------------------ 5. Сырой результат парсера
        try:
            storage.write_file(
                raw_uri,
                json.dumps(parse_result.model_dump(mode="json"), ensure_ascii=False).encode("utf-8"),
            )
        except Exception as exc:  # noqa: BLE001 — не повод терять обработанный документ
            logger.warning("Не удалось сохранить сырой результат %s: %s", raw_uri, exc)

        # -------------------------------------------------- 6. Ответ по файлу
        result = DocumentResult(
            s3_fileid=s3_fileid,
            document_metadata=doc_metadata,
            fragments=fragments,
        ).model_dump(mode="json")
        finalize(request_id, dialog_id, s3_fileid, result)

        if parse_result.degraded:
            logger.error(
                "Документ %s разобран с деградацией: %s",
                doc_id, "; ".join(parse_result.degraded),
            )
        logger.info("Документ %s обработан: %d фрагментов", doc_id, len(fragments))
        return {"status": "success", "doc_id": doc_id, "fragments": len(fragments)}

    except SoftTimeLimitExceeded:
        logger.error("Мягкий таймаут на документе %s", doc_id)
        _mark_error(session, doc_id)
        finalize(request_id, dialog_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", "processing timeout"
        ))
        return {"status": "error", "doc_id": doc_id, "reason": "timeout"}

    except Exception as exc:  # noqa: BLE001
        logger.error("Обработка %s не удалась: %s", s3_fileid, exc, exc_info=True)
        _mark_error(session, doc_id)

        if self.request.retries < self.max_retries:
            countdown = 60 * (self.request.retries + 1)
            logger.info("Повтор %s через %d с", s3_fileid, countdown)
            raise self.retry(exc=exc, countdown=countdown)

        logger.critical("Документ %s окончательно не обработан", doc_id)
        finalize(request_id, dialog_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", str(exc)
        ))
        return {"status": "error", "doc_id": doc_id, "reason": str(exc)}

    finally:
        session.close()


def _mark_error(session, doc_id: str) -> None:
    try:
        session.rollback()
        with session.begin():
            DocumentRepository(session).update_status(doc_id, "error")
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось выставить статус error для %s: %s", doc_id, exc)
