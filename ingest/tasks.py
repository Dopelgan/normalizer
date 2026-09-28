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
from typing import Any, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded

from core import filetypes, telemetry
from core.celery_app import app
from core.config import settings
from core.contract_finalize import finalize_fragments, normalized_content_hash
from core.db.session import SessionLocal
from core.drawing_processor import DrawingProcessor
from core.mappers import document_to_metadata, fragment_to_row, row_to_fragment
from core.models.contract import (
    DocumentMetadata,
    DocumentResult,
    Fragment,
    ParseResponse,
    ParserError,
    ResultsResponse,
)
from core.models.parse_result import ParseResult
from core.normalizer import TextNormalizer
from core.providers.document_parser import ParserFailed, ParserUnavailable
from core.providers.document_parser_factory import DocumentParserFactory
from core.providers.file_locator import FileLocator, LocatedFile, SourceFileNotFound
from core.providers.office_convert import ConversionFailed, ConversionUnavailable
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


# Исключение -> код причины и признак «повторять ли». Сообщение остаётся
# исходным: код нужен машине, текст — человеку.
_UNSUPPORTED_MARKERS = ("не поддерж", "unsupported", "неизвестный формат")


def classify_error(exc: BaseException, attempt: int = 1) -> ParserError:
    """
    Структурированная ошибка вместо строки.

    Потребителю важно одно: повторять запрос или отправлять файл человеку.
    По строке «Parser failed» этот вопрос не решается.
    """
    message = str(exc)
    lowered = message.lower()

    if isinstance(exc, SoftTimeLimitExceeded):
        return ParserError(reason_code="PARSER_TIMEOUT", message=message or "processing timeout",
                           retryable=True, attempt=attempt)
    if isinstance(exc, SourceFileNotFound) or isinstance(exc, FileNotFoundError):
        return ParserError(reason_code="SOURCE_NOT_FOUND", message=message,
                           retryable=False, attempt=attempt)
    if any(marker in lowered for marker in _UNSUPPORTED_MARKERS):
        return ParserError(reason_code="UNSUPPORTED_FORMAT", message=message,
                           retryable=False, attempt=attempt)
    if isinstance(exc, (ConversionUnavailable, ParserUnavailable)):
        return ParserError(reason_code="EXTRACTION_FAILED", message=message,
                           retryable=True, attempt=attempt)
    if isinstance(exc, (ConversionFailed, ParserFailed)):
        return ParserError(reason_code="EXTRACTION_FAILED", message=message,
                           retryable=False, attempt=attempt)
    if isinstance(exc, OSError):
        return ParserError(reason_code="SOURCE_READ_ERROR", message=message,
                           retryable=True, attempt=attempt)
    return ParserError(reason_code="INTERNAL_ERROR", message=message,
                       retryable=True, attempt=attempt)


def build_error_document(
    s3_fileid: str,
    doc_id: str,
    source_path: str,
    error: Any,
    attempt: int = 1,
) -> Dict[str, Any]:
    """Документ с ошибкой. `error` принимает и исключение, и готовый объект."""
    if isinstance(error, BaseException):
        parser_error = classify_error(error, attempt)
    else:
        parser_error = ParserError.coerce(error)
    return DocumentResult(
        s3_fileid=s3_fileid,
        status="error",
        document_metadata=DocumentMetadata(
            doc_id=doc_id,
            language=settings.DEFAULT_LANGUAGE,
            source_path=source_path,
            updated_at=datetime.now(timezone.utc),
            status="error",
        ),
        fragments=[],
        error=parser_error,
    ).model_dump(mode="json")


def page_report(parse_result: ParseResult) -> Dict[str, Any]:
    """
    Страницы, которые разобрались, и страницы, которые нет.

    Без этого списка потребитель по статусу `partial` знает только то, что
    что-то потерялось, но не знает — что именно.
    """
    failed = sorted({int(page) for page in parse_result.failed_pages})
    seen = {int(page) for page in parse_result.pages()}
    total = parse_result.page_count or (max(seen | set(failed)) if (seen or failed) else 0)
    expected = set(range(1, total + 1)) if total else seen
    processed = sorted((expected | seen) - set(failed))
    return {"pages": total or None, "processed_pages": processed, "failed_pages": failed}


def build_warnings(parse_result: ParseResult, failed_pages: List[int]) -> List[str]:
    """Деградации разбора и непрочитанные страницы — одним списком."""
    warnings = list(parse_result.degraded)
    warnings += [f"PAGE_NOT_PARSED: page {page}" for page in failed_pages]
    return warnings


def build_document_result(
    s3_fileid: str,
    document,
    fragments: List[Fragment],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Документ контракта из строки БД — для повторной выдачи без разбора.

    Поля контракта, которых нет отдельными колонками, лежат в `extra_data`
    именно ради этого случая: уже разобранный файл обязан отдаваться тем же
    набором полей, что и только что разобранный.
    """
    status = extra.get("document_status")
    if status not in ("completed", "partial", "error"):
        status = "error" if document.status == "error" else "completed"
    return DocumentResult(
        s3_fileid=s3_fileid,
        status=status,
        content_hash=f"sha256:{document.file_hash}" if document.file_hash else None,
        normalized_content_hash=extra.get("normalized_content_hash"),
        mime_type=extra.get("mime_type"),
        pages=extra.get("pages_reported") or extra.get("parsed_pages_count") or None,
        processed_pages=list(extra.get("processed_pages") or []),
        failed_pages=list(extra.get("failed_pages") or []),
        document_metadata=document_to_metadata(document),
        fragments=fragments,
        warnings=list(extra.get("warnings") or []),
    ).model_dump(mode="json")


def finalize(
    request_id: str,
    s3_fileid: str,
    document: Dict[str, Any],
    dialog_id: str = "",
    event_id: str = "",
) -> None:
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
    documents = [DocumentResult(**d) for d in state["documents"]]
    resolved_event = event_id or state.get("event_id", "")
    resolved_dialog = dialog_id or state.get("dialog_id", "")

    # Синхронный разбор отдаёт свой envelope с `dialog_id`, фоновый —
    # envelope контракта с `event_id`. Одного на оба случая быть не может:
    # именно этим отличаются два контракта.
    if resolved_event or not resolved_dialog:
        response = ResultsResponse(
            request_id=request_id,
            event_id=resolved_event,
            operation=state.get("operation"),
            status=state["status"],
            documents=documents,
            error=_batch_error(documents) if state["status"] in ("error", "partial") else None,
        )
    else:
        response = ParseResponse(
            request_id=request_id, dialog_id=resolved_dialog, documents=documents,
        )
    publish_result(response)


def _batch_error(documents: List[DocumentResult]) -> Optional[ParserError]:
    """Первая ошибка пакета: по ней потребитель решает, что делать дальше."""
    for document in documents:
        if document.error is not None:
            return document.error
    return None


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
    s3_fileid: str,
    operation: str = "create",
    dialog_id: str = "",
    event_id: str = "",
) -> Dict[str, Any]:
    """
    Обрабатывает один файл пакета. Возвращает краткий статус для логов.

    `event_id` — идентификатор lifecycle-события, по которому результат
    сопоставляется с исходным событием; в синхронном разборе его нет, там
    есть `dialog_id`. Один вместо другого не подставляется.
    """
    mark_started(request_id)
    doc_id = generate_doc_id(s3_fileid)
    storage = StorageProviderFactory.default()
    session = SessionLocal()
    located: Optional[LocatedFile] = None
    # Область сбора таймингов: все замеры внутри складываются сюда, и
    # словарь уезжает в extra_data документа. Без него на вопрос «почему
    # этот файл обрабатывался семь минут» отвечать нечем.
    timings_scope = telemetry.collect()
    timings = timings_scope.__enter__()

    try:
        # ------------------------------------------------------ 1. Поиск файла
        try:
            with telemetry.measure("parse.locate", s3_fileid=s3_fileid):
                located = FileLocator(storage).locate(s3_fileid)
        except SourceFileNotFound as exc:
            logger.error("Файл не найден: %s", exc)
            finalize(
                request_id, s3_fileid,
                build_error_document(s3_fileid, doc_id, "", exc, self.request.retries + 1),
                dialog_id, event_id,
            )
            return {"status": "error", "doc_id": doc_id, "reason": "source_not_found"}

        # ------------------------------- 2. Идемпотентность и запись документа
        with session.begin():
            doc_repo = DocumentRepository(session)
            document = doc_repo.get(doc_id, for_update=True)

            if document is not None and document.status == "indexed" and operation != "update":
                logger.info("Документ %s уже проиндексирован, отдаём из БД", doc_id)
                fragments = [row_to_fragment(r) for r in ChunkRepository(session).get_by_doc_id(doc_id)]
                cached = build_document_result(
                    s3_fileid, document, fragments, dict(document.extra_data or {})
                )
                finalize(request_id, s3_fileid, cached, dialog_id, event_id)
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
        with telemetry.measure(
            "parse.document", doc_id=doc_id, file_type=located.file_type
        ) as span:
            parse_result = parser.parse(located.uri, located.file_type, metadata)
            span["pages"] = parse_result.page_count or len(parse_result.pages())
            span["parser"] = parse_result.parser_name
        metadata["source_kind"] = parse_result.source_kind
        # Классификация до разбора нужна обработчику чертежей: по ней он
        # понимает, что сам документ — растровый лист, а не текст с
        # картинками. Без этого скан чертежа проходил мимо ветки чертежей.
        metadata["classification"] = (parse_result.ladder or {}).get("classification") or {}

        normalizer = TextNormalizer()
        with telemetry.measure("parse.normalize", doc_id=doc_id) as span:
            fragments, flags = normalizer.normalize_with_flags(parse_result, metadata)
            span["fragments"] = len(fragments)
        # Пометки привязываются к фрагменту по его идентификатору, а не по
        # месту в списке: обработчик чертежей вставляет разбор листа целиком
        # в начало, и позиционное сопоставление уезжает на один фрагмент.
        flags_by_id = {
            fragment.fragment_id: extra for fragment, extra in zip(fragments, flags)
        }

        with telemetry.measure("parse.drawings", doc_id=doc_id):
            fragments = DrawingProcessor(storage).enrich_fragments(fragments, metadata)

        # Идентификаторы, происхождение и факты — после ветки чертежей:
        # она заменяет фрагменты и вставляет разбор листа целиком, и только
        # теперь состав документа окончателен.
        with telemetry.measure("parse.facts", doc_id=doc_id) as span:
            fragments, id_map = finalize_fragments(fragments, doc_id, s3_fileid)
            span["facts"] = sum(len(f.extracted_facts) for f in fragments)
        flags_by_id = {
            id_map.get(fragment_id, fragment_id): extra
            for fragment_id, extra in flags_by_id.items()
        }

        # ----------------------------------------- 4. Сохранение (транзакция)
        with telemetry.measure("parse.file_hash"):
            file_hash = compute_file_hash(storage, located.uri)
        raw_uri = f"{settings.RAW_PARSE_S3_PREFIX.rstrip('/')}/{doc_id}.json"

        report = page_report(parse_result)
        warnings = build_warnings(parse_result, report["failed_pages"])
        document_status = "partial" if report["failed_pages"] else "completed"
        mime_type = filetypes.mime_for(located.file_type)
        normalized_hash = normalized_content_hash(fragments)

        with telemetry.measure("parse.save", doc_id=doc_id), session.begin():
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
                # Поля контракта, которые нужно отдать и при повторной
                # выдаче из БД, без нового разбора.
                "mime_type": mime_type,
                "document_status": document_status,
                "pages_reported": report["pages"],
                "processed_pages": report["processed_pages"],
                "failed_pages": report["failed_pages"],
                "warnings": warnings,
                "normalized_content_hash": normalized_hash,
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
                # Время каждой операции рядом с документом: по нему видно,
                # где именно он провёл время, без похода в логи воркера.
                "timings_ms": dict(timings),
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
            status=document_status,
            content_hash=f"sha256:{file_hash}" if file_hash else None,
            normalized_content_hash=normalized_hash,
            mime_type=mime_type,
            pages=report["pages"],
            processed_pages=report["processed_pages"],
            failed_pages=report["failed_pages"],
            document_metadata=doc_metadata,
            fragments=fragments,
            warnings=warnings,
        ).model_dump(mode="json")
        finalize(request_id, s3_fileid, result, dialog_id, event_id)

        if parse_result.degraded:
            logger.error(
                "Документ %s разобран с деградацией: %s",
                doc_id, "; ".join(parse_result.degraded),
            )
        logger.info("Документ %s обработан: %d фрагментов", doc_id, len(fragments))
        return {"status": "success", "doc_id": doc_id, "fragments": len(fragments)}

    except SoftTimeLimitExceeded as exc:
        logger.error("Мягкий таймаут на документе %s", doc_id)
        _mark_error(session, doc_id)
        finalize(request_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", exc,
            self.request.retries + 1,
        ), dialog_id, event_id)
        return {"status": "error", "doc_id": doc_id, "reason": "timeout"}

    except Exception as exc:  # noqa: BLE001
        logger.error("Обработка %s не удалась: %s", s3_fileid, exc, exc_info=True)
        _mark_error(session, doc_id)

        if self.request.retries < self.max_retries:
            countdown = 60 * (self.request.retries + 1)
            logger.info("Повтор %s через %d с", s3_fileid, countdown)
            raise self.retry(exc=exc, countdown=countdown)

        logger.critical("Документ %s окончательно не обработан", doc_id)
        finalize(request_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", exc,
            self.request.retries + 1,
        ), dialog_id, event_id)
        return {"status": "error", "doc_id": doc_id, "reason": str(exc)}

    finally:
        timings_scope.__exit__(None, None, None)
        session.close()


def _mark_error(session, doc_id: str) -> None:
    try:
        session.rollback()
        with session.begin():
            DocumentRepository(session).update_status(doc_id, "error")
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось выставить статус error для %s: %s", doc_id, exc)
