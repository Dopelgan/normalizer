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
from core import deadline
from core.config import settings
from core.processing_contract import PROCESSING_PROFILE_VERSION
from core.drawing_quality import document_quality
from core.contract_finalize import finalize_fragments, normalized_content_hash
from core.db.session import SessionLocal
from core.drawing_processor import DrawingProcessor
from core.ladder import drawing_route
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
    mark_handed_off,
    mark_started,
    parse_stale_after_seconds,
    put_document_result,
)
from core.result_publisher import publish_result

logger = logging.getLogger(__name__)

# Пауза между проверками, пока тот же файл разбирает другая задача.
PEER_WAIT_SECONDS = 60
PAGE_COVERAGE_VERSION = 1


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
        processing_profile_version=PROCESSING_PROFILE_VERSION,
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
    references = set(parse_result.pages()) | set(parse_result.failed_pages) | set(parse_result.confirmed_blank_pages)
    failed = {page for page in parse_result.failed_pages if page > 0}
    seen = {page for page in parse_result.pages() if page > 0}
    blank = {page for page in parse_result.confirmed_blank_pages if page > 0}
    total = parse_result.source_page_count or 0
    expected = set(range(1, total + 1)) if total > 0 else set()
    processed = (seen | blank) - failed
    unknown = sorted(expected - processed - failed)
    warnings = [f"PAGE_COVERAGE_UNKNOWN: page {page}" for page in unknown]
    mismatch = any(page <= 0 or (total > 0 and page > total) for page in references)
    if total <= 0:
        warnings.append("PAGE_COUNT_UNKNOWN")
    if mismatch:
        warnings.append("PAGE_COUNT_MISMATCH")
    return {"pages": total if total > 0 and not mismatch else None,
            "processed_pages": sorted(processed), "failed_pages": sorted(failed),
            "unknown_pages": unknown, "coverage_unknown": bool(unknown) or total <= 0 or mismatch,
            "coverage_warnings": warnings}


def build_warnings(parse_result: ParseResult, failed_pages: List[int]) -> List[str]:
    """Деградации разбора и непрочитанные страницы — одним списком."""
    warnings = list(parse_result.degraded)
    warnings += [f"PAGE_NOT_PARSED: page {page}" for page in failed_pages]
    warnings += page_report(parse_result)["coverage_warnings"]
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
        pages=(extra.get("pages_reported") if extra.get("page_coverage_version") == PAGE_COVERAGE_VERSION
               else extra.get("pages_reported") or extra.get("parsed_pages_count") or None),
        processed_pages=list(extra.get("processed_pages") or []),
        failed_pages=list(extra.get("failed_pages") or []),
        unknown_pages=list(extra.get("unknown_pages") or []),
        document_metadata=document_to_metadata(document),
        fragments=fragments,
        warnings=list(extra.get("warnings") or []),
        processing_profile_version=extra.get("processing_profile_version"),
        review_required=bool(extra.get("review_required", False)),
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
    soft_time_limit=settings.PARSE_TASK_SOFT_TIME_LIMIT,
    time_limit=settings.PARSE_TASK_TIME_LIMIT,
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
    mark_started(request_id, s3_fileid)
    doc_id = generate_doc_id(s3_fileid)
    storage = StorageProviderFactory.default()
    session = SessionLocal()
    located: Optional[LocatedFile] = None
    # Документ в БД наш (мы перевели его в pending). Чужой документ —
    # его разбирает другая задача — ошибкой не помечаем.
    owns_document = False
    # Область сбора таймингов: все замеры внутри складываются сюда, и
    # словарь уезжает в extra_data документа. Без него на вопрос «почему
    # этот файл обрабатывался семь минут» отвечать нечем.
    timings_scope = telemetry.collect()
    timings = timings_scope.__enter__()
    # Один срок на все уровни лестницы: запас оставлен на сохранение
    # результата и запись ошибки до мягкого лимита Celery.
    deadline_scope = deadline.scope(
        settings.PARSE_TASK_SOFT_TIME_LIMIT - settings.PARSE_DEADLINE_RESERVE
    )
    deadline_scope.__enter__()

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

            if (document is not None and document.status == "indexed" and operation != "update"
                    and (document.extra_data or {}).get("processing_profile_version") == PROCESSING_PROFILE_VERSION
                    and (document.extra_data or {}).get("page_coverage_version") == PAGE_COVERAGE_VERSION):
                logger.info("Документ %s уже проиндексирован, отдаём из БД", doc_id)
                fragments = [row_to_fragment(r) for r in ChunkRepository(session).get_by_doc_id(doc_id)]
                cached = build_document_result(
                    s3_fileid, document, fragments, dict(document.extra_data or {})
                )
                finalize(request_id, s3_fileid, cached, dialog_id, event_id)
                return {"status": "already_processed", "doc_id": doc_id}

            if document is not None and document.status == "pending":
                # Порог — жёсткий лимит разбора с запасом, а не таймаут
                # синхронного ответа (900 с): иначе живой часовой разбор
                # считается зависшим и запускается второй раз.
                if doc_repo.is_stale(document, parse_stale_after_seconds()):
                    logger.warning("Документ %s завис в pending, перезапускаем", doc_id)
                else:
                    # Файл сейчас разбирает другая задача (повторная доставка
                    # брокером или другой запрос с тем же файлом). Второй
                    # разбор — двойной OCR на общем GPU. Ждём первый и отдаём
                    # его результат из БД; если он умрёт, документ станет
                    # «зависшим», и следующая попытка начнёт заново.
                    logger.warning(
                        "Документ %s уже разбирается другой задачей — ждём %d с",
                        doc_id, PEER_WAIT_SECONDS,
                    )
                    # Новая задача, а не self.retry: ожидание не должно
                    # тратить бюджет повторов на настоящие ошибки.
                    process_document.apply_async(
                        kwargs={
                            "request_id": request_id, "s3_fileid": s3_fileid,
                            "operation": operation, "dialog_id": dialog_id,
                            "event_id": event_id,
                        },
                        countdown=PEER_WAIT_SECONDS,
                        queue=(self.request.delivery_info or {}).get("routing_key")
                        or settings.CELERY_ML_QUEUE,
                    )
                    mark_handed_off(request_id, s3_fileid)
                    return {"status": "waiting_for_peer", "doc_id": doc_id}

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

        # Транзакция закрылась без ошибки — документ теперь наш.
        owns_document = True

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
            # Лист, отданный чертёжной ветке авансом (LADDER_DRAWING_OVER_OCR):
            # теперь он прочитан, и объём сверяется с распознаванием заново.
            fragments, restored_flags = drawing_route.settle(
                fragments, parse_result, metadata, normalizer
            )
            flags_by_id.update(restored_flags)

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
        review_required, drawing_warnings = document_quality(fragments)
        if (metadata["classification"].get("genre") == "drawing"
                and not any(f.type == "drawing" for f in fragments)):
            review_required = True
            drawing_warnings.append("DRAWING_NOT_RECOGNIZED")
        if drawing_route.lost(parse_result):
            # Бедный разбор листа не выдаётся за полный: документ на проверку.
            review_required = True
            drawing_warnings.append(drawing_route.WARNING_THIN)
        elif drawing_route.thinner_kept(parse_result):
            # Недостача в пределах допуска: разбор оставлен, но лист всё
            # равно прочитан короче распознавания — тоже на проверку.
            review_required = True
            drawing_warnings.append(drawing_route.WARNING_THIN_KEPT)
        warnings = list(dict.fromkeys(warnings + drawing_warnings))
        document_status = "partial" if report["failed_pages"] or report["coverage_unknown"] or review_required else "completed"
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
                "unknown_pages": report["unknown_pages"],
                "page_coverage_version": PAGE_COVERAGE_VERSION,
                "warnings": warnings,
                "processing_profile_version": PROCESSING_PROFILE_VERSION,
                "review_required": review_required,
                "normalized_content_hash": normalized_hash,
                "parsed_pages_count": parse_result.page_count or len(parse_result.pages()),
                "total_fragments": len(fragments),
                "parser": parse_result.parser_name,
                "source_kind": parse_result.source_kind,
                "is_fallback": parse_result.is_fallback,
                "needs_review": sum(1 for f in fragments if f.review_required),
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
                # Итог сверки чертёжного разбора с распознаванием — только
                # у листов, где она была (LADDER_DRAWING_OVER_OCR).
                **({"drawing_route": ladder["drawing_route"]}
                   if ladder.get("drawing_route") else {}),
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
            unknown_pages=report["unknown_pages"],
            document_metadata=doc_metadata,
            fragments=fragments,
            warnings=warnings,
            processing_profile_version=PROCESSING_PROFILE_VERSION,
            review_required=review_required,
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
        if owns_document:
            _mark_error(session, doc_id)
        finalize(request_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", exc,
            self.request.retries + 1,
        ), dialog_id, event_id)
        return {"status": "error", "doc_id": doc_id, "reason": "timeout"}

    except Exception as exc:  # noqa: BLE001
        logger.error("Обработка %s не удалась: %s", s3_fileid, exc, exc_info=True)
        if owns_document:
            _mark_error(session, doc_id)

        if self.request.retries < self.max_retries:
            countdown = 60 * (self.request.retries + 1)
            logger.info("Повтор %s через %d с", s3_fileid, countdown)
            # Пока повтор ждёт в очереди, файл не «пропавший».
            mark_handed_off(request_id, s3_fileid)
            raise self.retry(exc=exc, countdown=countdown)

        logger.critical("Документ %s окончательно не обработан", doc_id)
        finalize(request_id, s3_fileid, build_error_document(
            s3_fileid, doc_id, located.uri if located else "", exc,
            self.request.retries + 1,
        ), dialog_id, event_id)
        return {"status": "error", "doc_id": doc_id, "reason": str(exc)}

    finally:
        deadline_scope.__exit__(None, None, None)
        timings_scope.__exit__(None, None, None)
        session.close()


def _mark_error(session, doc_id: str) -> None:
    try:
        session.rollback()
        with session.begin():
            DocumentRepository(session).update_status(doc_id, "error")
    except Exception as exc:  # noqa: BLE001
        logger.error("Не удалось выставить статус error для %s: %s", doc_id, exc)
