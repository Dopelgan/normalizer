"""
Доведение фрагментов до контракта: стабильные `fragment_id`, происхождение
и извлечённые факты.

Шаг выполняется последним — после нормализации и после обработки чертежей.
Раньше идентификатор выдавался нормализатором по порядковому номеру, и
обработчик чертежей, вставляя разбор листа в начало, сдвигал нумерацию:
при повторном разборе того же файла RAG получал другие `fragment_id`, а
значит — дубли вместо обновления.

Идентификатор считается от содержимого фрагмента, а не от его места в
списке: добавление фрагмента на первой странице не меняет идентификаторы
на десятой, а повторный разбор того же файла даёт те же значения.
"""

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.facts.extractor import extract_facts
from core.models.contract import Fragment

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")
_DIGEST_LENGTH = 12


def _content_signature(fragment: Fragment) -> str:
    """Слепок содержимого: то, что делает фрагмент этим фрагментом."""
    content = fragment.content
    if fragment.type == "table" and content.table_data is not None:
        return json.dumps(
            content.table_data.model_dump(), ensure_ascii=False, sort_keys=True,
            default=str,
        )
    if fragment.type == "drawing":
        fields = content.structured_drawing_fields or []
        parts = [f"{f.category}={f.value}" for f in fields]
        return f"{content.image_ref or ''}|" + "|".join(parts)
    if fragment.type == "image":
        return content.image_ref or ""
    if content.text:
        return _WHITESPACE.sub(" ", content.text).strip()
    return json.dumps(content.model_dump(), ensure_ascii=False, sort_keys=True, default=str)


def stable_fragment_id(doc_id: str, fragment: Fragment) -> str:
    """
    Идентификатор фрагмента, одинаковый при повторном разборе того же файла.

    В основе — тип, страница, лист и содержимое. Координаты в основу не
    берём: bbox дрожит между прогонами распознавания, а текст — нет.
    """
    position = fragment.position
    page = position.page if position else 0
    sheet = (position.sheet if position else None) or ""
    payload = f"{fragment.type}|{page}|{sheet}|{_content_signature(fragment)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
    return f"{doc_id or 'doc'}:p{page:03d}:{fragment.type}:{digest}"


def finalize_fragments(
    fragments: List[Fragment],
    doc_id: str,
    s3_fileid: str = "",
    extract: bool = True,
) -> Tuple[List[Fragment], Dict[str, str]]:
    """
    Проставляет стабильные идентификаторы, дополняет provenance и извлекает
    факты. Возвращает фрагменты и карту «прежний id -> новый id»: по ней
    пересобираются служебные пометки, собранные нормализатором.
    """
    id_map: Dict[str, str] = {}
    seen: Dict[str, int] = {}

    for order, fragment in enumerate(fragments, start=1):
        previous = fragment.fragment_id
        fragment_id = stable_fragment_id(doc_id, fragment)
        # Два одинаковых по содержимому фрагмента на одной странице —
        # редкость (пустая ячейка, повтор шапки), но идентификатор обязан
        # остаться уникальным, иначе один затрёт другой в БД.
        count = seen.get(fragment_id, 0) + 1
        seen[fragment_id] = count
        if count > 1:
            fragment_id = f"{fragment_id}-{count}"

        fragment.fragment_id = fragment_id
        if previous:
            id_map[previous] = fragment_id

        if fragment.position is not None:
            fragment.position.order = order
            fragment.provenance.source_page = fragment.position.page
        if s3_fileid:
            fragment.provenance.source_file_id = s3_fileid

        if extract:
            try:
                fragment.extracted_facts = extract_facts(fragment)
            except Exception as exc:  # noqa: BLE001 — факт не важнее фрагмента
                logger.warning(
                    "Извлечение фактов из %s не удалось: %s", fragment_id, exc
                )
                fragment.extracted_facts = []

    return fragments, id_map


def normalized_content_hash(fragments: List[Fragment]) -> Optional[str]:
    """
    `sha256:` нормализованного текста документа.

    Считается по содержимому фрагментов в их порядке: отвечает на вопрос
    «изменился ли смысл», когда байты файла изменились (пересохранение,
    другой генератор PDF), а текст остался прежним.
    """
    if not fragments:
        return None
    digest = hashlib.sha256()
    for fragment in fragments:
        digest.update(_content_signature(fragment).encode("utf-8"))
        digest.update(b"\x1e")
    return f"sha256:{digest.hexdigest()}"


def facts_of(fragments: List[Fragment]) -> List[Dict[str, Any]]:
    """Все факты документа одним списком — для логов и отладки."""
    return [
        fact.model_dump(mode="json")
        for fragment in fragments
        for fact in fragment.extracted_facts
    ]
