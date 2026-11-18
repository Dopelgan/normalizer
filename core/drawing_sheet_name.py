"""
Наименование листа 1 — на листы 2+ того же чертежа (`DRAWING_SHEET_NAME_CARRY`).

Зачем. У многолистового чертежа полный штамп (форма 1 ГОСТ 2.104) только на
листе 1; на листах 2+ — сокращённый (форма 2а): обозначение и номер листа,
наименования нет. Куски записи листа 2 уходят в индекс без названия детали,
и по вопросу «что сказано про корпус клапана» лист 2 не находится.

Что делает. Идёт по чертёжным фрагментам в порядке документа. Первый лист с
обозначением и наименованием становится источником для этого обозначения;
более поздний лист с тем же обозначением и без наименования получает его
наименование — в графу `structured_payload["title_block"]["name"]` и полем
категории `title_block`, как поле листа 1. Перенос помечается:
`structured_payload["title_block_sources"] = {"name": "sheet1"}` —
наименование не прочитано на этом листе, а взято с листа 1.

Чего не делает. Не переносит между разными обозначениями (другой чертёж в
том же файле — другая деталь), не заменяет наименование, прочитанное на
самом листе, и не трогает остальные графы.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from core.models.contract import DrawingField, Fragment
from core.providers.title_block import label

# Ключ в payload: откуда взята графа, если не с этого листа.
SOURCES_KEY = "title_block_sources"
# Пометка источника: графа взята с листа 1 того же обозначения.
CARRIED_FROM = "sheet1"

_SPACES = re.compile(r"\s+")


def _key(designation: object) -> str:
    """Обозначение для сравнения: без пробелов и регистра."""
    return _SPACES.sub("", str(designation or "")).casefold()


def _name_field(fragment: Fragment, name: str) -> DrawingField:
    """Поле наименования листа-источника, или новое с тем же значением."""
    prefix = label("name") + ":"
    for field in fragment.content.structured_drawing_fields or []:
        if field.category == "title_block" and field.value.startswith(prefix):
            return field.model_copy(deep=True)
    return DrawingField(category="title_block", value=f"{prefix} {name}",
                        confidence=fragment.confidence)


def apply(fragments: List[Fragment]) -> List[Fragment]:
    """Переносит наименование на листы без него; меняет фрагменты на месте."""
    sources: Dict[str, Tuple[str, Fragment]] = {}
    for fragment in fragments:
        if fragment.type != "drawing":
            continue
        payload = fragment.content.structured_payload or {}
        block: Optional[dict] = payload.get("title_block")
        if not isinstance(block, dict):
            continue
        key = _key(block.get("designation"))
        if not key:
            continue
        name = str(block.get("name") or "").strip()
        if name:
            sources.setdefault(key, (name, fragment))
            continue
        if key not in sources:
            continue
        name, source = sources[key]
        block["name"] = name
        payload.setdefault(SOURCES_KEY, {})["name"] = CARRIED_FROM
        fields = list(fragment.content.structured_drawing_fields or [])
        fields.append(_name_field(source, name))
        fragment.content.structured_drawing_fields = fields
        fragment.content.structured_payload = payload
    return fragments
