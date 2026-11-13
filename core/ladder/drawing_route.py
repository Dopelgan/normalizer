"""
Спор чертёжного разбора и сплошного распознавания на листе-картинке.

Зачем отдельный шаг. Лестница сравнивает уровни по числу символов, а на скане
чертежа это число у структурного разбора почти нулевое: MinerU возвращает лист
одной картинкой, и надписи, штамп, таблицы и весь текст листа читает модель
позже, в `DrawingProcessor`. На момент сравнения непрочитанный лист
соревнуется со сплошным распознаванием и проигрывает всегда: 0 символов
против 1692. В индекс уезжали строки Tesseract по чертежу, документ считался
разобранным фоллбэком, а вырезки видов моделью не читались вовсе.

Поэтому решение разнесено на два шага (флаг `LADDER_DRAWING_OVER_OCR`):

1. В лестнице (`deferred`): лист-картинка, который растровый детектор считает
   чертежом, не проигрывает распознаванию из-за числа символов. Лучший
   результат распознавания остаётся в запасе (`ParseResult.fallback_reserve`).
   Работает только вместе с `DRAWING_IMAGE_FULL_READ`: без него лист целиком
   не переписывается, и сравнивать с распознаванием было бы нечего.
2. После чтения листа моделью (`settle`): объём считается по тому, что
   чертёжная ветка действительно принесла, — текст листа, поля, таблицы.
   Если она не прочитала ничего или принесла меньше `LADDER_CONTENT_KEEP_RATIO`
   от распознавания, текст распознавания возвращается в документ, а документ
   уходит на проверку: бедный разбор не выдаётся за полный.

Объём считается осторожно. Поля и таблицы чертежа повторяют переписанный текст
листа, а вырезка вида повторяет лист, поэтому на страницу берётся одно лучшее
чтение, а не сумма всех: иначе один и тот же размер считался бы трижды.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from core import filetypes
from core.config import settings
from core.ladder.base import text_chars
from core.ladder.context import GENRE_DRAWING, SOURCE_RASTER, DocumentContext
from core.models.contract import Fragment
from core.models.parse_result import ParseResult

logger = logging.getLogger(__name__)

# Предупреждение документа: чертёжный разбор листа беднее распознавания.
WARNING_THIN = "DRAWING_ROUTE_THINNER_THAN_OCR"

_TITLE_BLOCK = "title_block"


@dataclass
class Volume:
    """Что принёс разбор листа — в тех единицах, в которых чертёж читают."""

    chars: int = 0             # символы без пробелов, лучшее чтение на страницу
    fields: int = 0            # поля чертежа: размеры, допуски, графы штампа
    tables: int = 0            # таблицы листа, прочитанные моделью
    table_rows: int = 0
    transcript_lines: int = 0  # строки переписанного текста листа

    @property
    def read(self) -> int:
        """Сколько единиц прочитала чертёжная ветка; ноль — лист не прочитан."""
        return self.fields + self.table_rows + self.transcript_lines

    def as_dict(self) -> Dict[str, int]:
        return {**asdict(self), "read": self.read}


def deferred(context: DocumentContext) -> bool:
    """
    Лист прочитает чертёжная ветка, и число символов у разбора с детекцией
    областей о потере текста не говорит. Только картинка: у скана PDF листов
    много, среди них бывают страницы текста, и там счёт символов — честная
    проверка сохранности.
    """
    # Без полного чтения листа-картинки модель читает только вырезки, текста
    # листа нет, и сверка почти всегда возвращала бы распознавание обратно.
    if not (settings.LADDER_DRAWING_OVER_OCR and settings.DRAWING_IMAGE_FULL_READ):
        return False
    if context.kind != filetypes.KIND_IMAGE:
        return False
    classification = context.classification
    return classification.source == SOURCE_RASTER and classification.genre == GENRE_DRAWING


def volume(fragments: List[Fragment]) -> Volume:
    """Объём разобранного: текст и таблицы суммой, чертёж — лучшим чтением листа."""
    total = Volume()
    best_drawing: Dict[int, int] = {}
    for fragment in fragments:
        content = fragment.content
        text = _chars(content.text)
        if fragment.type != "drawing":
            total.chars += text
            if content.table_data is not None:
                total.chars += _chars(content.table_data.headers) + _chars(content.table_data.rows)
            continue

        fields = content.structured_drawing_fields or []
        payload = content.structured_payload or {}
        # Таблица штампа уже разложена по полям: второй раз не считается.
        tables = [t for t in payload.get("tables") or [] if t.get("kind") != _TITLE_BLOCK]
        rows = [row for table in tables for row in table.get("rows") or []]
        total.fields += len(fields)
        total.tables += len(tables)
        total.table_rows += len(rows)
        total.transcript_lines += int(payload.get("transcript_line_count") or 0)

        structured = sum(_chars(f.value) + _chars(f.tolerance) for f in fields) + _chars(rows)
        page = fragment.position.page if fragment.position is not None else 0
        best_drawing[page] = max(best_drawing.get(page, 0), text, structured)
    total.chars += sum(best_drawing.values())
    return total


def settle(
    fragments: List[Fragment],
    parse_result: ParseResult,
    metadata: Dict[str, Any],
    normalizer,
) -> Tuple[List[Fragment], Dict[str, Dict[str, Any]]]:
    """
    Сверка после чтения листа моделью. Возвращает фрагменты документа и
    служебные пометки возвращённых фрагментов распознавания (по их id).

    Чертёжный разбор остаётся победителем, если прочитал хоть что-то и не
    потерял больше допустимого относительно распознавания. Иначе возвращается
    текст распознавания: добавляется к тому, что есть, ничего не заменяя, —
    итог не беднее того, что выдало бы распознавание само.
    """
    reserve = parse_result.fallback_reserve
    if reserve is None:
        return fragments, {}

    got = volume(fragments)
    reserve_chars = text_chars(reserve)
    won = got.read > 0 and got.chars >= reserve_chars * settings.LADDER_CONTENT_KEEP_RATIO
    outcome: Dict[str, Any] = {
        "won": won,
        "volume": got.as_dict(),
        "reserve_strategy": reserve.parser_name,
        "reserve_chars": reserve_chars,
    }
    parse_result.ladder["drawing_route"] = outcome
    parse_result.fallback_reserve = None
    if won:
        logger.info(
            "Лист прочитан чертёжной веткой: символов %d (полей %d, строк текста %d, "
            "строк таблиц %d) против %d у распознавания",
            got.chars, got.fields, got.transcript_lines, got.table_rows, reserve_chars,
        )
        return fragments, {}

    note = (
        f"чертёжный разбор беднее распознавания: {got.chars} символов "
        f"(прочитано единиц: {got.read}) против {reserve_chars} у {reserve.parser_name} — "
        "текст распознавания сохранён"
    )
    # Пометка деградации до нормализации запаса: с ней его фрагменты уходят
    # на проверку, как и при обычной победе фоллбэка.
    reserve.degraded = list(dict.fromkeys(list(reserve.degraded) + [note]))
    parse_result.degraded = list(dict.fromkeys(list(parse_result.degraded) + [note]))
    parse_result.ladder["fallback_won"] = True
    parse_result.ladder["degraded"] = list(parse_result.degraded)
    logger.warning("Разбор %s деградировал: %s", metadata.get("source_path") or "?", note)

    restored, flags = normalizer.normalize_with_flags(reserve, metadata)
    known = {fragment.fragment_id for fragment in fragments}
    extra_flags: Dict[str, Dict[str, Any]] = {}
    for fragment, flag in zip(restored, flags):
        if fragment.fragment_id in known:
            # Тот же текст на той же странице уже есть в документе.
            continue
        fragment.review_required = True
        fragments.append(fragment)
        extra_flags[fragment.fragment_id] = flag
    outcome["restored_fragments"] = len(extra_flags)
    return fragments, extra_flags


def lost(parse_result: ParseResult) -> bool:
    """Чертёжная ветка получила лист и проиграла сверку: документ не полон."""
    return (parse_result.ladder or {}).get("drawing_route", {}).get("won") is False


def _chars(value: Any) -> int:
    """Символы без пробелов в значении любой вложенности (строка, строки таблицы)."""
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_chars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_chars(item) for item in value)
    return len("".join(str(value).split()))
