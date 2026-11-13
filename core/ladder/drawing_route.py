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
   от распознавания, текст распознавания возвращается в документ: бедный
   разбор не выдаётся за полный.

С `LADDER_STRUCTURE_OVER_LENGTH` победителя определяет состав прочитанного, а
не объём (`core/ladder/structure.py`), и ни один признак качества не зовёт
человека: лист, прочитанный плохо, нужно прочитать заново, а решают люди не
это (куда документ относится, как называется, какая редакция старше). Причины
повторного чтения собирает `reprocess_candidate`.

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
from core.ladder.structure import drawing_structure
from core.models.contract import Fragment
from core.models.parse_result import ParseResult

logger = logging.getLogger(__name__)

# Предупреждение документа: чертёжный разбор листа беднее распознавания.
WARNING_THIN = "DRAWING_ROUTE_THINNER_THAN_OCR"

# Разбор уложился в допуск `LADDER_CONTENT_KEEP_RATIO`, но символов в нём
# всё равно меньше, чем выдало распознавание. Текст распознавания при этом
# не возвращается — иначе допуск не имел бы смысла, — поэтому недостача
# обязана быть видна: без неё лист, прочитанный наполовину, уходит как
# полный и молча.
WARNING_THIN_KEPT = "DRAWING_ROUTE_THINNER_KEPT"

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
    if classification.source != SOURCE_RASTER:
        return False
    if classification.genre == GENRE_DRAWING:
        return True
    # Жанр листа ставит растровый детектор по числу длинных линий и полос
    # строк, и плотный лист он называет страницей текста: у СМ1638 при
    # рамке формата нашлось 2 длинных линии вместо трёх и 7 полос строк.
    # Читает такой лист всё равно чертёжная ветка — по той же рамке
    # (`DrawingProcessor._full_read_image`), — значит и в лестнице его
    # число символов о потере текста не говорит. Условие здесь то же, что
    # у ветки: иначе два шага одного конвейера считают лист разным.
    return settings.LADDER_STRUCTURE_OVER_LENGTH and has_sheet_frame(classification)


def has_sheet_frame(classification) -> bool:
    """Растровый детектор видит на листе рамку формата."""
    signals = classification.signals or {}
    return bool(signals.get("frame") or signals.get("thin_line_frame"))


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
    found = drawing_structure(fragments)
    if settings.LADDER_STRUCTURE_OVER_LENGTH:
        # Решает состав прочитанного, а не объём. Плоский текст полей листа
        # не возвращает вовсе, поэтому чертёжный разбор с полями старше
        # распознавания при любом числе символов.
        won = found.structured
    else:
        won = got.read > 0 and got.chars >= reserve_chars * settings.LADDER_CONTENT_KEEP_RATIO
    outcome: Dict[str, Any] = {
        "won": won,
        "volume": got.as_dict(),
        "structure": found.as_dict(),
        "reserve_strategy": reserve.parser_name,
        "reserve_chars": reserve_chars,
    }
    parse_result.ladder["drawing_route"] = outcome
    parse_result.fallback_reserve = None
    if won:
        logger.info(
            "Лист прочитан чертёжной веткой: символов %d (полей %d, строк текста %d, "
            "строк таблиц %d) против %d у распознавания; состав: штамп %s, "
            "обозначение %s, величин %d, надписей видов %d",
            got.chars, got.fields, got.transcript_lines, got.table_rows, reserve_chars,
            found.title_block, found.designation, found.dimensions, found.view_titles,
        )
        if settings.LADDER_STRUCTURE_OVER_LENGTH and not found.complete:
            # Поля есть, а штампа и обозначения на листе прочитать не
            # удалось. Человеку такой лист показывать незачем: решать тут
            # нечего — либо штампа на листе нет (обрезанный лист, вырезка,
            # схема), либо модель его не нашла, и тогда нужен повторный
            # разбор, а не чужие глаза. Разбор принимается как есть, а
            # недочитанный штамп остаётся измеримым в `drawing_route`:
            # по нему видно, сколько таких листов в установке, и это повод
            # править чтение штампа, а не звать человека к каждому листу.
            outcome["thin_structure"] = True
            logger.info(
                "Лист %s прочитан без штампа и обозначения: полей %d, величин %d, "
                "надписей видов %d — разбор принят",
                metadata.get("source_path") or "?", found.fields,
                found.dimensions, found.view_titles,
            )
            return fragments, {}
        if got.chars < reserve_chars:
            # Недостача в допуске: текст распознавания не возвращается, но
            # документ уходит на проверку с этой строкой в предупреждениях.
            note = (
                f"чертёжный разбор короче распознавания: {got.chars} символов "
                f"против {reserve_chars} у {reserve.parser_name} — "
                "оставлен разбор чертежа"
            )
            outcome["thinner_kept"] = True
            outcome["note"] = note
            parse_result.degraded = list(dict.fromkeys(
                list(parse_result.degraded) + [note]
            ))
            logger.warning(
                "Лист %s прочитан короче распознавания: %s",
                metadata.get("source_path") or "?", note,
            )
        return fragments, {}

    if settings.LADDER_STRUCTURE_OVER_LENGTH:
        note = (
            "чертёжный разбор без структуры: полей "
            f"{found.fields}, штамп {'есть' if found.title_block else 'нет'}, "
            f"обозначение {'есть' if found.designation else 'нет'}, величин "
            f"{found.dimensions} — текст распознавания "
            f"({reserve_chars} символов, {reserve.parser_name}) сохранён"
        )
    else:
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


def thinner_kept(parse_result: ParseResult) -> bool:
    """Разбор оставлен победителем, но символов в нём меньше, чем у распознавания."""
    route = (parse_result.ladder or {}).get("drawing_route", {})
    return bool(route.get("thinner_kept"))


# Признаки качества, которые лечатся повторным чтением листа, а не человеком:
# модель не прочитала лист, прочитала его короче распознавания, не нашла
# штампа, не успела в бюджет времени. Все четыре — про то, как прошло чтение,
# и ни один не требует решения, которое может принять только человек.
#: Причина повторного чтения, а не предупреждение документа: штамп и
#: обозначение на листе прочитать не удалось.
THIN_STRUCTURE = "DRAWING_ROUTE_THIN_STRUCTURE"

REPROCESS_WARNINGS = (
    WARNING_THIN,
    WARNING_THIN_KEPT,
    "DRAWING_NOT_RECOGNIZED",
    "DRAWING_SKIPPED_NO_TIME",
    "DRAWING_SHEETS_SKIPPED",
)


def reprocess_candidate(warnings: List[str], parse_result: ParseResult) -> List[str]:
    """
    Причины, по которым документ стоит прочитать заново.

    Пусто — перечитывать нечего. Непустое значение уезжает в
    `documents.extra_data.quality_reprocess`, и по нему документы отбираются
    на повторное чтение запросом к БД, а не руками: ни одна из причин не
    требует человека, и держать их в очереди проверки незачем.
    """
    found = [warning for warning in dict.fromkeys(warnings)
             if warning.startswith(REPROCESS_WARNINGS)]
    if thin_structure(parse_result):
        # Предупреждением документа недочитанный штамп не становится:
        # предупреждение уехало бы потребителю и вернулось бы человеком.
        found.append(THIN_STRUCTURE)
    return found


def thin_structure(parse_result: ParseResult) -> bool:
    """
    Разбор чертёжный, но штампа и обозначения на листе прочитать не удалось.

    Наблюдение, а не повод звать человека: смотреть ему тут нечего, решать
    нечего, а недочитанный штамп лечится повторным чтением листа. Признак
    нужен, чтобы такие листы можно было посчитать в установке.
    """
    route = (parse_result.ladder or {}).get("drawing_route", {})
    return bool(route.get("thin_structure"))


def _chars(value: Any) -> int:
    """Символы без пробелов в значении любой вложенности (строка, строки таблицы)."""
    if value is None:
        return 0
    if isinstance(value, dict):
        return sum(_chars(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_chars(item) for item in value)
    return len("".join(str(value).split()))
