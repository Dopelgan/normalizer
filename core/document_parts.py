"""
Ярлык и его значения — один фрагмент, и он сам говорит, где у него заголовок.

Зачем это нужно. В обычном документе — не в чертеже — подпись и то, что
под ней стоит, приходят от разбора **разными блоками**: «Знак поверки:» —
один абзац, перечень под ним — следующий. Нормализатор делает из блока
фрагмент (`core/normalizer/normalizer.py`, ветка текста), поэтому
«Знак поверки:» становится отдельным фрагментом, а его значения —
соседним. Замер на 92 реальных разборах: подписей, стоящих отдельным
фрагментом перед своим текстом, — 650; подпись, у которой значения лежат
в том же фрагменте, нашлась **одна**. То есть это не редкость, а обычное
устройство разбора.

Чем это кончается у потребителя. Чанкер RAG режет по фрагментам, поэтому
такой фрагмент становится чанком из одной строки «Знак поверки:» — а
значения уезжают в следующий чанк. На том же замере таких чанков
«заголовок без значений» 226; правила чанкера, которым разметка не нужна
(не заканчивать окно на строке-ярлыке и уводить мелкий ярлык-фрагмент к
следующей группе), снимают 106 и оставляют 120, и 118 из этих 120 — ровно
этот случай: ярлык стоит отдельным фрагментом, и подвинуть границу внутри
фрагмента невозможно, потому что резать там нечего.

Что делает модуль. Две вещи, обе — на одном флаге:

1. **Соединяет** текстовый фрагмент, который кончается строкой-ярлыком, с
   идущим сразу за ним текстовым фрагментом той же страницы и того же
   раздела. Получается один фрагмент: «Знак поверки:» и перечень под ним.
   Символы не пропадают: текст склеивается через один перевод строки, из
   обоих текстов не удаляется ничего.
2. **Размечает** его частями (`content.structured_payload["record_parts"]`)
   по тому же контракту, что и запись чертежа: части подряд, без зазоров,
   от `0` до `len(text)`, `breaks` — начала строк строго внутри своей
   части. Чанкер (`super_rag/ingest/normalize/service.py::_record_parts`)
   режет только по `start` части и по её `breaks`, а на продолжении
   повторяет `title` — поэтому даже у большого соединённого фрагмента
   значения не придут без той подписи, которой они являются.

Чего модуль не делает — и это главное ограничение. Он не придумывает
структуру там, где текст о ней не говорит:

* подпись, за которой идёт **не текст**, а таблица, картинка, формула или
  чертёж (на замере 15 и 42 случая), не соединяется ни с чем: там
  соединять нечего, типы фрагментов разные, и догадка о принадлежности
  была бы именно догадкой;
* подпись, чьи значения — **на другой странице** (5 случаев), остаётся как
  была: перенос через страницу разбор не подтверждает;
* `total_row:` в конце нарисованной таблицы Parser исправить не может:
  эту строку рисует адаптер RAG из `table_data`, в тексте фрагмента её
  нет;
* длинная фраза, случайно кончающаяся двоеточием, под ярлык не попадает:
  строка ограничена по длине так же, как в чанкере.

Флаг `DOCUMENT_PARTS_MARKUP`, по умолчанию выключен: при выключенном флаге
ни один фрагмент не меняется ни на байт. Новых имён предупреждений модуль
не вводит и `review_required` не ставит — к человеку он ничего не
отправляет.

Связи. Вызывается из `ingest/tasks.py::process_document` после ветки
чертежей и **до** `core.contract_finalize.finalize_fragments`: поэтому
идентификаторы, порядок и факты считаются уже по соединённому фрагменту, и
второго способа выдавать `fragment_id` не появляется.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.config import settings
from core.models.contract import Fragment

# Ключ в payload: разметка частей фрагмента для чанкера. Имя то же, что у
# записи чертежа (`core/drawing_record.py`), потому что контракт один.
PARTS_KEY = "record_parts"

# Строка, которая что-то называет и останавливается: «Знак поверки:»,
# «Имеется:», «Диаметры —». Образец тот же, что у чанкера
# (`_LABEL_ONLY_LINE`): длина строки ограничена, иначе под ярлык попала бы
# любая фраза, случайно кончающаяся двоеточием.
_LABEL_LINE = re.compile(r"^[ \t]*(?:[-–—*•]\s*)?\S[^\n]{0,160}?[ \t]*[:：—–][ \t]*$")
# Номер пункта или перечня, за которым ничего нет: «4.», «5)», «4.1.».
_NUMBER_LINE = re.compile(r"^[ \t]*\d{1,3}(?:\.\d{1,3})*[.)][ \t]*$")
# Заголовок раздела (`DOCUMENT_HEADING_WITH_BODY`): номер раздела и имя
# либо слово-название с номером. Имя обязано быть — одинокий номер уже
# разобран `_NUMBER_LINE`.
_HEADING_LINE = re.compile(
    r"^(?:\d{1,3}(?:\.\d{1,3})*\.?\s+\S[^\n]*"
    r"|[А-ЯЁ]\.\d{1,2}\s+\S[^\n]*"
    r"|(?:(?:Таблица|Рисунок|Приложение|Раздел|Глава)\b|(?:Табл|Рис)\.)[^\n]*)$",
    re.IGNORECASE)
# Заголовок не бывает длинным: длиннее — это фраза, начатая с номера.
_HEADING_MAX_CHARS = 90

# Чем соединяются тексты двух фрагментов. Один перевод строки: он делает
# из подписи и значений те же строки, какими они стоят в документе, и
# ничего не удаляет.
_JOIN = "\n"


def enabled() -> bool:
    return bool(getattr(settings, "DOCUMENT_PARTS_MARKUP", False))


def is_label_line(line: str) -> bool:
    """Строка называет что-то и обрывается, не сказав значения."""
    if _LABEL_LINE.match(line) or _NUMBER_LINE.match(line):
        return True
    return _heading_enabled() and is_heading_line(line)


def _heading_enabled() -> bool:
    return bool(getattr(settings, "DOCUMENT_HEADING_WITH_BODY", False))


def is_heading_line(line: str) -> bool:
    """
    Строка — заголовок раздела (`DOCUMENT_HEADING_WITH_BODY`).

    Зачем отдельно от ярлыка. Ярлык узнаётся по тому, что обрывается на
    двоеточии или тире, и этого признака у заголовка нет: «1.1 НАЗНАЧЕНИЕ
    И КЛАССИФИКАЦИЯ», «2.5 МОНТАЖ», «9. Производственная деятельность»,
    «Таблица 5 — Нормы расхода лакокрасочных материалов», «Приложение В»
    кончаются ничем. Замер по 50 документам прода: таких чанков —
    заголовок без своего текста — 262 из 1195 коротких, и
    «4 ГАРАНТИЙНЫЕ ОБЯЗАТЕЛЬСТВА 2 ОСНОВНЫЕ ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ»
    повторяется 24 раза.

    Признак заголовка: строка начинается номером раздела или словом
    «Таблица»/«Рисунок»/«Приложение», коротка и не кончается точкой —
    точка в конце означает фразу, а не заголовок. Фраза, случайно
    начинающаяся с номера («4. Общие указания» — заголовок, а «4. При
    монтаже убедиться, что кабель обесточен.» — пункт требований),
    отделяется ровно этим: пункт требований кончается точкой.
    """
    body = " ".join((line or "").split())
    if not body or len(body) > _HEADING_MAX_CHARS or body.endswith((".", ";", ",")):
        return False
    return bool(_HEADING_LINE.match(body))


def label_tail(text: Optional[str]) -> Optional[int]:
    """
    Смещение начала последней строки `text`, если эта строка — ярлык.

    `None` — текст не кончается ярлыком, и соединять нечего.
    """
    if not text:
        return None
    stripped = text.rstrip()
    if not stripped:
        return None
    cut = stripped.rfind("\n")
    line = stripped[cut + 1:]
    if not is_label_line(line):
        return None
    return cut + 1


def apply(fragments: List[Fragment]) -> List[Fragment]:
    """
    Соединяет ярлык с его значениями и размечает получившийся фрагмент.

    При выключенном флаге возвращает тот же список без единого изменения.
    """
    if not enabled() or not fragments:
        return fragments

    result: List[Fragment] = []
    # Смещения подписей хранятся рядом, а не на фрагменте: фрагмент —
    # модель контракта, и служебное поле на нём уехало бы наружу.
    label_offsets: Dict[int, List[int]] = {}
    # Последний текстовый фрагмент, к которому ещё можно присоединить
    # значения. Фрагменты без текста (вырезка-картинка пустого поля
    # формы — на замере их между подписью и её значениями большинство)
    # candidate не сбрасывают: в индекс они не попадают вообще
    # (`super_rag/parser/adapter.py` пропускает фрагмент без текста),
    # поэтому между подписью и значениями у потребителя ничего не стоит.
    candidate: Optional[Fragment] = None
    for fragment in fragments:
        if candidate is not None and _joinable(candidate, fragment):
            _join_into(candidate, fragment, label_offsets)
            continue
        result.append(fragment)
        if _indexed_text(fragment):
            candidate = fragment if fragment.type == "text" else None
    # Разметка ставится только тем фрагментам, которые действительно
    # соединены: у остальных резать между подписью и значениями нечего, и
    # пустая разметка только мешала бы чанкеру.
    for fragment in result:
        payload = fragment.content.structured_payload or {}
        if payload.get(PARTS_KEY):
            continue
        marks = label_offsets.get(id(fragment))
        if not marks:
            continue
        parts = part_index(fragment.content.text or "", marks)
        if not parts:
            continue
        payload = dict(payload)
        payload[PARTS_KEY] = parts
        fragment.content.structured_payload = payload
    return result


def _joinable(first: Fragment, second: Fragment) -> bool:
    """
    Второй фрагмент — значения подписи, которой кончается первый.

    Требуется всё сразу: оба — текст, у обоих есть текст, первый кончается
    строкой-ярлыком, второй начинается не пустой строкой, и оба стоят на
    одной странице одного листа в одном разделе. Любое несовпадение —
    отказ: подпись остаётся там, где её поставил разбор.
    """
    if first.type != "text" or second.type != "text":
        return False
    head, tail = first.content.text or "", second.content.text or ""
    if not head.strip() or not tail.strip():
        return False
    if label_tail(head) is None:
        return False
    first_position, second_position = first.position, second.position
    if first_position is None or second_position is None:
        return False
    if first_position.page != second_position.page:
        return False
    if (first_position.sheet or None) != (second_position.sheet or None):
        return False
    # Через границу двух **названных** разделов подпись не переезжает: это
    # уже не «подпись и её значения», а другой раздел. Если у одного из
    # фрагментов названия раздела нет, разбор о границе ничего не сказал.
    if (first.section_title and second.section_title
            and first.section_title != second.section_title):
        return False
    # Соединённый фрагмент — всё ещё обычный текст: ни таблиц, ни полей
    # чертежа, ни готовой разметки частей у участников быть не должно.
    for fragment in (first, second):
        content = fragment.content
        if content.table_data or content.structured_drawing_fields:
            return False
        if (content.structured_payload or {}).get(PARTS_KEY):
            return False
    return True


def _indexed_text(fragment: Fragment) -> bool:
    """Фрагмент несёт что-то, что потребитель покажет и проиндексирует."""
    content = fragment.content
    if (content.text or "").strip():
        return True
    return bool(content.table_data or content.structured_drawing_fields)


def _join_into(first: Fragment, second: Fragment,
               label_offsets: Dict[int, List[int]]) -> None:
    """Переносит текст и происхождение второго фрагмента в первый."""
    head = (first.content.text or "").rstrip("\n")
    tail = second.content.text or ""
    offset = label_tail(head)
    marks = label_offsets.setdefault(id(first), [])
    if offset is not None:
        marks.append(offset)
    first.content.text = head + _JOIN + tail

    # Происхождение: фрагмент теперь покрывает оба блока, и об этом
    # говорят его спаны. Точные смещения больше не обещаются — текст
    # фрагмента уже не равен тексту одного блока.
    if first.position is not None and second.position is not None:
        first.position.source_spans = [
            *first.position.source_spans, *second.position.source_spans,
        ]
        first.position.text_offsets_exact = False
        first.position.bbox_exact = (
            first.position.bbox_exact and second.position.bbox_exact
            and first.position.bbox == second.position.bbox
        )
        first.position.bbox = _bbox_union(first.position.bbox, second.position.bbox)
        starts = [p for p in (first.position.page_start, second.position.page_start)
                  if p is not None]
        ends = [p for p in (first.position.page_end, second.position.page_end)
                if p is not None]
        if starts and ends:
            first.position.page_start, first.position.page_end = min(starts), max(ends)

    # Уверенность и полнота соединённого фрагмента — по худшему участнику:
    # соединение не делает прочитанное надёжнее.
    first.confidence = round(min(first.confidence, second.confidence), 3)
    if first.completeness is None or second.completeness is None:
        first.completeness = None
    else:
        first.completeness = min(first.completeness, second.completeness)
    # Пометки на проверку складываются, новых имён не появляется.
    first.review_required = bool(first.review_required or second.review_required)
    for warning in second.warnings:
        if warning not in first.warnings:
            first.warnings.append(warning)
    first.missing_regions = [*first.missing_regions, *second.missing_regions]
    if not first.section_path and second.section_path:
        first.section_path = list(second.section_path)
    # На конвейере фактов, узлов и связей здесь ещё нет — их считает
    # `finalize_fragments` позже и уже по соединённому тексту. Перенос
    # оставлен для случая, когда модуль вызван после извлечения: молча
    # потерять факт второго участника он не должен. У перенесённого факта
    # `fragment_id` обязан стать идентификатором фрагмента, в котором он
    # теперь лежит: потребитель проверяет это и отбрасывает чужой факт
    # (`super_rag/parser/types.py`).
    for fact in second.extracted_facts:
        if isinstance(fact, dict):
            if fact.get("fragment_id"):
                fact["fragment_id"] = first.fragment_id
        elif getattr(fact, "fragment_id", None):
            fact.fragment_id = first.fragment_id
    first.extracted_facts = [*first.extracted_facts, *second.extracted_facts]
    first.graph_nodes = [*first.graph_nodes, *second.graph_nodes]
    first.relations = [*first.relations, *second.relations]


def _bbox_union(first: List[float], second: List[float]) -> List[float]:
    if not first or len(first) != 4:
        return second
    if not second or len(second) != 4:
        return first
    return [min(first[0], second[0]), min(first[1], second[1]),
            max(first[2], second[2]), max(first[3], second[3])]


def part_index(text: str, label_offsets: List[int]) -> List[Dict[str, Any]]:
    """
    Части соединённого фрагмента: где подпись, где её значения, где резать.

    По контракту чанкера: части идут по порядку, касаются друг друга,
    покрывают текст от `0` до `len(text)`, `breaks` — начала строк строго
    внутри своей части. Разметка, которая этого не выполняет, чанкером
    отбрасывается целиком, поэтому здесь ничего не угадывается: границы
    берутся из смещений, записанных при соединении, и проверяются перед
    выдачей.

    Частей ровно столько, сколько подписей: текст до первой подписи — своя
    часть (её заголовок — её первая строка), дальше по одной части на
    «подпись и её значения». `title` части со значениями — строка подписи:
    её чанкер повторит на продолжении, и значения не придут без неё.
    """
    if not text:
        return []
    bounds = sorted({offset for offset in label_offsets if 0 <= offset < len(text)})
    if not bounds:
        return []
    # Подпись в самом начале фрагмента (весь первый фрагмент был ею) — это
    # не «часть текста до подписи», а сразу часть «подпись и значения».
    starts = list(bounds) if bounds[0] == 0 else [0, *bounds]
    parts: List[Dict[str, Any]] = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        body = text[start:end]
        title = body.split("\n", 1)[0].strip()
        # Резать внутри части можно по началам строк: там, где перед
        # смещением стоит перевод строки. Первая строка части — её
        # заголовок, и чанкер повторит его на продолжении.
        breaks = [
            offset for offset in range(start + 1, end)
            if text[offset - 1] == "\n"
        ]
        parts.append({
            "kind": "labelled_values" if start in bounds else "text",
            "title": title,
            "start": start,
            "end": end,
            "breaks": breaks,
        })
    return parts if _holds(text, parts) else []


def _holds(text: str, parts: List[Dict[str, Any]]) -> bool:
    """
    Та же проверка, что делает чанкер, — до выдачи, а не после.

    Разметка, которая не описывает текст в точности, бесполезна: чанкер
    отбросит её целиком и вернётся к нынешнему поведению. Проверять здесь
    дешевле, чем узнавать об этом по чужим чанкам.
    """
    cursor = 0
    for part in parts:
        if part["start"] != cursor or part["end"] <= part["start"]:
            return False
        if part["end"] > len(text):
            return False
        if not isinstance(part["kind"], str) or not part["kind"].strip():
            return False
        if not isinstance(part["title"], str):
            return False
        previous = part["start"]
        for offset in part["breaks"]:
            if not part["start"] < offset < part["end"]:
                return False
            if text[offset - 1] != "\n":
                return False
            if offset <= previous and previous != part["start"]:
                return False
            previous = offset
        cursor = part["end"]
    return cursor == len(text)
