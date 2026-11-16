"""
Одна запись номера свидетельства, заводского номера и обозначения
(`SCAN_CANONICAL_IDS`).

Зачем. Слой распознавания читает серию свидетельства о поверке по-разному
в одном и том же документе: «С-ВЫГ/27-09-2022/189200853» на одной странице,
«С-ВЫТ/…» на другой, «C-BbIT/…» латиницей на третьей. Ошибка системная
(Г ↔ Т в модели Tesseract плюс уход в латиницу), а не шум: на проде у 309
из 1 011 номеров серия прочитана двумя способами, и на вопрос «какой номер
свидетельства у датчика» ответ берётся из неверного прочтения. Тот же класс
у заводских номеров («22075о» в файле на 220723–220772) и у обозначения в
штампе («ТАдУ 405210.001» при «ТАДУ 405210.001» в имени файла).

На чём держится правка. У каждого из трёх видов жёсткая форма, и правится
только то, что под неё попало:

* **номер свидетельства** — `<префикс>-<серия>/<дд-мм-гггг>/<9 цифр>`.
  Серия — шифр поверителя, короткий закрытый список. Латинские двойники
  внутри серии сводятся к кириллице («C-BbIT» → «С-ВЫТ»). Верная серия
  берётся из доказательства, по порядку: другие упоминания того же хвоста
  (дата и девять цифр) в документе — голос большинства; список серий
  профиля (`scan_identity.certificate_series`) с расстоянием в один
  похожий знак (Г↔Т, В↔8, З↔3, лишняя буква). Серия, которой нет в
  профиле и которую документ не перекрыл голосами, остаётся как прочитана;
* **заводской номер** — когда имя файла или заголовок называют диапазон
  («зав. №№ 220723-220772»), номер той же длины, у которого вместо цифры
  стоит похожая буква (о↔0, З↔3, б↔6, l↔1) и который после замены попадает
  в диапазон, пишется цифрами. Номер вне диапазона не трогается;
* **обозначение** — обозначение из имени файла, совпадающее с прочитанным
  до регистра и латинских двойников при тех же цифрах, — каноническое.

Чего здесь нет: цифра не становится другой цифрой ни в одном из видов;
токен, не попавший под форму, не меняется; без доказательства не меняется
ничего, даже латиница. Прочтение не теряется: замена записана в
`completeness_mechanisms["canonical_ids"]` (поле, которое RAG не читает и в
текст не выводит) и в факте `identity.*` (`provenance.read_as`).

Связи. Вызывается из `core/contract_finalize.finalize_fragments` после
идентификаторов и фактов: `fragment_id` считается по прочтению, поэтому
включение флага идентификаторы не меняет. Факты вида `identity.*` строит
`core/facts/extractor.identity_fact`.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.facts.extractor import identity_fact

logger = logging.getLogger(__name__)

# Ключ учёта замен в `completeness_mechanisms` фрагмента.
MECHANISMS_KEY = "canonical_ids"
# Секция профиля установки и ключ списка серий в ней.
PROFILE_SECTION = "scan_identity"
PROFILE_SERIES = "certificate_series"

FACT_CERTIFICATE = "identity.certificate_number"
FACT_SERIAL = "identity.serial_number"
FACT_DESIGNATION = "identity.designation"

# Источник фрагмента — скан или картинка: только там текст пришёл из
# распознавания и может быть прочитан неверно.
_SCAN_SOURCES = frozenset({"scanned_pdf", "image"})

# Больше фактов идентичности на фрагмент не нужно: страница реестра
# свидетельств иначе дала бы сотни одинаковых.
_MAX_FACTS = 30

# ===========================================================================
# Номер свидетельства
# ===========================================================================

# Токен перед хвостом: всё, что не пробел и не косая черта. В нём ищется
# начало серии («С-» или «C-» с похожим дефисом); всё до него — знак
# номера в прочтении («№», «#», «N», «М»), он не трогается.
_CERTIFICATE = re.compile(
    r"(?P<token>[^\s/|]{1,16}?)(?P<sep1>\s?/\s?)"
    r"(?P<date>\d{2}[-.]\d{2}[-.]\d{4})(?P<sep2>\s?/\s?)(?P<number>\d{9})(?!\d)"
)
_SERIES_START = re.compile(r"[CСcс][-~–—][^\s/]{1,7}$")

# Латинские двойники кириллицы в серии. «bI» / «bl» — так распознавание
# пишет «Ы», поэтому сводится раньше одиночных букв.
_LATIN_TO_CYRILLIC = str.maketrans({
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "~": "-", "–": "-", "—": "-",
})
_YERU = re.compile(r"[bЬь][Il1|іІ]")

# Похожие знаки внутри серии: одна такая подмена — расстояние один.
_LOOKALIKE_PAIRS = frozenset(
    frozenset(pair) for pair in ("ГТ", "В8", "З3", "Б6", "О0")
)


def fold_series(series: str) -> str:
    """Серия в одном алфавите и регистре: «C-BbIT» -> «С-ВЫТ»."""
    text = _YERU.sub("Ы", series.strip())
    return text.upper().translate(_LATIN_TO_CYRILLIC)


def series_distance(left: str, right: str) -> int:
    """
    0 — одна серия, 1 — одна подмена похожим знаком или одна лишняя буква,
    2 — дальше. Расстояние считается по сведённым к одному алфавиту сериям.
    """
    if left == right:
        return 0
    if len(left) == len(right):
        diff = [(a, b) for a, b in zip(left, right) if a != b]
        if len(diff) == 1 and frozenset(diff[0]) in _LOOKALIKE_PAIRS:
            return 1
        return 2
    if abs(len(left) - len(right)) == 1:
        longer, shorter = (left, right) if len(left) > len(right) else (right, left)
        for index in range(len(longer)):
            if longer[:index] + longer[index + 1:] == shorter:
                # Лишняя буква — только буква: «С-ВЫТГ», но не «С-ВЫГ1».
                return 1 if longer[index].isalpha() else 2
    return 2


def _split_token(token: str) -> Optional[Tuple[str, str]]:
    """Токен перед хвостом -> (знак номера как прочитан, серия) или None."""
    match = _SERIES_START.search(token)
    if match is None:
        return None
    return token[:match.start()], token[match.start():]


def _tail_key(date: str, number: str) -> str:
    return re.sub(r"\D", "", date) + "/" + number


def _to_profile(folded: str, profile: Sequence[str]) -> Optional[str]:
    """Серия профиля, которой может быть прочтение: сама или единственная соседка."""
    if folded in profile:
        return folded
    near = [entry for entry in profile if series_distance(folded, entry) == 1]
    return near[0] if len(near) == 1 else None


def resolve_series(readings: Counter, profile: Sequence[str]) -> Optional[Tuple[str, str]]:
    """
    Прочтения серии одного хвоста в документе -> (серия, доказательство) или
    None, если доказательства нет.

    Порядок доказательств: профиль (каждое прочтение голосует за свою серию
    профиля; решает строгое большинство), затем голос документа среди
    прочтений, похожих друг на друга (строгое большинство). Ничья — нет
    доказательства.
    """
    if profile:
        votes: Counter = Counter()
        for folded, count in readings.items():
            entry = _to_profile(folded, profile)
            if entry is not None:
                votes[entry] += count
        if votes:
            ranked = votes.most_common(2)
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                return ranked[0][0], "profile"
            return None
    ranked = readings.most_common(2)
    if len(ranked) < 2 or ranked[0][1] <= ranked[1][1]:
        return None
    winner = ranked[0][0]
    if all(series_distance(other, winner) <= 1 for other in readings):
        return winner, "document_vote"
    return None


# ===========================================================================
# Заводской номер
# ===========================================================================

# Диапазон в имени файла или заголовке: «220723-220772», «220071 – 220080».
_RANGE = re.compile(r"(?<!\d)(\d{5,8})\s?[-–—]\s?(\d{5,8})(?!\d)")
# Знаки, которые распознавание ставит вместо цифры.
_DIGIT_LOOKALIKES = str.maketrans({
    "о": "0", "О": "0", "o": "0", "O": "0",
    "З": "3", "з": "3",
    "б": "6",
    "l": "1", "I": "1", "|": "1",
})
_SERIAL = re.compile(r"(?<![\w/.,-])([0-9оОoOЗзбlI|]{5,8})(?![\w/])")
# Диапазон шире этого — не партия изделий, а что-то другое (годы, коды).
_MAX_RANGE = 10_000


def serial_ranges(*sources: str) -> List[Tuple[int, int, int]]:
    """Диапазоны заводских номеров (начало, конец, число знаков)."""
    ranges: List[Tuple[int, int, int]] = []
    for source in sources:
        for match in _RANGE.finditer(source or ""):
            low, high = match.group(1), match.group(2)
            if len(low) != len(high):
                continue
            start, end = int(low), int(high)
            if start < end and end - start <= _MAX_RANGE:
                ranges.append((start, end, len(low)))
    return ranges


def canonical_serial(token: str, ranges: Sequence[Tuple[int, int, int]]) -> Optional[str]:
    """
    Номер с похожими на цифры буквами -> цифры, если результат в диапазоне.
    Номер из одних цифр не меняется (None): менять в нём нечего.
    """
    if token.isdigit():
        return None
    digits = token.translate(_DIGIT_LOOKALIKES)
    if not digits.isdigit() or sum(ch.isdigit() for ch in token) < len(token) - 2:
        return None
    value = int(digits)
    for start, end, width in ranges:
        if len(digits) == width and start <= value <= end:
            return digits
    return None


def _in_range(token: str, ranges: Sequence[Tuple[int, int, int]]) -> bool:
    return token.isdigit() and any(
        len(token) == width and start <= int(token) <= end for start, end, width in ranges
    )


# ===========================================================================
# Обозначение
# ===========================================================================

# Обозначение по ГОСТ 2.201: код организации, шесть цифр, порядковые номера.
_DESIGNATION = re.compile(
    r"(?<![\wА-ЯЁа-яё])([A-Za-zА-ЯЁа-яё]{2,4})([ ._]?)(\d{6}(?:\.\d{3})*(?:-\d{2,3})?)(?![\d.])"
)


def _fold_letters(code: str) -> str:
    return code.upper().translate(_LATIN_TO_CYRILLIC)


def name_designation(name: str) -> Optional[str]:
    """Обозначение из имени файла кириллицей, если оно там одно."""
    found = {
        f"{_fold_letters(m.group(1))} {m.group(3)}"
        for m in _DESIGNATION.finditer(name or "")
        if re.fullmatch(r"[А-ЯЁ]{2,4}", _fold_letters(m.group(1)))
    }
    return found.pop() if len(found) == 1 else None


def canonical_designation(read: str, from_name: str) -> Optional[str]:
    """
    Прочитанное обозначение -> запись из имени файла, если они одно и то же
    до регистра и латинских двойников при тех же цифрах; иначе None.
    """
    code, number = from_name.split(" ", 1)
    match = _DESIGNATION.fullmatch(read.strip())
    if match is None:
        return None
    if match.group(3) != number or _fold_letters(match.group(1)) != code:
        return None
    wanted = f"{code}{match.group(2) if match.group(2) != '_' else ' '}{number}"
    return None if wanted == read.strip() else wanted


# ===========================================================================
# Документ целиком
# ===========================================================================

def _is_scan(fragment) -> bool:
    provenance = getattr(fragment, "provenance", None)
    return getattr(provenance, "source", None) in _SCAN_SOURCES


def _texts(fragment) -> Iterable[Tuple[str, str]]:
    """(место, текст) фрагмента: текст и ячейки таблицы."""
    content = fragment.content
    if content.text:
        yield "text", content.text
    table = content.table_data
    if table is not None:
        for column, value in enumerate(table.headers or []):
            if isinstance(value, str) and value:
                yield f"cell:h:{column}", value
        for row_index, row in enumerate(table.rows or []):
            for column, value in enumerate(row or []):
                if isinstance(value, str) and value:
                    yield f"cell:{row_index}:{column}", value


def _set_text(fragment, where: str, value: str) -> None:
    content = fragment.content
    if where == "text":
        if fragment.position is not None and len(value) != len(content.text or ""):
            fragment.position.text_offsets_exact = False
        content.text = value
        return
    _, row, column = where.split(":")
    if row == "h":
        content.table_data.headers[int(column)] = value
    else:
        content.table_data.rows[int(row)][int(column)] = value


def _record(fragment, kind: str, before: str, after: str, evidence: str) -> None:
    mechanisms = dict(fragment.completeness_mechanisms or {})
    changes = list(mechanisms.get(MECHANISMS_KEY) or [])
    entry = {"kind": kind, "read_as": before, "canonical": after, "evidence": evidence}
    if entry not in changes:
        changes.append(entry)
    mechanisms[MECHANISMS_KEY] = changes
    fragment.completeness_mechanisms = mechanisms


def _certificate_plan(fragments, profile: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    readings: Dict[str, Counter] = {}
    for fragment in fragments:
        if not _is_scan(fragment):
            continue
        for _, text in _texts(fragment):
            for match in _CERTIFICATE.finditer(text):
                split = _split_token(match.group("token"))
                if split is None:
                    continue
                tail = _tail_key(match.group("date"), match.group("number"))
                readings.setdefault(tail, Counter())[fold_series(split[1])] += 1
    plan: Dict[str, Tuple[str, str]] = {}
    for tail, counter in readings.items():
        resolved = resolve_series(counter, profile)
        if resolved is not None:
            plan[tail] = resolved
    return plan


def _rewrite_certificates(
    text: str, plan, facts: list, profile: Sequence[str] = (),
) -> Tuple[str, List[Tuple[str, str, str]]]:
    """
    Серии свидетельств текста -> каноническая серия своего хвоста.

    Хвост (дата и девять цифр) называет одно свидетельство, поэтому у
    упоминания того же хвоста серия одна. Прочтение, у хвоста которого
    есть доказанная серия, заменяется ею, даже если оно дальше одного
    похожего знака («С-ВНI» при «С-ВЫГ» того же хвоста): спорным тут
    может быть только другое свидетельство, а его серия была бы в профиле.
    Прочтение, которое само есть серия профиля, не трогается.
    """
    changes: List[Tuple[str, str, str]] = []

    def replace(match: "re.Match[str]") -> str:
        split = _split_token(match.group("token"))
        if split is None:
            return match.group(0)
        prefix, series = split
        tail = _tail_key(match.group("date"), match.group("number"))
        resolved = plan.get(tail)
        folded = fold_series(series)
        canonical, evidence = series, ""
        if resolved is not None and (folded == resolved[0] or folded not in profile):
            canonical, evidence = resolved
        raw_number = f"{series}/{match.group('date')}/{match.group('number')}"
        number = f"{canonical}/{match.group('date')}/{match.group('number')}"
        facts.append((FACT_CERTIFICATE, number, raw_number, evidence))
        if canonical == series:
            return match.group(0)
        changes.append((raw_number, number, evidence))
        return (prefix + canonical + match.group("sep1") + match.group("date")
                + match.group("sep2") + match.group("number"))

    return _CERTIFICATE.sub(replace, text), changes


def _rewrite_serials(text: str, ranges, facts: list) -> Tuple[str, List[Tuple[str, str, str]]]:
    changes: List[Tuple[str, str, str]] = []

    def replace(match: "re.Match[str]") -> str:
        token = match.group(1)
        fixed = canonical_serial(token, ranges)
        if fixed is None:
            if _in_range(token, ranges):
                facts.append((FACT_SERIAL, token, token, ""))
            return token
        facts.append((FACT_SERIAL, fixed, token, "range"))
        changes.append((token, fixed, "range"))
        return fixed

    return _SERIAL.sub(replace, text), changes


def _rewrite_designations(text: str, from_name: str, facts: list):
    changes: List[Tuple[str, str, str]] = []

    def replace(match: "re.Match[str]") -> str:
        read = match.group(0)
        fixed = canonical_designation(read, from_name)
        if fixed is None:
            return read
        facts.append((FACT_DESIGNATION, fixed, read, "file_name"))
        changes.append((read, fixed, "file_name"))
        return fixed

    return _DESIGNATION.sub(replace, text), changes


def profile_series() -> List[str]:
    """
    Серии свидетельств из профиля установки
    (`scan_identity.certificate_series`): строка или `{series, note}`.
    """
    from core.gateway.profile import load_section

    entries = load_section(PROFILE_SECTION).get(PROFILE_SERIES) or []
    series: List[str] = []
    for entry in entries if isinstance(entries, list) else []:
        value = entry.get("series") if isinstance(entry, dict) else entry
        if isinstance(value, str) and value.strip():
            series.append(value.strip())
    return series


def apply(
    fragments, source_name: str = "", series: Optional[Sequence[str]] = None,
    file_id: str = "",
) -> int:
    """
    Канонические номера и обозначения во фрагментах документа. Возвращает
    число замен. Факты `identity.*` дописываются в `extracted_facts`.

    `source_name` — исходное имя файла (диапазон заводских номеров и
    обозначение); `file_id` — ключ файла в хранилище: в транслитерированном
    ключе снимка буквы потеряны, но цифры диапазона остаются, поэтому из него
    берётся только диапазон.
    """
    profile = [fold_series(entry) for entry in (series or []) if str(entry).strip()]
    plan = _certificate_plan(fragments, profile)
    headings = [getattr(f, "section_title", None) or "" for f in fragments]
    ranges = serial_ranges(source_name, file_id, *headings)
    from_name = name_designation(source_name)
    total = 0
    for fragment in fragments:
        scan = _is_scan(fragment)
        drawing = fragment.type == "drawing"
        if not scan and not drawing:
            continue
        facts: List[Tuple[str, str, str, str]] = []
        for where, text in list(_texts(fragment)):
            new_text, changes = text, []
            if scan:
                new_text, found = _rewrite_certificates(new_text, plan, facts, profile)
                changes += [("certificate_number",) + c for c in found]
                if ranges:
                    new_text, found = _rewrite_serials(new_text, ranges, facts)
                    changes += [("serial_number",) + c for c in found]
            if from_name:
                new_text, found = _rewrite_designations(new_text, from_name, facts)
                changes += [("designation",) + c for c in found]
            if new_text != text:
                _set_text(fragment, where, new_text)
            for kind, before, after, evidence in changes:
                _record(fragment, kind, before, after, evidence)
                total += 1
        if drawing and from_name:
            total += _drawing_designation(fragment, from_name, facts)
        _add_facts(fragment, facts)
    if total:
        logger.info("Канонические номера и обозначения: замен %d", total)
    return total


def _drawing_designation(fragment, from_name: str, facts: list) -> int:
    """Графа «Обозначение» штампа: в payload и в полях фрагмента."""
    changed = 0
    payload = fragment.content.structured_payload or {}
    block = payload.get("title_block")
    if isinstance(block, dict) and isinstance(block.get("designation"), str):
        read = block["designation"]
        match = _DESIGNATION.search(read)
        fixed = canonical_designation(match.group(0), from_name) if match else None
        if fixed is not None:
            block["designation"] = read.replace(match.group(0), fixed)
            facts.append((FACT_DESIGNATION, fixed, match.group(0), "file_name"))
            _record(fragment, "designation", match.group(0), fixed, "file_name")
            changed += 1
    for drawing_field in fragment.content.structured_drawing_fields or []:
        value, found = _rewrite_designations(drawing_field.value, from_name, [])
        if found:
            drawing_field.value = value
    return changed


def _add_facts(fragment, found: List[Tuple[str, str, str, str]]) -> None:
    if not found:
        return
    existing = list(fragment.extracted_facts or [])
    seen = {(f.key, f.value.raw) for f in existing}
    added = 0
    for key, value, read_as, evidence in found:
        if (key, value) in seen or added >= _MAX_FACTS:
            continue
        seen.add((key, value))
        existing.append(identity_fact(
            fragment.fragment_id, len(existing) + 1, key, value, read_as, evidence,
        ))
        added += 1
    fragment.extracted_facts = existing
