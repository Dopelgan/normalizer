"""
Основная надпись (штамп) чертежа: графы с подписями, боковые колонки,
рукописные записи.

Зачем отдельный проход и отдельный модуль. До него штамп читался тем же
способом, что и любая таблица листа: модель возвращала свободный словарь
`fields`, и в индекс попадало то, что она сама решила назвать графой. На
проверенных листах это давало три потери подряд.

* **Значение без имени графы.** В ответе приходило «12,35» и «1:4», а по
  какому полю — неизвестно. Ответить «какая масса у расширителя» по такой
  записи нельзя: число есть, графы нет. Поэтому здесь имя каждой графы
  приводится к канонической записи ГОСТ 2.104, и в контракт значение
  уходит только вместе с подписью.
* **Рукописное не возвращалось вовсе.** Даты подписей, сами подписи и
  инвентарные номера модель либо пропускала, либо отдавала как «подпись».
  Они нужны: «инв. № 1085-02» и «26.08.22» — это и есть ответ на вопрос,
  когда и кем лист подписан. Рукописное помечается отдельно: напечатанное
  и дописанное от руки — разной надёжности сведения, и смешивать их в
  одном поле нельзя.
* **Графа изменений и боковые графы терялись.** Строки «2 Нов.
  ТАДУ.409-23» приходили в `rows` таблицы, но полями не становились —
  `rows` в поля не раскладывались никогда. Боковые графы («Инв. № подл.»,
  «Перв. примен.») лежат вне рамки штампа, в поле подшивки, и в вырезку
  штампа просто не попадали.

Обозначения. Точки в обозначении по ГОСТ 2.201 модель теряет
(«ТАДУ 421451.001» -> «ТАДУ 421451001»), а код документа читает
гомоглифами («СБ» -> «C6», «ТЗ» -> «73»). Восстанавливаются только те два
случая, где запись однозначна: группировка цифр по три после шестизначного
классификатора и двухбуквенный код документа из списка ГОСТ 2.102.
Догадки вроде «сколько групп было в СМ1638.11.00.00.00.00» здесь не
делаются — потерянную группу восстановить нечем, это лечится чтением
вырезки в высоком разрешении, а не правилом.

Модуль — чистые функции над ответом модели: ни сети, ни PIL.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Канонические графы основной надписи и боковых колонок по ГОСТ 2.104.
# Ключ — внутреннее имя, значение — подпись, с которой поле уходит в индекс.
LABELS: Dict[str, str] = {
    "designation": "Обозначение",
    "name": "Наименование",
    "material": "Материал",
    "mass": "Масса",
    "scale": "Масштаб",
    "sheet": "Лист",
    "sheets": "Листов",
    "letter": "Литера",
    "organization": "Организация",
    "developed": "Разраб.",
    "checked": "Пров.",
    "tech_control": "Т.контр.",
    "norm_control": "Н.контр.",
    "approved": "Утв.",
    "format": "Формат",
    "changes": "Изм.",
    "inventory": "Инв. № подл.",
    "inventory_dup": "Инв. № дубл.",
    "replaced_inventory": "Взам. инв. №",
    "first_use": "Перв. примен.",
    "reference": "Справ. №",
    "signed_date": "Подп. и дата",
}

# Пометка, с которой рукописное значение уходит в контракт.
HANDWRITTEN_MARK = "рукописно"

# Графы, которые на листе заполняются от руки. Только они могут быть
# помечены рукописными: на проверенном листе модель назвала рукописными
# массу, масштаб и номер листа, напечатанные чертёжным шрифтом. Пометка не
# украшение — она понижает уверенность поля, и ошибаться в ней нельзя.
_HAND_FILLED = frozenset({
    "inventory", "inventory_dup", "replaced_inventory", "reference",
    "signed_date", "changes",
})

# Как модель называет графу -> каноническое имя. Ключи сравниваются по
# нормализованной записи (строчные, без точек, пробелов и знака «№»).
_SYNONYMS: Dict[str, str] = {}


def _register(key: str, *names: str) -> None:
    for name in names:
        _SYNONYMS[_fold(name)] = key


def _fold(name: str) -> str:
    """Имя графы без того, чем модель его украсила: точек, пробелов, «№»."""
    return re.sub(r"[\s.,:№\-_]+", "", str(name).strip().lower()).replace("ё", "е")


_register("designation", "обозначение", "обозначение документа", "шифр", "децимальный номер")
_register("name", "наименование", "название", "наименование изделия", "наименование документа")
_register("material", "материал", "марка материала", "сортамент")
_register("mass", "масса", "вес")
_register("scale", "масштаб")
_register("sheet", "лист", "номер листа")
_register("sheets", "листов", "всего листов", "кол-во листов")
_register("letter", "литера", "лит", "лит.")
_register("organization", "организация", "предприятие", "фирма", "завод")
_register("developed", "разработал", "разраб", "разраб.", "исполнитель")
_register("checked", "проверил", "пров", "пров.", "проверяющий")
_register("tech_control", "т контр", "тконтр", "техконтроль", "технический контроль", "т.контр.")
_register("norm_control", "н контр", "нконтр", "нормоконтроль", "нормоконтролер", "н.контр.")
_register("approved", "утвердил", "утв", "утв.", "утверждающий")
_register("format", "формат")
_register("changes", "изменения", "изм", "графа изменений", "таблица изменений")
_register("inventory", "инв подл", "инвподл", "инв номер подлинника",
          "инвентарный номер подлинника", "инв. № подл.", "подл")
_register("inventory_dup", "инв дубл", "инвдубл", "инвентарный номер дубликата")
_register("replaced_inventory", "взам инв", "взаминв", "взам. инв. №")
_register("first_use", "перв примен", "первприменение", "первичная применяемость",
          "перв. примен.", "первое применение")
_register("reference", "справ", "справочный номер", "справ. №")
_register("signed_date", "подп и дата", "подпись и дата", "подп. и дата")

# Подписант -> каноническая графа: по этим словам опознаётся строка
# подписей в `rows`, где ни одна ячейка не названа графой.
_SIGNERS = {
    "разраб": "developed", "пров": "checked", "проверил": "checked",
    "тконтр": "tech_control", "нконтр": "norm_control", "утв": "approved",
    "утвердил": "approved", "разработал": "developed",
}

# Слово «подпись» вместо фамилии: модель ставит его там, где на листе
# закорючка. Фамилией оно не является.
_NOT_A_NAME = {"подпись", "подп", "подписано", "рукописно"}

# Дата в графе: «26.08.22», «22.09.25», «14.10.16», «05.2016». Две части без
# третьей сюда не входят: «12,35» в массе и «1.4» в ссылке на пункт датами
# не являются, а раньше становились ими.
DATE = re.compile(r"^\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}$|^\d{1,2}[.\-/]\d{4}$")
# Номер извещения об изменении: «409-23», «355-22», «656/2-26»,
# «ТАДУ.409-23». Ячейка должна состоять из него целиком: иначе номером
# извещения становится любая ссылка на ГОСТ («ГОСТ 19903-74»), и обычная
# строка штампа превращалась в запись об изменении.
_NOTICE = re.compile(r"^(?:[А-ЯЁA-Z]{2,4}[. ])?\d+(?:/\d+)?-\d{2}$")


# ===========================================================================
# Имена граф
# ===========================================================================

def canonical(name: str) -> Optional[str]:
    """Имя графы от модели -> каноническое имя или `None`, если не узнали."""
    folded = _fold(name)
    if not folded:
        return None
    if folded in _SYNONYMS:
        return _SYNONYMS[folded]
    # «Наименование материала» и «обозначение материала» — это материал, а
    # не наименование изделия и не обозначение документа. Проверяется до
    # начала строки, иначе по префиксу выигрывает первое слово.
    if "материал" in folded:
        return "material"
    # Модель любит дописать к имени графы её содержимое («разраб фамилия»).
    # Длинное совпадение раньше короткого: «листов всего» — это «листов», а
    # не «лист», и порядок, в котором имена записаны, на это не влияет.
    for known in sorted(_SYNONYMS, key=len, reverse=True):
        if len(known) >= 4 and folded.startswith(known):
            return _SYNONYMS[known]
    return None


def label(key: str) -> str:
    """Подпись графы для индекса. Неизвестное имя идёт как есть."""
    return LABELS.get(key, key)


# ===========================================================================
# Обозначения
# ===========================================================================

# Обозначение по ГОСТ 2.201: код организации (4 буквы), шестизначный
# классификационный номер, дальше порядковые номера по три цифры.
_DESIGNATION = re.compile(r"^([А-ЯЁA-Z]{2,4})([ .]?)(\d+)((?:\.[.\d]*\d)?)$")
# Классификационный номер по ГОСТ 2.201 — шесть цифр.
_CLASSIFIER = 6
# Код документа по ГОСТ 2.102 / 2.701 — два знака в конце обозначения.
_DOC_CODES = {
    "СБ", "ВО", "ТЧ", "ГЧ", "МЭ", "МЧ", "УЧ", "ТБ", "РР", "ТУ", "ПЗ", "ТЗ",
    "ПМ", "РЭ", "ФО", "ПС", "ВС", "ВД", "ВП", "ТП", "ЭП", "И1", "Д1", "Д2",
    "Д3", "Э1", "Э2", "Э3", "Э4",
}
# Гомоглифы: чем модель заменяет букву кода. Применяется только к коду
# документа — в произвольном тексте такая замена ломает верно прочитанное.
_LOOKALIKE = {
    "A": "А", "B": "В", "C": "С", "E": "Е", "G": "Г", "H": "Н", "K": "К",
    "M": "М", "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У", "N": "И",
    "3": "З", "6": "Б", "0": "О", "4": "Ч", "9": "Э", "7": "Т",
}
# Код организации латиницей: «AEGH.01…» вместо «АЕГН.01…». Меняется только
# перед цифрами обозначения — код KKS («30KUA70AX001») начинается с цифр и
# под правило не попадает.
_LATIN_PREFIX = re.compile(r"^([A-Z]{2,4})(?=[ .]?\d)")


def fix_designation(text: str) -> str:
    """
    Обозначение, как его прочитала модель -> запись по ГОСТ 2.201.

    Восстанавливаются только однозначные случаи: точки между порядковыми
    номерами по три цифры, код организации латиницей и двухзначный код
    документа из списка ГОСТ 2.102. Всё остальное остаётся как прочитано:
    выдуманная точка хуже отсутствующей.
    """
    value = " ".join(str(text or "").split())
    if not value:
        return ""
    parts = value.split(" ")
    # Код документа — последнее слово из двух знаков.
    code = ""
    if len(parts) > 1 and len(parts[-1]) == 2:
        fixed = fix_document_code(parts[-1])
        if fixed:
            code, parts = fixed, parts[:-1]
    head = " ".join(parts)
    head = _LATIN_PREFIX.sub(lambda m: "".join(_LOOKALIKE.get(c, c) for c in m.group(1)), head)
    head = _split_triples(head)
    return f"{head} {code}".strip() if code else head


def fix_document_code(token: str) -> str:
    """Код документа («C6» -> «СБ», «73» -> «ТЗ») или пустая строка."""
    token = str(token).strip()
    if len(token) != 2:
        return ""
    if token.upper() in _DOC_CODES:
        return token.upper()
    guess = "".join(_LOOKALIKE.get(character.upper(), character.upper()) for character in token)
    return guess if guess in _DOC_CODES else ""


def _split_triples(head: str) -> str:
    """
    Точки между порядковыми номерами. Слипшийся хвост делится по три цифры
    только если его длина кратна трём: «687000001001» -> «687000.001.001»,
    а «6870000.001» (лишний ноль от модели) остаётся как есть.
    """
    match = _DESIGNATION.match(head)
    if not match:
        return head
    prefix, space, digits, tail = match.groups()
    classifier, glued = digits, ""
    extra = len(digits) - _CLASSIFIER
    if extra > 0 and extra % 3 == 0:
        # «421451001»: шестизначный классификатор плюс порядковые номера.
        classifier, glued = digits[:_CLASSIFIER], digits[_CLASSIFIER:]
    groups: List[str] = []
    for chunk in (glued + (tail or "")).split("."):
        if not chunk:
            continue
        if len(chunk) > 3 and len(chunk) % 3 == 0:
            groups.extend(chunk[i:i + 3] for i in range(0, len(chunk), 3))
        else:
            groups.append(chunk)
    joined = ".".join(groups)
    # Пробел между кодом и номером ставится только тот, что был на листе:
    # «СМ1638.11.00…» пишется без пробела, и добавить его значит испортить
    # верно прочитанное обозначение.
    return f"{prefix}{space}{classifier}" + (f".{joined}" if joined else "")


# ===========================================================================
# Разбор ответа модели
# ===========================================================================

def parse(answer: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ответ модели по вырезке штампа -> {'fields', 'handwritten', 'rows'}.

    `fields` — канонические графы со значениями, `handwritten` — те из них,
    которые модель назвала рукописными. Графа изменений и строка подписей
    добираются из `rows`, если в `fields` их нет: модель кладёт таблицу
    построчно почти всегда, а графами называет далеко не всё.
    """
    raw = answer.get("fields")
    raw = raw if isinstance(raw, dict) else {}
    hand_names = answer.get("handwritten")
    hand_names = hand_names if isinstance(hand_names, list) else []

    fields: Dict[str, str] = {}
    handwritten: Dict[str, str] = {}
    unknown: Dict[str, str] = {}
    marked = {_fold(name) for name in hand_names if isinstance(name, str)}

    for name, value in raw.items():
        text = _text(value)
        if not text:
            continue
        key = canonical(name)
        if key is None:
            unknown[" ".join(str(name).split())] = text
            continue
        if key in ("designation", "first_use"):
            text = fix_designation(text)
        elif key in ("signed_date", "changes"):
            text = _collapse(text)
        by_hand = (_fold(name) in marked or _is_handwritten(value)) and key in _HAND_FILLED
        (handwritten if by_hand else fields).setdefault(key, text)

    rows = _rows(answer.get("rows"))
    _from_rows(rows, fields, handwritten)
    return {"fields": fields, "handwritten": handwritten, "unknown": unknown, "rows": rows}


def _text(value: Any) -> str:
    """Значение графы: строкой или объектом {'value', 'handwritten'}."""
    if isinstance(value, dict):
        value = value.get("value") or value.get("значение") or ""
    if value is None or isinstance(value, (list, tuple, dict)):
        return ""
    return " ".join(str(value).split())


def _collapse(text: str) -> str:
    """
    Повторы в перечислении через «;». Все пятеро подписали лист в один
    день, и модель возвращает «26.08.22; 26.08.22; 26.08.22; 26.08.22;
    26.08.22» — одна дата, записанная пять раз, сведениями не является.
    """
    parts = [part.strip() for part in text.split(";") if part.strip()]
    return "; ".join(dict.fromkeys(parts)) if parts else text


def _is_handwritten(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    flag = value.get("handwritten")
    if flag is None:
        flag = value.get("рукописно")
    return bool(flag)


def _rows(raw: Any) -> List[List[str]]:
    if not isinstance(raw, list):
        return []
    rows: List[List[str]] = []
    for row in raw:
        if isinstance(row, list):
            rows.append([_text(cell) for cell in row])
        elif isinstance(row, dict):
            rows.append([_text(cell) for cell in row.values()])
    return rows


def _from_rows(
    rows: List[List[str]], fields: Dict[str, str], handwritten: Dict[str, str]
) -> None:
    """
    Графа изменений и строки подписей из таблицы штампа.

    Строка подписей — «Разраб. | Меснянкин | | | 26.08.22»: подписант,
    фамилия и дата. Дата рукописная почти всегда, поэтому она идёт в
    `handwritten` отдельной графой, а не приклеивается к фамилии. Дата
    остаётся при своём подписанте: «Разраб. 26.08.22; Утв. 30.09.25». Без
    этого на вопрос «кем и когда подписан лист» отвечал бы список дат, по
    которому не видно, чья какая.
    """
    changes: List[str] = []
    dates: List[str] = []
    for row in rows:
        cells = [cell for cell in row if cell]
        if not cells:
            continue
        key = _signer(cells[0])
        if key is not None:
            rest = cells[1:]
            names = [
                cell for cell in rest
                if not DATE.match(cell) and _fold(cell) not in _NOT_A_NAME
            ]
            if names:
                fields.setdefault(key, names[0])
            dates.extend(f"{label(key)} {cell}" for cell in rest if DATE.match(cell))
            continue
        entry = _change(cells)
        if entry:
            changes.append(entry)
            dates.extend(
                f"{label('changes')} {cell}" for cell in cells if DATE.match(cell)
            )
    # Графа, которую модель уже назвала — хоть печатной, хоть рукописной, —
    # из строк не добирается: иначе одно изменение приходит дважды.
    if changes and "changes" not in fields and "changes" not in handwritten:
        fields["changes"] = "; ".join(dict.fromkeys(changes))
    if dates and "signed_date" not in handwritten and "signed_date" not in fields:
        handwritten["signed_date"] = "; ".join(dict.fromkeys(dates))


def _signer(first: str) -> Optional[str]:
    folded = _fold(first)
    for word, key in _SIGNERS.items():
        if folded.startswith(word):
            return key
    return None


def _change(cells: List[str]) -> str:
    """
    Запись об изменении: «2 | Нов. | ТАДУ.409-23 | подпись | 22.10.16» ->
    «2 Нов. ТАДУ.409-23».

    Опознаётся по номеру извещения («409-23», «656/2-26»), а не по порядку
    колонок: у модели первой ячейкой бывает и номер изменения, и само слово
    «Изм». Шапка графы номера извещения не содержит и записью не станет.
    Номер извещения в запись входит всегда — без него запись ни о чём.
    """
    notice = next((cell for cell in cells if _NOTICE.match(cell)), "")
    if not notice:
        return ""
    meaningful = [
        cell for cell in cells
        if cell and not DATE.match(cell) and _fold(cell) not in _NOT_A_NAME
    ]
    head = [cell for cell in meaningful if cell != notice][:2]
    return " ".join(head + [notice]).strip()


# ===========================================================================
# Выдача в контракт
# ===========================================================================

def entries(parsed: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    """
    Поля штампа для фрагмента: (подпись графы, значение, рукописное ли).

    Порядок — как в `LABELS`, чтобы в индексе штамп всегда выглядел
    одинаково, а не в порядке, в котором модель перечислила графы.
    """
    fields = parsed.get("fields") or {}
    handwritten = parsed.get("handwritten") or {}
    result: List[Tuple[str, str, bool]] = []
    for key in LABELS:
        if key in fields:
            result.append((label(key), fields[key], False))
        if key in handwritten:
            result.append((label(key), handwritten[key], True))
    for name, value in (parsed.get("unknown") or {}).items():
        result.append((name, value, False))
    return result


def mark(value: str, handwritten: bool) -> str:
    """Значение с пометкой о рукописном. Пометка — часть текста поля."""
    return f"{value} ({HANDWRITTEN_MARK})" if handwritten else value
