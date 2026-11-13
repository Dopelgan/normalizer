"""
Восстановление обозначений и стандартов, испорченных путаницей латиницы и
кириллицы (`SCAN_DESIGNATION_REPAIR`).

Зачем. На скане чертежа обычное распознавание читает чертёжный шрифт ГОСТ
вперемешку: латинская «T» вместо «Т» («ТАДУ 405210.001TY»), кириллическая
«О» вместо нуля («12Х18Н1ОТ»), «З» вместо тройки («Св-04Х19Н11МЗ»),
строчные буквы вместо прописных («нП-о01-97»). В индексе «НП-001-97» и
«нП-о01-97» — разные строки: обозначение не находится поиском и не
сверяется с документом.

На чём это держится. У обозначения есть жёсткая форма, и форма сама
говорит, что в каком месте стоит. Поэтому правка идёт не по целому слову, а
по частям, и у каждой части своё правило:

* **префикс стандарта** («ГОСТ», «ГОСТ Р», «ОСТ», «СТО», «СТ ЦКБА», «НП»,
  «РД», «ТУ», «ИСО», «ПНАЭ»…) — закрытый список. Часть правится, если
  ровно одно прочтение по неразличимым знакам даёт член списка;
* **номер** — только цифры и разделители. Буква здесь стоять не может,
  поэтому «О» читается нулём, а «З» — тройкой. Обратно — никогда: цифра не
  становится буквой и не становится другой цифрой;
* **код документа** за обозначением ЕСКД («СБ», «ВО», «ТУ», «РР»…) —
  закрытый список;
* **код организации** в обозначении ЕСКД и **марка стали или сварочной
  проволоки** — здесь буквы и есть содержание, закрытой формы у них нет.
  Такая часть правится только тогда, когда прочтение совпало с настоящим
  обозначением из справочника ниже. Справочник собран по корпусу прода
  (`tools/harvest_designations.py`) и по ГОСТ 5632 и ГОСТ 2246 — это список
  настоящих обозначений, а не список ошибок;
* **шрифт по ГОСТ 26.008-85** — высота, «Пр» или «Шр», номер.

Правка делается, если прочтение одно. Два прочтения — значит неизвестно,
какое верно, и часть остаётся как была.

Чего здесь нет и не будет:

* цифра не превращается ни в букву, ни в другую цифру. «527 Не» не станет
  моментом затяжки, «061960.001» не станет «061360.001»;
* буква, непохожая на нужную, не меняется: «TARY» не станет «ТАДУ», потому
  что латинской «R» не соответствует «Д», а «AI» не станет «АЭ»;
* не правится то, что читается двумя способами. Латинское «OCT» на листе —
  это и «ОСТ», и «ГОСТ», у которого потерялась первая буква; «ОСТ 5582-75»
  вместо «ГОСТ 5582-75» было бы правдоподобной подменой стандарта, поэтому
  участок остаётся как был (на корпусе таких отказов 55);
* незнакомый код организации и незнакомая марка остаются как есть. Это
  плата за то, что придуманный код не выдаётся за настоящий: проверено на
  корпусе, где перебор без справочника делал из «TAAY» несуществующий
  «ТААУ», а из «12X1BH10T» — несуществующую марку «12Х1ВН10Т»;
* испорченное слово обычного текста («оргонодугобой», «Маркуровать»,
  «зоне») не трогается вовсе: у слова нет строгой формы, и правка по
  похожести дала бы правдоподобную подмену смысла. Такой текст исправляет
  перечитывание страницы моделью (`core/scan_text_rescan.py`).

Почему так. Правдоподобно неверное требование хуже видимо испорченного:
неверное цитируется как факт. Модуль правит только то, где форма или
справочник сами называют верный знак, и оставляет остальное сломанным —
заметно сломанным.

Связи. Вызывается из `core/contract_finalize.py` для каждого текстового
фрагмента до извлечения фактов.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

# Ключ в `structured_payload` фрагмента: что именно было восстановлено.
# Не пометка (`warnings`): для RAG пометка означает «текст неполный», а
# здесь всё наоборот — текст стал вернее. Документ с восстановленным
# обозначением не должен из-за этого терять полноту и чистое событие
# индексации. Ключ остаётся измеримым для нас и не трогает полноту у
# потребителя.
PAYLOAD_KEY = "designation_repairs"

# Классы неразличимых знаков: на листе знаки класса выглядят одинаково, и
# какой из них имелся в виду, решает форма обозначения. Двух разных цифр в
# одном классе нет — цифра не может превратиться в другую цифру.
_CLASSES: Tuple[str, ...] = (
    "АA", "аa", "ВB", "ЕE", "еe", "КK", "кk", "МM", "НH", "ОO0", "оo0",
    "РP", "рp", "СC", "сc", "ТT", "УY", "уy", "ХX", "хx", "З3", "з3",
    "Il1Іі", "ЈJ", "јj", "ЅS", "ѕs",
)


def _build_alternatives() -> Dict[str, Tuple[str, ...]]:
    """
    Знак -> знаки, которыми он мог быть на листе.

    У цифры прочтений нет: иначе перебор придумывал бы букву на месте
    цифры — ровно та догадка, которой здесь не должно быть. Буква читается
    и как своя пара из другого алфавита, и прописной, и (для «О» и «З») как
    цифра: место цифры задано формой обозначения.
    """
    groups: Dict[str, set] = {}
    for group in _CLASSES:
        members = set(group)
        for char in group:
            groups.setdefault(char, set()).update(members)
    table: Dict[str, Tuple[str, ...]] = {}
    lowercase = [chr(code) for code in range(0x430, 0x450)] + ["ё"]
    for char in list(groups) + lowercase:
        if char.isdigit():
            table[char] = (char,)
            continue
        variants = set(groups.get(char, ())) | {char}
        variants.update(variant.upper() for variant in list(variants))
        variants.update(groups.get(char.upper(), set()))
        table[char] = tuple(sorted(variants))
    return table


_ALTERNATIVES = _build_alternatives()
# Сколько прочтений части разбирается; больше — часть не правится.
_MAX_VARIANTS = 4096
# Буква, которая на месте цифры читается цифрой (и только так).
_AS_DIGIT = {"О": "0", "о": "0", "O": "0", "o": "0", "З": "3", "з": "3"}


def _loose(char: str) -> str:
    """Класс регулярного выражения: знак и все неразличимые с ним."""
    members = set(_ALTERNATIVES.get(char, (char,))) | {char}
    members.update(member.lower() for member in list(members))
    return "[" + "".join(re.escape(member) for member in sorted(members)) + "]"


def _loose_word(word: str) -> str:
    return "".join(_loose(char) for char in word)


def _loose_any(words: Sequence[str]) -> str:
    return "(?:" + "|".join(_loose_word(word) for word in words) + ")"


def readings(part: str) -> List[str]:
    """Прочтения части: каждый знак — своим классом неразличимых."""
    options = [_ALTERNATIVES.get(char, (char,)) for char in part]
    total = 1
    for option in options:
        total *= len(option)
        if total > _MAX_VARIANTS:
            return []
    results = [""]
    for option in options:
        results = [prefix + char for prefix in results for char in option]
    return results


def _only_reading(part: str, allowed: Sequence[str]) -> Optional[str]:
    """Единственное прочтение части, которое есть в списке, иначе None."""
    permitted = set(allowed)
    if part in permitted:
        return part
    found = {variant for variant in readings(part) if variant in permitted}
    return found.pop() if len(found) == 1 else None


def _only_entry(part: str, allowed: Sequence[str]) -> Optional[str]:
    """
    То же, но регистр сверяется без учёта различий, а возвращается написание
    из справочника: «CB-04X19H11M3» — это «Св-04Х19Н11М3», где «в» строчная
    по ГОСТ 2246, а не потому, что так прочиталось.
    """
    canonical = {entry.casefold(): entry for entry in allowed}
    if part in set(allowed):
        return part
    found = {canonical[variant.casefold()] for variant in readings(part)
             if variant.casefold() in canonical}
    return found.pop() if len(found) == 1 else None


def _digits_only(part: str) -> Optional[str]:
    """Номер: буква на месте цифры читается цифрой, прочее — как есть."""
    result = []
    for char in part:
        if char.isdigit() or char in ".-/  ":
            result.append(char)
        elif char in _AS_DIGIT:
            result.append(_AS_DIGIT[char])
        else:
            # Буква, не похожая ни на одну цифру: это не номер.
            return None
    return "".join(result)


# ------------------------------------------------------------- закрытые списки
# Префиксы стандартов, ТУ и руководящих документов. Номер за префиксом по
# справочнику не проверяется — только по форме, иначе незнакомый стандарт
# нового документа остался бы испорченным.
STD_PREFIXES: Tuple[str, ...] = (
    "ГОСТ Р", "ГОСТ", "ОСТ", "СТО", "СТ ЦКБА", "СТБ", "ПНАЭ", "СНиП",
    "СанПиН", "ОТТ", "НП", "РД", "РБ", "ТУ", "ИСО", "ТР ТС", "ПБ", "ВСН",
)
# Код документа за обозначением ЕСКД (ГОСТ 2.102, ГОСТ 2.601). Однобуквенных
# кодов здесь нет: иначе союз «и» после обозначения становился кодом «И».
ESKD_KINDS: Tuple[str, ...] = (
    "СБ", "ВО", "ГЧ", "МЧ", "ТЧ", "ПЗ", "РР", "ТУ", "ТБ", "ПМ", "РЭ", "ПС",
    "ФО", "ЭТ", "ИМ", "УЧ", "МЭ", "ВС", "ВД", "ВП", "ПТ",
)
# Буквы шрифта по ГОСТ 26.008-85.
_FONT_KINDS: Tuple[str, ...] = ("Пр", "Шр")

# ------------------------------------------------------------------ справочник
# Код организации в обозначении ЕСКД. Собрано по корпусу прода с порогом
# частоты: код, встреченный единично, сам может быть порчей другого кода
# («ГАДУ», «ТАЛУ», «САДУ» — это испорченное «ТАДУ»), и справочником он не
# становится. Пополняется `tools/harvest_designations.py`.
ESKD_CODES: Tuple[str, ...] = ("ТАДУ",)
# Марки стали и сварочной проволоки: ГОСТ 5632, ГОСТ 2246 и то, что
# встречается в корпусе написанным верно. Список настоящих марок — не список
# ошибок: прочтение, которого здесь нет, не правится.
GRADES: Tuple[str, ...] = (
    "12Х18Н10Т", "08Х18Н10Т", "12Х18Н9Т", "08Х18Н10", "12Х18Н10",
    "10Х17Н13М2Т", "10Х17Н13М3Т", "08Х17Н13М2Т", "14Х17Н2", "20Х13",
    "12Х13", "40Х13", "95Х18", "15Х25Т", "08Х22Н6Т", "06ХН28МДТ",
    "03Х17Н14М3", "02Х18Н11", "12Х18Н12Т", "15Х1М1Ф", "12Х1МФ",
    "20Х23Н18", "10Х23Н18", "45Х14Н14В2М", "04Х19Н11М3",
    "Св-04Х19Н11М3", "Св-08Х19Н10Г2Б", "Св-07Х25Н13", "Св-06Х19Н9Т",
    "Св-04Х19Н9", "Св-08Х20Н9Г7Т", "Св-01Х19Н9", "Св-04Х19Н10М3",
    "Св-04Х19Н10М2", "Св-08Г2С", "Св-08А", "Св-10ГА",
)


class Change(NamedTuple):
    """Что и на что заменено; для отчёта и для проверки на корпусе."""

    pattern: str
    before: str
    after: str


_SPACE = r"[  ]"
_LOOSE_DIGIT = r"[0-9OОoо3Зз]"
_NOT_WORD = r"(?![0-9A-Za-zА-Яа-яЁё])"
_NOT_WORD_BEFORE = r"(?<![0-9A-Za-zА-Яа-яЁё])"

# Стандарт: префикс из списка, разделитель, номер. Номер обязан кончаться
# цифрой — точка в конце предложения частью обозначения не становится.
_STANDARD = re.compile(
    _NOT_WORD_BEFORE + r"(?P<prefix>" + _loose_any(STD_PREFIXES) + r")"
    r"(?P<sep>" + _SPACE + r"|[-.])?"
    r"(?P<number>" + _LOOSE_DIGIT + r"(?:[0-9OОoо3Зз.\-/]*" + _LOOSE_DIGIT + r")?)"
    + _NOT_WORD)
# Обозначение ЕСКД: код организации, номер, исполнение, код документа.
_ESKD = re.compile(
    _NOT_WORD_BEFORE + r"(?P<code>[А-ЯЁA-Za-zа-яё]{4})(?P<gap>" + _SPACE + r"?)"
    r"(?P<number>" + _LOOSE_DIGIT + r"{6}\." + _LOOSE_DIGIT + r"{3}"
    r"(?:\." + _LOOSE_DIGIT + r"{2})?)"
    r"(?:(?P<kindgap>" + _SPACE + r"?)(?P<kind>" + _loose_any(ESKD_KINDS) + r")"
    r"(?P<kindnum>" + _LOOSE_DIGIT + r")?)?" + _NOT_WORD)
# Марка стали или сварочной проволоки: две цифры и легирующие буквы. Первые
# две цифры обязаны быть цифрами уже в исходном тексте — иначе обычное слово
# («зоне», «озеро») читалось бы как марка.
_GRADE = re.compile(
    _NOT_WORD_BEFORE + r"(?P<grade>(?:" + _loose_word("Св") + r"-)?\d{2}"
    r"(?:[А-ЯЁа-яёA-Za-z]" + _LOOSE_DIGIT + r"{0,2}){2,6})" + _NOT_WORD)
# Шрифт по ГОСТ 26.008-85: высота, «Пр» или «Шр», номер.
_FONT = re.compile(
    _NOT_WORD_BEFORE + r"(?P<height>\d{1,2})-(?P<kind>" + _loose_any(_FONT_KINDS)
    + r")(?P<number>" + _LOOSE_DIGIT + r")" + _NOT_WORD)

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
_UPPER = re.compile(r"[А-ЯЁA-Z]")
_SEPARATOR = re.compile(r"[.\-/]")
# Правка делается только в русском тексте: в английском абзаце латинская
# «OCT» — это месяц, а не «ОСТ».
_MIN_CYRILLIC_SHARE = 0.6


def _truncatable(prefix: str) -> bool:
    """Префикс совпадает с хвостом другого префикса из списка."""
    return any(other != prefix and other.endswith(prefix)
               for other in STD_PREFIXES)


def _repair_standard(match: "re.Match[str]") -> Optional[str]:
    """«OCT 5582-75» -> «ОСТ 5582-75», «нП-о01-97» -> «НП-001-97»."""
    prefix = _only_reading(match.group("prefix"), STD_PREFIXES)
    number = _digits_only(match.group("number"))
    if prefix is None or number is None:
        return None
    if prefix != match.group("prefix") and _truncatable(prefix):
        # «ОСТ» — это и настоящий префикс, и «ГОСТ», у которого потерялась
        # первая буква. Латинское «OCT» на листе читается и так и так, а
        # «ОСТ 5582-75» вместо «ГОСТ 5582-75» — правдоподобная подмена
        # стандарта. Прочтений два, значит участок остаётся как был.
        return None
    # Ссылка на документ — это номер: не меньше трёх цифр или две цифры с
    # разделителем. Иначе «НПО» (объединение) читалось бы как «НП» с номером
    # «О», а «Ty2» — как «ТУ2».
    digits = sum(char.isdigit() for char in number)
    if digits < 3 and not (digits >= 2 and _SEPARATOR.search(number)):
        return None
    # Префикс, набранный целиком строчными, — это обычное слово, а не
    # ссылка: «сто 50-60 деталей», «ту 100-5», «рд 12-3». Разделитель в
    # номере тут ничего не доказывает: диапазон «50-60» выглядит как номер
    # стандарта, а «сто» перед диапазоном — обычная русская речь. Прописная
    # буква в префиксе нужна всегда.
    if not _UPPER.search(match.group("prefix")):
        return None
    return prefix + (match.group("sep") or "") + number


def _repair_eskd(match: "re.Match[str]") -> Optional[str]:
    """«ТАДУ 405210.001TY» -> «ТАДУ 405210.001ТУ», «687о00» -> «687000»."""
    number = _digits_only(match.group("number"))
    if number is None:
        return None
    # Код организации правится только до настоящего кода из справочника:
    # иначе «TAAY» стало бы несуществующим «ТААУ».
    code = _only_entry(match.group("code"), ESKD_CODES) or match.group("code")
    result = code + match.group("gap") + number
    if match.group("kind") is None:
        return result
    kind = _only_reading(match.group("kind"), ESKD_KINDS)
    if kind is None:
        return None
    tail = _digits_only(match.group("kindnum") or "")
    if tail is None:
        return None
    return result + match.group("kindgap") + kind + tail


def _repair_grade(match: "re.Match[str]") -> Optional[str]:
    """«12X18H10T» -> «12Х18Н10Т»; незнакомая марка остаётся как есть."""
    return _only_entry(match.group("grade"), GRADES)


def _repair_font(match: "re.Match[str]") -> Optional[str]:
    """«5-ПрЗ» -> «5-Пр3»."""
    kind = _only_reading(match.group("kind"), _FONT_KINDS)
    number = _digits_only(match.group("number"))
    if kind is None or number is None:
        return None
    return match.group("height") + "-" + kind + number


_PATTERNS = (
    ("standard", _STANDARD, _repair_standard),
    ("eskd", _ESKD, _repair_eskd),
    ("grade", _GRADE, _repair_grade),
    ("font", _FONT, _repair_font),
)


def repair_text(text: str) -> Tuple[str, List[Change]]:
    """Текст с восстановленными обозначениями и список сделанных замен."""
    if not text:
        return text, []
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    if not cyrillic + latin:
        return text, []
    if cyrillic / (cyrillic + latin) < _MIN_CYRILLIC_SHARE:
        return text, []
    changes: List[Change] = []
    result = text
    for name, pattern, repair in _PATTERNS:
        pieces: List[str] = []
        position = 0
        for match in pattern.finditer(result):
            span = match.group(0)
            repaired = repair(match)
            if repaired is None or repaired == span:
                continue
            pieces.append(result[position:match.start()])
            pieces.append(repaired)
            position = match.end()
            changes.append(Change(name, span, repaired))
        if pieces:
            pieces.append(result[position:])
            result = "".join(pieces)
    return result, changes


def repair_fragment(fragment) -> List[Change]:
    """Обозначения текстового фрагмента; сделанные замены — в payload."""
    text = fragment.content.text
    if not text:
        return []
    repaired, changes = repair_text(text)
    if not changes:
        return []
    fragment.content.text = repaired
    payload = fragment.content.structured_payload or {}
    payload[PAYLOAD_KEY] = [
        {"pattern": change.pattern, "before": change.before, "after": change.after}
        for change in changes
    ]
    fragment.content.structured_payload = payload
    return changes
