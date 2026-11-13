"""
Устаревшие записи листа, который модель прочитала целиком.

Зачем. Когда весь текст листа переписан Qwen-VL, рядом с верным текстом
модели в том же документе остаются записи распознавания того же листа:
таблица MinerU со штампом, где масштаб «7:1» вместо «1:1», таблица сальника
с «M24×15» вместо «M24x1,5», абзацы Tesseract из одних обрывков («< - o a ni
aera = pian HM 2XIBHIOT [OCT»). В поиске они спорят с чтением модели и портят
ответ. Правка абзацев (`core/drawing_transcript.py`) переписывает только текст
абзаца: таблицы она не трогает и ничего не убирает.

Что делается (DRAWING_SUPPRESS_STALE_ON_FULL_READ). Для каждой таблицы и
каждого абзаца распознавания на листе с переписанным текстом решается одно из
трёх:

* **убрать как повтор** — всё содержимое блока есть у модели: каждая его
  значимая лексема нашлась среди лексем модели в том же написании и с теми же
  цифрами («Ra 6,3» у MinerU и «Ra 6,3» у модели);
* **убрать как шум распознавания** — в блоке нет ни одного русского слова и ни
  одного правильно записанного значения, которых нет у модели, а почти все его
  лексемы — обрывки (одиночные латинские буквы, латиница вперемешку с
  кириллицей, «слова» без гласных). Только на русском листе, текст которого
  переписан целиком и без недочитанных частей;
* **оставить** — во всех остальных случаях. Лексема, которой у модели нет,
  оставляет блок на месте всегда: модель могла её не прочитать.

Если цифры расходятся («7:1» против «1:1», «M24×15» против «M24x1,5»),
остаются оба чтения: блок сохраняется, а чтение модели кладётся рядом —
в `structured_payload.vlm_transcript_variants` с предупреждением
`TEXT_VARIANT_FROM_VLM_TRANSCRIPT`, тем же, что у правки абзацев
(`DRAWING_TRANSCRIPT_FIXES_OCR`). Какое из двух чтений верно, разбор не решает.

Чтение модели — это переписанный текст листа плюс её структурные данные:
поля `drawing`, графы штампа и строки прочитанных ею таблиц. Из них масштаб
«1:1» виден даже тогда, когда в переписанном тексте его нет.

Что убрано — видно у фрагмента листа: `structured_payload.stale_sources`
(список `dropped` и `marked`) и предупреждение `STALE_SOURCES_SUPPRESSED`.

Связи. Вызывается из `DrawingProcessor.enrich_fragments` после правки абзацев.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from core.drawing_transcript import VARIANT_WARNING, digits_one_apart
from core.models.contract import Fragment

WARNING = "STALE_SOURCES_SUPPRESSED"
PAYLOAD_KEY = "stale_sources"
VARIANTS_KEY = "vlm_transcript_variants"
REASON_DUPLICATE = "duplicate"
REASON_NOISE = "ocr_noise"

# Методы, записи которых считаются устаревшими рядом с чтением модели: это
# распознавание растра. Текстовый слой (`text_layer`) и таблицы, прочитанные
# самой моделью (`vlm_escalation`), сюда не входят.
_STALE_TEXT_METHODS = ("mineru_ocr", "ocr_layer", "ocr_layer_recovered", "tesseract")
_STALE_TABLE_METHODS = ("mineru_table", "ocr_layer", "ocr_layer_recovered", "tesseract")

# Лексема: буквы и цифры, склеенные разделителями значения («1:1», «6,3»,
# «М24х1,5», «687000.001.001», «2246-70»).
_TOKEN = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[.,:/\-×][A-Za-zА-Яа-яЁё0-9]+)*")
# Буквы, которые чертёжный шрифт делает неразличимыми: сравнение их не видит.
_LOOKALIKE = str.maketrans("abcehkmoptxyё", "авсенкмортхуе")
_VOWELS = set("аеиоуыэюяё")
# Знак умножения резьбы приводится к той же букве, что латинская «x».
_TIMES = {"×": "х", "х": "х"}

# Повтор убирается только у блока хотя бы из столькых лексем.
_MIN_DUPLICATE_TOKENS = 2
# Шум убирается только у блока хотя бы из столькых лексем: «Ra 6,3» шумом не бывает.
_MIN_NOISE_TOKENS = 4
# И только если обрывков в нём не меньше этой доли.
_MIN_NOISE_SHARE = 0.6
# Лист считается русским, если среди букв его переписанного текста столько кириллицы.
_MIN_RUSSIAN_SHARE = 0.6
# Слово — от столькых букв: «см», «по» совпадут с чем угодно.
_MIN_WORD_LETTERS = 3
# У значения не больше столькых букв: «М24х1,5», «Ra 1,6», «20°».
_MAX_VALUE_LETTERS = 3
# И цифры — не меньше этой доли его букв и цифр.
_MIN_VALUE_DIGIT_SHARE = 0.4
# Расхождение в цифрах ищется у лексемы хотя бы с этим числом цифр.
_MIN_VARIANT_DIGITS = 2


def suppress_stale_sources(fragments: List[Fragment]) -> List[Fragment]:
    """Убирает повторы и шум распознавания на листах с переписанным текстом."""
    sheets = _sheets_by_page(fragments)
    if not sheets:
        return fragments
    result: List[Fragment] = []
    for fragment in fragments:
        page = fragment.position.page if fragment.position is not None else None
        sheet = sheets.get(page)
        if sheet is None or not _is_stale(fragment):
            result.append(fragment)
            continue
        verdict, detail = _decide(fragment, sheet)
        if verdict == "drop":
            _note(sheet, "dropped", fragment, detail)
            continue
        if verdict == "mark":
            _mark(fragment, detail)
            _note(sheet, "marked", fragment, detail)
        result.append(fragment)
    return result


class _Sheet:
    """Чтение листа моделью: лексемы и то, переписан ли лист целиком."""

    def __init__(self, fragment: Fragment):
        self.fragment = fragment
        payload = fragment.content.structured_payload or {}
        self.complete = payload.get("transcript_complete") is not False
        transcript = fragment.content.text or ""
        self.tokens: Dict[str, str] = {}
        for text in [transcript, *_structured_texts(fragment)]:
            for token in _tokens(text):
                self.tokens.setdefault(_key(token), token)
        self.russian = _russian(transcript)
        self.has_tables = bool(payload.get("tables"))


def _sheets_by_page(fragments: List[Fragment]) -> Dict[int, _Sheet]:
    """Страница -> чтение её листа моделью, если лист прочитан целиком."""
    sheets: Dict[int, _Sheet] = {}
    for fragment in fragments:
        if fragment.type != "drawing" or fragment.position is None:
            continue
        payload = fragment.content.structured_payload or {}
        if not payload.get("transcript_line_count") or not (fragment.content.text or "").strip():
            continue
        page = int(fragment.position.page)
        if page not in sheets:
            sheets[page] = _Sheet(fragment)
    return sheets


def _structured_texts(fragment: Fragment) -> List[str]:
    """Структурные данные модели: поля, графы штампа, строки её таблиц."""
    texts: List[str] = []
    for field in fragment.content.structured_drawing_fields or []:
        texts.append(f"{field.value} {field.tolerance or ''}")
    payload = fragment.content.structured_payload or {}
    for name, value in (payload.get("title_block") or {}).items():
        texts.append(f"{name} {value}")
    for table in payload.get("tables") or []:
        for row in table.get("rows") or []:
            cells = row.values() if isinstance(row, dict) else row
            texts.extend(str(cell) for cell in (cells or []))
    return texts


def _is_stale(fragment: Fragment) -> bool:
    method = fragment.provenance.method
    if fragment.type == "text":
        return method in _STALE_TEXT_METHODS
    if fragment.type == "table":
        return method in _STALE_TABLE_METHODS
    return False


def _decide(fragment: Fragment, sheet: _Sheet) -> Tuple[str, Dict[str, Any]]:
    """keep | drop | mark для одного блока распознавания."""
    tokens = _tokens(_block_text(fragment))
    if not tokens:
        return "keep", {}
    unique: List[str] = []
    variants: List[Dict[str, str]] = []
    kinds: List[str] = []
    for token in tokens:
        kind = _kind(token)
        kinds.append(kind)
        if _key(token) in sheet.tokens:
            continue
        model = _variant(token, sheet.tokens)
        if model is not None:
            variants.append({"block": token, "model": model})
            continue
        if kind != "garbage":
            unique.append(token)
    if variants:
        # Цифры расходятся: кто прав — неизвестно, остаются оба чтения.
        return "mark", {"reason": "digits_differ", "variants": variants}
    if unique:
        return "keep", {}
    meaningful = [kind for kind in kinds if kind != "garbage"]
    if len(tokens) >= _MIN_DUPLICATE_TOKENS and meaningful:
        if fragment.type == "table" and not sheet.has_tables:
            # Строение таблицы строками текста не передаётся: без своей
            # таблицы модели убирать таблицу нельзя.
            return "keep", {}
        return "drop", {"reason": REASON_DUPLICATE}
    garbage = len(kinds) - len(meaningful)
    if (fragment.type == "text" and sheet.complete and sheet.russian
            and len(tokens) >= _MIN_NOISE_TOKENS
            and garbage / len(kinds) >= _MIN_NOISE_SHARE):
        return "drop", {"reason": REASON_NOISE}
    return "keep", {}


def _block_text(fragment: Fragment) -> str:
    """Весь текст блока: у таблицы — шапка и ячейки."""
    parts = [fragment.content.text or ""]
    table = fragment.content.table_data
    if table is not None:
        parts.extend(str(header) for header in table.headers)
        for row in table.rows:
            parts.extend(str(cell) for cell in row)
        for value in (table.total_row or {}).values():
            parts.append(str(value))
    return "\n".join(parts)


def _mark(fragment: Fragment, detail: Dict[str, Any]) -> None:
    """Чтение модели кладётся рядом с блоком, блок остаётся как был."""
    payload = dict(fragment.content.structured_payload or {})
    existing = list(payload.get(VARIANTS_KEY) or [])
    for variant in detail.get("variants") or []:
        if variant not in existing:
            existing.append(variant)
    payload[VARIANTS_KEY] = existing
    fragment.content.structured_payload = payload
    fragment.warnings = list(dict.fromkeys(fragment.warnings + [VARIANT_WARNING]))
    fragment.review_required = True


def _note(sheet: _Sheet, kind: str, fragment: Fragment, detail: Dict[str, Any]) -> None:
    """Запись о решении у фрагмента листа: что убрано и почему."""
    payload = dict(sheet.fragment.content.structured_payload or {})
    notes = dict(payload.get(PAYLOAD_KEY) or {})
    record: Dict[str, Any] = {
        "fragment_id": fragment.fragment_id,
        "type": fragment.type,
        "method": fragment.provenance.method,
        "reason": detail.get("reason", ""),
    }
    if detail.get("variants"):
        record["variants"] = detail["variants"]
    notes.setdefault(kind, []).append(record)
    payload[PAYLOAD_KEY] = notes
    sheet.fragment.content.structured_payload = payload
    sheet.fragment.warnings = list(dict.fromkeys(sheet.fragment.warnings + [WARNING]))


# ------------------------------------------------------------------ лексемы
def _tokens(text: str) -> List[str]:
    """Лексемы как в документе: сравнение приводит их к виду `_key`."""
    return _TOKEN.findall(text)


def _key(token: str) -> str:
    """
    Написание лексемы для сравнения: регистр не важен, запятая дроби и точка —
    одно и то же, знак умножения — та же буква, что «x», неразличимые буквы
    чертёжного шрифта сведены к кириллице.
    """
    lowered = token.lower()
    for sign, replacement in _TIMES.items():
        lowered = lowered.replace(sign, replacement)
    return lowered.translate(_LOOKALIKE).replace(",", ".")


def _letters(token: str) -> str:
    return "".join(char for char in _key(token) if char.isalpha())


def _digits(token: str) -> str:
    return "".join(char for char in token if char.isdigit())


def _mask(token: str) -> str:
    """Форма записи: «7:1» и «1:1» — одна форма, «16» и «1,6» — разные."""
    return re.sub(r"\d", "N", _key(token))


def _kind(token: str) -> str:
    """
    word — русское слово, value — записанное значение, garbage — обрывок.

    Алфавит берётся как в документе: латинское «[OCT» — обрывок «ГОСТ», а не
    русское слово, хотя для сравнения его буквы и сводятся к кириллице.
    """
    lowered = token.lower()
    letters = "".join(char for char in lowered if char.isalpha())
    digits = _digits(token)
    cyrillic = sum(1 for char in letters if "а" <= char <= "я" or char == "ё")
    if (len(letters) >= _MIN_WORD_LETTERS and cyrillic == len(letters)
            and any(char in _VOWELS for char in letters)):
        return "word"
    if (digits and len(letters) <= _MAX_VALUE_LETTERS
            and cyrillic in (0, len(letters))
            and len(digits) / (len(letters) + len(digits)) >= _MIN_VALUE_DIGIT_SHARE):
        return "value"
    return "garbage"


def _variant(token: str, model: Dict[str, str]) -> Optional[str]:
    """
    Чтение модели того же значения с другими цифрами, иначе None.

    Сравниваются только лексемы с одинаковыми буквами: «м24х15» и «м24х1,5».
    Расхождение — либо в разделителе при тех же цифрах («16» и «1,6»), либо
    в одной цифре при той же форме записи («7:1» и «1:1»). Запись без
    разделителя («001» и «002») вариантом не считается: это разные значения.
    """
    letters, digits, mask = _letters(token), _digits(token), _mask(token)
    if len(digits) < _MIN_VARIANT_DIGITS:
        return None
    separated = any(char in ".,:/-×" for char in token)
    for other in model.values():
        if _letters(other) != letters or _key(other) == _key(token):
            continue
        other_digits = _digits(other)
        if other_digits == digits:
            return other
        if ((separated or letters) and _mask(other) == mask
                and digits_one_apart(other_digits, digits)):
            return other
    return None


def _russian(text: str) -> bool:
    letters = [char for char in text.lower() if char.isalpha()]
    if not letters:
        return False
    cyrillic = sum(1 for char in letters if "а" <= char <= "я")
    return cyrillic / len(letters) >= _MIN_RUSSIAN_SHARE
