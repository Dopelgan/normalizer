"""
Правка чертёжного текста по закрытым словарям (флаг DRAWING_LEXICON_FIX).

Зачем. Есть надписи, у которых набор верных значений закрыт: фамилии
подписантов штампа, номера стандартов, слова предметной области и марки
материалов. Модель читает их с ошибками, которые на листе не видны
человеку, но ломают поиск: фамилия с одной-двумя чужими буквами («Пестрякин»
вместо «Пеструхин»), одна фамилия в двух написаниях на разных листах,
«ГОСТ 16.037-80» вместо «ГОСТ 16037-80», «Сдувка» вместо «Сдвижка». Хуже того: закорючку подписи модель читает фамилией, и в
графе «Подп. и дата» появляется человек, которого на листе нет.

Что делается.

* **Подписанты** (`DRAWING_LEXICON_SIGNERS_PATH`). Фамилия в графах
  «Разраб.», «Пров.», «Т.контр.», «Н.контр.», «Утв.» сравнивается со
  списком по взвешенному расстоянию правки: замена похожих на листе знаков
  («а»/«о», «и»/«н», «б»/«в», «Г»/«Е») дешевле обычной, цифра не
  становится другой цифрой. Фамилия ставится к ближайшей из списка, только
  если расхождение мало относительно длины и ближайшая одна с запасом.
  Иначе прочтение остаётся как было, а поле получает `unverified_signer`.
* **Подпись** — графы «Подп.», «Подп. и дата» и рукописное в графе
  подписанта. Здесь фамилию модель чаще придумывает по закорючке, поэтому
  слово, которое ни на кого из списка не похоже (или похоже на двоих), из
  значения убирается: выдуманная фамилия хуже отсутствующей.
* **Стандарты.** «ГОСТ 16.037-80» -> «ГОСТ 16037-80»: точка внутри номера
  из четырёх-пяти цифр снимается, если число до точки — не номер системы
  стандартов с точкой (ЕСКД 2.xxx, ЕСТПП 14.xxx и др.). При заданном
  `DRAWING_LEXICON_GOST_PATH` решает список: номер из списка не трогается,
  а номер не из списка правится только до номера из списка.
* **Термины и марки** (`DRAWING_LEXICON_TERMS_PATH`) — в графах штампа и
  материала: слово ставится к ближайшему термину на тех же условиях, что и
  фамилия, но не короче пяти знаков и с разницей длины не больше одного.

Прочтение до правки не теряется: каждая правка, отказ и снятая фамилия —
в `completeness_mechanisms["drawing_diagnostics"]["lexicon_fixes"]`
(`kind`, `before`, `after`, `where`). Не в `structured_payload`: оттуда
адаптер RAG выводит ключи строкой в индекс, и снятая выдуманная фамилия
вернулась бы в поиск через учёт.

Модуль на верхнем уровне импортирует только стандартную библиотеку: его
функции сравнения берёт сборщик списка подписантов
(`tools/build_signer_lexicon.py`), который запускается без пакета `core`.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

# Ключ учёта в `completeness_mechanisms["drawing_diagnostics"]`: что правлено
# и что прочитано до правки.
NOTES_KEY = "lexicon_fixes"
_DIAGNOSTICS = "drawing_diagnostics"

# Виды записей в `lexicon_fixes`.
SIGNER = "signer"            # фамилия поставлена к фамилии из списка
UNVERIFIED = "unverified"    # фамилия оставлена, но со списком не сошлась
DROPPED = "dropped"          # прочтение подписи снято: фамилии нет в списке
TERM = "term"                # слово поставлено к термину из списка
GOST = "gost"                # номер стандарта

# ===========================================================================
# Сравнение написаний
# ===========================================================================

# Латиница, неотличимая на листе от кириллицы, сравнивается как кириллица.
_TO_CYRILLIC = str.maketrans("ABCEHKMOPTXYacekopxy", "АВСЕНКМОРТХУасекорху")

# Пары знаков, которые модель путает на чертёжном шрифте и в рукописи;
# сравнение — в строчных. Замена внутри пары стоит `_CHEAP` вместо 1.
_CONFUSED = (
    "ао", "ин", "ип", "нп", "бв", "ге", "гт", "ес", "ео", "лп", "шщ", "цщ",
    "ий", "кх", "ьъ", "ьб", "зэ", "ая", "уч", "цу", "ли", "мл",
    # Буква на месте цифры и наоборот — только для неразличимых знаков.
    "о0", "з3", "б6", "ч4",
)
_CHEAP = 0.4
_PAIRS = frozenset(frozenset(pair) for pair in _CONFUSED)
# «Бесконечная» цена: цифра не становится другой цифрой, не появляется и не
# исчезает — номер в марке или стандарте правкой не меняется никогда.
_NEVER = 1e6


def fold(word: str) -> str:
    """Написание для сравнения: кириллица, строчные, «ё» как «е»."""
    return str(word).translate(_TO_CYRILLIC).lower().replace("ё", "е")


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in _PAIRS:
        return _CHEAP
    if a.isdigit() or b.isdigit():
        return _NEVER
    return 1.0


def _gap_cost(char: str) -> float:
    return _NEVER if char.isdigit() else 1.0


def distance(read: str, known: str) -> float:
    """
    Взвешенное расстояние правки между прочтением и записью из списка.

    Вставка и удаление — 1, замена похожих на листе знаков — `_CHEAP`,
    прочая замена — 1; цифра не заменяется другой цифрой, не вставляется и не
    удаляется (цена `_NEVER`).
    """
    a, b = fold(read), fold(known)
    previous = [0.0]
    for char in b:
        previous.append(previous[-1] + _gap_cost(char))
    for x in a:
        current = [previous[0] + _gap_cost(x)]
        for j, y in enumerate(b, start=1):
            current.append(min(
                previous[j] + _gap_cost(x),
                current[j - 1] + _gap_cost(y),
                previous[j - 1] + _sub_cost(x, y),
            ))
        previous = current
    return previous[-1]


# Допуск расхождения: доля длины записи из списка, но не меньше одной правки.
SIGNER_RATIO = 0.3
TERM_RATIO = 0.45
# Ближайшая запись должна быть ближе второй хотя бы на столько.
MARGIN = 1.0
# Термин: не короче и с разницей длины не больше.
_TERM_MIN_CHARS = 5
_TERM_LENGTH_SLACK = 1


def limit(known: str, ratio: float) -> float:
    return max(1.0, ratio * len(fold(known)))


def nearest(read: str, entries: Sequence[str], ratio: float = SIGNER_RATIO,
            length_slack: Optional[int] = None) -> Tuple[Optional[str], str]:
    """
    Запись из списка для прочтения и исход сравнения.

    Исходы: `exact` — прочтение и есть запись (с точностью до регистра,
    «ё» и латиницы-двойника); `snapped` — одна ближайшая запись в допуске и
    с запасом `MARGIN` до второй; `ambiguous` — в допуске, но запаса нет;
    `none` — ничего в допуске. Запись возвращается только при `exact` и
    `snapped`.
    """
    folded = fold(read)
    scored: List[Tuple[float, str]] = []
    for entry in entries:
        if fold(entry) == folded:
            return entry, "exact"
        if length_slack is not None and abs(len(fold(entry)) - len(folded)) > length_slack:
            continue
        # Слишком разная длина в допуск не уложится всё равно.
        if abs(len(fold(entry)) - len(folded)) > limit(entry, ratio):
            continue
        scored.append((distance(read, entry), entry))
    scored.sort(key=lambda item: (item[0], item[1]))
    if not scored or scored[0][0] > limit(scored[0][1], ratio):
        return None, "none"
    best_cost, best = scored[0]
    # Две записи на одном расстоянии с разным написанием — спор: какая из
    # них имелась в виду, решать нечем.
    rivals = [entry for cost, entry in scored[1:]
              if fold(entry) != fold(best) and cost - best_cost < MARGIN]
    if rivals:
        return None, "ambiguous"
    return best, "snapped"


# ===========================================================================
# Фамилии в значении графы
# ===========================================================================

# Фамилия: слово с прописной, не короче трёх букв, может быть двойной через
# дефис. Инициалы («И.И.») сюда не попадают. Латиница-двойник допускается —
# так модель и пишет «Иванoв».
_NAME = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])"
    r"[А-ЯЁA-Z][а-яёa-zА-ЯЁA-Z]{2,}(?:-[А-ЯЁA-Z][а-яёa-zА-ЯЁA-Z]+)?"
    r"(?![0-9A-Za-zА-Яа-яЁё])"
)
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")

# Слова графы, которые фамилией не являются (сравнение по `_label_fold`).
_NOT_NAMES = frozenset({
    "разраб", "разработал", "пров", "проверил", "тконтр", "нконтр", "утв",
    "утвердил", "подп", "подпись", "подписано", "дата", "рукописно", "копировал",
    "изм", "лист", "листов", "докум", "формат", "инв", "подл", "дубл", "взам",
    "перв", "примен", "справ", "масса", "масштаб", "лит", "кол", "уч", "контр",
    "нормоконтроль", "техконтроль", "исполнитель", "нач", "отд", "гл", "инж",
    "конструктор", "технолог", "согласовано", "утверждаю", "нет", "пусто",
})


def _label_fold(name: str) -> str:
    """Подпись графы без украшений — как `title_block._fold`."""
    return re.sub(r"[\s.,:№\-_]+", "", str(name).strip().lower()).replace("ё", "е")


def names_in(value: str) -> List[Tuple[int, int, str]]:
    """Фамилии в значении графы: (начало, конец, слово)."""
    found = []
    for match in _NAME.finditer(str(value or "")):
        word = match.group(0)
        if not _CYRILLIC.search(word) or _label_fold(word) in _NOT_NAMES:
            continue
        found.append((match.start(), match.end(), word))
    return found


def surname(line: str) -> str:
    """Фамилия из строки списка: «Иванов И.И.» -> «Иванов»."""
    found = names_in(line)
    return found[0][2] if found else ""


def _case_like(target: str, read: str) -> str:
    """Запись списка в регистре прочтения: «ИВАНОВ» -> прописными."""
    letters = [char for char in read if char.isalpha()]
    if letters and all(char.isupper() for char in letters) and len(letters) > 1:
        return target.upper()
    if letters and letters[0].islower():
        return target[:1].lower() + target[1:]
    if letters and letters[0].isupper():
        return target[:1].upper() + target[1:]
    return target


# ===========================================================================
# Графы
# ===========================================================================

# Графы подписантов основной надписи (сравнение по `_label_fold`).
ROLE_LABELS = frozenset({
    "разраб", "разработал", "пров", "проверил", "тконтр", "техконтроль",
    "нконтр", "нормоконтроль", "утв", "утвердил",
})
# Графы подписи: здесь прочтение — чаще закорючка, чем напечатанная фамилия.
SIGNATURE_LABELS = frozenset({"подп", "подпись", "подпидата", "подписьидата"})
# Ключи `title_block` с фамилиями подписантов.
ROLE_KEYS = ("developed", "checked", "tech_control", "norm_control", "approved")

_HAND_MARK = re.compile(r"\s*\((?:рукописно|от руки)\)\s*$", re.IGNORECASE)
_EMPTY_LEFT = re.compile(r"^[\s,;.:\-–—|/]*$")


def _split(value: str) -> Tuple[str, str, str]:
    """«Разраб.: Иванов (рукописно)» -> («Разраб.», «Иванов», « (рукописно)»)."""
    text = str(value or "")
    head, sep, rest = text.partition(":")
    if not sep:
        head, rest = "", text
    mark = _HAND_MARK.search(rest)
    tail = mark.group(0) if mark else ""
    body = rest[: mark.start()] if mark else rest
    return head, body, tail


def _join(head: str, body: str, tail: str) -> str:
    return f"{head}: {body.strip()}{tail}" if head else f"{body.strip()}{tail}"


class Lexicon:
    """Списки, по которым идёт правка. Пустой список — его правки нет."""

    def __init__(self, signers: Iterable[str] = (), terms: Iterable[str] = (),
                 gost: Iterable[str] = ()) -> None:
        names = [surname(line) for line in signers]
        self.signers: Tuple[str, ...] = tuple(dict.fromkeys(n for n in names if n))
        self.terms: Tuple[str, ...] = tuple(dict.fromkeys(
            t.strip() for t in terms if len(fold(t.strip())) >= _TERM_MIN_CHARS))
        self.gost = frozenset(_gost_key(entry) for entry in gost if _gost_key(entry))


# Инициалы за фамилией: «И.И.», «И. И.», «И.».
_INITIALS = re.compile(r"\s*[А-ЯЁA-Z]\.(?:\s?[А-ЯЁA-Z]\.)?")


def fix_signers(value: str, signers: Sequence[str], signature: bool,
                ) -> Tuple[str, bool, List[Dict[str, str]]]:
    """
    Фамилии в значении графы подписанта -> (новое значение, не сверено ли,
    записи учёта).

    `signature` — графа подписи или рукописное прочтение: слово, не
    сошедшееся со списком, из значения снимается. В печатной графе оно
    остаётся, но значение помечается несверенным.
    """
    if not signers:
        return value, False, []
    notes: List[Dict[str, str]] = []
    unverified = False
    pieces: List[str] = []
    position = 0
    for start, end, word in names_in(value):
        found, outcome = nearest(word, signers, SIGNER_RATIO)
        if found is not None:
            replacement = _case_like(found, word) if outcome == "snapped" else word
            if replacement != word:
                notes.append({"kind": SIGNER, "before": word, "after": replacement})
        else:
            unverified = True
            if signature:
                # Вместе с фамилией — и инициалы за ней: они прочитаны с той
                # же закорючки.
                initials = _INITIALS.match(value, end)
                if initials:
                    word, end = word + initials.group(0), initials.end()
                replacement = ""
                notes.append({"kind": DROPPED, "before": word.strip(), "after": "",
                              "reason": outcome})
            else:
                replacement = word
                notes.append({"kind": UNVERIFIED, "before": word, "after": word,
                              "reason": outcome})
        pieces.append(value[position:start])
        pieces.append(replacement)
        position = end
    pieces.append(value[position:])
    fixed = "".join(pieces)
    if fixed != value:
        fixed = " ".join(fixed.split())
    return fixed, unverified, notes


# Слово термина: буквы и цифры, части через дефис («Св-08Г2С»).
_WORD = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])[0-9A-Za-zА-Яа-яЁё]+(?:-[0-9A-Za-zА-Яа-яЁё]+)*"
    r"(?![0-9A-Za-zА-Яа-яЁё])")


def fix_terms(value: str, terms: Sequence[str]) -> Tuple[str, List[Dict[str, str]]]:
    """Слова и марки значения -> ближайший термин из списка (см. `nearest`)."""
    if not terms:
        return value, []
    notes: List[Dict[str, str]] = []

    def repair(match: "re.Match[str]") -> str:
        word = match.group(0)
        if len(fold(word)) < _TERM_MIN_CHARS or not _CYRILLIC.search(word):
            return word
        found, outcome = nearest(word, terms, TERM_RATIO, _TERM_LENGTH_SLACK)
        if found is None or outcome != "snapped":
            return word
        replacement = _case_like(found, word) if not any(c.isdigit() for c in found) else found
        if replacement != word:
            notes.append({"kind": TERM, "before": word, "after": replacement})
        return replacement

    return _WORD.sub(repair, value), notes


# ===========================================================================
# Стандарты
# ===========================================================================

# Системы стандартов, номер которых пишется с точкой: «ГОСТ 2.104-2006»
# (ЕСКД), «ГОСТ 14.004-83» (ЕСТПП), «ГОСТ 12.1.005-88» (ССБТ) и другие.
# Такой номер по форме не правится — только по списку.
DOTTED_SERIES = frozenset({
    "1", "2", "3", "4", "6", "7", "8", "9", "12", "14", "15", "19", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "34",
})
_GOST = re.compile(
    r"(?<![0-9A-Za-zА-Яа-яЁё])(?P<prefix>ГОСТ(?:[  ]?Р)?|ОСТ)(?P<sep>[  ]?)"
    r"(?P<number>\d{1,2}[.,][  ]?\d{2,3}|\d{4,5})"
    r"(?P<year>[-–]\d{2}(?:\d{2})?)(?![0-9])")


def _gost_key(text: str) -> str:
    """Запись стандарта для сравнения со списком: без пробелов, «–» как «-»."""
    match = _GOST.search(str(text or ""))
    if not match:
        return ""
    prefix = re.sub(r"[  ]", "", match.group("prefix"))
    number = re.sub(r"[  ]", "", match.group("number")).replace(",", ".")
    return f"{prefix} {number}{match.group('year').replace('–', '-')}"


def _dotted_is_real(series: str, rest: str) -> bool:
    """Номер с точкой похож на номер системы стандартов («2.104», «14.004»)."""
    return series in DOTTED_SERIES and len(rest) == 3


def fix_gost(text: str, known: frozenset = frozenset()) -> Tuple[str, List[Dict[str, str]]]:
    """Номера стандартов в тексте; со списком — только до номера из списка."""
    notes: List[Dict[str, str]] = []

    def repair(match: "re.Match[str]") -> str:
        span = match.group(0)
        number = re.sub(r"[  ]", "", match.group("number"))
        if "." not in number and "," not in number:
            return span
        series, _, rest = number.replace(",", ".").partition(".")
        joined = series + rest
        if not 4 <= len(joined) <= 5:
            return span
        prefix = match.group("prefix")
        fixed = f"{prefix}{match.group('sep')}{joined}{match.group('year')}"
        if known:
            if _gost_key(span) in known or _gost_key(fixed) not in known:
                return span
        elif prefix == "ОСТ" or _dotted_is_real(series, rest):
            # Отраслевой номер с точкой и номер системы — обычная запись.
            return span
        notes.append({"kind": GOST, "before": span, "after": fixed})
        return fixed

    return _GOST.sub(repair, str(text or "")), notes


# ===========================================================================
# Списки из файлов и ключ кэша
# ===========================================================================

def _read_list(path: str) -> Tuple[str, ...]:
    if not path:
        return ()
    # Тот же разбор файла и кэш по времени изменения, что у справочника
    # названий исполнений: строка — запись, «#» — комментарий.
    from core.providers.designation_repair import known_names

    return known_names(path)


def load(signers_path: str = "", terms_path: str = "", gost_path: str = "") -> Lexicon:
    return Lexicon(_read_list(signers_path), _read_list(terms_path), _read_list(gost_path))


def from_settings() -> Lexicon:
    from core.config import settings

    return load(settings.DRAWING_LEXICON_SIGNERS_PATH, settings.DRAWING_LEXICON_TERMS_PATH,
                settings.DRAWING_LEXICON_GOST_PATH)


def digest(paths: Sequence[str]) -> str:
    """
    Отпечаток содержимого списков — для ключа кэша результата. Пустой путь и
    отсутствующий файл дают свою метку: появление файла меняет ключ.
    """
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(b"\0")
        if not path:
            hasher.update(b"-")
            continue
        try:
            with open(path, "rb") as handle:
                hasher.update(handle.read().replace(b"\r\n", b"\n"))
        except OSError:
            hasher.update(b"missing")
    return hasher.hexdigest()


def settings_digest() -> Optional[str]:
    """Отпечаток для ключа кэша; `None` — флаг выключен, ключ не меняется."""
    from core.config import settings

    if not settings.DRAWING_LEXICON_FIX:
        return None
    return digest((settings.DRAWING_LEXICON_SIGNERS_PATH, settings.DRAWING_LEXICON_TERMS_PATH,
                   settings.DRAWING_LEXICON_GOST_PATH))


# ===========================================================================
# Фрагменты
# ===========================================================================

def _fix_value(value: str, kind: str, handwritten: bool, category: str,
               lexicon: Lexicon) -> Tuple[str, bool, List[Dict[str, str]]]:
    """Одно значение: фамилии, стандарты, термины. Возвращает и пометку."""
    notes: List[Dict[str, str]] = []
    unverified = False
    if kind in ("role", "signature"):
        value, unverified, found = fix_signers(
            value, lexicon.signers, signature=kind == "signature" or handwritten)
        notes += found
    value, found = fix_gost(value, lexicon.gost)
    notes += found
    if kind == "other" and category in ("title_block", "material"):
        value, found = fix_terms(value, lexicon.terms)
        notes += found
    return value, unverified, notes


def _kind(label: str) -> str:
    folded = _label_fold(label)
    if folded in ROLE_LABELS:
        return "role"
    if folded in SIGNATURE_LABELS:
        return "signature"
    return "other"


# Подписи одной графы: «Пров.» и «Проверил» — одна и та же графа.
_LABEL_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("разраб", "разработал"), ("пров", "проверил"), ("тконтр", "техконтроль"),
    ("нконтр", "нормоконтроль"), ("утв", "утвердил"), tuple(sorted(SIGNATURE_LABELS)),
)
# Ключи `title_block` и `title_block_handwritten` -> подписи их граф.
_KEY_LABELS: Dict[str, Tuple[str, ...]] = {
    "developed": _LABEL_GROUPS[0], "checked": _LABEL_GROUPS[1], "tech_control": _LABEL_GROUPS[2],
    "norm_control": _LABEL_GROUPS[3], "approved": _LABEL_GROUPS[4], "signed_date": _LABEL_GROUPS[5],
}
_LEFT = r"(?<![0-9A-Za-zА-Яа-яЁё])"
_RIGHT = r"(?![0-9A-Za-zА-Яа-яЁё])"
_LABEL_SEP = r"[\s.,:;№\-_|]"
# Правки слов, которые переносятся в текст листа и в прочтения поля.
_WORD_KINDS = (SIGNER, TERM, DROPPED)

Located = Tuple[Dict[str, str], Set[str], str]


def _words(note: Dict[str, str]) -> List[str]:
    """
    Что искать: само прочтение, а у снятой подписи — и фамилию без
    инициалов (другой проход мог прочитать её без них).
    """
    found = [note["before"]]
    if note["kind"] == DROPPED:
        bare = surname(note["before"])
        if bare and bare != note["before"]:
            found.append(bare)
    return found


def _label_group(folded: str) -> Tuple[str, ...]:
    for group in _LABEL_GROUPS:
        if folded in group:
            return group
    return (folded,) if folded else ()


def _label_pattern(folded: str) -> str:
    """Подпись графы в тексте: те же буквы, между ними точки и пробелы («Т. контр.»)."""
    chars = ["[её]" if char == "е" else re.escape(char) for char in folded]
    return "(?i:" + (_LABEL_SEP + "*").join(chars) + ")"


def _labels_before(value: str, word: str) -> Set[str]:
    """Подписи граф, за которыми слово стоит в самом значении («Разраб. Иванов»)."""
    found: Set[str] = set()
    for label in ROLE_LABELS | SIGNATURE_LABELS:
        pattern = _LEFT + _label_pattern(label) + _LABEL_SEP + "*" + re.escape(word) + _RIGHT
        if re.search(pattern, value):
            found.update(_label_group(label))
    return found


def _context(note: Dict[str, str], labels: Iterable[str], line: str) -> Located:
    return note, set(labels) | _labels_before(line, note["before"]), line


def _replace_at(text: str, spans: Iterable[Tuple[int, int]], after: str) -> str:
    """Замена по местам с конца; снятое слово уносит и пробелы за собой."""
    for start, end in sorted(set(spans), reverse=True):
        if after:
            text = text[:start] + after + text[end:]
            continue
        stop = end
        while stop < len(text) and text[stop] in " \t":
            stop += 1
        if stop == len(text) or text[stop] == "\n":
            while start > 0 and text[start - 1] in " \t":
                start -= 1
        text = text[:start] + text[stop:]
    return text


def _replace_words(text: str, located: Sequence[Located]) -> str:
    """
    Те же замены слов в тексте листа: запись и поля называют одинаково.

    Слово меняется только там, где оно стоит за подписью своей графы
    («Пров. Зорубякин») или в строке, совпадающей с прочтением поля целиком.
    По всему тексту не меняется: то же слово в другом месте листа может
    быть верным. Не нашлось такого места — текст не трогается.
    """
    for note, labels, line in located:
        if note["kind"] not in _WORD_KINDS or note["before"] == note["after"]:
            continue
        # Строка из одного слова — не место поля, если у поля есть подпись графы.
        by_line = bool(line.strip()) and (not labels or line.strip() != note["before"])
        for before in _words(note):
            word = re.escape(before)
            spans: List[Tuple[int, int]] = []
            for label in labels:
                pattern = _LEFT + _label_pattern(label) + _LABEL_SEP + "*(" + word + ")" + _RIGHT
                spans += [match.span(1) for match in re.finditer(pattern, text)]
            if by_line:
                whole = r"^[ \t]*" + re.escape(line.strip()) + r"[ \t]*$"
                for row in re.finditer(whole, text, re.MULTILINE):
                    spans += [(row.start() + match.start(), row.start() + match.end())
                              for match in re.finditer(_LEFT + word + _RIGHT, row.group(0))]
            if spans:
                text = _replace_at(text, spans, note["after"])
    return text


def _fix_readings(field: Any, found: Sequence[Dict[str, str]]) -> None:
    """
    Прочтения поля (`readings`, `alternatives`) — с той же правкой, что и
    значение: прежнее неверное прочтение остаётся только в учёте
    (`completeness_mechanisms`), иначе оно вернулось бы в индекс.
    """
    def fix(value: str) -> str:
        changed = False
        for note in found:
            if note["kind"] not in _WORD_KINDS or note["before"] == note["after"]:
                continue
            for before in _words(note):
                spans = [m.span() for m in re.finditer(_LEFT + re.escape(before) + _RIGHT, value)]
                if spans:
                    value, changed = _replace_at(value, spans, note["after"]), True
        return " ".join(value.split()) if changed else value

    if field.readings:
        readings = []
        for reading in field.readings:
            value = reading.get("value") if isinstance(reading, dict) else None
            if not isinstance(value, str):
                readings.append(reading)
                continue
            fixed = fix(value)
            if not _EMPTY_LEFT.match(fixed):
                readings.append({**reading, "value": fixed})
        field.readings = readings
    if field.alternatives:
        alternatives = [fix(value) if isinstance(value, str) else value for value in field.alternatives]
        field.alternatives = list(dict.fromkeys(
            value for value in alternatives if not (isinstance(value, str) and _EMPTY_LEFT.match(value))))


def apply_fragment(fragment: Any, lexicon: Lexicon) -> Dict[str, int]:
    """Чертёжный фрагмент на месте: поля, графы штампа, текст листа."""
    stats = {"changed": 0, "unverified": 0, "dropped": 0}
    content = fragment.content
    payload = content.structured_payload if isinstance(content.structured_payload, dict) else {}
    notes: List[Dict[str, str]] = []
    # Правки с местом, где их искать в тексте листа (`_replace_words`).
    located: List[Located] = []

    kept = []
    for field in content.structured_drawing_fields or []:
        head, body, tail = _split(field.value)
        kind = _kind(head) if head else "other"
        fixed, unverified, found = _fix_value(body, kind, bool(tail), field.category, lexicon)
        labels = _label_group(_label_fold(head)) if head else ()
        for note in found:
            note["where"] = head or field.category
            located.append(_context(note, labels, field.value))
        notes += found
        if unverified:
            field.unverified_signer = True
            stats["unverified"] += 1
        if fixed != body:
            if kind == "signature" and _EMPTY_LEFT.match(fixed):
                # От значения ничего не осталось — графа была одной фамилией
                # по закорючке. Поля нет: прочтение — в учёте.
                stats["dropped"] += 1
                continue
            field.value = _join(head, fixed, tail)
            stats["changed"] += 1
        if found:
            _fix_readings(field, found)
        kept.append(field)
    if content.structured_drawing_fields is not None:
        content.structured_drawing_fields = kept

    block = payload.get("title_block")
    if isinstance(block, dict):
        for key, value in list(block.items()):
            if not isinstance(value, str):
                continue
            kind = "role" if key in ROLE_KEYS else "other"
            category = "material" if key == "material" else "title_block"
            fixed, _, found = _fix_value(value, kind, False, category, lexicon)
            for note in found:
                note["where"] = f"title_block.{key}"
                located.append(_context(note, _KEY_LABELS.get(key, ()), value))
            notes += found
            block[key] = fixed
    hand = payload.get("title_block_handwritten")
    if isinstance(hand, dict):
        for key, value in list(hand.items()):
            if not isinstance(value, str):
                continue
            kind = "role" if key in ROLE_KEYS else (
                "signature" if key == "signed_date" else "other")
            fixed, _, found = _fix_value(value, kind, True, "title_block", lexicon)
            for note in found:
                note["where"] = f"title_block_handwritten.{key}"
                located.append(_context(note, _KEY_LABELS.get(key, ()), value))
            notes += found
            if kind == "signature" and _EMPTY_LEFT.match(fixed):
                del hand[key]
            else:
                hand[key] = fixed

    if content.text:
        text, found = fix_gost(content.text, lexicon.gost)
        for note in found:
            note["where"] = "text"
        notes += found
        content.text = _replace_words(text, located)

    if notes:
        diagnostics = dict(fragment.completeness_mechanisms.get(_DIAGNOSTICS) or {})
        diagnostics[NOTES_KEY] = list(diagnostics.get(NOTES_KEY) or []) + notes
        fragment.completeness_mechanisms[_DIAGNOSTICS] = diagnostics
    return stats


def apply(fragments: Sequence[Any], lexicon: Optional[Lexicon] = None) -> Dict[str, int]:
    """Все чертёжные фрагменты документа; правка на месте, итог — счётчики."""
    lexicon = lexicon if lexicon is not None else from_settings()
    total = {"changed": 0, "unverified": 0, "dropped": 0}
    for fragment in fragments:
        if getattr(fragment, "type", None) != "drawing":
            continue
        for key, count in apply_fragment(fragment, lexicon).items():
            total[key] += count
    return total
