"""
Выправка слов распознанного текста растра (`OCR_WORD_FIX_ENABLED`).

Tesseract и модель таблиц MinerU читают русский рукописный шрифт ГОСТ
латиницей и путают похожие буквы: «Bbod разъемный» (Ввод), «3aum.» (Зам.),
«mun» (тип), «TY» (ТУ), «no» (по), «Шайда» (Шайба), «кадельный» (кабельный).
Такое слово не находится ни поиском, ни судьёй ответа при сверке цитаты.

Что делается со словом, и только со словом без цифр (обозначения, коды и
номера не трогаются никогда):

* латинское слово в русской строке переводится в кириллицу по похожим
  буквам, в том числе по курсиву («u» — «и», «m» — «т», «d» — «д»), если
  ровно одно прочтение есть в словаре. Английское слово в строке, где
  латиница заметна, остаётся;
* русское слово, которого нет в словаре, выправляется заменой одной буквы,
  которую курсив даёт вместо настоящей (д вместо б или в, б/в, и/п/н,
  у/ц, ш/щ), если такая замена даёт ровно одно словарное слово;
* короткое прочтение (до трёх букв) принимается только из списка частых
  слов и обозначений: в словаре моделей много коротких обрывков;
* одиночные «|», «_», «~» и подобные обрывки линий сетки убираются.

Словарь — списки слов `rus` и `eng` из моделей самого Tesseract
(`combine_tessdata`, `dawg2wordlist` из пакета tesseract-ocr), плюс
обозначения документации, которых в общем словаре нет. Ничего не
скачивается. Нет утилит или моделей — словарь пуст, и выправка по словарю
не делается (обрывки сетки убираются и без него).
"""

import functools
import threading
import itertools
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from celery.exceptions import SoftTimeLimitExceeded
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from core.providers import ocr_model

logger = logging.getLogger(__name__)

# Латинская (и греческая) буква -> кириллические прочтения, первое — самое
# вероятное. Курсив ГОСТ даёт «u» вместо «и», «m» вместо «т», «n» вместо
# «п», «g»/«d» вместо «д», «b» вместо «в»/«б»/«ь». На примечаниях чертежей
# ещё «z» вместо «г»/«з», «s» вместо «з», «n» вместо «л», «a» вместо
# «я», «x» вместо «к», «u»/«y» вместо «ч», «C» вместо «Д»: «Cozobopa» —
# «Договора», «Macca usdenua» — «Масса изделия», «MEXHONOZUYECKUX» —
# «ТЕХНОЛОГИЧЕСКИХ». Лишние прочтения стоят дальше по рангу и проходят
# только через словарь, как и остальные.
_LATIN_TO_CYR: Dict[str, Tuple[str, ...]] = {
    "a": ("а", "я"), "b": ("в", "б", "ь"), "c": ("с",), "d": ("д", "б"),
    "e": ("е",), "g": ("д",), "h": ("н",), "i": ("й", "і"), "k": ("к",),
    "m": ("м", "т"), "n": ("п", "л"), "o": ("о",), "p": ("р",), "r": ("г",),
    "s": ("з",), "t": ("т",), "u": ("и", "ц", "ч"), "x": ("х", "к"),
    "y": ("у", "ч"), "z": ("г", "з"),
    "A": ("А",), "B": ("В",), "C": ("С", "Д"), "D": ("Д",), "E": ("Е",),
    "H": ("Н",), "K": ("К",), "M": ("М", "Т"), "N": ("П", "Л", "И"),
    "O": ("О",), "P": ("Р",), "S": ("З",), "T": ("Т", "Г"), "U": ("И", "Ц"),
    "X": ("Х",), "Y": ("У", "Ч"), "Z": ("Г", "З"), "F": ("Г",), "ü": ("й",),
    "Γ": ("Г",), "π": ("п",), "Π": ("П",), "ο": ("о",), "Ο": ("О",),
    "α": ("а",), "ρ": ("р",), "Ρ": ("Р",), "τ": ("т",), "Τ": ("Т",),
    "κ": ("к",), "Κ": ("К",), "χ": ("х",), "Χ": ("Х",), "ε": ("е",),
    "Ε": ("Е",), "Β": ("В",), "Η": ("Н",), "Μ": ("М",), "Υ": ("У",),
    # Цифры, похожие на буквы, — только внутри слова из одних букв.
    "3": ("З",), "0": ("О",),
}
# Буква, которую курсив даёт вместо настоящей -> настоящие. Направление
# важно: «б» читается как «д» («Шайда», «кадельный»), обратного не бывает,
# и «Бирка» не должна становиться «Диркой».
_CYR_CONFUSION: Dict[str, str] = {
    "д": "бв", "б": "в", "в": "б", "и": "пн", "п": "и", "н": "и",
    "ц": "у", "у": "ц", "щ": "ш", "ш": "щ",
}

# Короткое слово (до трёх букв) принимается только из этого списка или из
# обозначений: в словаре моделей Tesseract много коротких обрывков («ог»,
# «ап», «цро»), и через них мусор становился бы «словами».
SHORT_WORDS: FrozenSet[str] = frozenset("""
по от на не но до за из ко со во же ли ни то та те ту он она оно они мы вы
их его ее её ей им ими как так что это все всё при для без над под или год
гг тип вид ом ком тех нет да мин час сут раз два три вес вых вх""".split())

# Обозначения документации и сокращения: в общем словаре их нет, а латиница
# на их месте («TY», «TADY», «TOCT») встречается постоянно. Выигрывают у
# общего словаря: «TOCT» — это «ГОСТ», а не «тост».
DOMAIN_WORDS: FrozenSet[str] = frozenset({
    "ту", "гост", "ост", "сто", "рэ", "пс", "сб", "тз", "кд", "отк", "пси",
    "вик", "аэс", "таду", "зип", "мпа", "кпа", "окп", "окпд", "тс", "тп",
    "нтл", "ип", "ооо", "ао", "пао", "зав", "зам", "изм", "подп", "лит",
    "инв", "взам", "разраб", "пров", "утв", "нконтр", "т.контр", "лист",
    "листов", "кол", "поз", "шт", "кг", "мм", "см", "мин", "макс", "заим",
})

_TOKEN_SPLIT_RE = re.compile(r"(\s+)")
# Слово внутри токена: окружающая пунктуация («(Ввод», «Зам.») остаётся.
_CORE_RE = re.compile(r"^([^\w]*)(.*?)([^\w]*)$", re.UNICODE)
# Обрывки линий сетки: «|», «||», «_», «—_», «~», «^».
# Тильда («~ 220 В») и длинное поле бланка («______») — не обрывки.
_GRID_JUNK_RE = re.compile(r"^(?:[|¦_—–\-]*[|¦][|¦_—–\-]*|_{1,2}|—_|_—)$")

_MAX_VARIANTS = 4096
# Слово из букв; цифры 3 и 0 допускаются как буквы («3aum.»).
_WORDLIKE_RE = re.compile(r"(?:[^\W\d_]|[30])+", re.UNICODE)
# «No 651» в русском тексте — знак номера.
_NUMERO = frozenset({"No", "NO", "N°", "Nº", "Ne"})


def _script(ch: str) -> str:
    if "Ѐ" <= ch <= "ӿ":
        return "cyr"
    if "Ͱ" <= ch <= "Ͽ":
        return "grk"
    if ch.isascii() and ch.isalpha():
        return "lat"
    return ""


@dataclass
class Vocabulary:
    rus: FrozenSet[str] = frozenset()
    eng: FrozenSet[str] = frozenset()

    def is_russian(self, word: str) -> bool:
        low = word.lower()
        return low in DOMAIN_WORDS or low in self.rus

    def is_english(self, word: str) -> bool:
        return word.lower() in self.eng

    @property
    def empty(self) -> bool:
        return not self.rus


@dataclass
class CleanupStats:
    latin_words: int = 0
    confusion_words: int = 0
    junk_tokens: int = 0

    def add(self, other: "CleanupStats") -> None:
        self.latin_words += other.latin_words
        self.confusion_words += other.confusion_words
        self.junk_tokens += other.junk_tokens

    @property
    def total(self) -> int:
        return self.latin_words + self.confusion_words + self.junk_tokens

    def as_dict(self) -> Dict[str, int]:
        return {"latin_words": self.latin_words, "confusion_words": self.confusion_words,
                "junk_tokens": self.junk_tokens}


# ---------------------------------------------------------------- словарь
def _tessdata_dir() -> Optional[str]:
    # Выбранный каталог моделей идёт первым: словарь выправки должен быть
    # от той же модели, которой прочитан лист, иначе выправка сверяется с
    # чужим словарём.
    candidates = [ocr_model.tessdata_dir() or "", os.environ.get("TESSDATA_PREFIX") or ""]
    candidates += [
        "/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata",
        "/usr/share/tessdata", "/usr/local/share/tessdata",
    ]
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "rus.traineddata")):
            return path
    return None


def _wordlist(tessdata: str, lang: str, workdir: str) -> FrozenSet[str]:
    """Список слов модели Tesseract `lang` (нижний регистр)."""
    prefix = os.path.join(workdir, f"{lang}.")
    subprocess.run(
        ["combine_tessdata", "-u", os.path.join(tessdata, f"{lang}.traineddata"), prefix],
        check=True, capture_output=True, timeout=60,
    )
    out = os.path.join(workdir, f"{lang}.words")
    subprocess.run(
        ["dawg2wordlist", f"{prefix}lstm-unicharset", f"{prefix}lstm-word-dawg", out],
        check=True, capture_output=True, timeout=60,
    )
    with open(out, encoding="utf-8", errors="ignore") as handle:
        return frozenset(line.strip().lower() for line in handle if line.strip())


_VOCAB_LOCK = threading.Lock()


def load_vocabulary() -> Vocabulary:
    """
    Словарь из моделей Tesseract; читается один раз на процесс (около
    секунды). Сбой не фатален: пустой словарь выключает выправку по словарю.
    Страницы читаются в нескольких потоках — словарь собирает один.
    """
    with _VOCAB_LOCK:
        return _load_vocabulary()


@functools.lru_cache(maxsize=1)
def _load_vocabulary() -> Vocabulary:
    tessdata = _tessdata_dir()
    if not tessdata or not shutil.which("combine_tessdata") or not shutil.which("dawg2wordlist"):
        logger.warning("Словарь для выправки OCR недоступен: нет моделей или утилит Tesseract")
        return Vocabulary()
    try:
        workdir = tempfile.mkdtemp(prefix="ocr-vocab-")
    except OSError as exc:
        logger.warning("Словарь для выправки OCR не собрался: %s", exc)
        return Vocabulary()
    try:
        words = {}
        for lang in ("rus", "eng"):
            if not os.path.isfile(os.path.join(tessdata, f"{lang}.traineddata")):
                words[lang] = frozenset()
                continue
            try:
                words[lang] = _wordlist(tessdata, lang, workdir)
            except SoftTimeLimitExceeded:
                raise
            except Exception as exc:  # noqa: BLE001 — словарь не обязателен
                logger.warning("Словарь %s для выправки OCR не собрался: %s", lang, exc)
                words[lang] = frozenset()
        logger.info("Словарь выправки OCR: rus %d, eng %d слов",
                    len(words["rus"]), len(words["eng"]))
        return Vocabulary(rus=words["rus"], eng=words["eng"])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------- слово
def _case_like(word: str, original: str) -> str:
    # «3» и «0» на месте буквы — заглавные («3aum.» — «Заим.»).
    letters = [("A" if ch in "30" else ch) for ch in original if ch.isalpha() or ch in "30"]
    if len(letters) > 1 and all(ch.isupper() for ch in letters):
        return word.upper()
    # Обозначение, в котором заглавных не меньше половины («ГОСm»):
    # курсивная строчная латиница в нём — та же заглавная.
    if (word.lower() in DOMAIN_WORDS and len(letters) > 1
            and sum(ch.isupper() for ch in letters) * 2 >= len(letters)):
        return word.upper()
    if letters and letters[0].isupper():
        return word[:1].upper() + word[1:].lower()
    return word.lower()


def _latin_variants(core: str) -> Optional[List[Tuple[int, str]]]:
    """(ранг, кириллическое прочтение); None — есть буква без прочтения."""
    options: List[Tuple[Tuple[int, str], ...]] = []
    count = 1
    for ch in core:
        if _script(ch) == "cyr":
            options.append(((0, ch),))
            continue
        mapped = _LATIN_TO_CYR.get(ch)
        if not mapped:
            return None
        options.append(tuple(enumerate(mapped)))
        count *= len(mapped)
        if count > _MAX_VARIANTS:
            return None
    return [
        (sum(rank for rank, _ in combo), "".join(ch for _, ch in combo))
        for combo in itertools.product(*options)
    ]


def _acceptable(word: str, vocab: Vocabulary) -> bool:
    low = word.lower()
    if len(low) <= 3:
        return low in SHORT_WORDS or low in DOMAIN_WORDS
    return vocab.is_russian(low)


def _best(candidates: Iterable[Tuple[int, str]], vocab: Vocabulary) -> Optional[str]:
    """
    Словарное прочтение с наименьшим рангом. Обозначения документации
    старше общего словаря. Два разных слова на одном ранге — не выбираем.
    """
    found = [(rank, word) for rank, word in candidates if _acceptable(word, vocab)]
    if not found:
        return None
    domain = [(rank, word) for rank, word in found if word.lower() in DOMAIN_WORDS]
    pool = domain or found
    best_rank = min(rank for rank, _ in pool)
    words = {word.lower() for rank, word in pool if rank == best_rank}
    if len(words) != 1:
        return None
    return next(word for rank, word in pool if rank == best_rank)


def _fix_latin(core: str, vocab: Vocabulary, cyr_share: float) -> Optional[str]:
    if core in _NUMERO:
        return "№"
    if sum(1 for ch in core if ch.isalpha()) < 2:
        return None
    if not _WORDLIKE_RE.fullmatch(core):
        return None
    if all(_script(ch) == "cyr" for ch in core if ch.isalpha()):
        return None
    # Английское слово в строке, где латиница не случайна, остаётся.
    if all(_script(ch) == "lat" for ch in core) and vocab.is_english(core) and (
        cyr_share < 0.6 or len(core) >= 4
    ):
        return None
    variants = _latin_variants(core)
    if not variants:
        return None
    best = _best(variants, vocab)
    if best is None:
        return None
    return _case_like(best, core)


def _fix_confusion(core: str, vocab: Vocabulary, first: bool = True) -> Optional[str]:
    if len(core) < 4 or not all(_script(ch) == "cyr" for ch in core):
        return None
    # Слово с заглавной посреди текста — скорее имя собственное («Дубна»),
    # которого в словаре нет по праву.
    if not first and core[0].isupper() and not core.isupper():
        return None
    if vocab.is_russian(core):
        return None
    found = set()
    low = core.lower()
    for index, ch in enumerate(low):
        for other in _CYR_CONFUSION.get(ch, ""):
            candidate = low[:index] + other + low[index + 1:]
            if vocab.is_russian(candidate):
                found.add(candidate)
    if len(found) != 1:
        return None
    fixed = found.pop()
    # Регистр — побуквенно по исходному слову.
    return "".join(f.upper() if o.isupper() else f for f, o in zip(fixed, core))


def _upper_designation(core: str) -> Optional[str]:
    """«ГОСт» -> «ГОСТ»: обозначение, набранное заглавными, кроме курсивной буквы."""
    if core.lower() not in DOMAIN_WORDS or core.isupper() or len(core) < 3:
        return None
    if sum(ch.isupper() for ch in core) * 2 < len(core) or not core[0].isupper():
        return None
    return core.upper()


def cyrillic_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if _script(ch) == "cyr") / len(letters)


def clean_text(text: str, vocab: Vocabulary, cyr_share: Optional[float] = None) -> Tuple[str, CleanupStats]:
    """
    Текст с выправленными словами. `cyr_share` — доля кириллицы в контексте
    (для короткой ячейки — по всей таблице); по умолчанию — по самому тексту.
    Переводы строк и пробелы сохраняются.
    """
    stats = CleanupStats()
    if not text:
        return text, stats
    share = cyrillic_share(text) if cyr_share is None else cyr_share
    parts = _TOKEN_SPLIT_RE.split(text)
    out: List[str] = []
    sentence_start = True
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        if _GRID_JUNK_RE.match(part):
            stats.junk_tokens += 1
            # Пробел перед обрывком уходит вместе с ним; перевод строки — нет.
            if out and out[-1].isspace():
                tail_space = out.pop()
                if "\n" in tail_space:
                    out.append(tail_space[:tail_space.rindex("\n") + 1])
            continue
        first = sentence_start or (bool(out) and "\n" in out[-1])
        sentence_start = part[-1] in ".:;!?"
        if vocab.empty or "\\" in part or "$" in part:
            # LaTeX-разметка формул MinerU («$\\beta$») не трогается.
            out.append(part)
            continue
        match = _CORE_RE.match(part)
        lead, core, tail = match.group(1), match.group(2), match.group(3)
        fixed = None
        if core and share >= 0.5:
            fixed = _fix_latin(core, vocab, share)
            if fixed is not None:
                stats.latin_words += 1
            elif not tail.startswith("-"):
                # «изде-» — половина слова до переноса, словаря для неё нет.
                fixed = _fix_confusion(core, vocab, first) or _upper_designation(core)
                if fixed is not None:
                    stats.confusion_words += 1
        out.append(lead + (fixed if fixed is not None else core) + tail)
    cleaned = "".join(out)
    if stats.junk_tokens:
        # Обрывок в начале текста или строки оставлял пробел перед словом.
        cleaned = re.sub(r"(^|\n)[ \t]+", r"\1", cleaned)
    return cleaned, stats


# ------------------------------------------------------------- применение
def clean_layer(layer: Any, vocab: Vocabulary) -> Tuple[Any, CleanupStats]:
    """Слой распознавания (`TextLayer`) с выправленными строками; пустые строки уходят."""
    stats = CleanupStats()
    pages: Dict[int, List[Any]] = {}
    for page, lines in layer.pages.items():
        kept = []
        for line in lines:
            text, line_stats = clean_text(line.text, vocab)
            stats.add(line_stats)
            if not text.strip():
                continue
            parts = getattr(line, "parts", ())
            if parts:
                # Куски строки (развороты в две колонки) — с теми же правками.
                parts = tuple(
                    replace(part, text=fixed) for part in parts
                    for fixed in [clean_text(part.text, vocab)[0]] if fixed.strip()
                )
                kept.append(replace(line, text=text, parts=parts))
            else:
                kept.append(replace(line, text=text) if text != line.text else line)
        pages[page] = kept
    return replace(layer, pages=pages), stats


def clean_blocks(blocks: Sequence[Any], vocab: Vocabulary) -> CleanupStats:
    """
    Текст блоков и ячейки таблиц — на месте. Контекст ячейки — вся таблица:
    по одной короткой ячейке («TY», «mun») алфавит не определить.
    """
    stats = CleanupStats()
    for block in blocks:
        if block.type == "text" and block.text:
            block.text, block_stats = clean_text(block.text, vocab)
            stats.add(block_stats)
        elif block.type == "table" and block.table_data:
            table = block.table_data
            cells = [str(h) for h in (table.get("headers") or [])]
            for row in table.get("rows") or []:
                cells.extend(str(c) for c in row)
            share = cyrillic_share(" ".join(cells))

            def fix(value: Any, count: bool = True) -> Any:
                if not isinstance(value, str) or not value.strip():
                    return value
                fixed, cell_stats = clean_text(value, vocab, share)
                if count:
                    stats.add(cell_stats)
                return fixed

            table["headers"] = [fix(h) for h in (table.get("headers") or [])]
            table["rows"] = [[fix(c) for c in row] for row in (table.get("rows") or [])]
            block.table_data = table
            if getattr(block, "table_html", None):
                # Исходный HTML уходит в RAG рядом с ячейками — те же правки.
                block.table_html = _HTML_TEXT_RE.sub(
                    lambda m: ">" + fix(m.group(1), count=False) + "<", block.table_html
                )
    return stats


_HTML_TEXT_RE = re.compile(r">([^<>]+)<")
