"""
Текст листа, переписанный Qwen-VL, исправляет абзацы распознавания того же листа.

Зачем. На скане чертежа MinerU и Tesseract читают чертёжный шрифт с ошибками,
которые меняют смысл: «Ra 1,6» превращается в «Ра 16», «M24x1,5» — в «M24x15».
Модель тот же лист читает верно, но её текст лежит одним фрагментом `drawing`,
а абзацы распознавания остаются рядом и спорят с ним в поиске.

Что делается (DRAWING_TRANSCRIPT_FIXES_OCR). Для каждого абзаца распознавания
на странице с переписанным текстом ищется подряд идущая группа строк модели,
похожая на абзац. Нашлась — текст абзаца заменяется этими строками, фрагмент
получает пометку `TEXT_FROM_VLM_TRANSCRIPT`. Не нашлась — абзац остаётся как
был: ничего не удаляется. Замена делается только тогда, когда строки модели
покрывают абзац целиком (та же длина) и цифры в них те же и в том же порядке:
«Ра 16» и «Ra 1,6» — один текст, а «ГОСТ 1005» и «ГОСТ 1009» — разные.
Если строки модели очень похожи на абзац, но цифры расходятся («Св-04х19н1им3»
против «Св-04Х19Н11М3»), кто из двоих прав — неизвестно: абзац остаётся, а
чтение модели дописывается к нему отдельной строкой с пометкой
`TEXT_VARIANT_FROM_VLM_TRANSCRIPT`. Расхождение при этом — не больше одной
цифры: соседний пункт с другим номером и другим ГОСТом вариантом не считается.
Таблицы не трогаются: их строение модель строками не передаёт.

Связи. Вызывается из `DrawingProcessor.enrich_fragments` после разбора листов.
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from core.models.contract import Fragment

WARNING = "TEXT_FROM_VLM_TRANSCRIPT"
VARIANT_WARNING = "TEXT_VARIANT_FROM_VLM_TRANSCRIPT"
# Пометка строки-варианта. Нейтральная: какое из двух чтений верно — неизвестно.
VARIANT_MARK = " (вариант чтения модели, требует сверки)"

# Методы распознавания, текст которых модель вправе исправить.
_OCR_METHODS = ("mineru_ocr", "ocr_layer")
# Абзац считается тем же текстом, что и строки модели, с такой похожести.
_MIN_RATIO = 0.72
# С расхождением в цифрах — только при такой похожести, и не заменой, а вариантом.
_VARIANT_RATIO = 0.85
# Вариант дописывается, только если строки модели той же длины, что и абзац.
_VARIANT_COVER = 0.93
# Абзац листа занимает не больше стольких строк подряд; длиннее — не трогается.
_MAX_LINES = 12
# Строки модели покрывают абзац целиком: их длина — в этих долях длины абзаца.
_MIN_COVER, _MAX_COVER = 0.85, 1.2
# Короче этого абзац не сравнивается: «А», «п.4», «2*» совпадут с чем угодно.
_MIN_CHARS = 12
_DIGITS = re.compile(r"\D+")

_SQUEEZE = re.compile(r"[\W_]+", re.UNICODE)
# Буквы, которые чертёжный шрифт делает неразличимыми: сравнение их не видит.
_LOOKALIKE = str.maketrans("abcehkmoptxyё", "авсенкмортхуе")


def apply_transcripts(fragments: List[Fragment]) -> List[Fragment]:
    """Абзацы распознавания на переписанных листах заменяются текстом модели."""
    transcripts = _page_transcripts(fragments)
    if not transcripts:
        return fragments
    result: List[Fragment] = []
    for fragment in fragments:
        page = fragment.position.page if fragment.position is not None else None
        lines = transcripts.get(page)
        if (not lines or fragment.type != "text"
                or fragment.provenance.method not in _OCR_METHODS):
            result.append(fragment)
            continue
        text = fragment.content.text or ""
        match = _best_run(text, lines)
        if match is not None:
            start, end = match
            fragment.content.text = " ".join(line for line, _ in lines[start:end])
            fragment.warnings = list(dict.fromkeys(fragment.warnings + [WARNING]))
        else:
            variant = _best_run(text, lines, same_digits=False)
            if variant is not None:
                start, end = variant
                reading = " ".join(line for line, _ in lines[start:end])
                fragment.content.text = f"{text}\n{reading}{VARIANT_MARK}"
                fragment.warnings = list(dict.fromkeys(fragment.warnings + [VARIANT_WARNING]))
        result.append(fragment)
    return result


def _page_transcripts(fragments: List[Fragment]) -> Dict[int, List[Tuple[str, str]]]:
    """Страница -> строки модели (как на листе, для сравнения)."""
    pages: Dict[int, List[Tuple[str, str]]] = {}
    for fragment in fragments:
        if fragment.type != "drawing" or fragment.position is None:
            continue
        payload = fragment.content.structured_payload or {}
        if not payload.get("transcript_line_count"):
            continue
        lines = [line.strip() for line in (fragment.content.text or "").split("\n")]
        pages.setdefault(int(fragment.position.page), []).extend(
            (line, _key(line)) for line in lines if line)
    return pages


def _key(text: str) -> str:
    return _SQUEEZE.sub("", text.lower().translate(_LOOKALIKE))


def _best_run(
    text: str, lines: List[Tuple[str, str]], same_digits: bool = True,
) -> Optional[Tuple[int, int]]:
    """Границы группы строк модели, больше всего похожей на абзац, или None."""
    target = _key(text)
    if len(target) < _MIN_CHARS:
        return None
    digits = _DIGITS.sub("", target)
    best, best_ratio = None, (_MIN_RATIO if same_digits else _VARIANT_RATIO)
    matcher = SequenceMatcher(autojunk=False)
    matcher.set_seq2(target)
    for start in range(len(lines)):
        joined = ""
        for end in range(start + 1, min(len(lines), start + _MAX_LINES) + 1):
            joined += lines[end - 1][1]
            if len(joined) > len(target) * _MAX_COVER + 4:
                break
            if len(joined) < len(target) * (_MIN_COVER if same_digits else _VARIANT_COVER) - 4:
                continue
            # Другое число — другой текст, как бы ни были похожи слова вокруг.
            found = _DIGITS.sub("", joined)
            if found != digits and (same_digits or not _one_digit_apart(found, digits)):
                continue
            matcher.set_seq1(joined)
            if matcher.real_quick_ratio() < best_ratio or matcher.quick_ratio() < best_ratio:
                continue
            ratio = matcher.ratio()
            if ratio > best_ratio:
                best, best_ratio = (start, end), ratio
    return best


def _one_digit_apart(first: str, second: str) -> bool:
    """Цифры двух текстов расходятся одной вставкой, пропуском или заменой."""
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) > len(second):
        first, second = second, first
    index = 0
    while index < len(first) and first[index] == second[index]:
        index += 1
    if len(first) == len(second):
        return first[index + 1:] == second[index + 1:]
    return first[index:] == second[index + 1:]
