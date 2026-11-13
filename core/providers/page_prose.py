"""
Страница текста в рамке ЕСКД: проза, а не чертёж и не таблица.

ПОК, СТО, ТУ, РЭ, ПМ, ТЗ, извещения и письма оформляются в рамке формата
со штампом — геометрически лист похож на чертёж (рамка во весь формат,
сетка граф в правом нижнем углу), а MinerU видит в рамке одну таблицу.
Из-за этого текст страницы уезжает в поля чертежа «Примечание: …
(требует сверки)», страница идёт в Qwen-VL, а многостраничная книга целиком
становится одной «таблицей».

Отличает такую страницу сам текст: на чертеже строки короткие — размеры,
допуски, выноски, — а здесь это предложения, разделы «1 Общие положения»
и номер «Лист N» в штампе. Модуль чистый: на входе строки текста, на
выходе вердикт и признаки, по которым он вынесен.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable

# Строка-предложение: столько слов и символов у строки прозы как минимум.
_PROSE_LINE_WORDS = 5
_PROSE_LINE_CHARS = 30
# Порог страницы: доля символов в строках-предложениях, их число, доля
# чертёжных токенов (размеры, допуски, резьбы, шероховатость).
_MIN_CHARS = 400
_PROSE_SHARE = 0.6
_PROSE_LINES = 8
# С заголовком раздела или «Лист N» достаточно меньшего.
_PROSE_SHARE_SUPPORTED = 0.45
_PROSE_LINES_SUPPORTED = 5
_MAX_DIMENSION_SHARE = 0.12
# «Лист N» есть в штампе и у чертежа, нумерованные пункты — и в его
# технических требованиях: с поддержкой этих признаков размеров должно
# быть заметно меньше.
_MAX_DIMENSION_SHARE_SUPPORTED = 0.08

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")
_NUMBER = r"\d+(?:[.,]\d+)?"
_DIMENSION_TOKEN = re.compile(
    r"^(?:"
    r"[ØøΦ⌀]\s*" + _NUMBER +                 # диаметр
    r"|R[az]?\s*" + _NUMBER +                # радиус, шероховатость
    r"|M\s*" + _NUMBER + r"(?:[x×]" + _NUMBER + r")?"  # резьба
    r"|±\s*" + _NUMBER + r"|" + _NUMBER + r"±" + _NUMBER +  # допуск
    r"|[+\-−]" + _NUMBER +                   # отклонение
    r"|" + _NUMBER + r"°"                     # угол
    r"|[HhJjKkNnPpSsEeFfGg][sS]?\d{1,2}"      # посадка
    r"|" + _NUMBER +                          # голый размер
    r"|[√∇]"
    r")$"
)
# «1 Общие положения», «4.2 Требования к упаковке», «2. Назначение».
_HEADING = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?\s+[А-ЯЁA-Z][а-яёa-z]{2,}")
_ITEM_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")
# «Лист 3», «Лист 12 из 65», «Стр. 4», «Sheet 7» в английских версиях.
_SHEET_NUMBER = re.compile(r"^(?:Лист|Стр\.?|Страница|Sheet|Page)\s*\d{1,4}\b", re.IGNORECASE)
_SHEET_LABEL = re.compile(r"^(?:Лист|Листов|Sheet|Sheets)$", re.IGNORECASE)


@dataclass
class ProseVerdict:
    is_prose: bool
    signals: Dict[str, Any] = field(default_factory=dict)


def assess(lines: Iterable[str]) -> ProseVerdict:
    """Строки текста страницы (или рамки) -> проза ли это."""
    texts = [" ".join(str(line).split()) for line in lines]
    texts = [t for t in texts if t]

    total_chars = 0
    prose_chars = 0
    prose_lines = 0
    tokens = 0
    dimension_tokens = 0
    heading = False
    sheet_number = False

    for index, text in enumerate(texts):
        chars = len(text.replace(" ", ""))
        total_chars += chars
        words = _WORD.findall(text)
        if len(words) >= _PROSE_LINE_WORDS and len(text) >= _PROSE_LINE_CHARS:
            prose_lines += 1
            prose_chars += chars
        for position, token in enumerate(text.split()):
            tokens += 1
            if position == 0 and len(words) >= 2 and _ITEM_NUMBER.match(token):
                # «1.1 Настоящее …» — номер пункта, а не размер.
                continue
            if _DIMENSION_TOKEN.match(token.strip(",;:()")):
                dimension_tokens += 1
        if not heading and len(text) <= 80 and _HEADING.match(text):
            heading = True
        if not sheet_number:
            if _SHEET_NUMBER.match(text):
                sheet_number = True
            elif _SHEET_LABEL.match(text) and index + 1 < len(texts) \
                    and texts[index + 1].isdigit():
                # В штампе «Лист» и номер — разные строки.
                sheet_number = True

    prose_share = prose_chars / total_chars if total_chars else 0.0
    dimension_share = dimension_tokens / tokens if tokens else 0.0
    signals = {
        "chars": total_chars,
        "prose_lines": prose_lines,
        "prose_share": round(prose_share, 3),
        "dimension_share": round(dimension_share, 3),
        "heading": heading,
        "sheet_number": sheet_number,
    }

    if total_chars < _MIN_CHARS or dimension_share > _MAX_DIMENSION_SHARE:
        return ProseVerdict(False, signals)
    plain = prose_share >= _PROSE_SHARE and prose_lines >= _PROSE_LINES
    supported = (
        (heading or sheet_number)
        and prose_share >= _PROSE_SHARE_SUPPORTED
        and prose_lines >= _PROSE_LINES_SUPPORTED
        and dimension_share <= _MAX_DIMENSION_SHARE_SUPPORTED
    )
    return ProseVerdict(plain or supported, signals)
