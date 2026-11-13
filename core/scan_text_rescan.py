"""
Страница плохого скана перечитывается моделью, а не распознаванием
(`DRAWING_RESCAN_POOR_OCR_PAGES`).

Зачем. На скане чертежа обычное распознавание читает чертёжный шрифт ГОСТ
вперемешку с латиницей, и в индекс уезжает испорченное требование:
«Гайки поз, 6 затянуть 00 упора дез применения инструмента» вместо «затянуть
до упора без применения инструмента». Проверено на тех же документах: лист,
который прочитала модель проходом «весь текст листа», выходит чистым, а
соседний лист того же файла, оставшийся за распознаванием, — испорченным.
Значит, бумага читаемая, и портит её именно путь распознавания.

Что делается. Страница, у которой распознавание само себе ставит низкую
оценку или у которой текст несёт признак путаницы алфавитов, уходит в тот
же проход «весь текст листа», которым уже читаются листы при
`DRAWING_IMAGE_FULL_READ` и `DRAWING_PDF_FULL_READ`; её абзацы правятся
текстом модели существующим шагом `core/drawing_transcript.py`. Новой ветки
нет: меняется только то, какие страницы попадают в этот проход.

Признак выбран измеримый, по самой странице, а не по списку файлов:

* оценка страницы — средняя уверенность записей распознавания, взвешенная
  по числу символов: короткая подпись с уверенностью 0,9 не перевешивает
  абзац требований с 0,5;
* признак путаницы алфавитов — доля слов, в которых кириллица и латиница
  стоят рядом («Mapxupobame», «wmyyepax», «Россиц» рядом с «AI»), и слов,
  целиком набранных латиницей, на странице, где язык — русский. Верно
  прочитанная латиница («IP 67», «For NPP», «M8», «Ra 1,6») словом не
  считается: слово — это буквы без цифр, от трёх знаков.

Любого из двух признаков достаточно: порча бывает и при высокой оценке
распознавания, а низкая оценка бывает и без порчи (бледный лист).

Чего здесь нет. Ни одного имени файла и ни одного образца порчи: порог —
свойство страницы, поэтому незнакомый скан проходит его так же, как
знакомый.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, NamedTuple, Sequence

# Записи распознавания: их текст и оценку модуль и оценивает.
OCR_METHODS = ("ocr_layer", "mineru_ocr", "tesseract")

# Слово: только буквы, от трёх знаков. Короче — обрывок, с цифрой — код.
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_LATIN = re.compile(r"[A-Za-z]")
# Страница считается русской с такой долей кириллицы среди букв.
_RUSSIAN_SHARE = 0.6
# Страница короче этого не оценивается: на двух словах доли ничего не значат.
_MIN_CHARS = 60


class PageText(NamedTuple):
    """Запись распознавания страницы: оценка и текст."""

    confidence: float
    text: str


@dataclass
class Verdict:
    """Решение по странице и признаки, по которым оно вынесено."""

    rescan: bool = False
    score: float = 1.0
    mixed_share: float = 0.0
    chars: int = 0
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"rescan": self.rescan, "score": round(self.score, 3),
                "mixed_share": round(self.mixed_share, 3),
                "chars": self.chars, "reasons": list(self.reasons)}


def corruption_share(text: str) -> float:
    """Доля слов с путаницей алфавитов среди слов страницы."""
    words = _WORD.findall(text)
    if not words:
        return 0.0
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(_LATIN.findall(text))
    russian = bool(cyrillic + latin) and cyrillic / (cyrillic + latin) >= _RUSSIAN_SHARE
    bad = 0
    for word in words:
        has_cyrillic = bool(_CYRILLIC.search(word))
        has_latin = bool(_LATIN.search(word))
        if has_cyrillic and has_latin:
            bad += 1
        elif has_latin and russian:
            # Русская страница, слово целиком латиницей: либо чужое слово,
            # либо прочитанное не тем алфавитом. На русском чертеже второе
            # встречается несравнимо чаще.
            bad += 1
    return bad / len(words)


def page_verdict(
    items: Sequence[PageText],
    score_threshold: float = 0.8,
    mixed_threshold: float = 0.12,
) -> Verdict:
    """Нужно ли перечитать страницу моделью, и почему."""
    texts = [item.text or "" for item in items]
    chars = sum(len(text.strip()) for text in texts)
    verdict = Verdict(chars=chars)
    if chars < _MIN_CHARS:
        return verdict
    weight = sum(max(len((item.text or "").strip()), 1) for item in items)
    verdict.score = sum(
        item.confidence * max(len((item.text or "").strip()), 1) for item in items
    ) / weight
    verdict.mixed_share = corruption_share("\n".join(texts))
    if verdict.score < score_threshold:
        verdict.reasons.append("ocr_score")
    if verdict.mixed_share > mixed_threshold:
        verdict.reasons.append("script_mix")
    verdict.rescan = bool(verdict.reasons)
    return verdict


def poor_pages(
    fragments: Iterable,
    score_threshold: float = 0.8,
    mixed_threshold: float = 0.12,
) -> Dict[int, Verdict]:
    """Страницы документа, которые стоит перечитать моделью."""
    pages: Dict[int, List[PageText]] = {}
    for fragment in fragments:
        if fragment.type != "text" or fragment.position is None:
            continue
        if fragment.provenance.method not in OCR_METHODS:
            continue
        pages.setdefault(int(fragment.position.page), []).append(
            PageText(float(fragment.confidence or 0.0), fragment.content.text or "")
        )
    verdicts = {}
    for page, items in sorted(pages.items()):
        verdict = page_verdict(items, score_threshold, mixed_threshold)
        if verdict.rescan:
            verdicts[page] = verdict
    return verdicts
