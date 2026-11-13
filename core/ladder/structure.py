"""
Чем чертёжный разбор отличается от сплошного текста.

Спор чертёжной ветки и распознавания решался объёмом: у кого больше символов,
тот и прав. Объём — не та величина, по которой эти два способа различаются.
Распознавание выдаёт по чертежу строки вида `AL 10 SF Хх`: символов много,
смысла ноль. Чертёжный разбор выдаёт обозначение, графы штампа, надписи видов
и размеры: символов мало, и каждый — величина, по которой деталь ищут.

Поэтому здесь объём не считается вовсе. Считается то, из чего чертёжный разбор
состоит по ЕСКД:

* поля листа (`DrawingField.category`): размеры, допуски, резьба, радиусы,
  шероховатость, позиции, графы штампа;
* штамп — сам по себе, по полям `title_block` или по графам в тексте листа;
* обозначение — код по ЕСКД (`СМ1638.11.05.00.00.00`, `301421.007.00`);
* надписи видов и разрезов (`Г-Г`, `В-В`, `Вид сверху`, `Разрез А-А`);
* размерные величины (`⌀8,5`, `R12`, `M10`, `2 отв.`, `110`, `±0,2`).

Признаки не привязаны ни к одному документу: лист, который никто не смотрел,
описывается ими так же, как уже разобранный. Величины считаются и по полям, и
по тексту листа — модель кладёт одну и ту же надпись то в поле, то в строку
переписанного текста, и признак не должен зависеть от того, куда она попала.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List

# Категории полей, каждая из которых сама по себе — измеренная величина листа.
# `title_block` сюда не входит: штамп считается отдельным признаком.
MEASURED_CATEGORIES = frozenset(
    {"size", "tolerance", "radius", "thread", "roughness", "position", "callout"}
)

TITLE_BLOCK = "title_block"

# Обозначение по ЕСКД: необязательный буквенный код и группы цифр через точку.
# Три цифры до первой точки обязательны — иначе под обозначение подходили бы
# дата `13.11.2026` и номер пункта `1.2.3`.
_DESIGNATION = re.compile(r"[А-ЯA-Z]{0,4}\d{3,}(?:\.\d{2,3}){2,}")

# Надпись вида или разреза: повторённая буква (`Г-Г`, `А-А`) либо слово.
_VIEW_TITLE = re.compile(r"\b([А-ЯA-Z])\s?-\s?\1\b|\b(Вид|Разрез|Сечение)\b")

# Размерная величина: диаметр, радиус, метрическая резьба, число с дробной
# частью, число отверстий, предельное отклонение, угол.
_DIMENSION = re.compile(
    r"[⌀ØøΦφ]\s?\d|\bR\s?\d|\bM\s?\d+(?:[x×]\d)?|\d+\s*отв|\d+[,.]\d|±\s?\d|\d+\s*°"
)

# Графы штампа по ГОСТ 2.104. По ним лист со штампом узнаётся и тогда, когда
# модель вернула штамп не полями, а строками переписанного текста.
_TITLE_BLOCK_LABELS = (
    "инв. №", "взам. инв", "подп. и дата", "№ докум", "лит.", "масса",
    "масштаб", "разраб", "пров.", "утв.", "н. контр", "т. контр",
    "формат", "листов", "копировал",
)

# Сколько полей и величин делают разбор чертёжным, когда ни штампа, ни
# обозначения на листе прочитать не удалось. Два поля и одна величина — это
# уже не плоский текст: распознавание не возвращает полей вовсе.
_MIN_FIELDS = 2


#: Типы блоков, которыми разбор отдаёт лист целиком, ничего на нём не прочитав.
PICTURE_BLOCKS = ("image", "drawing")


def picture_only(result) -> bool:
    """
    Разбор с детекцией областей вернул лист картинкой и текста с него не взял.

    Это второй, независимый признак непрочитанного листа — и единственный,
    который приходит не от растрового детектора, а от самого разбора. Рамка
    формата у детектора находится и на обычной странице с таблицами: по
    аудиту прода рамку он видит на 595 листах из 4771, и среди них договоры,
    программы обеспечения качества и сертификаты. Но такую страницу MinerU
    читает — возвращает её текст блоками, — а лист чертежа отдаёт одной
    картинкой. Поэтому «рамка есть» само по себе ничего не решает, а «рамка
    есть и разбор не взял с листа ни одной строки» решает: читать лист будет
    модель, и счёт символов до её чтения ничего не меряет.
    """
    if not any(block.type in PICTURE_BLOCKS for block in result.blocks):
        return False
    return not any(
        (block.text or "").strip() or block.table_data
        for block in result.blocks
        if block.type not in PICTURE_BLOCKS
    )


@dataclass
class DrawingStructure:
    """Из чего состоит чертёжный разбор листа — без счёта символов."""

    fields: int = 0            # всего полей листа
    measured: int = 0          # из них величин: размеры, допуски, резьба
    title_block: bool = False  # штамп прочитан
    designation: bool = False  # обозначение по ЕСКД прочитано
    view_titles: int = 0       # надписи видов и разрезов
    dimensions: int = 0        # размерные величины в полях и в тексте листа
    table_rows: int = 0        # строки таблиц листа, кроме штампа
    transcript_lines: int = 0  # строки переписанного текста листа

    @property
    def structured(self) -> bool:
        """
        Разбор чертёжный, а не плоский текст.

        Штамп или обозначение — сами по себе достаточный признак: ни того, ни
        другого сплошное распознавание чертежа не возвращает. Без них нужны
        поля с величинами: поле — это разобранная надпись, а не строка.
        """
        if self.title_block or self.designation:
            return True
        return self.fields >= _MIN_FIELDS and (self.dimensions >= 1 or self.view_titles >= 1)

    @property
    def complete(self) -> bool:
        """
        Разбор не только чертёжный, но и полный настолько, чтобы не звать
        человека по одному составу прочитанного: лист опознан (штамп либо
        обозначение) и с него прочитана хоть одна величина или таблица.
        """
        return bool(
            (self.title_block or self.designation)
            and (self.dimensions >= 1 or self.view_titles >= 1 or self.table_rows >= 1)
        )

    def as_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "structured": self.structured, "complete": self.complete}


def drawing_structure(fragments: Iterable[Any]) -> DrawingStructure:
    """Признаки чертёжного разбора по фрагментам документа."""
    found = DrawingStructure()
    texts: List[str] = []
    for fragment in fragments:
        if getattr(fragment, "type", None) != "drawing":
            continue
        content = fragment.content
        fields = content.structured_drawing_fields or []
        payload = content.structured_payload or {}

        found.fields += len(fields)
        found.measured += sum(1 for f in fields if f.category in MEASURED_CATEGORIES)
        found.title_block = (
            found.title_block
            or bool(payload.get(TITLE_BLOCK))
            or any(f.category == TITLE_BLOCK for f in fields)
        )
        tables = [t for t in payload.get("tables") or [] if t.get("kind") != TITLE_BLOCK]
        found.table_rows += sum(len(t.get("rows") or []) for t in tables)
        found.transcript_lines += int(payload.get("transcript_line_count") or 0)

        texts.append(content.text or "")
        texts.extend(f"{f.value} {f.tolerance or ''}" for f in fields)
        texts.extend(str(cell) for t in tables for row in t.get("rows") or [] for cell in row)

    whole = "\n".join(texts)
    found.designation = bool(_DESIGNATION.search(whole))
    found.view_titles = len(_VIEW_TITLE.findall(whole))
    found.dimensions = len(_DIMENSION.findall(whole))
    if not found.title_block:
        lowered = whole.lower()
        found.title_block = sum(1 for label in _TITLE_BLOCK_LABELS if label in lowered) >= 3
    return found
