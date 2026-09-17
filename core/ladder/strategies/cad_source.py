"""
Уровень 1 — чтение исходного файла системы проектирования (DXF).

Самый дешёвый и самый точный уровень: в DXF текст, координаты, угол
поворота, слой и атрибуты блоков лежат как данные. Распознавать нечего —
ни одна ошибка OCR здесь невозможна в принципе.

Лист DXF (модельное пространство и каждый лист печати) считается
самостоятельной единицей, как того требует архитектура: ссылка на факт
ведёт на лист, а координаты фрагмента привязаны к листу.

DWG сюда не попадает: это закрытый двоичный формат, для него нужен внешний
конвертер. Такой файл отклоняется раньше, на Quality Gate, с объяснением.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Dict, List, Tuple

from core import filetypes
from core.ladder.base import Strategy
from core.ladder.context import DocumentContext
from core.models.parse_result import ParsedBlock, ParseResult
from core.providers.text_layer import TextLayer, TextLine
from core.providers.vector_drawing import extract_drawing_fields, structure_ratio

logger = logging.getLogger(__name__)

try:
    import ezdxf
    from ezdxf import recover as ezdxf_recover
    EZDXF_AVAILABLE = True
except ImportError:  # pragma: no cover
    ezdxf = None
    ezdxf_recover = None
    EZDXF_AVAILABLE = False

# Запас вокруг габаритов чертежа, чтобы крайние подписи не липли к краю.
_EXTENT_MARGIN = 0.02

# Имя слоя DXF — это данные файла, а не догадка: чертёж почти всегда
# разложен по слоям, и слой «Размеры» говорит о содержимом надписи больше,
# чем её текст. Категория по слою ставится только там, где правила разбора
# ничего не определили, и с меньшей уверенностью, чем разбор по тексту.
_LAYER_CATEGORIES = (
    (("разм", "dim", "size"), "size"),
    (("допуск", "toler", "gdt", "gd&t"), "tolerance"),
    (("шерох", "rough", "surf"), "roughness"),
    (("штамп", "основн", "title", "stamp", "titleblock"), "title_block"),
    (("матери", "material"), "material"),
    (("резьб", "thread"), "thread"),
    (("радиус", "radius"), "radius"),
    (("выноск", "leader", "callout", "поз"), "callout"),
)

class CadSourceStrategy(Strategy):
    """DXF -> фрагменты чертежа без единого шага распознавания."""

    level = 1
    name = "cad_source"
    method = "cad_source"
    exhaustive = True

    def applicable(self, context: DocumentContext) -> bool:
        if not EZDXF_AVAILABLE:
            return False
        return (
            context.kind == filetypes.KIND_CAD
            and filetypes.is_supported(context.file_type)
        )

    def run(self, context: DocumentContext) -> ParseResult:
        document = self._open(context.data)
        blocks: List[ParsedBlock] = []

        for page_no, (layout_name, entities) in enumerate(self._layouts(document), start=1):
            if not entities:
                continue
            layer, attrib_texts, layer_of = self._to_text_layer(entities, page_no)
            fields = extract_drawing_fields(layer, page_no)
            if not fields:
                continue
            fields = self._promote_by_layer(fields, layer_of)
            fields = self._promote_title_block(fields, attrib_texts)

            blocks.append(ParsedBlock(
                type="drawing",
                text=" ".join(f["value"] for f in fields),
                page=page_no,
                sheet_name=layout_name,
                bbox=[0.0, 0.0, 1.0, 1.0],
                confidence=1.0,          # значения взяты из файла, а не прочитаны
                method=self.method,
                drawing_fields=fields,
            ))
            logger.info(
                "DXF, лист %s: полей %d, разобрано %.0f%%",
                layout_name, len(fields), structure_ratio(fields) * 100,
            )

        if not blocks:
            raise ValueError("В DXF не нашлось ни одной текстовой подписи")

        return self.build_result(blocks, source_kind="cad", page_count=len(blocks))

    # ------------------------------------------------------------ внутреннее
    @staticmethod
    def _open(data: bytes):
        """Чтение с восстановлением: выгрузки из САПР часто слегка битые."""
        try:
            document, auditor = ezdxf_recover.read(io.BytesIO(data))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"DXF не читается: {exc}") from exc
        if auditor.has_errors:
            logger.warning(
                "DXF прочитан с замечаниями (%d), разбираем что есть",
                len(auditor.errors),
            )
        return document

    @staticmethod
    def _layouts(document) -> List[Tuple[str, List[Any]]]:
        """Модельное пространство и листы печати — каждый отдельной единицей."""
        layouts: List[Tuple[str, List[Any]]] = [
            ("Model", list(document.modelspace()))
        ]
        try:
            for name in document.layouts.names_in_taborder():
                if name.lower() == "model":
                    continue
                layouts.append((name, list(document.layouts.get(name))))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось перечислить листы DXF: %s", exc)
        return layouts

    def _to_text_layer(self, entities, page_no: int) -> Tuple[TextLayer, set, Dict[str, str]]:
        """Текстовые сущности DXF -> слой с координатами в долях листа."""
        raw: List[Tuple[str, float, float, float, float]] = []
        attrib_texts: set = set()
        layer_of: Dict[str, str] = {}

        for entity in entities:
            layer_name = str(getattr(getattr(entity, "dxf", None), "layer", "") or "")
            for text, x, y, rotation, height, is_attrib in self._texts_of(entity):
                cleaned = " ".join(str(text).split())
                if not cleaned:
                    continue
                raw.append((cleaned, x, y, rotation, height))
                layer_of.setdefault(cleaned, layer_name)
                if is_attrib:
                    attrib_texts.add(cleaned)

        if not raw:
            return TextLayer(pages={page_no: []}, page_count=page_no), attrib_texts, layer_of

        xs = [item[1] for item in raw]
        ys = [item[2] for item in raw]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span_x = (max_x - min_x) or 1.0
        span_y = (max_y - min_y) or 1.0

        lines: List[TextLine] = []
        for text, x, y, rotation, height in raw:
            # DXF считает Y снизу вверх, страница — сверху вниз.
            nx = _clamp((x - min_x) / span_x * (1 - 2 * _EXTENT_MARGIN) + _EXTENT_MARGIN)
            ny = _clamp(1.0 - ((y - min_y) / span_y * (1 - 2 * _EXTENT_MARGIN) + _EXTENT_MARGIN))
            width = _clamp(len(text) * (height / span_x) * 0.6) if span_x else 0.02
            lines.append(TextLine(
                text=text,
                bbox=[nx, ny, _clamp(nx + max(width, 0.005)), _clamp(ny + 0.012)],
                page=page_no,
                size=height,
                rotation=float(rotation),
                mono=False,
            ))

        return TextLayer(pages={page_no: lines}, page_count=page_no), attrib_texts, layer_of

    @staticmethod
    def _texts_of(entity):
        """Текст сущности DXF вместе с точкой вставки и поворотом."""
        dxftype = entity.dxftype()
        try:
            if dxftype == "TEXT":
                insert = entity.dxf.insert
                yield (entity.dxf.text, float(insert[0]), float(insert[1]),
                       float(getattr(entity.dxf, "rotation", 0.0) or 0.0),
                       float(getattr(entity.dxf, "height", 2.5) or 2.5), False)
            elif dxftype == "MTEXT":
                insert = entity.dxf.insert
                yield (entity.plain_text(), float(insert[0]), float(insert[1]),
                       float(getattr(entity.dxf, "rotation", 0.0) or 0.0),
                       float(getattr(entity.dxf, "char_height", 2.5) or 2.5), False)
            elif dxftype == "INSERT":
                # Основная надпись обычно вставлена блоком, а её графы —
                # атрибуты этого блока. Это самый надёжный источник штампа.
                for attrib in getattr(entity, "attribs", []) or []:
                    point = attrib.dxf.insert
                    yield (attrib.dxf.text, float(point[0]), float(point[1]),
                           float(getattr(attrib.dxf, "rotation", 0.0) or 0.0),
                           float(getattr(attrib.dxf, "height", 2.5) or 2.5), True)
            elif dxftype in ("ATTRIB", "ATTDEF"):
                point = entity.dxf.insert
                yield (entity.dxf.text, float(point[0]), float(point[1]),
                       float(getattr(entity.dxf, "rotation", 0.0) or 0.0),
                       float(getattr(entity.dxf, "height", 2.5) or 2.5), True)
            elif dxftype == "DIMENSION":
                text = getattr(entity.dxf, "text", "") or ""
                if text and text not in ("<>", " "):
                    point = entity.dxf.defpoint
                    yield (text, float(point[0]), float(point[1]), 0.0, 2.5, False)
        except Exception as exc:  # noqa: BLE001 — одна сущность не роняет лист
            logger.debug("Сущность %s пропущена: %s", dxftype, exc)

    @staticmethod
    def _promote_by_layer(
        fields: List[Dict[str, Any]], layer_of: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Категория по имени слоя — там, где по тексту не определилась. Надпись
        «SIZE 50 mm» на слое SIZE это размер, а не «не разобрано»: раньше
        такой лист давал полноту 0 при полностью читаемом содержимом.
        """
        for field in fields:
            if field["category"] != "note":
                continue
            name = (layer_of.get(field["value"]) or "").lower()
            if not name:
                continue
            for markers, category in _LAYER_CATEGORIES:
                if any(marker in name for marker in markers):
                    field["category"] = category
                    # Слой назван в самом файле, и это надёжнее, чем «не
                    # разобрано» с уверенностью 0.4, но слабее разбора по
                    # тексту: слой могли назвать как угодно.
                    field["confidence"] = 0.7
                    break
        return fields

    @staticmethod
    def _promote_title_block(
        fields: List[Dict[str, Any]], attrib_texts: set
    ) -> List[Dict[str, Any]]:
        """
        Атрибуты блока — это графы основной надписи, даже если геометрически
        они оказались не в правом нижнем углу листа.
        """
        for field in fields:
            if field["value"] in attrib_texts and field["category"] == "note":
                field["category"] = "title_block"
                field["confidence"] = 0.9
        return fields


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
