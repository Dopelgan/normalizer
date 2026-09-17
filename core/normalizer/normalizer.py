"""
Нормализатор: ParseResult -> список фрагментов контракта.

Принимает результат парсинга и метаданные документа, применяет стратегию
нарезки к текстовым блокам и превращает таблицы, формулы и изображения в
отдельные фрагменты. Заполняет все поля контракта, включая честные
`confidence`, `completeness` и `provenance`.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import settings
from core.models.contract import (
    Content,
    DrawingField,
    FormulaData,
    Fragment,
    Position,
    Provenance,
    TableData,
)
from core.models.parse_result import ParsedBlock, ParseResult
from core.normalizer.completeness import (
    PageCoverage,
    TextLayerRecall,
    formula_completeness,
    table_completeness,
)
from core.providers.vector_drawing import structure_ratio
from core.strategies.factory import StrategySelector

logger = logging.getLogger(__name__)

_ASSIGNMENT = re.compile(r"^\s*([A-Za-zА-Яа-яЁё_][\w\.]{0,31})\s*=\s*(.+?)\s*$", re.DOTALL)


def _sheet_of(blocks, default: Optional[str]) -> Optional[str]:
    """Имя листа у склеенного фрагмента берётся у его первого блока."""
    for block in blocks or []:
        if block.sheet_name:
            return block.sheet_name
    return default


def parse_expression(text: Optional[str]) -> Optional[FormulaData]:
    """
    Минимальный разбор формулы вида `X = <выражение>`.
    Большего из LaTeX без символьного движка честно не вытащить.
    """
    if not text:
        return None
    match = _ASSIGNMENT.match(text.replace("\n", " "))
    if not match:
        return None
    return FormulaData(
        base_var=match.group(1),
        operations=[{"type": "expression", "value": match.group(2).strip()}],
    )


class TextNormalizer:
    """Преобразует ParseResult в фрагменты контракта."""

    def __init__(self, strategy=None):
        self._strategy_override = strategy
        self.completeness_threshold = settings.COMPLETENESS_THRESHOLD

    # ------------------------------------------------------------- публичное
    def normalize(
        self, parse_result: ParseResult, metadata: Dict[str, Any]
    ) -> List[Fragment]:
        """Список фрагментов, отсортированный по (страница, порядок)."""
        fragments, _ = self.normalize_with_flags(parse_result, metadata)
        return fragments

    def normalize_with_flags(
        self, parse_result: ParseResult, metadata: Dict[str, Any]
    ) -> Tuple[List[Fragment], List[Dict[str, Any]]]:
        """
        То же, что normalize, плюс служебные пометки по каждому фрагменту
        (needs_review, is_fallback). В контракт они не входят и уезжают
        только в БД, в `chunks.extra_data`.
        """
        if not parse_result.blocks:
            return [], []

        doc_id = str(metadata.get("doc_id") or "")
        sheet = metadata.get("sheet")
        coverage = PageCoverage(parse_result.blocks, exhaustive=parse_result.exhaustive)
        # Если у документа есть текстовый слой, полнота считается прямым
        # сравнением с ним; покрытие площади остаётся только для сканов.
        recall = TextLayerRecall(parse_result.blocks, parse_result.text_layer_chars)

        meta = dict(metadata)
        meta.setdefault("source_kind", parse_result.source_kind)

        collected: List[Tuple[ParsedBlock, Fragment, Dict[str, Any]]] = []

        # ------------------------------------------------------------- текст
        # Нарезке отдаются не все текстовые блоки разом, а цепочками подряд
        # идущих в документе. Иначе она склеивает абзацы, между которыми в
        # документе стояла таблица: склейка встаёт на место своего первого
        # блока, таблица уезжает за неё, и порядок документа ломается.
        strategy = self._strategy_override or StrategySelector.build(meta)
        for run in self._text_runs(parse_result.blocks):
            for item in strategy.chunk(run, meta):
                # Короткие строки не выбрасываются: в технической документации
                # `POST /cart/add`, `race condition` или `Go:` — это и есть
                # содержание, а не мусор. Отбрасываем только пустое.
                if not item.text or not item.text.strip():
                    continue

                anchor = item.blocks[0] if item.blocks else run[0]
                is_fallback = item.is_fallback
                completeness = self._text_completeness(
                    item.page, is_fallback, recall, coverage
                )
                fragment = Fragment(
                    fragment_id="",
                    type="text",
                    content=Content(text=item.text),
                    position=Position(
                        page=item.page, sheet=_sheet_of(item.blocks, sheet),
                        bbox=item.bbox, order=item.order
                    ),
                    section_title=item.section_title or metadata.get("section_title"),
                    confidence=round(item.confidence, 3),
                    completeness=completeness,
                    provenance=self._provenance(
                        "text", parse_result, is_fallback, item.method
                    ),
                )
                collected.append((anchor, fragment, {"is_fallback": is_fallback}))

        # ----------------------------------------------------------- таблицы
        for block in parse_result.blocks:
            if block.type != "table":
                continue
            table_data = block.table_data or {}
            completeness = table_completeness(table_data)
            fragment = Fragment(
                fragment_id="",
                type="table",
                content=Content(
                    table_data=TableData(
                        headers=[str(h) for h in table_data.get("headers", [])],
                        rows=table_data.get("rows", []),
                        total_row=table_data.get("total_row"),
                    ) if table_data else None,
                    image_ref=block.image_ref,
                ),
                position=Position(
                    page=block.page, sheet=block.sheet_name or sheet,
                    bbox=block.bbox, order=block.order
                ),
                section_title=block.section_title or metadata.get("section_title"),
                confidence=round(block.confidence, 3),
                completeness=completeness,
                provenance=self._provenance("table", parse_result, block.is_fallback, block.method),
            )
            collected.append((block, fragment, {"is_fallback": block.is_fallback}))

        # ----------------------------------------------------------- формулы
        for block in parse_result.blocks:
            if block.type != "formula" or not block.text:
                continue
            parsed = parse_expression(block.text)
            completeness = formula_completeness(
                block.text, None, parsed.model_dump() if parsed else None
            )
            fragment = Fragment(
                fragment_id="",
                type="formula",
                content=Content(
                    text=block.text,
                    formula_mathml=None,   # MinerU отдаёт LaTeX, MathML не строим
                    parsed_expression=parsed,
                ),
                position=Position(
                    page=block.page, sheet=block.sheet_name or sheet,
                    bbox=block.bbox, order=block.order
                ),
                section_title=block.section_title or metadata.get("section_title"),
                confidence=round(block.confidence, 3),
                completeness=completeness,
                provenance=self._provenance("formula", parse_result, block.is_fallback, block.method),
            )
            collected.append((block, fragment, {"is_fallback": block.is_fallback}))

        # ----------------------------------------------------------- чертежи
        for block in parse_result.blocks:
            if block.type != "drawing" or not block.drawing_fields:
                continue
            fields = [DrawingField(**f) for f in block.drawing_fields]
            confidences = [f.confidence for f in fields]
            fragment = Fragment(
                fragment_id="",
                type="drawing",
                content=Content(
                    image_ref=block.image_ref,
                    structured_drawing_fields=fields,
                    structured_payload={
                        "detected_count": len(fields),
                        "recognition_mode": "vector_text_layer",
                        "page_score": round(block.confidence, 3),
                    },
                ),
                position=Position(
                    page=block.page, sheet=block.sheet_name or sheet,
                    bbox=block.bbox, order=block.order
                ),
                section_title=block.section_title or metadata.get("section_title"),
                confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                # Полнота разбора — доля полей с определённой категорией.
                # Текст при этом взят из файла целиком и не потерян.
                completeness=structure_ratio(block.drawing_fields),
                provenance=self._provenance(
                    "drawing", parse_result, block.is_fallback, block.method
                ),
            )
            collected.append((block, fragment, {"is_fallback": block.is_fallback}))

        # ------------------------------------------------------- изображения
        for block in parse_result.blocks:
            if block.type != "image" or not block.image_ref:
                continue
            fragment = Fragment(
                fragment_id="",
                type="image",
                content=Content(image_ref=block.image_ref),
                position=Position(
                    page=block.page, sheet=block.sheet_name or sheet,
                    bbox=block.bbox, order=block.order
                ),
                section_title=block.section_title or metadata.get("section_title"),
                confidence=round(block.confidence, 3),
                completeness=1.0,   # само изображение извлечено целиком
                provenance=self._provenance("image", parse_result, block.is_fallback, block.method),
            )
            collected.append((block, fragment, {"is_fallback": block.is_fallback}))

        # ---------------------------------------- порядок, id, служебные метки
        # Сортировка идёт в одной шкале — в той, в которой блоки пришли от
        # парсера. У текста `position.order` — это номер куска после нарезки,
        # и с `order` таблицы или чертежа он несравним: второй кусок обходил
        # десятую таблицу на той же странице, хотя нарезан из блока, который
        # шёл после неё. Место фрагмента задаёт блок-источник: для таблиц,
        # формул, чертежей и картинок это сам блок, для текста — первый блок
        # куска; номер куска остаётся только для кусков одного блока.
        source_order = {
            id(block): (block.order if block.order is not None else index)
            for index, block in enumerate(parse_result.blocks)
        }

        def place(triple: Tuple[ParsedBlock, Fragment, Dict[str, Any]]) -> tuple:
            block, fragment, _flag = triple
            position = fragment.position
            return (
                position.page if position else 0,
                source_order.get(id(block), 0),
                position.order if position and position.order is not None else 0,
                fragment.type,
            )

        collected.sort(key=place)

        fragments: List[Fragment] = []
        flags: List[Dict[str, Any]] = []
        degraded = bool(parse_result.degraded)
        for index, (_block, fragment, flag) in enumerate(collected, start=1):
            fragment.fragment_id = self.make_fragment_id(doc_id, index)
            if fragment.position is not None:
                fragment.position.order = index
            # Признак остаётся служебным: набор ключей контракта
            # зафиксирован, и расширять его в одностороннем порядке нельзя.
            # Наружу деградация видна через честный `completeness`.
            # Деградация разбора — повод на проверку независимо от полноты.
            # Победивший фоллбэк и недоступный MinerU дают внешне приличные
            # цифры: у OCR-страницы полнота 0.72 при пороге 0.5, и документ,
            # в котором формулы превратились в кашу, выглядел нормальным.
            flag["needs_review"] = (
                fragment.completeness < self.completeness_threshold or degraded
            )
            fragments.append(fragment)
            flags.append(flag)

        low = sum(1 for f in flags if f["needs_review"])
        if degraded:
            logger.warning(
                "Нормализация doc_id=%s: разбор деградировал (%s) — все фрагменты "
                "помечены на проверку", doc_id or "?", "; ".join(parse_result.degraded),
            )
        logger.info(
            "Нормализация doc_id=%s: фрагментов %d, ниже порога полноты %d, "
            "полнота по текстовому слою %s, fallback=%s",
            doc_id or "?", len(fragments), low,
            recall.overall() if recall.available else "н/д", parse_result.is_fallback,
        )
        return fragments, flags

    # ------------------------------------------------------------ внутреннее
    @staticmethod
    def make_fragment_id(doc_id: str, index: int) -> str:
        return f"{doc_id or 'doc'}-frag-{index:03d}"

    @staticmethod
    def _text_runs(blocks: List[ParsedBlock]) -> List[List[ParsedBlock]]:
        """
        Текстовые блоки, разбитые на цепочки подряд идущих в документе.

        Стратегия нарезки склеивает соседние блоки, а соседями в списке
        «только текст» оказываются абзацы, между которыми стояла таблица,
        формула или картинка. Место склеенного куска задаёт его первый блок,
        поэтому всё, что стояло между склеенными абзацами, уезжало в конец
        страницы. Разрыв цепочки на любом нетекстовом блоке чинит это, не
        трогая сами стратегии: внутри цепочки склейка по-прежнему работает.

        Порядок берётся из (страница, `order` блока), а не из порядка в
        списке: список приходит от парсера и после сшивки с текстовым слоем
        может быть перетасован, а `order` расставлен по чтению.
        """
        ordered = sorted(
            enumerate(blocks),
            key=lambda pair: (
                pair[1].page,
                pair[1].order if pair[1].order is not None else pair[0],
            ),
        )
        runs: List[List[ParsedBlock]] = []
        current: List[ParsedBlock] = []
        for _index, block in ordered:
            if block.type == "text":
                current.append(block)
            elif current:
                runs.append(current)
                current = []
        if current:
            runs.append(current)
        return runs

    # Способ получения блока -> (метод в контракте, уровень стратегии).
    _METHOD_MAP = {
        "cad_source": ("cad_source", 1),
        "text_layer": ("text_layer_extraction", 2),
        "office_extraction": ("office_extraction", 2),
        "plain_text": ("plain_text_extraction", 2),
        "tabular_extraction": ("tabular_extraction", 3),
        "restored_layout": ("restored_layout", 6),
        "vlm_escalation": ("vlm_escalation", 7),
        "text_layer_recovered": ("text_layer_extraction", 2),
        # Слой, собранный построчным распознаванием растра: текст оттуда,
        # структура — от разбора с детекцией областей, поэтому уровень 5.
        "ocr_layer": ("ocr_layer", 5),
        "ocr_layer_recovered": ("ocr_layer", 5),
        "mineru_ocr": ("mineru_ocr", 5),
        "mineru_table": ("mineru_table", 5),
        "mineru_formula": ("mineru_formula", 5),
        "mineru_image": ("mineru_image", 5),
        "tesseract": ("tesseract", 4),
        "vector_text_layer": ("vector_text_layer", 2),
    }
    # Если блок не сказал, чем получен, — угадываем по типу, но никогда не
    # называем это чтением текстового слоя: раньше так подписывался OCR.
    _KIND_DEFAULT = {
        "text": ("mineru_ocr", 5),
        "table": ("mineru_table", 5),
        "formula": ("mineru_formula", 5),
        "image": ("mineru_image", 5),
        "drawing": ("vector_text_layer", 2),
    }

    @classmethod
    def _provenance(
        cls,
        kind: str,
        parse_result: ParseResult,
        is_fallback: bool,
        method: Optional[str] = None,
    ) -> Provenance:
        """Метод и источник — из того, чем фрагмент реально получен."""
        source = parse_result.source_kind or "vector_pdf"
        if method == "tesseract" or (is_fallback and not method):
            # Уровень берётся из того, кто звал распознавание. `tesseract_full`
            # — это фоллбэк **внутри** разбора с детекцией областей: MinerU
            # звали, он ответил без текста, страницу дочитал OCR. Раньше оба
            # случая подписывались уровнем 4, а источник был жёстко прописан
            # как `scanned_pdf`, из-за чего PNG уезжал в контракт сканом PDF,
            # а три разных сценария были снаружи неразличимы.
            level = 5 if parse_result.parser_name == "tesseract_full" else 4
            return Provenance(method="tesseract", strategy_level=level, source=source)

        label, level = cls._METHOD_MAP.get(
            method or "", cls._KIND_DEFAULT.get(kind, cls._METHOD_MAP["mineru_ocr"])
        )
        return Provenance(method=label, strategy_level=level, source=source)

    def _text_completeness(
        self,
        page: int,
        is_fallback: bool,
        recall: TextLayerRecall,
        coverage: PageCoverage,
    ) -> float:
        """Полнота по текстовому слою, если он есть; иначе по покрытию площади."""
        measured = recall.of(page) if recall.available else None
        if measured is None:
            return coverage.text_completeness(page, is_fallback)
        return round(measured * (0.8 if is_fallback else 1.0), 3)
