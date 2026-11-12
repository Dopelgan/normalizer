"""
Текст в рамке ЕСКД: не чертёж, не таблица; мусор OCR-слоя рядом с MinerU.

Страницы смоделированы по разбору прод-индекса 12.11 (parser-quality):
ТЗ/ПОК/СТО в рамке со штампом уходили в поля чертежа «Примечание: …»
и в Qwen-VL, ПМ1 (изм. 4) становилась одной таблицей на всю книгу,
Tesseract-слой со строками conf < 0.4 дублировал MinerU на сканах.
"""

from dataclasses import dataclass

import pytest

from core.config import settings
from core.models.parse_result import ParsedBlock
from core.providers import page_prose
from core.providers.mineru_parser import drop_low_confidence_lines
from core.providers.text_layer import TextLayer, TextLine, reconcile_with_text_layer
from core.providers.vector_drawing import _score_page


# ------------------------------------------------------------ синтетика
TZ_PAGE = [
    "1 Общие положения",
    "1.1 Настоящее техническое задание распространяется на установку",
    "автоматизированную дозирования ТАДУ 421451.003 и устанавливает требования",
    "к её изготовлению, приёмке, поставке и вводу в эксплуатацию на объекте.",
    "1.2 Установка предназначена для дозирования химических реагентов в",
    "контуры энергоблока и должна работать в непрерывном режиме круглосуточно.",
    "2 Назначение и область применения",
    "2.1 Изготовитель обязан обеспечить соответствие установки требованиям",
    "нормативных документов, перечисленных в разделе 3 настоящего задания.",
    "2.2 Поставка осуществляется комплектно с эксплуатационной документацией",
    "на русском и английском языках в объёме, согласованном с заказчиком.",
    "3 Требования к качеству",
    "3.1 Входной контроль материалов проводится по программе обеспечения",
    "качества изготовителя с оформлением протоколов по установленной форме.",
    # Штамп ГОСТ 2.104 в правом нижнем углу.
    "Изм.", "Лист", "№ докум.", "Подп.", "Дата",
    "ТАДУ 421451.003 ТЗ",
    "Лист", "7",
]

DRAWING_PAGE = [
    "Ø42H7", "Ø60", "R5", "M12x1,5", "Ra 3,2", "45°", "120", "86±0,2", "+0,1",
    "-0,05", "35", "18", "А-А", "Б", "2 фаски 1x45°", "Ø8,5", "4 отв.", "R2",
    "1 *Размеры для справок.", "2 Неуказанные предельные отклонения по H14, h14.",
    "3 Покрытие Хим.Окс.прм.", "Сталь 40Х ГОСТ 4543-2016",
    "Изм.", "Лист", "№ докум.", "Подп.", "Дата", "ТАДУ 305179.001", "Лист", "1",
]


@dataclass
class _Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0


class _Page:
    """Лист A4 в рамке ЕСКД со штампом — как ПОК/ТЗ, и как чертёж."""

    def __init__(self, lines, extra_paths=0):
        self.rect = _Rect(0, 0, 595, 842)
        self._text = "\n".join(lines)
        paths = [{"rect": _Rect(20, 14, 580, 828)}]  # рамка формата
        for row in range(10):                          # сетка штампа
            y = 760 + row * 6
            paths.append({"rect": _Rect(340, y, 580, y + 0.5)})
        for i in range(extra_paths):                   # геометрия детали
            paths.append({"rect": _Rect(100 + i % 200, 200, 140 + i % 200, 320)})
        self._paths = paths

    def get_drawings(self):
        return self._paths

    def get_text(self, kind="text"):
        return self._text


@pytest.fixture
def text_in_frame(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_TEXT_IN_FRAME_ENABLED", True)


# ------------------------------------------------------------ проза
class TestProse:
    def test_tz_page_is_prose(self):
        verdict = page_prose.assess(TZ_PAGE)
        assert verdict.is_prose
        assert verdict.signals["heading"] and verdict.signals["sheet_number"]

    def test_english_tz_page_is_prose(self):
        # ТЗ (анг): худший документ отчёта, 65 страниц в виде полей чертежа.
        lines = [
            "1 General provisions",
            "1.1 This technical specification applies to the automated dosing unit",
            "and establishes requirements for its manufacture, acceptance and delivery.",
            "1.2 The unit is intended for dosing of chemical reagents into the circuits",
            "of the power unit and shall operate continuously around the clock.",
            "2 Purpose and scope of application of the equipment",
            "2.1 The manufacturer shall ensure compliance of the unit with the documents",
            "listed in section 3 of this specification and the contract annexes.",
            "Sheet", "12",
        ]
        verdict = page_prose.assess(lines)
        assert verdict.is_prose and verdict.signals["sheet_number"]

    def test_drawing_page_is_not_prose(self):
        assert not page_prose.assess(DRAWING_PAGE).is_prose

    def test_stamp_only_is_not_prose(self):
        assert not page_prose.assess(["Изм.", "Лист", "№ докум.", "Подп.", "Дата", "Лист", "1"]).is_prose

    def test_dimension_heavy_notes_stay_drawing(self):
        # Технические требования длинные, но размеров на листе больше.
        lines = DRAWING_PAGE * 3 + [
            "1 Неуказанные предельные отклонения размеров по H14, h14, IT14/2.",
            "2 Покрытие деталей по ГОСТ 9.306 после окончательной механообработки.",
        ]
        assert not page_prose.assess(lines).is_prose


# ------------------------------------------------------- классификатор
class TestDrawingClassifier:
    def test_flag_off_keeps_old_verdict(self):
        # Без флага поведение прежнее: рамка + штамп + геометрия = чертёж.
        verdict = _score_page(_Page(TZ_PAGE, extra_paths=150), 1)
        assert verdict.is_drawing
        assert "text_in_frame" not in verdict.signals

    def test_tz_in_eskd_frame_is_text_page(self, text_in_frame):
        verdict = _score_page(_Page(TZ_PAGE, extra_paths=150), 7)
        assert not verdict.is_drawing
        assert verdict.signals["text_in_frame"] is True
        assert verdict.signals["format_frame"] is True

    def test_real_drawing_stays_drawing(self, text_in_frame):
        verdict = _score_page(_Page(DRAWING_PAGE, extra_paths=150), 1)
        assert verdict.is_drawing
        assert "text_in_frame" not in verdict.signals


    def test_assembly_drawing_with_long_requirements_stays_drawing(self, text_in_frame):
        # Сборочный чертёж: длинные технические требования, мало размеров
        # текстом, но геометрии на листе — сотни путей.
        lines = ASSEMBLY_REQUIREMENTS + ["1", "2", "3", "4", "5", "Ø40", "120", "R5",
                                         "Изм.", "Лист", "№ докум.", "Подп.", "Дата",
                                         "ТАДУ 305179.001 СБ", "Лист", "1"]
        assert page_prose.assess(lines).is_prose  # текст сам по себе похож на прозу
        verdict = _score_page(_Page(lines, extra_paths=600), 1)
        assert verdict.is_drawing
        assert "text_in_frame" not in verdict.signals


ASSEMBLY_REQUIREMENTS = [
    "1 Сборку производить в соответствии с требованиями ОСТ 26-07-2033",
    "и технологического процесса изготовителя на данное изделие в целом.",
    "2 Перед сборкой все детали промыть, продуть сжатым воздухом и осмотреть",
    "на отсутствие забоин, заусенцев, следов коррозии и прочих дефектов.",
    "3 Резьбовые соединения затянуть с моментом согласно таблице на листе",
    "и застопорить краской, нанесённой полосой по всей длине соединения.",
    "4 Установку испытать на прочность и плотность пробным давлением воды",
    "в течение не менее десяти минут без падения давления по манометру.",
    "5 Маркировать шрифтом по ГОСТ 2.304 краской контрастного цвета на",
    "табличке, закреплённой на раме установки в доступном для осмотра месте.",
]


# --------------------------------------------------- рамка ≠ таблица
def _layer(lines_by_page, conf=1.0):
    pages = {}
    for page, texts in lines_by_page.items():
        pages[page] = [
            TextLine(text=t, page=page, bbox=[0.08, 0.05 + i * 0.03, 0.9, 0.07 + i * 0.03],
                     confidence=conf)
            for i, t in enumerate(texts)
        ]
    return TextLayer(pages=pages, page_count=len(pages))


def _frame_table(page):
    return ParsedBlock(
        type="table", text="<table><tr><td>" + " ".join(TZ_PAGE) + "</td></tr></table>",
        page=page, bbox=[0.03, 0.02, 0.97, 0.98], confidence=0.497,
        table_data={"rows": [[" ".join(TZ_PAGE)]]},
    )


class TestFrameTable:
    def test_multipage_frame_table_becomes_text(self):
        layer = _layer({1: TZ_PAGE, 2: TZ_PAGE, 3: TZ_PAGE})
        blocks = [_frame_table(1), _frame_table(2), _frame_table(3)]

        result, stats = reconcile_with_text_layer(blocks, layer, frame_table_guard=True)

        assert stats.frame_tables == 3
        assert not [b for b in result if b.type == "table"]
        text = " ".join(b.text for b in result if b.type == "text")
        assert "1 Общие положения" in text
        assert {b.page for b in result} == {1, 2, 3}

    def test_head_table_of_book_dropped_with_continuations(self):
        # MinerU: вся книга в «таблице» титульного листа, дальше пустые рамки.
        # Титул сам не проза, но его таблица держит текст продолжений —
        # оставь её, и текст книги ушёл бы в индекс дважды.
        title = ["УТВЕРЖДАЮ", "Генеральный директор", "ТАДУ 421451.003 ПМ1", "2026"]
        layer = _layer({1: title, 2: TZ_PAGE, 3: TZ_PAGE})
        head = ParsedBlock(type="table", text="", page=1, bbox=[0.03, 0.02, 0.97, 0.98],
                           table_data={"rows": [[t] for t in title + TZ_PAGE + TZ_PAGE]})
        empty = [ParsedBlock(type="table", text="", page=p, bbox=[0.03, 0.02, 0.97, 0.98])
                 for p in (2, 3)]

        result, stats = reconcile_with_text_layer([head] + empty, layer, frame_table_guard=True)

        assert stats.frame_tables == 3
        assert not [b for b in result if b.type == "table"]
        text = " ".join(b.text for b in result if b.type == "text")
        assert text.count("1 Общие положения") == 2  # по разу на стр. 2 и 3
        assert "УТВЕРЖДАЮ" in text

    def test_guard_off_keeps_table(self):
        layer = _layer({1: TZ_PAGE})
        result, stats = reconcile_with_text_layer([_frame_table(1)], layer)
        assert stats.frame_tables == 0
        assert [b for b in result if b.type == "table"]

    def test_real_full_page_table_kept(self):
        rows = [f"{i} Ø{20 + i} 12 ±0,1 35 3,2" for i in range(30)]
        layer = _layer({1: rows})
        table = ParsedBlock(type="table", text="<table></table>", page=1,
                            bbox=[0.03, 0.02, 0.97, 0.98], table_data={"rows": [["x"]]})
        result, stats = reconcile_with_text_layer([table], layer, frame_table_guard=True)
        assert stats.frame_tables == 0
        assert [b for b in result if b.type == "table"]

    def test_small_table_not_touched(self):
        layer = _layer({1: TZ_PAGE})
        table = _frame_table(1)
        table.bbox = [0.1, 0.1, 0.5, 0.4]
        _, stats = reconcile_with_text_layer([table], layer, frame_table_guard=True)
        assert stats.frame_tables == 0


# -------------------------------------------- мусор слоя распознавания
class TestLowConfidenceOcr:
    def _layer(self):
        good = TextLine(text="Акт приёмки оборудования", page=1, bbox=[0.1, 0.1, 0.6, 0.12], confidence=0.92)
        junk = TextLine(text="едприяувие = Fe SES axtne", page=1, bbox=[0.1, 0.5, 0.6, 0.52], confidence=0.31)
        alone = TextLine(text="ыва прол", page=2, bbox=[0.1, 0.5, 0.6, 0.52], confidence=0.2)
        return TextLayer(pages={1: [good, junk], 2: [alone]}, page_count=2)

    BODY = "Акт приёмки оборудования " * 10

    def test_footer_only_page_keeps_layer(self):
        # У MinerU на странице только колонтитул — тело ещё может быть
        # только в слое, его строки не трогаем.
        blocks = [ParsedBlock(type="text", text="Лист 3", page=1, bbox=[0.8, 0.95, 0.9, 0.97])]
        layer = drop_low_confidence_lines(self._layer(), blocks, 0.4)
        assert len(layer.lines(1)) == 2

    def test_junk_dropped_where_mineru_has_text(self):
        blocks = [ParsedBlock(type="text", text=self.BODY, page=1,
                              bbox=[0.1, 0.1, 0.6, 0.12])]
        layer = drop_low_confidence_lines(self._layer(), blocks, 0.4)
        assert [l.text for l in layer.lines(1)] == ["Акт приёмки оборудования"]
        # На стр. 2 у MinerU текста нет — слой единственный источник.
        assert [l.text for l in layer.lines(2)] == ["ыва прол"]

    def test_reconcile_no_longer_recovers_junk(self):
        blocks = [ParsedBlock(type="text", text=self.BODY, page=1,
                              bbox=[0.1, 0.1, 0.6, 0.12])]
        from core.providers.text_layer import OCR_LAYER

        before, _ = reconcile_with_text_layer(
            [ParsedBlock(**{**b.__dict__}) for b in blocks], self._layer(), source=OCR_LAYER)
        assert any("едприяувие" in (b.text or "") for b in before)

        layer = drop_low_confidence_lines(self._layer(), blocks, 0.4)
        after, _ = reconcile_with_text_layer(blocks, layer, source=OCR_LAYER)
        assert not any("едприяувие" in (b.text or "") for b in after)

    def test_flag_default_off(self):
        assert settings.OCR_LAYER_DROP_LOW_CONFIDENCE is False
        assert settings.DRAWING_TEXT_IN_FRAME_ENABLED is False
        assert settings.FRAME_TABLE_GUARD_ENABLED is False
