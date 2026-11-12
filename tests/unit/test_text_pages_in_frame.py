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


# ------------------------------------------- сетка таблиц без чертежа
@dataclass
class _Point:
    x: float
    y: float


TABLE_CELLS = [
    "Наименование", "Кол.", "Примечание", "Клапан запорный Ду10", "28",
    "Фильтр 400 мм", "1", "Расходомер", "2", "Манометр электронный", "1",
    "Изм.", "Лист", "№ докум.", "Подп.", "Дата", "ТАДУ 421451.003 ПМ1", "Лист", "66",
]


class _GeometryPage(_Page):
    """Лист с известным составом путей (как у PyMuPDF `items`) и картинками."""

    def __init__(self, lines, grid=200, curves=0, diagonals=0, image_bbox=None):
        super().__init__(lines)
        paths = [{"rect": _Rect(20, 14, 580, 828),
                  "items": [("re", _Rect(20, 14, 580, 828), 1)]}]
        for row in range(10):
            y = 760 + row * 6
            paths.append({"rect": _Rect(340, y, 580, y + 0.5),
                          "items": [("l", _Point(340, y), _Point(580, y))]})
        for i in range(grid):  # сетка таблицы: только горизонтали и вертикали
            x = 40 + i % 500
            paths.append({"rect": _Rect(x, 100, x + 0.5, 700),
                          "items": [("l", _Point(x, 100), _Point(x, 700))]})
        for i in range(curves):  # окружности и дуги вида
            paths.append({"rect": _Rect(200, 200, 260, 260),
                          "items": [("c", _Point(200, 230), _Point(200, 200),
                                     _Point(260, 200), _Point(260, 230))]})
        for i in range(diagonals):  # штриховка разреза
            paths.append({"rect": _Rect(300, 300, 340, 340),
                          "items": [("l", _Point(300 + i % 30, 300), _Point(340, 340 - i % 30))]})
        self._paths = paths
        self._images = [] if image_bbox is None else [{"bbox": image_bbox}]

    def get_image_info(self):
        return self._images


class TestGridOnlyPage:
    def test_flag_off_table_in_frame_stays_drawing(self):
        assert _score_page(_GeometryPage(TABLE_CELLS), 66).is_drawing

    def test_table_in_frame_is_text_page(self, text_in_frame):
        # ПМ1 (изм. 4), ТЗ (анг), ПОК: таблица в рамке, строки ячеек короткие —
        # прозой не считается, но вся графика — сетка.
        assert not page_prose.assess(TABLE_CELLS).is_prose
        verdict = _score_page(_GeometryPage(TABLE_CELLS, grid=900), 66)
        assert not verdict.is_drawing
        assert verdict.signals["text_in_frame"] is True
        assert verdict.signals["geometry"]["grid_only"] is True

    def test_vector_drawing_with_curves_stays_drawing(self, text_in_frame):
        verdict = _score_page(_GeometryPage(DRAWING_PAGE, grid=150, curves=60), 1)
        assert verdict.is_drawing
        assert "text_in_frame" not in verdict.signals

    def test_hatched_section_stays_drawing(self, text_in_frame):
        verdict = _score_page(_GeometryPage(DRAWING_PAGE, grid=150, diagonals=40), 1)
        assert verdict.is_drawing

    def test_inserted_drawing_image_stays_drawing(self, text_in_frame):
        # ТУ 681821.001, прил. А: габаритный чертёж вставлен картинкой в рамку.
        verdict = _score_page(_GeometryPage(DRAWING_PAGE, grid=120,
                                            image_bbox=(60, 60, 560, 520)), 29)
        assert verdict.is_drawing
        assert verdict.signals["geometry"]["image_coverage"] > 0.15

    def test_large_orthogonal_scheme_with_terminals_stays_drawing(self, text_in_frame):
        # Схема Э3: тысячи проводов, доли кривых < 2 %, но клемм-кружков десятки.
        verdict = _score_page(_GeometryPage(DRAWING_PAGE, grid=3000, curves=50), 1)
        assert verdict.signals["geometry"]["curve_share"] <= 0.02
        assert verdict.is_drawing

    def test_unreadable_images_fall_back_to_prose(self, text_in_frame):
        class _Broken(_GeometryPage):
            def get_image_info(self):
                raise RuntimeError("broken xref")

        verdict = _score_page(_Broken(TABLE_CELLS, grid=200), 66)
        assert "geometry" not in verdict.signals
        assert verdict.is_drawing  # ячейки — не проза, прежний вердикт


# ------------------------------------------- классификатор листа (флаг 2)
FIGURE_PAGE = [
    "Приложение Б (справочное)",
    "Общий вид, габаритные размеры установки подготовки проб",
    "Поз.", "Наименование патрубка", "Материал", "Тип сварного соединения",
    "I", "Вход пробы", "12Х18Н10Т", "С-22",
    "II", "Выход охлаждающей воды", "20", "С-17",
    "Изм.", "Лист", "№ докум.", "Подп.", "Дата", "ТАДУ 421451.003 ТЗ", "Лист", "87",
]

DRAWING_FORM_STAMP = [
    "Корпус", "Лит.", "Масса", "Масштаб", "1:2",
    "Изм.", "Лист", "№ докум.", "Подп.", "Дата", "ТАДУ 305179.001", "Лист", "1",
]


@pytest.fixture
def page_classifier(text_in_frame, monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_PAGE_CLASSIFIER_ENABLED", True)


class TestPageClassifier:
    """Лист ТЗ/ПМ/ТУ с рисунком-картинкой: текст, а не чертёж (флаг 2)."""

    IMAGE = (60, 160, 560, 560)   # треть листа: рисунок в приложении ТЗ

    def test_flag_off_figure_page_stays_drawing(self, text_in_frame, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PAGE_CLASSIFIER_ENABLED", False)
        verdict = _score_page(_GeometryPage(FIGURE_PAGE, image_bbox=self.IMAGE), 87)
        assert verdict.is_drawing
        assert verdict.signals["geometry"]["lines_only"] is True

    def test_figure_in_text_page_is_text(self, page_classifier):
        verdict = _score_page(_GeometryPage(FIGURE_PAGE, image_bbox=self.IMAGE), 87)
        assert not verdict.is_drawing
        assert verdict.signals["text_in_frame"] is True
        assert verdict.signals["figure_in_text_page"] is True

    def test_drawing_form_stamp_stays_drawing(self, page_classifier):
        # Штамп формы 1 (графы «Масса», «Масштаб») — чертёж, хоть вид и картинкой.
        verdict = _score_page(
            _GeometryPage(FIGURE_PAGE + DRAWING_FORM_STAMP, image_bbox=self.IMAGE), 1)
        assert verdict.is_drawing

    def test_glued_stamp_labels_stay_drawing(self, page_classifier):
        lines = FIGURE_PAGE + ["Лит. Масса Масштаб", "1:2"]
        assert _score_page(_GeometryPage(lines, image_bbox=self.IMAGE), 1).is_drawing

    def test_mass_in_prose_is_not_a_stamp(self, page_classifier):
        lines = FIGURE_PAGE + ["Масса установки не более 350 кг, масштаб рисунка условный."]
        assert not _score_page(_GeometryPage(lines, image_bbox=self.IMAGE), 87).is_drawing

    def test_dimensions_over_picture_stay_drawing(self, page_classifier):
        # Вид картинкой, а размеры Ø/R/резьба — текстом: это чертёж.
        verdict = _score_page(_GeometryPage(DRAWING_PAGE, image_bbox=self.IMAGE), 29)
        assert verdict.is_drawing

    def test_page_mostly_picture_stays_drawing(self, page_classifier):
        verdict = _score_page(_GeometryPage(FIGURE_PAGE, image_bbox=(20, 14, 580, 828)), 87)
        assert verdict.is_drawing

    def test_curves_stay_drawing(self, page_classifier):
        verdict = _score_page(
            _GeometryPage(FIGURE_PAGE, grid=150, curves=60, image_bbox=self.IMAGE), 87)
        assert verdict.is_drawing

    def test_flag_default_off(self):
        assert settings.model_fields["DRAWING_PAGE_CLASSIFIER_ENABLED"].default is False
