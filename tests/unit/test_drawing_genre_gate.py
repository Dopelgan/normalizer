"""
Жанровый гейт: лист считается чертёжным только по признакам чертежа.

Страницы взяты из настоящих документов корпуса заказчика, вместе с их
собственными геометрическими признаками из разбора:

* «ПОК 01.25 (изм. 1)», стр. 25 — таблица должностей и полномочий.
  Признаки разбора: `vector_paths=126`, `title_block_lines=14`,
  `format_frame=false`, `chars=1576`. Балл 0,75 = 0,35 (путей больше 120)
  + 0,25 (пучок отрезков в зоне штампа) + 0,15 (путей много, текста меньше
  2500 символов). Рамки формата у страницы нет вовсе — чертежом её сделали
  сетка таблицы и нижние строки, попавшие в зону штампа.
* «Программа обеспечения качества … АЭС Эль-Дабаа», стр. 4 — лист рассылки.
  `vector_paths=510`, `title_block_lines=31`, `chars=260`.
* «ТАДУ 421451.003И» (инструкция по монтажу), титул и стр. 2 — обозначение
  ЕСКД есть, но с кодом текстового документа.
* Настоящий сборочный чертёж: размеры, разрез, основная надпись формы 1.
* Скан чертежа: текста в файле нет — таких листов в корпусе большинство
  среди настоящих чертежей, и гейт обязан молчать.
"""

from dataclasses import dataclass

import pytest

from core.config import settings
from core.providers import drawing_genre
from core.providers.vector_drawing import _score_page


# ---------------------------------------------------------- реальные страницы
POK_TABLE_PAGE = """ПРОГРАММА ОБЕСПЕЧЕНИЯ КАЧЕСТВА ПРИ
КОНСТРУИРОВАНИИ И ИЗГОТОВЛЕНИИ ПРОДУКЦИИ,
ПОСТАВЛЯЕМОЙ НА АТОМНЫЕ ЭЛЕКТРИЧЕСКИЕ
СТАНЦИИ (ПОК (К,И))
Рев. 1
Лист: 25/90
ПОК 01.25
Должность
(Подразделение / специалист)
Функциональные обязаности
Полномочия
Ответственность
Начальник волочильного цеха
(Волочильный цех)
Руководство цехом;
Организация текущего производственного планирования;
Изготовление кабельной продукции.
Давать подчиненным работникам обязательные для исполнения
указания по вопросам производственной деятельности и
осуществлять контроль за их выполнением
За ненадлежащее исполнение или неисполнение своих
должностных обязанностей
"""

QUALITY_PROGRAMME_PAGE = """ПРОГРАММА ОБЕСПЕЧЕНИЯ КАЧЕСТВА
КОМПАНИИ ООО НТЛ-ПРИБОР ПРИ
ИЗГОТОВЛЕНИИ И ПОСТАВКЕ
ОБОРУДОВАНИЯ
NPPA
АЭС Эль-Дабаа
ООО НТЛ-ПРИБОР
ED.S.A054.&.&&&&&&.&&&&&&.089.QF.0001.E-3.0
Страница 4из 97
ЛИСТ РАССЫЛКИ
Организация/ Подразделение
Количество Контрольная копия (КК) / Для информации (И)
№ копии
"""

INSTALLATION_MANUAL_PAGE = """RPR.1099.10UKC.KUA60.TM.DC003
ТАДУ 421451.003И
Изм Лист № документа
Подпись Дата
Разраб.
Анцупова Н.Ю.
СИСТЕМА ПОДГОТОВКИ ПРОБ И ИЗМЕРЕНИЙ
в составе системы автоматизированного химического контроля первого контура
Инструкция по монтажу
ДЛЯ АЭС
"""

BANK_GUARANTEE_PAGE = """БАНКОВСКАЯ ГАРАНТИЯ № 10340336
г. Москва
10 марта 2022 г.
Банк-гарант безотзывно гарантирует уплату Бенефициару суммы, не превышающей
сумму гарантии, по первому письменному требованию Бенефициара.
Настоящая гарантия вступает в силу с даты её выдачи и действует до 31.12.2023.
Подписи уполномоченных лиц банка
"""

ASSEMBLY_DRAWING_PAGE = """Ø42H7
Ø60
R5
M12x1,5
Ra 3,2
45°
120
86±0,2
А-А
Вид Б
Масштаб 1:2
2 фаски 1x45°
Ø8,5
4 отв.
1 *Размеры для справок.
2 Неуказанные предельные отклонения размеров по H14, h14.
Сталь 40Х ГОСТ 4543-2016
Изм. Лист № докум. Подп. Дата
ТАДУ 305179.001 СБ
Лит. Масса Масштаб
"""

CYRILLIC_DRAWING_PAGE = """Ф42Н7
М12х1,5
R5
Ra 6,3
А-А
Разрез Б-Б
Лит. Масса Масштаб
ТАДУ 421451.003 СБ
Неуказанные предельные отклонения по Н14, н14
"""

SCHEME_PAGE = """Схема электрическая принципиальная
Лит. Масса Масштаб
ТАДУ 421451.003 Э3
Перечень элементов
R1 R2 R3 резисторы
"""


# ----------------------------------------------------------------- признаки
class TestEvidence:
    def test_pok_table_page_has_no_drawing_evidence(self):
        verdict = drawing_genre.gate(POK_TABLE_PAGE)
        assert verdict["weight"] == 0
        assert verdict["judgeable"] is True
        assert verdict["demote"] is True

    def test_quality_programme_distribution_sheet_demoted(self):
        verdict = drawing_genre.gate(QUALITY_PROGRAMME_PAGE)
        assert verdict["demote"] is True
        assert verdict["dimension_marks"] == 0

    def test_installation_manual_designation_is_not_evidence(self):
        # Обозначение ЕСКД есть, но код «И» — код текстового документа.
        verdict = drawing_genre.gate(INSTALLATION_MANUAL_PAGE)
        assert verdict["text_designation"] is True
        assert verdict["drawing_designation"] is False
        assert verdict["demote"] is True

    def test_bank_guarantee_demoted(self):
        assert drawing_genre.gate(BANK_GUARANTEE_PAGE)["demote"] is True

    def test_assembly_drawing_proves_itself(self):
        verdict = drawing_genre.gate(ASSEMBLY_DRAWING_PAGE)
        assert verdict["proves_drawing"] is True
        assert verdict["demote"] is False
        assert verdict["title_block_form1"] is True
        assert verdict["dimension_marks"] >= drawing_genre.MIN_DIMENSION_MARKS
        assert verdict["view_titles"] >= 1
        assert verdict["scale_mark"] is True
        assert verdict["drawing_designation"] is True

    def test_cyrillic_dimension_notation_counts(self):
        # «Ф», «М», «Н7» набирают в российских САПР кириллицей.
        verdict = drawing_genre.gate(CYRILLIC_DRAWING_PAGE)
        assert verdict["dimension_marks"] >= drawing_genre.MIN_DIMENSION_MARKS
        assert verdict["demote"] is False

    def test_scheme_with_form_one_stamp_is_kept(self):
        # Схема Э3: размеров на ней нет, но основная надпись — формы 1.
        verdict = drawing_genre.gate(SCHEME_PAGE)
        assert verdict["title_block_form1"] is True
        assert verdict["demote"] is False

    def test_page_without_text_is_not_judged(self):
        # Скан чертежа: текста в файле нет, признаков взять неоткуда.
        for text in ("", "   \n \n", "Лист 1\n"):
            verdict = drawing_genre.gate(text)
            assert verdict["judgeable"] is False
            assert verdict["demote"] is False

    def test_single_dimension_mark_is_not_enough(self):
        # «болт М12» в строке спецификации чертежом страницу не делает.
        text = ("Спецификация покупных изделий\n"
                "Болт М12 ГОСТ 7798-70, количество 24 шт.\n"
                + "Поставка комплектно с эксплуатационной документацией.\n" * 3)
        verdict = drawing_genre.gate(text)
        assert verdict["dimension_marks"] < drawing_genre.MIN_DIMENSION_MARKS
        assert verdict["demote"] is True

    def test_arbitrary_ratio_is_not_a_scale(self):
        # «1:3» и «1:7» в ряд ГОСТ 2.302 не входят.
        assert drawing_genre.assess("соотношение 1:3 и 1:7").scale_mark is False
        assert drawing_genre.assess("Масштаб 1:50").scale_mark is True


# ------------------------------------------------------ страница целиком
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
    """
    Лист A4 с сеткой таблицы и пучком отрезков в зоне штампа — геометрия
    «ПОК 01.25», стр. 25, как её вернул разбор: 126 путей, 14 отрезков в
    зоне штампа, рамки формата нет.
    """

    def __init__(self, text, paths=126, stamp_lines=14, frame=False):
        self.rect = _Rect(0, 0, 595, 842)
        self.rotation = 0
        self._text = text
        items = []
        if frame:
            items.append({"rect": _Rect(20, 14, 580, 828)})
        for row in range(stamp_lines):
            y = 700 + row * 8
            items.append({"rect": _Rect(340, y, 580, y + 0.5)})
        for i in range(max(0, paths - len(items))):
            y = 60 + (i % 60) * 10
            items.append({"rect": _Rect(40, y, 555, y + 0.4)})
        self._paths = items

    def get_drawings(self):
        return self._paths

    def get_text(self, kind="text"):
        return self._text

    def get_image_info(self):
        return []


@pytest.fixture
def gate_on(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_GENRE_GATE", True)


class TestScorePage:
    def test_flag_off_keeps_old_verdict(self):
        # Базовое поведение: сетка таблицы и пучок в зоне штампа = чертёж.
        verdict = _score_page(_Page(POK_TABLE_PAGE), 25)
        assert verdict.is_drawing
        assert verdict.score == pytest.approx(0.75)
        assert verdict.signals["format_frame"] is False
        assert verdict.signals["title_block_lines"] >= 6
        assert "genre_gate" not in verdict.signals

    def test_gate_demotes_table_page(self, gate_on):
        verdict = _score_page(_Page(POK_TABLE_PAGE), 25)
        assert not verdict.is_drawing
        assert verdict.signals["genre_gate"]["demote"] is True
        assert verdict.signals["genre_gate"]["weight"] == 0

    def test_gate_keeps_real_drawing(self, gate_on):
        verdict = _score_page(_Page(ASSEMBLY_DRAWING_PAGE), 1)
        assert verdict.is_drawing
        assert verdict.signals["genre_gate"]["demote"] is False

    def test_gate_silent_on_page_without_text(self, gate_on):
        # Скан: геометрия чертёжная, текста нет — решение остаётся прежним.
        verdict = _score_page(_Page(""), 1)
        assert verdict.is_drawing
        assert verdict.signals["genre_gate"]["demote"] is False

    def test_gate_does_not_promote(self, gate_on):
        # Признаки чертежа в тексте есть, а геометрии нет: гейт жанр не выдаёт.
        page = _Page(ASSEMBLY_DRAWING_PAGE, paths=4, stamp_lines=2)
        assert not _score_page(page, 1).is_drawing


class TestTextIsNotLost:
    """
    Снятый жанр не должен стоить ни одного символа. Лист, который перестал
    быть чертежом, разбирается обычной веткой: его текстовые блоки остаются
    на месте, а блок `drawing` вместо них не подставляется.

    На корпусе это проверено пофайлово: на 1334 снятых страницах 81
    документа чертёжный блок несёт 1 616 387 символов — ровно столько же,
    сколько строки текстового слоя тех же страниц, потому что поля чертежа
    и собирались из этих строк.
    """

    def test_text_blocks_survive_demotion(self):
        from core.models.parse_result import ParsedBlock
        from core.providers.mineru_parser import MinerUParserProvider
        from core.providers.text_layer import TextLayer, TextLine
        from core.providers.vector_drawing import DrawingPage

        lines = [TextLine(text=part, bbox=[0.1, 0.1 + i * 0.03, 0.9, 0.12 + i * 0.03],
                          page=1, size=10.0)
                 for i, part in enumerate(POK_TABLE_PAGE.strip().split("\n"))]
        layer = TextLayer(pages={1: lines}, page_count=1)
        blocks = [ParsedBlock(type="text", text=line.text, page=1, bbox=[0.1, 0.1, 0.9, 0.2])
                  for line in lines]
        parser = MinerUParserProvider.__new__(MinerUParserProvider)

        # Геометрия назвала лист чертежом: текстовые блоки исчезают.
        drawing = {1: DrawingPage(page=1, score=0.75)}
        after = parser._apply_vector_drawings(list(blocks), layer, drawing)
        assert [b.type for b in after] == ["drawing"]

        # Гейт снял жанр: текст страницы остаётся целиком.
        demoted = {1: DrawingPage(page=1, score=0.4)}
        after = parser._apply_vector_drawings(list(blocks), layer, demoted)
        assert all(b.type == "text" for b in after)
        kept = "".join("".join((b.text or "").split()) for b in after)
        assert kept == "".join(POK_TABLE_PAGE.split())


class TestRasterScanPages:
    """Гейт на ветке скана (`core.ladder.context._raster_drawing_pages`)."""

    def test_hybrid_scan_text_page_dropped(self, gate_on, monkeypatch):
        from core.ladder import context
        from core.providers import raster_drawing

        class _Pixmap:
            def tobytes(self, _fmt):
                return b"png"

        class _ScanPage:
            def __init__(self, text):
                self._text = text

            def get_pixmap(self, dpi=100):
                return _Pixmap()

            def get_text(self, kind="text"):
                return self._text

        class _Doc:
            page_count = 2

            def __init__(self):
                self._pages = [_ScanPage(BANK_GUARANTEE_PAGE),
                               _ScanPage(ASSEMBLY_DRAWING_PAGE)]

            def __getitem__(self, index):
                return self._pages[index]

            def close(self):
                pass

        monkeypatch.setattr(context, "_raster_drawing_pages",
                            context._raster_drawing_pages)
        monkeypatch.setattr("fitz.open", lambda **kwargs: _Doc())
        monkeypatch.setattr(
            raster_drawing, "analyse",
            lambda data: raster_drawing.RasterVerdict(raster_drawing.KIND_DRAWING, 0.8),
        )
        pages, checked = context._raster_drawing_pages(b"%PDF-")
        assert checked == 2
        # Гарантия отброшена, лист с размерами остался.
        assert pages == [2]
