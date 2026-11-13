"""
Слой распознавания на листе, повёрнутом боком.

Скан альбомной страницы, сохранённый книжным без /Rotate (график поверки,
альбомный чертёж), Tesseract читает после поворота по OSD, а рамки
строк возвращает в систему исходного листа — чтобы сшить их с рамками
MinerU. Строки на таком листе — вертикальные полосы, и всё, что дальше
считало «сверху вниз, слева направо» по этим рамкам, путало порядок:
абзацы склеивались в один с перемешанными строками, колонки таблицы
искались поперёк строк, а штамп у правого нижнего угла листа оказывался
«наверху» и не узнавался. Данные синтетические.
"""

import random

import pytest
from PIL import Image

from core.models.parse_result import ParsedBlock
from core.providers import tesseract_fallback
from core.providers.ocr_prepare import PreparedPage, _rotate
from core.providers.tesseract_fallback import OcrLine, TesseractFallbackProvider
from core.providers.text_layer import (
    OCR_LAYER,
    TextLayer,
    TextLine,
    group_lines,
    lines_to_text,
    page_turn,
    reconcile_with_text_layer,
    upright_bbox,
)
from tests.unit.test_mangled_tables import FORMULA_SIMPLE, ocr_lines, table_block


def to_source(bbox, turn):
    """
    Рамка на прямо стоящем листе -> рамка на исходном листе, текст которого
    повёрнут на `turn` против часовой (90 — строки идут снизу вверх).
    Считается независимо от кода: поворот точки (u, v) прямого листа.
    """
    u1, v1, u2, v2 = bbox
    if turn == 90:      # прямой лист повёрнут против часовой: (u, v) -> (v, 1 - u)
        return [v1, 1 - u2, v2, 1 - u1]
    if turn == 270:     # по часовой: (u, v) -> (1 - v, u)
        return [1 - v2, u1, 1 - v1, u2]
    if turn == 180:
        return [1 - u2, 1 - v2, 1 - u1, 1 - v1]
    return list(bbox)


def signed(turn):
    return turn if turn <= 180 else turn - 360


# Прямо стоящий лист: заголовок, утверждение, абзац из трёх строк.
UPRIGHT = [
    ("ГРАФИК поверки эталонов", [0.30, 0.05, 0.70, 0.08]),
    ("и средств измерений на 2024 год", [0.25, 0.09, 0.75, 0.12]),
    ("Калибратор температуры КТ-3 зав. 063", [0.05, 0.30, 0.60, 0.33]),
    ("Термометр ПТСВ-1-2 зав. 2203", [0.05, 0.34, 0.45, 0.37]),
    ("Преобразователь ППО зав. 190", [0.05, 0.38, 0.52, 0.41]),
    ("Генеральный директор И.Ю. Шаповалов", [0.50, 0.70, 0.95, 0.73]),
]
EXPECTED = [
    "ГРАФИК поверки эталонов и средств измерений на 2024 год",
    "Калибратор температуры КТ-3 зав. 063 Термометр ПТСВ-1-2 зав. 2203 "
    "Преобразователь ППО зав. 190",
    "Генеральный директор И.Ю. Шаповалов",
]


def turned_lines(turn, shuffle=True):
    lines = [
        TextLine(text=text, bbox=to_source(bbox, turn), page=1,
                 rotation=signed(turn), confidence=0.9)
        for text, bbox in UPRIGHT
    ]
    if shuffle:
        random.Random(7).shuffle(lines)
    return lines


class TestReadingOrder:
    @pytest.mark.parametrize("turn", [0, 90, 180, 270])
    def test_paragraphs_follow_upright_reading_order(self, turn):
        groups = group_lines(turned_lines(turn))
        assert [lines_to_text(g) for g in groups] == EXPECTED

    @pytest.mark.parametrize("turn", [90, 270])
    def test_page_turn_from_lines(self, turn):
        assert page_turn(turned_lines(turn)) == turn

    def test_minority_of_turned_lines_does_not_turn_page(self):
        # Чертёж: подписи размеров повёрнуты, но текста по горизонтали больше.
        lines = turned_lines(0) + [
            TextLine(text="R96", bbox=[0.5, 0.5, 0.52, 0.6], page=1, rotation=90.0),
        ]
        assert page_turn(lines) == 0

    @pytest.mark.parametrize("turn", [90, 270])
    def test_recovered_blocks_on_turned_page_keep_order(self, turn):
        layer = TextLayer(pages={1: turned_lines(turn)}, page_count=1)
        # MinerU нашёл только картинку где-то в стороне — строки уходят
        # восстановленными блоками.
        image = ParsedBlock(type="image", page=1, bbox=to_source([0.8, 0.9, 0.9, 0.95], turn),
                            image_ref="assets/x.jpg")
        blocks, _ = reconcile_with_text_layer([image], layer, source=OCR_LAYER)
        texts = [b.text for b in blocks if b.type == "text"]
        assert texts == EXPECTED
        # Картинка внизу прямого листа — после текста.
        assert blocks[-1].type == "image"

    @pytest.mark.parametrize("turn", [90, 270])
    def test_text_block_rebuilt_in_reading_order(self, turn):
        block = ParsedBlock(type="text", page=1, bbox=to_source([0.04, 0.29, 0.62, 0.42], turn),
                            text="КТ-3 063 ПТСВ-1-2 2203 ППО 190 Калибратор Термометр Преобразователь")
        layer = TextLayer(pages={1: [l for l in turned_lines(turn) if "зав." in l.text]},
                          page_count=1)
        blocks, _ = reconcile_with_text_layer([block], layer, source=OCR_LAYER)
        assert blocks[0].text == EXPECTED[1]


class TestStampOnTurnedSheet:
    """Штамп альбомного листа, отсканированного книжным."""

    # Штамп формы 1 на A3 альбомном: 185×55 мм в правом нижнем углу.
    STAMP = [0.53, 0.79, 0.97, 0.97]

    def _rows(self, turn, stamp_bbox=None):
        bbox = to_source(stamp_bbox or self.STAMP, turn)
        block = ParsedBlock(
            type="table", page=1, bbox=bbox,
            table_data={"headers": ["Изм.", "Лист", "№ докум.", "Подп.", "Дата"],
                        "rows": [["Разрад.", "Иванов", "Macca", "", ""],
                                 ["Пров.", "Сидоров", "Луcт12X18H10T", "", ""]],
                        "total_row": None},
        )
        inside = to_source([0.55, 0.82, 0.90, 0.85], turn)
        lines = [
            TextLine(text="Иванов Сидоров", bbox=inside, page=1, rotation=signed(turn),
                     confidence=0.9),
            # Остальной текст листа повёрнут так же.
            *[TextLine(text=f"Технические требования пункт {i}", page=1,
                       bbox=to_source([0.05, 0.1 + 0.05 * i, 0.5, 0.13 + 0.05 * i], turn),
                       rotation=signed(turn), confidence=0.9) for i in range(4)],
        ]
        blocks, _ = reconcile_with_text_layer(
            [block], TextLayer(pages={1: lines}, page_count=1), source=OCR_LAYER
        )
        return next(b for b in blocks if b.type == "table").table_data["rows"]

    @pytest.mark.parametrize("turn", [0, 90, 180, 270])
    def test_stamp_labels_and_material_fixed_on_turned_sheet(self, turn):
        rows = self._rows(turn)
        assert rows[0][0] == "Разраб."
        assert rows[0][2] == "Масса"
        assert rows[1][2] == "Лист 12Х18Н10Т"

    @pytest.mark.parametrize("turn", [0, 90])
    def test_table_at_top_of_upright_sheet_is_not_a_stamp(self, turn):
        rows = self._rows(turn, stamp_bbox=[0.05, 0.05, 0.95, 0.30])
        assert rows[0][0] == "Разрад."


class TestMangledTableOnTurnedPage:
    """Таблица на повёрнутом листе: колонки ищутся по прямому листу."""

    @pytest.mark.parametrize("turn", [90, 270])
    def test_prose_cells_repaired_on_turned_page(self, turn):
        block = table_block()
        block.bbox = to_source(block.bbox, turn)
        lines = [
            TextLine(text=l.text, bbox=to_source(l.bbox, turn), page=1,
                     rotation=signed(turn), confidence=l.confidence)
            for l in ocr_lines()
        ]
        blocks, stats = reconcile_with_text_layer(
            [block], TextLayer(pages={1: lines}, page_count=1), source=OCR_LAYER
        )

        assert stats.repaired_tables == 1
        assert stats.degraded_tables == 0
        table = blocks[0].table_data
        assert table["rows"][0][0] == FORMULA_SIMPLE
        assert "Прямолинейное равномерное движение" in table["rows"][0][1]
        assert "Движение по окружности" in table["rows"][2][1]
        # Рамка таблицы осталась в системе исходного листа.
        assert blocks[0].bbox == block.bbox


class TestOcrLinesCarryOrientation:
    """Поворот листа по OSD доезжает до строк слоя."""

    def _provider(self, monkeypatch, orientation):
        image = Image.new("L", (400, 600), 255)
        page = PreparedPage(image=image, source_size=image.size, orientation=orientation)
        monkeypatch.setattr(tesseract_fallback, "prepare_page", lambda *_a, **_k: page)
        data = {
            "text": ["Калибратор", "температуры"], "conf": ["90", "91"],
            "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1],
            "left": [10, 120], "top": [10, 10], "width": [100, 120], "height": [20, 20],
        }
        monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_data",
                            lambda *_a, **_k: data)
        monkeypatch.setattr(tesseract_fallback.settings, "OCR_PREPROCESS_ENABLED", True)
        return TesseractFallbackProvider(storage=object()), image

    @pytest.mark.parametrize("orientation, rotation", [(0, 0.0), (90, 90.0), (270, -90.0),
                                                       (180, 180.0)])
    def test_rotation_from_osd(self, monkeypatch, orientation, rotation):
        provider, image = self._provider(monkeypatch, orientation)
        lines = provider.ocr_lines(image)
        assert [l.rotation for l in lines] == [rotation]

    @pytest.mark.parametrize("orientation", [90, 180, 270])
    def test_upright_frame_matches_ocr_preparation(self, orientation):
        """
        Соглашение об угле сходится с подготовкой листа: рамка строки на
        выпрямленном листе, переведённая `to_source` в исходный и обратно
        `upright_bbox` с углом строки, возвращается на место.
        """
        page = PreparedPage(image=Image.new("L", (600, 900), 255), source_size=(600, 900))
        _rotate(page, -orientation)  # так поворачивает `prepare_page`
        on_prepared = [0.1, 0.2, 0.6, 0.25]
        rotation = orientation - 360 if orientation > 180 else orientation
        turn = page_turn([TextLine(text="строка", bbox=[0, 0, 1, 1], page=1,
                                   rotation=float(rotation))])
        assert upright_bbox(page.to_source(on_prepared), turn) == pytest.approx(
            on_prepared, abs=0.01)

    def test_text_layer_lines_keep_rotation(self, monkeypatch):
        provider, image = self._provider(monkeypatch, 90)
        monkeypatch.setattr(provider, "_materialize",
                            lambda uri, data=None, file_type=None: ("/tmp/x.png", lambda: None))
        monkeypatch.setattr(provider, "_images_of", lambda *_a, **_k: [(1, image)])
        layer = provider.text_layer("x.png", "png")
        assert [l.rotation for l in layer.lines(1)] == [90.0]

    def test_full_ocr_blocks_follow_upright_order(self, monkeypatch):
        provider = TesseractFallbackProvider(storage=object())
        monkeypatch.setattr(provider, "ocr_lines", lambda _image: [
            OcrLine(text=l.text, bbox=l.bbox, confidence=0.9, rotation=90.0)
            for l in turned_lines(90)
        ])
        blocks = provider._blocks_from_image(Image.new("L", (10, 10), 255), page=1)
        assert [b.text.replace("\n", " ") for b in blocks] == EXPECTED


class TestUprightPagesUnchanged:
    """Повёрнутое меньшинство не поворачивает лист и не путает строки."""

    def test_dimension_labels_do_not_turn_sparse_drawing(self):
        # Разреженный прямой чертёж: цифр в повёрнутых подписях размеров
        # больше, чем букв в штампе, но букв в них нет.
        labels = [TextLine(text="100±0,2", page=1, bbox=[0.1 + 0.05 * i, 0.3, 0.12 + 0.05 * i, 0.4],
                           rotation=90.0) for i in range(6)]
        stamp = TextLine(text="АБВГ.301261.005", page=1, bbox=[0.80, 0.90, 0.88, 0.92])
        assert page_turn(labels + [stamp]) == 0

        from core.providers.vector_drawing import extract_drawing_fields
        fields = extract_drawing_fields(TextLayer(pages={1: labels + [stamp]}, page_count=1), 1)
        assert {f["value"]: f["category"] for f in fields}["АБВГ.301261.005"] == "title_block"

    def test_horizontal_notes_on_turned_page_are_not_glued(self):
        notes = [
            TextLine(text="Note top", page=1, bbox=[0.40, 0.05, 0.60, 0.07]),
            TextLine(text="Note bottom", page=1, bbox=[0.42, 0.90, 0.62, 0.92]),
        ]
        groups = group_lines(turned_lines(90) + notes)
        texts = [lines_to_text(g) for g in groups]
        assert "Note top" in texts and "Note bottom" in texts

    def test_rotated_leftovers_on_upright_page_keep_page_order(self):
        # Лист стоит прямо; среди строк, не попавших в рамки, большинство
        # повёрнуто. Поворот решается по странице, а не по остатку.
        body = [TextLine(text=f"Обычный горизонтальный текст абзаца номер {i}", page=1,
                         bbox=[0.2, 0.1 + 0.03 * i, 0.8, 0.12 + 0.03 * i]) for i in range(8)]
        block = ParsedBlock(type="text", page=1, bbox=[0.15, 0.08, 0.85, 0.40], text="x")
        side = [
            TextLine(text="Вертикальная подпись первая", page=1, bbox=[0.05, 0.5, 0.07, 0.9],
                     rotation=90.0),
            TextLine(text="Вертикальная подпись вторая", page=1, bbox=[0.08, 0.3, 0.10, 0.9],
                     rotation=90.0),
        ]
        blocks, _ = reconcile_with_text_layer(
            [block], TextLayer(pages={1: body + side}, page_count=1), source=OCR_LAYER)
        recovered = [b.text for b in blocks if b is not block]
        assert recovered == ["Вертикальная подпись вторая Вертикальная подпись первая"]
