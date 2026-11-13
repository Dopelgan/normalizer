"""
Отдельный проход по основной надписи (DRAWING_TITLE_BLOCK_PASS).

Строки здесь — не выдуманные примеры, а то, что на проверенных листах
действительно приходило от модели: значение без имени графы, потерянная
точка в обозначении, «C6» вместо «СБ», пропавшие даты подписей и «инв. №
1085-02». Каждое правило проверяется своей строкой.
"""

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers import drawing_vision, title_block
from core.providers.qwen_client import SheetReading

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "assets/doc-1/sheet.png"


# ===========================================================================
# Обозначения
# ===========================================================================

class TestDesignation:
    """Точки по ГОСТ 2.201 и код документа по ГОСТ 2.102."""

    @pytest.mark.parametrize("read, expected", [
        # Слипшийся порядковый номер: длина хвоста кратна трём.
        ("ТАДУ 421451001", "ТАДУ 421451.001"),
        ("ТАДУ 687000.001001", "ТАДУ 687000.001.001"),
        # Уже верное обозначение не портится.
        ("ТАДУ 687000.001.001", "ТАДУ 687000.001.001"),
        ("ТАДУ 302182.001 ГЧ", "ТАДУ 302182.001 ГЧ"),
    ])
    def test_dots_restored(self, read, expected):
        assert title_block.fix_designation(read) == expected

    def test_invented_dot_not_added(self):
        """
        «6870000.001» — лишний ноль от модели, а не потерянная точка. Хвост
        не кратен трём, и правило молчит: выдуманная точка хуже отсутствующей.
        """
        assert title_block.fix_designation("ТАДУ 6870000.001 СБ") == "ТАДУ 6870000.001 СБ"

    @pytest.mark.parametrize("read, expected", [
        ("ТАДУ 421451.001 C6", "ТАДУ 421451.001 СБ"),
        ("ТАДУ 305179.001.03 73", "ТАДУ 305179.001.03 ТЗ"),
    ])
    def test_document_code(self, read, expected):
        assert title_block.fix_designation(read) == expected

    def test_latin_organisation_code(self):
        """MinerU и модель читают код организации латиницей."""
        assert title_block.fix_designation(
            "AEGH.01.00017.111.0015 C6"
        ) == "АЕГН.01.00017.111.0015 СБ"

    def test_unknown_code_left_alone(self):
        """«AF» в графе формата кодом документа не является."""
        assert title_block.fix_document_code("AF") == ""


# ===========================================================================
# Имена граф
# ===========================================================================

class TestGraphNames:
    @pytest.mark.parametrize("name, key", [
        ("масса", "mass"), ("Масса", "mass"), ("масштаб", "scale"),
        ("разработал", "developed"), ("Разраб.", "developed"),
        ("Пров.", "checked"), ("Т.контр.", "tech_control"),
        ("Н.контр.", "norm_control"), ("Утв.", "approved"),
        ("Инв. № подл.", "inventory"), ("Перв. примен.", "first_use"),
        ("Взам. инв. №", "replaced_inventory"), ("Подп. и дата", "signed_date"),
    ])
    def test_canonical(self, name, key):
        assert title_block.canonical(name) == key

    def test_unknown_name(self):
        assert title_block.canonical("примечание к листу") is None


# ===========================================================================
# Разбор ответа модели
# ===========================================================================

class TestParse:
    def test_value_keeps_its_label(self):
        """
        Главная потеря: «12,35» и «1:4» без имени графы. Ответить по такой
        записи, что это масса и масштаб, нельзя.
        """
        parsed = title_block.parse({"fields": {"Масса": "12,35", "Масштаб": "1:4"}})
        assert parsed["fields"] == {"mass": "12,35", "scale": "1:4"}
        assert ("Масса", "12,35", False) in title_block.entries(parsed)
        assert ("Масштаб", "1:4", False) in title_block.entries(parsed)

    def test_handwritten_marked_separately(self):
        parsed = title_block.parse({
            "fields": {"Инв. № подл.": "1085-02", "Обозначение": "ТАДУ 302182.001 ГЧ"},
            "handwritten": ["Инв. № подл."],
        })
        assert parsed["handwritten"] == {"inventory": "1085-02"}
        assert parsed["fields"] == {"designation": "ТАДУ 302182.001 ГЧ"}
        assert ("Инв. № подл.", "1085-02", True) in title_block.entries(parsed)
        assert title_block.mark("1085-02", True) == "1085-02 (рукописно)"

    def test_handwritten_as_object(self):
        """Модель помечает графу и прямо в значении."""
        parsed = title_block.parse(
            {"fields": {"Подп. и дата": {"value": "26.08.22", "handwritten": True}}}
        )
        assert parsed["handwritten"] == {"signed_date": "26.08.22"}

    def test_change_entries_from_rows(self):
        """
        Графа изменений приходила в `rows` и полем не становилась никогда.
        Номер изменения бывает и первой ячейкой, и после слова «Изм».
        """
        parsed = title_block.parse({"fields": {}, "rows": [
            ["Изм.", "Лист", "№ докум.", "Подп.", "Дата"],
            ["2", "Нов.", "ТАДУ.409-23", "подпись", "22.10.16"],
            ["Изм", "1", "ТАДУ.355-22", "", "14.10.16"],
        ]})
        assert parsed["fields"]["changes"] == "2 Нов. ТАДУ.409-23; Изм 1 ТАДУ.355-22"

    def test_change_header_is_not_an_entry(self):
        parsed = title_block.parse(
            {"fields": {}, "rows": [["Изм.", "Лист", "№ докум.", "Подп.", "Дата"]]}
        )
        assert "changes" not in parsed["fields"]

    def test_signers_and_dates_from_rows(self):
        """
        Строка подписей: фамилия идёт в свою графу, дата — в «Подп. и дата»
        и помечается рукописной. Раньше дат не было вовсе.
        """
        parsed = title_block.parse({"fields": {}, "rows": [
            ["Разраб.", "Меснянкин", "", "", "26.08.22"],
            ["Пров.", "Бажко", "", "", "26.08.22"],
            ["Т.контр.", "Дивин", "", "", "26.08.22"],
            ["Утв.", "Глинко", "", "", "26.08.22"],
        ]})
        assert parsed["fields"]["developed"] == "Меснянкин"
        assert parsed["fields"]["checked"] == "Бажко"
        assert parsed["fields"]["approved"] == "Глинко"
        assert parsed["handwritten"]["signed_date"] == (
            "Разраб. 26.08.22; Пров. 26.08.22; Т.контр. 26.08.22; Утв. 26.08.22"
        )

    def test_designation_fixed_in_fields(self):
        parsed = title_block.parse({"fields": {"Обозначение": "ТАДУ 421451001 C6"}})
        assert parsed["fields"]["designation"] == "ТАДУ 421451.001 СБ"

    def test_first_use_column(self):
        """«Перв. примен.» — боковая графа, её не было ни на одном листе."""
        parsed = title_block.parse({"fields": {"Перв. примен.": "АЕГН.01.00017.111.0015"}})
        assert parsed["fields"]["first_use"] == "АЕГН.01.00017.111.0015"

    def test_unknown_graph_survives(self):
        """Неизвестное имя графы не выбрасывается: значение на листе есть."""
        parsed = title_block.parse({"fields": {"Копировал": "Иванов"}})
        assert ("Копировал", "Иванов", False) in title_block.entries(parsed)

    def test_empty_answer(self):
        parsed = title_block.parse({})
        assert title_block.entries(parsed) == []

    def test_entries_order_is_stable(self):
        """Порядок граф — свой, а не тот, в котором ответила модель."""
        parsed = title_block.parse(
            {"fields": {"Масштаб": "1:4", "Обозначение": "ТАДУ 302182.001 ГЧ"}}
        )
        assert [name for name, _, _ in title_block.entries(parsed)] == ["Обозначение", "Масштаб"]


# ===========================================================================
# Колонка дополнительных граф
# ===========================================================================

class TestSideColumn:
    def test_band_left_of_inner_frame(self):
        layout = drawing_vision.SheetLayout(
            image=None, frame=[0.02, 0.02, 0.98, 0.98],
            signals={"inner_frame": [0.095, 0.03, 0.97, 0.96]},
        )
        assert drawing_vision.side_column(layout) == [0.005, 0.03, 0.095, 0.96]

    def test_no_binding_margin(self):
        """Рамка у самого края: поля подшивки на листе нет."""
        layout = drawing_vision.SheetLayout(image=None, frame=[0.0, 0.0, 1.0, 1.0])
        assert drawing_vision.side_column(layout) is None

    def test_no_frame(self):
        assert drawing_vision.side_column(drawing_vision.SheetLayout(image=None)) is None


# ===========================================================================
# Проход целиком
# ===========================================================================

class FakeQwen:
    """Модель, которая отвечает заготовленным и запоминает, о чём спросили."""

    def __init__(self, block=None, side=None):
        self.available = True
        self.model = "fake-vl"
        self.block = block
        self.side = side
        self.asked = []

    def read_sheet(self, image, hint=""):
        return SheetReading(sheet_type="detail", annotations=[])

    def read_title_block(self, image):
        self.asked.append(("old", image.size))
        return {"fields": {"масса": "12,35"}, "rows": []}

    def read_title_block_fields(self, image):
        self.asked.append(("fields", image.size))
        return self.block

    def read_side_column(self, image):
        self.asked.append(("side", image.size))
        return self.side

    def read_table(self, image):
        return None


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


@pytest.fixture
def big_storage(temp_storage):
    """Лист в разрешении настоящего скана: на нём видны потолки вырезки."""
    import io

    buffer = io.BytesIO()
    _scan_sized_sheet().save(buffer, format="PNG")
    temp_storage.write_file(SHEET_URI, buffer.getvalue())
    return temp_storage


def sheet_fragment() -> Fragment:
    return Fragment(
        fragment_id="doc-1-frag-001",
        type="image",
        content=Content(image_ref=SHEET_URI),
        position=Position(page=1, bbox=[0.0, 0.0, 1.0, 1.0], order=1),
        confidence=0.9,
        completeness=1.0,
        provenance=Provenance(method="mineru_image", strategy_level=2, source="vector_pdf"),
    )


def values(fragment: Fragment) -> list:
    return [f.value for f in fragment.content.structured_drawing_fields]


class TestProcessorPass:
    def test_flag_off_keeps_old_reading(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", False)
        qwen = FakeQwen()
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})
        assert all(kind != "fields" for kind, _ in qwen.asked)
        assert result["payload"].get("title_block") == {"масса": "12,35"}

    def test_fields_carry_labels_and_handwritten(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        qwen = FakeQwen(
            block=title_block.parse({
                "fields": {
                    "Обозначение": "ТАДУ 421451001 C6",
                    "Масса": "12,35", "Масштаб": "1:4", "Пров.": "Бажко",
                },
                "rows": [["2", "Нов.", "ТАДУ.409-23", "подпись", "22.10.16"]],
            }),
            side=title_block.parse({
                "fields": {"Инв. № подл.": "1085-02"}, "handwritten": ["Инв. № подл."],
            }),
        )
        fragment = DrawingProcessor(storage, qwen=qwen).enrich_fragments(
            [sheet_fragment()], {}
        )[0]
        assert fragment.type == "drawing"
        shown = values(fragment)
        assert "Обозначение: ТАДУ 421451.001 СБ" in shown
        assert "Масса: 12,35" in shown
        assert "Масштаб: 1:4" in shown
        assert "Пров.: Бажко" in shown
        assert "Изм.: 2 Нов. ТАДУ.409-23" in shown
        assert "Инв. № подл.: 1085-02 (рукописно)" in shown

    def test_side_column_alone_is_enough(self, storage, monkeypatch):
        """Штамп не прочитан, боковая колонка прочитана — область не потеряна."""
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        qwen = FakeQwen(block=None, side=title_block.parse(
            {"fields": {"Перв. примен.": "АЕГН.01.00017.111.0015"}}
        ))
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})
        assert result["payload"].get("title_block") == {
            "first_use": "АЕГН.01.00017.111.0015"
        }

    def test_nothing_read_is_not_invented(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        qwen = FakeQwen(block=None, side=None)
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})
        assert "title_block" not in result["payload"]


# ===========================================================================
# Чего правила делать не должны
# ===========================================================================

class TestNoInvention:
    """
    Случаи из независимого ревью: правило, которое что-то восстанавливает,
    обязано молчать там, где восстанавливать нечего.
    """

    @pytest.mark.parametrize("value", [
        # Обозначение без пробела между кодом и номером: пробела не было на
        # листе, и добавлять его нельзя.
        "СМ1638.11.00.00.00.00 СБ",
        "АЕГН.01.00017.111.0015 СБ",
        "ТАДУ 687000.001.001",
        # Не обозначения вовсе.
        "30KUA70AX001",
        "ГОСТ 19904-90",
        "12Х18Н10Т",
    ])
    def test_correct_value_survives(self, value):
        assert title_block.fix_designation(value) == value

    def test_gost_reference_is_not_a_change_entry(self):
        """
        «ГОСТ 19903-74» похож на номер извещения хвостом «-74». Записью об
        изменении обычная строка штампа становиться не должна.
        """
        parsed = title_block.parse(
            {"fields": {}, "rows": [["Лист 2x1000x2000 ГОСТ 19903-74", "", "", ""]]}
        )
        assert "changes" not in parsed["fields"]

    def test_notice_number_stays_in_the_entry(self):
        """Запись об изменении без номера извещения ни о чём."""
        parsed = title_block.parse({"fields": {}, "rows": [
            ["2", "1", "лист 3", "ТАДУ.409-23", "подпись", "22.10.16"],
        ]})
        assert "ТАДУ.409-23" in parsed["fields"]["changes"]

    def test_mass_is_not_a_date(self):
        """«12,35» и «1.4» — масса и ссылка на пункт, а не даты подписей."""
        parsed = title_block.parse(
            {"fields": {}, "rows": [["Разраб.", "12.35", "Меснянкин"]]}
        )
        assert "signed_date" not in parsed["handwritten"]
        assert parsed["fields"]["developed"] == "12.35"

    def test_signature_word_is_not_a_surname(self):
        """Модель ставит «подпись» там, где на листе закорючка."""
        parsed = title_block.parse(
            {"fields": {}, "rows": [["Разраб.", "подпись", "Меснянкин", "26.08.22"]]}
        )
        assert parsed["fields"]["developed"] == "Меснянкин"

    def test_date_keeps_its_signer(self):
        parsed = title_block.parse({"fields": {}, "rows": [
            ["Разраб.", "Меснянкин", "26.08.22"],
            ["Утв.", "Глинко", "30.09.25"],
        ]})
        assert parsed["handwritten"]["signed_date"] == "Разраб. 26.08.22; Утв. 30.09.25"

    @pytest.mark.parametrize("name, key", [
        ("листов всего", "sheets"),
        ("наименование материала", "material"),
        ("обозначение материала", "material"),
    ])
    def test_longest_name_wins(self, name, key):
        """Короткое имя графы не должно перебивать длинное."""
        assert title_block.canonical(name) == key


def _scan_sized_sheet():
    """Синтетический лист, увеличенный до размера листа A3 в 200 dpi."""
    from PIL import Image

    image = sheet.build()
    return image.resize((image.width * 2, image.height * 2), Image.LANCZOS)


class TestHighResolutionCrop:
    """
    Вырезка штампа обязана доехать до модели в своём разрешении: потолок в
    `crop` и потолок при отправке — одно и то же число.
    """

    def test_crop_respects_its_own_limit(self):
        # Лист в разрешении настоящего скана: на синтетическом листе 1600 px
        # вырезка штампа до потолков не дотягивает и разницы не видно.
        image = _scan_sized_sheet()
        piece = drawing_vision.crop(
            image, sheet.title_block_bbox(), upscale=3.0, max_side=2600,
        )
        plain = drawing_vision.crop(image, sheet.title_block_bbox(), upscale=3.0)
        assert max(piece.size) > max(plain.size)
        assert max(piece.size) <= 2600

    def test_pass_sends_the_enlarged_piece(self, big_storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        monkeypatch.setattr(settings, "QWEN_TITLE_BLOCK_MAX_SIDE", 2600)
        monkeypatch.setattr(settings, "QWEN_TITLE_BLOCK_UPSCALE", 3.0)
        qwen = FakeQwen(block=title_block.parse({"fields": {"Масса": "12,35"}}))
        DrawingProcessor(big_storage, qwen=qwen).process(SHEET_URI, {})
        sent = [size for kind, size in qwen.asked if kind == "fields"]
        assert sent and max(sent[0]) > settings.QWEN_REGION_MAX_SIDE


class TestMergeBlocks:
    """Штамп и боковая колонка в одном разборе."""

    def test_different_values_both_survive(self):
        from core.drawing_processor import _merge_blocks

        merged = _merge_blocks(
            {"fields": {"inventory": "1085-02"}, "handwritten": {}, "unknown": {}, "rows": []},
            {"fields": {}, "handwritten": {"inventory": "1085"}, "unknown": {}, "rows": []},
        )
        assert merged["fields"]["inventory"] == "1085-02"
        assert merged["handwritten"]["inventory"] == "1085"

    def test_same_value_stays_handwritten_only(self):
        from core.drawing_processor import _merge_blocks

        merged = _merge_blocks(
            {"fields": {"inventory": "1085-02"}, "handwritten": {}, "unknown": {}, "rows": []},
            {"fields": {}, "handwritten": {"inventory": "1085-02"}, "unknown": {},
             "rows": []},
        )
        assert "inventory" not in merged["fields"]
        assert merged["handwritten"]["inventory"] == "1085-02"

    def test_stamp_overrides_side_column(self):
        from core.drawing_processor import _merge_blocks

        merged = _merge_blocks(
            {"fields": {"designation": "ТАДУ 302182.001 ГЧ"}, "handwritten": {},
             "unknown": {}, "rows": []},
            {"fields": {"designation": "ТАДУ 302182.001"}, "handwritten": {},
             "unknown": {}, "rows": []},
        )
        assert merged["fields"]["designation"] == "ТАДУ 302182.001 ГЧ"


class TestUpright:
    def test_column_turns_clockwise(self):
        """
        Надписи колонки читаются снизу вверх: после поворота нижняя графа
        должна оказаться слева. Проверяется по тёмной метке внизу полосы.
        """
        from PIL import Image

        strip = Image.new("L", (40, 200), 255)
        strip.paste(0, (0, 180, 40, 200))       # метка у нижнего края
        turned = drawing_vision.upright(strip)
        assert turned.size == (200, 40)
        assert turned.getpixel((5, 20)) == 0    # метка ушла налево
        assert turned.getpixel((195, 20)) == 255


class TestStampByStandard:
    """
    Вырезка штампа по ГОСТ 2.104, а не по уцелевшим линиям сетки. На листе
    АЕГН…0015 СБ сетка нашла только правый блок «Лит. | Масса | Масштаб»:
    обозначения, наименования и фамилий в вырезке не было вовсе.
    """

    def layout(self, frame, size=(1653, 2339)):
        from PIL import Image

        return drawing_vision.SheetLayout(
            image=Image.new("L", size, 255), frame=frame,
            signals={"inner_frame": frame},
        )

    def test_a4_portrait_stamp_spans_the_frame(self):
        """У A4 штамп шириной 185 мм — это вся ширина внутренней рамки."""
        box = drawing_vision.stamp_bbox(self.layout([0.0824, 0.0105, 0.9738, 0.9885]))
        assert box is not None
        assert box[0] == pytest.approx(0.0824, abs=0.005)
        assert box[2] == 0.9738 and box[3] == 0.9885
        # 55 мм из 287 мм высоты рамки.
        assert (box[3] - box[1]) / (0.9885 - 0.0105) == pytest.approx(55 / 287, abs=0.01)

    def test_unknown_format_falls_back_to_the_grid(self):
        """Рамка не той пропорции — подгонять по ГОСТ нечего."""
        layout = self.layout([0.1, 0.1, 0.9, 0.2])
        assert drawing_vision.stamp_bbox(layout) is None
        assert drawing_vision.stamp_bbox(layout) is None
        region = drawing_vision.Region(kind="title_block", bbox=[0.7, 0.8, 0.97, 0.98])
        assert drawing_vision.title_block_crop(layout, region) == [0.7, 0.8, 0.97, 0.98]

    def test_grid_piece_is_replaced_not_merged(self):
        """Найденный сеткой кусок не сужает и не раздувает вырезку по ГОСТ."""
        layout = self.layout([0.0824, 0.0105, 0.9738, 0.9885])
        piece = drawing_vision.Region(kind="title_block", bbox=[0.7334, 0.868, 0.9738, 0.9885])
        assert drawing_vision.title_block_crop(layout, piece) == drawing_vision.stamp_bbox(layout)


class TestRepeats:
    def test_one_date_written_five_times(self):
        parsed = title_block.parse({"fields": {
            "Подп. и дата": "26.08.22; 26.08.22; 26.08.22; 26.08.22; 26.08.22",
        }})
        assert parsed["fields"]["signed_date"] == "26.08.22"

    def test_change_is_not_taken_twice(self):
        """Графа, названную моделью рукописной, из строк не добирают."""
        parsed = title_block.parse({
            "fields": {"Изм.": "1 Зам. 356-22 26.09.22"},
            "handwritten": ["Изм."],
            "rows": [["1", "Зам.", "356-22", "", "26.09.22"]],
        })
        assert "changes" not in parsed["fields"]
        assert parsed["handwritten"]["changes"] == "1 Зам. 356-22 26.09.22"


class TestHandwrittenOnlyWhereItHappens:
    """
    Пометка «рукописно» понижает уверенность поля, и ставить её наугад
    нельзя. На листе ТАДУ 302182.001 ГЧ модель назвала рукописными массу,
    масштаб и номер листа, напечатанные чертёжным шрифтом.
    """

    @pytest.mark.parametrize("name, key", [
        ("Масса", "mass"), ("Масштаб", "scale"), ("Лист", "sheet"),
        ("Обозначение", "designation"), ("Организация", "organization"),
    ])
    def test_printed_graph_is_never_handwritten(self, name, key):
        parsed = title_block.parse({"fields": {name: "1:4"}, "handwritten": [name]})
        assert key in parsed["fields"] and not parsed["handwritten"]

    @pytest.mark.parametrize("name, key", [
        ("Инв. № подл.", "inventory"), ("Подп. и дата", "signed_date"),
        ("Взам. инв. №", "replaced_inventory"), ("Изм.", "changes"),
    ])
    def test_hand_filled_graph_keeps_the_mark(self, name, key):
        parsed = title_block.parse({"fields": {name: "1085-02"}, "handwritten": [name]})
        assert key in parsed["handwritten"]


class TestFrameFallback:
    def test_outer_frame_when_inner_is_wrong(self):
        """
        На листе АЕГН…0015 СБ нижняя линия внутренней рамки найдена на 0,85
        высоты: её пропорция не отвечает ни одному формату, и штамп ищется
        по рамке формата.
        """
        from PIL import Image

        layout = drawing_vision.SheetLayout(
            image=Image.new("L", (1653, 2339), 255),
            frame=[0.0824, 0.0105, 0.9738, 0.9885],
            signals={"inner_frame": [0.0824, 0.0105, 0.9738, 0.8522]},
        )
        box = drawing_vision.stamp_bbox(layout)
        assert box is not None and box[3] == 0.9885
        assert box[0] == pytest.approx(0.0824, abs=0.005)


class TestWholeSheetIsNotAFrame:
    def test_sheet_edge_does_not_pass_for_a_format(self):
        """
        Пропорция бумаги (0,707) случайно сходится с рамкой A2 книжного.
        Рамка во весь лист — обрез скана, и штамп по ней не строится.
        """
        from PIL import Image

        layout = drawing_vision.SheetLayout(
            image=Image.new("L", (1653, 2339), 255), frame=[0.0, 0.0, 1.0, 1.0],
        )
        assert drawing_vision.stamp_bbox(layout) is None

# ===========================================================================
# Стык с достоверностью размеров (пачка 4)
# ===========================================================================

class TestFidelityOnTitleBlockPass:
    """
    Отдельный проход по штампу (DRAWING_TITLE_BLOCK_PASS) заменил прежнее
    чтение области, а правки достоверности размеров
    (DRAWING_DIMENSION_FIDELITY) были сделаны в прежнем пути. На проде оба
    флага включены вместе, поэтому проверяется именно эта пара: правка
    масштаба и правило про путающиеся буквы обязаны доезжать и здесь.
    """

    def test_scale_graph_repaired_with_both_flags(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
        qwen = FakeQwen(block=title_block.parse(
            {"fields": {"Масштаб": "7:1", "Масса": "12,35"}}
        ))
        processor = DrawingProcessor(storage, qwen=qwen)
        result = processor.process(SHEET_URI, {})
        # Значение одно и то же и в structured_payload, и в поле чертежа:
        # иначе запись MinerU с «7:1» окажется повтором исправленного.
        assert result["payload"]["title_block"]["scale"] == "1:1"
        fragment = processor.enrich_fragments([sheet_fragment()], {})[0]
        assert "Масштаб: 1:1" in values(fragment)

    def test_scale_graph_untouched_without_fidelity_flag(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
        qwen = FakeQwen(block=title_block.parse({"fields": {"Масштаб": "7:1"}}))
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})
        assert result["payload"]["title_block"]["scale"] == "7:1"

    def test_mass_is_not_mistaken_for_a_scale(self, storage, monkeypatch):
        """Правка зовётся только для графы «Масштаб»."""
        monkeypatch.setattr(settings, "DRAWING_TITLE_BLOCK_PASS", True)
        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
        qwen = FakeQwen(block=title_block.parse(
            {"fields": {"Масса": "7:1", "Лист": "7"}}
        ))
        result = DrawingProcessor(storage, qwen=qwen).process(SHEET_URI, {})
        assert result["payload"]["title_block"]["mass"] == "7:1"
        assert result["payload"]["title_block"]["sheet"] == "7"

    @pytest.mark.parametrize("method", ["read_title_block_fields", "read_side_column"])
    def test_careful_letters_rule_reaches_both_stamp_prompts(self, method, monkeypatch):
        """
        Правило про «б/в», «ж/х», «н/п» и обозначение шва нужно и вырезке
        штампа, и колонке у левого поля: фамилии путаются в обеих.
        """
        from core.providers import qwen_client

        asked = []

        def fake_ask(self, image, prompt, max_side):
            asked.append(prompt)
            return None

        monkeypatch.setattr(qwen_client.QwenVisionClient, "_ask", fake_ask)
        client = qwen_client.QwenVisionClient.__new__(qwen_client.QwenVisionClient)

        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
        getattr(client, method)(object())
        assert "Ербасов" in asked[-1] and "Н1-ИНп" in asked[-1]

        monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)
        getattr(client, method)(object())
        assert "Ербасов" not in asked[-1]
