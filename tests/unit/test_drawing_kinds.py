"""
Род значения на чертеже (core/drawing_kinds.py): шероховатость неуказанных
поверхностей, фаска, изображение, на котором стоит размер.

Формы здесь взяты с проверенных листов. «Крышка»: на листе «√ Ra 6,3 (√)»
в правом верхнем углу, а в разборе — голая строка «Ra 6,3» в тексте листа и
поле только у местной «Ra 1,6». «Расширитель»: углы 15°, 20° и 35° без
указания, на каком изображении какой. Главное, что проверяется, — отказы:
подпись не ставится там, где оформление листа её не несёт.
"""

import pytest

from core import drawing_kinds as kinds
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.models.contract import DrawingField
from core.providers import drawing_vision
from core.providers.qwen_client import QwenVisionClient, SheetReading

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet as sheet  # noqa: E402

SHEET_URI = "src/sheet.png"
CORNER = [0.70, 0.03, 0.98, 0.14]
IN_CORNER = [0.86, 0.06, 0.95, 0.09]
UNKNOWN = [0.0, 0.0, 1.0, 1.0]


def field(value, category="size", bbox=None):
    return {"category": category, "value": value, "bbox": list(bbox or UNKNOWN)}


# ===========================================================================
# Знак шероховатости
# ===========================================================================

class TestRoughnessSign:
    @pytest.mark.parametrize("text, expected", [
        ("Ra 6,3", ("Ra 6,3", False)),
        ("√ Ra 6,3 (√)", ("Ra 6,3", True)),
        # Распознавание теряет сам значок, а скобки оставляет.
        ("Ra 6,3()", ("Ra 6,3", True)),
        ("Rz 40", ("Rz 40", False)),
        ("Ra 6.3", ("Ra 6,3", False)),
    ])
    def test_sign_alone(self, text, expected):
        assert kinds.roughness_sign(text) == expected

    @pytest.mark.parametrize("text", [
        "8. Лицевую сторону шлифовать, шерох. не хуже, чем Ra 1,6",
        "R1", "Ra", "17*", "",
    ])
    def test_mention_in_text_is_not_a_sign(self, text):
        assert kinds.roughness_sign(text) is None


class TestGeneralRoughness:
    def test_corner_crop_and_transcript_line(self):
        """«Крышка»: знак на вырезке угла и та же строка в тексте листа."""
        decision = kinds.general_roughness(
            [field("Ra 1,6", "roughness", [0.4, 0.3, 0.42, 0.32])],
            ["ТАДУ 687000.001001", "Ra 6,3", "17*"], CORNER,
            {"sign": True, "value": "Ra 6,3", "bracket_sign": True},
        )
        assert decision == {"value": "Ra 6,3", "bracket_sign": True,
                            "evidence": ["corner_crop", "transcript_line"]}

    def test_sheet_field_in_corner_and_transcript_line(self):
        """Поле листа стоит в углу, строка текста листа та же: вырезка не нужна."""
        fields = [field("Ra 6,3", "roughness", IN_CORNER)]
        assert not kinds.needs_corner_reading(fields, CORNER)
        decision = kinds.general_roughness(fields, ["Ra 6,3"], CORNER, None)
        assert decision["value"] == "Ra 6,3"
        assert decision["evidence"] == ["sheet_field_in_corner", "transcript_line"]
        # Вырезку не читали: стоит ли знак в скобках, неизвестно.
        assert decision["bracket_sign"] is None

    def test_bracket_seen_by_any_reading_counts(self):
        decision = kinds.general_roughness(
            [], ["Ra 6,3 (√)"], CORNER, {"sign": True, "value": "Ra 6,3", "bracket_sign": False})
        assert decision["bracket_sign"] is True

    def test_bracket_not_seen_stays_false(self):
        decision = kinds.general_roughness(
            [], ["Ra 6,3"], CORNER, {"sign": True, "value": "Ra 6,3", "bracket_sign": False})
        assert decision["bracket_sign"] is False

    def test_bare_transcript_line_is_not_enough(self):
        """Строка «Ra 6,3» без места — это и местный знак тоже."""
        decision = kinds.general_roughness([], ["Ra 6,3"], CORNER, None)
        assert decision == {"refused": kinds.REFUSED_NOT_READ}

    def test_empty_corner(self):
        """Сборочный лист: в углу знака нет, местная «Ra 1,6» общей не становится."""
        decision = kinds.general_roughness(
            [field("Ra 1,6", "roughness", [0.4, 0.3, 0.42, 0.32])], ["Ra 1,6"],
            CORNER, {"sign": False},
        )
        assert decision == {"refused": kinds.REFUSED_NO_SIGN}

    def test_corner_alone_is_not_confirmed(self):
        """Знак «увиден» на вырезке, но больше нигде на листе такого значения нет."""
        decision = kinds.general_roughness(
            [], ["17*", "R1"], CORNER, {"sign": True, "value": "Ra 3,2"})
        assert decision == {"refused": kinds.REFUSED_NOT_CONFIRMED, "value": "Ra 3,2"}

    def test_local_sign_elsewhere_does_not_confirm(self):
        """То же значение полем в другом месте листа — местный знак, не подтверждение."""
        decision = kinds.general_roughness(
            [field("Ra 3,2", "roughness", [0.3, 0.5, 0.33, 0.52])], [], CORNER,
            {"sign": True, "value": "Ra 3,2"},
        )
        assert decision["refused"] == kinds.REFUSED_NOT_CONFIRMED

    def test_view_in_the_corner_forbids_the_label(self):
        """В углу стоит размер — там изображение, и знак в нём может быть местным."""
        fields = [field("⌀32", "size", [0.80, 0.05, 0.84, 0.07]),
                  field("Ra 3,2", "roughness", IN_CORNER)]
        assert kinds.general_roughness(fields, ["Ra 3,2"], CORNER, None) == {
            "refused": kinds.REFUSED_VIEW_IN_CORNER}
        assert kinds.general_roughness(
            fields[:1], ["Ra 3,2"], CORNER, {"sign": True, "value": "Ra 3,2"},
        ) == {"refused": kinds.REFUSED_VIEW_IN_CORNER}

    def test_crop_misread_as_the_local_value_is_not_confirmed(self):
        """
        Вырезка угла прочитана как «Ra 3,2», а это значение местного знака:
        строка «Ra 3,2» в тексте листа — его прочтение, а не знака в углу.
        """
        decision = kinds.general_roughness(
            [field("Ra 3,2", "roughness", [0.3, 0.5, 0.33, 0.52])], ["Ra 3,2", "Ra 6,3"],
            CORNER, {"sign": True, "value": "Ra 3,2"},
        )
        assert decision == {"refused": kinds.REFUSED_NOT_CONFIRMED, "value": "Ra 3,2"}

    def test_field_without_place_confirms_the_crop(self):
        decision = kinds.general_roughness(
            [field("Ra 12,5", "roughness")], None, CORNER,
            {"sign": True, "value": "Ra 12,5"},
        )
        assert decision["evidence"] == ["corner_crop", "sheet_field"]
        assert decision["bracket_sign"] is False

    def test_two_readings_of_the_corner_disagree(self):
        decision = kinds.general_roughness(
            [field("Ra 6,3", "roughness", IN_CORNER)], ["Ra 6,3"], CORNER,
            {"sign": True, "value": "Ra 3,2"},
        )
        assert decision == {"refused": kinds.REFUSED_DISAGREE, "values": ["Ra 3,2", "Ra 6,3"]}

    def test_crop_empty_but_field_says_corner(self):
        decision = kinds.general_roughness(
            [field("Ra 6,3", "roughness", IN_CORNER)], ["Ra 6,3"], CORNER, {"sign": False})
        assert decision == {"refused": kinds.REFUSED_DISAGREE}

    def test_unreadable_value(self):
        decision = kinds.general_roughness([], ["Ra 6,3"], CORNER, {"sign": True, "value": "√"})
        assert decision == {"refused": kinds.REFUSED_NO_VALUE}

    def test_no_frame(self):
        assert kinds.general_roughness([], ["Ra 6,3"], None, None) == {
            "refused": kinds.REFUSED_NO_FRAME}

    def test_mark_keeps_local_sign_unlabelled(self):
        local = field("Ra 6,3", "roughness", [0.3, 0.5, 0.33, 0.52])
        general = field("Ra 6,3", "roughness", IN_CORNER)
        assert kinds.mark_general_roughness([local, general], "Ra 6,3", CORNER)
        assert "kind" not in local
        assert general["kind"] == kinds.KIND_ROUGHNESS_UNSPECIFIED
        assert general["kind_label"] == "Шероховатость неуказанных поверхностей"

    def test_mark_reports_missing_field(self):
        assert not kinds.mark_general_roughness([field("17*")], "Ra 6,3", CORNER)


# ===========================================================================
# Фаска
# ===========================================================================

class TestChamfer:
    @pytest.mark.parametrize("text", ["2×45°", "1,6x45°", "2х45°", "2 фаски 1×45°", "Фаска 2×45°"])
    def test_chamfer_notation(self, text):
        assert kinds.chamfer(text)

    @pytest.mark.parametrize("text", [
        "35°", "20°", "M20x1,5", "4x90°", "2x30°", "⌀32x2,5", "45°", "770x520x320",
    ])
    def test_everything_else_is_not_a_chamfer(self, text):
        """Угол сам по себе фаской не называется: «35°» у «Расширителя» — разделка кромки."""
        assert not kinds.chamfer(text)

    def test_mark(self):
        fields = [field("2×45°"), field("35°")]
        assert kinds.mark_chamfers(fields) == 1
        assert fields[0]["kind"] == kinds.KIND_CHAMFER and fields[0]["kind_label"] == "Фаска"
        assert "kind" not in fields[1]


# ===========================================================================
# Изображение, на котором стоит размер
# ===========================================================================

class TestViewTitles:
    @pytest.mark.parametrize("text, key", [
        ("Б(1:2)", "Б(1:2)"), ("Б (1:2)", "Б(1:2)"), ("A (1:2)", "А(1:2)"),
        ("В–В", "В–В"), ("B-B", "В–В"), ("А-А (2:1)", "А–А(2:1)"), ("Вид А", "ВИДА"),
        ("Б (4:1) 4 места", "Б(4:1)"),
    ])
    def test_title(self, text, key):
        assert kinds.view_title(text) == key

    @pytest.mark.parametrize("text", ["А", "Б", "35°", "A-Б", "1:2", "п.4", "Таблица 1"])
    def test_not_a_title(self, text):
        """Одна буква — это и стрелка взгляда, и кружок выносного элемента."""
        assert kinds.view_title(text) is None

    def test_title_needs_both_readings(self):
        """«Крышка»: знак «развёрнуто» один проход прочитал «B», другой «Q»."""
        fields = [field("A (1:2)", "callout"), field("B (1:2,5)", "callout"),
                  field("Б (4:1) 4 места", "callout")]
        titles = kinds.view_titles(fields, ["Q (1:2,5)", "A (1:2)", "Б (4:1)", "4 места"])
        assert sorted(titles) == ["А(1:2)", "Б(4:1)"]
        assert kinds.title_text(titles["Б(4:1)"]) == "Б (4:1)"

    def test_no_transcript_no_titles(self):
        assert kinds.view_titles([field("Б(1:2)", "callout")], None) == {}


class TestViewMembership:
    """Лист в два столбца: «А (1:1)» сверху справа, «Б (1:1)» под ним."""

    def sheet(self):
        fields = [
            field("А (1:1)", "callout", [0.74, 0.08, 0.81, 0.10]),
            field("35°", "size", [0.89, 0.19, 0.92, 0.21]),
            field("⌀32", "size", [0.75, 0.13, 0.79, 0.15]),
            field("Б (1:1)", "callout", [0.77, 0.36, 0.83, 0.38]),
            field("20°", "size", [0.65, 0.44, 0.68, 0.46]),
            field("3", "size", [0.68, 0.51, 0.70, 0.53]),
            field("3", "size", [0.60, 0.49, 0.62, 0.51]),
            field("505±10", "size", [0.55, 0.20, 0.57, 0.30]),
            field("15°", "size"),
        ]
        return fields, kinds.view_titles(fields, ["А (1:1)", "Б (1:1)"])

    def labelled(self, fields):
        return {f["value"]: f["view"] for f in fields if f.get("view")}

    def test_confirmed_values_get_the_view(self):
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "А (1:1)", "values": ["35°", "⌀32"]},
            {"title": "Б(1:1)", "values": ["20°"]},
        ]})
        assert self.labelled(fields) == {"35°": "А (1:1)", "⌀32": "А (1:1)", "20°": "Б (1:1)"}
        assert result == {"labelled": 3, "refused": []}

    def test_place_vetoes_the_model(self):
        """Модель отнесла «35°» к «Б», а надпись стоит выше заголовка «Б»."""
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "Б (1:1)", "values": ["35°"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_POSITION

    def test_far_value_is_not_of_this_view(self):
        """
        Размер главного вида ниже и в стороне от заголовка: ближайший
        заголовок над ним — «А», но до него дальше, чем бывает свой размер.
        """
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "А (1:1)", "values": ["505±10"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_POSITION

    def test_repeated_value_is_ambiguous(self):
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "Б (1:1)", "values": ["3"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_AMBIGUOUS

    def test_value_repeated_in_the_transcript_is_ambiguous(self):
        """Два «35°» на листе сведены в одно поле, но в тексте листа их два."""
        fields, titles = self.sheet()
        result = kinds.view_membership(
            fields, titles, {"views": [{"title": "А (1:1)", "values": ["35°"]}]},
            ["А (1:1)", "35°", "Б (1:1)", "35°"])
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_AMBIGUOUS

    @pytest.mark.parametrize("rival", ["В-В", "В"])
    def test_unconfirmed_title_in_between_still_blocks(self, rival):
        """
        Прямо над размером стоит заголовок, который второй проход прочитал
        иначе, или одна буква вида по стрелке: размер — его, а не дальнего «Б».
        """
        fields, titles = self.sheet()
        fields.append(field(rival, "callout", [0.66, 0.40, 0.70, 0.42]))
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "Б (1:1)", "values": ["20°"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_POSITION

    def test_general_roughness_belongs_to_no_view(self):
        fields, titles = self.sheet()
        general = field("Ra 6,3", "roughness", [0.80, 0.12, 0.84, 0.14])
        general["kind"] = kinds.KIND_ROUGHNESS_UNSPECIFIED
        fields.append(general)
        kinds.view_membership(fields, titles, {"views": [
            {"title": "А (1:1)", "values": ["Ra 6,3"]}]})
        assert "view" not in general

    def test_value_claimed_by_two_views(self):
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "А (1:1)", "values": ["20°"]}, {"title": "Б (1:1)", "values": ["20°"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_AMBIGUOUS

    def test_value_without_place_is_left_alone(self):
        """«Расширитель»: модель вернула поля без координат — подписи нет."""
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "А (1:1)", "values": ["15°"]}]})
        assert self.labelled(fields) == {}
        assert result["refused"][0]["reason"] == kinds.REFUSED_NO_POSITION

    def test_invented_title_and_value(self):
        fields, titles = self.sheet()
        result = kinds.view_membership(fields, titles, {"views": [
            {"title": "Г (2:1)", "values": ["35°"]}, {"title": "А (1:1)", "values": ["99"]}]})
        assert self.labelled(fields) == {}
        assert [item["reason"] for item in result["refused"]] == [
            kinds.REFUSED_UNKNOWN_TITLE, kinds.REFUSED_UNKNOWN_VALUE]

    def test_no_answer_changes_nothing(self):
        fields, titles = self.sheet()
        assert kinds.view_membership(fields, titles, None) == {"labelled": 0, "refused": []}
        assert self.labelled(fields) == {}


# ===========================================================================
# Контракт
# ===========================================================================

class TestContract:
    def test_unlabelled_field_looks_as_before(self):
        dumped = DrawingField(category="size", value="20").model_dump(mode="json")
        assert not {"kind", "kind_label", "view"} & set(dumped)

    def test_labelled_field_carries_the_kind(self):
        dumped = DrawingField(
            category="roughness", value="Ra 6,3", kind=kinds.KIND_ROUGHNESS_UNSPECIFIED,
            kind_label="Шероховатость неуказанных поверхностей",
        ).model_dump(mode="json")
        assert dumped["kind"] == "roughness_unspecified_surfaces"
        assert dumped["kind_label"] == "Шероховатость неуказанных поверхностей"
        assert "view" not in dumped


# ===========================================================================
# Запросы к модели
# ===========================================================================

class TestRequests:
    def client(self, monkeypatch, answer):
        client = QwenVisionClient(endpoint="http://qwen.test")
        asked = []

        def ask(image, prompt, max_side, max_tokens=None, list_key=None):
            asked.append(prompt)
            return answer

        monkeypatch.setattr(client, "_ask", ask)
        return client, asked

    def test_corner_answer(self, monkeypatch):
        client, asked = self.client(monkeypatch, {"sign": True, "value": "Ra 6,3", "bracket_sign": True})
        assert client.read_corner_roughness(object()) == {
            "sign": True, "value": "Ra 6,3", "bracket_sign": True}
        assert "правый верхний угол" in asked[0]

    def test_corner_answer_of_wrong_shape_is_no_sign(self, monkeypatch):
        client, _ = self.client(monkeypatch, {"sign": "да", "value": None})
        assert client.read_corner_roughness(object()) == {
            "sign": False, "value": "", "bracket_sign": False}

    def test_view_values(self, monkeypatch):
        client, asked = self.client(monkeypatch, {"views": [{"title": "Б (1:2)", "values": ["35°"]}, "мусор"]})
        assert client.read_view_values(object(), ["Б (1:2)", "В (1:2)"]) == {
            "views": [{"title": "Б (1:2)", "values": ["35°"]}]}
        assert "«Б (1:2)», «В (1:2)»" in asked[0]

    def test_view_values_without_list(self, monkeypatch):
        client, _ = self.client(monkeypatch, {"views": "нет"})
        assert client.read_view_values(object(), ["Б (1:2)"]) is None


# ===========================================================================
# Ветка чертежей
# ===========================================================================

class FakeQwen:
    """Модель, которая отвечает заготовленным и запоминает, о чём спросили."""

    def __init__(self, annotations=(), lines=(), corner=None, views=None):
        self.available = True
        self.model = "fake-vl"
        self.annotations = list(annotations)
        self.lines = list(lines)
        self.corner = corner
        self.views = views
        self.asked = []

    def read_sheet(self, image, hint=""):
        return SheetReading(sheet_type="detail", annotations=self.annotations)

    def transcribe(self, image):
        return self.lines

    def read_title_block(self, image):
        return None

    def read_table(self, image):
        return None

    def read_corner_roughness(self, image):
        self.asked.append(("corner", image.size))
        return self.corner

    def read_view_values(self, image, titles):
        self.asked.append(("views", list(titles)))
        return self.views


KRYSHKA = [
    {"category": "callout", "value": "A (1:2)", "bbox": [0.26, 0.30, 0.30, 0.32]},
    {"category": "size", "value": "(X+12)*", "bbox": [0.24, 0.67, 0.29, 0.69]},
    {"category": "size", "value": "17*", "bbox": [0.44, 0.17, 0.46, 0.22]},
    {"category": "size", "value": "⌀8,5", "bbox": [0.27, 0.40, 0.31, 0.42]},
]
KRYSHKA_LINES = ["ТАДУ 687000.001001", "Q (1:2,5)", "Ra 6,3", "A (1:2)", "(X+12)*", "17*"]


@pytest.fixture
def storage(temp_storage):
    temp_storage.write_file(SHEET_URI, sheet.png_bytes())
    return temp_storage


def read(storage, qwen, image_ref=SHEET_URI):
    return DrawingProcessor(storage, qwen=qwen).process(
        image_ref, {"source_path": SHEET_URI}, transcribe=True)


def by_value(result):
    return {f["value"]: f for f in result["parsed_fields"]}


class TestProcessor:
    def test_flags_off_nothing_is_asked_or_added(self, storage):
        qwen = FakeQwen(KRYSHKA, KRYSHKA_LINES, corner={"sign": True, "value": "Ra 6,3"})
        result = read(storage, qwen)
        assert qwen.asked == []
        assert "corner_roughness" not in result["payload"]
        assert "view_membership" not in result["payload"]
        assert all("kind" not in f and "view" not in f for f in result["parsed_fields"])
        assert "Ra 6,3" not in by_value(result)

    def test_corner_roughness_becomes_a_field(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
        qwen = FakeQwen(KRYSHKA, KRYSHKA_LINES,
                        corner={"sign": True, "value": "Ra 6,3", "bracket_sign": True})
        before = by_value(read(storage, FakeQwen(KRYSHKA, KRYSHKA_LINES)))
        result = read(storage, qwen)

        assert [kind for kind, _ in qwen.asked] == ["corner"]
        general = by_value(result)["Ra 6,3"]
        assert general["category"] == "roughness"
        assert general["kind"] == kinds.KIND_ROUGHNESS_UNSPECIFIED
        corner = result["payload"]["corner_roughness"]
        assert corner["value"] == "Ra 6,3" and corner["bracket_sign"] is True
        assert corner["evidence"] == ["corner_crop", "transcript_line"]
        assert general["bbox"] == corner["bbox"]
        # Прежние поля не тронуты.
        assert {value: f for value, f in by_value(result).items() if value != "Ra 6,3"} == {
            value: f for value, f in before.items()}
        # Поле проходит модель контракта.
        assert DrawingField(**general).kind_label == "Шероховатость неуказанных поверхностей"

    def test_refusal_is_recorded(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
        result = read(storage, FakeQwen(KRYSHKA, KRYSHKA_LINES, corner={"sign": False}))
        assert result["payload"]["corner_roughness"]["refused"] == kinds.REFUSED_NO_SIGN
        assert "Ra 6,3" not in by_value(result)

    def test_silent_model_is_recorded(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
        result = read(storage, FakeQwen(KRYSHKA, KRYSHKA_LINES, corner=None))
        assert result["payload"]["corner_roughness"]["refused"] == kinds.REFUSED_NOT_READ

    def test_view_cutout_has_no_corner(self, storage, monkeypatch):
        """Вырезка вида — не лист: у неё нет ни угла поля, ни заголовков листа."""
        monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
        monkeypatch.setattr(settings, "DRAWING_VALUE_KIND", True)
        storage.write_file("assets/doc-1/crop.png", sheet.png_bytes())
        qwen = FakeQwen(KRYSHKA, KRYSHKA_LINES, corner={"sign": True, "value": "Ra 6,3"})
        result = read(storage, qwen, image_ref="assets/doc-1/crop.png")
        assert qwen.asked == []
        assert "corner_roughness" not in result["payload"]

    def test_view_and_chamfer(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_VALUE_KIND", True)
        annotations = KRYSHKA + [{"category": "size", "value": "2×45°", "bbox": [0.5, 0.5, 0.55, 0.52]}]
        qwen = FakeQwen(annotations, KRYSHKA_LINES, views={"views": [
            {"title": "A (1:2)", "values": ["⌀8,5", "17*", "(X+12)*"]}]})
        result = read(storage, qwen)

        assert qwen.asked == [("views", ["A (1:2)"])]
        fields = by_value(result)
        assert fields["⌀8,5"]["view"] == "A (1:2)"
        # «17*» стоит выше заголовка «A (1:2)», а «(X+12)*» — слишком далеко
        # под ним: место с ответом не согласно, подписи нет.
        assert "view" not in fields["17*"] and "view" not in fields["(X+12)*"]
        assert fields["2×45°"]["kind"] == kinds.KIND_CHAMFER
        membership = result["payload"]["view_membership"]
        assert membership["labelled"] == 1 and membership["read"] is True
        assert [(item["value"], item["reason"]) for item in membership["refused"]] == [
            ("17*", kinds.REFUSED_POSITION), ("(X+12)*", kinds.REFUSED_POSITION)]

    def test_no_time_no_question(self, storage, monkeypatch):
        """Бюджет листа исчерпан: модель не спрашивается, отказ записан."""
        monkeypatch.setattr(settings, "DRAWING_CORNER_ROUGHNESS", True)
        monkeypatch.setattr(settings, "DRAWING_VALUE_KIND", True)
        monkeypatch.setattr(DrawingProcessor, "_can_ask", lambda self, started: False)
        qwen = FakeQwen(KRYSHKA, KRYSHKA_LINES, corner={"sign": True, "value": "Ra 6,3"})
        result = read(storage, qwen)
        assert qwen.asked == []
        assert result["payload"]["corner_roughness"]["refused"] == kinds.REFUSED_NOT_READ
        assert result["payload"]["view_membership"]["read"] is False
        assert all("kind" not in f and "view" not in f for f in result["parsed_fields"])

    def test_sheet_without_titles_is_not_asked(self, storage, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_VALUE_KIND", True)
        qwen = FakeQwen(KRYSHKA[1:], KRYSHKA_LINES)
        result = read(storage, qwen)
        assert qwen.asked == []
        assert "view_membership" not in result["payload"]


class TestCorner:
    def test_corner_is_inside_the_frame(self):
        from PIL import Image

        layout = drawing_vision.SheetLayout(
            image=Image.new("L", (3307, 2338), 255), frame=[0.0631, 0.0248, 0.9792, 0.9664])
        left, top, right, bottom = drawing_vision.roughness_corner(layout)
        assert (right, top) == (0.9792, 0.0248)
        # Знак «Крышки» лежит в этой вырезке целиком.
        assert left < 0.865 and bottom > 0.085

    def test_no_frame_no_corner(self):
        from PIL import Image

        layout = drawing_vision.SheetLayout(image=Image.new("L", (100, 100), 255))
        assert drawing_vision.roughness_corner(layout) is None
