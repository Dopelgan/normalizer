"""
Оборванный ответ прохода «лист целиком» (DRAWING_SHEET_HIGH_TOKENS,
DRAWING_SHEET_TILES, DRAWING_SHEET_CUT_VISIBLE).

Список надписей плотного листа не влезает в потолок ответа: JSON обрывается
посреди списка, спасаются только дописанные надписи, а хвост пропадал молча —
оборванный ответ оставался успешным ответом. Модель здесь поддельная: у неё
ограничен ответ, и лист, на котором надписей больше, она обрывает посреди
очередной надписи, как vLLM с finish_reason=length. Места надписей она
отвечает честно — по центру тяжести своей четверти на присланной картинке,
поэтому надпись, прочитанная и на листе, и в части листа, приходит в одном
и том же месте листа.
"""

import base64
import io
import json

import pytest
import requests

from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.drawing_quality import document_quality
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers.qwen_client import (
    QwenVisionClient, merge_annotations, place_in_sheet,
)

pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

# Четверти листа окрашены по-разному: по цветам на картинке поддельная
# модель понимает, какую часть листа ей прислали и где она на ней.
COLORS = {
    "top_left": (255, 0, 0), "top_right": (0, 255, 0),
    "bottom_left": (0, 0, 255), "bottom_right": (255, 255, 0),
}
# В каждой четверти шесть своих надписей и один общий «⌀8»: на чертеже такой
# размер стоит у каждого отверстия, и четыре «⌀8» — это четыре размера.
TEXTS = {name: [f"{name} надпись {index}" for index in range(6)] + ["⌀8"]
         for name in COLORS}
ALL_TEXTS = [text for name in COLORS for text in TEXTS[name]]


def quartered(size=(800, 600)):
    width, height = size
    image = Image.new("RGB", size)
    half_w, half_h = width // 2, height // 2
    image.paste(COLORS["top_left"], (0, 0, half_w, half_h))
    image.paste(COLORS["top_right"], (half_w, 0, width, half_h))
    image.paste(COLORS["bottom_left"], (0, half_h, half_w, height))
    image.paste(COLORS["bottom_right"], (half_w, half_h, width, height))
    return image


def sent_image(body):
    url = body["messages"][1]["content"][0]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB")


def shown_parts(image):
    """Четверти, которые занимают заметную долю присланной картинки."""
    counts = dict((color, count) for count, color in image.getcolors(1 << 24))
    total = image.size[0] * image.size[1]
    return [name for name, color in COLORS.items() if counts.get(color, 0) / total > 0.2]


def centroid(image, color):
    """Середина пятна цвета в долях присланной картинки."""
    width, height = image.size
    pixels = image.load()
    xs, ys, count = 0, 0, 0
    for y in range(0, height, 4):
        for x in range(0, width, 4):
            if pixels[x, y] == color:
                xs, ys, count = xs + x, ys + y, count + 1
    if not count:
        return 0.5, 0.5
    return xs / count / width, ys / count / height


def annotation(text, spot):
    cx, cy = spot
    return {"category": "size", "value": text,
            "bbox": [round(cx - 0.02, 4), round(cy - 0.02, 4),
                     round(cx + 0.02, 4), round(cy + 0.02, 4)]}


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code, self.text = payload, status, ""

    def json(self):
        return self.payload


def reply(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class SheetModel:
    """
    Отвечает надписями тех четвертей, что видны на картинке. Если надписей
    больше `capacity`, ответ обрывается посреди очередной надписи.
    `fail` — четверти, на которых модель не отвечает (обрыв связи);
    `cut_always` — обрывает ответ и по части листа тоже.
    """

    def __init__(self, capacity=10, fail=(), cut_always=False, cut_parts=()):
        self.capacity = capacity
        self.fail = set(fail)
        self.cut_always = cut_always
        # Четверти, ответ по которым обрывается и отдельным вызовом: так
        # ведёт себя лист, у которого вся плотность собрана в одном углу
        # (лист-таблица, спецификация).
        self.cut_parts = set(cut_parts)
        self.calls = []

    def post(self, url, json=None, headers=None, timeout=None):
        image = sent_image(json)
        parts = shown_parts(image)
        self.calls.append({"parts": parts, "max_tokens": json["max_tokens"]})
        if parts and set(parts) <= self.fail:
            raise requests.ConnectionError("обрыв")
        items = [annotation(text, centroid(image, COLORS[name]))
                 for name in parts for text in TEXTS[name]]
        dense = bool(parts) and set(parts) <= self.cut_parts
        if len(items) <= self.capacity and not self.cut_always and not dense:
            return Response(reply(_whole(items)))
        kept = items[: self.capacity] if len(items) > self.capacity else items[:-1]
        return Response(reply(_cut(kept), finish="length"))

    @property
    def sheet_calls(self):
        return [call for call in self.calls if len(call["parts"]) == 4]

    @property
    def tile_calls(self):
        return [call for call in self.calls if len(call["parts"]) < 4]


def _whole(items):
    return json.dumps({"sheet_type": "detail", "annotations": items}, ensure_ascii=False)


def _cut(kept):
    """Ответ, оборванный посреди очередной надписи, — как у vLLM."""
    head = '{"sheet_type": "detail", "annotations": ['
    body = ", ".join(json.dumps(item, ensure_ascii=False) for item in kept)
    return head + body + ', {"category": "size", "value": "незаконч'


def client(model):
    return QwenVisionClient(endpoint="http://qwen:8000/v1", model="fake-vl", session=model)


@pytest.fixture(autouse=True)
def quiet(monkeypatch):
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)
    monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", False)
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", False)


@pytest.fixture
def tiles(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", True)


@pytest.fixture
def visible(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_SHEET_CUT_VISIBLE", True)


# --------------------------------------------------------------- слияние
class TestMergeAnnotations:
    """Слияние списков надписей частей листа: ключ — текст и место."""

    def test_same_text_in_the_same_place_is_one_annotation(self):
        left = [annotation("⌀44H7", (0.50, 0.40))]
        right = [annotation("⌀44H7", (0.51, 0.41))]
        assert len(merge_annotations([left, right], near=0.05)) == 1

    def test_same_text_elsewhere_on_the_sheet_is_kept(self):
        """«⌀8» у каждого отверстия — восемь размеров, а не один."""
        first = [annotation("⌀8", (0.20, 0.20))]
        second = [annotation("⌀8", (0.80, 0.80))]
        merged = merge_annotations([first, second], near=0.05)
        assert [a["bbox"] for a in merged] == [first[0]["bbox"], second[0]["bbox"]]

    def test_repeat_inside_one_part_in_the_same_place_is_dropped(self):
        part = [annotation("Ra 6,3", (0.3, 0.3)), annotation("Ra 6,3", (0.3, 0.3))]
        assert len(merge_annotations([part], near=0.05)) == 1

    def test_order_of_parts_is_kept(self):
        parts = [[annotation("A", (0.1, 0.1))], [annotation("B", (0.9, 0.9))]]
        assert [a["value"] for a in merge_annotations(parts, near=0.05)] == ["A", "B"]

    def test_annotation_without_a_place_falls_back_to_the_sheet_centre(self):
        """Без координат два одинаковых текста не различить — остаётся один."""
        parts = [[{"category": "note", "value": "см. п.4"}],
                 [{"category": "note", "value": "см. п.4"}]]
        assert len(merge_annotations(parts, near=0.05)) == 1

    def test_not_a_dictionary_is_skipped(self):
        assert merge_annotations([["⌀8", None, 7]], near=0.05) == []

    def test_text_is_compared_without_case_and_extra_spaces(self):
        parts = [[annotation("Ra  6,3", (0.3, 0.3))], [annotation("ra 6,3", (0.3, 0.3))]]
        assert len(merge_annotations([*parts], near=0.05)) == 1


class TestPlaceInSheet:
    """Место надписи из долей части листа в доли всего листа."""

    def test_centre_of_a_quarter_becomes_a_quarter_of_the_sheet(self):
        placed = place_in_sheet(annotation("⌀8", (0.5, 0.5)), [0.0, 0.0, 0.5, 0.5])
        assert placed["bbox"] == [0.24, 0.24, 0.26, 0.26]

    def test_bottom_right_quarter_is_shifted(self):
        placed = place_in_sheet(annotation("⌀8", (0.5, 0.5)), [0.5, 0.5, 1.0, 1.0])
        assert placed["bbox"] == [0.74, 0.74, 0.76, 0.76]

    def test_pixels_are_left_alone_for_the_parser_to_reject(self):
        pixel = {"category": "size", "value": "⌀8", "bbox": [100, 200, 140, 220]}
        assert place_in_sheet(pixel, [0.5, 0.5, 1.0, 1.0])["bbox"] == [100, 200, 140, 220]

    def test_annotation_without_a_bbox_is_returned_as_is(self):
        plain = {"category": "note", "value": "см. п.4"}
        assert place_in_sheet(plain, [0.5, 0.5, 1.0, 1.0]) == plain

    def test_the_original_annotation_is_not_mutated(self):
        original = annotation("⌀8", (0.5, 0.5))
        before = list(original["bbox"])
        place_in_sheet(original, [0.5, 0.5, 1.0, 1.0])
        assert original["bbox"] == before


# ----------------------------------------------------------- потолок ответа
class TestOutputLimit:
    def test_flag_off_keeps_the_shared_ceiling(self):
        model = SheetModel(capacity=100)
        client(model).read_sheet(quartered())
        assert model.calls[0]["max_tokens"] == settings.QWEN_MAX_TOKENS

    def test_flag_on_gives_the_pass_its_own_ceiling(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SHEET_HIGH_TOKENS", True)
        model = SheetModel(capacity=100)
        client(model).read_sheet(quartered())
        assert model.calls[0]["max_tokens"] == settings.DRAWING_SHEET_MAX_TOKENS
        assert settings.DRAWING_SHEET_MAX_TOKENS > settings.QWEN_MAX_TOKENS

    def test_the_raised_ceiling_fits_the_model_context(self):
        """Запрос с листом — около 4.4 тыс. токенов, --max-model-len 32768."""
        assert settings.DRAWING_SHEET_MAX_TOKENS + 4400 < 32768

    def test_other_passes_keep_their_own_ceilings(self, monkeypatch):
        """Потолок прохода «лист целиком» не трогает проход «весь текст»."""
        monkeypatch.setattr(settings, "DRAWING_SHEET_HIGH_TOKENS", True)
        model = SheetModel(capacity=100)
        client(model).transcribe(quartered())
        assert model.calls[0]["max_tokens"] == settings.DRAWING_TRANSCRIBE_MAX_TOKENS


# --------------------------------------------------------------- обрыв
class TestCutAnswer:
    def test_whole_answer_is_not_called_cut(self):
        reading = client(SheetModel(capacity=100)).read_sheet(quartered())
        assert len(reading.annotations) == len(ALL_TEXTS)
        assert not reading.cut and reading.complete

    def test_cut_answer_keeps_the_finished_annotations(self):
        reading = client(SheetModel(capacity=10)).read_sheet(quartered())
        assert len(reading.annotations) == 10
        assert reading.cut and not reading.complete

    def test_flag_off_does_not_read_the_sheet_in_parts(self):
        model = SheetModel(capacity=10)
        client(model).read_sheet(quartered())
        assert len(model.calls) == 1

    def test_silent_model_is_still_nothing(self):
        model = SheetModel(fail=COLORS)
        assert client(model).read_sheet(quartered()) is None


class TestReadingInParts:
    def test_cut_sheet_is_read_in_parts_and_the_tail_comes_back(self, tiles):
        model = SheetModel(capacity=10)
        reading = client(model).read_sheet(quartered())
        assert len(model.sheet_calls) == 1 and len(model.tile_calls) == 4
        assert sorted(a["value"] for a in reading.annotations) == sorted(ALL_TEXTS)
        assert reading.tiles == 4 and reading.complete

    def test_one_dimension_in_two_parts_is_not_counted_twice(self, tiles):
        """«⌀8» стоит в каждой четверти: четыре размера, а не восемь."""
        reading = client(SheetModel(capacity=10)).read_sheet(quartered())
        assert [a["value"] for a in reading.annotations].count("⌀8") == 4

    def test_parts_land_in_their_places_on_the_sheet(self, tiles):
        reading = client(SheetModel(capacity=10)).read_sheet(quartered())
        spots = {a["value"]: a["bbox"] for a in reading.annotations}
        assert spots["top_left надпись 0"][0] < 0.5
        assert spots["bottom_right надпись 0"][0] > 0.5
        assert spots["bottom_right надпись 0"][1] > 0.5

    def test_whole_answer_is_not_read_in_parts_at_all(self, tiles):
        model = SheetModel(capacity=100)
        client(model).read_sheet(quartered())
        assert len(model.calls) == 1

    def test_part_that_is_cut_again_is_not_read_deeper(self, tiles):
        """Цена ограничена: лист плюс четыре части, без делений вглубь."""
        model = SheetModel(capacity=10, cut_always=True)
        reading = client(model).read_sheet(quartered())
        assert len(model.calls) == 5
        assert len(reading.failed) == 4 and not reading.complete

    def test_part_without_an_answer_is_a_failed_part(self, tiles):
        model = SheetModel(capacity=10, fail=["bottom_right"])
        reading = client(model).read_sheet(quartered())
        assert reading.tiles == 3 and len(reading.failed) == 1
        assert not reading.complete

    def test_no_time_left_leaves_the_parts_unread(self, tiles, monkeypatch):
        import time as clock
        model = SheetModel(capacity=10)
        reading = client(model).read_sheet(quartered(), until=clock.monotonic() - 1)
        assert len(model.tile_calls) == 0
        assert len(reading.failed) == 4 and not reading.complete


# ------------------------------------------------------- видимость обрыва
def processor(model, monkeypatch):
    """
    Разбор без зрения: размечать нечего, и лист уходит в модель как есть.
    Так проверяется один проход «лист целиком», а не геометрия.
    """
    worker = DrawingProcessor(storage=object(), qwen=client(model))
    worker.vision_enabled = False
    monkeypatch.setattr(worker, "_open", lambda ref, page=1: quartered())
    return worker


def source_fragment():
    return Fragment(
        fragment_id="f1", type="image",
        content=Content(image_ref="sheet.jpg"),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.5, completeness=None,
        provenance=Provenance(method="mineru", strategy_level=1,
                              source="image", source_unit_id="u1"),
    )


def built(model, monkeypatch):
    worker = processor(model, monkeypatch)
    result = worker.process("sheet.jpg", {}, page=1)
    return worker.build_fragment(result, source_fragment(), {})


class TestCutIsVisible:
    def test_flag_off_a_cut_answer_still_passes_as_before(self, monkeypatch):
        """Выключенный флаг ничего не меняет: ни пометки, ни области."""
        fragment = built(SheetModel(capacity=10), monkeypatch)
        assert "DRAWING_SHEET_ANSWER_CUT" not in fragment.warnings
        assert fragment.missing_regions == []
        assert "sheet_answer_cut" not in fragment.content.structured_payload

    def test_cut_sheet_carries_a_warning_and_an_unread_region(
        self, monkeypatch, visible,
    ):
        fragment = built(SheetModel(capacity=10), monkeypatch)
        assert "DRAWING_SHEET_ANSWER_CUT" in fragment.warnings
        assert [r["kind"] for r in fragment.missing_regions] == ["sheet_tile"]
        payload = fragment.content.structured_payload
        assert payload["sheet_answer_cut"] is True
        assert payload["sheet_annotations_complete"] is False

    def test_cut_sheet_makes_the_document_parsed_incompletely(
        self, monkeypatch, visible,
    ):
        """И при DRAWING_UNMEASURED_NOT_PARTIAL: это реальная потеря."""
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10), monkeypatch)
        partial, warnings = document_quality([fragment])
        assert partial and "DRAWING_SHEET_ANSWER_CUT" in warnings
        assert fragment.review_required

    def test_whole_answer_is_not_marked(self, monkeypatch, visible):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=100), monkeypatch)
        assert "DRAWING_SHEET_ANSWER_CUT" not in fragment.warnings
        assert fragment.missing_regions == []
        assert document_quality([fragment])[0] is False

    def test_sheet_read_in_parts_is_not_marked_as_lost(
        self, monkeypatch, tiles, visible,
    ):
        """Дочитанный частями лист прочитан: пометки потери на нём нет."""
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10), monkeypatch)
        assert "DRAWING_SHEET_ANSWER_CUT" not in fragment.warnings
        assert fragment.missing_regions == []
        assert fragment.content.structured_payload["sheet_tiles"] == 4
        assert document_quality([fragment])[0] is False

    def test_part_that_stayed_cut_is_named_as_an_unread_region(
        self, monkeypatch, tiles, visible,
    ):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10, cut_always=True), monkeypatch)
        assert "DRAWING_SHEET_ANSWER_CUT" in fragment.warnings
        kinds = [r["kind"] for r in fragment.missing_regions]
        assert kinds == ["sheet_tile"] * 4
        assert document_quality([fragment])[0] is True

    def test_dense_corner_leaves_only_that_part_unread(
        self, monkeypatch, tiles, visible,
    ):
        """
        Лист-таблица: вся плотность в одном углу, и часть с ней обрывается
        и отдельным вызовом. Три части прочитаны, недочитана одна — она и
        названа, а не весь лист.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        model = SheetModel(capacity=10, cut_parts=["bottom_right"])
        fragment = built(model, monkeypatch)
        regions = fragment.missing_regions
        assert [r["kind"] for r in regions] == ["sheet_tile"]
        # Недочитана правая нижняя часть: её bbox начинается за серединой.
        assert regions[0]["bbox"][0] > 0.4 and regions[0]["bbox"][1] > 0.4
        assert "DRAWING_SHEET_ANSWER_CUT" in fragment.warnings
        assert document_quality([fragment])[0] is True

    def test_more_annotations_come_back_than_before(self, monkeypatch):
        """Та же плотная страница: до — спасённый хвост, после — весь лист."""
        before = built(SheetModel(capacity=10), monkeypatch)
        monkeypatch.setattr(settings, "DRAWING_SHEET_TILES", True)
        after = built(SheetModel(capacity=10), monkeypatch)
        assert len(after.content.structured_drawing_fields) > len(
            before.content.structured_drawing_fields)


# ----------------------------------------------------------- виды листов
class TestSheetShapes:
    """
    Листы, которых не было в разборе сегодняшнего дня: вытянутый A1/A0,
    многолистовой документ, лист-фотография. Проверяется не модель, а то,
    что от формы листа зависят места надписей и ничего больше.
    """

    # A0 — 1189 x 841 мм: доля стороны у такого листа та же, а вот пиксели
    # у частей совсем разные, и перенос места обязан это учитывать.
    A0 = (1189, 841)

    def test_long_sheet_is_tiled_and_annotations_keep_their_places(self, tiles):
        reading = client(SheetModel(capacity=10)).read_sheet(quartered(self.A0))
        assert sorted(a["value"] for a in reading.annotations) == sorted(ALL_TEXTS)
        spots = {a["value"]: a["bbox"] for a in reading.annotations}
        assert spots["top_right надпись 0"][0] > 0.5
        assert spots["top_right надпись 0"][1] < 0.5
        assert spots["bottom_left надпись 0"][0] < 0.5
        assert spots["bottom_left надпись 0"][1] > 0.5

    def test_long_sheet_annotations_stay_inside_the_sheet(self, tiles):
        reading = client(SheetModel(capacity=10)).read_sheet(quartered(self.A0))
        for item in reading.annotations:
            assert all(0.0 <= value <= 1.0 for value in item["bbox"])

    def test_narrow_sheet_is_tiled_the_same_way(self, tiles):
        """Лист A4 стоймя: сетка та же, доли те же."""
        reading = client(SheetModel(capacity=10)).read_sheet(quartered((595, 842)))
        assert sorted(a["value"] for a in reading.annotations) == sorted(ALL_TEXTS)

    def test_photograph_without_a_cut_costs_nothing(self, tiles, visible):
        """
        Фотография чертежа: надписей мало, ответ целый — ни одного лишнего
        вызова, ни пометки, ни непрочитанной области.
        """
        model = SheetModel(capacity=100)
        reading = client(model).read_sheet(quartered((1200, 900)))
        assert len(model.calls) == 1
        assert reading.complete and not reading.cut and not reading.failed

    def test_each_page_of_a_multi_sheet_document_is_judged_on_its_own(
        self, monkeypatch, tiles, visible,
    ):
        """
        Многолистовой PDF: плотный лист помечается, соседний целый — нет.
        Пометка и непрочитанная область живут во фрагменте своего листа.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        dense = built(SheetModel(capacity=10, cut_always=True), monkeypatch)
        plain = built(SheetModel(capacity=100), monkeypatch)
        plain.fragment_id, plain.position.page = "f2", 2
        for region in dense.missing_regions:
            region["page"] = 1
        assert "DRAWING_SHEET_ANSWER_CUT" in dense.warnings
        assert "DRAWING_SHEET_ANSWER_CUT" not in plain.warnings
        assert plain.missing_regions == []
        # Документ уходит человеку из-за одного листа, а не из-за всех.
        assert document_quality([dense, plain])[0] is True
        assert document_quality([plain])[0] is False
