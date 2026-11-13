"""
Оборванный ответ прохода «лист целиком» (DRAWING_SHEET_HIGH_TOKENS,
DRAWING_SHEET_TILES) и учёт потерянных прочтений.

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
    QwenVisionClient, as_annotations, merge_annotations, place_in_sheet,
    salvage_list,
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

    def __init__(self, capacity=10, fail=(), cut_always=False, cut_parts=(), junk=()):
        # Элементы списка, которые надписью стать не могут (доли bbox,
        # пришедшие отдельными элементами): они обязаны дойти до человека.
        self.junk = list(junk)
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
        items += self.junk
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


# ------------------------------------------- элементы списка не только объекты
# Настоящий лист: «Паспорт ТАДУ 405210.001ПС Зав (2).pdf», стр. 6 — таблица
# технических характеристик. 13.11.2026 проход «лист целиком» на нём
# оборвался, из списка спасено 108 элементов, а до разбора дошли 24: всё,
# что не объект, выбрасывал `isinstance(a, dict)`. Значения ниже — те самые
# 24, прочитанные из сохранённого payload на проде.
REAL_SHEET = [
    ("note", "2 ОСНОВНЫЕ ТЕХНИЧЕСКИЕ ХАРАКТЕРИСТИКИ", [0.92, 0.04, 0.96, 0.37]),
    ("note", "2.1 Условное обозначение изделия", [0.52, 0.04, 0.92, 0.06]),
    ("note", "СБ210/СП-01-50П-В- 4.0.32/К03-0.12", [0.42, 0.04, 0.52, 0.1]),
    ("note", "2.2 Номинальная статическая характеристика преобразования (НСХ) "
             "по ГОСТ 6651-2009", [0.52, 0.06, 0.92, 0.1]),
    ("note", "50П", [0.42, 0.1, 0.52, 0.11]),
    ("note", "2.3 Рабочий диапазон измеряемых температур, °С", [0.52, 0.1, 0.92, 0.12]),
    ("note", "От -50 до + 400", [0.42, 0.12, 0.52, 0.14]),
    ("note", "2.4 Классы допуска по ГОСТ 6651-2009", [0.52, 0.12, 0.92, 0.14]),
    ("note", "В", [0.42, 0.14, 0.52, 0.15]),
    ("note", "2.9 Длина монтажной части, Л, м", [0.52, 0.27, 0.92, 0.29]),
    ("note", "0,32", [0.42, 0.29, 0.52, 0.3]),
    ("note", "2.12 Масса, не более, кг", [0.52, 0.33, 0.92, 0.35]),
]
# Значения, которые на этом листе шли дальше по таблице и пропадали.
REAL_TAIL = ["2.13 Степень защиты по ГОСТ 14254-2015", "IP65",
             "2.14 Средний срок службы, лет", "10"]


def real_annotations():
    return [{"category": c, "value": v, "bbox": b} for c, v, b in REAL_SHEET]


class TestSalvagedShapesAreNotDropped:
    """
    `salvage_list` отдаёт то, что возвращает `json.JSONDecoder.raw_decode`:
    объект, массив, строку, число, true/false/null. Виды перечислены по
    коду, а не наугад; каким именно был пропавший хвост того листа, по
    сохранённому ответу уже не узнать — сырые ответы модели не хранятся.
    """

    def test_objects_are_annotations_as_before(self):
        readable, lost = as_annotations(real_annotations())
        assert len(readable) == len(REAL_SHEET) and lost == {}

    def test_bare_strings_become_annotations_without_a_place(self):
        """Модель бросила обёртку объекта и перечисляет значения подряд."""
        readable, lost = as_annotations(real_annotations() + REAL_TAIL)
        assert lost == {}
        values = [a["value"] for a in readable]
        assert values[-len(REAL_TAIL):] == REAL_TAIL
        assert "bbox" not in readable[-1]

    def test_a_table_row_as_an_array_is_read_cell_by_cell(self):
        readable, lost = as_annotations([["2.13 Степень защиты", "IP65"]])
        assert [a["value"] for a in readable] == ["2.13 Степень защиты", "IP65"]
        assert lost == {}

    def test_nested_objects_in_an_array_are_read(self):
        readable, lost = as_annotations([real_annotations()[:2]])
        assert len(readable) == 2 and lost == {}

    def test_a_stray_bbox_is_counted_lost_not_turned_into_text(self):
        """«0.52» — доля из bbox, а не прочтение листа: в индекс не идёт."""
        readable, lost = as_annotations([[0.52, 0.04, 0.92, 0.06]])
        assert readable == [] and sum(lost.values()) == 4

    def test_the_category_token_does_not_become_a_field(self):
        """
        Та самая поломка, под которую написан разбор: поля схемы идут по
        порядку {category, value, bbox}, и, бросив обёртку, модель
        перечисляет их все. «size» и «note» — разметка, а не прочтение, но
        `looks_like_text` их пропускает, и полями они доходили до индекса.
        """
        stream = ('{"annotations": ["size", "D80", [0.1,0.2,0.3,0.4], '
                  '"note", "Stal 3", "dimension", "110"')
        items = salvage_list(stream, "annotations")
        assert items == ["size", "D80", [0.1, 0.2, 0.3, 0.4],
                         "note", "Stal 3", "dimension", "110"]
        readable, lost = as_annotations(items)
        # Остаются прочтения, уходят слова схемы и доли bbox.
        assert [a["value"] for a in readable] == ["D80", "Stal 3", "110"]
        assert lost["слово схемы"] == 3     # size, note, dimension
        assert lost["float"] == 4           # доли bbox

    def test_every_category_name_is_refused(self):
        from core.providers.qwen_client import CATEGORIES
        readable, lost = as_annotations(list(CATEGORIES))
        assert readable == []
        assert lost["слово схемы"] == len(CATEGORIES)

    def test_sheet_type_words_are_refused_too(self):
        readable, _ = as_annotations(["detail", "assembly", "scheme", "unknown"])
        assert readable == []

    def test_a_quoted_coordinate_is_refused(self):
        """«0.1» в кавычках обходил отказ от чисел и ничего не считал."""
        readable, lost = as_annotations([["0.1", "0.2"]])
        assert readable == [] and lost["доля bbox строкой"] == 2

    def test_readings_that_look_like_numbers_are_kept(self):
        """
        На чертеже по ЕСКД десятичный разделитель запятая: «0,32» —
        настоящее прочтение с листа-паспорта, а «0.32» — доля из bbox.
        Целые числа не отбрасываются вовсе: «110» — размер, «1» — позиция.
        """
        keep = ["0,32", "0,12", "0,35", "110", "1", "1.5", "50П", "Ra 6,3",
                "M20x1,5", "H7", "IP65", "УХЛ4", "12Х18Н10Т", "⌀44H7",
                "От -50 до + 400", "1:2", "Ст3"]
        readable, lost = as_annotations(keep)
        assert [a["value"] for a in readable] == keep
        assert lost == {}

    def test_refusals_are_counted_so_over_refusal_would_show(self):
        """Отказ не тишина: он виден числом в логе и в payload."""
        _, lost = as_annotations(["size", "0.1", ""])
        assert sum(lost.values()) == 3

    def test_numbers_and_nulls_are_counted_lost(self):
        readable, lost = as_annotations([1, 2.5, None, True])
        assert readable == [] and sum(lost.values()) == 4

    def test_empty_strings_and_arrays_are_counted_lost(self):
        readable, lost = as_annotations(["", "   ", []])
        assert readable == [] and sum(lost.values()) == 3

    def test_lost_shapes_are_named_for_the_log(self):
        _, lost = as_annotations([1, None, ""])
        assert set(lost) == {"int", "NoneType", "пустая строка"}

    def test_the_real_sheet_loses_nothing_it_used_to_lose(self):
        """
        Тот же лист: раньше из объектов и хвоста строк доходили только
        объекты, теперь — всё, что на листе прочитано.
        """
        items = real_annotations() + REAL_TAIL
        before = [a for a in items if isinstance(a, dict)]
        readable, lost = as_annotations(items)
        assert len(before) == len(REAL_SHEET)
        assert len(readable) == len(REAL_SHEET) + len(REAL_TAIL)
        assert lost == {}

    def test_a_cut_answer_of_the_real_shape_salvages_then_converts(self):
        """От оборванного текста ответа до надписей — весь путь целиком."""
        body = ", ".join(
            json.dumps({"category": c, "value": v, "bbox": b}, ensure_ascii=False)
            for c, v, b in REAL_SHEET)
        tail = ", ".join(json.dumps(v, ensure_ascii=False) for v in REAL_TAIL)
        content = ('{"sheet_type": "detail", "annotations": [' + body + ", " + tail
                   + ', {"category": "note", "value": "2.15 Габарит')
        items = salvage_list(content, "annotations")
        assert len(items) == len(REAL_SHEET) + len(REAL_TAIL)
        readable, lost = as_annotations(items)
        assert len(readable) == len(items) and lost == {}


class TestLostElementsReachAHuman:
    def test_lost_elements_make_the_reading_incomplete(self):
        from core.providers.qwen_client import SheetReading
        assert SheetReading(lost=3).complete is False
        assert SheetReading().complete is True


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


class TestNobodyIsCalledForParseQuality:
    """
    Файл уходит человеку только за тем, чего он один и может решить:
    кластер, имя, конфликт версий. Насколько хорошо прочитан лист — наша
    забота, поэтому обрыв ответа не ставит ни пометки в `warnings`, ни
    области в `missing_regions` и документ partial от него не становится.
    Лист при этом не изображает прочитанным целиком: своими данными он
    говорит правду и просит машинного перечитывания.
    """

    def test_a_cut_answer_changes_nothing_a_person_would_see(self, monkeypatch):
        """Пометки и области остаются ровно такими, как без этой ветки."""
        fragment = built(SheetModel(capacity=10), monkeypatch)
        assert fragment.warnings == ["DRAWING_COMPLETENESS_UNKNOWN"]
        assert fragment.missing_regions == []

    def test_the_count_is_stored_without_any_flag(self, monkeypatch):
        """
        Учёт и эскалация развязаны: считать обрывы нужно по сохранённым
        данным, а не грепом по логам воркера, поэтому ключи пишутся всегда.
        """
        payload = built(SheetModel(capacity=10), monkeypatch).content.structured_payload
        assert payload["sheet_answer_cut"] is True
        assert payload["sheet_annotations_complete"] is False
        assert payload["sheet_reparse_candidate"] is True

    def test_cut_sheet_is_recorded_but_sends_nobody_anywhere(
        self, monkeypatch,
    ):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10), monkeypatch)
        payload = fragment.content.structured_payload
        # Учёт есть.
        assert payload["sheet_answer_cut"] is True
        assert payload["sheet_annotations_complete"] is False
        # Человека не зовут.
        assert fragment.warnings == ["DRAWING_COMPLETENESS_UNKNOWN"]
        assert fragment.missing_regions == []
        assert document_quality([fragment])[0] is False

    def test_a_sheet_with_lost_readings_asks_to_be_reparsed(
        self, monkeypatch,
    ):
        """Правильное автоматическое действие — перечитать, а не спросить."""
        fragment = built(SheetModel(capacity=10), monkeypatch)
        payload = fragment.content.structured_payload
        assert payload["sheet_reparse_candidate"] is True
        assert payload["sheet_reparse_reason"] == "sheet_answer_cut"

    def test_unread_tiles_are_named_for_the_reparse(
        self, monkeypatch, tiles,
    ):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10, cut_always=True), monkeypatch)
        payload = fragment.content.structured_payload
        assert payload["sheet_reparse_reason"] == "sheet_tiles_unread"
        assert len(payload["sheet_unread_tiles"]) == 4
        # И всё равно никого не зовут.
        assert fragment.missing_regions == []
        assert document_quality([fragment])[0] is False

    def test_whole_answer_is_not_recorded_at_all(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=100), monkeypatch)
        payload = fragment.content.structured_payload
        assert "sheet_answer_cut" not in payload
        assert "sheet_reparse_candidate" not in payload
        assert document_quality([fragment])[0] is False

    def test_sheet_recovered_in_parts_is_not_a_reparse_candidate(
        self, monkeypatch, tiles,
    ):
        """
        Дочитанный частями лист прочитан: перечитывать нечего, записывать
        потерю нечего, спрашивать некого.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        fragment = built(SheetModel(capacity=10), monkeypatch)
        payload = fragment.content.structured_payload
        assert payload["sheet_tiles"] == 4
        assert payload["sheet_answer_cut"] is True       # обрыв был
        assert "sheet_reparse_candidate" not in payload  # но текст вернулся
        assert "sheet_annotations_complete" not in payload
        assert fragment.missing_regions == []
        assert document_quality([fragment])[0] is False

    def test_dense_corner_names_only_that_part_for_the_reparse(
        self, monkeypatch, tiles,
    ):
        """
        Лист-таблица: вся плотность в одном углу, и часть с ней обрывается
        и отдельным вызовом. Недочитана одна часть — она и названа.
        """
        model = SheetModel(capacity=10, cut_parts=["bottom_right"])
        payload = built(model, monkeypatch).content.structured_payload
        boxes = payload["sheet_unread_tiles"]
        assert len(boxes) == 1
        assert boxes[0][0] > 0.4 and boxes[0][1] > 0.4

    def test_lost_elements_are_recorded_even_on_a_whole_answer(
        self, monkeypatch,
    ):
        """
        Ответ целый, но часть элементов списка — не прочтения (доли bbox
        отдельными элементами). Молча они больше не пропадают — и человека
        всё равно не зовут.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        model = SheetModel(capacity=100, junk=[0.52, 0.04, None])
        fragment = built(model, monkeypatch)
        payload = fragment.content.structured_payload
        assert payload["sheet_annotations_lost"] == 3
        assert payload["sheet_annotations_complete"] is False
        assert payload["sheet_reparse_reason"] == "sheet_annotations_lost"
        assert "sheet_answer_cut" not in payload
        assert fragment.missing_regions == []
        assert document_quality([fragment])[0] is False

    def test_no_drawing_warning_of_ours_can_make_a_document_partial(
        self, monkeypatch,
    ):
        """
        Страховка от возврата старого поведения: ни один лист этой ветки не
        добавляет пометку потери, по которой документ стал бы partial.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        for model in (SheetModel(capacity=10),
                      SheetModel(capacity=10, cut_always=True),
                      SheetModel(capacity=100, junk=[None, 0.5])):
            fragment = built(model, monkeypatch)
            partial, warnings = document_quality([fragment])
            assert partial is False
            assert warnings == ["DRAWING_COMPLETENESS_UNKNOWN"]

    def test_flag_off_lost_elements_are_still_recovered_but_not_marked(self, monkeypatch):
        """
        Спасение прочтений — исправление потери, а не механизм под флагом:
        строки из списка становятся полями и без флагов. Пометка — под флагом.
        """
        model = SheetModel(capacity=100, junk=["2.13 Степень защиты по ГОСТ 14254-2015"])
        fragment = built(model, monkeypatch)
        values = [f.value for f in fragment.content.structured_drawing_fields]
        assert "2.13 Степень защиты по ГОСТ 14254-2015" in values
        assert "sheet_annotations_lost" not in fragment.content.structured_payload
        assert fragment.missing_regions == []

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

    def test_photograph_without_a_cut_costs_nothing(self, tiles):
        """
        Фотография чертежа: надписей мало, ответ целый — ни одного лишнего
        вызова, ни пометки, ни непрочитанной области.
        """
        model = SheetModel(capacity=100)
        reading = client(model).read_sheet(quartered((1200, 900)))
        assert len(model.calls) == 1
        assert reading.complete and not reading.cut and not reading.failed

    def test_each_page_of_a_multi_sheet_document_is_judged_on_its_own(
        self, monkeypatch, tiles,
    ):
        """
        Многолистовой PDF: учёт потери живёт в payload своего листа, а
        соседний целый лист остаётся чистым. Человека не зовут ни из-за
        одного листа, ни из-за обоих.
        """
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        dense = built(SheetModel(capacity=10, cut_always=True), monkeypatch)
        plain = built(SheetModel(capacity=100), monkeypatch)
        plain.fragment_id, plain.position.page = "f2", 2
        assert dense.content.structured_payload["sheet_reparse_candidate"] is True
        assert "sheet_reparse_candidate" not in plain.content.structured_payload
        assert document_quality([dense, plain])[0] is False
        assert document_quality([plain])[0] is False
