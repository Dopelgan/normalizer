"""
Проводка связи величин с видами (`DRAWING_VALUE_VIEW_LINKS`).

Сам `core/providers/drawing_views.py` живёт на ветке связывания величин с
видами — здесь проверяется не он, а проводка: доходят ли до `attach`
разметка листа и уже сведённые поля, попадает ли результат в
`structured_payload["value_links"]`, и что происходит в дереве, где того
модуля нет вовсе. Поведение самого `attach` закреплено на его ветке.
"""

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

import core.providers as providers_pkg
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.providers.qwen_client import SheetReading

# Модуль видов и настройку приносит чужая ветка. До её слияния их здесь нет,
# после — есть, и проверки на отсутствие верны только до.
PAIRED = hasattr(settings, "DRAWING_VALUE_VIEW_LINKS")

LINKS = {
    "coordinate_reference": "deskewed_page",
    "views": [{"id": "v1", "name": "А", "kind": "section",
               "scale": "1:2", "bbox": [0.1, 0.1, 0.5, 0.5]}],
    "fields": [{"field_index": 0, "view_id": "v1", "basis": "inside_view"}],
    "counts": {"linked": 1, "unattributed": 0},
}


class Model:
    """Одна надпись: что именно читает модель, здесь не важно."""

    model = "fake-vl"
    available = True

    def read_sheet(self, image, hint=""):
        return SheetReading(sheet_type="detail", annotations=[
            {"category": "size", "value": "⌀8,5", "bbox": [0.2, 0.2, 0.3, 0.3]},
        ])


def worker(monkeypatch):
    work = DrawingProcessor(storage=object(), qwen=Model())
    work.vision_enabled = False
    monkeypatch.setattr(work, "_open",
                        lambda ref, page=1: Image.new("RGB", (400, 300), "white"))
    return work


@pytest.fixture
def links_on(monkeypatch):
    """
    Настройку приносит чужая ветка, и в этом дереве её в `Settings` нет:
    pydantic отказывается заводить неизвестное поле даже на запись. Поэтому
    подменяется не настройка, а модуль, которого тоже ещё нет.
    """
    calls = []

    class Views:
        @staticmethod
        def attach(layout, fields):
            calls.append((layout, fields))
            return dict(LINKS)

    from core import drawing_processor

    # Подменяется флаг и модуль — но не `_value_links`: проверяется именно
    # он, вместе с ленивым импортом внутри него. Подмена идёт атрибутом
    # пакета, а не записью в `sys.modules`: `from core.providers import
    # drawing_views` берёт готовый атрибут пакета, когда он есть, и только
    # иначе лезет за подмодулем — то есть атрибут работает в дереве и с
    # чужой ветвью, и без неё.
    monkeypatch.setattr(providers_pkg, "drawing_views", Views, raising=False)
    monkeypatch.setattr(drawing_processor, "_value_view_links", lambda: True)
    return calls


class TestSafeWithoutItsOwnBranch:
    """
    Пока чужой ветки нет, правка обязана быть мертва и ничего не ронять.
    Импорт наверху файла уронил бы весь `core.drawing_processor` на загрузке
    — то есть ни один документ не был бы разобран, не только чертёж.
    """

    @pytest.mark.skipif(PAIRED, reason="ветка связи величин уже в дереве")
    def test_the_module_is_absent_here(self):
        with pytest.raises(ImportError):
            __import__("core.providers.drawing_views")

    @pytest.mark.skipif(PAIRED, reason="ветка связи величин уже в дереве")
    def test_the_setting_is_absent_here(self):
        assert not hasattr(settings, "DRAWING_VALUE_VIEW_LINKS")

    def test_but_the_parser_module_imports_fine(self):
        """Главное свойство: модуль разбора грузится без чужой ветки."""
        import importlib

        import core.drawing_processor as module
        assert importlib.reload(module) is module

    def test_and_a_sheet_is_still_parsed(self, monkeypatch):
        from core import drawing_processor

        monkeypatch.setattr(drawing_processor, "_value_view_links", lambda: False)
        result = worker(monkeypatch).process("sheet.jpg", {}, page=1)
        assert result["parsed_fields"]
        assert "value_links" not in result["payload"]

    def test_the_helper_answers_without_importing_anything(self, monkeypatch):
        """Флаг выключен — до импорта дело не доходит, отвечает `None`."""
        from core import drawing_processor

        monkeypatch.setattr(drawing_processor, "_value_view_links", lambda: False)
        assert DrawingProcessor._value_links(object(), []) is None

    def test_the_flag_reader_never_raises(self):
        """Главное свойство: читатель отвечает, а не падает, в любом дереве."""
        from core.drawing_processor import _value_view_links

        assert _value_view_links() in (True, False)


class TestWiring:
    def test_layout_and_fields_reach_attach(self, monkeypatch, links_on):
        result = worker(monkeypatch).process("sheet.jpg", {}, page=1)
        assert len(links_on) == 1
        layout, fields = links_on[0]
        # Разметка листа — та же, что ушла в модель; поля — уже сведённые.
        assert hasattr(layout, "image") and hasattr(layout, "regions")
        assert [f["value"] for f in fields] == ["⌀8,5"]
        assert fields is result["parsed_fields"]

    def test_the_block_lands_in_the_payload(self, monkeypatch, links_on):
        result = worker(monkeypatch).process("sheet.jpg", {}, page=1)
        assert result["payload"]["value_links"] == LINKS

    def test_none_adds_no_key_at_all(self, monkeypatch):
        """«Связывать нечего» — это отсутствие ключа, а не пустой блок."""
        from core import drawing_processor

        class Silent:
            @staticmethod
            def attach(layout, fields):
                return None

        monkeypatch.setattr(providers_pkg, "drawing_views", Silent, raising=False)
        monkeypatch.setattr(drawing_processor, "_value_view_links", lambda: True)
        result = worker(monkeypatch).process("sheet.jpg", {}, page=1)
        assert "value_links" not in result["payload"]

    def test_the_block_survives_into_the_fragment(self, monkeypatch, links_on):
        """Блок должен дойти до потребителя, а не остаться в результате."""
        from core.models.contract import Content, Fragment, Position, Provenance

        work = worker(monkeypatch)
        result = work.process("sheet.jpg", {}, page=1)
        source = Fragment(
            fragment_id="f1", type="image", content=Content(image_ref="sheet.jpg"),
            position=Position(page=1, bbox=[0, 0, 1, 1]),
            confidence=0.5, completeness=None,
            provenance=Provenance(method="mineru", strategy_level=1,
                                  source="image", source_unit_id="u1"),
        )
        fragment = work.build_fragment(result, source, {})
        assert fragment.content.structured_payload["value_links"] == LINKS

    def test_fields_are_not_touched_by_the_linking(self, monkeypatch, links_on):
        """Привязка ссылается на поля, а не правит их."""
        result = worker(monkeypatch).process("sheet.jpg", {}, page=1)
        assert [f["value"] for f in result["parsed_fields"]] == ["⌀8,5"]
        assert all("view_id" not in f for f in result["parsed_fields"])
