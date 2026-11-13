"""
Картинка читается как лист чертежа только в чертёжном документе
(DRAWING_IMAGE_GENRE_GATE).

Без флага каждая картинка MinerU уходит в Qwen-VL как чертёжный лист, если
только это не мелкая картинка в текстовом документе. Так иллюстрации в
договорах, программах обеспечения качества и банковских гарантиях читаются
листами: вызов модели впустую и псевдочертёжная запись в индексе.

Жанр здесь только читается из классификации лестницы; считает его лестница
(core/ladder/context.py). Поэтому модель в этих тестах не нужна вовсе:
проверяется, дошло ли дело до вызова.
"""

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from core.config import settings
from core.drawing_processor import DrawingProcessor, _genre_allows_sheet
from core.models.contract import Content, Fragment, Position, Provenance


def image_fragment(page=1, fid="f1", bbox=(0.1, 0.1, 0.9, 0.9)):
    return Fragment(
        fragment_id=fid, type="image",
        content=Content(image_ref="pic.jpg"),
        position=Position(page=page, bbox=list(bbox)),
        confidence=0.5, completeness=None,
        provenance=Provenance(method="mineru", strategy_level=1,
                              source="image", source_unit_id="u1"),
    )


def meta(genre=None, drawing_pages=None, raster_pages=None, **extra):
    classification = {}
    if genre is not None:
        classification["genre"] = genre
    signals = {}
    if drawing_pages is not None:
        signals["drawing_pages"] = list(drawing_pages)
    if raster_pages is not None:
        signals["raster_drawing_pages"] = list(raster_pages)
    if signals:
        classification["signals"] = signals
    data = dict(extra)
    if classification:
        data["classification"] = classification
    return data


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.setattr(settings, "DRAWING_IMAGE_GENRE_GATE", True)


class TestTheRule:
    """Гейт только отказывает и никогда не разрешает сверх сегодняшнего."""

    def test_flag_off_reads_everything_as_before(self):
        assert _genre_allows_sheet(image_fragment(), meta(genre="text")) is True

    def test_drawing_document_is_read(self, gate):
        assert _genre_allows_sheet(image_fragment(), meta(genre="drawing")) is True

    def test_text_document_is_not_read(self, gate):
        assert _genre_allows_sheet(image_fragment(), meta(genre="text")) is False

    def test_spreadsheet_document_is_not_read(self, gate):
        assert _genre_allows_sheet(image_fragment(), meta(genre="spreadsheet")) is False

    @pytest.mark.parametrize("genre", [None, "", "unknown"])
    def test_unknown_genre_behaves_as_today(self, gate, genre):
        """
        Молчание классификации — не запрет: иначе пропавший жанр тихо
        прекратил бы чтение настоящих листов.
        """
        assert _genre_allows_sheet(image_fragment(), meta(genre=genre)) is True

    def test_no_classification_at_all_behaves_as_today(self, gate):
        assert _genre_allows_sheet(image_fragment(), {}) is True
        assert _genre_allows_sheet(image_fragment(), {"classification": None}) is True

    def test_fragment_without_a_page_is_read(self, gate):
        """Страницы нет — доказать, что лист не чертёжный, нечем."""
        fragment = image_fragment()
        fragment.position = None
        assert _genre_allows_sheet(fragment, meta(genre="text")) is True


class TestADrawingInsideATextDocument:
    """
    Главное требование: чертёж, вплетённый в текстовый документ, не теряется.
    Такой лист несёт свою основную надпись формы 1, лестница называет его
    страницу чертёжной — и правило читает именно её, а не весь документ.
    """

    def test_the_drawing_page_of_a_text_document_is_read(self, gate):
        data = meta(genre="mixed", drawing_pages=[7, 8])
        assert _genre_allows_sheet(image_fragment(page=7), data) is True
        assert _genre_allows_sheet(image_fragment(page=8), data) is True

    def test_the_text_pages_of_the_same_document_are_not(self, gate):
        data = meta(genre="mixed", drawing_pages=[7, 8])
        assert _genre_allows_sheet(image_fragment(page=1), data) is False
        assert _genre_allows_sheet(image_fragment(page=9), data) is False

    def test_a_scanned_drawing_page_counts_too(self, gate):
        """У скана чертёжные листы перечислены вторым списком."""
        data = meta(genre="mixed", drawing_pages=[], raster_pages=[3, 5])
        assert _genre_allows_sheet(image_fragment(page=3), data) is True
        assert _genre_allows_sheet(image_fragment(page=4), data) is False

    def test_both_lists_are_taken_together(self, gate):
        data = meta(genre="text", drawing_pages=[2], raster_pages=[9])
        assert _genre_allows_sheet(image_fragment(page=2), data) is True
        assert _genre_allows_sheet(image_fragment(page=9), data) is True
        assert _genre_allows_sheet(image_fragment(page=5), data) is False

    def test_page_numbers_as_strings_are_understood(self, gate):
        data = meta(genre="text", drawing_pages=["4"])
        assert _genre_allows_sheet(image_fragment(page=4), data) is True

    def test_rubbish_in_the_page_list_does_not_open_the_gate(self, gate):
        data = meta(genre="text", drawing_pages=[None, "стр. 2", {}])
        assert _genre_allows_sheet(image_fragment(page=2), data) is False


class Counting:
    """Модель, которая ничего не читает: важно только, звали ли её."""

    model = "fake-vl"
    available = True

    def __init__(self):
        self.calls = 0

    def read_sheet(self, *a, **k):
        self.calls += 1
        return None          # разбор не удался — документ от этого не падает


class TestNoModelCallIsMade:
    """Правило проверяется там, где тратятся деньги: до вызова модели."""

    @staticmethod
    def worker(monkeypatch):
        qwen = Counting()
        worker = DrawingProcessor(storage=object(), qwen=qwen)
        worker.vision_enabled = False
        # Картинка открывается без хранилища: проверяется не чтение файла,
        # а то, дошло ли дело до вызова модели.
        monkeypatch.setattr(worker, "_open",
                            lambda ref, page=1: Image.new("RGB", (400, 300), "white"))
        return worker, qwen

    def test_illustration_in_a_contract_costs_no_call(self, monkeypatch, gate):
        worker, qwen = self.worker(monkeypatch)
        fragments = [image_fragment(page=1), image_fragment(page=2, fid="f2")]
        result = worker.enrich_fragments(fragments, meta(genre="text"))
        assert qwen.calls == 0
        assert [f.type for f in result] == ["image", "image"]
        # Картинка остаётся картинкой и ничем не помечена: это не потеря.
        assert all(not f.warnings and not f.review_required for f in result)

    def test_the_drawing_page_still_reaches_the_model(self, monkeypatch, gate):
        worker, qwen = self.worker(monkeypatch)
        worker.enrich_fragments([image_fragment(page=4)],
                                meta(genre="mixed", drawing_pages=[4]))
        assert qwen.calls == 1

    def test_flag_off_still_reaches_the_model(self, monkeypatch):
        worker, qwen = self.worker(monkeypatch)
        worker.enrich_fragments([image_fragment()], meta(genre="text"))
        assert qwen.calls == 1

    def test_unknown_genre_still_reaches_the_model(self, monkeypatch, gate):
        worker, qwen = self.worker(monkeypatch)
        worker.enrich_fragments([image_fragment()], meta(genre="unknown"))
        assert qwen.calls == 1
