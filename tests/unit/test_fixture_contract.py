"""
Контракт приёма и разбора на реальных файлах из tests/fixtures/parser_contract.

Каждый тест здесь закрывает случай, который однажды уже прошёл мимо: набор
собран по итогам ручного прогона конвейера, и ожидания в нём описаны в
manifest.json рядом с самими файлами.
"""

import pytest

from core import filetypes
from core.gateway.profile import (
    IMAGE_DRAWING,
    IMAGE_PERSONAL_PHOTO,
)
from core.gateway.service import ACCEPT, QUARANTINE, REJECT, DataGateway
from core.ladder.context import GENRE_DRAWING, GENRE_TEXT, DocumentContext, classify
from core.ladder.strategies.cad_source import EZDXF_AVAILABLE, CadSourceStrategy
from core.ladder.strategies.native_text import PlainTextStrategy
from core.providers import raster_drawing
from core.providers.storage import LocalStorageProvider
from core.providers.vector_drawing import structure_ratio

pytestmark = pytest.mark.unit

CONTRACT_DIR_NAME = "parser_contract"
IMAGE_FORMATS = ["valid_image.png", "valid_image.jpg", "valid_image.bmp", "valid_image.tiff"]


@pytest.fixture
def contract_dir(fixtures_dir):
    directory = fixtures_dir / CONTRACT_DIR_NAME
    if not directory.exists():  # pragma: no cover — набор лежит в репозитории
        pytest.skip("Нет набора фикстур контракта")
    return directory


@pytest.fixture
def gateway(contract_dir):
    return DataGateway(storage=LocalStorageProvider(base_path=str(contract_dir)))


def _read(contract_dir, name: str) -> bytes:
    return (contract_dir / name).read_bytes()


def _context(contract_dir, name: str) -> DocumentContext:
    return DocumentContext(
        uri=name, file_type=filetypes.extension_of(name),
        storage=LocalStorageProvider(base_path=str(contract_dir)),
        data=_read(contract_dir, name),
    )


# ===========================================================================
# Приём: решение зависит от содержимого, а не от формата хранения
# ===========================================================================

class TestSameImageEveryFormat:
    """
    Один и тот же чертёж в четырёх форматах. bmp и tiff уезжали в карантин с
    уверенностью 0.06, потому что классификация смотрела первые 64 КБ, а у
    несжатых форматов это кусок заголовка.
    """

    def test_verdict_does_not_depend_on_format(self, gateway, contract_dir):
        verdicts = {
            name: gateway.evaluate(name, size=(contract_dir / name).stat().st_size)
            for name in IMAGE_FORMATS
        }
        assert {v.outcome for v in verdicts.values()} == {ACCEPT}, {
            name: (v.outcome, v.category, v.confidence) for name, v in verdicts.items()
        }
        assert {v.category for v in verdicts.values()} == {IMAGE_DRAWING}
        assert len({round(v.confidence, 2) for v in verdicts.values()}) == 1

    def test_whole_file_is_inspected(self, gateway, contract_dir):
        verdict = gateway.evaluate(
            "valid_image.tiff", size=(contract_dir / "valid_image.tiff").stat().st_size
        )
        assert verdict.signals["inspected_fraction"] == 1.0


class TestLowQualityImage:
    """Скан 80x50: на нём не видно ни рамки, ни строк, и чертежом он не был."""

    def test_thumbnail_is_not_a_drawing(self, gateway, contract_dir):
        verdict = gateway.evaluate(
            "low_quality.png", size=(contract_dir / "low_quality.png").stat().st_size
        )
        assert verdict.category != IMAGE_DRAWING
        assert verdict.outcome in (QUARANTINE, REJECT)

    def test_reason_names_resolution(self, contract_dir):
        verdict = raster_drawing.analyse(_read(contract_dir, "low_quality.png"))
        assert verdict.kind == raster_drawing.KIND_UNREADABLE
        assert "разрешение" in verdict.reason


class TestPersonalName:
    """Имя говорит «личное», а на листе чертёж — решает человек, не правило."""

    def test_conflict_goes_to_quarantine(self, gateway, contract_dir):
        name = "личное_отпуск.jpg"
        verdict = gateway.evaluate(name, size=(contract_dir / name).stat().st_size)
        assert verdict.outcome in (QUARANTINE, REJECT)
        if verdict.outcome == QUARANTINE:
            assert verdict.category in (IMAGE_DRAWING, IMAGE_PERSONAL_PHOTO)


# ===========================================================================
# Классификация до разбора
# ===========================================================================

class TestRasterClassification:
    def test_drawing_image_gets_drawing_genre(self, contract_dir):
        classification = classify(_context(contract_dir, "valid_image.png"))
        assert classification.genre == GENRE_DRAWING
        assert classification.signals["raster_kind"] == raster_drawing.KIND_DRAWING

    def test_formula_page_is_a_page_of_text(self, contract_dir):
        """Лист с формулами — страница текста, а не чертёж."""
        classification = classify(_context(contract_dir, "formulas.png"))
        assert classification.genre == GENRE_TEXT


# ===========================================================================
# Разбор
# ===========================================================================

class TestMarkdownTables:
    def test_table_is_parsed_as_table(self, contract_dir):
        result = PlainTextStrategy().run(_context(contract_dir, "valid_markdown.md"))
        tables = [b for b in result.blocks if b.type == "table"]
        assert len(tables) == 1
        table = tables[0].table_data
        assert table["headers"] == ["Parameter", "Value"]
        assert ["Pressure", "8.5 bar"] in table["rows"]

    def test_table_does_not_leak_into_text(self, contract_dir):
        result = PlainTextStrategy().run(_context(contract_dir, "valid_markdown.md"))
        for block in result.blocks:
            if block.type == "text":
                assert "|---" not in (block.text or "")


@pytest.mark.skipif(not EZDXF_AVAILABLE, reason="ezdxf не установлен")
class TestCadLayers:
    def test_layer_names_give_categories(self, contract_dir):
        result = CadSourceStrategy().run(_context(contract_dir, "valid_drawing.dxf"))
        drawings = [b for b in result.blocks if b.type == "drawing"]
        assert drawings
        fields = drawings[0].drawing_fields
        categories = {f["category"] for f in fields}
        assert "size" in categories
        assert "title_block" in categories
        # Полнота — доля полей с определённой категорией; раньше был ноль.
        assert structure_ratio(fields) > 0.5


class TestRasterDrawingFields:
    """
    Растровый чертёж из набора: лист доходит до внешней модели и
    возвращается фрагментом с полями. Модель поддельная — проверяется
    дорога до неё и сборка контракта, а не качество чтения.
    """

    def test_sheet_is_parsed_into_fields(self, contract_dir):
        from core.drawing_processor import DrawingProcessor

        storage = LocalStorageProvider(base_path=str(contract_dir))
        processor = DrawingProcessor(storage, qwen=_FakeQwen())

        result = processor.process("valid_image.png", {})
        assert result["parsed_fields"]
        assert all(0.0 <= f["confidence"] <= 1.0 for f in result["parsed_fields"])
        assert result["payload"]["model"]

    def test_whole_sheet_becomes_a_drawing_fragment(self, contract_dir):
        from core.drawing_processor import DrawingProcessor

        storage = LocalStorageProvider(base_path=str(contract_dir))
        processor = DrawingProcessor(storage, qwen=_FakeQwen())

        metadata = {
            "doc_id": "doc-test", "source_path": "valid_image.png",
            "file_type": "png", "source_kind": "image",
            "classification": {"genre": GENRE_DRAWING, "source": "raster_only"},
        }
        fragments = processor.enrich_fragments([], metadata)
        assert [f.type for f in fragments] == ["drawing"]
        assert fragments[0].content.structured_drawing_fields

    def test_silent_model_leaves_the_sheet_unparsed(self, contract_dir):
        from core.drawing_processor import MODE_NONE, DrawingProcessor

        storage = LocalStorageProvider(base_path=str(contract_dir))
        processor = DrawingProcessor(storage, qwen=_FakeQwen(silent=True))

        result = processor.process("valid_image.png", {})
        assert result["mode"] == MODE_NONE
        assert result["parsed_fields"] == []


class _FakeQwen:
    """Внешняя модель в тестах контракта: отвечает одним и тем же."""

    available = True
    model = "fake-vl"

    def __init__(self, silent: bool = False):
        self.silent = silent

    def read_sheet(self, image, hint=""):
        if self.silent:
            return None
        from core.providers.qwen_client import SheetReading

        return SheetReading(sheet_type="detail", annotations=[
            {"category": "size", "value": "⌀44H7", "bbox": [0.2, 0.3, 0.3, 0.35]},
            {"category": "roughness", "value": "Ra 3.2", "bbox": [0.4, 0.2, 0.5, 0.25]},
        ])

    def read_title_block(self, image):
        return None if self.silent else {"fields": {"Наименование": "Вилка"}, "rows": []}

    def read_table(self, image):
        return None


# ===========================================================================
# Размер фикстур: набор должен доезжать до ворот, а не отсеиваться раньше
# ===========================================================================

class TestFixtureSizes:
    """
    `valid_text.txt` (219 Б), `valid_table.csv` (111 Б) и `valid_drawing.dxf`
    (295 Б) отклонялись на QG-1 по `min_size_bytes`, то есть строки T04 и T05
    матрицы на этом наборе были непроверяемы: манифест обещает приём, а файл
    до проверки читаемости не доходил.
    """

    def test_every_accepted_fixture_passes_the_size_rule(self, contract_dir):
        import json

        from core.gateway.profile import GatewayProfile

        manifest = json.loads((contract_dir / "manifest.json").read_text("utf-8"))
        minimum = GatewayProfile().min_size_bytes
        small = {
            name: (contract_dir / name).stat().st_size
            for name, expected in manifest["expected"].items()
            if expected.startswith("accept") and (contract_dir / name).exists()
            and (contract_dir / name).stat().st_size < minimum
        }
        assert not small, f"меньше {minimum} Б: {small}"

    def test_drawing_fixture_keeps_its_categories(self, contract_dir):
        """Дополнение файла не должно размывать долю опознанных полей."""
        if not EZDXF_AVAILABLE:  # pragma: no cover
            pytest.skip("ezdxf не установлен")
        result = CadSourceStrategy().run(_context(contract_dir, "valid_drawing.dxf"))
        fields = [b for b in result.blocks if b.type == "drawing"][0].drawing_fields
        assert structure_ratio(fields) > 0.5

    def test_text_fixture_keeps_exact_tokens(self, contract_dir):
        text = (contract_dir / "valid_text.txt").read_text("utf-8")
        for token in ("expires_at", "rps", "2026-12-31", "8.5 bar", "X = 3.14 * D"):
            assert token in text
