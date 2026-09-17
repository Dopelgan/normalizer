"""Приём документов: Data Gateway, Quality Gate и журнал решений."""

import io
from typing import Optional

import pytest

from core.gateway.profile import (
    CATEGORY_BUSINESS,
    CATEGORY_PERSONAL,
    CATEGORY_TECHNICAL,
    IMAGE_DRAWING,
    GatewayProfile,
)
from core.gateway.service import ACCEPT, QUARANTINE, REJECT, DataGateway
from core.quality.service import (
    ROUTE_DRAWING,
    ROUTE_TABULAR,
    ROUTE_TEXT,
    QualityGate,
    _simhash,
)


class FakeStorage:
    """Хранилище в памяти: проверяем логику приёма, а не ввод-вывод."""

    def __init__(self, files: Optional[dict] = None):
        self.files = dict(files or {})

    def read_bytes(self, uri: str) -> bytes:
        if uri not in self.files:
            raise FileNotFoundError(uri)
        return self.files[uri]

    def size(self, uri: str) -> Optional[int]:
        return len(self.files[uri]) if uri in self.files else None

    def exists(self, uri: str) -> bool:
        return uri in self.files

    def write_file(self, uri: str, content: bytes) -> None:
        self.files[uri] = content


class SilentClassifier:
    """Классификатор не настроен — решают правила."""

    available = False

    def classify_document(self, path, sample):
        return None

    def classify_image(self, path, sample):
        return None

    def resolve_ambiguous(self, path, sample, company_profile):
        return None


def gateway(files=None, profile=None, classifier=None) -> DataGateway:
    return DataGateway(
        profile=profile or GatewayProfile(),
        storage=FakeStorage(files),
        classifier=classifier or SilentClassifier(),
    )


def pdf_bytes(lines=30, pages=1) -> bytes:
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page(width=600, height=800)
        for row in range(lines):
            page.insert_text((60, 50 + row * 24), "corporate document body text",
                             fontsize=11)
    data = document.tobytes()
    document.close()
    return data


# ===========================================================================
# G-1. Формальные признаки
# ===========================================================================

class TestProfileLoading:
    def test_values_are_coerced_to_default_types(self):
        """
        Регрессия: `min_size_bytes: "512"` из YAML доезжал строкой до
        сравнения с размером файла и ронял приём пятисотой на каждом файле.
        """
        from core.gateway.profile import _from_mapping

        profile = _from_mapping({
            "min_size_bytes": "512",
            "max_size_bytes": "1048576",
            "accept_personal_areas": "true",
            "forbidden_paths": ["/tmp/"],
        })
        assert profile.min_size_bytes == 512
        assert profile.max_size_bytes == 1048576
        assert profile.accept_personal_areas is True
        assert profile.forbidden_paths == ["/tmp/"]

    def test_broken_value_falls_back_to_default(self):
        from core.gateway.profile import GatewayProfile, _from_mapping

        profile = _from_mapping({"min_size_bytes": "пятьсот"})
        assert profile.min_size_bytes == GatewayProfile().min_size_bytes

    def test_unknown_key_ignored(self):
        from core.gateway.profile import GatewayProfile, _from_mapping

        assert _from_mapping({"нет_такого": 1}) == GatewayProfile()

    def test_shipped_example_is_used_when_the_profile_is_not_copied(self, tmp_path):
        """
        В штатной поставке `config/profile.yaml` нет, и механизм профиля был
        выключен целиком: правила правили в образце, а работали умолчания.
        """
        from core.gateway import profile as profile_module

        (tmp_path / "profile.example.yaml").write_text(
            "data_gateway:\n  min_size_bytes: 4096\n", encoding="utf-8"
        )
        profile_module.reset_cache()
        loaded = profile_module.load_profile(str(tmp_path / "profile.yaml"))
        assert loaded.min_size_bytes == 4096

    def test_own_profile_wins_over_the_example(self, tmp_path):
        from core.gateway import profile as profile_module

        (tmp_path / "profile.example.yaml").write_text(
            "data_gateway:\n  min_size_bytes: 4096\n", encoding="utf-8"
        )
        (tmp_path / "profile.yaml").write_text(
            "data_gateway:\n  min_size_bytes: 8192\n", encoding="utf-8"
        )
        profile_module.reset_cache()
        loaded = profile_module.load_profile(str(tmp_path / "profile.yaml"))
        assert loaded.min_size_bytes == 8192

    def test_no_file_and_no_example_means_defaults(self, tmp_path):
        from core.gateway.profile import GatewayProfile
        from core.gateway import profile as profile_module

        profile_module.reset_cache()
        loaded = profile_module.load_profile(str(tmp_path / "profile.yaml"))
        assert loaded == GatewayProfile()


class TestFormalLayer:
    def test_system_file_is_rejected(self):
        verdict = gateway().formal_layer("documents/Thumbs.db", 4096)
        assert verdict.outcome == REJECT
        assert verdict.layer == "G-1"

    def test_forbidden_directory_is_rejected(self):
        verdict = gateway().formal_layer("documents/tmp/отчёт.pdf", 40960)
        assert verdict.outcome == REJECT

    def test_personal_area_goes_to_quarantine_not_trash(self):
        """
        В личной области бывает единственный экземпляр документа, поэтому
        запрет настраиваемый, а исход — карантин, а не отказ.
        """
        verdict = gateway().formal_layer("documents/личное/скан.pdf", 40960)
        assert verdict.outcome == QUARANTINE
        assert verdict.category == CATEGORY_PERSONAL

    def test_personal_area_can_be_allowed(self):
        profile = GatewayProfile(accept_personal_areas=True)
        assert gateway(profile=profile).formal_layer("documents/личное/с.pdf", 40960) is None

    def test_personal_exception_path_passes(self):
        profile = GatewayProfile(personal_exceptions=["/личное/архив/"])
        assert gateway(profile=profile).formal_layer(
            "documents/личное/архив/устав.pdf", 40960
        ) is None

    def test_unsupported_type_is_rejected(self):
        verdict = gateway().formal_layer("documents/архив.zip", 40960)
        assert verdict.outcome == REJECT
        assert ".pdf" in verdict.reason

    def test_dwg_explains_what_to_do(self):
        verdict = gateway().formal_layer("documents/вал.dwg", 40960)
        assert verdict.outcome == REJECT
        assert "DXF" in verdict.reason

    def test_too_small_is_rejected(self):
        verdict = gateway().formal_layer("documents/пусто.pdf", 10)
        assert verdict.outcome == REJECT

    def test_too_large_goes_to_quarantine(self):
        profile = GatewayProfile(max_size_bytes=1024)
        verdict = gateway(profile=profile).formal_layer("documents/большой.pdf", 99999)
        assert verdict.outcome == QUARANTINE

    def test_ordinary_file_passes_to_next_layer(self):
        assert gateway().formal_layer("documents/регламент.pdf", 40960) is None


# ===========================================================================
# G-2. Быстрый классификатор
# ===========================================================================

class TestContentLayer:
    def test_technical_documentation_is_accepted(self):
        verdict = gateway().content_layer(
            "documents/чертёж вала.pdf", "ГОСТ 2.307 допуск".encode(), 1.0
        )
        assert verdict.category == CATEGORY_TECHNICAL
        assert verdict.outcome == ACCEPT

    def test_business_document_is_accepted(self):
        verdict = gateway().content_layer(
            "documents/договор.pdf", "Настоящий договор".encode(), 1.0
        )
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.outcome == ACCEPT

    def test_correspondence_goes_to_quarantine_by_default(self):
        verdict = gateway().content_layer(
            "documents/письмо.txt", "Re: добрый день, с уважением".encode(), 1.0
        )
        assert verdict.outcome == QUARANTINE

    def test_personal_is_rejected(self):
        verdict = gateway().content_layer(
            "documents/отпуск.txt", "фото с отпуска".encode(), 1.0
        )
        assert verdict.outcome == REJECT

    def test_small_inspected_fraction_lowers_confidence(self):
        """Осмотрев пять процентов, классификатор не может быть уверен."""
        sample = "ГОСТ 2.307 допуск".encode()
        full = gateway().content_layer("documents/ч.pdf", sample, 1.0)
        partial = gateway().content_layer("documents/ч.pdf", sample, 0.02)
        assert partial.confidence < full.confidence
        assert partial.signals["inspected_fraction"] == 0.02

    def test_model_is_preferred_when_configured(self):
        class Model(SilentClassifier):
            def classify_document(self, path, sample):
                return {"category": CATEGORY_TECHNICAL, "confidence": 0.95}

        verdict = gateway(classifier=Model()).content_layer("x.pdf", b"", 1.0)
        assert verdict.signals["decided_by"] == "модель"
        assert verdict.confidence == pytest.approx(0.95)


# ===========================================================================
# G-3 и G-4
# ===========================================================================

class TestImageLayer:
    """
    Размеры картинок здесь не случайны: род листа определяется по геометрии
    (рамка формата, длинные линии, полосы строк), и на превью 300x200 ничего
    этого не разглядеть. Такая картинка теперь честно уходит в карантин, а не
    назначается чертежом.
    """

    def _png(self, colourful: bool, size=(900, 600)) -> bytes:
        pytest.importorskip("numpy")
        from PIL import Image, ImageDraw

        width, height = size
        if colourful:
            image = Image.new("RGB", size)
            pixels = image.load()
            for x in range(width):
                for y in range(height):
                    pixels[x, y] = (x % 256, y % 256, (x * y) % 256)
        else:
            image = Image.new("RGB", size, color=(255, 255, 255))
            draw = ImageDraw.Draw(image)
            # Рамка формата и внутренняя графика — то, чем чертёж отличается
            # от страницы текста.
            margin = max(2, width // 40)
            draw.rectangle(
                (margin, margin, width - margin, height - margin),
                outline=(0, 0, 0), width=2,
            )
            draw.line(
                (2 * margin, height // 2, width - 2 * margin, height // 2),
                fill=(0, 0, 0), width=2,
            )
            draw.rectangle(
                (width // 8, int(height * 0.6), width // 2, int(height * 0.9)),
                outline=(0, 0, 0), width=2,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def test_line_drawing_is_accepted(self):
        verdict = gateway().image_layer("documents/лист.png", self._png(False), 1.0)
        assert verdict.category == IMAGE_DRAWING
        assert verdict.outcome == ACCEPT

    def test_colourful_photo_is_not_accepted(self):
        verdict = gateway().image_layer("documents/IMG_2231.png", self._png(True), 1.0)
        assert verdict.outcome in (REJECT, QUARANTINE)

    def test_thumbnail_is_not_declared_a_drawing(self):
        """На картинке 80x50 не видно ни рамки, ни строк — это не чертёж."""
        verdict = gateway().image_layer(
            "documents/мелкая.png", self._png(False, size=(80, 50)), 1.0
        )
        assert verdict.category != IMAGE_DRAWING
        assert verdict.outcome in (REJECT, QUARANTINE)


class TestEscalation:
    def test_unsure_without_model_goes_to_quarantine(self):
        """Спорный случай не выбрасывается — его смотрит человек."""
        previous = gateway().content_layer("documents/x.pdf", "ни о чём".encode(), 1.0)
        verdict = gateway().escalation_layer("documents/x.pdf", b"", previous)
        assert verdict.outcome == QUARANTINE
        assert verdict.layer == "G-4"

    def test_main_model_resolves_the_case(self):
        class Model(SilentClassifier):
            def resolve_ambiguous(self, path, sample, company_profile):
                return {"category": CATEGORY_TECHNICAL, "confidence": 0.9}

        previous = gateway().content_layer("documents/x.pdf", "ни о чём".encode(), 1.0)
        verdict = gateway(classifier=Model()).escalation_layer("x.pdf", b"", previous)
        assert verdict.outcome == ACCEPT
        assert verdict.signals["escalated_from"] == "G-2"


class TestGatewayEndToEnd:
    def test_technical_pdf_is_accepted(self):
        data = pdf_bytes()
        service = gateway({"documents/чертёж ГОСТ.pdf": data})
        verdict = service.evaluate("чертёж ГОСТ.pdf", "documents/чертёж ГОСТ.pdf", len(data))
        assert verdict.outcome == ACCEPT

    def test_gateway_does_not_judge_technical_quality(self):
        """
        Data Gateway отвечает только за принадлежность к знаниям. Битый,
        но деловой файл отсюда уходит дальше — его забракует Quality Gate.
        """
        service = gateway({"documents/договор.pdf": "%PDF-1.4 битый".encode() * 100})
        verdict = service.evaluate("договор.pdf", "documents/договор.pdf", 1400)
        assert verdict.outcome == ACCEPT


# ===========================================================================
# Quality Gate
# ===========================================================================

def quality(files=None, **kwargs) -> QualityGate:
    return QualityGate(profile=GatewayProfile(), storage=FakeStorage(files), **kwargs)


class TestTechnicalValidation:
    def test_broken_pdf_is_rejected(self):
        data = b"%PDF-1.4 broken" * 100
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.outcome == REJECT
        assert verdict.stage == "QG-1"

    def test_missing_file_is_rejected(self):
        verdict = quality().evaluate("нет.pdf", "documents/нет.pdf")
        assert verdict.outcome == REJECT

    def test_unsupported_type_is_rejected(self):
        verdict = quality({"documents/x.zip": b"x" * 5000}).evaluate("x.zip", "documents/x.zip")
        assert verdict.outcome == REJECT

    def test_valid_pdf_passes(self):
        data = pdf_bytes()
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.outcome == ACCEPT


class TestDuplicates:
    def test_exact_duplicate_is_rejected(self):
        import hashlib

        data = pdf_bytes()
        digest = hashlib.sha256(data).hexdigest()
        gate = quality({"documents/копия.pdf": data}, known_hashes={digest: "оригинал.pdf"})
        verdict = gate.evaluate("копия.pdf", "documents/копия.pdf")
        assert verdict.outcome == REJECT
        assert "оригинал.pdf" in verdict.reason

    def test_near_duplicate_only_raises_a_flag(self):
        """Похожий документ — это «обновление или дубль», решает человек."""
        data = pdf_bytes()
        gate = quality({"documents/новая_редакция.pdf": data})
        fingerprint = _simhash(" ".join(["corporate document body text"] * 30))
        gate.known_fingerprints = [("прежняя_редакция.pdf", fingerprint)]

        verdict = gate.evaluate("новая_редакция.pdf", "documents/новая_редакция.pdf")
        assert verdict.outcome == ACCEPT
        assert any("прежняя_редакция.pdf" in w for w in verdict.warnings)

    def test_different_documents_do_not_collide(self):
        assert _simhash("совершенно один текст про валы") != _simhash("другой текст о бюджете")


class TestFreshnessAndReadability:
    def test_missing_date_is_a_warning_not_a_refusal(self):
        data = pdf_bytes()
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.outcome == ACCEPT
        assert any("дат" in w for w in verdict.warnings)

    def test_text_layer_proves_readability_without_ocr(self):
        data = pdf_bytes()
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.confidence == 1.0

    def test_spreadsheet_is_read_directly(self):
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active.append(["Обозначение", "Кол."])
        workbook.active.append(["АБВГ.01", "2"])
        buffer = io.BytesIO()
        workbook.save(buffer)

        verdict = quality({"documents/в.xlsx": buffer.getvalue()}).evaluate(
            "в.xlsx", "documents/в.xlsx"
        )
        assert verdict.outcome == ACCEPT


class TestRouting:
    def test_spreadsheet_routes_to_tabular(self):
        openpyxl = pytest.importorskip("openpyxl")
        workbook = openpyxl.Workbook()
        workbook.active.append(["a", "b"])
        workbook.active.append(["1", "2"])
        buffer = io.BytesIO()
        workbook.save(buffer)

        verdict = quality({"documents/в.xlsx": buffer.getvalue()}).evaluate(
            "в.xlsx", "documents/в.xlsx"
        )
        assert verdict.routing["document_type"] == ROUTE_TABULAR

    def test_text_pdf_routes_to_text(self):
        data = pdf_bytes()
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.routing["document_type"] == ROUTE_TEXT

    def test_cad_routes_to_drawing_and_is_confidential(self):
        ezdxf = pytest.importorskip("ezdxf")
        document = ezdxf.new(setup=True)
        document.modelspace().add_text("Ø20", height=3.5).set_placement((10, 10))
        buffer = io.StringIO()
        document.write(buffer)

        verdict = quality({"documents/вал.dxf": buffer.getvalue().encode()}).evaluate(
            "вал.dxf", "documents/вал.dxf"
        )
        assert verdict.routing["document_type"] == ROUTE_DRAWING
        assert verdict.routing["suggested_sensitivity"] == "confidential"

    def test_placement_is_a_proposal_for_a_human(self):
        """Предложение формирует система, утверждает человек."""
        data = pdf_bytes()
        verdict = quality({"documents/x.pdf": data}).evaluate("x.pdf", "documents/x.pdf")
        assert verdict.routing["needs_moderation"] is True
        assert verdict.routing["suggested_container"]
        assert verdict.routing["suggested_access_labels"]
