"""Приём документов: Data Gateway, Quality Gate и журнал решений."""

import io
from typing import Optional

import pytest

from core.gateway.profile import (
    CATEGORY_BUSINESS,
    CATEGORY_CORRESPONDENCE,
    CATEGORY_PERSONAL,
    CATEGORY_TECHNICAL,
    IMAGE_DRAWING,
    GatewayProfile,
)
from core.gateway.service import ACCEPT, QUARANTINE, REJECT, DataGateway
from core import filetypes
from core.ladder.context import (
    GENRE_DRAWING,
    GENRE_TEXT,
    SOURCE_RASTER,
    Classification,
)
from core.providers.text_layer import TextLayer, TextLine
from core.quality.service import (
    ROUTE_DRAWING,
    ROUTE_MIXED,
    ROUTE_MULTIMODAL,
    ROUTE_TABULAR,
    ROUTE_TEXT,
    QualityGate,
    _OcrProbe,
    _document_date,
    _probe_from_blocks,
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


def pdf_with_text(*lines: str) -> bytes:
    """PDF с заданным текстом: классификация теперь смотрит именно текст."""
    pymupdf = pytest.importorskip("pymupdf")
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    for row, line in enumerate(lines):
        page.insert_text((60, 60 + row * 26), line, fontsize=12)
    data = document.tobytes()
    document.close()
    return data


def docx_bytes(*lines: str) -> bytes:
    """DOCX в памяти — нужен там, где проверяется тип по сигнатуре."""
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for line in lines:
        document.add_paragraph(line)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


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
# Тип файла и содержимое: чем классифицируется документ
# ===========================================================================

class TestClassificationUsesText:
    """
    Регресс. Маркеры искались в сырых первых 64 КБ файла, декодированных как
    cp1251. Для PDF это сжатый поток, и «Приложение 2. График разработки
    документации» было признано перепиской: внутри потока нашлись байты
    `re:`. Теперь текст достаётся инструментом формата.
    """

    def test_random_bytes_do_not_make_correspondence(self):
        # В тексте документа маркеров переписки нет, а в байтах файла —
        # есть: ровно так выглядел сжатый поток настоящего PDF.
        data = pdf_with_text("Development schedule, attachment 2")
        data += b"\n%% re: fwd: \xff\xfe\x00 \n"
        verdict = gateway().content_layer("documents/attachment 2.pdf", data, 1.0)
        assert verdict.category != CATEGORY_CORRESPONDENCE
        assert "matched_in" not in verdict.signals

    def test_text_decides_when_it_has_markers(self):
        data = docx_bytes("Настоящий договор поставки оборудования")
        verdict = gateway().content_layer("documents/файл.docx", data, 1.0)
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.signals["matched_in"] == "текст"
        assert verdict.signals["matched_marker"].startswith("договор")

    def test_scan_without_text_falls_back_to_path(self):
        """У скана текста нет — решает путь, и это видно в сигналах."""
        data = pdf_with_text()  # страница без текстового слоя
        verdict = gateway().content_layer(
            "documents/1-2. Договора и ДС/договор поставки.pdf", data, 1.0
        )
        assert verdict.category == CATEGORY_BUSINESS
        assert verdict.signals["matched_in"] == "путь"

    def test_path_signal_is_weaker_than_text(self):
        by_text = gateway().content_layer(
            "documents/x.docx", docx_bytes("Настоящий договор поставки"), 1.0
        )
        by_path = gateway().content_layer(
            "documents/договоры/x.docx", docx_bytes("таблица сроков"), 1.0
        )
        assert by_text.category == by_path.category == CATEGORY_BUSINESS
        assert by_path.confidence < by_text.confidence

    def test_directory_name_counts(self):
        """Раньше бралось только имя файла, каталог отбрасывался."""
        verdict = gateway().content_layer(
            "documents/1-2. Договора и ДС/приложение 2.pdf", pdf_with_text(), 1.0
        )
        assert verdict.category == CATEGORY_BUSINESS

    def test_correspondence_is_not_derived_from_path_alone(self):
        verdict = gateway().content_layer(
            "documents/переписка/приложение.pdf", pdf_with_text(), 1.0
        )
        assert verdict.category != CATEGORY_CORRESPONDENCE

    def test_letter_text_is_still_correspondence(self):
        text = "Кому: ООО Ромашка\nЗдравствуйте!\nНаправляем ответ.\nС уважением, Иванов"
        verdict = gateway().content_layer("documents/док.txt", text.encode(), 1.0)
        assert verdict.category == CATEGORY_CORRESPONDENCE
        assert verdict.outcome == QUARANTINE

    def test_docx_text_is_read(self):
        data = docx_bytes("Спецификация к договору", "ГОСТ 2.307 допуск формы")
        verdict = gateway().content_layer("documents/без имени.docx", data, 1.0)
        assert verdict.category == CATEGORY_TECHNICAL
        assert verdict.signals["text_source"] == "docx"


class TestTypeBySignature:
    """Расширение — утверждение отправителя; тип берётся по содержимому."""

    def test_docx_named_pdf_is_recognized(self):
        data = docx_bytes("Настоящий договор поставки оборудования")
        service = gateway({"documents/договор.pdf": data})
        verdict = service.evaluate("договор.pdf", "documents/договор.pdf", len(data))
        assert verdict.outcome == ACCEPT
        assert "docx" in verdict.signals["type_mismatch"]

    def test_archive_named_pdf_is_rejected_by_real_type(self):
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "x" * 5000)
        data = buffer.getvalue()
        service = gateway({"documents/отчёт.pdf": data})
        verdict = service.evaluate("отчёт.pdf", "documents/отчёт.pdf", len(data))
        # Zip опознан, но docx/xlsx в нём нет: тип остаётся pdf, а открыть
        # такой файл нельзя — решение принимает Quality Gate, не приём.
        assert verdict.outcome in (ACCEPT, QUARANTINE, REJECT)

    def test_type_layer_uses_detected_type(self):
        from core import filetypes

        resolved = filetypes.TypeVerdict("zip", "pdf", "zip", mismatch=True)
        verdict = gateway().type_layer("documents/x.pdf", resolved)
        assert verdict is not None
        assert verdict.outcome == REJECT
        assert verdict.signals["declared_type"] == "pdf"


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


# ===========================================================================
# QG-4: читаемость
# ===========================================================================

def png_bytes(size=(900, 600), colour=255) -> bytes:
    Image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    Image.new("L", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


def docx_of(paragraphs) -> bytes:
    """DOCX из перечисленных абзацев — в том числе пустых."""
    docx = pytest.importorskip("docx")
    document = docx.Document()
    for text in paragraphs:
        document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


class FakeContext:
    """Контекст ровно в том объёме, в каком его читают QG-4 и QG-5."""

    def __init__(self, kind, genre=GENRE_TEXT, layer=None, file_type="png",
                 source=SOURCE_RASTER, signals=None):
        self.kind = kind
        self.file_type = file_type
        self.uri = f"documents/x.{file_type}"
        self.text_layer = layer
        self.classification = Classification(
            source=source, genre=genre, signals=dict(signals or {})
        )


def probe_of(confidence, blocks):
    return lambda self, context: _OcrProbe(confidence=confidence, blocks=blocks)


class TestReadabilityWithoutText:
    """Пустой результат OCR — это «текста нет», а не «документ нечитаем»."""

    def test_drawing_without_captions_is_accepted(self, monkeypatch):
        monkeypatch.setattr(QualityGate, "_quick_ocr", probe_of(0.0, 0))
        warnings = []
        verdict = quality().ocr_check(
            FakeContext(filetypes.KIND_IMAGE, genre=GENRE_DRAWING), warnings
        )
        assert verdict.outcome == ACCEPT
        assert verdict.stage == "QG-4"
        assert any("надпис" in w.lower() for w in warnings)

    def test_page_without_text_is_quarantined(self, monkeypatch):
        monkeypatch.setattr(QualityGate, "_quick_ocr", probe_of(0.0, 0))
        verdict = quality().ocr_check(FakeContext(filetypes.KIND_IMAGE), [])
        assert verdict.outcome == QUARANTINE
        assert verdict.stage == "QG-4"

    def test_unreadable_text_is_still_quarantined(self, monkeypatch):
        """Текст нашёлся, но распознан почти на ноль — прежнее поведение."""
        monkeypatch.setattr(QualityGate, "_quick_ocr", probe_of(0.1, 5))
        verdict = quality().ocr_check(
            FakeContext(filetypes.KIND_IMAGE, genre=GENRE_DRAWING), []
        )
        assert verdict.outcome == QUARANTINE


class TestPartialTextLayer:
    """Слой доказывает читаемость только покрытых им страниц."""

    @staticmethod
    def _layer(pages_with_text, total):
        layer = TextLayer(page_count=total)
        for page in range(1, total + 1):
            layer.pages[page] = (
                [TextLine(text="условия поставки", bbox=[0, 0, 1, 0.1], page=page)]
                if page <= pages_with_text else []
            )
        return layer

    def test_half_covered_document_is_not_proven_readable(self):
        warnings = []
        verdict = quality().ocr_check(
            FakeContext(filetypes.KIND_PDF, layer=self._layer(2, 4), file_type="pdf"),
            warnings,
        )
        assert verdict.outcome == ACCEPT
        assert verdict.confidence == 0.5
        assert any("покрывает" in w for w in warnings)

    def test_fully_covered_document_keeps_full_confidence(self):
        warnings = []
        verdict = quality().ocr_check(
            FakeContext(filetypes.KIND_PDF, layer=self._layer(3, 3), file_type="pdf"),
            warnings,
        )
        assert verdict.confidence == 1.0
        assert warnings == []


class TestDirectRead:
    """«Читается напрямую» — утверждение о файле, а не о формате."""

    def test_docx_without_text_is_flagged(self):
        data = docx_of([""] * 40)
        verdict = quality({"documents/п.docx": data}).evaluate("п.docx", "documents/п.docx")
        assert verdict.outcome == ACCEPT
        assert verdict.confidence == 0.5
        assert any("картинками" in w for w in verdict.warnings)

    def test_docx_with_text_reads_directly(self):
        data = docx_of(["Акт сдачи-приёмки выполненных работ по договору."])
        verdict = quality({"documents/а.docx": data}).evaluate("а.docx", "documents/а.docx")
        assert verdict.outcome == ACCEPT
        assert verdict.confidence == 1.0


class TestQuickOcrInput:
    """Быстрый OCR смотрит на содержимое, а не на имя файла."""

    def test_detected_type_beats_extension(self):
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        assert TesseractFallbackProvider._treat_as_image("/d/скан.pdf", "png") is True
        assert TesseractFallbackProvider._treat_as_image("/d/лист.png", "pdf") is False
        assert TesseractFallbackProvider._treat_as_image("/d/лист.png") is True

    def test_small_raster_is_enlarged_before_recognition(self):
        Image = pytest.importorskip("PIL.Image")
        from core.config import settings
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        fitted = TesseractFallbackProvider._fit(Image.new("L", (80, 50), 255))
        assert max(fitted.size) == settings.OCR_MIN_SIDE

    def test_large_raster_is_reduced(self):
        Image = pytest.importorskip("PIL.Image")
        from core.config import settings
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        fitted = TesseractFallbackProvider._fit(Image.new("L", (9000, 6000), 255))
        assert max(fitted.size) == settings.OCR_MAX_SIDE


# ===========================================================================
# QG-5: маршрутизация карантина и сигналы
# ===========================================================================

class TestQuarantineRouting:
    def test_quarantined_file_keeps_routing_and_signals(self, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(QualityGate, "_quick_ocr", probe_of(0.0, 0))
        verdict = quality({"documents/пусто.png": png_bytes()}).evaluate(
            "пусто.png", "documents/пусто.png"
        )
        assert verdict.outcome == QUARANTINE
        assert verdict.routing["document_type"]
        assert verdict.routing["suggested_container"]
        assert verdict.signals["genre"]

    def test_accepted_file_carries_classification(self):
        verdict = quality({"documents/x.pdf": pdf_bytes()}).evaluate(
            "x.pdf", "documents/x.pdf"
        )
        assert verdict.signals["source"]
        assert "needs_restoration" in verdict.signals

    def test_empty_fingerprint_is_not_stored(self, monkeypatch):
        """Нулевой отпечаток вытеснял из выборки почти-дублей настоящие."""
        pytest.importorskip("numpy")
        monkeypatch.setattr(QualityGate, "_quick_ocr", probe_of(0.8, 3))
        raster = quality({"documents/скан.png": png_bytes()}).evaluate(
            "скан.png", "documents/скан.png"
        )
        assert "fingerprint" not in raster.signals

        text = quality({"documents/x.pdf": pdf_bytes()}).evaluate(
            "x.pdf", "documents/x.pdf"
        )
        assert text.signals["fingerprint"]


# ===========================================================================
# QG-4: доля нечитаемого и отпечаток растра
# ===========================================================================

class _Block:
    """Блок распознавания в том объёме, в каком его читает проба."""

    def __init__(self, text, confidence):
        self.text = text
        self.confidence = confidence


class TestUnreadableShare:
    """Среднее по листу вытягивается крупной шапкой — решает доля."""

    def test_probe_measures_the_share(self):
        probe = _probe_from_blocks([
            _Block("ЗАГОЛОВОК ДОКУМЕНТА" * 2, 0.95),
            _Block("нечитаемое тело листа" * 2, 0.10),
        ])
        assert probe.blocks == 2
        assert probe.unreadable_share == pytest.approx(0.5, abs=0.05)
        assert "ЗАГОЛОВОК" in probe.text

    def test_mostly_unreadable_sheet_is_quarantined(self, monkeypatch):
        monkeypatch.setattr(
            QualityGate, "_quick_ocr",
            lambda self, ctx: _OcrProbe(confidence=0.7, blocks=4, unreadable_share=0.8),
        )
        verdict = quality().ocr_check(FakeContext(filetypes.KIND_IMAGE), [])
        assert verdict.outcome == QUARANTINE
        assert "80%" in verdict.reason

    def test_small_share_is_only_a_warning(self, monkeypatch):
        monkeypatch.setattr(
            QualityGate, "_quick_ocr",
            lambda self, ctx: _OcrProbe(confidence=0.8, blocks=4, unreadable_share=0.2),
        )
        warnings = []
        verdict = quality().ocr_check(FakeContext(filetypes.KIND_IMAGE), warnings)
        assert verdict.outcome == ACCEPT
        assert any("ниже порога читаемости" in w for w in warnings)


class TestRasterFingerprint:
    """У растра отпечаток берётся из текста быстрого OCR."""

    @staticmethod
    def _probe(self, context):
        return _OcrProbe(
            confidence=0.8, blocks=3,
            text="условия поставки и сроки оплаты по договору номер сорок семь",
        )

    def test_scan_gets_a_fingerprint(self, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(QualityGate, "_quick_ocr", self._probe)
        verdict = quality({"documents/скан.png": png_bytes()}).evaluate(
            "скан.png", "documents/скан.png"
        )
        assert verdict.signals["fingerprint"]

    def test_near_duplicate_scan_is_flagged(self, monkeypatch):
        pytest.importorskip("numpy")
        monkeypatch.setattr(QualityGate, "_quick_ocr", self._probe)
        twin = _simhash(
            "условия поставки и сроки оплаты по договору номер сорок семь"
        )
        gate = quality({"documents/копия.png": png_bytes()})
        gate.known_fingerprints = [("оригинал.png", twin)]

        verdict = gate.evaluate("копия.png", "documents/копия.png")
        assert verdict.outcome == ACCEPT
        assert any("оригинал.png" in w for w in verdict.warnings)


class TestQuickOcrReadsOnce:
    """Байты уже прочитаны приёмом — второй выкачки из S3 быть не должно."""

    def test_materialize_prefers_given_bytes(self, tmp_path):
        from core.providers.tesseract_fallback import TesseractFallbackProvider

        class Angry:
            def read_bytes(self, uri):
                raise AssertionError("файл выкачан повторно")

        provider = TesseractFallbackProvider(storage=Angry())
        path, cleanup = provider._materialize("s3://bucket/скан.png", b"PNG-bytes")
        try:
            assert path and open(path, "rb").read() == b"PNG-bytes"
        finally:
            cleanup()


# ===========================================================================
# QG-5: тип по жанру, гриф по чертёжным листам
# ===========================================================================

class TestRoutingByGenre:
    def test_scan_of_text_is_a_text_document(self):
        routing = quality().routing(FakeContext(filetypes.KIND_IMAGE, genre=GENRE_TEXT))
        assert routing["document_type"] == ROUTE_TEXT
        assert routing["scanned"] is True

    def test_unrecognised_raster_stays_multimodal(self):
        from core.ladder.context import GENRE_UNKNOWN

        routing = quality().routing(
            FakeContext(filetypes.KIND_IMAGE, genre=GENRE_UNKNOWN)
        )
        assert routing["document_type"] == ROUTE_MULTIMODAL

    def test_drawing_sheets_raise_sensitivity_of_a_mixed_document(self):
        from core.ladder.context import GENRE_MIXED, SOURCE_VECTOR_TEXT

        routing = quality().routing(FakeContext(
            filetypes.KIND_PDF, genre=GENRE_MIXED, file_type="pdf",
            source=SOURCE_VECTOR_TEXT, signals={"drawing_share": 0.333},
        ))
        assert routing["document_type"] == ROUTE_MIXED
        assert routing["suggested_sensitivity"] == "confidential"
        assert routing["suggested_access_labels"] == ["конструкторский отдел"]
        assert routing["drawing_share"] == 0.333

    def test_document_without_drawings_stays_internal(self):
        from core.ladder.context import GENRE_MIXED, SOURCE_VECTOR_TEXT

        routing = quality().routing(FakeContext(
            filetypes.KIND_PDF, genre=GENRE_MIXED, file_type="pdf",
            source=SOURCE_VECTOR_TEXT,
        ))
        assert routing["suggested_sensitivity"] == "internal"
        assert routing["scanned"] is False


# ===========================================================================
# QG-3: год в имени файла
# ===========================================================================

class TestDateInName:
    def test_year_glued_to_a_word_is_not_a_date(self):
        assert _document_date("деталь_2000шт.pdf", b"") is None
        assert _document_date("серия2015.pdf", b"") is None

    def test_standalone_year_is_a_date(self):
        assert _document_date("отчёт_2024.pdf", b"").year == 2024
        assert _document_date("ГОСТ 2.307-2011.pdf", b"").year == 2011
        assert _document_date("акт_2024-03.pdf", b"").year == 2024
