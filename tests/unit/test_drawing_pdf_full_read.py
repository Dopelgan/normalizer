"""
Скан чертежа в PDF читается моделью целиком.

1. У скана PDF, из которого MinerU нарезал виды, Qwen-VL читал только
   вырезки: штамп, технические требования, общая шероховатость и размеры вне
   вырезок оставались за распознаванием без модели («Ra 1,6» → «Ра 16»,
   «(Y+12)*» потерян). DRAWING_PDF_FULL_READ читает каждый чертёжный лист
   целиком и переписывает весь его текст.
2. Ответ модели, оборванный потолком токенов, разбирался как первый
   вложенный объект: от листа оставалась одна надпись.
3. Абзац распознавания на переписанном листе заменяется строками модели
   (DRAWING_TRANSCRIPT_FIXES_OCR).

Синтетика повторяет форму реальных случаев; сами листы в тесты не кладутся.
"""

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor, _field
from core.drawing_quality import document_quality
from core.drawing_transcript import VARIANT_MARK, VARIANT_WARNING, WARNING, apply_transcripts
from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
from core.providers.qwen_client import QwenVisionClient, enlarge_small, salvage_list

pytest.importorskip("PIL")


def crop(page):
    return Fragment(
        fragment_id=f"doc-1-frag-{page:03d}", type="drawing",
        content=Content(image_ref=f"assets/doc-1/crop-{page}.png", structured_drawing_fields=[
            DrawingField(category="size", value="20", confidence=0.6)]),
        position=Position(page=page, bbox=[0.2, 0.2, 0.6, 0.6], order=1),
        confidence=0.6, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7, source="drawing"),
    )


def paragraph(page, text, method="mineru_ocr", confidence=0.95, order=5):
    return Fragment(
        fragment_id=f"doc-1-p{page}-{order}", type="text", content=Content(text=text),
        position=Position(page=page, bbox=[0.5, 0.5, 0.9, 0.6], order=order),
        confidence=confidence, completeness=1.0,
        provenance=Provenance(method=method, strategy_level=5, source="scanned_pdf"),
    )


def scan_metadata(pages=(1, 2), file_type="pdf"):
    return {
        "doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": file_type,
        "classification": {"genre": "drawing", "source": "raster_only",
                           "signals": {"raster_drawing_pages": list(pages)}},
    }


SHEET_LINES = [
    "Ra 6,3 (√)",
    "(Y+12)*",
    "8. Лицевую сторону шлифовать, шерохов. не хуже, чем Ra 1,6 (операцию допускается",
    "выполнить до гибки). Обеспечить защиту поверхности от повреждений при проведении",
    "технологических операций.",
    "ТАДУ 687000.001.001",
]


class FakeQwen:
    available = True
    model = "fake-vl"


@pytest.fixture
def processor(temp_storage, monkeypatch):
    proc = DrawingProcessor(temp_storage, qwen=FakeQwen())
    asked = []

    def process(image_ref, metadata, page=1, transcribe=False):
        asked.append((page, transcribe))
        payload = {"tables": []}
        if transcribe:
            payload["transcript_lines"] = list(SHEET_LINES)
        return {"detected_count": 1, "avg_confidence": 0.75, "mode": "sheet_only",
                "parsed_fields": [_field("size", "120*", [0.1, 0.1, 0.2, 0.2], 0.75)],
                "payload": payload, "warnings": ["DRAWING_COMPLETENESS_UNKNOWN"]}

    monkeypatch.setattr(proc, "process", process)
    proc.asked = asked
    return proc


# --------------------------------------------------------------------------
# 1. Лист скана PDF читается целиком
# --------------------------------------------------------------------------

class TestPdfSheetFullRead:
    def test_flag_off_crops_cancel_the_sheet(self, processor):
        out = processor.enrich_fragments([crop(1), crop(2)], scan_metadata())
        assert processor.asked == []
        assert not any(f.content.text for f in out if f.type == "drawing")

    def test_flag_on_every_sheet_is_read_and_transcribed(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        out = processor.enrich_fragments([crop(1), crop(2)], scan_metadata())
        assert processor.asked == [(1, True), (2, True)]
        sheets = [f for f in out if f.type == "drawing" and f.content.text]
        assert [f.position.page for f in sheets] == [1, 2]
        assert "Ra 1,6" in sheets[0].content.text and "(Y+12)*" in sheets[0].content.text
        assert sheets[0].content.structured_payload["transcript_line_count"] == len(SHEET_LINES)
        # Вырезки видов остаются: у них свои поля с местом на листе.
        assert sum(1 for f in out if f.type == "drawing") == 4

    def test_flag_on_sheet_stands_before_its_page(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        out = processor.enrich_fragments([crop(1), paragraph(1, "текст"), crop(2)], scan_metadata())
        pages = [f.position.page for f in out]
        assert pages == sorted(pages)
        assert out[0].content.text and out[3].content.text

    def test_flag_on_sheets_keep_page_order_without_crops(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        out = processor.enrich_fragments([paragraph(1, "лист 1"), paragraph(2, "лист 2")], scan_metadata())
        assert [(f.type, f.position.page) for f in out] == [
            ("drawing", 1), ("text", 1), ("drawing", 2), ("text", 2)]

    def test_flag_on_image_input_is_not_affected(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        processor.enrich_fragments([crop(1)], scan_metadata(pages=(1,), file_type="jpg"))
        assert processor.asked == []

    def test_flag_on_text_pdf_is_left_alone(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        metadata = scan_metadata()
        metadata["classification"]["genre"] = "mixed"
        processor.enrich_fragments([crop(1)], metadata)
        assert processor.asked == []

    def test_unread_region_on_transcribed_sheet_is_not_a_loss(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        monkeypatch.setattr(settings, "DRAWING_UNMEASURED_NOT_PARTIAL", True)
        out = processor.enrich_fragments([crop(1)], scan_metadata(pages=(1,)))
        out[0].missing_regions = [{"kind": "table", "bbox": [0, 0, 1, 1], "page": 1,
                                   "reason": "region_not_read"}]
        partial, _ = document_quality(out)
        assert partial is False


# --------------------------------------------------------------------------
# 2. Оборванный ответ модели
# --------------------------------------------------------------------------

CUT = ('{"sheet_type": "detail", "annotations": [\n'
       ' {"category": "title_block", "value": "ТАДУ 687000.001.001", "bbox": [0.06, 0.03, 0.22, 0.07]},\n'
       ' {"category": "size", "value": "(Y+12)*", "bbox": [0.4, 0.5, 0.42, 0.6]},\n'
       ' {"category": "size", "value": "15±0')


def answer(content, finish="stop"):
    """Ответ vLLM: текст и причина остановки."""
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


class TestCutAnswer:
    def test_whole_items_survive(self):
        items = salvage_list(CUT, "annotations")
        assert [item["value"] for item in items] == ["ТАДУ 687000.001.001", "(Y+12)*"]

    def test_lines_survive(self):
        assert salvage_list('{"lines": ["Ra 6,3", "п.4", "Лицевая сто', "lines") == ["Ra 6,3", "п.4"]

    def test_no_list_gives_nothing(self):
        assert salvage_list('{"sheet_type": "detail"', "annotations") == []

    def test_read_sheet_keeps_annotations_of_cut_answer(self, monkeypatch):
        from PIL import Image
        client = QwenVisionClient(endpoint="http://qwen.test/v1")
        monkeypatch.setattr(client, "_post", lambda body: answer(CUT, "length"))
        reading = client.read_sheet(Image.new("RGB", (40, 30), "white"))
        # Раньше разбор брал первый вложенный объект, и надписей не было вовсе.
        assert [a["value"] for a in reading.annotations] == ["ТАДУ 687000.001.001", "(Y+12)*"]
        assert reading.sheet_type == "detail"

    def test_whole_answer_is_parsed_as_before(self, monkeypatch):
        from PIL import Image
        client = QwenVisionClient(endpoint="http://qwen.test/v1")
        monkeypatch.setattr(client, "_post", lambda body: answer(CUT + ',1"}]}'))
        reading = client.read_sheet(Image.new("RGB", (40, 30), "white"))
        assert reading.sheet_type == "detail" and len(reading.annotations) == 3


class TestSmallPrint:
    """Вырезка вида 700 px: отклонение «2+1,5» модель через раз читала как «2+15»."""

    def ask(self, monkeypatch, size):
        from PIL import Image
        client = QwenVisionClient(endpoint="http://qwen.test/v1")
        seen = {}

        def post(body):
            content = body["messages"][1]["content"]
            seen["prompt"] = content[1]["text"]
            seen["image"] = content[0]["image_url"]["url"]
            return answer('{"sheet_type": "detail", "annotations": []}')

        monkeypatch.setattr(client, "_post", post)
        client.read_sheet(Image.new("RGB", size, "white"))
        return seen

    def decoded_size(self, url):
        import base64, io
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size

    def test_flag_off_request_is_unchanged(self, monkeypatch):
        seen = self.ask(monkeypatch, (700, 500))
        assert "Запятую в десятичной дроби" not in seen["prompt"]
        assert self.decoded_size(seen["image"]) == (700, 500)

    def test_flag_on_small_crop_is_enlarged_and_rule_added(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", True)
        seen = self.ask(monkeypatch, (700, 500))
        assert "Запятую в десятичной дроби" in seen["prompt"]
        assert self.decoded_size(seen["image"]) == (1400, 1000)

    def test_flag_on_large_sheet_keeps_its_size(self, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", True)
        seen = self.ask(monkeypatch, (1600, 1100))
        assert self.decoded_size(seen["image"]) == (1600, 1100)

    def test_flag_on_transcript_prompt_has_the_rule(self, monkeypatch):
        from PIL import Image
        client = QwenVisionClient(endpoint="http://qwen.test/v1")
        asked = []

        def post(body):
            asked.append(body["messages"][1]["content"][1]["text"])
            return answer('{"lines": ["Ra 1,6"]}')

        monkeypatch.setattr(client, "_post", post)
        image = Image.new("RGB", (40, 30), "white")
        assert client.transcribe(image) == ["Ra 1,6"]
        monkeypatch.setattr(settings, "DRAWING_SMALL_PRINT_CARE", True)
        assert client.transcribe_sheet(image).lines == ["Ra 1,6"]
        assert "Запятую в десятичной дроби" not in asked[0]
        assert "Запятую в десятичной дроби" in asked[1]

    def test_enlarge_small_leaves_large_image(self):
        from PIL import Image
        image = Image.new("RGB", (1300, 900), "white")
        assert enlarge_small(image, 1200) is image


# --------------------------------------------------------------------------
# 3. Абзац распознавания исправляется текстом модели
# --------------------------------------------------------------------------

def sheet(page=1, lines=SHEET_LINES):
    return Fragment(
        fragment_id=f"doc-1-frag-000-{page}", type="drawing",
        content=Content(text="\n".join(lines),
                        structured_payload={"transcript_line_count": len(lines)}),
        position=Position(page=page, bbox=[0, 0, 1, 1], order=0),
        confidence=0.75, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


WRONG = ("8.Лицевую сторону шлифовать, шерохов. не хуже, чемРа 16 (опероцию допускается "
         "выполнить до гибки). Обеспечить защиту поверхности от повреждений при проведении "
         "технологических операций.")


class TestTranscriptFixesOcr:
    def test_paragraph_takes_the_model_reading(self):
        out = apply_transcripts([sheet(), paragraph(1, WRONG)])
        fixed = out[1]
        assert "Ra 1,6" in fixed.content.text and "Ра 16" not in fixed.content.text
        assert fixed.content.text.startswith("8. Лицевую сторону")
        assert fixed.content.text.endswith("технологических операций.")
        assert WARNING in fixed.warnings

    def test_other_page_is_not_touched(self):
        out = apply_transcripts([sheet(1), paragraph(2, WRONG)])
        assert out[1].content.text == WRONG and WARNING not in out[1].warnings

    def test_unmatched_paragraph_stays(self):
        text = "Допускается замена материала в соответствии с ведомостью замены"
        out = apply_transcripts([sheet(), paragraph(1, text)])
        assert out[1].content.text == text

    def test_short_text_is_not_compared(self):
        out = apply_transcripts([sheet(), paragraph(1, "(Y+12)")])
        assert out[1].content.text == "(Y+12)"

    def test_unmatched_tesseract_noise_is_kept(self):
        # Ничего не удаляется: текст модели может быть неполным.
        noise = paragraph(1, "< - o a ni aera = pian HM 2XIBHIOT [OCT", method="ocr_layer", confidence=0.5)
        short = paragraph(1, "Ra 6,3", method="ocr_layer", confidence=0.4)
        assert len(apply_transcripts([sheet(), noise, short])) == 3

    def test_other_number_is_kept_and_model_reading_is_added(self):
        # Цифры расходятся: кто прав — неизвестно. Абзац остаётся, чтение модели — рядом.
        lines = ["4. Сварка аргонодуговая неплавящимся электродом. Сварочная проволока Св-04Х19Н11М3",
                 "ГОСТ 2246-70."]
        text = "4. Сборка дргонодуговая неплавящимся электродом. Сворочная проволока Св-04х19н1им3 ГОСТ 2246-70."
        out = apply_transcripts([sheet(lines=lines), paragraph(1, text)])
        first, second = out[1].content.text.split("\n")
        assert first == text
        assert second == " ".join(lines) + VARIANT_MARK
        assert VARIANT_WARNING in out[1].warnings and WARNING not in out[1].warnings

    def test_neighbour_requirement_is_not_a_variant(self):
        # Своей строки абзаца в тексте модели нет (ответ оборван): соседний пункт
        # с другим номером и другим ГОСТом к нему не дописывается.
        lines = ["5. Требование номер 5 по документу ГОСТ 1005", "6. Требование номер 6 по документу ГОСТ 1006"]
        text = "5. Требование номер 7 по документу ГОСТ 1009"
        out = apply_transcripts([sheet(lines=lines), paragraph(1, text)])
        assert out[1].content.text == text and not out[1].warnings

    def test_unlike_text_with_other_numbers_is_left_alone(self):
        lines = ["5. Требование про маркировку изделия лазером по документу ГОСТ 1005"]
        text = "5. Оценка качества сварных соединений по документу ГОСТ 1009"
        out = apply_transcripts([sheet(lines=lines), paragraph(1, text)])
        assert out[1].content.text == text and not out[1].warnings

    def test_digit_error_of_recognition_is_not_overwritten(self):
        # «З,2» с буквой вместо тройки: цифры расходятся — замены нет, только вариант рядом.
        lines = ["Шероховатость обработанных поверхностей не хуже Rz 3,2 по всему контуру"]
        text = "Шероховатость обработанных поверхностей не хуже Rz З,2 по всему контуру"
        out = apply_transcripts([sheet(lines=lines), paragraph(1, text)])
        assert out[1].content.text.startswith(text + "\n") and VARIANT_WARNING in out[1].warnings

    def test_long_paragraph_is_not_cut_to_a_slice(self):
        lines = [f"{n}. Пункт технических требований номер {n} про сварку и контроль" for n in range(1, 15)]
        text = " ".join(lines).replace("сварку", "сворку")
        out = apply_transcripts([sheet(lines=lines), paragraph(1, text)])
        # Абзац длиннее предела строк: кусок вместо целого не подставляется.
        assert out[1].content.text == text

    def test_lost_comma_is_the_same_number(self):
        lines = ["a | 4 - 16 | 41 | 27 | M24x1,5 | 2 | Кабельный"]
        out = apply_transcripts([sheet(lines=lines), paragraph(1, "а | 4-16 | 41 | 27 | M24x15 | 2 | Кабельный")])
        assert "M24x1,5" in out[1].content.text

    def test_table_is_not_touched(self):
        table = paragraph(1, WRONG)
        table.type = "table"
        out = apply_transcripts([sheet(), table])
        assert out[1].content.text == WRONG

    def test_no_transcript_changes_nothing(self):
        fragments = [crop(1), paragraph(1, WRONG, method="ocr_layer", confidence=0.3)]
        assert apply_transcripts(fragments) == fragments

    def test_processor_applies_it_only_with_the_flag(self, processor, monkeypatch):
        monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
        out = processor.enrich_fragments([crop(1), paragraph(1, WRONG)], scan_metadata(pages=(1,)))
        assert any(f.content.text == WRONG for f in out)
        monkeypatch.setattr(settings, "DRAWING_TRANSCRIPT_FIXES_OCR", True)
        out = processor.enrich_fragments([crop(1), paragraph(1, WRONG)], scan_metadata(pages=(1,)))
        assert not any(f.content.text == WRONG for f in out)
