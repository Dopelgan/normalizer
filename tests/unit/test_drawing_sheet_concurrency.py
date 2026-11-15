"""
Листы одного документа читаются моделью одновременно (VLM_SHEET_CONCURRENCY).

Листы шли друг за другом, хотя vLLM держит 16 запросов разом. Параллельное
чтение не должно менять ничего, кроме времени: те же фрагменты в том же
порядке страниц, те же ключи кэша ответов, тот же разбор отказа листа и те
же пометки о листах, на которые не хватило срока.

Модель подменена на уровне HTTP-запроса: клиент Qwen настоящий, поэтому
ключи кэша и разбор ответов — рабочие. Ответы приходят в обратном порядке
(последний лист отвечает первым), один лист падает с исключением, другой
остаётся без ответа.
"""

import base64
import io
import json
import threading
import time

import pytest

from core import drawing_processor as dp
from core.config import settings
from core.drawing_processor import DrawingProcessor
from core.models.contract import Content, Fragment, Position, Provenance
from core.providers import qwen_client
from core.providers.qwen_client import QwenVisionClient

fitz = pytest.importorskip("fitz")
Image = pytest.importorskip("PIL.Image")

PAGES = [1, 2, 3, 4, 5]
RAISES = 2      # лист, на котором вызов модели падает исключением
SILENT = 4      # лист, по которому модель не отвечает


def _shade(page: int) -> int:
    return 40 * page


def _scan_pdf() -> bytes:
    """Скан на пять листов: лист узнаётся по оттенку серого."""
    document = fitz.open()
    for page in PAGES:
        sheet = document.new_page(width=72, height=72)
        grey = _shade(page) / 255
        sheet.draw_rect(sheet.rect, color=(grey, grey, grey), fill=(grey, grey, grey))
    data = document.tobytes()
    document.close()
    return data


def _page_of(body) -> int:
    url = body["messages"][1]["content"][0]["image_url"]["url"]
    image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("L")
    value = image.getpixel((image.width // 2, image.height // 2))
    return min(PAGES, key=lambda page: abs(_shade(page) - value))


class FakeVlm:
    """Подмена `_post`: отвечает по листу, с задержкой, обратной номеру."""

    def __init__(self, raises=RAISES, silent=SILENT, delay=0.04):
        self.raises, self.silent, self.delay = raises, silent, delay
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def __call__(self, client, body):
        page = _page_of(body)
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self.delay * (len(PAGES) + 1 - page))
            if page == self.raises:
                raise RuntimeError("связь с моделью оборвалась")
            if page == self.silent:
                return None
            if body["messages"][1]["content"][1]["text"] == qwen_client._transcribe_prompt():
                content = {"lines": [f"ТАДУ 687000.00{page}", f"Ra {page},6", "Сталь 20"]}
            else:
                content = {"sheet_type": "detail", "annotations": [
                    {"value": f"⌀{page}0", "category": "size", "bbox": [0.1, 0.1, 0.2, 0.2]},
                    {"value": f"{page}2±0,5", "category": "size", "bbox": [0.3, 0.3, 0.4, 0.4]},
                ]}
            return {"choices": [{"message": {"content": json.dumps(content, ensure_ascii=False)},
                                 "finish_reason": "stop"}]}
        finally:
            with self.lock:
                self.in_flight -= 1


def _paragraph(page):
    return Fragment(
        fragment_id=f"doc-1-p{page}", type="text", content=Content(text=f"Абзац {page}"),
        position=Position(page=page, bbox=[0.1, 0.5, 0.9, 0.6], order=1),
        confidence=0.9, completeness=1.0,
        provenance=Provenance(method="mineru_ocr", strategy_level=5, source="scanned_pdf"),
    )


METADATA = {
    "doc_id": "doc-1", "source_path": "src/scan.pdf", "file_type": "pdf",
    "classification": {"genre": "drawing", "source": "raster_only",
                       "signals": {"raster_drawing_pages": list(PAGES)}},
}


@pytest.fixture
def run(temp_storage, monkeypatch, tmp_path):
    temp_storage.write_file("src/scan.pdf", _scan_pdf())
    monkeypatch.setattr(settings, "DRAWING_PDF_FULL_READ", True)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIBE_TILES", False)
    monkeypatch.setattr(settings, "DRAWING_DIMENSION_FIDELITY", True)
    monkeypatch.setattr(settings, "DRAWING_TRANSCRIPT_FIXES_OCR", False)
    monkeypatch.setattr(settings, "DRAWING_SUPPRESS_STALE_ON_FULL_READ", False)
    monkeypatch.setattr(settings, "DRAWING_RESCAN_POOR_OCR_PAGES", False)
    monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE", True)
    monkeypatch.setattr(settings, "QWEN_RETRIES", 0)

    def go(concurrency, fake=None, remaining=None):
        fake = fake or FakeVlm()
        monkeypatch.setattr(settings, "VLM_SHEET_CONCURRENCY", concurrency)
        # Свой каталог кэша на прогон: иначе второй прогон брал бы ответы
        # из кэша первого и сравнение ничего бы не доказывало.
        monkeypatch.setattr(settings, "VLM_RESPONSE_CACHE_DIR", str(tmp_path / f"cache-{concurrency}"))
        keys = []
        put = qwen_client.cache_put
        monkeypatch.setattr(qwen_client, "cache_put",
                            lambda key, answer: (keys.append(key), put(key, answer)))
        monkeypatch.setattr(QwenVisionClient, "_post", lambda self, body: fake(self, body))
        if remaining is not None:
            monkeypatch.setattr(dp.deadline, "remaining", remaining)
        qwen = QwenVisionClient(endpoint="http://vlm.test/v1", model="fake-vl", api_key="")
        processor = DrawingProcessor(temp_storage, qwen=qwen)
        processor.vision_enabled = False
        fragments = [_paragraph(page) for page in PAGES]
        result = processor.enrich_fragments(fragments, dict(METADATA))
        return [f.model_dump() for f in result], sorted(keys), fake

    return go


def test_parallel_sheets_give_the_same_document(run):
    serial, serial_keys, serial_fake = run(1)
    parallel, parallel_keys, parallel_fake = run(3)

    assert serial_fake.max_in_flight == 1
    assert parallel_fake.max_in_flight > 1
    assert parallel == serial
    assert parallel_keys == serial_keys and serial_keys

    # Порядок страниц: лист встаёт перед абзацем своей страницы; упавший
    # и молчащий листы остаются без фрагмента drawing, соседи — с ним.
    order = [(f["type"], f["position"]["page"]) for f in parallel]
    drawn = [page for page in PAGES if page not in (RAISES, SILENT)]
    expected = []
    for page in PAGES:
        if page in drawn:
            expected.append(("drawing", page))
        expected.append(("text", page))
    assert order == expected
    for fragment in parallel:
        if fragment["type"] == "drawing":
            page = fragment["position"]["page"]
            assert f"ТАДУ 687000.00{page}" in fragment["content"]["text"]


@pytest.mark.parametrize("concurrency", [2, 5, 16])
def test_any_bound_gives_the_same_document(run, concurrency):
    serial, serial_keys, _ = run(1)
    parallel, parallel_keys, fake = run(concurrency)
    assert parallel == serial
    assert parallel_keys == serial_keys
    assert fake.max_in_flight <= concurrency


def test_out_of_time_marks_the_same_sheets(run):
    """
    Срок кончился после двух листов: прочитаны первые два, остальные — в
    пометке DRAWING_SHEETS_SKIPPED, как при чтении по очереди. С двумя
    потоками третий лист проверяет срок только после одного из первых.
    """
    def remaining_after(allowed):
        calls = {"n": 0}
        lock = threading.Lock()

        def remaining():
            with lock:
                calls["n"] += 1
                return 10_000.0 if calls["n"] <= allowed else 1.0
        return remaining

    no_failures = dict(raises=None, silent=None)
    serial, _, _ = run(1, FakeVlm(**no_failures), remaining_after(2))
    parallel, _, _ = run(2, FakeVlm(**no_failures), remaining_after(2))
    assert parallel == serial
    drawings = [f for f in serial if f["type"] == "drawing"]
    assert [f["position"]["page"] for f in drawings] == [1, 2]
    assert "DRAWING_SHEETS_SKIPPED:3,4,5" in drawings[0]["warnings"]


def test_repeated_page_is_read_in_order(temp_storage, monkeypatch):
    """Страница дважды в списке — по очереди: второй вызов видит прочтения первого."""
    monkeypatch.setattr(settings, "VLM_SHEET_CONCURRENCY", 4)
    processor = DrawingProcessor(temp_storage, qwen=QwenVisionClient(endpoint="http://vlm.test/v1"))
    seen = []

    def process(uri, metadata, page, transcribe=False, page_fields=None):
        seen.append((page, len(page_fields), threading.current_thread() is threading.main_thread()))
        page_fields.append({"page": page})
        return None

    monkeypatch.setattr(processor, "process", process)
    shared = []
    processor._read_sheets("src/scan.pdf", {}, [3, 3], False, [shared, shared])
    assert seen == [(3, 0, True), (3, 1, True)]
