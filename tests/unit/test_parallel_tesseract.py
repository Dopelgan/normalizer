"""
Параллельное чтение страниц Tesseract (`TESSERACT_PAGE_WORKERS`).

Проверяется то, что не должно измениться от параллельности: порядок
страниц, память OCR между уровнями лестницы, бюджет времени и поведение
при сбое одной страницы. По умолчанию (1) всё идёт в вызывающем потоке.
"""

import os
import random
import threading
import time

import pytest
from PIL import Image

from core.config import settings
from core.providers import tesseract_fallback
from core.providers.tesseract_fallback import TesseractFallbackProvider, ocr_memo

PAGES = 12


@pytest.fixture(autouse=True)
def _restore_omp(monkeypatch):
    # Параллельный режим ставит OMP_THREAD_LIMIT процессу; после теста — как было.
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)


def _render(path, first_page, last_page, dpi):
    # Номер страницы зашит в яркость листа: так подмена tesseract знает,
    # какую страницу читает, и после масштабирования тоже.
    return [Image.new("L", (300, 200), page * 10) for page in range(first_page, last_page + 1)]


class _Tesseract:
    """Подмена `image_to_data`: случайная задержка, учёт одновременных вызовов."""

    def __init__(self, fail_pages=(), delay=0.02):
        self.fail_pages = set(fail_pages)
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.threads = set()
        self.lock = threading.Lock()

    def __call__(self, image, lang=None, config=None, output_type=None):
        page = image.getpixel((0, 0)) // 10
        with self.lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.current_thread().name)
        try:
            # Поздние страницы отвечают раньше ранних — порядок должен устоять.
            time.sleep(self.delay * random.random() + (PAGES - page) * self.delay / 10)
            if page in self.fail_pages:
                raise RuntimeError(f"tesseract упал на странице {page}")
            return {
                "text": ["стр", str(page)], "conf": [90, 90],
                "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 1],
                "left": [10, 70], "top": [10, 10], "width": [50, 30], "height": [20, 20],
            }
        finally:
            with self.lock:
                self.active -= 1


@pytest.fixture
def provider(monkeypatch):
    provider = TesseractFallbackProvider(storage=object())
    monkeypatch.setattr(provider, "_materialize", lambda uri, data=None, file_type=None: ("/tmp/doc.pdf", lambda: None))
    monkeypatch.setattr(provider, "_page_count", staticmethod(lambda path: PAGES))
    monkeypatch.setattr(tesseract_fallback, "convert_from_path", _render)
    monkeypatch.setattr(settings, "OCR_BATCH_PAGES", 5)
    monkeypatch.setattr(settings, "OCR_PREPROCESS_ENABLED", False)
    monkeypatch.setattr(settings, "TESSERACT_PAGE_WORKERS", 4)
    return provider


def _tesseract(monkeypatch, **kwargs):
    fake = _Tesseract(**kwargs)
    monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_data", fake)

    def no_string(*args, **kw):
        raise RuntimeError("простое распознавание тоже упало")

    monkeypatch.setattr(tesseract_fallback.pytesseract, "image_to_string", no_string)
    return fake


def _texts(blocks):
    return [(b.page, b.text) for b in blocks]


def test_parallel_keeps_page_order_and_text(provider, monkeypatch):
    fake = _tesseract(monkeypatch)
    blocks = provider.parse_all_pages("documents/doc.pdf", "pdf")
    assert _texts(blocks) == [(p, f"стр {p}") for p in range(1, PAGES + 1)]
    assert 1 < fake.peak <= 4


def test_parallel_matches_sequential(provider, monkeypatch):
    _tesseract(monkeypatch)
    parallel = provider.parse_all_pages("documents/doc.pdf", "pdf")
    monkeypatch.setattr(settings, "TESSERACT_PAGE_WORKERS", 1)
    fake = _tesseract(monkeypatch)
    sequential = provider.parse_all_pages("documents/doc.pdf", "pdf")
    assert parallel == sequential
    assert fake.peak == 1
    assert fake.threads == {threading.current_thread().name}


def test_text_layer_parallel_order(provider, monkeypatch):
    fake = _tesseract(monkeypatch)
    layer = provider.text_layer("documents/doc.pdf", "pdf")
    assert sorted(layer.pages) == list(range(1, PAGES + 1))
    assert list(layer.pages) == list(range(1, PAGES + 1))
    for page, lines in layer.pages.items():
        assert [line.text for line in lines] == [f"стр {page}"]
        assert all(line.page == page for line in lines)
    assert fake.peak > 1


def test_memo_shared_with_worker_threads(provider, monkeypatch):
    """Уровень 4 и слой уровня 5 в одном документе: каждая страница — один вызов."""
    fake = _tesseract(monkeypatch)
    with ocr_memo():
        provider.parse_all_pages("documents/doc.pdf", "pdf")
        layer = provider.text_layer("documents/doc.pdf", "pdf")
    assert fake.calls == PAGES
    assert len(layer.pages) == PAGES
    assert tesseract_fallback._MEMO.get() is None


def test_one_failed_page_does_not_break_others(provider, monkeypatch):
    _tesseract(monkeypatch, fail_pages={3, 8})
    blocks = provider.parse_all_pages("documents/doc.pdf", "pdf")
    expected = [p for p in range(1, PAGES + 1) if p not in (3, 8)]
    assert [b.page for b in blocks] == expected


def test_time_budget_stops_new_pages(provider, monkeypatch):
    """Срок вышел посреди документа: прочитанное — сплошное начало, усечение честное."""
    _tesseract(monkeypatch, delay=0.0)
    clock = {"now": 0.0}
    monkeypatch.setattr(tesseract_fallback.time, "monotonic", lambda: clock["now"])
    real_read = provider._blocks_from_image
    lock = threading.Lock()

    def read(image, page):
        with lock:
            clock["now"] += 10.0
        return real_read(image, page)

    monkeypatch.setattr(provider, "_blocks_from_image", read)
    monkeypatch.setattr(settings, "OCR_TIME_BUDGET", 45)
    blocks = provider.parse_all_pages("documents/doc.pdf", "pdf")
    pages = [b.page for b in blocks]
    assert pages == list(range(1, len(pages) + 1))
    assert provider.truncated_at == (len(pages), PAGES)
    assert 5 <= len(pages) < PAGES


def test_expired_budget_reads_nothing(provider, monkeypatch):
    fake = _tesseract(monkeypatch)
    monkeypatch.setattr(settings, "OCR_TIME_BUDGET", -1)
    assert provider.parse_all_pages("documents/doc.pdf", "pdf") == []
    assert provider.truncated_at == (0, PAGES)
    assert fake.calls == 0


def test_in_flight_pages_are_bounded(monkeypatch):
    """Из генератора рендера берётся не больше N страниц сверх выданных."""
    monkeypatch.setattr(settings, "TESSERACT_PAGE_WORKERS", 3)
    taken = []

    def pages():
        for page in range(1, 11):
            taken.append(page)
            yield page, page

    seen = []
    for page, result in TesseractFallbackProvider._map_pages(pages(), lambda image, page: page * 2):
        assert len(taken) - len(seen) <= 3
        seen.append((page, result))
    assert seen == [(p, p * 2) for p in range(1, 11)]


def test_unexpected_error_propagates_in_page_order(monkeypatch):
    monkeypatch.setattr(settings, "TESSERACT_PAGE_WORKERS", 4)
    started = []

    def work(image, page):
        started.append(page)
        if page == 2:
            raise ValueError("страница 2")
        return page

    pages = ((p, None) for p in range(1, 100))
    got = []
    with pytest.raises(ValueError):
        for page, result in TesseractFallbackProvider._map_pages(pages, work):
            got.append(page)
    assert got == [1]
    # Остальное отменено, а не дочитано до конца документа.
    time.sleep(0.05)
    assert len(started) < 10


def test_parallel_limits_openmp_threads(provider, monkeypatch):
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    _tesseract(monkeypatch)
    provider.parse_all_pages("documents/doc.pdf", "pdf")
    assert os.environ.get("OMP_THREAD_LIMIT") == "1"


def test_default_is_sequential(monkeypatch):
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)
    monkeypatch.setattr(settings, "TESSERACT_PAGE_WORKERS", 1)
    threads = set()

    def work(image, page):
        threads.add(threading.current_thread().name)
        return page

    result = list(TesseractFallbackProvider._map_pages(((p, None) for p in range(5)), work))
    assert result == [(p, p) for p in range(5)]
    assert threads == {threading.current_thread().name}
    assert "OMP_THREAD_LIMIT" not in os.environ
