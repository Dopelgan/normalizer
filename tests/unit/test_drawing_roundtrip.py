"""
Ветка чертежей целиком, через настоящий HTTP.

Здесь ничего не подменяется моками: поднимается локальный сервер, который
отвечает как vLLM (OpenAI-совместимый `/v1`), и лист проходит весь путь —
разметка зрением, вырезка областей, запросы к модели, приведение к ГОСТ,
сборка фрагмента контракта. Мок-тесты проверяют договор по частям, этот —
что части сходятся: именно на стыках раньше и ломалось.
"""

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.drawing_processor import MODE_VISION_VLM, DrawingProcessor
from core.providers.qwen_client import QwenVisionClient

pytest.importorskip("numpy")
pytest.importorskip("PIL")

from tests.fixtures import synthetic_sheet  # noqa: E402

SHEET_URI = "sheet.png"

SHEET_ANSWER = {
    "sheet_type": "assembly",
    "annotations": [
        # Модель отвечает так, как отвечает живая: кириллические гомоглифы
        # в обозначениях и обрывок графики среди надписей.
        {"category": "size", "value": "Ф44Н7", "bbox": [0.2, 0.3, 0.26, 0.33]},
        {"category": "roughness", "value": "Ка 3,2", "bbox": [0.4, 0.2, 0.45, 0.23]},
        {"category": "position", "value": "5", "bbox": [0.5, 0.5, 0.52, 0.53]},
        {"category": "note", "value": "|| ---"},
    ],
}
TITLE_BLOCK_ANSWER = {
    "fields": {"Наименование": "Вилка", "Материал": "Сталь 45", "Пусто": ""},
    "rows": [["Разраб.", "Иванов"]],
}
TABLE_ANSWER = {
    "title": "Спецификация",
    "headers": ["Поз.", "Наименование", "Кол."],
    "rows": [["1", "Корпус", "1"], ["2", "Гайка M4", "2"]],
}


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):        # тишина в выводе тестов
        pass

    def do_GET(self):
        self._send({"data": [{"id": "Qwen3-VL-8B-Instruct"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        content = body["messages"][1]["content"]
        raw = base64.b64decode(content[0]["image_url"]["url"].split(",", 1)[1])
        prompt = content[1]["text"]

        from PIL import Image

        kind = ("title_block" if "основная надпись" in prompt
                else "table" if "таблица, вырезанная" in prompt else "sheet")
        _Handler.seen.append({"kind": kind, "size": Image.open(io.BytesIO(raw)).size})

        answer = {"sheet": SHEET_ANSWER, "title_block": TITLE_BLOCK_ANSWER,
                  "table": TABLE_ANSWER}[kind]
        # Живая модель охотно заворачивает JSON в markdown — так и отвечаем.
        fenced = "```json\n" + json.dumps(answer, ensure_ascii=False) + "\n```"
        self._send({"choices": [{"message": {"content": fenced}}]})

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def endpoint():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


@pytest.fixture
def result(endpoint, temp_storage):
    _Handler.seen = []
    temp_storage.write_file(SHEET_URI, synthetic_sheet.png_bytes())
    processor = DrawingProcessor(temp_storage, qwen=QwenVisionClient(endpoint=endpoint))
    return processor.process(SHEET_URI, {})


class TestRoundTrip:
    def test_host_is_alive(self, endpoint):
        assert QwenVisionClient(endpoint=endpoint).health() is True

    def test_sheet_and_every_region_are_asked_separately(self, result):
        kinds = [item["kind"] for item in _Handler.seen]
        assert kinds[0] == "sheet"
        assert sorted(kinds[1:]) == ["table", "title_block"]

    def test_regions_are_sent_enlarged(self, result):
        sheet_size, *regions = [item["size"] for item in _Handler.seen]
        for width, height in regions:
            assert width < sheet_size[0]              # это вырезка
            assert width > 0.2 * sheet_size[0]        # и она увеличена

    def test_fields_come_back_in_gost_notation(self, result):
        by_category = {f["category"]: f for f in result["parsed_fields"]}
        assert by_category["size"]["value"] == "⌀44H7"
        assert by_category["size"]["tolerance"] == "H7"
        assert by_category["roughness"]["value"] == "Ra 3,2"

    def test_graphics_leftovers_do_not_reach_the_contract(self, result):
        assert all("||" not in f["value"] for f in result["parsed_fields"])

    def test_title_block_is_read_from_its_own_region(self, result):
        values = [f["value"] for f in result["parsed_fields"]]
        assert "Наименование: Вилка" in values
        assert result["payload"]["title_block"]["Материал"] == "Сталь 45"
        assert all("Пусто" not in value for value in values)

    def test_specification_lands_in_payload(self, result):
        tables = {t["kind"]: t for t in result["payload"]["tables"]}
        assert tables["table"]["rows"][1] == ["2", "Гайка M4", "2"]
        assert all("Корпус" not in f["value"] for f in result["parsed_fields"])

    def test_mode_and_metrics(self, result):
        assert result["mode"] == MODE_VISION_VLM
        assert result["payload"]["sheet_type"] == "assembly"
        assert 0 < result["avg_confidence"] <= 1
        assert result["completeness"] == 1.0
