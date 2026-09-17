"""
Сквозной сценарий по контракту: положили файл -> запросили -> получили фрагменты.

Требуется полный стек: docker compose up -d
"""

import time
import uuid

import pytest
import requests

from core.config import settings

API = f"http://127.0.0.1:{settings.API_PORT}"
SHARED_ROOT = "/data/shared"


def require_api():
    try:
        response = requests.get(f"{API}/health", timeout=5)
    except requests.RequestException as exc:
        pytest.skip(f"API недоступен: {exc}")
    body = response.json()
    if body.get("status") != "ok":
        pytest.skip(f"API в состоянии {body.get('status')}: {body.get('dependencies')}")


@pytest.fixture
def uploaded_file():
    """Кладёт фикстуру туда, где Parser ищет файлы по s3_fileid."""
    from core.providers.storage import LocalStorageProvider

    storage = LocalStorageProvider(base_path=SHARED_ROOT)
    if not storage.exists("fixtures/sample.pdf"):
        pytest.skip("Фикстура fixtures/sample.pdf не смонтирована")

    fileid = f"it-{uuid.uuid4().hex[:8]}.pdf"
    storage.write_file(
        f"{settings.SOURCE_PREFIX}{fileid}", storage.read_bytes("fixtures/sample.pdf")
    )
    yield fileid
    try:
        storage.delete_file(f"{settings.SOURCE_PREFIX}{fileid}")
    except Exception:  # noqa: BLE001
        pass


@pytest.mark.integration
class TestBackgroundFlow:
    def test_full_cycle(self, uploaded_file):
        require_api()
        request_id = f"it-{uuid.uuid4().hex[:8]}"

        accepted = requests.post(f"{API}/internal/v1/parse/background", json={
            "request_id": request_id,
            "dialog_id": "integration",
            "operation": "create",
            "s3_fileid": [uploaded_file],
        }, timeout=30)
        assert accepted.status_code == 202
        assert accepted.json()["status"] == "queued"

        deadline = time.time() + settings.PROCESSING_TIMEOUT_SECONDS
        body = None
        while time.time() < deadline:
            body = requests.get(
                f"{API}/internal/v1/parse/results/{request_id}", timeout=30
            ).json()
            if body["status"] in ("completed", "partial", "error"):
                break
            time.sleep(2)

        assert body is not None and body["status"] != "error", body
        document = body["documents"][0]
        assert document["s3_fileid"] == uploaded_file
        assert document["document_metadata"]["status"] == "indexed"
        assert document["fragments"], "Документ без фрагментов"

        fragment = document["fragments"][0]
        assert set(fragment) == {
            "fragment_id", "type", "content", "position", "section_title",
            "confidence", "completeness", "provenance", "graph_nodes", "relations",
        }
        assert 0 <= fragment["confidence"] <= 1
        assert 0 <= fragment["completeness"] <= 1
        assert all(0 <= c <= 1 for c in fragment["position"]["bbox"])

    def test_unknown_file_reports_error_not_crash(self):
        require_api()
        request_id = f"it-{uuid.uuid4().hex[:8]}"

        requests.post(f"{API}/internal/v1/parse/background", json={
            "request_id": request_id, "dialog_id": "integration",
            "s3_fileid": ["точно-нет-такого-файла"],
        }, timeout=30)

        deadline = time.time() + 120
        body = None
        while time.time() < deadline:
            body = requests.get(
                f"{API}/internal/v1/parse/results/{request_id}", timeout=30
            ).json()
            if body["status"] in ("completed", "partial", "error"):
                break
            time.sleep(2)

        assert body["status"] == "error"
        assert body["documents"][0]["document_metadata"]["status"] == "error"

    def test_results_of_unknown_request(self):
        require_api()
        response = requests.get(f"{API}/internal/v1/parse/results/нет-такого", timeout=10)
        assert response.status_code == 404


@pytest.mark.integration
class TestSyncFlow:
    def test_sync_returns_documents(self, uploaded_file):
        require_api()
        request_id = f"it-{uuid.uuid4().hex[:8]}"

        response = requests.post(f"{API}/internal/v1/parse", json={
            "request_id": request_id,
            "dialog_id": "integration",
            "s3_fileid": [uploaded_file],
        }, timeout=settings.PROCESSING_TIMEOUT_SECONDS + 30)

        assert response.status_code in (200, 504)
        body = response.json()
        assert body["request_id"] == request_id
        if response.status_code == 200:
            assert body["documents"][0]["fragments"]


@pytest.mark.integration
class TestIdempotency:
    def test_same_file_twice_gives_same_doc_id(self, uploaded_file):
        require_api()
        doc_ids = []

        for _ in range(2):
            request_id = f"it-{uuid.uuid4().hex[:8]}"
            requests.post(f"{API}/internal/v1/parse/background", json={
                "request_id": request_id, "dialog_id": "integration",
                "s3_fileid": [uploaded_file],
            }, timeout=30)

            deadline = time.time() + settings.PROCESSING_TIMEOUT_SECONDS
            while time.time() < deadline:
                body = requests.get(
                    f"{API}/internal/v1/parse/results/{request_id}", timeout=30
                ).json()
                if body["status"] in ("completed", "partial", "error"):
                    break
                time.sleep(2)
            doc_ids.append(body["documents"][0]["document_metadata"]["doc_id"])

        assert doc_ids[0] == doc_ids[1]
