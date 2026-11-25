"""Общий bearer-токен на ручках ingest/gateway/quality (SEC-02)."""

import pytest
from fastapi.testclient import TestClient

from core import api_auth
from core.config import settings
from core.providers.file_locator import FileLocator, SourceFileNotFound
from gateway import api as gateway_api
from ingest import api as ingest_api
from quality import api as quality_api

FAKE_TOKEN = "test-token-not-a-secret"
UUID = "0f8fad5b-d9cb-469f-a165-70867728950e"


@pytest.fixture
def token_on(monkeypatch):
    monkeypatch.setattr(settings, "NORMALIZER_REQUIRE_TOKEN", True)
    monkeypatch.setattr(settings, "NORMALIZER_API_TOKEN", FAKE_TOKEN)


@pytest.fixture
def ingest_client(fake_redis, mocker):
    mocker.patch("ingest.api._dispatch")
    with TestClient(ingest_api.app) as client:
        yield client


def _results(client, headers=None):
    return client.get("/internal/v1/parse/results/req-missing", headers=headers or {})


class TestFlagOff:
    def test_routes_open_without_header(self, ingest_client):
        # Флаг выключен — поведение прежнее: без заголовка 404 «нет запроса», не 401.
        assert _results(ingest_client).status_code == 404


class TestFlagOn:
    def test_missing_header_is_401(self, token_on, ingest_client):
        response = _results(ingest_client)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_wrong_token_is_401(self, token_on, ingest_client):
        response = _results(ingest_client, {"Authorization": "Bearer wrong-token"})
        assert response.status_code == 401

    def test_wrong_scheme_is_401(self, token_on, ingest_client):
        response = _results(ingest_client, {"Authorization": f"Basic {FAKE_TOKEN}"})
        assert response.status_code == 401

    def test_right_token_passes(self, token_on, ingest_client):
        response = _results(ingest_client, {"Authorization": f"Bearer {FAKE_TOKEN}"})
        assert response.status_code == 404  # дошли до ручки: запроса нет

    def test_post_routes_checked_before_body(self, token_on, ingest_client):
        response = ingest_client.post("/internal/v1/parse/background", json={
            "request_id": "r", "event_id": "e", "operation": "create", "s3_fileid": ["a"],
        })
        assert response.status_code == 401

    def test_health_stays_open(self, token_on, ingest_client):
        assert ingest_client.get("/health").status_code == 200

    def test_gateway_and_quality_routes_protected(self, token_on):
        with TestClient(gateway_api.app) as client:
            assert client.get("/internal/v1/intake/quarantine").status_code == 401
            assert client.get("/internal/v1/intake/report").status_code == 401
            assert client.post("/internal/v1/intake/quarantine/x/resolve").status_code == 401
        with TestClient(quality_api.app) as client:
            assert client.post("/internal/v1/quality", json={"files": []}).status_code == 401

    @pytest.mark.parametrize("app", [ingest_api.app, gateway_api.app, quality_api.app])
    def test_every_internal_route_has_the_dependency(self, app):
        """Новая ручка без токена — ошибка: защищено всё, кроме проб."""
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/internal/"):
                continue
            calls = [dep.call for dep in route.dependant.dependencies]
            assert api_auth.require_api_token in calls, path


class TestDocsAndQuality:
    @pytest.mark.parametrize("path", ["/docs", "/openapi.json", "/redoc"])
    def test_docs_need_token_when_flag_on(self, token_on, ingest_client, path):
        assert ingest_client.get(path).status_code == 401
        ok = ingest_client.get(path, headers={"Authorization": f"Bearer {FAKE_TOKEN}"})
        assert ok.status_code == 200

    def test_docs_open_when_flag_off(self, ingest_client):
        assert ingest_client.get("/openapi.json").status_code == 200

    def test_quality_forward_sends_bearer(self, token_on, mocker):
        from core.models.intake import IntakeFile

        post = mocker.patch.object(quality_api.requests, "post")
        post.return_value.raise_for_status.return_value = None
        quality_api._forward("req-1", [IntakeFile(s3_fileid="a.pdf")])
        assert post.call_args.kwargs["headers"] == {"Authorization": f"Bearer {FAKE_TOKEN}"}


class TestFailClosed:
    def test_flag_on_without_token_refuses_to_start(self, monkeypatch, fake_redis):
        monkeypatch.setattr(settings, "NORMALIZER_REQUIRE_TOKEN", True)
        monkeypatch.setattr(settings, "NORMALIZER_API_TOKEN", "  ")
        with pytest.raises(api_auth.ApiTokenMisconfigured):
            with TestClient(ingest_api.app):
                pass

    def test_flag_on_with_token_starts(self, token_on, fake_redis):
        with TestClient(ingest_api.app):
            pass


class TestOutbound:
    def test_gateway_forward_sends_bearer(self, token_on, mocker):
        from core.intake.pipeline import IntakePipeline
        from core.models.intake import IntakeFile

        post = mocker.patch("core.intake.pipeline.requests.post")
        post.return_value.raise_for_status.return_value = None
        IntakePipeline.forward("req-1", [IntakeFile(s3_fileid="a.pdf")])
        assert post.call_args.kwargs["headers"] == {"Authorization": f"Bearer {FAKE_TOKEN}"}


class _Storage:
    def __init__(self, existing):
        self.existing = set(existing)

    def exists(self, uri):
        return uri in self.existing

    def metadata(self, uri):
        return {"filename": "scan.pdf"}


class TestAttachmentFallback:
    @pytest.fixture
    def s3(self, monkeypatch):
        monkeypatch.setattr(settings, "STORAGE_TYPE", "s3")
        monkeypatch.setattr(settings, "S3_BUCKET", "kb")
        monkeypatch.setattr(settings, "S3_ATTACHMENTS_BUCKET", "chat-attachments")
        monkeypatch.setattr(settings, "SOURCE_PREFIX", "")
        return _Storage({f"s3://chat-attachments/{UUID}"})

    def test_bare_uuid_found_when_flag_off(self, s3):
        located = FileLocator(s3).locate(UUID)
        assert located.uri == f"s3://chat-attachments/{UUID}"

    def test_bare_uuid_not_found_when_flag_on(self, s3, token_on):
        with pytest.raises(SourceFileNotFound):
            FileLocator(s3).locate(UUID)

    def test_named_bucket_found_when_flag_on(self, s3, token_on):
        located = FileLocator(s3).locate(f"chat-attachments/{UUID}")
        assert located.uri == f"s3://chat-attachments/{UUID}"
        located = FileLocator(s3).locate(f"s3://chat-attachments/{UUID}")
        assert located.uri == f"s3://chat-attachments/{UUID}"

    def test_presence_ignores_bare_uuid_when_flag_on(self, s3, token_on):
        assert FileLocator(s3).is_present(UUID) is False
        assert FileLocator(s3).is_present(f"chat-attachments/{UUID}") is True

    def test_presence_sees_bare_uuid_when_flag_off(self, s3):
        assert FileLocator(s3).is_present(UUID) is True
