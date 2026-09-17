"""
Нагрузочные сценарии.

    locust -f tests/load/locustfile.py --host=http://127.0.0.1:8000 \
           --users=20 --spawn-rate=2 --run-time=5m

Файлы берутся из переменной LOAD_TEST_FILEIDS (через запятую) — это должны
быть существующие s3_fileid, иначе тест измерит только скорость ошибок.
"""

import os
import uuid

from locust import HttpUser, between, task

FILEIDS = [f.strip() for f in os.environ.get("LOAD_TEST_FILEIDS", "").split(",") if f.strip()]


class ParserUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.submitted = []

    @task(5)
    def submit_background(self):
        if not FILEIDS:
            return
        request_id = f"load-{uuid.uuid4().hex[:12]}"
        with self.client.post(
            "/internal/v1/parse/background",
            json={
                "request_id": request_id,
                "dialog_id": "load-test",
                "operation": "create",
                "s3_fileid": FILEIDS[:3],
            },
            name="POST /internal/v1/parse/background",
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                self.submitted.append(request_id)
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @task(10)
    def poll_results(self):
        if not self.submitted:
            return
        request_id = self.submitted[-1]
        self.client.get(
            f"/internal/v1/parse/results/{request_id}",
            name="GET /internal/v1/parse/results/{id}",
        )

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
