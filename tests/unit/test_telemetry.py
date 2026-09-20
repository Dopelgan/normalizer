"""
Замер времени операций.

Повод конкретный: пачка из пятидесяти сканов не уложилась в таймаут
Quality Gate, и по логам нельзя было сказать, что именно отняло время.
Тайминг обязан попадать разом в три места — в лог, в ответ и в метрики, —
и при этом не иметь права уронить измеряемый код.
"""

import json
import logging

import pytest

from core import logging_setup, telemetry


class TestMeasure:
    def test_duration_lands_in_collector(self):
        with telemetry.collect() as timings:
            with telemetry.measure("шаг"):
                pass
        assert "шаг" in timings
        assert timings["шаг"] >= 0

    def test_repeated_stage_is_summed(self):
        """Этап, выполненный дважды, — это его суммарное время, а не последнее."""
        with telemetry.collect() as timings:
            for _ in range(3):
                with telemetry.measure("шаг"):
                    pass
        assert len(timings) == 1
        assert timings["шаг"] >= 0

    def test_measure_outside_collector_does_not_fail(self):
        """Замер вне области сбора — обычное дело: лог и метрика всё равно есть."""
        with telemetry.measure("шаг"):
            pass
        assert telemetry.collected() is None

    def test_exception_passes_through(self):
        """Замер не подменяет собой ошибку измеряемого кода."""
        with pytest.raises(ValueError):
            with telemetry.collect():
                with telemetry.measure("шаг"):
                    raise ValueError("своя ошибка")

    def test_failed_stage_is_still_timed(self):
        with telemetry.collect() as timings:
            with pytest.raises(ValueError):
                with telemetry.measure("шаг"):
                    raise ValueError("боом")
        assert "шаг" in timings

    def test_nested_stages_are_all_recorded(self):
        with telemetry.collect() as timings:
            with telemetry.measure("внешний"):
                with telemetry.measure("внутренний"):
                    pass
        assert set(timings) == {"внешний", "внутренний"}

    def test_collector_is_restored_after_scope(self):
        with telemetry.collect():
            with telemetry.collect() as inner:
                with telemetry.measure("вложенный"):
                    pass
            assert "вложенный" in inner
        assert telemetry.collected() is None


class TestLogPayload:
    def test_stage_is_logged_with_fields(self, caplog):
        with caplog.at_level(logging.INFO, logger="normalizer.timing"):
            with telemetry.measure("quality_gate.ocr", s3_fileid="скан.pdf") as span:
                span["outcome"] = "accept"

        record = caplog.records[-1]
        fields = record.fields
        assert fields["event"] == "stage"
        assert fields["stage"] == "quality_gate.ocr"
        assert fields["outcome"] == "accept"
        assert fields["s3_fileid"] == "скан.pdf"
        assert fields["duration_ms"] >= 0

    def test_error_outcome_is_marked(self, caplog):
        with caplog.at_level(logging.INFO, logger="normalizer.timing"):
            with pytest.raises(RuntimeError):
                with telemetry.measure("шаг"):
                    raise RuntimeError("не вышло")
        fields = caplog.records[-1].fields
        assert fields["outcome"] == "error"
        assert "RuntimeError" in fields["error"]


class TestJsonFormatter:
    def test_record_becomes_one_json_line(self):
        formatter = logging_setup.JsonFormatter("data_gateway")
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1, "принято %d", (3,), None
        )
        record.fields = {"event": "stage", "duration_ms": 12.5}

        payload = json.loads(formatter.format(record))
        assert payload["service"] == "data_gateway"
        assert payload["message"] == "принято 3"
        assert payload["event"] == "stage"
        assert payload["duration_ms"] == 12.5
        assert payload["level"] == "INFO"

    def test_exception_is_included(self):
        formatter = logging_setup.JsonFormatter("parser")
        try:
            raise ValueError("боом")
        except ValueError:
            import sys

            record = logging.LogRecord(
                "test", logging.ERROR, __file__, 1, "упало", (), sys.exc_info()
            )
        payload = json.loads(formatter.format(record))
        assert "ValueError" in payload["exception"]


class TestMetrics:
    def test_payload_is_produced(self):
        pytest.importorskip("prometheus_client")
        with telemetry.measure("тест.метрика"):
            pass
        payload = telemetry.metrics_payload().decode("utf-8")
        assert "normalizer_stage_duration_seconds" in payload
