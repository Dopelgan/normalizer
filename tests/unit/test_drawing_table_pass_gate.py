"""
Пропуск пустых проходов по таблице листа (DRAWING_TABLE_PASS_GATE).

Проход по таблице не нужен, если MinerU разобрал страницу, а в этой области
у него нет таблицы со строками. Без слоя MinerU и на вырезках пропускать
нечего: сравнить не с чем. С выключенным флагом всё как раньше.
"""

import pytest

from core.config import settings
from core.drawing_processor import DrawingProcessor

from tests.unit.test_drawing_passes_parallel import (  # noqa: F401 — фикстуры
    SHEET_URI, SPEC, SlowQwen, layout, storage,
)


def _process(storage, monkeypatch, gate, tables, lines=10, **kwargs):
    monkeypatch.setattr(settings, "DRAWING_TABLE_PASS_GATE", gate)
    monkeypatch.setattr(settings, "DRAWING_PASSES_PARALLEL", 1)
    qwen = SlowQwen(delay=0.0)
    processor = DrawingProcessor(storage, qwen=qwen)
    processor._mineru_tables = {1: tables}
    processor._layer_lines = {1: lines} if lines else {}
    result = processor.process(SHEET_URI, {}, transcribe=True, **kwargs)
    return result, qwen.asked


def test_gate_off_asks_every_table(storage, layout, monkeypatch):
    _, asked = _process(storage, monkeypatch, False, [])
    assert asked.count("table") == 2


def test_table_without_mineru_rows_is_skipped(storage, layout, monkeypatch):
    # MinerU видит только спецификацию; пустая таблица в углу не спрашивается.
    result, asked = _process(storage, monkeypatch, True, [(list(SPEC.bbox), 4)])
    assert asked.count("table") == 1
    assert result["payload"]["regions_gated"] == 1
    assert [t["kind"] for t in result["payload"]["tables"]].count("table") == 1
    # Пропуск — не «область не прочитана»: человеку лист не уходит.
    assert [r["reason"] for r in result["missing_regions"]] == []


def test_mineru_table_with_zero_rows_does_not_keep_the_pass(storage, layout, monkeypatch):
    _, asked = _process(storage, monkeypatch, True, [(list(SPEC.bbox), 0)])
    assert asked.count("table") == 0


def test_page_without_mineru_layer_is_not_gated(storage, layout, monkeypatch):
    _, asked = _process(storage, monkeypatch, True, [], lines=0)
    assert asked.count("table") == 2


def test_crop_is_not_gated(storage, layout, monkeypatch):
    # У вырезки MinerU рамки области в долях вырезки, не страницы.
    _, asked = _process(storage, monkeypatch, True, [], log_page=1)
    assert asked.count("table") == 2


def test_title_block_is_never_gated(storage, layout, monkeypatch):
    _, asked = _process(storage, monkeypatch, True, [])
    assert "title_block" in asked
