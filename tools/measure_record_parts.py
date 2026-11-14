"""
Замер: что каждое представление листа добавляет к остальным.

Зачем. Один лист уходит в индекс несколькими представлениями сразу, и
прежде чем убирать любое из них, надо знать, что оно несёт такого, чего нет
в других. Представлений три, и они разные по происхождению:

* **R1** — `content.text` собранной записи, части 1–8
  (`core/drawing_record.py`): значения под подписями, по роду величины.
* **R2** — часть 9 записи, «Надписи листа в порядке чтения»: прежний текст
  листа строками в порядке выдачи модели.
* **R3** — строки «Метка: значение», которые из `structured_drawing_fields`
  строит адаптер RAG (`super_rag/parser/adapter.drawing_search_text`).
  Он делает это **только** для фрагмента с пустым `content.text`; у
  фрагмента с текстом индексируется текст.

Что считается. Для каждого листа: значения (надписи листа посимвольно) и
то, в каких представлениях они стоят. Значение считается «несомым»
представлением, если оно входит в его текст целой подстрокой.

Вход — снятые с настоящих листов надписи `tests/fixtures/real_sheet_views`
(ТАДУ 687000.001.001 «Крышка», ТАДУ 302182.001 ГЧ, ТАДУ 305179.001.03 СБ).
Выборка маленькая и названа маленькой: три листа, 104 надписи. Геометрию
замер не трогает — сравниваются только тексты представлений.

Запуск: `python tools/measure_record_parts.py`
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict, List

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("QWEN_ENDPOINT", "http://localhost:1")


def fragment_of(annotations: List[Any]) -> Any:
    """Фрагмент чертежа той же формы, какую даёт ветка чертежей."""
    from core.models.contract import Content, DrawingField, Fragment, Position, Provenance
    from core.providers import vector_drawing

    fields = []
    for value, bbox, _truth in annotations:
        category, confidence = vector_drawing.classify(value)
        fields.append(DrawingField(category=category, value=value, confidence=confidence,
                                   confirmed=confidence >= 0.95))
    # Строки листа — надписи в порядке чтения: сверху вниз, слева вправо.
    lines = [value for value, _bbox, _truth in
             sorted(annotations, key=lambda item: (round(item[1][1], 2), item[1][0]))]
    return Fragment(
        fragment_id="f1", type="drawing",
        content=Content(text="\n".join(lines),
                        structured_drawing_fields=fields,
                        structured_payload={"recognition_mode": "sheet_only"}),
        position=Position(page=1, bbox=[0, 0, 1, 1]),
        confidence=0.8, completeness=None,
        provenance=Provenance(method="vlm_full_page", strategy_level=7, source="drawing"),
    )


def measure(name: str) -> Dict[str, Any]:
    from core.config import settings
    from core import drawing_record
    from tests.fixtures import real_sheet_views

    settings.DRAWING_RECORD_COMPOSED = True
    data = real_sheet_views.sheet(name)
    annotations = data["annotations"]
    fragment = fragment_of(annotations)
    values = [value for value, _bbox, _truth in annotations if value.strip()]

    drawing_record.compose_fragment(fragment)
    text = fragment.content.text or ""
    parts = (fragment.content.structured_payload or {}).get("record_parts") or []
    tail = next((p for p in parts if p["kind"] == "sheet_lines"), None)
    r2 = text[tail["start"]:tail["end"]] if tail else ""
    r1 = text[:tail["start"]] if tail else text
    # R3 — то, что из полей собрал бы адаптер RAG: «Метка: значение».
    r3 = "\n".join(f"{f.category}: {f.value}" for f in fragment.content.structured_drawing_fields)

    def carried(where: str) -> List[str]:
        return [value for value in values if value.strip() in where]

    in_r1, in_r2, in_r3 = set(carried(r1)), set(carried(r2)), set(carried(r3))
    return {
        "sheet": data.get("sheet", name),
        "annotations": len(values),
        "chars": {"r1_parts_1_8": len(r1), "r2_sheet_lines": len(r2), "r3_field_lines": len(r3)},
        "values_in": {"r1": len(in_r1), "r2": len(in_r2), "r3": len(in_r3)},
        "only_in_r2": sorted(in_r2 - in_r1),
        "only_in_r3": sorted(in_r3 - in_r1),
        "nowhere": sorted(set(values) - in_r1 - in_r2 - in_r3),
    }


def main() -> int:
    from tests.fixtures import real_sheet_views

    out = [measure(name) for name in real_sheet_views.SHEETS]
    total = {
        "annotations": sum(item["annotations"] for item in out),
        "only_in_r2": sum(len(item["only_in_r2"]) for item in out),
        "only_in_r3": sum(len(item["only_in_r3"]) for item in out),
        "nowhere": sum(len(item["nowhere"]) for item in out),
    }
    print(json.dumps({"sheets": out, "total": total}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
