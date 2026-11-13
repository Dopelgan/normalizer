"""
Замер связывания величин с видами по настоящим листам.

Зачем. `core/providers/drawing_views.py` привязывает прочитанную величину к
виду по геометрии листа, и эту привязку надо мерить до того, как её включат:
сколько величин привязано, сколько осталось без привязки и — главное —
сколько привязано **неверно**. Неверная привязка хуже отсутствия, потому что
утверждает, что размер относится к детали, к которой он не относится.

Как устроено. На вход — картинка листа (как её рендерит разбор,
`DRAWING_RENDER_DPI`) и таблица надписей с рамками и эталоном:

    {"<имя листа>": {"annotations": [["⌀8,5 N отв.", [x1,y1,x2,y2], "Q"], ...]}}

Третий элемент — эталон: имя вида, которому надпись принадлежит на самом
деле; `""` — не принадлежит ни одному виду (общая шероховатость, требования,
надписи главного вида без заголовка), `"title"` — это сам заголовок вида,
`"pointer"` — буква-указание у стрелки или выноски.

Геометрия берётся с настоящего растра, рамки надписей сняты с него же. В бою
рамки приходят от модели: разница в их точности, не в устройстве привязки, и
в отчёте это названо прямо.

Итог — JSON: найденные виды, привязка по каждой надписи и сверка с эталоном
(верно / **неверно** / без привязки).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("QWEN_ENDPOINT", "http://localhost:1")


def as_fields(annotations: List[Any]) -> List[Dict[str, Any]]:
    """Надписи -> поля контракта той же формы, что даёт ветка чертежей."""
    from core.providers import vector_drawing

    fields = []
    for value, bbox, _ in annotations:
        category, _confidence = vector_drawing.classify(value)
        fields.append({"category": category, "value": value, "bbox": list(bbox)})
    return fields


def measure(sheet: pathlib.Path, sheet_data: Dict[str, Any]) -> Dict[str, Any]:
    from PIL import Image

    from core.config import settings
    from core.providers import drawing_views, drawing_vision

    settings.DRAWING_VALUE_VIEW_LINKS = True

    annotations = sheet_data["annotations"]
    image = Image.open(sheet).convert("RGB")
    layout = drawing_vision.analyse(image)
    fields = as_fields(annotations)
    linked = drawing_views.attach(layout, fields)

    result: Dict[str, Any] = {
        "sheet": sheet.name,
        "fields": len(fields),
        "frame": layout.frame,
        "regions": [region.kind for region in layout.regions],
    }
    if linked is None:
        result["value_links"] = None
        return result
    result.update({
        "views": linked["views"],
        "counts": linked["counts"],
        "signals": linked["signals"],
        "check": verify(linked, annotations),
    })
    return result


def verify(linked: Dict[str, Any], annotations: List[Any]) -> Dict[str, Any]:
    """Сверка привязки с эталоном по каждой надписи."""
    # Имя вида в эталоне — текст его заголовка до скобки; вид без буквы
    # (знак «развёрнуто») сопоставляется по рамке заголовка.
    def plain(name: str) -> str:
        return name.replace("–", "-").replace("—", "-").strip()

    truth_name: Dict[str, str] = {}
    for value, bbox, role in annotations:
        if role == "title":
            truth_name[json.dumps([round(c, 4) for c in bbox])] = plain(
                value.split("(")[0]) or value
    names: Dict[str, str] = {}
    for view in linked["views"]:
        key = json.dumps(view["title_bbox"])
        names[view["id"]] = truth_name.get(key, view.get("name") or view["id"])

    want_by_index = {index: role for index, (_v, _b, role) in enumerate(annotations)}
    right, wrong, missed, extra = [], [], [], []
    for record in linked["fields"]:
        want = want_by_index.get(record["field_index"])
        if want in (None, "title", "pointer"):
            continue
        got = names.get(record.get("view") or "")
        if record.get("view") is None:
            if want:
                missed.append({"value": record["value"], "expected": want,
                               "why": record.get("why")})
            continue
        if not want:
            # Привязана надпись, которая виду не принадлежит, — это неверная
            # привязка, и считается она так же.
            extra.append({"value": record["value"], "got": got})
        elif plain(got or "") == plain(want):
            right.append({"value": record["value"], "view": got})
        else:
            wrong.append({"value": record["value"], "expected": want, "got": got})
    return {
        "right": len(right),
        "wrong": len(wrong) + len(extra),
        "unattributed_but_expected": len(missed),
        "wrong_links": wrong,
        "linked_though_no_view": extra,
        "missed": missed,
        "right_links": right,
        "views_named": names,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets", required=True, type=pathlib.Path,
                        help="каталог с картинками листов")
    parser.add_argument("--annotations", required=True, type=pathlib.Path)
    args = parser.parse_args()

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    out = []
    for name, sheet_data in data.items():
        sheet = args.sheets / (name + ".png")
        if not sheet.exists():
            out.append({"sheet": name, "skip": "нет картинки листа"})
            continue
        out.append(measure(sheet, sheet_data))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
