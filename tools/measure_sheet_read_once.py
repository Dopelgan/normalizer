"""
Чего стоит и что теряет чтение листа один раз (DRAWING_SHEET_READ_ONCE).

Зачем. Флаг убирает проходы модели по вырезкам видов там, где лист уже
прочитан целиком. Экономия измерима сразу, а цена — нет: вырезка уходит в
модель крупнее, чем тот же вид на листе целиком (QWEN_SHEET_MAX_SIDE на
четверть листа — примерно вчетверо больше точек на миллиметр), и мелкие
размеры внутри видов за этим и читались. Поэтому мерить надо не число
полей, а сами значения: три поля вырезки против одного поля листа могут
быть одним и тем же размером, прочитанным по-разному.

Как устроено. Один прогон на документ, с флагом выключенным: он даёт и
чтение листа целиком, и чтения вырезок. Включённый флаг — это тот же
результат без полей вырезок, поэтому второго прогона модели не нужно, и
разброс модели между прогонами в сравнение не попадает. Чтобы это
равенство не осталось на словах, `--verify-flag` повторяет прогон с
включённым флагом и сверяет, что проход листа остался ровно один и чтение
листа совпало.

Итог по странице:
  * время — сколько заняли проходы вырезок (это и есть экономия) и сколько
    занял проход листа целиком;
  * значения — какие значения вырезок не нашлись в чтении листа. Это цена,
    и она называется поимённо: категория, значение, рамка. Отдельно
    считаются величины (размеры, допуски, резьба, шероховатость) — именно
    про них вопрос;
  * объём — `core/ladder/drawing_route.volume` и состав
    `drawing_structure` при обоих исходах: видно, двинулся ли вердикт
    LADDER_DRAWING_OVER_OCR.

Нужны поднятые MinerU (нарезка видов) и Qwen-VL. Ни прод, ни очередь не
трогает: файлы читаются с диска, результат — в JSON Lines.

    python tools/measure_sheet_read_once.py samples/*.pdf --out out.jsonl
    python tools/measure_sheet_read_once.py --summary out.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MEASURED = frozenset(
    {"size", "tolerance", "radius", "thread", "roughness", "position", "callout"}
)


# --------------------------------------------------------------------- прогон
def _flags(**wanted: bool) -> None:
    from core.config import settings
    for name, value in wanted.items():
        setattr(settings, name, value)


def _page_count(path: pathlib.Path) -> int:
    import fitz
    with fitz.open(str(path)) as doc:
        return doc.page_count


def _metadata(uri: str, pages: int) -> Dict[str, Any]:
    """
    Разметка лестницы для скана чертежа. Жанр ставится прямо: все листы
    корпуса проверены как растровые (текстового слоя нет), а гонять ради
    этого всю лестницу замер не просит.
    """
    return {
        "doc_id": pathlib.Path(uri).stem,
        "source_path": uri,
        "file_type": "pdf",
        "classification": {
            "genre": "drawing", "source": "raster_only",
            "signals": {"raster_drawing_pages": list(range(1, pages + 1))},
        },
    }


def _timed(processor, source_uri: str) -> List[Dict[str, Any]]:
    """Оборачивает `process`: каждый проход модели попадает в журнал."""
    calls: List[Dict[str, Any]] = []
    original = processor.process

    def process(image_ref, metadata, page=1, transcribe=False,
                log_page=None, text_only=False):
        started = time.monotonic()
        try:
            return original(image_ref, metadata, page, transcribe,
                            log_page=log_page, text_only=text_only)
        finally:
            calls.append({
                "ref": image_ref,
                "page": int(log_page or page),
                "sheet": image_ref == source_uri,
                "seconds": round(time.monotonic() - started, 2),
            })

    processor.process = process
    return calls


def _values(fragment) -> List[Dict[str, Any]]:
    fields = fragment.content.structured_drawing_fields or []
    return [{"category": f.category, "value": f.value,
             "tolerance": f.tolerance, "bbox": f.bbox} for f in fields]


def _sheet_haystack(sheet) -> str:
    """Всё, что прочитано с листа целиком: текст, поля, таблицы."""
    if sheet is None:
        return ""
    parts = [sheet.content.text or ""]
    for field in sheet.content.structured_drawing_fields or []:
        parts.append(f"{field.value} {field.tolerance or ''}")
    payload = sheet.content.structured_payload or {}
    for table in payload.get("tables") or []:
        parts.extend(str(cell) for row in table.get("rows") or [] for cell in row)
    return "\n".join(parts)


def _missing(crops: List[Dict[str, Any]], haystack: str) -> List[Dict[str, Any]]:
    """
    Значения вырезок, которых в чтении листа нет. Сравнение — по значению
    после нормализации ГОСТ, чтобы «Ф44Н7» и «⌀44H7» считались одним.
    """
    from core.providers import drawing_text
    found = drawing_text.normalize(haystack)
    missing = []
    for field in crops:
        value = drawing_text.normalize(str(field["value"] or ""))
        if value and value not in found:
            missing.append({**field, "normalized": value})
    return missing


def measure(path: pathlib.Path, verify_flag: bool = False) -> Dict[str, Any]:
    from core.drawing_processor import DrawingProcessor
    from core.ladder.drawing_route import volume
    from core.ladder.structure import drawing_structure
    from core.providers.mineru_parser import MinerUParserProvider
    from core.providers.storage import LocalStorageProvider

    storage = LocalStorageProvider(base_path=str(path.parent))
    uri = path.name
    pages = _page_count(path)
    metadata = _metadata(uri, pages)

    # Нарезка видов — настоящая, из MinerU: вырезки, которых флаг и касается.
    parsed = MinerUParserProvider(storage=storage).parse(uri, "pdf", dict(metadata))
    crops_in = [f for f in parsed.fragments if f.type == "image"]

    _flags(DRAWING_PDF_FULL_READ=True, DRAWING_SHEET_READ_ONCE=False)
    processor = DrawingProcessor(storage)
    calls = _timed(processor, uri)
    fragments = processor.enrich_fragments(list(parsed.fragments), dict(metadata))

    sheets = {}
    crop_fields: Dict[int, List[Dict[str, Any]]] = {}
    crop_ids = {f.fragment_id for f in crops_in}
    for fragment in fragments:
        if fragment.type != "drawing" or fragment.position is None:
            continue
        page = int(fragment.position.page)
        payload = fragment.content.structured_payload or {}
        if payload.get("transcript_line_count"):
            sheets[page] = fragment
        elif fragment.fragment_id in crop_ids:
            crop_fields.setdefault(page, []).extend(_values(fragment))

    # «Флаг включён» — тот же результат без полей вырезок прочитанных листов.
    kept = [f for f in fragments
            if not (f.type == "drawing" and f.fragment_id in crop_ids
                    and f.position is not None
                    and int(f.position.page) in sheets)]

    per_page = []
    for page in sorted(set(list(sheets) + list(crop_fields))):
        crops = crop_fields.get(page, [])
        missing = _missing(crops, _sheet_haystack(sheets.get(page)))
        page_calls = [c for c in calls if c["page"] == page]
        per_page.append({
            "page": page,
            "sheet_read": page in sheets,
            "crop_passes": sum(1 for c in page_calls if not c["sheet"]),
            "seconds_crops": round(sum(c["seconds"] for c in page_calls
                                       if not c["sheet"]), 2),
            "seconds_sheet": round(sum(c["seconds"] for c in page_calls
                                       if c["sheet"]), 2),
            "crop_fields": len(crops),
            "missing_from_sheet": missing,
            "missing_measured": [m for m in missing if m["category"] in MEASURED],
        })

    return {
        "file": path.name,
        "pages": pages,
        "crops_from_mineru": len(crops_in),
        "passes_total": len(calls),
        "passes_saved": sum(p["crop_passes"] for p in per_page if p["sheet_read"]),
        "seconds_total": round(sum(c["seconds"] for c in calls), 2),
        "seconds_saved": round(sum(p["seconds_crops"] for p in per_page
                                   if p["sheet_read"]), 2),
        "volume_off": volume(fragments).as_dict(),
        "volume_on": volume(kept).as_dict(),
        "structure_off": drawing_structure(fragments).as_dict(),
        "structure_on": drawing_structure(kept).as_dict(),
        "pages_detail": per_page,
        "verified_flag": _verify(path, metadata, parsed, sheets) if verify_flag else None,
    }


def _verify(path, metadata, parsed, sheets) -> Dict[str, Any]:
    """
    Повтор с включённым флагом: проход листа остался один на страницу, и
    чтение листа то же. Стоит ещё одного прогона модели, поэтому по просьбе.
    """
    from core.drawing_processor import DrawingProcessor
    from core.providers.storage import LocalStorageProvider

    _flags(DRAWING_PDF_FULL_READ=True, DRAWING_SHEET_READ_ONCE=True)
    processor = DrawingProcessor(LocalStorageProvider(base_path=str(path.parent)))
    calls = _timed(processor, path.name)
    again = processor.enrich_fragments(list(parsed.fragments), dict(metadata))
    lines = {int(f.position.page): (f.content.structured_payload or {}).get(
        "transcript_line_count")
        for f in again if f.type == "drawing" and f.position is not None}
    return {
        "crop_passes": sum(1 for c in calls if not c["sheet"]),
        "sheet_passes": sum(1 for c in calls if c["sheet"]),
        "seconds_total": round(sum(c["seconds"] for c in calls), 2),
        "transcript_lines_match": all(
            lines.get(page) == (sheets[page].content.structured_payload or {}).get(
                "transcript_line_count")
            for page in sheets),
    }


# ---------------------------------------------------------------------- сводка
def summarize(rows: List[Dict[str, Any]]) -> str:
    out = []
    saved = sum(r.get("seconds_saved", 0) for r in rows)
    total = sum(r.get("seconds_total", 0) for r in rows)
    passes = sum(r.get("passes_total", 0) for r in rows)
    cut = sum(r.get("passes_saved", 0) for r in rows)
    share = f"{saved / total * 100:.0f} %" if total else "—"
    out.append(f"Документов {len(rows)}; проходов модели {passes}, из них "
               f"убирается {cut}; времени {total:.0f} с, экономия {saved:.0f} с ({share}).")
    lost = [(r["file"], p) for r in rows for p in r.get("pages_detail", [])
            if p.get("missing_measured")]
    if not lost:
        out.append("Величин, прочитанных вырезкой и не прочитанных листом, нет.")
    else:
        out.append(f"Величины, которых лист не прочитал, на {len(lost)} стр.:")
        for name, page in lost:
            for miss in page["missing_measured"]:
                out.append(f"  {name} стр.{page['page']}: "
                           f"{miss['category']} {miss['value']!r} {miss['bbox']}")
    for row in rows:
        off, on = row.get("structure_off", {}), row.get("structure_on", {})
        if off.get("structured") != on.get("structured"):
            out.append(f"  ВЕРДИКТ ДВИНУЛСЯ: {row['file']} "
                       f"structured {off.get('structured')} -> {on.get('structured')}")
        vo, vn = row.get("volume_off", {}), row.get("volume_on", {})
        if vo.get("chars") != vn.get("chars"):
            out.append(f"  символов {vo.get('chars')} -> {vn.get('chars')} "
                       f"({row['file']})")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=pathlib.Path,
                        help="PDF сканов чертежей или готовый JSONL для --summary")
    parser.add_argument("--out", type=pathlib.Path, help="куда писать JSON Lines")
    parser.add_argument("--summary", action="store_true",
                        help="разобрать готовый JSONL и напечатать сводку")
    parser.add_argument("--verify-flag", action="store_true",
                        help="повторить прогон с включённым флагом и сверить")
    args = parser.parse_args()

    if args.summary:
        rows = []
        for path in args.paths:
            with open(path, encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
        print(summarize(rows))
        return 0

    # Без модели замер молча вернул бы пустые листы и «экономию» в ноль:
    # лучше остановиться на входе, чем докладывать такой результат.
    if not os.environ.get("QWEN_ENDPOINT"):
        from core.config import settings
        if not settings.QWEN_ENDPOINT:
            print("QWEN_ENDPOINT не задан: без модели мерить нечего.",
                  file=sys.stderr)
            return 2

    rows = []
    for path in args.paths:
        print(f"-- {path.name}", file=sys.stderr)
        try:
            rows.append(measure(path, verify_flag=args.verify_flag))
        except Exception:  # noqa: BLE001 — один файл не роняет замер
            traceback.print_exc()
            rows.append({"file": path.name, "error": traceback.format_exc(limit=1)})
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
