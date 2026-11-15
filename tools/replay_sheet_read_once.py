"""
Что теряет и что снимает чтение листа один раз (DRAWING_SHEET_READ_ONCE) —
по тому, что прод уже прочитал.

Зачем. Флаг убирает проходы модели по вырезкам видов на странице, где лист
уже прочитан целиком. Вызывать для этого модель заново незачем и даже
вредно: прод сохранил оба чтения каждой страницы — и лист целиком, и
каждую вырезку, — поэтому сравнение идёт по записанному, а разброс модели
между прогонами в него не попадает вовсе. Это не перечитывание, это то,
что прод прочитал на самом деле.

Что считается.
  * Цена. Для каждой страницы, где лист прочитан целиком (есть
    `transcript_line_count`), значения полей вырезок ищутся в чтении листа:
    в тексте листа, его полях и его таблицах. Чего там нет — цена флага, и
    она называется поимённо: страница, категория, значение. Величины
    (размеры, допуски, резьба, шероховатость, радиусы, позиции, выноски)
    считаются отдельно: вырезки добавляли именно их.
  * Вердикт. `core/ladder/drawing_route.volume` и
    `core/ladder/structure.drawing_structure` считаются по фрагментам
    страницы дважды: как есть и без полей вырезок. Видно, двинулся ли
    исход LADDER_DRAWING_OVER_OCR.
  * Время. Из журнала модели (`Qwen-VL итог:`) берутся секунды вызовов по
    тем вырезкам, что лежат на страницах с прочитанным листом. Это и есть
    снимаемое время — замер, а не оценка.

Откуда данные. Выгрузка чертёжных фрагментов из
`platform_parser_ingest_queue.parser_payload` (чтение, без записи) —
колонки `event_id, s3_fileid, fragment_id, page, image_ref, text, fields,
payload` — и строки `Qwen-VL итог:` из журналов воркеров за тот же период.

    python tools/replay_sheet_read_once.py drawfrag.csv.gz \
        --log qwen-lines.gz --out out.jsonl

Важно про область применимости: выгрузка — это корпус, разобранный при
включённых на 14.11 флагах, то есть сегодняшнее поведение прода, а не
поведение флага, который предстоит включить. Сравнение показывает, что
флаг выбросил бы из уже прочитанного, и ничего не говорит о том, как
модель прочитала бы лист иначе.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import pathlib
import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

csv.field_size_limit(10 ** 9)

MEASURED = frozenset(
    {"size", "tolerance", "radius", "thread", "roughness", "position", "callout"}
)
# Строка журнала: один вызов `process` — лист, страница и секунды.
_LINE = re.compile(
    r"Qwen-VL итог: лист (?P<ref>\S+) стр\. (?P<page>\d+).*?(?P<seconds>[\d.]+) с\s*$"
)


def _open(path: pathlib.Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


# ------------------------------------------------------------------ фрагменты
def _fragment(row: Dict[str, str]):
    """Фрагмент контракта из строки выгрузки — теми же моделями, что у прода."""
    from core.models.contract import (
        Content, DrawingField, Fragment, Position, Provenance,
    )
    fields = []
    for raw in json.loads(row["fields"] or "[]") or []:
        try:
            fields.append(DrawingField(**raw))
        except Exception:  # noqa: BLE001 — поле чужой версии не роняет замер
            pass
    payload = json.loads(row["payload"] or "{}") or {}
    page = int(row["page"] or 1)
    return Fragment(
        fragment_id=row["fragment_id"] or "frag",
        type="drawing",
        content=Content(text=row["text"] or None,
                        image_ref=row["image_ref"] or None,
                        structured_drawing_fields=fields or None,
                        structured_payload=payload or None),
        position=Position(page=page, bbox=[0.0, 0.0, 1.0, 1.0], order=0),
        confidence=0.0, completeness=None,
        provenance=Provenance(method="vision_plus_vlm", strategy_level=7,
                              source="drawing"),
    )


def _is_crop(fragment) -> bool:
    return str(fragment.content.image_ref or "").startswith("assets/")


def _transcribed(fragment) -> bool:
    return bool((fragment.content.structured_payload or {}).get("transcript_line_count"))


def _haystack(sheets: List[Any]) -> str:
    """Всё, что прочитано с листа целиком: текст, поля, таблицы."""
    parts: List[str] = []
    for sheet in sheets:
        parts.append(sheet.content.text or "")
        for field in sheet.content.structured_drawing_fields or []:
            parts.append(f"{field.value} {field.tolerance or ''}")
        for table in (sheet.content.structured_payload or {}).get("tables") or []:
            parts.append(str(table.get("title") or ""))
            parts.extend(str(cell) for cell in table.get("headers") or [])
            parts.extend(str(cell) for row in table.get("rows") or []
                         for cell in row)
    return "\n".join(parts)


def _missing(crops: List[Any], haystack: str) -> List[Dict[str, Any]]:
    """
    Значения вырезок, которых в чтении листа нет. Сравнение — по значению
    после нормализации ГОСТ, чтобы «Ф44Н7» и «⌀44H7» были одним значением.

    Короткие числовые значения помечаются: «20» находится в тексте листа
    почти всегда, и такое совпадение ничего не доказывает. Поэтому в итоге
    они считаются отдельно, а не растворяются в общем числе.
    """
    from core.providers import drawing_text
    found = drawing_text.normalize(haystack)
    missing: List[Dict[str, Any]] = []
    for crop in crops:
        for field in crop.content.structured_drawing_fields or []:
            value = drawing_text.normalize(str(field.value or ""))
            if not value:
                continue
            if value in found:
                continue
            # Подстрока — тест строгий: лист мог прочитать то же значение с
            # другими пробелами («P = 16 МПа» против «P=16 МПа»), и тогда
            # «не нашлось» — это промах сравнения, а не промах листа.
            # Поэтому у каждого значения считается доля его собственных
            # заметных слов, которые на листе всё-таки есть.
            share = _token_share(value, found)
            missing.append({
                "category": field.category,
                "value": field.value,
                "tolerance": field.tolerance,
                "bbox": field.bbox,
                "measured": field.category in MEASURED,
                "short": len(value) <= 3,
                "token_share": round(share, 2),
                # Заметных слов на листе почти нет: значение и правда не
                # прочитано, а не прочитано иначе.
                "absent": share < 0.5,
            })
    return missing


_TOKEN = re.compile(r"[0-9A-Za-zА-Яа-яЁё]{3,}")


def _token_share(value: str, haystack: str) -> float:
    """Доля заметных слов значения, которые есть в чтении листа."""
    tokens = _TOKEN.findall(value)
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if t in haystack) / len(tokens)


def _weak_hits(crops: List[Any], haystack: str) -> int:
    """
    Сколько значений «нашлось» только как короткая подстрока. Это предел
    метода, и его лучше назвать числом, чем умолчать.
    """
    from core.providers import drawing_text
    found = drawing_text.normalize(haystack)
    weak = 0
    for crop in crops:
        for field in crop.content.structured_drawing_fields or []:
            value = drawing_text.normalize(str(field.value or ""))
            if value and len(value) <= 3 and value in found:
                weak += 1
    return weak


# ---------------------------------------------------------------------- журнал
def read_log(path: pathlib.Path) -> Tuple[Dict[str, float], float, int]:
    """Секунды по каждому `лист` из журнала, их сумма и число строк."""
    seconds: Dict[str, float] = collections.defaultdict(float)
    total = 0.0
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(
            path).endswith(".gz") else open(
            path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _LINE.search(line.strip())
            if not match:
                continue
            value = float(match.group("seconds"))
            seconds[match.group("ref")] += value
            total += value
            count += 1
    return dict(seconds), total, count


# ----------------------------------------------------------------------- разбор
def replay(csv_path: pathlib.Path, log_path: Optional[pathlib.Path]) -> Dict[str, Any]:
    from core.ladder.drawing_route import volume
    from core.ladder.structure import drawing_structure

    groups: Dict[Tuple[str, int], List[Any]] = collections.defaultdict(list)
    rows = 0
    with _open(csv_path) as handle:
        for row in csv.DictReader(handle):
            rows += 1
            try:
                fragment = _fragment(row)
            except Exception:  # noqa: BLE001 — строка чужой версии не роняет замер
                continue
            groups[(row["event_id"], int(row["page"] or 1))].append(fragment)

    seconds, log_total, log_lines = read_log(log_path) if log_path else ({}, 0.0, 0)

    pages: List[Dict[str, Any]] = []
    affected_refs: List[str] = []
    for (event, page), fragments in sorted(groups.items()):
        sheets = [f for f in fragments if _transcribed(f)]
        crops = [f for f in fragments if _is_crop(f) and not _transcribed(f)]
        if not sheets or not crops:
            continue
        haystack = _haystack(sheets)
        missing = _missing(crops, haystack)
        kept = [f for f in fragments if f not in crops]
        before, after = volume(fragments), volume(kept)
        sbefore, safter = drawing_structure(fragments), drawing_structure(kept)
        refs = [str(c.content.image_ref) for c in crops]
        affected_refs.extend(refs)
        pages.append({
            "event_id": event,
            "page": page,
            "crops": len(crops),
            "crop_fields": sum(len(c.content.structured_drawing_fields or [])
                               for c in crops),
            "sheet_transcript_lines": sum(
                int((s.content.structured_payload or {}).get(
                    "transcript_line_count") or 0) for s in sheets),
            "missing": missing,
            "missing_measured": [m for m in missing if m["measured"]],
            "absent_measured": [m for m in missing
                                if m["measured"] and m["absent"]],
            "weak_hits": _weak_hits(crops, haystack),
            "seconds_crops": round(sum(seconds.get(r, 0.0) for r in refs), 1),
            "logged_crops": sum(1 for r in refs if r in seconds),
            "volume_before": before.as_dict(), "volume_after": after.as_dict(),
            "structured_before": sbefore.structured,
            "structured_after": safter.structured,
            "complete_before": sbefore.complete, "complete_after": safter.complete,
        })

    return {
        "rows": rows,
        "groups": len(groups),
        "pages_flag_applies": len(pages),
        "crops_removed": sum(p["crops"] for p in pages),
        "log_lines": log_lines,
        "log_seconds_total": round(log_total, 1),
        "seconds_saved": round(sum(p["seconds_crops"] for p in pages), 1),
        "crops_found_in_log": sum(p["logged_crops"] for p in pages),
        "pages": pages,
    }


def summarize(result: Dict[str, Any]) -> str:
    pages = result["pages"]
    out: List[str] = []
    out.append(
        f"Фрагментов в выгрузке {result['rows']}; страниц, где флаг действует, "
        f"{result['pages_flag_applies']}; вырезок снимается "
        f"{result['crops_removed']}.")
    saved, total = result["seconds_saved"], result["log_seconds_total"]
    share = f"{saved / total * 100:.1f} %" if total else "—"
    out.append(
        f"Время: журнал {result['log_lines']} вызовов, {total:.0f} с всего; "
        f"снимается {saved:.0f} с ({share}); из снимаемых вырезок нашлось в "
        f"журнале {result['crops_found_in_log']} из {result['crops_removed']}.")

    fields = sum(p["crop_fields"] for p in pages)
    missing = [m for p in pages for m in p["missing"]]
    meas = [m for p in pages for m in p["missing_measured"]]
    weak = sum(p["weak_hits"] for p in pages)
    absent = [m for m in missing if m["absent"]]
    absent_meas = [m for m in meas if m["absent"]]
    out.append(
        f"Цена: полей у вырезок {fields}; не нашлось подстрокой {len(missing)} "
        f"({len(meas)} величин). Из них заметных слов нет на листе у "
        f"{len(absent)} ({len(absent_meas)} величин) — это и есть потеря; "
        f"остальные лист прочитал иначе (пробелы, регистр).")
    out.append(
        f"Пределы метода: совпадений только короткой подстрокой {weak} — "
        f"столько «нашлось» ничего не доказывает.")

    moved = [p for p in pages if p["structured_before"] != p["structured_after"]]
    lost_complete = [p for p in pages
                     if p["complete_before"] and not p["complete_after"]]
    thinner = [p for p in pages
               if p["volume_after"]["chars"] < p["volume_before"]["chars"]]
    out.append(
        f"Вердикт: состав `structured` двинулся на {len(moved)} стр.; "
        f"`complete` потерян на {len(lost_complete)} стр.; символов стало "
        f"меньше на {len(thinner)} стр. из {len(pages)}.")

    by_category = collections.Counter(m["category"] for m in absent_meas)
    if by_category:
        out.append("Потерянные величины по категориям: "
                   + ", ".join(f"{k} {v}" for k, v in by_category.most_common()))
    pages_hit = [p for p in pages if p["absent_measured"]]
    out.append(f"Страниц с потерянными величинами: {len(pages_hit)} из {len(pages)}.")
    worst = sorted(pages, key=lambda p: -len(p["absent_measured"]))[:12]
    for page in worst:
        if not page["absent_measured"]:
            break
        out.append(f"  {page['event_id'][:12]} стр.{page['page']} "
                   f"(вырезок {page['crops']}, строк листа "
                   f"{page['sheet_transcript_lines']}): "
                   f"величин потеряно {len(page['absent_measured'])}")
        for miss in page["absent_measured"][:5]:
            mark = " [короткое]" if miss["short"] else ""
            out.append(f"      {miss['category']} {miss['value']!r}{mark}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=pathlib.Path, help="выгрузка чертёжных фрагментов")
    parser.add_argument("--log", type=pathlib.Path, help="строки Qwen-VL итог:")
    parser.add_argument("--out", type=pathlib.Path, help="куда писать подробности")
    args = parser.parse_args()

    # Значения чертежа — это «⌀», «±», «√»: в кодировке консоли Windows их
    # нет, и сводка падала на выводе уже посчитанного.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 — на поток без перекодировки не ругаемся
        pass

    result = replay(args.csv, args.log)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for page in result["pages"]:
                handle.write(json.dumps(page, ensure_ascii=False) + "\n")
    print(summarize(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
