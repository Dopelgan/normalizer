"""
Прогон обнаружения рамки формата по уже разобранным листам.

Зачем. Восстановление рамки (`core/providers/drawing_vision.py`, раздел
«Восстановление рамки») включается флагами, и каждый флаг надо мерить до
того, как его включат: сколько листов получили рамку, сколько сохранили
прежнюю, сколько сменили её на другую и — главное — по какой причине
механизм отказался. Этот прогон отвечает на всё это по корпусу разобранных
листов, не трогая ни прод, ни очередь.

Как устроено. На вход — выгрузка разметки с прода и те же картинки, на
которых она считалась:

    layouts.csv   колонки src, page, image_ref, layout (JSON разметки);
                  выгружается из platform_parser_ingest_queue.parser_payload,
                  запрос — в docs/frame-recovery-measurement-20261114.md;
    --images      каталог, где по image_ref лежат картинки страниц;
    --sources     каталог исходных файлов и MAP.json вида
                  {"0001.pdf": "<src из выгрузки>"} для листов, у которых
                  image_ref — это PDF (тогда страница рендерится заново).

Для каждого листа разметка считается дважды: с выключенными флагами (обязана
совпасть с выгруженной — это проверка, что прогон воспроизводит бой) и, если
рамки нет, по одному разу на каждый механизм. Механизмы не складываются:
восстановление вызывается только там, где `_frame` вернул `None`, поэтому
сменить уже найденную рамку оно не может, и в выводе это видно.

Итог — JSON Lines, по строке на лист. Разбор и сводка — `--summary`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("QWEN_ENDPOINT", "http://localhost:1")

csv.field_size_limit(10 ** 9)

MECHANISMS: Tuple[Tuple[str, Dict[str, bool]], ...] = (
    ("stitch", {"DRAWING_FRAME_STITCH": True}),
    ("four_sides", {"DRAWING_FRAME_FOUR_SIDES": True}),
    ("title_block", {"DRAWING_FRAME_FROM_TITLE_BLOCK": True}),
    ("all", {"DRAWING_FRAME_STITCH": True, "DRAWING_FRAME_FOUR_SIDES": True,
             "DRAWING_FRAME_FROM_TITLE_BLOCK": True}),
)
FLAGS = ("DRAWING_FRAME_STITCH", "DRAWING_FRAME_FOUR_SIDES",
         "DRAWING_FRAME_FROM_TITLE_BLOCK")


def _set_flags(wanted: Dict[str, bool]) -> None:
    from core.config import settings
    for flag in FLAGS:
        setattr(settings, flag, wanted.get(flag, False))


def index_images(images: pathlib.Path, sources: Optional[pathlib.Path]):
    """image_ref -> файл картинки; src -> исходный файл."""
    by_ref: Dict[str, pathlib.Path] = {}
    for path in images.rglob("*.jpg"):
        by_ref["assets/%s/%s" % (path.parent.name, path.name)] = path
    mapping = images / "MAP.json"
    if mapping.exists():
        for name, ref in json.loads(mapping.read_text(encoding="utf-8")).items():
            by_ref[ref] = images / name

    by_src: Dict[str, pathlib.Path] = {}
    if sources is not None and (sources / "MAP.json").exists():
        for name, src in json.loads((sources / "MAP.json").read_text(encoding="utf-8")).items():
            by_src[src] = sources / name
    return by_ref, by_src


def open_sheet(row: Dict[str, str], by_ref, by_src):
    """Лист ровно так, как его открывал разбор: картинка или страница PDF."""
    from PIL import Image
    from core.config import settings

    ref = (row.get("image_ref") or "").strip()
    if ref.lower().endswith(".jpg"):
        path = by_ref.get(ref)
        if path is None:
            return None, "нет картинки страницы"
        return Image.open(path).convert("RGB"), ""

    path = by_src.get(row["src"])
    if path is None:
        return None, "нет исходного файла"
    import fitz
    document = fitz.open(path)
    try:
        page = int(row["page"] or 1)
        if not 1 <= page <= document.page_count:
            return None, "страницы нет в файле"
        pixmap = document[page - 1].get_pixmap(dpi=settings.DRAWING_RENDER_DPI)
        return Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples), ""
    finally:
        document.close()


def replay(row: Dict[str, str], by_ref, by_src) -> Dict[str, Any]:
    from core.providers import drawing_vision as vision

    image, why = open_sheet(row, by_ref, by_src)
    if image is None:
        return {"skip": why}

    _set_flags({})
    base = vision.analyse(image)
    stored = (json.loads(row["layout"]) or {}).get("frame") if row.get("layout") else None
    result: Dict[str, Any] = {
        "before": base.frame,
        "stored": stored,
        "faithful": base.frame == stored,
        "angle": round(base.angle, 2),
        "turn": base.turn,
        "size": list(base.image.size),
        "stamp": base.title_block.bbox if base.title_block else None,
    }
    if base.frame is not None:
        # Восстановление сюда не доходит — рамка уже есть.
        for name, _ in MECHANISMS:
            result[name] = {"frame": base.frame, "source": None, "format_mm": None,
                            "refused": None}
        return result

    for name, wanted in MECHANISMS:
        _set_flags(wanted)
        signals = dict(base.signals)
        frame = vision._recover_frame(base.image, base.regions, signals)
        report = signals.get("frame_recovery") or {}
        result[name] = {"frame": frame, "source": report.get("source"),
                        "format_mm": report.get("format_mm"),
                        "format_certain": report.get("format_certain"),
                        "refused": report.get("refused")}
    return result


def run(args: argparse.Namespace) -> None:
    by_ref, by_src = index_images(pathlib.Path(args.images),
                                  pathlib.Path(args.sources) if args.sources else None)
    with open(args.layouts, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for number, row in enumerate(rows) if number % args.shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    with open(args.out, "w", encoding="utf-8") as output:
        for number, row in enumerate(rows):
            record = {"src": row["src"], "page": row["page"],
                      "ref": row.get("image_ref", "")}
            try:
                record.update(replay(row, by_ref, by_src))
            except Exception as error:               # noqa: BLE001 — прогон не должен падать целиком
                record["error"] = "%r" % (error,)
                record["trace"] = traceback.format_exc()[-400:]
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if number % 25 == 0:
                print("%d/%d" % (number, len(rows)), flush=True)


def _whole_sheet(record: Dict[str, Any]) -> bool:
    """Целый лист, а не вырезка: не мельче A5 в пикселях и с пропорцией листа."""
    width, height = record["size"]
    return min(width, height) >= 700 and 1.2 <= max(width, height) / min(width, height) <= 1.65


def summarise(paths: Iterable[str]) -> None:
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)

    good = [r for r in records if not r.get("skip") and not r.get("error")]
    print("листов: %d, разобрано %d, пропущено %d, ошибок %d" % (
        len(records), len(good),
        sum(1 for r in records if r.get("skip")),
        sum(1 for r in records if r.get("error"))))
    print("прогон воспроизводит бой: %d из %d"
          % (sum(1 for r in good if r["faithful"]), len(good)))

    had = [r for r in good if r["before"] is not None]
    none = [r for r in good if r["before"] is None]
    print()
    print("рамка была: %d; сменилась на другую: %d"
          % (len(had), sum(1 for r in had
                           if any(r[name]["frame"] != r["before"] for name, _ in MECHANISMS))))
    print("рамки не было: %d (целых листов %d)"
          % (len(none), sum(1 for r in none if _whole_sheet(r))))
    print()
    for name, _ in MECHANISMS:
        got = [r for r in none if r[name]["frame"]]
        formats: Dict[str, int] = {}
        for record in got:
            key = record[name]["format_mm"] or "-"
            formats[key] = formats.get(key, 0) + 1
        unsure = sum(1 for r in got if r[name].get("format_certain") is False)
        print("%-12s восстановлено %4d (целых листов %3d; формат под сомнением %d) %s"
              % (name, len(got), sum(1 for r in got if _whole_sheet(r)), unsure,
                 dict(sorted(formats.items(), key=lambda item: -item[1]))))

    print()
    print("причины отказа сшивки на целых листах:")
    reasons: Dict[str, int] = {}
    missing_side = 0
    misses = [r for r in none if _whole_sheet(r) and not r["stitch"]["frame"]]
    for record in misses:
        why = (record["stitch"]["refused"] or {}).get("stitch", "-")
        if why.startswith("нет линии у края"):
            missing_side += 1
        head = why.split(" (кандидатов")[0].split(";")[0]
        reasons[head] = reasons.get(head, 0) + 1
    for why, count in sorted(reasons.items(), key=lambda item: -item[1])[:15]:
        print("  %5d  %s" % (count, why))
    print()
    print("целых листов без рамки, сшивка не помогла: %d" % len(misses))
    print("  нет хотя бы одной стороны в картинке: %d" % missing_side)
    print("  все четыре стороны видны (адресуемые): %d" % (len(misses) - missing_side))


def export_csv(paths: Iterable[str], destination: str) -> None:
    """
    Целые листы, на которых рамки не было, — в CSV: что сделал каждый
    механизм и от чего отказался. По этой таблице отказы проверяются
    поштучно.

    Вырезки MinerU и иллюстрации внутри документов сюда не идут: рамки на
    них нет и быть не должно, а в таблице они только мешают — их на порядок
    больше, чем листов.
    """
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle)
    rows = [r for r in records
            if not r.get("skip") and not r.get("error") and r["before"] is None
            and _whole_sheet(r)]

    columns = ["src", "page", "image_ref", "width", "height", "whole_sheet",
               "angle", "turn", "stamp_found"]
    for name, _ in MECHANISMS:
        columns += ["%s_frame" % name, "%s_format_mm" % name,
                    "%s_format_certain" % name, "%s_refused" % name]

    with open(destination, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for record in sorted(rows, key=lambda r: (r["src"], int(r["page"] or 0))):
            row = [record["src"], record["page"], record["ref"],
                   record["size"][0], record["size"][1],
                   int(_whole_sheet(record)), record["angle"], record["turn"],
                   int(bool(record["stamp"]))]
            for name, _ in MECHANISMS:
                outcome = record[name] or {}
                frame = outcome.get("frame")
                refused = outcome.get("refused") or {}
                row += [
                    "" if not frame else " ".join("%.4f" % c for c in frame),
                    outcome.get("format_mm") or "",
                    "" if outcome.get("format_certain") is None
                    else int(outcome["format_certain"]),
                    "; ".join("%s: %s" % (key, value) for key, value in refused.items()),
                ]
            writer.writerow(row)
    print("целых листов без рамки выгружено: %d -> %s" % (len(rows), destination))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layouts", help="выгрузка разметки, CSV")
    parser.add_argument("--images", help="каталог картинок страниц")
    parser.add_argument("--sources", help="каталог исходных файлов с MAP.json")
    parser.add_argument("--out", help="куда писать JSON Lines")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary", nargs="*", help="свести готовые JSON Lines")
    parser.add_argument("--csv", help="выгрузить листы без рамки в CSV (с --summary)")
    args = parser.parse_args()

    if args.summary:
        summarise(args.summary)
        if args.csv:
            export_csv(args.summary, args.csv)
        return
    if not (args.layouts and args.images and args.out):
        parser.error("нужны --layouts, --images и --out (или --summary)")
    run(args)


if __name__ == "__main__":
    main()
