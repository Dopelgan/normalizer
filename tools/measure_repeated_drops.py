#!/usr/bin/env python
"""
Сколько удалений дедупликатора повторов были не повторами — на сырых
результатах разбора.

Единственное место, где уцелели удалённые блоки, — сырой результат разбора
документа, разобранного с выключенным `OCR_LAYER_DROP_REPEATED`
(`documents.extra_data.raw_parse_uri`, по умолчанию `raw_parse/<doc_id>.json`).
В `chunks` удалённого блока уже нет: `drop_repeated_recovered_blocks`
работает внутри `MinerUParserProvider.parse` до присваивания `result.blocks`,
и сырой результат сохраняется уже после удаления. Поэтому абсолютное число
удалений на проде восстановить нельзя ничем, кроме повторного разбора, а
доля ложных удалений считается здесь — на выборке, где флаг был выключен и
все восстановленные блоки на месте.

Запуск (JSON-файлы уже выкачаны локально):

    python tools/measure_repeated_drops.py /путь/к/raw_parse

Как выбрать выборку (prod, только чтение):

    select id, extra_data->>'raw_parse_uri' from documents
     where extra_data->>'source_kind' in ('scanned_pdf','image')
       and coalesce((extra_data->'text_layer_stats'->>'dropped_repeated')::int,0)=0
       and extra_data->'text_layer_stats'->>'layer'='ocr';

Выводится: сколько блоков удалило бы прежнее правило (доля основ по
объединению слов всей страницы, без порядка, порог `--old-share`), сколько
удаляет действующее (содержание в одном блоке по порядку слов, порог
`--contain`) и какая доля удалений прежнего правила была настоящим текстом.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from core.models.parse_result import ParsedBlock  # noqa: E402
from core.providers.mineru_parser import (  # noqa: E402
    _LAT_WORD_RE,
    _MINERU_PAGE_TEXT_CHARS,
    _TEXTISH,
    _contained_share,
    _stems,
    drop_repeated_recovered_blocks,
)
from core.providers.text_layer import METHOD_OCR_LAYER_RECOVERED as REC  # noqa: E402


def _old_rule_drops(blocks: list[ParsedBlock], min_share: float) -> list[ParsedBlock]:
    """Прежнее правило: доля основ по объединению слов страницы, без порядка."""
    page_chars: dict[int, int] = {}
    page_stems: dict[int, set] = {}
    for b in blocks:
        if b.method == REC:
            continue
        if b.type in _TEXTISH or b.type == "table":
            page_stems.setdefault(b.page, set()).update(_stems(b.text or ""))
        if b.type in _TEXTISH and b.text:
            page_chars[b.page] = page_chars.get(b.page, 0) + len("".join(b.text.split()))
    dropped = []
    for b in blocks:
        if b.method != REC or page_chars.get(b.page, 0) < _MINERU_PAGE_TEXT_CHARS:
            continue
        stems = _stems(b.text or "")
        known = page_stems.get(b.page, set())
        if not stems or sum(s in known for s in stems) >= min_share * len(stems):
            dropped.append(b)
    return dropped


def _prose(text: str, stems: list[str]) -> bool:
    letters = sum(1 for ch in text if ch.isalpha())
    cyr = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    return len(stems) >= 5 and bool(letters) and cyr / letters >= 0.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="каталог с сырыми результатами разбора (*.json)")
    parser.add_argument("--contain", type=float, default=0.9, help="порог нового правила")
    parser.add_argument("--old-share", type=float, default=0.6, help="порог прежнего правила")
    parser.add_argument("--examples", type=int, default=5, help="сколько примеров печатать")
    args = parser.parse_args()

    files = sorted(pathlib.Path(args.path).glob("*.json"))
    if not files:
        print(f"В {args.path} нет файлов *.json", file=sys.stderr)
        return 2

    docs = old_total = new_total = rescued = rescued_prose = 0
    docs_old = docs_rescued = 0
    examples: list[tuple[float, str, int, str]] = []
    for path in files:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — один битый файл не повод падать
            print(f"{path.name}: не прочитан ({exc})", file=sys.stderr)
            continue
        blocks = [ParsedBlock.model_validate(b) for b in raw.get("blocks") or []]
        if not any(b.method == REC for b in blocks):
            continue
        docs += 1
        old = _old_rule_drops(blocks, args.old_share)
        _, counts = drop_repeated_recovered_blocks(blocks, args.contain)
        old_total += len(old)
        new_total += counts["total"]
        if old:
            docs_old += 1
        here = 0
        for b in old:
            text = b.text or ""
            stems = _stems(text)
            if not stems:
                # Правило графики: мусор обводки по-прежнему удаляется, а
                # похожее на текст (русские слова латиницей) — остаётся.
                if _LAT_WORD_RE.search(text):
                    here += 1
                    rescued_prose += _prose(text, stems)
                    examples.append((0.0, path.stem, b.page, text))
                continue
            others = [_stems(o.text or "") for o in blocks
                      if o.method != REC and o.page == b.page and _stems(o.text or "")]
            best = max((_contained_share(stems, o) for o in others), default=0.0)
            if best < args.contain:
                here += 1
                rescued_prose += _prose(text, stems)
                examples.append((best, path.stem, b.page, text))
        rescued += here
        if here:
            docs_rescued += 1

    print(f"документов с восстановленными блоками: {docs}")
    print(f"удалило бы прежнее правило (порог {args.old_share}): {old_total}"
          f" в {docs_old} документах")
    print(f"удаляет действующее правило (порог {args.contain}): {new_total}")
    share = (100.0 * rescued / old_total) if old_total else 0.0
    print(f"сохранено настоящего текста: {rescued} ({share:.1f}% удалений прежнего правила)"
          f" в {docs_rescued} документах")
    print(f"из них связная русская проза: {rescued_prose}")
    if args.examples:
        print("\n--- примеры сохранённого ---")
        examples.sort(key=lambda e: -len(e[3]))
        for best, doc, page, text in examples[: args.examples]:
            print(f"[{best:.2f}] {doc} стр.{page}\n    {text[:300]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
