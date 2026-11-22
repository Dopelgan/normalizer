"""Bound physical table rows for the existing Parser→RAG structural item ceiling."""

import hashlib
import json

from core.models.contract import TableData

TABLE_MAX_ITEMS = 100_000
TABLE_MAX_PARTS = 4000


def _items(value):
    return 1 + (sum(_items(v) for v in value.values()) if isinstance(value, dict) else
                sum(_items(v) for v in value) if isinstance(value, list) else 0)


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def table_parts(table):
    """Partition after source identity attachment; never prune cell metadata."""
    whole = table.model_dump()
    if whole.get("table_id") is None:
        raise ValueError("table parts require a table_id")
    if _items(whole) <= TABLE_MAX_ITEMS:
        return [table]
    source = whole.get("source_structure")
    if not isinstance(source, dict) or source.get("schema") != "source-table-v1":
        raise ValueError("TABLE_RESOURCE_LIMIT: oversized table has no physical source grid")
    physical = source.get("rows") or []
    data_rows = source.get("data_rows") or []
    if len(data_rows) != len(whole["rows"]) or not physical:
        raise ValueError("TABLE_RESOURCE_LIMIT: source row projection cannot be partitioned")
    if (any(type(number) is not int or not 1 <= number <= len(physical) for number in data_rows)
            or data_rows != sorted(set(data_rows))):
        raise ValueError("TABLE_RESOURCE_LIMIT: source data row mapping is invalid")
    data = dict(zip(data_rows, whole["rows"]))
    if len(data) != len(data_rows) or [r["row"] for r in physical] != list(range(1, len(physical) + 1)):
        raise ValueError("TABLE_RESOURCE_LIMIT: physical source rows are not contiguous")
    digest = _digest({k: v for k, v in whole.items() if k not in {"table_id", "row_start", "row_end"}})
    anchor_names = {m["anchor"] for m in source.get("merges", [])}
    anchors = {c["coordinate"]: c for row in physical for c in row["cells"]
               if c["coordinate"] in anchor_names}
    if set(anchors) != anchor_names:
        raise ValueError("TABLE_RESOURCE_LIMIT: source merge anchor is missing")
    full_cells = sum(len(row["cells"]) for row in physical)

    def build(rows, start, index=0, count=1):
        numbers = {r["row"] for r in rows}
        selected = [r for r in data_rows if r in numbers]
        grid = {**source, "rows": rows, "data_rows": selected,
                "part_index": index, "part_count": count,
                "physical_row_start": rows[0]["row"], "physical_row_end": rows[-1]["row"],
                "full_row_count": len(physical), "full_cell_count": full_cells,
                "whole_table_digest": digest,
                "merge_anchors": [cell for cell in anchors.values() if cell["row"] not in numbers]}
        # Первая часть держит table_id целой таблицы, следующие получают свой
        # номер: у каждой части должен быть отдельный table_id.
        table_id = whole["table_id"]
        if index > 0:
            table_id = f"{table_id}:part-{index + 1}"
        result = {**whole, "table_id": table_id,
                  "rows": [data[r] for r in selected], "source_structure": grid,
                  "row_start": start, "row_end": start + len(selected),
                  "total_row": whole["total_row"] if rows[-1]["row"] == len(physical) else None}
        return TableData(**result)

    anchor_cost = {}
    for cell in anchors.values():
        anchor_cost[cell["row"]] = anchor_cost.get(cell["row"], 0) + _items(cell)

    def delta(row):
        number = row["row"]
        return (_items(row) + (_items(data[number]) + 1 if number in data else 0)
                - anchor_cost.get(number, 0)
                + (_items(whole["total_row"]) - 1 if number == len(physical) else 0))

    # Every variable-size container is additive. Scalar range/index/count fields
    # have constant item cost, so the exact final model budget can be packed once.
    base_cost = _items(build([physical[0]], 0).model_dump()) - delta(physical[0])
    groups, current, start, cost = [], [], 0, base_cost
    for row in physical:
        addition = delta(row)
        if cost + addition > TABLE_MAX_ITEMS:
            if not current:
                raise ValueError("TABLE_RESOURCE_LIMIT: one physical row/anchors exceeds item budget")
            groups.append((current, start))
            if len(groups) >= TABLE_MAX_PARTS:
                raise ValueError("TABLE_RESOURCE_LIMIT: table parts exceed fragment ceiling")
            start += len([r for r in current if r["row"] in data])
            current = [row]
            cost = base_cost + addition
            if cost > TABLE_MAX_ITEMS:
                raise ValueError("TABLE_RESOURCE_LIMIT: one physical row/anchors exceeds item budget")
        else:
            current.append(row)
            cost += addition
    groups.append((current, start))
    result = [build(rows, start, index, len(groups)) for index, (rows, start) in enumerate(groups)]
    if any(_items(part.model_dump()) > TABLE_MAX_ITEMS for part in result):
        raise ValueError("TABLE_RESOURCE_LIMIT: final table part exceeds item budget")
    return result
