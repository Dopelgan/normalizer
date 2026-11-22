# Source table coordinates

Processing profile `structural-rag-v2` introduces this source grid. It is enabled
only by `STRUCTURAL_RAG_V2=true` (`core/config.py`, default `false`). With the flag
off the Parser output is byte-identical to `structural-rag-v1`: profile version,
legacy `headers/rows/total_row` tables without `source_structure`, no table parts,
PDF span text joined without geometric spaces, v1 source-unit identity, and no
`text_observations` / `text_correction_evidence`. The effective version comes from
`core/processing_contract.processing_profile_version()`; the cached-result gate in
`ingest/tasks.py` compares the stored version with it, so switching the flag either
way reprocesses documents instead of serving the other profile from the cache.
Cached results from the other profile are reprocessed by the existing task profile
gate; they cannot masquerade as fresh results carrying the new coordinates. The additive
grid schema remains `source-table-v1`. Model revision and chunking settings are
unchanged. `tests/unit/test_tasks.py::TestStructuralProfile` exercises the actual
task decision and parser invocation count, plus fresh/cache equality with merge
anchors and partial/unknown-page coverage.

`table_data.source_structure` is an additive `source-table-v1` contract for ODS,
XLSX and CSV. `headers`, `rows` and `total_row` remain the legacy display projection.
The source grid is independent of that projection; consumers must not derive
physical positions by adding one to a legacy row index.

* `rows[].row`, `cells[].row/column` are one-based physical sheet indices.
  `coordinate` is the corresponding A1 locator, scoped by fragment position.sheet.
  Blank rows/cells present in the source grid are retained here.
* `header_rows`, `title_rows` and `data_rows` map the legacy projection back to
  physical rows. Explicit ODS header groups are preserved, including multiple
  rows. Without explicit markup the first nonempty row is a layout candidate;
  a leading singleton in an explicit horizontal merge is a title candidate.
  This heuristic does not prove the semantic meaning of column headers.
* `merges` retain inclusive physical ranges and anchors. Each represented cell
  inside a merge has `merge_anchor`; its original `display` stays literal.
  `effective_display` inherits only from that explicit anchor, never from a
  preceding unmerged row. ODS repeat counts are retained on expanded cells.
* `raw_value`, `value_type`, `formula`, `cached_value` and `style_id` preserve
  the source representation where available. XLSX numeric values and style IDs
  are read from XML before openpyxl date conversion; `number_format` accompanies
  them. A date-looking display therefore does not establish a business event.
  XLSX shared strings retain their XML index in raw_value and decoded display;
  inline strings additionally retain inline_text. Formula attributes are retained
  for shared/array formulas, without claiming to evaluate them.
* `display_origin` distinguishes ODS source text from openpyxl's converted value.
  XLSX display is not a pixel-exact Excel number-format renderer. Missing formula
  cache is explicit in `formula_cache_status`; formula text fallback is not a
  computed result. `sheet_visibility` is metadata, not an ACL.
* `coverage=complete` means the supported source grid was read within budget;
  it does not certify header inference, formula recalculation or semantic facts.
  A sheet (or CSV file) without any nonempty display emits no fragment, as in v1.
  Invalid XML/repeats or exceeded resource
  budgets raise an extraction error; no successful truncated prefix is returned.
  XLSX ZIP compressed input and total expanded entries are bounded by the existing
  `QG_MAX_SIZE_BYTES` ceiling before any member is read. Source worksheet XML is
  parsed incrementally, with the cell budget applied during parsing. Worksheet
  association uses its workbook relationship and exact name, excluding chartsheets.
  The grid extends to the last nonempty cell (value, formula or inline text) and
  only that extent counts toward `SPREADSHEET_MAX_CELLS`. A declared `<dimension>`
  neither extends nor hides the grid; style-only empty cells carry no XML record;
  merges anchored outside the extent are dropped and the rest are applied clipped
  to it (their recorded range stays literal). ODS repeated rows/columns after the
  last cell with text, value, formula, cover or span are not expanded and do not
  count toward the budget; repeated blank rows and cells inside the extent are kept.

The whole table contributes to source-unit and final fragment identity. A change
in raw values, merge structure or physical coordinates invalidates the identity
even if the displayed legacy rows happen to match. Existing XLS/BIFF extraction
has not gained this source grid and must not be assumed to provide these fields.

Large physical grids are emitted as bounded transport parts **after** the whole
source-unit identity is calculated. Each final `TableData.model_dump()` stays
within the consumer's existing 100,000-item ceiling, including repeated headers,
defaults, merges and proof fields. An unsplittable physical row/anchor set produces
`TABLE_RESOURCE_LIMIT`; no cell fields are discarded. Parts keep the same
`source_unit_id`, absolute source coordinates and complete merge ranges. Each part
has its own `table_id`: the first keeps the whole-table id, part N > 1 gets
`<id>:part-N` (nested payload tables likewise get `<unit>:part-N:table-K`).
`merge_anchors` carries full anchor records outside the part's physical rows.

The flat source proof contains zero-based `part_index`, positive `part_count`,
inclusive one-based `physical_row_start/end`, `full_row_count`, `full_cell_count`
and `whole_table_digest`. The digest covers the original normalized table without
`table_id` or legacy row ranges, using sorted compact UTF-8 JSON. Legacy
`row_start/end` are zero-based half-open data-row ranges; `data_rows` maps only the
part's legacy rows to absolute physical rows. Physical title/header/blank rows are
partitioned once rather than silently lost. The last part alone retains total_row.
RAG verifies group identity, indices/counts, contiguity, coordinates, anchor copies
and the whole digest before accepting a complete group. A missing part rejects a
completed document; an explicitly partial document keeps known parts with an
incomplete-group warning and cannot establish whole-table totals.

Sync/results HTTP responses, including timeout partial envelopes, are encoded with
the exact UTF-8 JSON representation and capped at 16,000,000 bytes. Oversized
envelopes return HTTP 413 with `PARSER_RESPONSE_RESOURCE_LIMIT`, without a document
prefix or a completed status. The 413 body keeps the endpoint's response model:
`/parse/results` returns a `ResultsResponse` (request_id, event_id, operation,
`status=error`, `documents=[]`, error); `/parse` returns the `SyncPendingResponse`
shape of its 504 (a `ParseResponse` with request_id, dialog_id, `documents=[]` plus
`status=error` and error). Per-part item limits do not claim that an arbitrary
large document fits this independent whole-response byte budget.

Focused causal regressions: `tests/unit/test_source_table_coordinates.py`.
Run these and `tests/unit/test_structural_contract.py` in the project's authorized
Docker runner. The synthetic gold fixes literal XML positions/merges before the
implementation, and is independent of customer file names or serial values.

ODS and XLSX share the existing expanded ZIP byte budget and duplicate-member
preflight. No source member is opened before these checks. A rejected container
does not become a truncated successful table.

XLSX `position.page` is the worksheet ordinal, as in v1: chart sheets are not
numbered. A chart before a worksheet does not rebind the worksheet's XML; the
worksheet is page 1 and the source-count/page-coverage checks keep the unread chart
as an unknown page. Empty worksheets emit no fragment but keep their ordinal.

Text correction evidence (v2 only): `text_correction_evidence` entries of the
mixed-script rule carry `raw_sha256`, `raw_length`, page/bbox and per-edit
`source_start/source_end/original/replacement`, not a copy of the full raw text;
the raw text is reconstructable from the emitted text and the edits. Repeating the
repair over the same raw text replaces its entry instead of appending a duplicate.
