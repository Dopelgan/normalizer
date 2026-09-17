# Parser fixture test matrix

## Intake / quality gate

| ID | File | Expected | What to verify |
|---|---|---|---|
| T01 | valid_mixed.pdf | accept | 2 pages, text/table fragments, provenance
| T02 | valid_mixed.docx | accept | exact text, table structure
| T03 | valid_table.xlsx | accept | rows/cells and formula-derived totals
| T04 | valid_table.csv | accept | tabular strategy
| T05 | valid_drawing.dxf | accept | CAD/vector strategy
| T06 | valid_image.png | accept | image/OCR path
| T07 | личное_отпуск.jpg | quarantine/reject | Data Gateway classification by profile
| T08 | unsupported.dwg | reject | unsupported closed binary format
| T09 | corrupt.pdf | reject/quarantine | unreadable technical artifact
| T10 | low_quality.png | quarantine/degraded | readability gate and degraded OCR
| T11 | no_extension | discover/reject | extension fallback behavior
| T12 | duplicate of valid_text.txt | quarantine/reject | duplicate detection

## Parse / polling

- Submit T01/T02/T03 via `/internal/v1/parse` and verify one result per file.
- Submit T01 via `/internal/v1/parse/background`, then poll `/internal/v1/parse/results/{request_id}` until terminal status.
- Run two concurrent requests for the same `s3_fileid`; each request must receive its own result and must not create duplicate chunks.

## Content assertions

The strings `expires_at`, `rps`, `2026-12-31`, `8.5 bar`, and `X = 3.14 * D` are deliberate exact-token checks.
