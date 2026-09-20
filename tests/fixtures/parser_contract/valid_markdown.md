# Test procedure

## Acceptance criteria
- status is `approved`
- expires_at is preserved exactly
- UTF-8 text remains intact

| Parameter | Value |
|---|---|
| Pressure | 8.5 bar |
| Temperature | 72 C |
| Flow rate | 120 l/min |
| Duration | 30 min |

## Method

Bring the rig up to the pressure in the table above and hold it there for the
duration given in the same table. Record the readings once a minute; the
operator signs the log at the end of the run and attaches it to this
procedure. A run interrupted for any reason is void and repeated in full.

## Reporting

The report states the serial number of the rig, the name of the operator and
the exact value of every parameter above. Values are quoted as written here,
including units: `8.5 bar`, not `8,5 bar` and not `850 kPa`. The formula
`X = 3.14 * D` is quoted verbatim as well, with the spaces around the signs.
The field `expires_at` is copied from the accompanying text file unchanged.
