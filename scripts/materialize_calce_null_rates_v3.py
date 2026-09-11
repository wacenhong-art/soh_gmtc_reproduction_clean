"""Create a versioned null-rate view from the completed raw-row audit.

No raw data are opened or altered.  This preserves the original audit artifact
and makes the requested NULL_RATE explicit for every audited column.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "data_readiness"


def main() -> None:
    source = OUT / "calce_null_duplicate_audit.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    for sheet in payload["column_null_audit"]:
        for values in sheet.get("column_null_audit", {}).values():
            total = values["TOTAL_ROWS"]
            values["NULL_RATE"] = values["NULL_COUNT"] / total if total else None
    payload["derived_from"] = str(source.relative_to(ROOT))
    payload["null_rate_definition"] = "NULL_COUNT / TOTAL_ROWS, per battery/workbook/sheet/column; source raw-row audit was complete."
    payload["artifact_version"] = "v3_explicit_null_rate"
    (OUT / "calce_null_duplicate_audit_v3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
