"""Create a source-keyed identity v3 view without replacing prior evidence."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "data_readiness"


def main() -> None:
    identity = json.loads((OUT / "calce_cycle_identity_audit.json").read_text(encoding="utf-8"))
    boundary = json.loads((OUT / "calce_cycle_boundary_audit_v3.json").read_text(encoding="utf-8"))
    summaries = {item["BATTERY_ID"]: item for item in boundary["battery_summaries"]}
    for battery in identity["batteries"]:
        summary = summaries[battery["battery_id"]]
        duplicates = sum(x["DUPLICATE_REPORTED_CYCLE_INDEX_SEGMENTS_WITHIN_WORKBOOK"] for x in summary["WORKBOOK_SUMMARY"])
        behavior = battery["cycle_index_behavior"]
        behavior["CYCLE_INDEX_DUPLICATES_WITHIN_WORKBOOK"] = duplicates > 0
        behavior["CONTIGUOUS_SEGMENT_AUDIT"] = "output/data_readiness/calce_cycle_boundary_audit_v3.json"
        behavior["SOURCE_KEY_SCHEMA"] = summary["CANONICAL_SOURCE_KEY_SCHEMA"]
    identity["derived_from"] = ["output/data_readiness/calce_cycle_identity_audit.json", "output/data_readiness/calce_cycle_boundary_audit_v3.json"]
    identity["artifact_version"] = "v3_source_keyed_identity_correction"
    (OUT / "calce_cycle_identity_audit_v3.json").write_text(json.dumps(identity, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
