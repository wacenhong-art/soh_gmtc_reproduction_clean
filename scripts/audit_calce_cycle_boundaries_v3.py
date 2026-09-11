"""Persist source-keyed contiguous CALCE Cycle_Index segments (read-only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_calce_readiness_v2 import BATTERIES, OUT, RAW, iter_rows, number, shared_strings, workbook_sheets  # noqa: E402
import zipfile


def scan_battery(battery: str):
    rows, per_workbook = [], []
    for workbook in sorted((RAW / battery / battery).glob("*.xlsx")):
        workbook_rows = []
        with zipfile.ZipFile(workbook) as archive:
            strings = shared_strings(archive)
            for sheet, xml in workbook_sheets(archive):
                source = iter_rows(archive, xml, strings)
                _, header = next(source, (None, None))
                if not header:
                    continue
                mapping = {value: key for key, value in header.items() if value}
                if "Cycle_Index" not in mapping:
                    continue
                active = None
                ordinal = 0
                def close(segment):
                    if segment is None:
                        return
                    segment["BOUNDARY_STATUS"] = "EXACT__CONTIGUOUS_REPORTED_CYCLE_INDEX_RUN"
                    rows.append(segment); workbook_rows.append(segment)
                for row_number, values in source:
                    cycle = values.get(mapping["Cycle_Index"])
                    timestamp = values.get(mapping.get("Date_Time", ""))
                    if active is None or cycle != active["REPORTED_CYCLE_INDEX"]:
                        close(active); ordinal += 1
                        key = f"{battery}|{workbook.relative_to(Path(__file__).resolve().parents[1])}|{sheet}|segment={ordinal}|cycle={cycle}"
                        active = {"BATTERY_ID": battery, "SOURCE_WORKBOOK": str(workbook.relative_to(Path(__file__).resolve().parents[1])),
                                  "SOURCE_SHEET": sheet, "REPORTED_CYCLE_INDEX": cycle,
                                  "CANONICAL_CALCE_SOURCE_KEY": key, "START_ROW": row_number, "END_ROW": row_number,
                                  "START_TIMESTAMP": timestamp, "END_TIMESTAMP": timestamp, "ROW_COUNT": 1,
                                  "OPERATION_OR_SEGMENT_TYPE_IF_DETERMINABLE": "UNRESOLVED__REQUIRES_OPERATION_RULE"}
                    else:
                        active["END_ROW"] = row_number; active["END_TIMESTAMP"] = timestamp; active["ROW_COUNT"] += 1
                close(active)
        reported = [r["REPORTED_CYCLE_INDEX"] for r in workbook_rows if r["REPORTED_CYCLE_INDEX"] is not None]
        per_workbook.append({"SOURCE_WORKBOOK": str(workbook.relative_to(Path(__file__).resolve().parents[1])),
                             "SEGMENT_COUNT": len(workbook_rows), "CYCLE_INDEX_MIN": min(reported, key=float) if reported else None,
                             "CYCLE_INDEX_MAX": max(reported, key=float) if reported else None,
                             "DUPLICATE_REPORTED_CYCLE_INDEX_SEGMENTS_WITHIN_WORKBOOK": len(reported) - len(set(reported))})
    return {"BATTERY_ID": battery, "WORKBOOK_COUNT": len(per_workbook), "WORKBOOK_SUMMARY": per_workbook,
            "CANONICAL_SOURCE_KEY_SCHEMA": "BATTERY_ID|SOURCE_WORKBOOK|SOURCE_SHEET|segment=<contiguous ordinal>|cycle=<reported Cycle_Index>",
            "SEGMENT_COUNT": len(rows)}, rows


def main():
    summaries, inventory = [], []
    for battery in BATTERIES:
        summary, rows = scan_battery(battery); summaries.append(summary); inventory.extend(rows)
        print(f"COMPLETED_BATTERY={battery}", flush=True)
    payload = {"scope": "CALCE_SOURCE_KEYED_CONTIGUOUS_CYCLE_BOUNDARY_AUDIT_V3", "battery_summaries": summaries,
               "cycle_boundary_inventory": inventory,
               "boundary_rule": "A segment is a maximal contiguous row run having the same raw reported Cycle_Index within one source workbook and sheet. The rule does not alter the raw Cycle_Index or infer a global lifecycle cycle.",
               "flags": {"TRAIN_SET_ASSIGNED": False, "VALIDATION_SET_ASSIGNED": False, "TEST_SET_ASSIGNED": False,
                         "PA001_UNCHANGED": True, "PA004_UNCHANGED": True}, "status": "COMPLETE"}
    (OUT / "calce_cycle_boundary_audit_v3.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
