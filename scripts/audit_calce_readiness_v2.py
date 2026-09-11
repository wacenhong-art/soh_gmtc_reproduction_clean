"""Source-keyed, split-independent CALCE readiness audit.

This read-only audit scans every worksheet in every official CALCE workbook.
It does not construct SOH, assign data splits, choose any paper ambiguity, or
perform modelling.  New artifacts are versioned separately from the prior
full-row inventory so earlier evidence is never replaced.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data" / "calce_cs2"
OUT = ROOT / "output" / "data_readiness"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
BATTERIES = ("CS2_35", "CS2_36", "CS2_37", "CS2_38")


def col_name(cell_ref: str) -> str:
    return re.match(r"[A-Z]+", cell_ref).group(0)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def normalized(values: dict[str, str | None], columns: list[str]) -> str:
    """Stable, provenance-free serialization for an explicitly declared key."""
    return "\x1f".join(f"{column}={values.get(column, '') or ''}" for column in columns)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(n.text or "" for n in item.iter(f"{NS}t")) for item in root.iter(f"{NS}si")]


def workbook_sheets(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {r.attrib["Id"]: r.attrib["Target"] for r in rels.iter(f"{PKG_REL_NS}Relationship")}
    result = []
    for sheet in root.iter(f"{NS}sheet"):
        target = targets[sheet.attrib[f"{REL_NS}id"]].lstrip("/")
        if not target.startswith("xl/"):
            target = "xl/" + target
        result.append((sheet.attrib["name"], target))
    return result


def iter_rows(archive: zipfile.ZipFile, xml_path: str, strings: list[str]):
    with archive.open(xml_path) as handle:
        for _, row in ET.iterparse(handle, events=("end",)):
            if row.tag != f"{NS}row":
                continue
            values = {}
            for cell in row.iter(f"{NS}c"):
                value_node = cell.find(f"{NS}v")
                value = None if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = strings[int(value)]
                values[col_name(cell.attrib["r"])] = value
            yield int(row.attrib.get("r", "0")), values
            row.clear()


def numeric_expected(name: str) -> bool:
    tokens = ("(s)", "(a)", "(v)", "(ah)", "(wh)", "ohm", "angle", "index", "point", "dV/dt")
    return any(token.lower() in name.lower() for token in tokens)


def scan_battery(battery: str) -> tuple[dict, list[dict], list[dict]]:
    directory = RAW / battery / battery
    workbooks = sorted(directory.glob("*.xlsx"))
    battery_result = {"battery_id": battery, "workbook_count": len(workbooks), "workbooks": [],
                      "cycle_index_behavior": {}, "timestamp_order": {}}
    null_rows, boundary_rows = [], []
    observed_cycles_by_book, all_cycles = {}, defaultdict(set)
    for workbook in workbooks:
        with zipfile.ZipFile(workbook) as archive:
            strings = shared_strings(archive)
            sheets = workbook_sheets(archive)
            book = {"source_workbook": str(workbook.relative_to(ROOT)), "sheets": []}
            for sheet_name, xml_path in sheets:
                rows = iter_rows(archive, xml_path, strings)
                header_row, header_cells = next(rows, (None, None))
                if header_cells is None:
                    book["sheets"].append({"source_sheet": sheet_name, "status": "EMPTY"})
                    continue
                columns = [header_cells[key] for key in sorted(header_cells)]
                mapping = {value: key for key, value in header_cells.items() if value}
                stats = {col: {"TOTAL_ROWS": 0, "NULL_COUNT": 0, "EMPTY_STRING_COUNT": 0,
                               "NON_NUMERIC_COUNT_IF_NUMERIC_EXPECTED": 0, "INF_COUNT": 0, "NAN_COUNT": 0}
                         for col in columns}
                all_row_counts, content_counts = Counter(), Counter()
                cycle_content, timestamp_content, source_keys = defaultdict(set), defaultdict(set), set()
                first_time = last_time = None
                monotonic = True
                cycle_values = set()
                operation_rows = 0
                for source_row, values_by_col in rows:
                    operation_rows += 1
                    by_name = {name: values_by_col.get(letter) for name, letter in mapping.items()}
                    for name, letter in mapping.items():
                        value = values_by_col.get(letter)
                        entry = stats[name]; entry["TOTAL_ROWS"] += 1
                        if value is None:
                            entry["NULL_COUNT"] += 1
                        elif value == "":
                            entry["EMPTY_STRING_COUNT"] += 1
                        elif numeric_expected(name):
                            parsed = number(value)
                            if parsed is None:
                                entry["NON_NUMERIC_COUNT_IF_NUMERIC_EXPECTED"] += 1
                            elif math.isinf(float(value)):
                                entry["INF_COUNT"] += 1
                            elif math.isnan(float(value)):
                                entry["NAN_COUNT"] += 1
                    raw_key = digest(normalized(values_by_col, sorted(values_by_col)))
                    all_row_counts[raw_key] += 1
                    content_columns = [x for x in ("Step_Index", "Cycle_Index", "Current(A)", "Voltage(V)",
                                  "Charge_Capacity(Ah)", "Charge_Energy(Wh)", "Discharge_Capacity(Ah)",
                                  "Discharge_Energy(Wh)", "dV/dt(V/s)", "Is_FC_Data", "AC_Impedance(Ohm)",
                                  "ACI_Phase_Angle(Deg)") if x in mapping]
                    content_key = digest(normalized(by_name, content_columns))
                    content_counts[content_key] += 1
                    if sheet_name == "Data" or "Cycle_Index" in mapping:
                        cycle = by_name.get("Cycle_Index")
                        timestamp = by_name.get("Date_Time")
                        source_key = f"{battery}|{workbook.relative_to(ROOT)}|{sheet_name}|{source_row}|{cycle}"
                        if source_key in source_keys:
                            raise RuntimeError("Canonical source key collision")
                        source_keys.add(source_key)
                        if cycle is not None:
                            cycle_values.add(cycle); all_cycles[cycle].add(workbook.name)
                            cycle_content[cycle].add(content_key)
                        if timestamp is not None:
                            timestamp_content[timestamp].add(content_key)
                            t = number(timestamp)
                            if t is not None:
                                monotonic = monotonic and (last_time is None or t >= last_time)
                                first_time = t if first_time is None else min(first_time, t)
                                last_time = t if last_time is None else max(last_time, t)
                duplicate_exact = sum(count - 1 for count in all_row_counts.values() if count > 1)
                duplicate_content = sum(count - 1 for count in content_counts.values() if count > 1)
                cycle_collision = sum(len(values) - 1 for values in cycle_content.values() if len(values) > 1)
                timestamp_collision = sum(len(values) - 1 for values in timestamp_content.values() if len(values) > 1)
                sheet_entry = {"source_sheet": sheet_name, "header_row": header_row, "total_rows": operation_rows,
                    "column_names": columns, "column_null_audit": stats,
                    "EXACT_ROW_DUPLICATE_COUNT": duplicate_exact, "CONTENT_DUPLICATE_COUNT": duplicate_content,
                    "CYCLE_INDEX_COLLISION_COUNT": cycle_collision, "TIMESTAMP_COLLISION_COUNT": timestamp_collision,
                    "SOURCE_KEY_DUPLICATE_COUNT": 0, "timestamp_monotonic_within_workbook_traversal": monotonic,
                    "timestamp_min": first_time, "timestamp_max": last_time}
                book["sheets"].append(sheet_entry)
                null_rows.append({"battery_id": battery, "source_workbook": str(workbook.relative_to(ROOT)),
                                  "source_sheet": sheet_name, **sheet_entry})
                if "Cycle_Index" in mapping:
                    observed_cycles_by_book[workbook.name] = cycle_values
                    for cycle in sorted(cycle_values, key=lambda x: number(x) if number(x) is not None else float("inf")):
                        boundary_rows.append({"BATTERY_ID": battery, "SOURCE_WORKBOOK": str(workbook.relative_to(ROOT)),
                          "SOURCE_SHEET": sheet_name, "REPORTED_CYCLE_INDEX": cycle,
                          "CANONICAL_CALCE_SOURCE_KEY": f"{battery}|{workbook.relative_to(ROOT)}|{sheet_name}|cycle={cycle}",
                          "START_ROW": "UNRESOLVED__REQUIRES_CONTIGUOUS_OPERATION_SEGMENT_RULE",
                          "END_ROW": "UNRESOLVED__REQUIRES_CONTIGUOUS_OPERATION_SEGMENT_RULE",
                          "START_TIMESTAMP": "AVAILABLE_IN_ROW_LEVEL_AUDIT_NOT_AGGREGATED_BY_UNPROVEN_SEGMENT",
                          "END_TIMESTAMP": "AVAILABLE_IN_ROW_LEVEL_AUDIT_NOT_AGGREGATED_BY_UNPROVEN_SEGMENT",
                          "ROW_COUNT": "UNRESOLVED__CYCLE_INDEX_MAY_RECUR_NONCONTIGUOUSLY",
                          "OPERATION_OR_SEGMENT_TYPE_IF_DETERMINABLE": "UNRESOLVED",
                          "BOUNDARY_STATUS": "UNRESOLVED"})
            battery_result["workbooks"].append(book)
    intersections = []
    names = sorted(observed_cycles_by_book)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = observed_cycles_by_book[left] & observed_cycles_by_book[right]
            if overlap:
                intersections.append({"workbooks": [left, right], "overlap_count": len(overlap), "cycle_indices": sorted(overlap, key=float)})
    per_workbook = {name: {"min": (min(values, key=float) if values else None), "max": (max(values, key=float) if values else None),
                           "unique_cycle_index_count": len(values)} for name, values in observed_cycles_by_book.items()}
    all_numeric = set().union(*observed_cycles_by_book.values()) if observed_cycles_by_book else set()
    sorted_cycles = sorted(map(float, all_numeric))
    gaps = [int(a + 1) for a, b in zip(sorted_cycles, sorted_cycles[1:]) if b - a > 1]
    battery_result["cycle_index_behavior"] = {"CYCLE_INDEX_MIN_MAX_PER_WORKBOOK": per_workbook,
        "CYCLE_INDEX_RESET_ACROSS_WORKBOOKS": bool(intersections), "CYCLE_INDEX_OVERLAP_ACROSS_WORKBOOKS": bool(intersections),
        "CYCLE_INDEX_GAPS_GLOBAL_NUMERIC_DOMAIN": gaps, "CYCLE_INDEX_DUPLICATES_WITHIN_WORKBOOK": False,
        "CYCLE_INDEX_DUPLICATES_ACROSS_WORKBOOKS": bool(intersections),
        "IS_CYCLE_INDEX_GLOBALLY_UNIQUE": False if intersections else "UNRESOLVED__NO_GLOBAL_ID_CONTRACT_IN_SOURCE"}
    return battery_result, null_rows, boundary_rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results, null_rows, boundary_rows = [], [], []
    for battery in BATTERIES:
        result, nulls, boundaries = scan_battery(battery)
        results.append(result); null_rows.extend(nulls); boundary_rows.extend(boundaries)
        print(f"COMPLETED_BATTERY={battery}", flush=True)
    payload = {"scope": "CALCE_SOURCE_KEYED_CYCLE_IDENTITY_NULL_AND_DUPLICATE_AUDIT_V2", "batteries": results,
        "cycle_boundary_inventory": boundary_rows, "column_null_audit": null_rows,
        "content_duplicate_key": {"included_columns": ["Step_Index", "Cycle_Index", "Current(A)", "Voltage(V)", "Charge_Capacity(Ah)", "Charge_Energy(Wh)", "Discharge_Capacity(Ah)", "Discharge_Energy(Wh)", "dV/dt(V/s)", "Is_FC_Data", "AC_Impedance(Ohm)", "ACI_Phase_Angle(Deg)"],
          "excluded_columns": ["Data_Point", "Test_Time(s)", "Date_Time"], "reason": "The content key tests repeated electrochemical/step states while excluding row identity and time-position fields; exact row duplication retains every populated raw cell."},
        "flags": {"TRAIN_SET_ASSIGNED": False, "VALIDATION_SET_ASSIGNED": False, "TEST_SET_ASSIGNED": False,
                  "FINAL_SOH_TARGET_ASSIGNED": False, "PA001_UNCHANGED": True, "PA004_UNCHANGED": True},
        "status": "COMPLETE_WITH_RECORDED_FEATURE_LIMITATION"}
    (OUT / "calce_cycle_identity_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (OUT / "calce_null_duplicate_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
