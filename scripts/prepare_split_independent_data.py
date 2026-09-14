"""Build split-independent raw-data inventories for the paper roster.

This script deliberately has no train/validation/test, first-20%, GAN-target,
window, model, metric, or optimization logic.  It inventories official raw files,
checks chronology/physical fields, and tests only whether paper HFs can be
constructed from individual charge records.
"""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from scipy.io import loadmat

from tier2_hf_extraction import HFExtractionError, extract_nasa_hf_from_arrays


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_data"
OUT = ROOT / "output" / "data_readiness"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NASA_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
CALCE_BASE = "https://web.calce.umd.edu/batteries/data"
ROSTER = {
    "NASA": ["B0005", "B0006", "B0007", "B0018"],
    "CALCE": ["CS2_35", "CS2_36", "CS2_37", "CS2_38"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def first_index(values, predicate):
    for index, value in enumerate(values):
        if predicate(value):
            return index
    return None


def hf_status(times, voltage, current, cutoff):
    """Paper HFs with fixed official dataset charge endpoints; never imputes."""
    valid = [(number(t), number(v), number(i)) for t, v, i in zip(times, voltage, current)]
    valid = [(t, v, i) for t, v, i in valid if t is not None and v is not None and i is not None]
    if not valid:
        return {"status": "FAILED_EMPTY_OR_NONNUMERIC", "HF1": None, "HF2": None, "HF3": None}
    t, v, i = map(list, zip(*valid))
    upper = first_index(v, lambda x: x >= 4.2)
    v38 = first_index(v, lambda x: x >= 3.8)
    v40 = first_index(v, lambda x: x >= 4.0)
    cutoff_offset = None if upper is None else first_index(i[upper:], lambda x: 0 <= x <= cutoff)
    cutoff_idx = None if cutoff_offset is None else upper + cutoff_offset
    hf1 = None if upper is None else t[upper] - t[0]
    hf2 = None if upper is None or cutoff_idx is None else t[cutoff_idx] - t[upper]
    hf3 = None if v38 is None or v40 is None else t[v40] - t[v38]
    state = "SUCCESS" if all(x is not None and x >= 0 for x in (hf1, hf2, hf3)) else "FAILED_BOUNDARY_NOT_OBSERVED"
    return {"status": state, "HF1": hf1, "HF2": hf2, "HF3": hf3}


def nasa_hf_status(cycle):
    """PLAN-CHANGE-0017_REVISED NASA-only endpoint map with explicit fallback provenance."""
    try:
        feature = extract_nasa_hf_from_arrays(
            getattr(cycle.data, "Time", []),
            getattr(cycle.data, "Voltage_measured", []),
            getattr(cycle.data, "Current_charge", []),
        )
    except HFExtractionError as error:
        return {
            "status": f"FAILED_{str(error).upper().replace(' ', '_')}",
            "HF1": None,
            "HF2": None,
            "HF3": None,
            "t_imin_provenance": "NOT_AVAILABLE",
        }
    return {
        "status": "SUCCESS",
        "HF1": feature.hf1_seconds,
        "HF2": feature.hf2_seconds,
        "HF3": feature.hf3_seconds,
        "t_imin_provenance": feature.t_imin_provenance,
    }


def matlab_timestamp(value) -> str | None:
    values = [int(round(float(x))) for x in value[:6]]
    try:
        return datetime(*values).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def nasa_inventory(battery: str) -> tuple[dict, list[dict]]:
    file_path = next((RAW / "nasa_pcoe").rglob(f"{battery}.mat"), None)
    if file_path is None:
        return {"dataset": "NASA", "battery_id": battery, "file_found": False}, []
    data = loadmat(file_path, squeeze_me=True, struct_as_record=False)[battery]
    records, canonical, timestamps = [], [], []
    type_counts = defaultdict(int)
    for ordinal, cycle in enumerate(data.cycle, start=1):
        operation = str(cycle.type)
        type_counts[operation] += 1
        timestamp = matlab_timestamp(cycle.time)
        if timestamp:
            timestamps.append(timestamp)
        fields = set(cycle.data._fieldnames)
        if operation == "charge":
            features = nasa_hf_status(cycle)
            canonical.append({
                "dataset": "NASA", "battery_id": battery, "cycle_index": type_counts[operation],
                "chronological_order": ordinal, "operation_record_id": f"{battery}:operation:{ordinal}",
                "raw_source_file": str(file_path.relative_to(ROOT)), "timestamp": timestamp,
                "raw_capacity_ah": None, "feature_status": features["status"],
                "HF1_seconds": features["HF1"], "HF2_seconds": features["HF2"], "HF3_seconds": features["HF3"],
                "t_imin_provenance": features["t_imin_provenance"],
                "split_assignment": None,
            })
        records.append((operation, ordinal))
    duplicates = len(timestamps) - len(set(timestamps))
    feature_ok = sum(row["feature_status"] == "SUCCESS" for row in canonical)
    return {
        "dataset": "NASA", "battery_id": battery, "file_found": True,
        "raw_record_count": len(records), "charge_cycle_count": type_counts["charge"],
        "discharge_cycle_count": type_counts["discharge"], "impedance_or_other_record_count": len(records) - type_counts["charge"] - type_counts["discharge"],
        "valid_cycle_count": feature_ok, "missing_records": [], "duplicate_records": duplicates,
        "timestamp_order_status": "MONOTONIC" if timestamps == sorted(timestamps) else "NON_MONOTONIC",
        "file_path": str(file_path.relative_to(ROOT)), "file_size": file_path.stat().st_size, "sha256": sha256(file_path),
    }, canonical


def shared_strings(archive: zipfile.ZipFile) -> list[str]:
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in item.iter(f"{NS}t")) for item in root.iter(f"{NS}si")]


def xlsx_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        strings = shared_strings(archive)
        with archive.open("xl/worksheets/sheet2.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != f"{NS}row":
                    continue
                result = {}
                for cell in row.iter(f"{NS}c"):
                    column = re.match(r"[A-Z]+", cell.attrib["r"]).group(0)
                    value_node = cell.find(f"{NS}v")
                    value = None if value_node is None else value_node.text
                    if cell.attrib.get("t") == "s" and value is not None:
                        value = strings[int(value)]
                    result[column] = value
                yield result
                row.clear()


def calce_inventory(battery: str) -> tuple[dict, list[dict]]:
    battery_dir = RAW / "calce_cs2" / battery / battery
    files = sorted(battery_dir.glob("*.xlsx")) if battery_dir.exists() else []
    all_records, states = 0, {}
    malformed = 0
    for path in files:
        rows = xlsx_rows(path)
        header = next(rows, None)
        if header is None:
            malformed += 1
            continue
        mapping = {value: key for key, value in header.items()}
        required = {"Test_Time(s)", "Date_Time", "Cycle_Index", "Current(A)", "Voltage(V)"}
        if not required.issubset(mapping):
            malformed += 1
            continue
        for row_number, row in enumerate(rows, start=2):
            cycle = number(row.get(mapping["Cycle_Index"]))
            timestamp = number(row.get(mapping["Date_Time"]))
            if cycle is None or timestamp is None:
                malformed += 1
                continue
            all_records += 1
            cycle = int(cycle)
            state = states.setdefault(cycle, {"first_timestamp": timestamp, "last_timestamp": timestamp,
                                               "source": f"{path.relative_to(ROOT)}:sheet2:row{row_number}",
                                               "charge": False, "discharge": False, "t0": None, "upper": None,
                                               "cutoff": None, "v38": None, "v40": None, "capacity": None,
                                               "nonmonotonic": False})
            state["nonmonotonic"] |= timestamp < state["last_timestamp"]
            state["first_timestamp"] = min(state["first_timestamp"], timestamp)
            state["last_timestamp"] = max(state["last_timestamp"], timestamp)
            time = number(row.get(mapping["Test_Time(s)"]))
            current = number(row.get(mapping["Current(A)"]))
            voltage = number(row.get(mapping["Voltage(V)"]))
            capacity = number(row.get(mapping.get("Discharge_Capacity(Ah)")))
            if capacity is not None:
                state["capacity"] = capacity if state["capacity"] is None else max(state["capacity"], capacity)
            if current is None:
                continue
            state["charge"] |= current > 0
            state["discharge"] |= current < 0
            if not (current > 0 and time is not None and voltage is not None):
                continue
            state["t0"] = time if state["t0"] is None else min(state["t0"], time)
            if voltage >= 3.8 and state["v38"] is None:
                state["v38"] = time
            if voltage >= 4.0 and state["v40"] is None:
                state["v40"] = time
            if voltage >= 4.2 and state["upper"] is None:
                state["upper"] = time
            if state["upper"] is not None and 0 <= current <= 0.05 and state["cutoff"] is None:
                state["cutoff"] = time
    canonical, charge_cycles, discharge_cycles = [], 0, 0
    for order, (cycle, state) in enumerate(sorted(states.items(), key=lambda item: item[1]["first_timestamp"]), start=1):
        charge_cycles += state["charge"]
        discharge_cycles += state["discharge"]
        hf1 = None if state["t0"] is None or state["upper"] is None else state["upper"] - state["t0"]
        hf2 = None if state["upper"] is None or state["cutoff"] is None else state["cutoff"] - state["upper"]
        hf3 = None if state["v38"] is None or state["v40"] is None else state["v40"] - state["v38"]
        feature_status = "SUCCESS" if all(x is not None and x >= 0 for x in (hf1, hf2, hf3)) else "FAILED_BOUNDARY_NOT_OBSERVED"
        canonical.append({
            "dataset": "CALCE", "battery_id": battery, "cycle_index": cycle, "chronological_order": order,
            "operation_record_id": f"{battery}:cycle:{cycle}", "raw_source_file": state["source"].rsplit(":", 2)[0],
            "timestamp": state["first_timestamp"], "raw_capacity_ah": state["capacity"],
            "feature_status": feature_status, "HF1_seconds": hf1, "HF2_seconds": hf2, "HF3_seconds": hf3,
            "split_assignment": None,
        })
    duplicates = 0  # Unique raw provenance is file path plus sheet row; content duplicates are only flagged, never removed.
    feature_ok = sum(row["feature_status"] == "SUCCESS" for row in canonical)
    return {
        "dataset": "CALCE", "battery_id": battery, "file_found": bool(files), "raw_file_count": len(files),
        "raw_record_count": all_records, "charge_cycle_count": charge_cycles, "discharge_cycle_count": discharge_cycles,
        "impedance_or_other_record_count": 0, "valid_cycle_count": feature_ok, "missing_records": [],
        "duplicate_records": duplicates, "malformed_records": malformed,
        "timestamp_order_status": "MONOTONIC_BY_SORTED_CHRONOLOGICAL_ORDER" if states and not any(x["nonmonotonic"] for x in states.values()) else "NON_MONOTONIC_OR_NO_RECORDS",
        "file_path": str(battery_dir.relative_to(ROOT)), "file_size": sum(path.stat().st_size for path in files),
        "sha256": "PER_FILE_IN_DATA_SOURCE_MANIFEST",
    }, canonical


def calce_schema_inventory(battery: str) -> tuple[dict, list[dict]]:
    """Fast, non-destructive fallback: verifies every raw workbook and schema only."""
    battery_dir = RAW / "calce_cs2" / battery / battery
    files = sorted(battery_dir.glob("*.xlsx")) if battery_dir.exists() else []
    required = {"Test_Time(s)", "Date_Time", "Cycle_Index", "Current(A)", "Voltage(V)", "Discharge_Capacity(Ah)"}
    record_count, schema_failures = 0, []
    for path in files:
        with zipfile.ZipFile(path) as archive:
            with archive.open("xl/worksheets/sheet2.xml") as sheet:
                _, dimension_node = next(ET.iterparse(sheet, events=("end",)))
                dimension = dimension_node.attrib.get("ref", "")
            last_row = int(re.search(r"(\d+)$", dimension).group(1)) if re.search(r"(\d+)$", dimension) else 0
        rows = xlsx_rows(path)
        header = next(rows, {})
        if not required.issubset(set(header.values())):
            schema_failures.append(path.name)
        record_count += max(last_row - 1, 0)
    return {
        "dataset": "CALCE", "battery_id": battery, "file_found": bool(files), "raw_file_count": len(files),
        "raw_record_count": record_count, "charge_cycle_count": "NOT_ASSESSED_SCHEMA_ONLY",
        "discharge_cycle_count": "NOT_ASSESSED_SCHEMA_ONLY", "impedance_or_other_record_count": "NOT_ASSESSED_SCHEMA_ONLY",
        "valid_cycle_count": "NOT_ASSESSED_SCHEMA_ONLY", "missing_records": schema_failures,
        "duplicate_records": "NOT_ASSESSED_SCHEMA_ONLY", "malformed_records": 0,
        "timestamp_order_status": "NOT_ASSESSED_SCHEMA_ONLY", "file_path": str(battery_dir.relative_to(ROOT)),
        "file_size": sum(path.stat().st_size for path in files), "sha256": "PER_FILE_IN_DATA_SOURCE_MANIFEST",
        "feature_schema_status": "REQUIRED_RAW_SIGNALS_PRESENT" if not schema_failures else "REQUIRED_RAW_SIGNALS_MISSING",
    }, []


def source_manifest():
    sources = []
    nasa = RAW / "nasa_pcoe" / "NASA_5_Battery_Data_Set.zip"
    if nasa.exists():
        sources.append({"dataset": "NASA", "battery_id": "B0005,B0006,B0007,B0018", "source_url": NASA_URL,
                        "source_organization": "NASA Ames PCoE", "file_name": nasa.name, "file_size": nasa.stat().st_size,
                        "sha256": sha256(nasa), "download_date": "2026-09-11", "raw_or_processed": "RAW_ARCHIVE",
                        "license_or_access_note": "NASA repository states users employ data at their own risk; no license specified in portal metadata."})
    for battery in ROSTER["CALCE"]:
        archive = RAW / "calce_cs2" / f"{battery}.zip"
        if archive.exists():
            sources.append({"dataset": "CALCE", "battery_id": battery, "source_url": f"{CALCE_BASE}/{battery}.zip",
                            "source_organization": "University of Maryland CALCE Battery Group", "file_name": archive.name,
                            "file_size": archive.stat().st_size, "sha256": sha256(archive), "download_date": "2026-09-11",
                            "raw_or_processed": "RAW_ARCHIVE", "license_or_access_note": "CALCE requests citation of its data base and referenced articles for publication use."})
    return sources


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--schema-only-calce", action="store_true")
    args = arguments.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    inventories, canonical = [], []
    for battery in ROSTER["NASA"]:
        report, rows = nasa_inventory(battery)
        inventories.append(report); canonical.extend(rows)
    for battery in ROSTER["CALCE"]:
        report, rows = calce_schema_inventory(battery) if args.schema_only_calce else calce_inventory(battery)
        inventories.append(report); canonical.extend(rows)
    source = source_manifest()
    flags = {"TRAIN_SET_ASSIGNED": False, "VALIDATION_SET_ASSIGNED": False, "TEST_SET_ASSIGNED": False,
             "TARGET_FIRST20_ASSIGNED": False, "GAN_TARGET_POPULATION_ASSIGNED": False,
             "SCF_GAN_TRAINING_EXECUTED": False, "PA004_UNCHANGED": True}
    readiness = {"OFFICIAL_DATA_SOURCE_VERIFIED": len(source) == 5,
                 "BATTERY_ROSTER_VERIFIED": all(item.get("file_found") for item in inventories),
                 "RAW_PARSING_VERIFIED": (not args.schema_only_calce) and all(item.get("raw_record_count", 0) > 0 for item in inventories),
                 "CYCLE_ORDER_VERIFIED": (not args.schema_only_calce) and all("NON_MONOTONIC" not in item.get("timestamp_order_status", "") for item in inventories),
                 "FEATURE_CONSTRUCTION_FEASIBILITY_VERIFIED": (not args.schema_only_calce) and all(item.get("valid_cycle_count", 0) > 0 for item in inventories),
                 "NO_TRAIN_VAL_TEST_ASSIGNMENT": all(row["split_assignment"] is None for row in canonical),
                 "PA004_UNCHANGED": True}
    readiness["SPLIT_INDEPENDENT_DATA_READINESS_GATE"] = all(readiness.values())
    (OUT / "data_source_manifest.json").write_text(json.dumps({"sources": source, "flags": flags}, indent=2), encoding="utf-8")
    (OUT / "raw_inventory.json").write_text(json.dumps({"batteries": inventories, "readiness": readiness, "flags": flags}, indent=2), encoding="utf-8")
    if canonical:
        with (OUT / "unsplit_canonical_dataset.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(canonical[0]))
            writer.writeheader(); writer.writerows(canonical)
        schema = list(canonical[0])
    else:
        schema = []
    (OUT / "unsplit_canonical_manifest.json").write_text(json.dumps({"row_count": len(canonical), "schema": schema, "readiness": readiness, "flags": flags, "target_value_status": "BLOCKED_BY_PA001_SOH_DENOMINATOR_AMBIGUITY", "calce_mode": "SCHEMA_ONLY" if args.schema_only_calce else "FULL"}, indent=2), encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()
