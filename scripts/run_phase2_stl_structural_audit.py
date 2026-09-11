"""Phase 2B train-only structural period discovery and STL audit.

The script reads protocol membership metadata, then converts only TRAIN /
TRAIN_REAL raw capacity values into in-memory arrays.  Validation and test
capacity values are neither selected, converted, retained, nor analysed.
No model, metric, grid search, synthetic data, or component file is produced.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import scipy
import statsmodels
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = ROOT / "output" / "protocol" / "phase1_membership_manifest.csv"
OUT = ROOT / "output" / "stl"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _train_role(protocol: str) -> str:
    return "TRAIN" if protocol == "A" else "TRAIN_REAL"


def _membership() -> tuple[dict[tuple[str, str, str], list[dict[str, str]]], dict[tuple[str, str, str], int]]:
    selected: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    validation_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    with MEMBERSHIP.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["active_protocol"], row["dataset"], row["battery_id"])
            if row["role"] == _train_role(row["active_protocol"]):
                selected[key].append(row)
            elif row["role"] == "VALIDATION":
                validation_counts[key] += 1
    for rows in selected.values():
        rows.sort(key=lambda row: int(row["chronological_index"]))
    return selected, validation_counts


def _nasa_train_capacity(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict]:
    """Read only requested early discharge capacities by matched cycle ordinal."""
    battery = rows[0]["battery_id"]
    requested = [int(row["reported_cycle_index"]) for row in rows]
    if requested != list(range(1, len(requested) + 1)):
        raise ValueError(f"{battery}: selected charge cycle ordinals are not a contiguous prefix")
    source = ROOT / Path(rows[0]["raw_source_provenance"].replace("\\", "/"))
    data = loadmat(source, squeeze_me=True, struct_as_record=False)[battery]
    values: dict[int, float] = {}
    discharge_ordinal = 0
    for cycle in data.cycle:
        if str(cycle.type) != "discharge":
            continue
        discharge_ordinal += 1
        if discharge_ordinal not in set(requested):
            continue
        capacity = float(getattr(cycle.data, "Capacity"))
        if not math.isfinite(capacity):
            raise ValueError(f"{battery}: non-finite requested discharge capacity")
        values[discharge_ordinal] = capacity
        if len(values) == len(requested):
            break
    if list(values) != requested:
        raise ValueError(f"{battery}: requested charge/discharge ordinal alignment is incomplete")
    return np.asarray([values[index] for index in requested], dtype=float), {
        "capacity_source": "NASA_DISCHARGE_RECORD_CAPACITY",
        "alignment_rule": "match the early charge membership reported_cycle_index to the same ordinal discharge record; only requested prefix capacity values are converted",
        "raw_source": str(source.relative_to(ROOT)).replace("\\", "/"),
        "test_capacity_values_selected": False,
    }


def _col_name(cell_ref: str) -> str:
    return "".join(ch for ch in cell_ref if ch.isalpha())


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(n.text or "" for n in item.iter(f"{NS}t")) for item in root.iter(f"{NS}si")]


def _sheet_xml_path(archive: zipfile.ZipFile, requested_sheet: str) -> str:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {node.attrib["Id"]: node.attrib["Target"] for node in relationships.iter(f"{PKG_REL_NS}Relationship")}
    for sheet in workbook.iter(f"{NS}sheet"):
        if sheet.attrib["name"] == requested_sheet:
            target = targets[sheet.attrib[f"{REL_NS}id"]].lstrip("/")
            return target if target.startswith("xl/") else "xl/" + target
    raise KeyError(f"sheet not found: {requested_sheet}")


def _iter_rows(archive: zipfile.ZipFile, xml_path: str, strings: list[str]):
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
                values[_col_name(cell.attrib["r"])] = value
            yield int(row.attrib.get("r", "0")), values
            row.clear()


def _calce_train_capacity(rows: list[dict[str, str]]) -> tuple[np.ndarray, dict]:
    """Read Discharge_Capacity(Ah) only from selected source-keyed train segments."""
    by_sheet: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_sheet[(row["raw_source_provenance"], row["source_sheet"])].append(row)
    values_by_uid: dict[str, float] = {}
    for (raw_source, sheet), group in by_sheet.items():
        source = ROOT / Path(raw_source.replace("\\", "/"))
        ranges = {row["sample_uid"]: (int(row["source_row_start"]), int(row["source_row_end"])) for row in group}
        max_needed_row = max(end for _, end in ranges.values())
        with zipfile.ZipFile(source) as archive:
            strings = _shared_strings(archive)
            iterator = _iter_rows(archive, _sheet_xml_path(archive, sheet), strings)
            _, header = next(iterator)
            if header is None:
                raise ValueError(f"empty CALCE sheet: {raw_source}::{sheet}")
            by_name = {value: key for key, value in header.items() if value}
            column = by_name.get("Discharge_Capacity(Ah)")
            if column is None:
                raise ValueError(f"CALCE discharge-capacity column missing: {raw_source}::{sheet}")
            for source_row, fields in iterator:
                if source_row > max_needed_row:
                    break
                for uid, (start, end) in ranges.items():
                    if start <= source_row <= end:
                        raw = fields.get(column)
                        if raw in (None, ""):
                            continue
                        value = float(raw)
                        if not math.isfinite(value):
                            raise ValueError(f"CALCE non-finite selected capacity: {uid}")
                        values_by_uid[uid] = max(value, values_by_uid.get(uid, value))
    missing = [row["sample_uid"] for row in rows if row["sample_uid"] not in values_by_uid]
    if missing:
        raise ValueError(f"CALCE selected source segments without finite discharge capacity: {len(missing)}")
    return np.asarray([values_by_uid[row["sample_uid"]] for row in rows], dtype=float), {
        "capacity_source": "CALCE_SELECTED_SEGMENT_MAX_DISCHARGE_CAPACITY_AH",
        "alignment_rule": "maximum finite Discharge_Capacity(Ah) within each selected canonical source-keyed segment",
        "test_capacity_values_selected": False,
    }


def _linear_detrend(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x, values, deg=1)
    return values - (slope * x + intercept)


def _acf_candidate(detrended: np.ndarray, max_period: int) -> tuple[int | None, dict]:
    denominator = float(np.dot(detrended, detrended))
    if denominator <= np.finfo(float).eps:
        return None, {"reason": "ZERO_VARIANCE_AFTER_LINEAR_DETREND"}
    lags = range(2, max_period + 1)
    scores = {lag: float(np.dot(detrended[:-lag], detrended[lag:]) / denominator) for lag in lags}
    candidate = max(scores, key=scores.get)
    return candidate, {
        "candidate": candidate,
        "score": scores[candidate],
        "eligible_lag_count": len(scores),
        "runner_up_score": sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else None,
    }


def _spectral_candidate(detrended: np.ndarray, max_period: int) -> tuple[int | None, dict]:
    n = detrended.size
    spectrum = np.abs(np.fft.rfft(detrended)) ** 2
    options = {}
    for frequency_bin in range(1, spectrum.size):
        period = int(round(n / frequency_bin))
        if 2 <= period <= max_period:
            options[frequency_bin] = {"period": period, "power": float(spectrum[frequency_bin])}
    if not options:
        return None, {"reason": "NO_FFT_BIN_MAPS_TO_EXECUTABLE_PERIOD"}
    selected_bin = max(options, key=lambda item: options[item]["power"])
    selected = options[selected_bin]
    return int(selected["period"]), {
        "candidate": int(selected["period"]),
        "frequency_bin": selected_bin,
        "power": selected["power"],
        "eligible_frequency_bin_count": len(options),
        "runner_up_power": sorted((entry["power"] for entry in options.values()), reverse=True)[1] if len(options) > 1 else None,
    }


def _estimate_period(values: np.ndarray, validation_n: int) -> dict:
    max_period = min(values.size // 2, validation_n // 2)
    if max_period < 2:
        return {"status": "UNRESOLVED", "reason": "NO_PERIOD_EXECUTABLE_IN_BOTH_TRAIN_AND_VALIDATION", "max_period": int(max_period)}
    detrended = _linear_detrend(values)
    acf, acf_evidence = _acf_candidate(detrended, max_period)
    spectral, spectral_evidence = _spectral_candidate(detrended, max_period)
    if acf is None or spectral is None:
        return {"status": "UNRESOLVED", "reason": "ACF_OR_SPECTRAL_CANDIDATE_MISSING", "max_period": int(max_period), "acf": acf_evidence, "spectral": spectral_evidence}
    difference = abs(acf - spectral)
    if difference > 1:
        return {"status": "UNRESOLVED", "reason": "ACF_SPECTRAL_DISAGREEMENT_EXCEEDS_ONE_CYCLE", "max_period": int(max_period), "acf_period_candidate": acf, "spectral_period_candidate": spectral, "agreement_difference": difference, "acf": acf_evidence, "spectral": spectral_evidence}
    selected = int(round((acf + spectral) / 2))
    return {"status": "RESOLVED", "max_period": int(max_period), "acf_period_candidate": acf, "spectral_period_candidate": spectral, "agreement_difference": difference, "selected_period": selected, "selection_source": "ACF_SPECTRAL_AGREEMENT__ABSOLUTE_DIFFERENCE_LE_1", "acf": acf_evidence, "spectral": spectral_evidence}


def _stability(values: np.ndarray, validation_n: int) -> dict:
    results = {}
    selected = []
    for fraction in (0.8, 0.9, 1.0):
        prefix_n = math.floor(fraction * values.size)
        estimate = _estimate_period(values[:prefix_n], validation_n)
        results[str(fraction)] = {"prefix_n": prefix_n, "estimate": estimate}
        if estimate["status"] == "RESOLVED":
            selected.append(estimate["selected_period"])
    if len(selected) != 3:
        return {"status": "UNSTABLE_OR_UNRESOLVED", "reason": "AT_LEAST_ONE_PREFIX_HAS_NO_AGREED_PERIOD", "prefixes": results}
    spread = max(selected) - min(selected)
    return {"status": "STABLE" if spread <= 1 else "UNSTABLE", "selected_period_spread": spread, "prefixes": results}


def main() -> None:
    selected, validation_counts = _membership()
    period_rows = []
    all_ready = True
    for (protocol, dataset, battery), rows in sorted(selected.items()):
        values, capacity_provenance = (_nasa_train_capacity(rows) if dataset == "NASA" else _calce_train_capacity(rows))
        estimate = _estimate_period(values, validation_counts[(protocol, dataset, battery)])
        stability = _stability(values, validation_counts[(protocol, dataset, battery)])
        ready = estimate["status"] == "RESOLVED" and stability["status"] == "STABLE"
        all_ready = all_ready and ready
        period_rows.append({
            "protocol": protocol, "dataset": dataset, "battery_id": battery,
            "n_train": int(values.size), "n_validation_metadata_only": validation_counts[(protocol, dataset, battery)],
            "capacity_provenance": capacity_provenance, "period_estimate": estimate, "period_stability": stability,
            "period_frozen": ready,
        })
    payload = {
        "scope": "PHASE_2B_TRAIN_ONLY_STRUCTURAL_PERIOD_DISCOVERY",
        "result_blind": True,
        "performance_blind": True,
        "period_discovery_detrend_method": "ORDINARY_LEAST_SQUARES_LINEAR_DETREND__NO_STL_USED_FOR_DETRENDING",
        "period_selection_uses_validation_values": False,
        "period_selection_uses_test_values": False,
        "period_selection_uses_model_metrics": False,
        "validation_counts_metadata_only": True,
        "test_loaded": False,
        "test_capacity_values_accessed": False,
        "period_rows": period_rows,
        "all_target_periods_frozen": all_ready,
        "statsmodels_version": statsmodels.__version__,
        "scipy_version": scipy.__version__,
        "numpy_version": np.__version__,
        "membership_manifest_sha256": _sha256(MEMBERSHIP),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stl_period_discovery.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "scope": "PHASE_2B_TRAIN_STL_STRUCTURAL_AUDIT",
        "train_stl_executed": False,
        "all_target_periods_frozen": all_ready,
        "structural_rows": [],
        "validation_stl_executability": "NOT_EXECUTED__PARAMETERS_ARE_CHECKED_ONLY_BY_BOUND",
        "test_stl_executed": False,
        "test_loaded": False,
    }
    (OUT / "stl_structural_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_target_periods_frozen": all_ready, "period_rows": period_rows, "structural_rows": []}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
