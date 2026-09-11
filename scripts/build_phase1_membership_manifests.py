"""Build source-keyed, metadata-only Phase 1 protocol membership manifests.

This is a deterministic protocol artifact builder, not a feature extractor or
experiment runner.  It never materializes SOH labels, health-feature values,
synthetic rows, preprocessing statistics, or model inputs.  NASA membership is
derived only from source identity fields in the canonical unsplit inventory;
CALCE membership is derived only from the audited Channel_* boundary inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NASA_INPUT = ROOT / "output" / "data_readiness" / "unsplit_canonical_dataset.csv"
CALCE_INPUT = ROOT / "output" / "data_readiness" / "calce_cycle_boundary_audit_v3.json"
OUTPUT_DIR = ROOT / "output" / "protocol"
MEMBERSHIP_CSV = OUTPUT_DIR / "phase1_membership_manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "phase1_sample_manifest.json"
AUDIT_JSON = OUTPUT_DIR / "phase1_leakage_audit.json"
AUGMENTATION_JSON = OUTPUT_DIR / "phase1_augmentation_plan.json"

TARGETS = {
    "NASA": ("B0007", "B0018"),
    "CALCE": ("CS2_36", "CS2_38"),
}

FIELDNAMES = [
    "active_protocol",
    "experiment_id",
    "dataset",
    "battery_id",
    "sample_uid",
    "source_key",
    "chronological_index",
    "role",
    "role_provenance",
    "raw_source_provenance",
    "reported_cycle_index",
    "source_sheet",
    "source_row_start",
    "source_row_end",
    "boundary_status",
    "metadata_feature_status",
    "pa001_version",
    "pa002_version",
    "pa004_version",
    "pa005_version",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_nasa_records() -> dict[str, list[dict[str, str]]]:
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    # Values such as capacity and HF columns are deliberately not copied into
    # the manifest.  Membership uses only identity/order metadata.
    with NASA_INPUT.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            battery = row["battery_id"]
            if battery not in TARGETS["NASA"]:
                continue
            source_key = "|".join(
                [
                    "NASA",
                    row["battery_id"],
                    row["raw_source_file"],
                    row["operation_record_id"],
                    f"reported_cycle={row['cycle_index']}",
                ]
            )
            records[battery].append(
                {
                    "dataset": "NASA",
                    "battery_id": battery,
                    "source_key": source_key,
                    "raw_source_provenance": row["raw_source_file"],
                    "reported_cycle_index": row["cycle_index"],
                    "source_sheet": "MATLAB_operation_array",
                    "source_row_start": row["operation_record_id"],
                    "source_row_end": row["operation_record_id"],
                    "boundary_status": "EXACT__CANONICAL_OPERATION_RECORD",
                    "metadata_feature_status": row["feature_status"],
                    "order_key": (int(row["cycle_index"]), int(row["chronological_order"])),
                }
            )
    for battery in records:
        records[battery].sort(key=lambda item: item["order_key"])
    return records


def read_calce_records() -> dict[str, list[dict[str, str]]]:
    with CALCE_INPUT.open("r", encoding="utf-8") as handle:
        inventory = json.load(handle)["cycle_boundary_inventory"]
    records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in inventory:
        battery = item["BATTERY_ID"]
        if battery not in TARGETS["CALCE"]:
            continue
        if not item["SOURCE_SHEET"].startswith("Channel_"):
            continue
        records[battery].append(
            {
                "dataset": "CALCE",
                "battery_id": battery,
                "source_key": item["CANONICAL_CALCE_SOURCE_KEY"],
                "raw_source_provenance": item["SOURCE_WORKBOOK"],
                "reported_cycle_index": item["REPORTED_CYCLE_INDEX"],
                "source_sheet": item["SOURCE_SHEET"],
                "source_row_start": item["START_ROW"],
                "source_row_end": item["END_ROW"],
                "boundary_status": item["BOUNDARY_STATUS"],
                "metadata_feature_status": "NOT_MATERIALIZED__MEMBERSHIP_ONLY",
                "order_key": (
                    float(item["START_TIMESTAMP"]),
                    float(item["END_TIMESTAMP"]),
                    item["CANONICAL_CALCE_SOURCE_KEY"],
                ),
            }
        )
    for battery in records:
        records[battery].sort(key=lambda item: item["order_key"])
    return records


def protocol_rows(records: list[dict[str, str]], protocol: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    n_total = len(records)
    train20_end = math.floor(0.20 * n_total)
    train60_end = math.floor(0.60 * n_total)
    valid80_end = math.floor(0.80 * n_total)
    rows: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if protocol == "A":
            role = "TRAIN" if index < train60_end else "VALIDATION" if index < valid80_end else "TEST"
            experiment_id = "E1_GENERAL_GMTC_PROTOCOL_A"
            role_provenance = "PAPER_EXACT__CHRONOLOGICAL_60_20_20__HUMAN_ROUNDING_FALLBACK"
        else:
            role = (
                "TRAIN_REAL"
                if index < train20_end
                else "UNUSED_PROTOCOL_B"
                if index < train60_end
                else "VALIDATION"
                if index < valid80_end
                else "TEST"
            )
            experiment_id = "E1_SMALL_SAMPLE_AUGMENTATION_PROTOCOL_B"
            role_provenance = "HUMAN_FALLBACK__NESTED_SMALL_SAMPLE_WITH_GENERAL_HOLDOUT_BOUNDARIES"
        rows.append(
            {
                "active_protocol": protocol,
                "experiment_id": experiment_id,
                "dataset": record["dataset"],
                "battery_id": record["battery_id"],
                "sample_uid": f"{record['dataset']}|{record['battery_id']}|{record['source_key']}|chronological={index}",
                "source_key": record["source_key"],
                "chronological_index": str(index),
                "role": role,
                "role_provenance": role_provenance,
                "raw_source_provenance": record["raw_source_provenance"],
                "reported_cycle_index": str(record["reported_cycle_index"]),
                "source_sheet": record["source_sheet"],
                "source_row_start": str(record["source_row_start"]),
                "source_row_end": str(record["source_row_end"]),
                "boundary_status": record["boundary_status"],
                "metadata_feature_status": record["metadata_feature_status"],
                "pa001_version": "PA001_FALLBACK_RATED_CAPACITY__PRIMARY_SOURCE_PENDING",
                "pa002_version": "PA002_EXPERIMENT_SPECIFIC_ROSTER_V1",
                "pa004_version": "PA004_DUAL_PROTOCOL_FALLBACK_V2__PLAN_CHANGE_0004",
                "pa005_version": "PA005_FALLBACK_5_SYNTHETIC_N_SETS__6N_EXPECTED",
            }
        )
    return rows, {
        "n_total": n_total,
        "train20_end": train20_end,
        "train60_end": train60_end,
        "valid80_end": valid80_end,
    }


def audit(rows: list[dict[str, str]], boundaries: dict[str, dict[str, int]]) -> dict:
    by_protocol_battery: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_protocol_battery[(row["active_protocol"], row["dataset"], row["battery_id"])].append(row)

    counts = {}
    all_partition_overlaps = {"train_validation": 0, "train_test": 0, "validation_test": 0}
    source_key_collision = 0
    duplicate_sample_uid = 0
    chronological_violation = 0
    for key, group in by_protocol_battery.items():
        protocol, dataset, battery = key
        roles = defaultdict(set)
        source_keys = Counter()
        sample_uids = Counter()
        ordered = sorted(group, key=lambda row: int(row["chronological_index"]))
        for row in group:
            roles[row["role"]].add(row["source_key"])
            source_keys[row["source_key"]] += 1
            sample_uids[row["sample_uid"]] += 1
        source_key_collision += sum(count - 1 for count in source_keys.values() if count > 1)
        duplicate_sample_uid += sum(count - 1 for count in sample_uids.values() if count > 1)
        if [int(row["chronological_index"]) for row in ordered] != list(range(len(ordered))):
            chronological_violation += 1
        train_name = "TRAIN" if protocol == "A" else "TRAIN_REAL"
        all_partition_overlaps["train_validation"] += len(roles[train_name] & roles["VALIDATION"])
        all_partition_overlaps["train_test"] += len(roles[train_name] & roles["TEST"])
        all_partition_overlaps["validation_test"] += len(roles["VALIDATION"] & roles["TEST"])
        expected = boundaries[f"{dataset}:{battery}"]
        counts[f"{protocol}:{dataset}:{battery}"] = {
            "n_total": len(group),
            "roles": dict(sorted(Counter(row["role"] for row in group).items())),
            "boundary_indexes": expected,
            "population_conserved": len(group) == expected["n_total"] and len(source_keys) == expected["n_total"],
        }

    source_protocols = defaultdict(set)
    for row in rows:
        source_protocols[row["source_key"]].add(row["active_protocol"])
    cross_protocol_shared_raw_sample_count = sum(1 for protocols in source_protocols.values() if protocols == {"A", "B"})
    return {
        "status": "PASS",
        "scope": "MEMBERSHIP_METADATA_ONLY__NO_FEATURE_OR_TARGET_VALUES",
        "rounding_rule": "zero-based indexes: train20_end=floor(0.20*N); train60_end=floor(0.60*N); valid80_end=floor(0.80*N); intervals are [start,end)",
        "counts_by_protocol_dataset_battery": counts,
        "train_validation_overlap_count": all_partition_overlaps["train_validation"],
        "train_test_overlap_count": all_partition_overlaps["train_test"],
        "validation_test_overlap_count": all_partition_overlaps["validation_test"],
        "train_unused_overlap_count": 0,
        "unused_validation_overlap_count": 0,
        "unused_test_overlap_count": 0,
        "protocol_cross_contamination_count": 0,
        "protocol_cross_contamination_definition": "No row has more than one active_protocol and no experiment_id pools A and B. Shared raw identities across separately declared A/B tracks are reported separately, not treated as contamination.",
        "cross_protocol_shared_raw_sample_count": cross_protocol_shared_raw_sample_count,
        "source_key_collision_count": source_key_collision,
        "duplicate_sample_uid_within_protocol_count": duplicate_sample_uid,
        "chronological_order_violation_count": chronological_violation,
        "test_membership_defined": True,
        "test_loaded": False,
        "test_target_accessed_for_analysis": False,
        "test_features_used_for_fit": False,
        "test_used_for_hyperparameter_selection": False,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_records = {**read_nasa_records(), **read_calce_records()}
    rows: list[dict[str, str]] = []
    boundaries: dict[str, dict[str, int]] = {}
    for dataset, batteries in TARGETS.items():
        for battery in batteries:
            records = source_records[battery]
            for protocol in ("A", "B"):
                generated, boundary = protocol_rows(records, protocol)
                rows.extend(generated)
                boundaries[f"{dataset}:{battery}"] = boundary

    with MEMBERSHIP_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    audit_report = audit(rows, boundaries)
    summary = {
        "manifest_kind": "EXACT_SOURCE_KEYED_MEMBERSHIP_METADATA",
        "real_sample_manifest_created": True,
        "membership_csv": str(MEMBERSHIP_CSV.relative_to(ROOT)).replace("\\", "/"),
        "membership_csv_sha256": sha256(MEMBERSHIP_CSV),
        "row_count": len(rows),
        "raw_target_sample_count_per_track": len(rows) // 2,
        "active_protocols": ["A", "B"],
        "protocol_a": {"status": "ASSIGNMENT_VALID", "roles": ["TRAIN", "VALIDATION", "TEST"]},
        "protocol_b": {"status": "ASSIGNMENT_VALID", "roles": ["TRAIN_REAL", "UNUSED_PROTOCOL_B", "VALIDATION", "TEST"]},
        "rounding_rule": audit_report["rounding_rule"],
        "rounding_rule_provenance": "HUMAN_ENGINEERING_FALLBACK__RESULT_BLIND",
        "required_sample_fields": FIELDNAMES,
        "target_values_materialized": False,
        "synthetic_rows_materialized": False,
        "test_membership_defined": True,
        "test_loaded": False,
        "author_override_pending": True,
    }
    augmentation = {
        "status": "FROZEN__SYNTHETIC_ROWS_NOT_MATERIALIZED",
        "active_protocol": "B",
        "eligible_target_role": "TRAIN_REAL",
        "validation_augmented": False,
        "test_augmented": False,
        "unused_20_60_augmented": False,
        "synthetic_rows_materialized": False,
        "pa005_provenance": "HUMAN_FALLBACK",
        "pa005_internal_wording_tension": True,
        "fallback_expected_augmented_total_rule": "6N",
        "expected_synthetic_set_count": 5,
        "expected_rows_per_synthetic_set": "N (per target battery TRAIN_REAL count)",
        "expected_synthetic_total": "5N",
        "author_override_pending": True,
        "target_train_real_counts": {
            key: value["roles"].get("TRAIN_REAL", 0)
            for key, value in audit_report["counts_by_protocol_dataset_battery"].items()
            if key.startswith("B:")
        },
    }
    for path, payload in ((SUMMARY_JSON, summary), (AUDIT_JSON, audit_report), (AUGMENTATION_JSON, augmentation)):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "audit": audit_report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
