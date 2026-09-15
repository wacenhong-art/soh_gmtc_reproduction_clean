"""Read-only readiness audit for the paper's voltage-range condition wording.

The audit looks at B0005/B0006 paired charge traces only. It never reads
capacity/SOH fields, materializes TEST data, trains a model, or defines a new
voltage-window health feature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_nasa_discharge_centered_pairing import audit
from run_phase9a_tier2_paired_train_val_dry_run import _cycles
from tier2_hf_extraction import NASA_CANONICAL_VOLTAGE_FIELD, V_END, V_MAX, V_START


ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "PHASE9C_VOLTAGE_RANGE_CONDITION_READINESS_METRICS.json"
REPORT_PATH = ROOT / "PHASE9C_VOLTAGE_RANGE_CONDITION_READINESS_REPORT.md"
SOURCE_BATTERIES = ("B0005", "B0006")
FULL_RANGE = (3.6, 4.2)
PARTIAL_RANGE = (3.8, 4.0)


class AuditError(RuntimeError):
    """Raised when source-only structural audit cannot be completed faithfully."""


def finite_voltage(charge: Any) -> np.ndarray:
    values = np.asarray(getattr(charge.data, NASA_CANONICAL_VOLTAGE_FIELD), dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise AuditError("charge Voltage_measured is empty or non-finite")
    return values


def covers(values: np.ndarray, voltage_range: tuple[float, float]) -> bool:
    return bool(values.min() <= voltage_range[0] and values.max() >= voltage_range[1])


def audit_source_battery(battery: str) -> dict[str, Any]:
    metadata = audit(battery)
    if not metadata["one_to_one"] or metadata["n_valid_pairs"] != 168:
        raise AuditError(f"source pairing contract failed for {battery}")
    cycles = _cycles(battery)
    records: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(metadata["pair_identities"]):
        charge_index = int(pair["charge_record_index"])
        voltage = finite_voltage(cycles[charge_index])
        records.append({
            "battery_id": battery,
            "pair_index": pair_index,
            "charge_record_index": charge_index,
            "observed_start_voltage": float(voltage[0]),
            "observed_end_voltage": float(voltage[-1]),
            "observed_min_voltage": float(voltage.min()),
            "observed_max_voltage": float(voltage.max()),
            "covers_3p6_to_4p2": covers(voltage, FULL_RANGE),
            "covers_3p8_to_4p0": covers(voltage, PARTIAL_RANGE),
        })
    return {
        "battery_id": battery,
        "legal_source_charge_count": len(records),
        "coverage_counts": {
            "covers_3p6_to_4p2": int(sum(item["covers_3p6_to_4p2"] for item in records)),
            "covers_3p8_to_4p0": int(sum(item["covers_3p8_to_4p0"] for item in records)),
        },
        "records": records,
    }


def hf_provenance() -> dict[str, Any]:
    return {
        "HF1": {
            "frozen_equation": "t_Vmax - t_0",
            "voltage_field": NASA_CANONICAL_VOLTAGE_FIELD,
            "voltage_start_metadata": "NOT_FIXED__CHARGE_OPERATION_START_t0",
            "voltage_end_metadata": V_MAX,
            "requires_data_outside_3p8_to_4p0_only": True,
            "computable_from_3p8_to_4p0_only": False,
            "reason": "requires charge-operation start t0 and the 4.2-V endpoint",
        },
        "HF2": {
            "frozen_equation": "t_Imin - t_Vmax",
            "voltage_field": NASA_CANONICAL_VOLTAGE_FIELD,
            "voltage_start_metadata": V_MAX,
            "voltage_end_metadata": "NOT_FIXED__CURRENT_CHARGE_CUTOFF_OR_FINAL_TIMESTAMP",
            "requires_data_outside_3p8_to_4p0_only": True,
            "computable_from_3p8_to_4p0_only": False,
            "reason": "requires the 4.2-V endpoint and post-4.2-V charger-current/cutoff context",
        },
        "HF3": {
            "frozen_equation": "t_4p0 - t_3p8",
            "voltage_field": NASA_CANONICAL_VOLTAGE_FIELD,
            "voltage_start_metadata": V_START,
            "voltage_end_metadata": V_END,
            "requires_data_outside_3p8_to_4p0_only": False,
            "computable_from_3p8_to_4p0_only": True,
            "reason": "both frozen endpoints are 3.8 V and 4.0 V",
        },
    }


def observed_range_variability(batteries: dict[str, Any]) -> dict[str, Any]:
    all_records = [record for battery in batteries.values() for record in battery["records"]]
    observed_pairs = {(record["observed_start_voltage"], record["observed_end_voltage"]) for record in all_records}
    return {
        "raw_observed_start_end_pairs_distinct_count": len(observed_pairs),
        "raw_observed_start_end_ranges_observable": True,
        "multiple_raw_observed_start_end_pairs": len(observed_pairs) > 1,
        "current_pipeline_emits_voltage_range_condition": False,
        "current_pipeline_condition": "NORMALIZED_CYCLE_POSITION",
        "reason_not_directly_available": "Observed trace endpoints are raw metadata, but the frozen SCF-GAN pipeline has no authorized mapping from per-cycle observed endpoints or selected voltage windows to c=[V_start,V_end].",
    }


def write_report(metrics: dict[str, Any]) -> None:
    lines = [
        "# Phase 9C voltage-range condition readiness audit",
        "",
        "## Scope controls",
        "",
        "```text",
        "SOURCE_BATTERIES=B0005,B0006",
        "MODEL_TRAINED=False",
        "SYNTHETIC_HF_GENERATED=False",
        "TEST_LOADED=False",
        "TEST_ACCESS_COUNT=0",
        "PLAN_CHANGE_0018_MODIFIED=False",
        "```",
        "",
        "## Source trace coverage",
        "",
        "| Battery | legal paired charges | covers [3.6,4.2] V | covers [3.8,4.0] V |",
        "| --- | ---: | ---: | ---: |",
    ]
    for battery, result in metrics["source_batteries"].items():
        counts = result["coverage_counts"]
        lines.append(f"| {battery} | {result['legal_source_charge_count']} | {counts['covers_3p6_to_4p2']} | {counts['covers_3p8_to_4p0']} |")
    lines.extend([
        "",
        "## Frozen HF voltage provenance",
        "",
        "| HF | frozen endpoints / prerequisite | computable from [3.8,4.0] only |",
        "| --- | --- | --- |",
    ])
    for feature, detail in metrics["hf_provenance"].items():
        endpoint = f"start={detail['voltage_start_metadata']}; end={detail['voltage_end_metadata']}"
        lines.append(f"| {feature} | {endpoint} | {detail['computable_from_3p8_to_4p0_only']} |")
    lines.extend([
        "",
        "## Condition evidence and readiness",
        "",
        "```text",
        "PAPER_CONDITION_EVIDENCE:",
        "Eq5=CYCLE_PERIOD",
        "Fig3=CYCLE_PLUS_BATTERY_TYPE",
        "METHODOLOGY_PROSE=VOLTAGE_RANGE",
        "Fig2_VOLTAGE_RANGE_REPRESENTATION=START_VOLTAGE_PLUS_END_VOLTAGE",
        "PAPER_CONDITION_SEMANTICS_CONTRADICTION=True",
        f"SOURCE_MULTI_RANGE_CONDITION_AVAILABLE={metrics['source_multi_range_condition_available']}",
        f"HF1_COMPUTABLE_FROM_3P8_4P0_ONLY={metrics['hf_provenance']['HF1']['computable_from_3p8_to_4p0_only']}",
        f"HF2_COMPUTABLE_FROM_3P8_4P0_ONLY={metrics['hf_provenance']['HF2']['computable_from_3p8_to_4p0_only']}",
        f"HF3_COMPUTABLE_FROM_3P8_4P0_ONLY={metrics['hf_provenance']['HF3']['computable_from_3p8_to_4p0_only']}",
        f"VOLTAGE_RANGE_CONDITION_IMPLEMENTATION_READY={metrics['voltage_range_condition_implementation_ready']}",
        "```",
        "",
        "## Missing scientific definitions",
        "",
    ])
    for missing in metrics["missing_scientific_definitions"]:
        lines.append(f"- {missing}")
    lines.append("")
    lines.append("The full JSON manifest records observed start/end/min/max and both coverage flags for every legal source charge. No new voltage-window HF formula or condition encoding was created.")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    source_results = {battery: audit_source_battery(battery) for battery in SOURCE_BATTERIES}
    provenance = hf_provenance()
    range_info = observed_range_variability(source_results)
    missing = [
        "An author- or protocol-supported rule that maps NASA raw trace voltage windows or observed endpoints to the SCF-GAN condition vector c=[V_start,V_end].",
        "A source-training sample policy defining which multiple voltage ranges are presented for a single charge cycle and how their labels/feature vectors are paired.",
        "A paper-supported replacement or restriction rule for HF1 and HF2 when only [3.8,4.0] V observations are available; their frozen equations require 4.2-V and/or post-4.2-V context.",
        "An executable reconciliation of Eq. (5) cycle-period, Fig. 3 Cycle + Battery Type, and methodology voltage-range condition semantics.",
    ]
    metrics = {
        "task": "PHASE9C_VOLTAGE_RANGE_CONDITION_READINESS_AUDIT",
        "source_batteries": source_results,
        "hf_provenance": provenance,
        "raw_voltage_range_observability": range_info,
        "source_multi_range_condition_available": False,
        "hf1_computable_from_3p8_4p0_only": False,
        "hf2_computable_from_3p8_4p0_only": False,
        "hf3_computable_from_3p8_4p0_only": True,
        "voltage_range_condition_implementation_ready": False,
        "paper_condition_evidence": {
            "eq5": "CYCLE_PERIOD",
            "fig3": "CYCLE_PLUS_BATTERY_TYPE",
            "methodology_prose": "VOLTAGE_RANGE",
            "fig2_voltage_range_representation": "START_VOLTAGE_PLUS_END_VOLTAGE",
            "paper_condition_semantics_contradiction": True,
        },
        "missing_scientific_definitions": missing,
        "model_trained": False,
        "synthetic_hf_generated": False,
        "test_loaded": False,
        "test_access_count": 0,
        "plan_change_0018_modified": False,
        "next_action": "STOP__VOLTAGE_RANGE_CONDITION_REQUIRES_SCIENTIFIC_DEFINITION",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(metrics)
    print(json.dumps({key: metrics[key] for key in ("source_multi_range_condition_available", "hf1_computable_from_3p8_4p0_only", "hf2_computable_from_3p8_4p0_only", "hf3_computable_from_3p8_4p0_only", "voltage_range_condition_implementation_ready", "test_access_count")}, indent=2))


if __name__ == "__main__":
    main()
