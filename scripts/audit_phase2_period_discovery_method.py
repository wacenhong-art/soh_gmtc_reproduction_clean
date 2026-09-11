"""Train-only methodology audit for the Phase 2B STL period detector.

This is a diagnostic audit, not an STL run and not a replacement period
selector.  It uses only Protocol-A TRAIN and Protocol-B TRAIN_REAL values via
the already-reviewed extraction functions.  Validation counts bound observable
lags; validation/test values, models, metrics, and component decomposition are
not used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from run_phase2_stl_structural_audit import _calce_train_capacity, _linear_detrend, _membership, _nasa_train_capacity


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "stl"


def _max_lag(n_train: int, n_validation: int) -> int:
    return min(n_train // 2, n_validation // 2)


def _acf(values: np.ndarray, max_lag: int) -> dict:
    denominator = float(np.dot(values, values))
    if denominator <= np.finfo(float).eps:
        return {"status": "UNAVAILABLE_ZERO_VARIANCE"}
    scores = {lag: float(np.dot(values[:-lag], values[lag:]) / denominator) for lag in range(1, max_lag + 1)}
    local_peaks = []
    for lag in range(2, max_lag):
        center, left, right = scores[lag], scores[lag - 1], scores[lag + 1]
        if center > left and center >= right:
            local_peaks.append({
                "lag": lag,
                "acf": center,
                "immediate_neighbor_prominence": center - max(left, right),
                "above_approx_95_white_noise_bound": abs(center) > (1.96 / math.sqrt(values.size)),
            })
    first_zero = next((lag for lag, score in scores.items() if score <= 0), None)
    return {
        "status": "DIAGNOSTIC_ONLY",
        "acf_by_lag": scores,
        "local_peaks": local_peaks,
        "first_zero_crossing": first_zero,
        "approx_95_white_noise_bound": 1.96 / math.sqrt(values.size),
        "global_argmax_lag_ge_2": max(range(2, max_lag + 1), key=lambda lag: scores[lag]) if max_lag >= 2 else None,
    }


def _spectral(values: np.ndarray, max_lag: int) -> dict:
    n = values.size
    power = np.abs(np.fft.rfft(values)) ** 2
    options = []
    for bin_index in range(1, power.size):  # Explicit non-DC restriction.
        rounded_period = int(round(n / bin_index))
        if 2 <= rounded_period <= max_lag:
            options.append({
                "frequency_bin": bin_index,
                "frequency_cycles_per_sample": bin_index / n,
                "rounded_period": rounded_period,
                "period_fraction_of_series": rounded_period / n,
                "power": float(power[bin_index]),
            })
    if not options:
        return {"status": "UNAVAILABLE_NO_EXECUTABLE_NON_DC_BIN"}
    ranked = sorted(options, key=lambda row: row["power"], reverse=True)
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else None
    return {
        "status": "DIAGNOSTIC_ONLY",
        "top_peak": top,
        "second_peak": second,
        "top1_top2_power_ratio": None if second is None or second["power"] == 0 else top["power"] / second["power"],
        "top_peak_is_first_or_second_non_dc_bin": top["frequency_bin"] <= 2,
        "low_frequency_trend_contamination_risk": "NOT_EXCLUDABLE_WITH_OLS_DETREND_ONLY",
        "eligible_peak_count": len(ranked),
    }


def _signal_views(values: np.ndarray) -> dict[str, np.ndarray]:
    ols = _linear_detrend(values)
    first_difference = np.diff(values)
    scale = float(np.std(ols, ddof=0))
    normalized = ols / scale if scale > np.finfo(float).eps else ols
    return {
        "OLS_DETRENDED_CAPACITY": ols,
        "FIRST_DIFFERENCE_CAPACITY": first_difference,
        "NORMALIZED_OLS_DETRENDED_CAPACITY": normalized,
    }


def _harmonic_relations(acf_peaks: list[dict], spectral_period: int | None) -> list[dict]:
    if spectral_period is None:
        return []
    results = []
    for peak in acf_peaks:
        for order in (1, 2, 3):
            for relation, expected in (("ACF_EQUALS_SPECTRAL_TIMES_ORDER", spectral_period * order), ("SPECTRAL_EQUALS_ACF_TIMES_ORDER", spectral_period / order)):
                if abs(peak["lag"] - expected) <= 1:
                    results.append({"acf_lag": peak["lag"], "spectral_period": spectral_period, "relation": relation, "harmonic_order": order})
    return results


def _one_prefix(values: np.ndarray, validation_n: int, fraction: float) -> dict:
    prefix_n = math.floor(values.size * fraction)
    max_lag = _max_lag(prefix_n, validation_n)
    if max_lag < 3:
        return {"prefix_n": prefix_n, "max_lag": max_lag, "status": "UNRESOLVED_INSUFFICIENT_LAGS"}
    views = _signal_views(values[:prefix_n])
    diagnostics = {}
    for name, signal in views.items():
        signal_max_lag = min(max_lag, signal.size // 2)
        diagnostics[name] = {"acf": _acf(signal, signal_max_lag), "spectral": _spectral(signal, signal_max_lag)}
    return {"prefix_n": prefix_n, "max_lag": max_lag, "status": "DIAGNOSTIC_ONLY__NO_PERIOD_SELECTED", "signals": diagnostics}


def _track(protocol: str, dataset: str, battery: str, rows: list[dict[str, str]], validation_n: int) -> dict:
    values, provenance = (_nasa_train_capacity(rows) if dataset == "NASA" else _calce_train_capacity(rows))
    prefixes = {str(fraction): _one_prefix(values, validation_n, fraction) for fraction in (0.8, 0.9, 1.0)}
    full = prefixes["1.0"]
    ols = full.get("signals", {}).get("OLS_DETRENDED_CAPACITY", {})
    spectral_period = ols.get("spectral", {}).get("top_peak", {}).get("rounded_period")
    harmonics = _harmonic_relations(ols.get("acf", {}).get("local_peaks", []), spectral_period)
    return {
        "protocol": protocol,
        "dataset": dataset,
        "battery_id": battery,
        "n_train": int(values.size),
        "n_validation_metadata_only": validation_n,
        "capacity_provenance": provenance,
        "prefix_diagnostics": prefixes,
        "harmonic_relations_OLS_DETRENDED_ONLY_order_le_3": harmonics,
        "revised_detector_result": "UNRESOLVED__MIN_PERIOD_NOT_PAPER_OR_DOMAIN_JUSTIFIED__DIAGNOSTICS_DO_NOT_SELECT_A_PERIOD",
    }


def main() -> None:
    selected, validation_counts = _membership()
    tracks = []
    for (protocol, dataset, battery), rows in sorted(selected.items()):
        tracks.append(_track(protocol, dataset, battery, rows, validation_counts[(protocol, dataset, battery)]))
    output = {
        "scope": "PHASE_2C_STL_PERIOD_DISCOVERY_METHOD_AUDIT",
        "train_only": True,
        "test_loaded": False,
        "validation_values_used": False,
        "test_values_used": False,
        "model_metrics_used": False,
        "stl_executed": False,
        "original_phase2b_acf_rule": "linear_detrend; normalized autocorrelation; global argmax over every lag in [2,max_lag]; no local-peak, prominence, significance, or persistence exclusion",
        "original_phase2b_fft_rule": "linear_detrend; numpy rfft with implicit sampling interval one observation; exclude DC; choose largest power among bins mapping by rounded n/bin into [2,max_lag]",
        "revised_acf_rule_status": "NOT_FROZEN__MIN_PERIOD_HAS_NO_PAPER_OR_DOMAIN_AUTHORITY",
        "agreement_hierarchy": {
            "level_1": "ACF periodic local peak and spectral peak agree within predeclared tolerance",
            "level_2": "ACF peak and spectral peak have harmonic/subharmonic relation of order <=3",
            "level_3": "spectral evidence only",
            "level_4": "no reliable periodic evidence",
        },
        "tracks": tracks,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stl_period_method_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"track_count": len(tracks), "stl_executed": False, "test_loaded": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
