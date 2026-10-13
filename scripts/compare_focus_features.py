"""Compare feature sets on identical time partitions without tuning on labels."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from time import perf_counter

import pandas as pd
from threadpoolctl import threadpool_limits

from app.flow.definitions import observation_valid
from app.flow.evaluacion import CausalFlowRunner, labelled_report, statistical_labelled_report, _merge_report_groups
from app.modulos.estadistico.continuos_temporales import (
    FiltroContinuoTemporal, FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_SCHEMA_VERSION)
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.reproduccion import MotorReproduccionValidacion

TARGETS = {"sensor.hall_sensor_temperature", "sensor.office_sensor_temperature",
           "sensor.office_sensor_illuminance", "sensor.weatherstation_temp_2",
           "sensor.weatherstation_inside_temp_2"}


def evaluate_statistical_batches(filtro, baseline, events, entities, truth, *, chunk_size=2000, progress=None):
    """Statistical-only flow: keep per-entity chronology and simultaneous groups.

    This uses the same motor, evidence adapter and final-decision functions as
    the flow runner. No contextual or classifier node reads a batch's future.
    """
    columns = [name for name in events if name not in {"es_sintetica", "tipo", "tipo_falla", "fault_type", "label", "evaluation"}]
    ordered = events[columns].sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
    runtime_base = deepcopy(baseline)
    motor = MotorReproduccionValidacion(filtro)
    runner = CausalFlowRunner(filtro, None, None)
    runner.execution_id = "statistical-comparison"
    runner.baselines = {"statistical_focus": runtime_base}
    runner._evidence_nodes = {name: {"type": kind, "config": {"evidence_policy": "combine_v1"}}
        for name, kind in [("statistical_focus", "statistical_focus"), ("output", "output")]}
    truth_index = {int(row["raw_index"]): row for row in truth}
    groups, statistical_groups, decisions, start = {}, {}, Counter(), 0
    while start < len(ordered):
        end = min(start + chunk_size, len(ordered))
        while end < len(ordered) and ordered.iloc[end].timestamp_dt == ordered.iloc[end - 1].timestamp_dt:
            end += 1
        batch = ordered.iloc[start:end]
        runner.statistical_results = {"statistical_focus": motor.analizar_lote(batch, runtime_base, .6)}
        incoming = [{"observation_id": f"row:{row['_raw_index']}", "source_raw_index": row["_raw_index"],
            "entity_id": str(row["entity_id"]), "timestamp": row["timestamp_dt"].isoformat(),
            "sensor_kind": row["sensor_kind"], "state": str(row["state"]), "value": row.get("value"),
            "is_valid_value": observation_valid(row), "signal": "unknown", "evidence": []}
            for row in batch.to_dict("records")]
        output = runner._statistical("statistical_focus", incoming, batch, entities, {})
        _merge_report_groups(statistical_groups,
            statistical_labelled_report(output, truth, "statistical_focus", truth_index=truth_index))
        output = runner._merge("output", output, {"mode": "output"})
        for item in output:
            item["final_decision"] = runner._final_decision(item)
        block = labelled_report(output, truth, truth_index=truth_index)
        _merge_report_groups(groups, block)
        decisions.update(item["final_decision"] for item in output)
        start = end
        if progress:
            progress(start, len(ordered))
    return {"candidate_count": len(ordered), "by_final_decision": dict(decisions),
        "statistical_evaluation": {"statistical_focus": {"labelled": bool(truth), "scope": "received_by_node",
            "groups": list(statistical_groups.values()), "matched_rows": sum(g["total"] for g in statistical_groups.values())}},
        "evaluation": {"version": "separate_non_evaluable_v2", "labelled": bool(truth),
            "groups": list(groups.values()), "matched_rows": sum(g["total"] for g in groups.values())}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--truth-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-schema", nargs="+",
        choices=(LEGACY_FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION),
        default=(LEGACY_FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION))
    args = parser.parse_args()
    loader = CargadorHistorial()
    development, evaluation = loader.cargar_real(args.development), loader.cargar_real(args.evaluation)
    development.eventos = development.eventos[development.eventos.entity_id.isin(TARGETS)].copy()
    development.entidades = development.entidades[development.entidades.entity_id.isin(TARGETS)].copy()
    future = evaluation.eventos[evaluation.eventos.entity_id.isin(TARGETS)].copy()
    with sqlite3.connect(f"file:{args.truth_database.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        truth = [dict(row) for row in db.execute("SELECT raw_index,entity,tipo,synthetic,fault_type FROM evaluation_truth")]
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for version in args.feature_schema:
        print(f"Preparing {version}", flush=True)
        filtro = FiltroEstadistico()
        filtro.continuos_temporales = FiltroContinuoTemporal(version)
        started = perf_counter()
        with threadpool_limits(limits=1):
            baseline = MotorReproduccionValidacion(filtro).construir_baseline(
                development.eventos, development.entidades,
                progress_callback=lambda phase, percent, message: print(f"{phase}: {message}", flush=True))
            summary = evaluate_statistical_batches(filtro, baseline, future, development.entidades, truth,
                progress=lambda done, total: print(f"{version}: {done}/{total}", flush=True))
        results[version] = {"summary": summary, "models": baseline["continuous_model_manifest"],
            "elapsed_seconds": perf_counter() - started}
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
        print(f"{version}: false positives = {sum(row['false_positives'] for row in summary['evaluation']['groups'])}", flush=True)
    for entity in TARGETS if all(version in results for version in (LEGACY_FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_VERSION)) else ():
        old = results[LEGACY_FEATURE_SCHEMA_VERSION]["models"][entity]["partitions"]
        new = results[FEATURE_SCHEMA_VERSION]["models"][entity]["partitions"]
        if old != new:
            raise AssertionError(f"Unequal comparison partitions for {entity}")


if __name__ == "__main__":
    main()
