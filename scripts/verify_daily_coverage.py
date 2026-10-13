"""Rebuild statistical references and replay existing labels into new artifacts.

Run with PYTHONPATH=backend. The original database and runs are read-only.
"""
import argparse
import json
from pathlib import Path
import pickle
import sqlite3
from threadpoolctl import threadpool_limits

from app.modulos.estadistico.residuos_secuenciales import IMPLEMENTATION_VERSION
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.procedencia_normalidad import effective_parameters
from app.services.reproduccion import MotorReproduccionValidacion
from compare_focus_features import evaluate_statistical_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--truth-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-baseline", action="store_true", help="Reuse this output directory's baseline with unchanged development data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    loader = CargadorHistorial()
    print("Loading development and evaluation", flush=True)
    development = loader.cargar_real(args.development)
    evaluation = loader.cargar_real(args.evaluation)
    with sqlite3.connect(f"file:{args.truth_database.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        truth = [dict(row) for row in db.execute("SELECT raw_index,entity,tipo,synthetic,fault_type FROM evaluation_truth")]
    filtro = FiltroEstadistico()
    def progress(phase, percent, message):
        print(f"{phase} {percent:.1f}% {message}", flush=True)
    if args.reuse_baseline:
        with (args.output / "baseline.pkl").open("rb") as stream:
            baseline = pickle.load(stream)
        # Restrict reuse to the same chronological per-entity endpoints.
        for entity, model in baseline["continuous_model_manifest"].items():
            if model.get("implementation_version") != IMPLEMENTATION_VERSION:
                raise ValueError("Detector implementation changed; rebuild the reference")
            rows = development.eventos[development.eventos.entity_id.eq(entity)]
            latest = rows.iloc[-1]
            if model["last_observation_at"] != latest.timestamp_dt.isoformat() or model["last_observed_value"] != str(latest.state):
                raise ValueError("Development changed; rebuild the reference")
    else:
        baseline = MotorReproduccionValidacion(filtro).construir_baseline(
            development.eventos, development.entidades, progress_callback=progress)
    with (args.output / "baseline.pkl").open("wb") as stream:
        pickle.dump(baseline, stream)
    (args.output / "baseline.json").write_text(json.dumps(effective_parameters(baseline), ensure_ascii=False, indent=2))
    last = -1
    def replay_progress(processed, total, phase, detail=None, **metadata):
        nonlocal last
        bucket = int(10 * processed / max(total, 1))
        if bucket != last:
            last = bucket
            print(f"Replay {processed}/{total}", flush=True)
    with threadpool_limits(limits=1):
        summary = evaluate_statistical_batches(filtro, baseline, evaluation.eventos,
            development.entidades, truth, progress=lambda done, total: replay_progress(done, total, "replay"))
    report = {"development": str(args.development), "evaluation": str(args.evaluation),
              "summary": summary}
    (args.output / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    counts = {"prepared": sum(m["eligible"] for m in baseline["continuous_model_manifest"].values()),
              "continuous_total": len(baseline["continuous_model_manifest"])}
    print(json.dumps(counts), flush=True)
    for row in summary["evaluation"]["groups"]:
        if row["expected"] == "Falla":
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
