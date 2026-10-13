"""Replay one sensor with rebuilt references, without changing saved runs or labels."""
import argparse
import json
from pathlib import Path
import sqlite3
from threadpoolctl import threadpool_limits

from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.reproduccion import MotorReproduccionValidacion
from compare_focus_features import evaluate_statistical_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--truth-database", type=Path, required=True)
    parser.add_argument("--entity", default="sensor.weatherstation_inside_temp_2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loader = CargadorHistorial()
    print("Loading historical observations", flush=True)
    development, evaluation = loader.cargar_real(args.development), loader.cargar_real(args.evaluation)
    development.eventos = development.eventos[development.eventos.entity_id.eq(args.entity)].copy()
    development.entidades = development.entidades[development.entidades.entity_id.eq(args.entity)].copy()
    future = evaluation.eventos[evaluation.eventos.entity_id.eq(args.entity)].copy()
    if development.eventos.empty or future.empty:
        raise ValueError("The requested entity must exist in both datasets")
    if future.timestamp_dt.min() < development.eventos.timestamp_dt.max():
        raise ValueError("Evaluation must be subsequent to development")
    with sqlite3.connect(f"file:{args.truth_database.resolve()}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        truth = [dict(row) for row in db.execute(
            "SELECT raw_index,entity,tipo,synthetic,fault_type FROM evaluation_truth WHERE entity=?", (args.entity,))]
    filtro = FiltroEstadistico()
    with threadpool_limits(limits=1):
        baseline = MotorReproduccionValidacion(filtro).construir_baseline(
            development.eventos, development.entidades,
            progress_callback=lambda phase, percent, message: print(message, flush=True))
        summary = evaluate_statistical_batches(filtro, baseline, future, development.entidades, truth,
            progress=lambda done, total: print(f"Replay {done}/{total}", flush=True))
    report = {"entity_id": args.entity, "development": str(args.development),
              "evaluation": str(args.evaluation), "models": baseline["continuous_model_manifest"],
              "summary": summary}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary["statistical_evaluation"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
