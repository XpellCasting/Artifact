"""Mide almacenamiento en una muestra distribuida; no importa CSVs ni entrena."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from app.database.review import ReviewStore
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.pipeline import build_sequence_dataset, development_sources, load_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, type=Path, help="Experimento existente, con history y review")
    parser.add_argument("--output", required=True, type=Path, help="Carpeta nueva para la muestra y reporte")
    parser.add_argument("--windows", type=int, default=256, help="Máximo de instantes; conserva sus etiquetas")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-device", default="cpu")
    parser.add_argument("--max-output-mib", type=int, default=128)
    args = parser.parse_args()
    if not 1 <= args.windows <= 1024 or args.workers < 1 or not 8 <= args.max_output_mib <= 256:
        parser.error("La prueba admite 1–1024 instantes, procesos positivos y límite de 8–256 MiB")
    output = args.output.resolve()
    if output.exists():
        parser.error("--output debe ser una carpeta nueva; no se sobrescriben resultados")
    experiment = args.experiment.resolve()
    contract = json.loads((experiment / "experiment.json").read_text(encoding="utf-8"))
    config = Config(**contract["config"])
    Config(device=args.prepare_device)
    review = ReviewStore(experiment / "review")
    manifest = experiment / "history" / "particiones_seleccionadas_resumen.json"
    selected = [item["source_id"] for item in development_sources(manifest, review)
                if item["sha256"] == contract["source"]["sha256"]]
    if len(selected) != 1:
        parser.error("El experimento debe contener una única fuente idéntica al contrato")
    os.environ.update(GAT_PREPARE_WORKERS=str(args.workers), GAT_PREPARE_DEVICE=args.prepare_device,
                      GAT_STORAGE_MAX_BYTES=str(args.max_output_mib * 1024**2))
    output.mkdir(parents=True)
    def progress(percent, message, **details):
        print(json.dumps({"percent": percent, "message": message, **details}, ensure_ascii=False), flush=True)
    started = perf_counter()
    dataset = build_sequence_dataset(review, selected, [], config, output, manifest, progress,
                                     probe_samples=args.windows)
    info = dataset["storage_probe"]
    stats = dataset["storage_runtime"]
    total_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    source_dir = Path(dataset["sequence_dir"])
    feature_bytes = sum(path.stat().st_size for path in source_dir.glob("*.h5"))
    metadata_bytes = sum(path.stat().st_size for path in source_dir.glob("*.jsonl.gz"))
    shared_extras = total_bytes - feature_bytes - metadata_bytes
    estimated = (feature_bytes * info["full_unique_windows"] / max(1, info["probe_unique_windows"])
                 + metadata_bytes * info["full_samples"] / max(1, info["probe_samples"]) + shared_extras)
    reads, seconds = 0, 0.
    for part in ("train_fit", "dev_tune"):
        rows = load_partition(dataset, part)
        if not rows:
            continue
        try:
            indices = np.random.default_rng(42).permutation(len(rows)).tolist()
            began = perf_counter()
            for start in range(0, len(indices), 16):
                batch = rows.read_batch(indices[start:start + 16])
                reads += len(batch["X"])
            seconds += perf_counter() - began
        finally:
            rows.close()
    report = {"source_sha256": contract["source"]["sha256"], "boundaries": contract["boundaries"],
              "probe_only": True, **info, "storage": stats, "total_probe_bytes": total_bytes,
              "estimated_full_bytes": round(estimated),
              "suggested_budget_bytes": round(estimated * 1.5),
              "estimate_note": "Extrapolación de muestra estratificada; no es una medición completa. Margen operativo del 50%.",
              "read_samples": reads, "read_seconds": seconds,
              "read_samples_per_second": reads / max(seconds, 1e-9),
              "wall_seconds": perf_counter() - started, "dataset": dataset}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"dataset", "strata"}},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
