"""Microbenchmark con ventanas reales: compara salida original, cachés, procesos y GPU DTW."""
import argparse
from collections import defaultdict
import csv
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
from app.modulos.clasificador.config import Config, parse_utc
from app.modulos.clasificador.dataset import Target
from app.modulos.clasificador.features import _finite_number, fit_preprocessor
from app.modulos.clasificador.pipeline import _provider
from app.modulos.clasificador.preparation import prepared_targets
from app.modulos.clasificador.segments import build_sample, fit_spec
from app.services.estados_validos import valid_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--start", required=True, help="Inicio UTC de una muestra de cuatro días")
    parser.add_argument("--windows", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", nargs="+", default=["cpu"])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.windows < 1 or args.workers < 1:
        parser.error("Ventanas y procesos deben ser positivos")
    start = parse_utc(args.start)
    fit_end, end = start + timedelta(days=3), start + timedelta(days=4)
    index = defaultdict(list)
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        for source_row, raw in enumerate(csv.DictReader(stream), 2):
            at = parse_utc(raw.get("last_changed") or raw["timestamp"])
            if start <= at <= end:
                index[raw["entity_id"]].append({"entity_id": raw["entity_id"], "timestamp": at.isoformat(),
                    "_time": at, "state": raw["state"], "source_row": source_row, "provenance": "benchmark"})
    for rows in index.values():
        rows.sort(key=lambda row: (row["_time"], row["source_row"]))
    fit = [row for rows in index.values() for row in rows if row["_time"] < fit_end]
    kinds = {}
    for entity, rows in index.items():
        values = [row["state"] for row in rows if row["_time"] < fit_end and valid_state(row["state"])]
        kinds[entity] = "continuous" if values and all(_finite_number(v) for v in values) else "multistate"
    config = Config()
    prep = fit_preprocessor(fit, [], config, kinds)
    graph = fit_spec(fit, prep, config, (start, fit_end))
    stamps = sorted({row["timestamp"] for rows in index.values() for row in rows if fit_end <= row["_time"] <= end})[:args.windows]
    if not stamps:
        parser.error("La muestra no contiene instantes con 72 horas de contexto")
    targets = [Target(prep["entity_ids"][i % len(kinds)], stamp, 0, str(i), "train_fit", str(i), None)
               for i, stamp in enumerate(stamps)]
    reference_provider = _provider(index)
    started = perf_counter()
    reference = [build_sample(reference_provider, target.timestamp, prep, graph, config,
                              {target.entity_id: target.label}) for target in targets]
    baseline_seconds = perf_counter() - started
    report = {"source": args.csv.name, "sample_start": start.isoformat(), "nodes": len(kinds),
              "windows": len(targets), "reference_seconds": baseline_seconds, "runs": []}
    print(json.dumps({"reference_seconds": baseline_seconds, "windows": len(targets)}), flush=True)
    for device in args.devices:
        if device != "cpu":
            from app.modulos.clasificador.batched_dtw import distances
            distances([[0., 1.]], [[0., 1.]], .1, device=device)
        for workers in dict.fromkeys((1, args.workers)):
            os.environ.update(GAT_PREPARE_WORKERS=str(workers), GAT_PREPARE_DEVICE=device,
                              GAT_PREPARE_BATCH_SIZE="16")
            stats = {}
            started = perf_counter()
            with tempfile.TemporaryDirectory(prefix="gat-benchmark-") as temp:
                observed = list(prepared_targets(targets, index, prep, graph, config, Path(temp), stats))
            elapsed = perf_counter() - started
            for (_, actual), expected in zip(observed, reference):
                for field in ("X", "state_ids", "node_present", "y", "label_mask"):
                    dtype = np.float32 if field == "X" else None
                    np.testing.assert_array_equal(actual[field], np.asarray(expected[field], dtype=dtype))
                if actual["metadata"] != expected["metadata"]:
                    raise AssertionError("Los metadatos difieren de la referencia")
            row = {**stats, "wall_seconds": elapsed, "speedup": baseline_seconds / elapsed, "exact_arrays": True}
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
