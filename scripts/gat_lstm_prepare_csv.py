"""Audita o prepara un CSV etiquetado en un experimento con límites temporales propios."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modulos.clasificador.config import Config, parse_utc, partition_bounds
from app.modulos.clasificador.digests import sha256_file
from app.services.csv_etiquetado import labelled_columns, read_label


def inspect_csv(path):
    """Valida todas las etiquetas sin cargar el histórico completo en RAM."""
    classes, faults, entities = Counter(), Counter(), set()
    start = end = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader, [])
        names = [name.strip().lower() for name in headers]
        timestamp = "last_changed" if "last_changed" in names else "timestamp"
        if len(set(names)) != len(names) or not {"entity_id", "state", timestamp}.issubset(names):
            raise ValueError("Cabecera inválida: se requieren entity_id, state y last_changed/timestamp sin duplicados")
        columns = labelled_columns(headers)
        if columns is None:
            raise ValueError("El CSV requiere es_sintetica, tipo y tipo_falla")
        entity_col, time_col = names.index("entity_id"), names.index(timestamp)
        for line, raw in enumerate(reader, 2):
            try:
                if len(raw) != len(headers):
                    raise ValueError("Número de columnas incorrecto")
                entity = raw[entity_col]
                if not entity or entity != entity.strip():
                    raise ValueError("entity_id vacío o con espacios exteriores")
                at = parse_utc(raw[time_col])
                label = read_label(raw, columns)
                if label["label"] == "sensor_fault" and label["fault_type"] is None:
                    raise ValueError("La falla requiere tipo_falla para entrenar la cabeza de subtipos")
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Fila {line}: {exc}") from exc
            start = min(start, at) if start else at
            end = max(end, at) if end else at
            classes[label["label"]] += 1
            if label["fault_type"]:
                faults[label["fault_type"]] += 1
            entities.add(entity)
    if start is None:
        raise ValueError("El CSV no contiene lecturas")
    return {"filename": Path(path).name, "sha256": sha256_file(path),
            "rows": sum(classes.values()), "start": start.isoformat(), "end": end.isoformat(),
            "entity_ids": sorted(entities), "classes": dict(classes), "fault_types": dict(faults)}


def experiment_manifest(boundaries, audit):
    moments = [parse_utc(value) for value in boundaries]
    if len(moments) != 4 or any(right <= left for left, right in zip(moments, moments[1:])):
        raise ValueError("Se requieren cuatro límites UTC estrictamente crecientes")
    if parse_utc(audit["start"]) < moments[0] or parse_utc(audit["end"]) >= moments[1]:
        raise ValueError("El CSV completo debe estar dentro de Desarrollo [inicio, fin)")
    return {"boundaries": [value.isoformat() for value in moments], "partitions": {},
            "purpose": "isolated_labelled_csv_development", "source_sha256": audit["sha256"],
            "evaluation_data_loaded": False}


def write_contract(path, value):
    """Reanudar el mismo experimento es seguro; cambiar su identidad se rechaza."""
    value = json.loads(json.dumps(value, ensure_ascii=False))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"El experimento existente no coincide; usa otro --output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--inspect", action="store_true", help="Solo valida y resume; no prepara ni entrena")
    parser.add_argument("--output", type=Path, help="Carpeta aislada para este CSV y configuración")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="Procesos CPU para preparar ventanas; predeterminado hasta 4")
    parser.add_argument("--prepare-device", default="cpu", help="Backend DTW: cpu o cuda:0 (AMD/ROCm)")
    parser.add_argument("--prepare-batch-size", type=int, default=16, help="Instantes por lote de preparación")
    parser.add_argument("--max-storage-gib", type=float, default=4., help="Límite total de secuencias comprimidas, por defecto 4 GiB")
    parser.add_argument("--reserve-free-gib", type=float, default=5., help="Reserva mínima libre, incluye C: bajo WSL")
    parser.add_argument("--boundaries", nargs=4, metavar=("START", "DEV_END", "VAL_END", "TEST_END"),
                        help="Límites explícitos; Desarrollo se divide 80/20 en train_fit/dev_tune")
    args = parser.parse_args()
    try:
        audit = inspect_csv(args.csv)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        if args.inspect:
            return
        if args.output is None or args.boundaries is None:
            parser.error("Preparar requiere --output y --boundaries; usa --inspect para solo auditar")
        config = Config.from_json(args.config) if args.config else Config()
        if args.workers < 1 or args.prepare_batch_size < 1:
            raise ValueError("--workers y --prepare-batch-size deben ser positivos")
        if not 0 < args.max_storage_gib < float("inf") or not 0 <= args.reserve_free_gib < float("inf"):
            raise ValueError("Presupuesto y reserva de almacenamiento inválidos")
        os.environ["GAT_STORAGE_MAX_BYTES"] = str(int(args.max_storage_gib * 1024**3))
        os.environ["GAT_STORAGE_RESERVE_BYTES"] = str(int(args.reserve_free_gib * 1024**3))
        Config(device=args.prepare_device)
        os.environ["GAT_PREPARE_WORKERS"] = str(args.workers)
        os.environ["GAT_PREPARE_DEVICE"] = args.prepare_device
        os.environ["GAT_PREPARE_BATCH_SIZE"] = str(args.prepare_batch_size)
        if len(audit["entity_ids"]) > config.max_nodes:
            raise ValueError(f"El CSV supera max_nodes={config.max_nodes}")
        manifest = experiment_manifest(args.boundaries, audit)
        output = args.output.resolve()
        write_contract(output / "experiment.json", {"source": audit, "config": config.as_dict(),
                                                    "boundaries": manifest["boundaries"]})
        # Config.setting reads this override dynamically. The main application's
        # canonical manifest, databases and active model are never overwritten.
        history = output / "history"
        write_contract(history / "particiones_seleccionadas_resumen.json", manifest)
        os.environ["HISTORICAL_DATA_DIR"] = str(history)
        from app.services.diagnostico import DiagnosticWorkbench
        workbench = DiagnosticWorkbench(output / "review")
        imported = workbench.review.import_csv(args.csv, args.csv.name, "development")
        def progress(percent, message, **details):
            print(json.dumps({"percent": percent, "message": message, **details}, ensure_ascii=False), flush=True)
        prepared = output / "prepared.json"
        if prepared.exists():
            # The immutable experiment contract above has already been checked.
            dataset = json.loads(prepared.read_text(encoding="utf-8"))
        else:
            dataset = workbench.build_gat_dataset([f"review:{imported['dataset_id']}"], [],
                                                  config.as_dict(), progress)
            dataset = {**dataset, "review_dir": str(output / "review"),
                       "temporal_bounds": {key: [start.isoformat(), end.isoformat() if end else None]
                                           for key, (start, end) in partition_bounds(manifest).items()}}
            write_contract(prepared, dataset)
        print(json.dumps(dataset, ensure_ascii=False, indent=2), flush=True)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
