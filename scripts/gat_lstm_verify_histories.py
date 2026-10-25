"""Verifica secuencias portables completas sin cargar ni entrenar el modelo."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.history_preparation import read_manifest, save_json, source_path, validate_coverage, world_bounds


def verify(manifest_path, prepared):
    import h5py
    import numpy as np
    prepared = Path(prepared)
    root, manifest = read_manifest(manifest_path)
    dataset = json.loads((prepared / "prepared.json").read_text(encoding="utf-8"))
    coverage = json.loads((prepared / "coverage.json").read_text(encoding="utf-8"))
    if dataset.get("preparation_verification", {}).get("status") != "VERIFIED":
        raise ValueError("La preparación está incompleta o es solo una prueba")
    with (prepared / "targets.csv").open(encoding="utf-8", newline="") as f:
        targets = list(csv.DictReader(f))
    if sha256_file(prepared / "targets.csv") != coverage["targets_sha256"]:
        raise ValueError("El índice de objetivos cambió")
    expected = {(r["world_id"], int(r["source_row"])): r for r in targets}
    worlds = {w["world_id"]: w for w in manifest["worlds"]}
    for world in worlds.values():
        if dataset["source_hashes"].get(world["world_id"]) != world["sha256"]:
            raise ValueError("El dataset usa otros históricos")
        for key, checksum in (("path", "sha256"), ("audit", "audit_sha256")):
            if sha256_file(source_path(root, manifest, world, key)) != world[checksum]:
                raise ValueError("El histórico o su auditoría cambió")
    sequences = prepared / "review" / "diagnostics" / "sequences" / dataset["dataset_id"]
    pre = json.loads((sequences / "preprocessor.json").read_text(encoding="utf-8"))
    entities = pre["entity_ids"]
    if len(entities) != 15 or set(entities) != set(manifest["classifier_entities"]) or dataset["entity_ids"] != entities:
        raise ValueError("El preprocesador no conserva los 15 sensores")
    observed, seen, world_counts, shapes = [], set(), Counter(), {}
    for part in ("train_fit", "dev_tune"):
        with h5py.File(sequences / f"{part}.h5", "r") as file, gzip.open(
                sequences / f"{part}_metadata.jsonl.gz", "rt", encoding="utf-8") as stream:
            metadata = [json.loads(line) for line in stream]
            if not file.attrs.get("complete") or file["X"].shape[2] != 15 or len(file["y"]) != len(metadata):
                raise ValueError("Arrays incompletos o con sensores/filas perdidos")
            if len(metadata) != dataset["partition_counts"][part]:
                raise ValueError("El número de secuencias difiere del manifiesto")
            shapes[part] = {k: list(file[k].shape) for k in ("X", "y", "label_mask")}
            windows = file["window_index"][:]
            if windows.min() < 0 or windows.max() >= len(file["X"]):
                raise ValueError("Índice de ventanas fuera de rango")
            for offset in range(0, len(file["X"]), 64):
                if not np.isfinite(file["X"][offset:offset+64]).all():
                    raise ValueError("Características NaN o infinitas")
            labels, masks = file["y"][:], file["label_mask"][:]
            for i, meta in enumerate(metadata):
                world = worlds[meta["world_id"]]
                key = (meta["world_id"], meta["source_row"])
                original = expected.get(key)
                if key in seen or original is None:
                    raise ValueError("Secuencia duplicada o sin objetivo original")
                seen.add(key)
                if meta["entity_ids"] != entities or meta["context_entities"] != entities:
                    raise ValueError("Orden de sensores distinto")
                node = entities.index(meta["next_entity"])
                if not masks[i, node] or masks[i].sum() != 1 or labels[i, node] != LABELS.index(original["label"]):
                    raise ValueError("Etiqueta enmascarada o asignada a otro sensor")
                if not file["node_present"][windows[i], :, node].any():
                    raise ValueError("Sensor objetivo ausente de todas las escalas")
                lo, hi = world_bounds(world)[part]
                stamp, start = parse_utc(meta["context_end"]), parse_utc(meta["context_start"])
                if not lo <= start <= stamp < hi or meta["partition"] != part:
                    raise ValueError("La ventana cruza el corte UTC de su partición")
                if (meta["target_time"] != original["timestamp"] or meta["parent_episode_id"] != original["episode_id"]
                    or meta["next_entity"] != original["entity_id"]
                    or (meta["fault_subtype"] or "") != original["fault_subtype"]
                    or meta["source_sha256"] != world["sha256"]
                    or set(meta["input_provenance"]) != {world["world_id"]}):
                    raise ValueError("Se mezcló un mundo, instante, etiqueta o episodio")
                observed.append(original)
                world_counts[world["world_id"] + "/" + part] += 1
    if seen != set(expected):
        raise ValueError("Se perdieron objetivos durante la preparación")
    episodes = json.loads((root / manifest["episodes"]).read_text(encoding="utf-8-sig"))
    report = validate_coverage(observed, episodes, manifest["classifier_entities"])
    report.update(targets=len(observed), world_partition_counts=dict(world_counts), shapes=shapes,
                  all_features_finite=True, worlds_never_mixed=True, utc_context_bounds_preserved=True,
                  dataset_id=dataset["dataset_id"], training_executed=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("prepared", type=Path)
    args = parser.parse_args()
    result = verify(args.manifest, args.prepared)
    save_json(args.prepared / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
