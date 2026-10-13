"""Migración offline del almacenamiento anterior, con inventario y reanudación.

Los movimientos ocurren dentro del mismo Artifact; las bases no se exportan ni
se reconstruyen. Cada movimiento queda registrado antes de retirar el origen.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sqlite3

VERSION = "artifact-layout-v2"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".migration.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_supported(value):
    return value.get("method", "gat_lstm") == "gat_lstm" and value.get("config", {}).get("architecture_version") == "gat_lstm_v3_segments_multiscale"


def safe(root, path):
    path = Path(path).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Ruta fuera de Artifact: {path}")
    return path


def database_inventory(path):
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        check = db.execute("PRAGMA quick_check").fetchall()
        if check != [("ok",)]:
            raise ValueError(f"Base dañada {path}: {check}")
        if db.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError(f"Referencias SQLite incompatibles: {path}")
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        return {name: db.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0] for name in tables}


def old_artifacts(directory):
    """Clasificar por manifiestos, nunca por coincidencias del nombre del archivo."""
    maps = {folder: {p.stem: read_json(p) for p in (directory / folder).glob("*.json")}
            for folder in ("models", "datasets", "configurations", "encoders")}
    kept = {key: value for key, value in maps["models"].items() if is_supported(value)}
    retained_datasets = {m.get("dataset_id") for m in kept.values()}
    removed = {folder: sorted(key for key, value in entries.items() if not is_supported(value))
               for folder, entries in maps.items()}
    removed["datasets"] = [key for key in removed["datasets"] if key not in retained_datasets]
    # Some historical metadata is absent; model-owned weights and references are
    # still identified by the model manifest, not by a directory-wide deletion.
    refs = set().union(*(set(m.get("reference_ids", [])) for m in maps["models"].values()))
    refs.update(p.stem for p in (directory / "references").glob("*.json")
                if read_json(p).get("feature_version") == "causal-window-1.0")
    retained_refs = set().union(*(set(m.get("reference_ids", [])) for m in kept.values()))
    removed["references"] = sorted(refs - retained_refs)
    return removed, kept


def cleanup_retired_references(root, report):
    """Incluye referencias huérfanas cuyo manifiesto identifica el motor retirado."""
    directory = root / "models/gat_lstm"
    retained = set()
    for key in report["retained_model_ids"]:
        retained.update(read_json(directory / "models" / f"{key}.json").get("reference_ids", []))
    removed = set(report["removed"]["references"])
    for path in (directory / "references").glob("*.json"):
        if path.stem not in retained and read_json(path).get("feature_version") == "causal-window-1.0":
            removed.add(path.stem)
            remove_checked(root, path)
    report["removed"]["references"] = sorted(removed)


def move_pairs(root):
    old = root / "data/review/diagnostics"
    pairs = []
    for folder in ("models", "weights", "graphs", "preprocessors", "configurations", "sensors", "references", "encoders", "experiments"):
        pairs.append((old / folder, root / "models/gat_lstm" / folder))
    pairs += [(old / "active.json", root / "models/gat_lstm/active.json"),
              (old / "datasets", root / "data/gat_lstm/datasets"),
              (old / "sequences", root / "data/gat_lstm/sequences"),
              (old / "jobs", root / "storage/jobs/diagnostics"),
              (root / "data/review/review.sqlite3", root / "storage/database/review.sqlite3"),
              (old / "diagnostics.sqlite3", root / "storage/database/diagnostics.sqlite3"),
              (root / "HistoricosEnUso", root / "data/history")]
    cache = root / "data/cache"
    if cache.exists():
        for child in cache.iterdir():
            if child.name.endswith(("-wal", "-shm")):
                continue
            dest = root / "storage" / ("database" if child.suffix in {".sqlite3", ".sqlite", ".db"} else "cache") / child.name
            pairs.append((child, dest))
    pairs.append((root / "backend/app/conf.txt", root / "data/defaults/conf.txt"))
    return pairs


def merge_check(source, target):
    if not target.exists():
        return
    if source.is_dir() and target.is_dir():
        for child in source.iterdir():
            merge_check(child, target / child.name)
    elif source.is_file() and target.is_file() and source.stat().st_size == target.stat().st_size and digest(source) == digest(target):
        return
    else:
        raise ValueError(f"Destino incompatible; no se sobrescribe: {target}")


def move_checked(root, source, target):
    source, target = safe(root, source), safe(root, target)
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        source.rename(target)
    elif source.is_dir():
        for child in list(source.iterdir()):
            move_checked(root, child, target / child.name)
        source.rmdir()
    else:
        merge_check(source, target)
        source.unlink()


def remove_checked(root, path):
    path = safe(root, path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def referenced_models(value):
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model_id", "model_version", "diagnostic_model_id"} and isinstance(item, str):
                found.add(item)
            found.update(referenced_models(item))
    elif isinstance(value, list):
        for item in value:
            found.update(referenced_models(item))
    return found


def purge_database(root, removed, report):
    retired = set(removed["models"])
    path = root / "storage/database/diagnostics.sqlite3"
    if path.exists():
        with closing(sqlite3.connect(path, timeout=1)) as db, db:
            runs = [(key, read) for key, raw in db.execute("SELECT run_id,payload FROM runs") if not is_supported((read := json.loads(raw)).get("model") or {})]
            keys = [r[0] for r in runs]
            # The subquery avoids a parameter limit and removes orphaned old
            # diagnoses as well as snapshots created by the catalogue fallback.
            db.execute("CREATE TEMP TABLE retired_runs (run_id TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO retired_runs VALUES (?)", [(key,) for key in keys])
            count = db.execute("DELETE FROM decisions WHERE run_id IN (SELECT run_id FROM retired_runs)").rowcount
            db.execute("DELETE FROM runs WHERE run_id IN (SELECT run_id FROM retired_runs)")
            report["removed_diagnostic_decisions"] = report.get("removed_diagnostic_decisions", 0) + count
    path = root / "storage/database/flow_lab.sqlite3"
    if path.exists():
        with closing(sqlite3.connect(path, timeout=1)) as db, db:
            for key, raw in db.execute("SELECT flow_id,payload FROM flows").fetchall():
                flow = json.loads(raw)
                missing = referenced_models(flow) & retired
                if missing:
                    flow["requires_model_selection"] = sorted(missing)
                    db.execute("UPDATE flows SET payload=? WHERE flow_id=?", (json.dumps(flow, ensure_ascii=False), key))
            deleted_runs = []
            for key, raw in db.execute("SELECT run_id,snapshot FROM flow_runs").fetchall():
                snapshot = json.loads(raw)
                ids = referenced_models(snapshot)
                classifier_nodes = [n["id"] for n in snapshot.get("flow", {}).get("nodes", []) if n.get("type") == "classifier"]
                for node_id in classifier_nodes:
                    row = db.execute("SELECT payload FROM flow_results WHERE run_id=? AND node_id=? LIMIT 1", (key, node_id)).fetchone()
                    if row:
                        ids.update(referenced_models(json.loads(row[0])))
                if ids & retired or any(identifier.startswith("catalogue:") for identifier in ids):
                    deleted_runs.append(key)
            for key in deleted_runs:
                db.execute("DELETE FROM flow_results WHERE run_id=?", (key,))
                db.execute("DELETE FROM flow_runs WHERE run_id=?", (key,))
            report["removed_flow_runs"] = report.get("removed_flow_runs", 0) + len(deleted_runs)


def migrate(root, *, apply=False, offline=False, progress=print):
    root = Path(root).resolve()
    journal = root / "storage/migrations/layout-v2.json"
    if apply:
        if not offline:
            raise ValueError("Detén la aplicación y usa --offline para migrar")
        with socket.socket() as probe:
            probe.settimeout(.2)
            if probe.connect_ex(("127.0.0.1", 8000)) == 0:
                raise ValueError("El puerto 8000 sigue activo; detén la API antes de migrar")
    if journal.exists() and read_json(journal).get("status") == "complete":
        report = read_json(journal)
        if apply:
            cleanup_retired_references(root, report)
            write_json(journal, report)
        return report
    old = root / "data/review/diagnostics"
    report = read_json(journal) if journal.exists() else None
    if report is None:
        removed, kept = old_artifacts(old)
        report = {"version": VERSION, "status": "planned", "removed": removed,
                  "retained_model_ids": sorted(kept), "moves": [], "databases": {}}
    pairs = move_pairs(root)
    for source, target in pairs:
        safe(root, source); safe(root, target)
        if source.exists():
            merge_check(source, target)
    if not apply:
        return {**report, "pending_moves": [[str(s.relative_to(root)), str(t.relative_to(root))] for s,t in pairs if s.exists()]}
    report["status"] = "moving"
    write_json(journal, report)
    for source, target in pairs:
        if not source.exists():
            continue
        progress(f"Migrando {source.relative_to(root)}", flush=True)
        is_database = source.suffix in {".sqlite3", ".sqlite", ".db"}
        if is_database:
            # Fail if a writer still owns the DB; checkpoint before moving WAL.
            with closing(sqlite3.connect(source, timeout=1)) as db:
                db.execute("BEGIN IMMEDIATE"); db.rollback()
                checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint and checkpoint[0]:
                    raise ValueError(f"Base en uso: {source}")
            counts = database_inventory(source)
            report["databases"][target.name] = counts
            write_json(journal, report)
        move_checked(root, source, target)
        if is_database:
            if database_inventory(target) != counts:
                raise ValueError(f"Verificación fallida después de mover {target}")
            for suffix in ("-wal", "-shm"):
                Path(str(source) + suffix).unlink(missing_ok=True)
        report["moves"].append([str(source.relative_to(root)), str(target.relative_to(root))])
        write_json(journal, report)
    progress("Verificando referencias y retirando artefactos antiguos", flush=True)
    models = root / "models/gat_lstm"
    for p in (root / "data/gat_lstm/datasets").glob("*.json"):
        item = read_json(p)
        if is_supported(item):
            sequences = root / "data/gat_lstm/sequences" / p.stem
            if not sequences.is_dir():
                raise ValueError(f"Faltan secuencias retenidas: {p.stem}")
            item["sequence_dir"] = str(sequences)
            write_json(p, item)
    for key in report["retained_model_ids"]:
        item = read_json(models / "models" / f"{key}.json")
        if digest(models / "weights" / f"{key}.pt") != item["weights_sha256"]:
            raise ValueError(f"Pesos v2 incompatibles: {key}")
        for folder in ("graphs", "preprocessors"):
            read_json(models / folder / f"{key}.json")
    active_path = models / "active.json"
    if active_path.exists() and read_json(active_path).get("model_id") not in report["retained_model_ids"]:
        write_json(active_path, {"model_id": None, "reason": "retired_architecture"})
    report["status"] = "cleaning"
    write_json(journal, report)
    removed = report["removed"]
    purge_database(root, removed, report)
    for key in removed["models"]:
        for folder, suffix in (("models", ".json"), ("weights", ".pt"), ("graphs", ".json"), ("preprocessors", ".json")):
            remove_checked(root, models / folder / (key + suffix))
        remove_checked(root, models / "experiments" / key)
    for key in removed["datasets"]:
        remove_checked(root, root / "data/gat_lstm/datasets" / f"{key}.json")
        remove_checked(root, root / "data/gat_lstm/sequences" / key)
    for folder in ("configurations", "encoders", "references"):
        for key in removed[folder]:
            remove_checked(root, models / folder / f"{key}.json")
            if folder == "encoders":
                remove_checked(root, models / folder / f"{key}.pt")
    for path in (root / "storage/jobs/diagnostics").glob("*.json"):
        job = read_json(path)
        result = job.get("result") or {}
        if (referenced_models(job) & set(removed["models"]) or result.get("dataset_id") in removed["datasets"]
                or job.get("operation") in {"build_dataset", "train", "pretrain_gat"}):
            remove_checked(root, path)
    # Manual annotations, source data and revision history must remain identical.
    review = root / "storage/database/review.sqlite3"
    if review.exists() and database_inventory(review) != report["databases"].get("review.sqlite3", {}):
        raise ValueError("Cambió el inventario de revisión durante la migración")
    report["status"] = "complete"
    cleanup_retired_references(root, report)
    write_json(journal, report)
    return report
