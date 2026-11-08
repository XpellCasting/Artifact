"""Recover reports for a saved CSV run after verifying its source fingerprint."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
from itertools import islice
from uuid import uuid4
import zlib

import pandas as pd

from app.database.evaluacion_etiquetada import separate_csv
from app.flow.definitions import _json
from app.flow.evaluacion import labelled_report, statistical_labelled_report, _merge_report_groups
from app.flow.result_spool import archive_for_run, iter_run_results, decode_observation, sync_directory
from app.services.historial.cargar_historial import CargadorHistorial


def recover_run_labels(database: Path, csv_path: Path, run_id: str, *, apply: bool = False):
    if not database.is_file():
        raise FileNotFoundError(database)
    content = csv_path.read_bytes()
    _, labels = separate_csv(content)
    events = CargadorHistorial().cargar_real(csv_path).eventos
    events = events.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
    identity = events[["entity_id", "timestamp_dt", "state", "_raw_index"]]
    fingerprint = hashlib.sha256(pd.util.hash_pandas_object(identity, index=False).values.tobytes()).hexdigest()
    truth = [{"raw_index": index, "entity": entity, "tipo": kind, "synthetic": synthetic, "fault_type": fault}
             for index, entity, kind, synthetic, fault in labels]
    truth_index = {row["raw_index"]: row for row in truth}
    backup = None
    with closing(sqlite3.connect(database, timeout=30)) as db, db:
        row = db.execute("SELECT snapshot,summary FROM flow_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise ValueError("Ejecución no encontrada.")
        snapshot, summary = map(json.loads, row)
        source = snapshot.get("source", {})
        if (snapshot.get("execution_mode") != "causal_future" or source.get("fingerprint") != fingerprint
                or source.get("rows") != len(events)):
            raise ValueError("El CSV no coincide con la huella y las filas de la ejecución guardada.")
        if summary.get("evaluation", {}).get("labelled"):
            raise ValueError("La ejecución ya tiene etiquetas de referencia.")
        if apply:
            backup_dir = database.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            backup = backup_dir / f"{database.stem}-before-labels-{stamp}.sqlite3"
            with closing(sqlite3.connect(backup)) as target:
                db.backup(target)
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT summary FROM flow_runs WHERE run_id=?", (run_id,)).fetchone()[0] != row[1]:
                raise ValueError("La ejecución cambió durante la recuperación.")
        statistical_nodes = [node["id"] for node in snapshot["flow"]["nodes"]
                             if node["type"] == "statistical_focus" and node.get("config", {}).get("enabled") is not False]
        reports = {}
        archive = archive_for_run(db, database, run_id)
        for node_id in ["output", *statistical_nodes]:
            groups = {}
            cursor = iter_run_results(db, database, run_id, node_id)
            while batch := list(islice(cursor, 500)):
                items = [record["payload"] for record in batch]
                report = (labelled_report(items, truth, truth_index=truth_index) if node_id == "output" else
                          statistical_labelled_report(items, truth, node_id, truth_index=truth_index))
                _merge_report_groups(groups, report)
                if apply and node_id == "output" and archive is None:
                    db.executemany("UPDATE flow_results SET payload=? WHERE run_id=? AND node_id='output' AND observation_id=?",
                                   ((_json(item), run_id, record["observation_id"]) for record, item in zip(batch, items)))
            reports[node_id] = {"labelled": True, "groups": list(groups.values()),
                                "matched_rows": sum(group["total"] for group in groups.values())}
        report = {"version": "separate_non_evaluable_v2", **reports.pop("output")}
        if report["matched_rows"] != summary.get("output_count"):
            raise ValueError("No se pudieron asociar todos los resultados de salida con sus etiquetas.")
        summary["evaluation"] = report
        summary["statistical_evaluation"] = {node: {"scope": "received_by_node", **value} for node, value in reports.items()}
        if apply:
            source.update(labelled=True, labels_fingerprint=hashlib.sha256(content).hexdigest(),
                          labels_recovered_at=datetime.now(timezone.utc).isoformat())
            replacement = None
            try:
                if archive is not None:
                    # Copy-on-write: the original sealed file remains usable by
                    # concurrent readers and by the pre-recovery database backup.
                    replacement = archive.with_name(f"labels-{uuid4().hex}.sqlite3")
                    shutil.copyfile(archive, replacement)
                    with closing(sqlite3.connect(replacement)) as target, target:
                        target.execute("PRAGMA journal_mode=OFF")
                        target.execute("PRAGMA synchronous=OFF")
                        cursor = iter_run_results(db, database, run_id, "output")
                        delta = 0
                        while batch := list(islice(cursor, 500)):
                            items = [record["payload"] for record in batch]
                            labelled_report(items, truth, truth_index=truth_index)
                            for record, item in zip(batch, items):
                                identifier, blob, slot = target.execute("""SELECT observation.id,observation.payload,result.slot
                                    FROM observations AS observation JOIN results AS result ON result.observation=observation.id
                                    WHERE observation.identity=? AND result.node_id='output'""", (record["observation_id"],)).fetchone()
                                body = decode_observation(blob)
                                old_size = len(_json(body).encode("utf-8"))
                                body["nodes"][slot]["fields"]["evaluation"] = item["evaluation"]
                                raw = _json(body).encode("utf-8")
                                delta += len(raw) - old_size
                                target.execute("UPDATE observations SET payload=? WHERE id=?", (zlib.compress(raw, 1), identifier))
                        target.commit()
                        if target.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                            raise ValueError("El archivo anotado no superó la comprobación de integridad")
                    with replacement.open("r+b") as stream:
                        os.fsync(stream.fileno())
                    sync_directory(replacement.parent)
                    counts = json.loads(db.execute("SELECT counts FROM flow_run_archives WHERE run_id=?", (run_id,)).fetchone()[0])
                    counts["archive_bytes"] = replacement.stat().st_size
                    counts["uncompressed_bytes"] += delta
                    db.execute("UPDATE flow_run_archives SET filename=?,counts=? WHERE run_id=?",
                               (replacement.name, _json(counts), run_id))
                db.execute("UPDATE flow_runs SET snapshot=?,summary=? WHERE run_id=?", (_json(snapshot), _json(summary), run_id))
                db.commit()
            except BaseException:
                db.rollback()
                if replacement is not None:
                    replacement.unlink(missing_ok=True)
                raise
    return {"run_id": run_id, "applied": apply, "matched_rows": report["matched_rows"],
            "groups": len(report["groups"]), "backup": str(backup) if backup else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover_run_labels(args.database, args.csv, args.run_id, apply=args.apply), ensure_ascii=False))
