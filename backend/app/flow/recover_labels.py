"""Recover reports for a saved CSV run after verifying its source fingerprint."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import pandas as pd

from app.database.evaluacion_etiquetada import separate_csv
from app.flow.definitions import _json
from app.flow.evaluacion import labelled_report, statistical_labelled_report, _merge_report_groups
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
        for node_id in ["output", *statistical_nodes]:
            groups = {}
            cursor = db.execute("SELECT observation_id,payload FROM flow_results WHERE run_id=? AND node_id=? ORDER BY sequence",
                                (run_id, node_id))
            while batch := cursor.fetchmany(500):
                items = [json.loads(payload) for _, payload in batch]
                report = (labelled_report(items, truth, truth_index=truth_index) if node_id == "output" else
                          statistical_labelled_report(items, truth, node_id, truth_index=truth_index))
                _merge_report_groups(groups, report)
                if apply and node_id == "output":
                    db.executemany("UPDATE flow_results SET payload=? WHERE run_id=? AND node_id='output' AND observation_id=?",
                                   ((_json(item), run_id, identifier) for (identifier, _), item in zip(batch, items)))
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
            db.execute("UPDATE flow_runs SET snapshot=?,summary=? WHERE run_id=?", (_json(snapshot), _json(summary), run_id))
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
