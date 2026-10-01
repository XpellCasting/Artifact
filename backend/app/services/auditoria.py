"""Auditoría reproducible de fuentes y etiquetas sin escrituras operativas."""
from __future__ import annotations
from app.config import REVIEW_DATA_DIR, database_path

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.config import actual_partition
from app.modulos.clasificador.config import historical_manifest_path
from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.config import partition_bounds
from app.modulos.clasificador.config import required_context_seconds
from app.modulos.clasificador.config import resolve_manifest_source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_ro(path: Path):
    # mode=ro evita crear o migrar bases. URI de SQLite conserva rutas Windows.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _declared_matches(declared: str, actual: str) -> bool:
    return {"development": {"train_fit", "dev_tune"},
            "validation": {"validation"}, "test": {"test"}}.get(declared, set()).__contains__(actual)


def _annotation_parts(item: dict, bounds: dict, margin: timedelta) -> tuple[list[str], str | None]:
    start, end = parse_utc(item["start"]), parse_utc(item["end"])
    # ReviewStore usa ambos extremos inclusivos. Un punto tiene start == end.
    if end < start:
        return [], "inverted_interval"
    covered = []
    for name, (lo, hi) in bounds.items():
        if (hi is None or start < hi) and end >= lo:
            covered.append(name)
    if len(covered) != 1:
        return covered, "crosses_partition"
    name = covered[0]
    lo, hi = bounds[name]
    if start - margin < lo or (hi is not None and end >= hi):
        return covered, "context_crosses_partition"
    return covered, None


@dataclass
class AuditReport:
    payload: dict

    def as_dict(self):
        return self.payload


def audit_sources(config: Config, review_store=None, *, manifest_path: Path | None = None,
                  review_dir: Path | None = None) -> AuditReport:
    """Lee manifiesto, archivos y SQLite sin modificar sus contenidos.

    ``review_store`` se acepta por contrato, pero se usa solo su ruta: los métodos
    públicos de ReviewStore pueden refrescar índices y escribir en la base.
    """
    if manifest_path is None:
        manifest_path = historical_manifest_path()
    if review_dir is None:
        review_dir = Path(review_store.directory) if review_store is not None else REVIEW_DATA_DIR
    manifest_path, review_dir = Path(manifest_path), Path(review_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds = partition_bounds(manifest)
    margin = timedelta(seconds=required_context_seconds(config))
    report = {
        "schema_version": config.schema_version, "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "bounds": {name: [lo.isoformat(), hi.isoformat() if hi else None] for name, (lo, hi) in bounds.items()},
        "context_margin_seconds": int(margin.total_seconds()),
        "files": {}, "imported_sources": [], "annotations": {}, "issues": [], "aptitude": {},
        "torch_available": importlib.util.find_spec("torch") is not None,
    }
    all_entities = set()
    for name, entry in manifest["partitions"].items():
        path = resolve_manifest_source(manifest_path, entry["path"])
        info = {"path": str(path), "expected_sha256": entry["sha256"],
                "expected_rows": entry["rows"], "declared_start": entry.get("start"),
                "declared_end": entry.get("end"), "entities": entry.get("entities", {})}
        if path.is_file():
            info["sha256"] = sha256_file(path)
            info["hash_match"] = info["sha256"] == entry["sha256"]
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                import csv
                reader = csv.reader(stream)
                info["columns"] = next(reader, [])
                info["actual_rows"] = sum(1 for _ in reader)
            info["row_count_match"] = info["actual_rows"] == entry["rows"]
            if not info["hash_match"] or not info["row_count_match"]:
                report["issues"].append({"kind": "manifest_mismatch", "partition": name})
        else:
            info["exists"] = False
            report["issues"].append({"kind": "missing_file", "partition": name})
        report["files"][name] = info
        all_entities.update(entry.get("entities", {}))

    connection = _connect_ro(Path(review_store.database) if review_store is not None and hasattr(review_store, "database") else database_path("review.sqlite3", review_dir))
    try:
        datasets = [(json.loads(row["metadata"]), row["source_name"]) for row in connection.execute(
            "SELECT metadata,source_name FROM datasets")]
        annotations = [json.loads(row[0]) for row in connection.execute("SELECT payload FROM annotations")]
        source_by_id = {item["dataset_id"]: item for item, _ in datasets}
        for source, source_name in datasets:
            start, end = source.get("start"), source.get("end")
            sql = """SELECT COUNT(*) rows, COUNT(DISTINCT entity_id) entities,
                     SUM(CASE WHEN is_valid=0 THEN 1 ELSE 0 END) invalid_rows,
                     MIN(ts) first_us, MAX(ts) last_us
                     FROM events WHERE dataset_id=?"""
            stats = dict(connection.execute(sql, (source["dataset_id"],)).fetchone())
            actual_start = (datetime.fromtimestamp(stats["first_us"] / 1e6, timezone.utc).isoformat()
                            if stats["first_us"] is not None else None)
            actual_end = (datetime.fromtimestamp(stats["last_us"] / 1e6, timezone.utc).isoformat()
                          if stats["last_us"] is not None else None)
            first = actual_partition(parse_utc(actual_start), bounds) if actual_start else None
            last = actual_partition(parse_utc(actual_end), bounds) if actual_end else None
            mismatch = not (_declared_matches(source["partition"], first)
                            and _declared_matches(source["partition"], last))
            source_path = review_dir / "sources" / source_name
            source_hash = sha256_file(source_path) if source_path.is_file() else None
            irregularity = {}
            previous_entity, previous_ts = None, None
            for entity, ts in connection.execute(
                    "SELECT entity_id,ts FROM events WHERE dataset_id=? ORDER BY entity_id,ts,source_row",
                    (source["dataset_id"],)):
                if entity != previous_entity:
                    previous_entity, previous_ts = entity, ts
                    irregularity[entity] = {"intervals": 0, "zero_gap": 0,
                                            "gap_over_15_minutes": 0, "max_gap_seconds": 0.0}
                    continue
                gap = (ts - previous_ts) / 1e6
                record = irregularity[entity]
                record["intervals"] += 1
                record["zero_gap"] += gap == 0
                record["gap_over_15_minutes"] += gap > 900
                record["max_gap_seconds"] = max(record["max_gap_seconds"], gap)
                previous_ts = ts
            duplicate = connection.execute("""SELECT COUNT(*) FROM (
                SELECT entity_id,ts FROM events WHERE dataset_id=?
                GROUP BY entity_id,ts HAVING COUNT(*)>1)""", (source["dataset_id"],)).fetchone()[0]
            conflicts = connection.execute("""SELECT COUNT(*) FROM (
                SELECT entity_id,ts FROM events WHERE dataset_id=?
                GROUP BY entity_id,ts HAVING COUNT(DISTINCT state)>1)""", (source["dataset_id"],)).fetchone()[0]
            item = {"dataset_id": source["dataset_id"], "filename": source.get("filename"),
                    "sha256": source.get("sha256"), "declared_partition": source["partition"],
                    "actual_start_partition": first, "actual_end_partition": last,
                    "partition_mismatch": mismatch, "start": start, "end": end,
                    "actual_start": actual_start, "actual_end": actual_end,
                    "source_copy_path": str(source_path), "source_copy_sha256": source_hash,
                    "source_copy_hash_match": (source_hash == source.get("sha256")
                                               if source_hash is not None else None),
                    "temporal_irregularity": irregularity,
                    "columns": source.get("columns"), "rows": source.get("rows"),
                    "indexed_rows": source.get("indexed_rows"), "rejected_rows": source.get("rejected_rows"),
                    "rejection_summary": source.get("rejection_summary"),
                    "entity_kinds": {e["entity_id"]: e.get("kind") for e in source.get("entities", [])},
                    "duplicate_entity_timestamps": duplicate, "conflicting_entity_timestamps": conflicts,
                    **stats}
            report["imported_sources"].append(item)
            if source_hash != source.get("sha256"):
                report["issues"].append({"kind": ("missing_imported_source_copy" if source_hash is None
                                                  else "imported_source_hash_mismatch"),
                                         "dataset_id": source["dataset_id"]})
            if mismatch:
                report["issues"].append({"kind": "partition_mismatch", "dataset_id": source["dataset_id"],
                                         "declared": source["partition"], "actual": [first, last]})
        # A point is one target. Intervals remain one independent episode; no
        # artificial normal labels are generated from unannotated neighbors.
        by_partition = {name: {label: 0 for label in LABELS} for name in bounds}
        entities = {name: defaultdict(Counter) for name in bounds}
        statuses, origins, labels, kinds = Counter(), Counter(), Counter(), Counter()
        exclusions = []
        groups = defaultdict(set)
        accepted = []
        for item in annotations:
            label, status = item.get("label"), item.get("status")
            statuses[status] += 1; origins[item.get("reference_origin", "unspecified")] += 1
            labels[label] += 1
            is_point = item.get("source_row") is not None or item["start"] == item["end"]
            kinds["point" if is_point else "interval"] += 1
            parts, reason = _annotation_parts(item, bounds, margin)
            group = item.get("parent_episode_id") or item.get("annotation_id")
            groups[group].update(parts)
            if reason or status not in {"reviewed", "verified"} or label not in LABELS:
                exclusions.append({"annotation_id": item.get("annotation_id"),
                                   "reason": reason or "not_supervised", "partitions": parts})
                continue
            source = source_by_id.get(item.get("dataset_id"))
            if source is None or source.get("sha256") != item.get("source_sha256"):
                exclusions.append({"annotation_id": item.get("annotation_id"), "reason": "source_identity_mismatch"})
                continue
            part = parts[0]
            by_partition[part][label] += 1
            entities[part][item["entity_id"]][label] += 1
            accepted.append((item, part))
        shared_groups = sorted(group for group, parts in groups.items() if len(parts) > 1)
        if shared_groups:
            report["issues"].append({"kind": "shared_episode_across_partitions", "groups": shared_groups})
        conflicts = []
        for index, (left, _) in enumerate(accepted):
            for right, _ in accepted[index + 1:]:
                if (left["dataset_id"] == right["dataset_id"] and left["entity_id"] == right["entity_id"]
                    and left["label"] != right["label"] and left["start"] <= right["end"]
                    and right["start"] <= left["end"]):
                    conflicts.append([left["annotation_id"], right["annotation_id"]])
        if conflicts:
            report["issues"].append({"kind": "annotation_conflicts", "pairs": conflicts})
        report["annotations"] = {
            "total": len(annotations), "status": dict(statuses), "origin": dict(origins),
            "label": dict(labels), "geometry": dict(kinds),
            "eligible_by_partition": by_partition,
            "eligible_by_entity": {part: {e: dict(count) for e, count in rows.items()}
                                   for part, rows in entities.items()},
            "excluded": exclusions, "conflicts": conflicts,
            "independent_episode_groups": len(groups), "shared_groups": shared_groups,
            "interval_semantics": "inclusive_start_inclusive_end",
        }
    finally:
        connection.close()
    hashes_ok = all(item.get("hash_match") and item.get("row_count_match")
                    for item in report["files"].values())
    enough_entities = len(all_entities) <= config.max_nodes
    train_support = by_partition["train_fit"]
    dev_support = by_partition["dev_tune"]
    no_leak = not shared_groups and not conflicts
    report["aptitude"] = {
        "build_tensors": {"ok": hashes_ok and enough_entities,
                          "reason": [] if hashes_ok and enough_entities else ["manifest_or_node_limit"]},
        "pretrain": {"ok": None,
                     "data_ready": hashes_ok and enough_entities,
                     "missing_dependency": not report["torch_available"],
                     "reason": (["future_target_coverage_not_audited"] if hashes_ok and enough_entities
                                else ["manifest_or_node_limit"])},
        "train_three_classes": {"ok": hashes_ok and no_leak and all(train_support.values())
                                and all(dev_support.values()) and report["torch_available"],
                                "support_train_fit": train_support, "support_dev_tune": dev_support,
                                "missing_dependency": not report["torch_available"]},
        "evaluate_three_classes": {"ok": hashes_ok and no_leak
                                   and all(by_partition["validation"].values())
                                   and all(by_partition["test"].values()),
                                   "support_validation": by_partition["validation"],
                                   "support_test": by_partition["test"]},
    }
    return AuditReport(report)
