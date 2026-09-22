"""Resolve labels for the injected history without changing source annotations."""
import json
from collections import defaultdict

from .Revision_Fragmentos import parse_time, iso_time

TRUTH_FIELDS = ["dataset_id", "source_sha256", "partition", "entity_id", "start", "end",
    "label", "kind", "reference_origin", "evaluable", "review_status", "evaluation_scope",
    "normality_basis", "baseline_normality_basis", "externally_verified", "intervention_known",
    "annotation_id", "annotation_revision", "annotation_references", "injection_id",
    "source_row", "excluded_source_rows", "evaluation_note", "evidence", "reviewer"]
INTERVENTION_FIELDS = ["source_row", "entity_id", "timestamp", "original_state", "injected_state", "injection_id", "kind", "label"]


def flag(value):
    return value is True or value == "true"


def manual_segment(manuals, entity, start, stop, source_row="", excluded_rows=()):
    known = [a for a in manuals if flag(a.get("evaluable")) and a["label"] != "uncertain"]
    eligible = known or [a for a in manuals if a.get("review_status") == "reviewed"] or manuals
    ambiguous = len({(a["label"], a.get("fault_type") or "") for a in eligible}) > 1
    representative = sorted(eligible, key=lambda a: (a.get("updated_at", ""), a["annotation_id"]))[-1]
    label = "uncertain" if ambiguous else representative["label"]
    record = {key: representative.get(key, "") for key in TRUTH_FIELDS}
    record.update(entity_id=entity, start=iso_time(start), end=iso_time(stop - 1), label=label,
        kind=(representative.get("fault_type") or "") if label == "sensor_fault" else "",
        reference_origin="historical_review", intervention_known="false", injection_id="", source_row=source_row,
        annotation_revision=representative["revision"],
        annotation_references=json.dumps([{"annotation_id": a["annotation_id"], "revision": a["revision"]} for a in manuals]),
        excluded_source_rows=json.dumps(list(excluded_rows)), evaluable="true" if known and not ambiguous else "false",
        externally_verified="true" if representative.get("externally_verified") and not ambiguous else "false",
        evaluation_note="Interpretaciones contradictorias; excluido de métricas." if ambiguous else "Marcación manual del original; las etiquetas de fila prevalecen sobre las de intervalo.")
    if ambiguous:
        record.update(evaluation_scope="historical_review", normality_basis="not_applicable")
    return record


def compose_truth(annotations, injections):
    """Sweep closed microsecond intervals. Point spikes affect only their source row."""
    entities = defaultdict(list)
    for annotation in annotations:
        entities[annotation["entity_id"]].append(("manual", annotation))
    for injection in injections:
        entities[injection["entity_id"]].append(("injection", injection))
    output = []
    for entity, records in sorted(entities.items()):
        events = defaultdict(lambda: {"add": [], "remove": []})
        for index, (_, record) in enumerate(records):
            start, end = parse_time(record["start"])[0], parse_time(record["end"])[0]
            events[start]["add"].append(index)
            events[end + 1]["remove"].append(index)
        boundaries = sorted(events)
        active = set()
        for start, stop in zip(boundaries, boundaries[1:]):
            active.difference_update(events[start]["remove"])
            active.update(events[start]["add"])
            manuals = [records[i][1] for i in sorted(active) if records[i][0] == "manual"]
            synthetic = [records[i][1] for i in sorted(active) if records[i][0] == "injection"]
            references = json.dumps([{"annotation_id": a["annotation_id"], "revision": a["revision"]} for a in manuals], separators=(",", ":"))
            excluded_rows = []
            if synthetic:
                injection = synthetic[0]
                conflict = any(a.get("review_status") == "reviewed" and a["label"] != "normal" for a in manuals)
                output.append({**{k: v for k, v in injection.items() if k != "changed_rows"},
                    "injection_changed_rows_total": injection.get("changed_rows", ""),
                    "start": iso_time(start), "end": iso_time(stop - 1),
                    "reference_origin": "controlled_injection", "intervention_known": "true",
                    "evaluable": "true" if flag(injection.get("evaluable")) and not conflict else "false",
                    "annotation_references": references, "excluded_source_rows": "[]",
                    "evaluation_note": "La revisión actual contradice la normalidad de la base. Intervención conservada, fuera de métricas." if conflict else "Intervención controlada sobre la base documentada."})
                # With duplicate timestamps, other records retain their manual label.
                if injection.get("source_row") not in (None, ""):
                    excluded_rows = [int(injection["source_row"])]
                else:
                    continue
            if not manuals:
                continue
            point_labels = defaultdict(list)
            for annotation in manuals:
                if annotation.get("source_row") is not None:
                    point_labels[int(annotation["source_row"])].append(annotation)
            for row, labels in sorted(point_labels.items()):
                if row not in excluded_rows:
                    output.append(manual_segment(labels, entity, start, stop, source_row=row))
            intervals = [a for a in manuals if a.get("source_row") is None]
            if intervals:
                output.append(manual_segment(intervals, entity, start, stop, excluded_rows=sorted(set(excluded_rows) | set(point_labels))))
    return output


FAULT_NAMES = {"bias": "Desplazamiento", "drift": "Deriva", "noise": "Ruido", "spike": "Pico",
               "stuck": "Valor congelado", "unavailable": "Indisponibilidad", "state_replace": "Estado sustituido"}


class RowLabels:
    """Fast point/interval lookup for the requested labelled CSV, in any source order."""
    def __init__(self, annotations):
        self.points = {}
        self.intervals = defaultdict(list)
        for row in compose_truth(annotations, []):
            if row.get("source_row") not in (None, ""):
                self.points[int(row["source_row"])] = row
            else:
                self.intervals[row["entity_id"]].append((parse_time(row["start"])[0], parse_time(row["end"])[0], row))
        self.starts = {entity: [item[0] for item in rows] for entity, rows in self.intervals.items()}

    def values(self, source_row, entity, timestamp, change=None, inherited=None):
        if change:
            return ["true", "Falla", FAULT_NAMES.get(change["kind"], change["kind"])]
        row = self.points.get(source_row)
        if row is None and entity in self.intervals:
            from bisect import bisect_right
            try:
                at = parse_time(timestamp)[0]
                index = bisect_right(self.starts[entity], at) - 1
                if index >= 0:
                    _, end, candidate = self.intervals[entity][index]
                    if at <= end and source_row not in json.loads(candidate["excluded_source_rows"]):
                        row = candidate
            except ValueError:
                pass
        synthetic = inherited["es_sintetica"] if inherited else "false"
        if row is None and inherited:
            return [synthetic, inherited["tipo"], inherited["tipo_falla"]]
        if not row or not flag(row.get("evaluable")):
            return [synthetic, "NULL", "NULL"]
        label = {"normal": "normal", "environmental_anomaly": "anomalia", "sensor_fault": "Falla"}.get(row["label"], "NULL")
        fault = FAULT_NAMES.get(row.get("kind"), row.get("kind")) if label == "Falla" else None
        return [synthetic, label, fault or "NULL"]
