"""Split a labelled four-month history without exposing holdout rows to fitting."""
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.services.csv_etiquetado import labelled_columns, read_label


def month_start(at, offset=0):
    at = at.astimezone(timezone.utc)
    absolute = at.year * 12 + at.month - 1 + offset
    year, month = divmod(absolute, 12)
    return datetime(year, month + 1, 1, tzinfo=timezone.utc)


def temporal_plan(first, tune_days=14):
    start, boundary, end = month_start(first), month_start(first, 2), month_start(first, 4)
    tune = boundary - timedelta(days=tune_days)
    if not start < tune < boundary or tune_days < 4:
        raise ValueError("La selección interna requiere al menos cuatro días y debe quedar en los primeros dos meses")
    return {"version": "calendar_2_plus_2_v1", "timezone": "UTC",
            "development_start": start.isoformat(), "development_end": boundary.isoformat(),
            "train_fit_end": tune.isoformat(), "evaluation_start": boundary.isoformat(),
            "evaluation_end": end.isoformat(), "tune_days": tune_days}


def _rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        names = [name.strip().lower() for name in reader.fieldnames or []]
        if (len(set(names)) != len(names) or not {"entity_id", "state", "last_changed"}.issubset(names)
                or labelled_columns(names) is None):
            raise ValueError("Se requiere el CSV etiquetado: entity_id, state, last_changed, es_sintetica, tipo y tipo_falla. No se inventan etiquetas para un histórico de tres columnas.")
        for number, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Fila {number}: número de columnas inválido")
            row = dict(zip(names, raw.values()))
            if not row["entity_id"].strip():
                raise ValueError(f"Fila {number}: entidad vacía")
            at = parse_utc(row["last_changed"]).astimezone(timezone.utc)
            label = read_label([row[key] for key in ("es_sintetica", "tipo", "tipo_falla")], [0,1,2])
            yield number, names, row, at, label


def split_four_month_csv(path, output, *, tune_days=14, episode_gap_seconds=900):
    path, output = Path(path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("La carpeta de salida ya existe; usa una nueva para conservar resultados anteriores")
    if episode_gap_seconds < 1:
        raise ValueError("El intervalo de episodio debe ser positivo")
    original_sha = sha256_file(path)
    first = last = None
    months, current, positive_episodes = set(), {}, []
    for number, names, row, at, label in _rows(path):
        entity = row["entity_id"]
        first = min(first, at) if first else at
        last = max(last, at) if last else at
        months.add((at.year, at.month))
        signature = (label["label"], label["fault_type"], label["es_sintetica"])
        previous = current.get(entity)
        if previous and at < previous["end"]:
            raise ValueError(f"Fila {number}: las fechas de {entity} no están ordenadas")
        if (previous and previous["signature"] == signature
                and (at-previous["end"]).total_seconds() <= episode_gap_seconds):
            previous["end"] = at
        else:
            if previous and previous["signature"][0] in {"sensor_fault", "environmental_anomaly"}:
                positive_episodes.append(previous)
            current[entity] = dict(entity_id=entity, start=at, end=at, signature=signature)
    positive_episodes.extend(row for row in current.values()
                             if row["signature"][0] in {"sensor_fault", "environmental_anomaly"})
    if first is None:
        raise ValueError("El CSV está vacío")
    plan = temporal_plan(first, tune_days)
    start, boundary, end = (parse_utc(plan[key]) for key in ("development_start", "evaluation_start", "evaluation_end"))
    expected = {(month_start(first, offset).year, month_start(first, offset).month) for offset in range(4)}
    if months != expected or not start <= first <= last < end:
        raise ValueError("El CSV debe abarcar cuatro meses calendario consecutivos; revisa sus fechas")
    crossing = defaultdict(list)
    for episode in positive_episodes:
        if episode["start"] < boundary <= episode["end"]:
            crossing[episode["entity_id"]].append(episode)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, masked, labels = Counter(), 0, {"development":Counter(), "evaluation":Counter()}
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        with (temporary/"development.csv").open("w",encoding="utf-8",newline="") as development, \
                (temporary/"evaluation.csv").open("w",encoding="utf-8",newline="") as evaluation:
            writers = None
            for number, names, row, at, label in _rows(path):
                if writers is None:
                    writers = {key:csv.DictWriter(stream, fieldnames=names)
                               for key,stream in (("development",development),("evaluation",evaluation))}
                    for writer in writers.values():
                        writer.writeheader()
                part = "development" if at < boundary else "evaluation"
                if part == "development" and any(e["start"] <= at <= e["end"] for e in crossing[row["entity_id"]]):
                    # Preserve observations as causal context but exclude this
                    # abnormal episode from training labels on the earlier side.
                    row["tipo"] = row["tipo_falla"] = "NULL"
                    masked += 1
                writers[part].writerow(row)
                counts[part] += 1
                labels[part][row["tipo"]] += 1
        if sha256_file(path) != original_sha:
            raise ValueError("El CSV cambió durante la separación; no se publicaron los resultados")
        manifest = {"boundaries":[start.isoformat(),boundary.isoformat(),end.isoformat(),end.isoformat()],
                    "train_fit_end":plan["train_fit_end"],"partitions":{}}
        report = {**plan, "source_sha256":original_sha, "source_name":path.name,
                  "rows":dict(counts), "label_counts_raw":{key:dict(value) for key,value in labels.items()},
                  "crossing_abnormal_episodes":sum(map(len,crossing.values())),
                  "development_labels_excluded_at_holdout_boundary":masked,
                  "development_sha256":sha256_file(temporary/"development.csv"),
                  "evaluation_sha256":sha256_file(temporary/"evaluation.csv"),
                  "evaluation_labels_modified":False,
                  "episode_gap_seconds":episode_gap_seconds}
        for name,value in (("split_manifest.json",manifest),("split_report.json",report)):
            (temporary/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        os.rename(temporary,output)
    return report
