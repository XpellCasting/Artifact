"""Ground truth stays outside detector inputs; metrics count unique source rows."""
import csv
import io
import sqlite3
import math
from collections import defaultdict
from contextlib import contextmanager, closing
from .CSV_Etiquetado import labelled_columns, read_label
from .Revision_Fragmentos import parse_time
from .Estados_Validos import valid_state


@contextmanager
def database(path):
    with closing(sqlite3.connect(path, timeout=30)) as db:
        with db:
            yield db


def separate_csv(content):
    text = content.decode("utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text, newline=""), dialect)
    headers = [h.strip().lower() for h in next(reader, [])]
    columns = labelled_columns(headers)
    if columns is None or not all(h in headers for h in ("entity_id", "state", "last_changed")):
        raise ValueError("Se requieren entity_id,state,last_changed,es_sintetica,tipo,tipo_falla.")
    positions = [headers.index(h) for h in ("entity_id", "state", "last_changed")]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["entity_id", "state", "last_changed"])
    labels = []
    for index, raw in enumerate(reader):
        try:
            if len(raw) != len(headers):
                raise ValueError("Cantidad de columnas incorrecta.")
            values = [raw[i] for i in positions]
            if not values[0].strip():
                raise ValueError("Entidad vacía.")
            parse_time(values[2])
            label = read_label(raw, columns)
        except ValueError as exc:
            raise ValueError(f"Fila {index + 2}: {exc}") from exc
        labels.append((index, values[0].strip(), label["tipo"], label["es_sintetica"], label["fault_type"]))
        writer.writerow(values)
    if not labels:
        raise ValueError("El conjunto etiquetado está vacío.")
    return output.getvalue().encode("utf-8"), labels


def install(path, labels, events):
    with database(path) as db:
        db.execute("DROP TABLE IF EXISTS evaluation_truth")
        db.execute("CREATE TABLE evaluation_truth (raw_index INTEGER PRIMARY KEY, entity TEXT, tipo TEXT, synthetic TEXT, replay_index INTEGER, detected INTEGER DEFAULT 0, fault_type TEXT, statistical_evaluable INTEGER NULL, non_evaluable_reason TEXT NULL)")
        order = dict(zip(events["_raw_index"].astype(int), events["_replay_index"].astype(int)))
        db.executemany("INSERT INTO evaluation_truth(raw_index,entity,tipo,synthetic,replay_index,fault_type) VALUES (?,?,?,?,?,?)", ((*row[:4], order[row[0]], row[4] if len(row) > 4 else None) for row in labels))


def read_truth(path):
    with database(path) as db:
        db.row_factory = sqlite3.Row
        columns = {row[1] for row in db.execute("PRAGMA table_info(evaluation_truth)")}
        if not columns:
            return []
        fault = "fault_type" if "fault_type" in columns else "NULL AS fault_type"
        return [dict(row) for row in db.execute(f"SELECT raw_index,entity,tipo,synthetic,{fault} FROM evaluation_truth ORDER BY replay_index")]


def reset(path):
    with database(path) as db:
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='evaluation_truth'").fetchone():
            _migrate(db)
            db.execute("UPDATE evaluation_truth SET detected=0, statistical_evaluable=NULL, non_evaluable_reason=NULL")


def _migrate(db):
    columns = {row[1] for row in db.execute("PRAGMA table_info(evaluation_truth)")}
    if "statistical_evaluable" not in columns:
        db.execute("ALTER TABLE evaluation_truth ADD COLUMN statistical_evaluable INTEGER NULL")
    if "non_evaluable_reason" not in columns:
        db.execute("ALTER TABLE evaluation_truth ADD COLUMN non_evaluable_reason TEXT NULL")


def _event_evaluability(event, baseline):
    explicit = event.get("is_valid_value")
    valid = ("is_valid_value" not in event or bool(explicit)) and valid_state(event.get("state"))
    if valid and event.get("sensor_kind") == "continuous":
        try:
            valid = math.isfinite(float(event.get("value")))
        except (TypeError, ValueError, OverflowError):
            valid = False
    if not valid:
        return 0, "invalid_state"
    kind = str(event.get("sensor_kind", ""))
    if str(event.get("entity_id")) not in (baseline or {}).get(kind, {}):
        return 0, "missing_baseline"
    return 1, None


def record(path, results, batch, baseline=None):
    # Binary windows can lack a source index. Match only an unambiguous trigger;
    # never label every point in a window as detected.
    candidates = defaultdict(list)
    for event in batch.to_dict("records"):
        candidates[(str(event["entity_id"]), parse_time(str(event["timestamp"]))[0], str(event["state"]))].append(int(event["_raw_index"]))
    detected, unmatched = set(), 0
    for result in results:
        index = result.get("source_raw_index")
        if index is None:
            key = (str(result["entity_id"]), parse_time(str(result["timestamp"]))[0], str(result.get("observed_state", "")))
            matches = candidates.get(key, [])
            index = matches[0] if len(matches) == 1 else None
        if index is None:
            unmatched += 1
        else:
            detected.add(int(index))
    with database(path) as db:
        _migrate(db)
        db.executemany(
            "UPDATE evaluation_truth SET statistical_evaluable=?, non_evaluable_reason=? WHERE raw_index=?",
            ((*_event_evaluability(event, baseline), int(event["_raw_index"])) for event in batch.to_dict("records")),
        )
        db.executemany("UPDATE evaluation_truth SET detected=1 WHERE raw_index=?", ((index,) for index in detected))
    return unmatched


def summary(path, cursor):
    with database(path) as db:
        _migrate(db)
        db.row_factory = sqlite3.Row
        rows = db.execute("""SELECT entity AS entity_id, COUNT(*) AS total,
            SUM(replay_index < ?) AS processed, SUM(tipo='NULL') AS unlabelled,
            SUM(tipo='normal') AS normal,
            SUM(tipo='normal' AND detected=1) AS false_positives,
            SUM(tipo='Falla' AND synthetic='false') AS real_faults,
            SUM(tipo='Falla' AND synthetic='false' AND detected=1) AS real_detected,
            SUM(tipo='Falla' AND synthetic='true') AS synthetic_faults,
            SUM(tipo='Falla' AND synthetic='true' AND detected=1) AS synthetic_detected,
            SUM(tipo='anomalia') AS anomalies,
            SUM(tipo='anomalia' AND detected=1) AS anomalies_detected,
            SUM(replay_index < ? AND statistical_evaluable=1) AS evaluable,
            SUM(replay_index < ? AND statistical_evaluable=0) AS non_evaluable,
            SUM(replay_index < ? AND non_evaluable_reason='invalid_state') AS non_evaluable_invalid_state,
            SUM(replay_index < ? AND non_evaluable_reason='missing_baseline') AS non_evaluable_missing_baseline
            FROM evaluation_truth GROUP BY entity ORDER BY entity""", (cursor, cursor, cursor, cursor, cursor)).fetchall()
    return [dict(row) for row in rows]
