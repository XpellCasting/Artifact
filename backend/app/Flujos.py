"""Configuración y ejecución reproducible de flujos de filtros.

El módulo no sustituye al pipeline histórico.  Ofrece un laboratorio aislado
para componer los mismos filtros sin compartir su estado ni sus resultados con
la reproducción convencional.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from time import perf_counter
from typing import Any
from uuid import uuid4

import pandas as pd

from .Estados_Validos import valid_state


NODE_TYPES = {"input", "statistical", "statistical_focus", "contextual", "classifier", "merge", "decision", "output"}
SIGNALS = {"positive", "negative", "unknown"}
CONTINUOUS_METHODS = {"temporal_quantile_v1", "residual_ewma_v1"}


def _json(value: Any) -> str:
    return json.dumps(_clean(value), ensure_ascii=False, default=str, separators=(",", ":"), allow_nan=False)


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _observation_key(row: dict[str, Any], fallback: int) -> str:
    raw = row.get("_raw_index", fallback)
    return f"observation:{raw}"


def _time(value: Any) -> str:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return "" if pd.isna(parsed) else parsed.isoformat()


def observation_valid(item: dict[str, Any]) -> bool:
    if "is_valid_value" in item and not bool(item.get("is_valid_value")):
        return False
    if not valid_state(item.get("state")):
        return False
    if item.get("sensor_kind") == "continuous":
        try:
            return math.isfinite(float(item.get("value")))
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def default_flow() -> dict[str, Any]:
    return {
        "name": "Flujo secuencial de referencia",
        "nodes": [
            {"id": "input", "type": "input", "position": {"x": 30, "y": 180}, "config": {}},
            {"id": "statistical", "type": "statistical", "position": {"x": 260, "y": 180}, "config": {"suspicion_threshold": 0.6, "continuous_method": "temporal_quantile_v1"}},
            {"id": "contextual", "type": "contextual", "position": {"x": 510, "y": 180}, "config": {}},
            {"id": "classifier", "type": "classifier", "position": {"x": 760, "y": 180}, "config": {}},
            {"id": "output", "type": "output", "position": {"x": 1010, "y": 180}, "config": {}},
        ],
        "edges": [
            {"id": "input-statistical", "source": "input", "target": "statistical"},
            {"id": "statistical-contextual", "source": "statistical", "target": "contextual", "when": ["positive", "unknown"]},
            {"id": "statistical-output", "source": "statistical", "target": "output", "when": ["negative"]},
            {"id": "contextual-classifier", "source": "contextual", "target": "classifier", "when": ["positive", "unknown"]},
            {"id": "contextual-output", "source": "contextual", "target": "output", "when": ["negative"]},
            {"id": "classifier-output", "source": "classifier", "target": "output"},
        ],
    }


def validate_flow(payload: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {"valid": False, "errors": ["El flujo debe ser un objeto."], "order": []}
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not nodes:
        return {"valid": False, "errors": ["El flujo requiere nodos."], "order": []}
    if not isinstance(edges, list):
        return {"valid": False, "errors": ["Las conexiones deben ser una lista."], "order": []}
    ids = [str(item.get("id", "")) for item in nodes if isinstance(item, dict)]
    if len(ids) != len(nodes) or not all(ids) or len(set(ids)) != len(ids):
        errors.append("Cada nodo requiere un identificador único.")
    if any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in nodes):
        return {"valid": False, "errors": ["Cada nodo requiere un identificador de texto."], "order": []}
    kinds = {str(item.get("id")): item.get("type") for item in nodes if isinstance(item, dict)}
    invalid = [identifier for identifier, kind in kinds.items() if not isinstance(kind, str) or kind not in NODE_TYPES]
    if invalid:
        return {"valid": False, "errors": ["Hay tipos de nodo no reconocidos: " + ", ".join(invalid)], "order": []}
    if sum(kind == "input" for kind in kinds.values()) != 1:
        errors.append("El flujo requiere exactamente una Entrada.")
    if sum(kind == "output" for kind in kinds.values()) != 1:
        errors.append("El flujo requiere exactamente una Salida.")
    if "output" in kinds and kinds["output"] != "output":
        errors.append("El identificador output está reservado para la Salida.")
    for node in nodes:
        config = node.get("config", {})
        if not isinstance(config, dict):
            errors.append(f"La configuración de {node['id']} debe ser un objeto.")
            continue
        if "enabled" in config and not isinstance(config["enabled"], bool):
            errors.append("La activación de un nodo debe ser booleana.")
        if node["type"] in {"statistical", "statistical_focus"}:
            threshold = config.get("suspicion_threshold", .6)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not .01 <= threshold <= .99:
                errors.append("El umbral estadístico debe estar entre 0,01 y 0,99.")
            method = "residual_focus_v1" if node["type"] == "statistical_focus" else config.get("continuous_method", "temporal_quantile_v1")
            if method not in {*CONTINUOUS_METHODS, "residual_focus_v1"}:
                errors.append("El método para continuos no es reconocido.")
            if node["type"] == "statistical_focus" and "continuous_method" in config and config["continuous_method"] != "residual_focus_v1":
                errors.append("Estadístico FOCuS usa residual_focus_v1 de forma fija.")
        if node["type"] == "merge" and config.get("mode", "any") not in ("any", "all"):
            errors.append("La unión debe usar cualquiera (any) o todas (all).")
        if node["type"] == "decision":
            accept = config.get("accept", ["positive"])
            if not isinstance(accept, list) or not accept or any(not isinstance(item, str) or item not in SIGNALS for item in accept):
                errors.append("La decisión debe seleccionar al menos una señal válida.")
        position = node.get("position", {})
        if not isinstance(position, dict) or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in position.values()):
            errors.append("Las coordenadas de los nodos deben ser números finitos.")
    incoming: dict[str, int] = defaultdict(int)
    outgoing: dict[str, list[str]] = defaultdict(list)
    pairs = set()
    for edge in edges:
        if not isinstance(edge, dict):
            errors.append("Cada conexión debe ser un objeto.")
            continue
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in kinds or target not in kinds:
            errors.append("Una conexión referencia un nodo inexistente.")
            continue
        if source == target:
            errors.append("Un nodo no puede conectarse consigo mismo.")
        if (source, target) in pairs:
            errors.append("No puede repetirse una conexión entre los mismos nodos.")
        pairs.add((source, target))
        if kinds[source] == "output":
            errors.append("Salida no admite conexiones salientes.")
        when = edge.get("when")
        if "when" in edge and (not isinstance(when, list) or not when or any(not isinstance(item, str) or item not in SIGNALS for item in when)):
            errors.append("Las condiciones de una conexión deben usar positive, negative o unknown.")
        incoming[target] += 1
        outgoing[source].append(target)
    input_id = next((key for key, value in kinds.items() if value == "input"), None)
    if input_id and incoming[input_id]:
        errors.append("Entrada no admite conexiones entrantes.")
    for identifier, kind in kinds.items():
        if kind != "input" and not incoming[identifier]:
            errors.append(f"El nodo {identifier} no tiene entrada.")
        if kind != "output" and not outgoing[identifier]:
            errors.append(f"El nodo {identifier} no tiene salida.")
        if incoming[identifier] > 1 and kind not in {"merge", "output"}:
            errors.append(f"Usa una Unión antes de {identifier} para reunir ramas.")
    indegree = {identifier: incoming[identifier] for identifier in kinds}
    ready = sorted(identifier for identifier, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while ready:
        identifier = ready.pop(0)
        order.append(identifier)
        for target in outgoing[identifier]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
                ready.sort()
    if len(order) != len(kinds):
        errors.append("El flujo no puede contener ciclos.")
    return {"valid": not errors, "errors": errors, "order": order}


class FlowStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._database() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS flows (
                    flow_id TEXT PRIMARY KEY, name TEXT NOT NULL, payload TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_runs (
                    run_id TEXT PRIMARY KEY, flow_id TEXT NOT NULL, snapshot TEXT NOT NULL, summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS flow_results (
                    run_id TEXT NOT NULL, node_id TEXT NOT NULL, observation_id TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY (run_id, node_id, observation_id)
                );
                CREATE INDEX IF NOT EXISTS flow_results_page ON flow_results(run_id, node_id, sequence);
            """)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    @contextmanager
    def _database(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        checked = validate_flow(payload)
        if not checked["valid"]:
            raise ValueError("; ".join(checked["errors"]))
        flow_id = uuid4().hex
        item = {"flow_id": flow_id, "name": str(payload.get("name") or "Flujo sin nombre"), "nodes": payload["nodes"], "edges": payload["edges"]}
        with self._database() as db:
            db.execute("INSERT INTO flows VALUES (?,?,?,?)", (flow_id, item["name"], _json(item), _now()))
        return item

    def list(self) -> list[dict[str, Any]]:
        with self._database() as db:
            rows = db.execute("SELECT payload,updated_at FROM flows ORDER BY updated_at DESC").fetchall()
        return [{**json.loads(payload), "updated_at": updated} for payload, updated in rows]

    def get(self, flow_id: str) -> dict[str, Any]:
        with self._database() as db:
            row = db.execute("SELECT payload,updated_at FROM flows WHERE flow_id=?", (flow_id,)).fetchone()
        if not row:
            raise KeyError(flow_id)
        return {**json.loads(row[0]), "updated_at": row[1]}

    def update(self, flow_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.get(flow_id)
        checked = validate_flow(payload)
        if not checked["valid"]:
            raise ValueError("; ".join(checked["errors"]))
        item = {"flow_id": flow_id, "name": str(payload.get("name") or "Flujo sin nombre"), "nodes": payload["nodes"], "edges": payload["edges"]}
        updated_at = _now()
        with self._database() as db:
            db.execute("UPDATE flows SET name=?,payload=?,updated_at=? WHERE flow_id=?", (item["name"], _json(item), updated_at, flow_id))
        return {**item, "updated_at": updated_at}

    def save_run(self, flow_id: str, snapshot: dict[str, Any], summary: dict[str, Any], results: dict[str, list[dict[str, Any]]]) -> str:
        run_id = uuid4().hex
        with self._database() as db:
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)", (run_id, flow_id, _json(snapshot), _json(summary), _now()))
            records = []
            for node_id, items in results.items():
                for sequence, item in enumerate(items):
                    records.append((run_id, node_id, str(item["observation_id"]), sequence, _json(item)))
            db.executemany("INSERT INTO flow_results VALUES (?,?,?,?,?)", records)
        return run_id

    def save_run_stream(self, flow_id: str, snapshot: dict[str, Any], summary: dict[str, Any], records) -> tuple[str, float]:
        """Persist a completed spool atomically without materializing its rows."""
        run_id = uuid4().hex
        started = perf_counter()
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            def values():
                for record in records:
                    yield (run_id, record["node_id"], record["observation_id"],
                           record["sequence"], _json(record["payload"]))
            db.executemany("INSERT INTO flow_results VALUES (?,?,?,?,?)", values())
            insert_ms = round((perf_counter() - started) * 1000, 3)
            performance = summary.setdefault("performance", {})
            phases = performance.setdefault("phases_ms", {})
            phases["result_persistence_before_commit"] = insert_ms
            phases["total_with_result_inserts"] = round(phases.get("total", 0) + insert_ms, 3)
            included = performance.setdefault("included_in", {})
            included["result_persistence_before_commit"] = "total_with_result_inserts"
            included["total_with_result_inserts"] = "evaluation and result insertion; excludes commit latency"
            db.execute("INSERT INTO flow_runs VALUES (?,?,?,?,?)",
                       (run_id, flow_id, _json(snapshot), _json(summary), _now()))
            db.commit()
            return run_id, round((perf_counter() - started) * 1000, 3)
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_runs(self, flow_id: str | None = None) -> list[dict[str, Any]]:
        clause, args = (" WHERE flow_id=?", [flow_id]) if flow_id else ("", [])
        with self._database() as db:
            rows = db.execute("SELECT run_id,flow_id,snapshot,summary,created_at FROM flow_runs" + clause + " ORDER BY created_at DESC", args).fetchall()
        return [{"run_id": row[0], "flow_id": row[1], "snapshot": json.loads(row[2]), "summary": json.loads(row[3]), "created_at": row[4]} for row in rows]

    def results(self, run_id: str, node_id: str = "output", offset: int = 0, limit: int = 100) -> dict[str, Any]:
        with self._database() as db:
            if not db.execute("SELECT 1 FROM flow_runs WHERE run_id=?", (run_id,)).fetchone():
                raise KeyError(run_id)
            total = db.execute("SELECT COUNT(*) FROM flow_results WHERE run_id=? AND node_id=?", (run_id, node_id)).fetchone()[0]
            rows = db.execute("SELECT payload FROM flow_results WHERE run_id=? AND node_id=? ORDER BY sequence LIMIT ? OFFSET ?", (run_id, node_id, limit, offset)).fetchall()
        items = [json.loads(row[0]) for row in rows]
        return {"run_id": run_id, "node_id": node_id, "total": total, "offset": offset, "items": items, "next_offset": offset + len(items) if offset + len(items) < total else None}


@dataclass
class FlowRunner:
    statistical: Any
    contextual: Any
    diagnostic: Any

    @staticmethod
    def compile(flow: dict[str, Any]) -> dict[str, Any]:
        checked = validate_flow(flow)
        if not checked["valid"]:
            raise ValueError("; ".join(checked["errors"]))
        nodes = {node["id"]: node for node in flow["nodes"]}
        edges_from: dict[str, list[dict[str, Any]]] = defaultdict(list)
        edges_to: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in flow["edges"]:
            edges_from[edge["source"]].append(edge); edges_to[edge["target"]].append(edge)
        return {"checked": checked, "nodes": nodes, "edges_from": edges_from, "edges_to": edges_to,
                "input_id": next(node_id for node_id, node in nodes.items() if node["type"] == "input")}

    def run(self, flow: dict[str, Any], events: pd.DataFrame, entities: pd.DataFrame, context_map: dict[str, Any], candidate_limit: int = 1000, *, candidate_events=None, source_id=None, compiled=None, events_sorted=False, include_snapshot=True) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
        compiled = compiled or self.compile(flow)
        checked = compiled["checked"]
        if events is None or events.empty:
            raise ValueError("No hay histórico cargado para ejecutar el flujo.")
        limit = max(1, min(int(candidate_limit), 20000))
        ordered = events if events_sorted else events.copy()
        if candidate_events is None and ("_raw_index" not in ordered or ordered["_raw_index"].isna().any() or ordered["_raw_index"].duplicated().any()):
            ordered["_raw_index"] = range(len(ordered))
        if not events_sorted:
            ordered = ordered.sort_values(["timestamp_dt", "entity_id", "_raw_index"], kind="stable")
        sample = ordered.tail(limit).copy() if candidate_events is None else candidate_events
        source_columns = [column for column in ("entity_id", "timestamp_dt", "state", "sensor_kind", "_raw_index") if column in ordered]
        source_hash = source_id or hashlib.sha256(pd.util.hash_pandas_object(ordered[source_columns], index=False).values.tobytes()).hexdigest()
        candidates = []
        for ordinal, row in enumerate(sample.to_dict("records")):
            observation_id = source_hash + ":" + _observation_key(row, ordinal)
            candidates.append(_clean({
                "observation_id": observation_id, "entity_id": str(row.get("entity_id", "")),
                "timestamp": _time(row.get("timestamp_dt", row.get("timestamp"))),
                "sensor_kind": row.get("sensor_kind"), "state": str(row.get("state", "")),
                "value": row.get("value"), "source_raw_index": row.get("_raw_index"),
                "is_valid_value": observation_valid(row),
                "signal": "unknown", "evidence": [],
            }))
        nodes, edges_from, edges_to = compiled["nodes"], compiled["edges_from"], compiled["edges_to"]
        delivered: dict[str, list[dict[str, Any]]] = defaultdict(list)
        outputs: dict[str, list[dict[str, Any]]] = {}
        input_id = compiled["input_id"]
        delivered[input_id] = candidates
        started = perf_counter()
        node_timings: dict[str, float] = {}
        for node_id in checked["order"]:
            node = nodes[node_id]
            incoming = delivered[node_id]
            node_started = perf_counter()
            if node["type"] == "input":
                produced = incoming
            elif node.get("config", {}).get("enabled") is False:
                produced = [self._with_evidence(item, node_id, item.get("signal", "unknown"), "omitted", {}) for item in incoming]
            elif node["type"] in {"statistical", "statistical_focus"}:
                produced = self._statistical(node_id, incoming, ordered, entities, node.get("config", {}))
            elif node["type"] == "contextual":
                produced = self._contextual(node_id, incoming, ordered, entities, context_map)
            elif node["type"] == "classifier":
                produced = self._classifier(node_id, incoming, ordered)
            elif node["type"] == "merge":
                produced = self._merge(node_id, incoming, {**node.get("config", {}), "expected_inputs": len(edges_to[node_id])})
            elif node["type"] == "decision":
                produced = self._decision(node_id, incoming, node.get("config", {}))
            else:
                produced = self._merge(node_id, incoming, {"mode": "output"})
            node_timings[node_id] = round((perf_counter() - node_started) * 1000, 3)
            outputs[node_id] = produced
            for edge in edges_from[node_id]:
                allowed = set(edge.get("when", SIGNALS))
                delivered[edge["target"]].extend(item for item in produced if item.get("signal") in allowed)
        final = []
        for node_id, node in nodes.items():
            if node["type"] == "output":
                final.extend(outputs.get(node_id, []))
        final = [deepcopy(item) for item in final]
        seen = {item["observation_id"] for item in final}
        # Una ruta que descarta un candidato debe seguir siendo auditable.
        for candidate in candidates:
            if candidate["observation_id"] not in seen:
                last = next((item for node_id in reversed(checked["order"]) for item in outputs.get(node_id, []) if item["observation_id"] == candidate["observation_id"]), candidate)
                final.append({**deepcopy(last), "final_decision": "not_routed"})
        for item in final:
            item.setdefault("final_decision", self._final_decision(item))
        outputs["output"] = final
        counts = Counter(item.get("final_decision") for item in final)
        summary = {"candidate_count": len(candidates), "output_count": len(final), "by_final_decision": dict(counts), "node_counts": {key: len(value) for key, value in outputs.items()}, "node_timings_ms": node_timings, "elapsed_ms": round((perf_counter() - started) * 1000, 3)}
        snapshot = {"flow": deepcopy(flow), "execution_mode": "retrospective", "candidate_limit": limit, "source": {"rows": len(events), "sample": "latest_chronological_rows", "sample_rows": len(sample), "sample_start": _time(sample["timestamp_dt"].min()), "sample_end": _time(sample["timestamp_dt"].max()), "fingerprint": source_hash}, "context_map_version": context_map.get("map_version"), "created_at": _now()} if include_snapshot else {}
        return summary, outputs, snapshot

    @staticmethod
    def _with_evidence(item: dict[str, Any], node_id: str, signal: str, status: str, data: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(item)
        result["signal"] = signal
        result["evidence"] = [*result.get("evidence", []), {"node_id": node_id, "signal": signal, "status": status, "data": data}]
        return result

    def _statistical(self, node_id: str, incoming: list[dict[str, Any]], events: pd.DataFrame, entities: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
        if not incoming:
            return []
        if any(item.get("sensor_kind") == "continuous" for item in incoming):
            raise ValueError("El método temporal requiere Desarrollo y eventos posteriores")
        threshold = float(config.get("suspicion_threshold", 0.6))
        result = deepcopy(self.statistical).analizar_historial(events, entities, suspicion_threshold=threshold)
        suspects = result.get("all_results", result.get("results", []))
        by_raw = {str(item.get("source_raw_index")): item for item in suspects if item.get("source_raw_index") is not None}
        by_point = {(str(item.get("entity_id")), _time(item.get("timestamp"))): item for item in suspects if item.get("source_raw_index") is None}
        duplicate_points = set(events.loc[events.duplicated(["entity_id", "timestamp_dt"], keep=False)].apply(lambda row: (str(row["entity_id"]), _time(row["timestamp_dt"])), axis=1))
        output = []
        for item in incoming:
            if not observation_valid(item):
                output.append(self._with_evidence(item, node_id, "unknown", "not_evaluable", {"reason": "invalid_state"}))
                continue
            entity_rows = events[events["entity_id"].astype(str).eq(str(item["entity_id"]))]
            valid_rows = sum(observation_valid(row) for row in entity_rows.to_dict("records"))
            kind = item.get("sensor_kind")
            minimum = (
                getattr(getattr(self.statistical, "continuos", None), "window_hampel", 12)
                if kind == "continuous"
                else getattr(getattr(self.statistical, "binarios" if kind == "binary" else "multivalor", None), "min_history_events", 24)
            )
            if hasattr(self.statistical, "continuos") and valid_rows < minimum:
                output.append(self._with_evidence(item, node_id, "unknown", "not_evaluable", {"reason": "missing_baseline"}))
                continue
            point = (item["entity_id"], item["timestamp"])
            evidence = by_raw.get(str(item.get("source_raw_index"))) or (by_point.get(point) if point not in duplicate_points else None)
            output.append(self._with_evidence(item, node_id, "positive" if evidence else "negative", "suspicion" if evidence else "no_suspicion", evidence or {}))
        return output

    def _contextual(self, node_id: str, incoming: list[dict[str, Any]], events: pd.DataFrame, entities: pd.DataFrame, context_map: dict[str, Any]) -> list[dict[str, Any]]:
        if not incoming:
            return []
        requests = []
        for item in incoming:
            has_statistical = any(e.get("node_id") and e.get("status") == "suspicion" and e.get("signal") == "positive" for e in item.get("evidence", []))
            requests.append({**item, "is_suspicious": True, "passed_to_contextual_filter": True, "suspicion_score": next((e["data"].get("suspicion_score") for e in reversed(item.get("evidence", [])) if isinstance(e.get("data"), dict) and e["data"].get("suspicion_score") is not None), None), "_flow_has_statistical": has_statistical})
        requests_by_id = {item["observation_id"]: item for item in requests}
        causal_options = {"concurrence_state": self.context_states.setdefault(node_id, {})} if hasattr(self, "context_states") else {}
        prepared_state = getattr(self, "context_sessions", {}).get(node_id)
        if prepared_state is not None:
            causal_options["prepared_state"] = prepared_state
        raw = self.contextual.analizar_causal(requests, events, entities, context_map, accept_all=True, **causal_options)
        indexed = {str(item.get("observation_id")): item for item in raw.get("results", [])}
        output = []
        for item in incoming:
            evidence = indexed.get(item["observation_id"], {})
            outcome = evidence.get("context_outcome")
            signal = "positive" if outcome == "contradicted" else "negative" if outcome == "supported" else "unknown"
            if evidence.get("map_quality", {}).get("stale") is not False:
                signal = "unknown"
            if not requests_by_id[item["observation_id"]].get("_flow_has_statistical"):
                evidence = deepcopy(evidence)
                evidence["contextual_class"] = None
                evidence["alert"] = None
                evidence["statistical_score"] = None
                evidence["adjusted_suspicion_score"] = None
                evidence["routing"] = {"terminal": False, "destination": "flow", "send_to_classifier": True, "reason": "standalone_contextual_evidence"}
            output.append(self._with_evidence(item, node_id, signal, outcome or "not_evaluable", evidence))
        return output

    def _classifier(self, node_id: str, incoming: list[dict[str, Any]], events: pd.DataFrame) -> list[dict[str, Any]]:
        if not incoming:
            return []
        contexts, statistical = [], []
        for item in incoming:
            identity = {key: item.get(key) for key in ("observation_id", "entity_id", "timestamp", "sensor_kind", "source_raw_index")}
            f1 = next((e for e in reversed(item.get("evidence", [])) if e.get("status") in {"suspicion", "no_suspicion"}), None)
            f2 = next((e.get("data", {}) for e in reversed(item.get("evidence", [])) if "context_outcome" in e.get("data", {})), {})
            contexts.append({**deepcopy(f2), **identity, "event_id": item["observation_id"], "routing": {"terminal": False}})
            candidate = {**(deepcopy(f1.get("data", {})) if f1 else {}), **identity}
            if f1 is not None:
                candidate["is_suspicious"] = f1["signal"] == "positive"
            statistical.append(candidate)
        run_id = f"{self.execution_id}:{node_id}" if hasattr(self, "execution_id") else uuid4().hex
        session = getattr(self, "diagnostic_sessions", {}).get(node_id)
        enriched = self.diagnostic.enrich_contextual(statistical, contexts, events=None if session is not None else events, source="flow_evaluation", run_id=run_id, session=session)
        indexed = {str(item.get("observation_id")): item.get("diagnosis", {}) for item in enriched}
        output = []
        for item in incoming:
            diagnosis = indexed.get(item["observation_id"], {})
            status = diagnosis.get("status")
            signal = "positive" if status == "compatible" else "negative" if status == "no_fault_evidence" else "unknown"
            output.append(self._with_evidence(item, node_id, signal, status or "indeterminate", diagnosis))
        return output

    def _merge(self, node_id: str, incoming: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in incoming:
            grouped[item["observation_id"]].append(item)
        mode = config.get("mode", "any")
        output = []
        for items in grouped.values():
            signals = [item.get("signal", "unknown") for item in items]
            missing = max(0, config.get("expected_inputs", len(items)) - len(items))
            signals.extend(["unknown"] * missing)
            if mode == "output":
                signal = signals[0] if len(set(signals)) == 1 else "unknown"
            elif mode == "all":
                signal = "negative" if "negative" in signals else "positive" if signals and all(value == "positive" for value in signals) else "unknown"
            else:
                signal = "positive" if "positive" in signals else "negative" if signals and all(value == "negative" for value in signals) else "unknown"
            base = deepcopy(items[0])
            base["evidence"] = list({entry["node_id"]: entry for item in items for entry in item.get("evidence", [])}.values())
            output.append(self._with_evidence(base, node_id, signal, f"merge_{mode}", {"signals": signals, "missing_branches": missing}))
        return output

    def _decision(self, node_id: str, incoming: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        accepted = set(config.get("accept", ["positive"]))
        return [self._with_evidence(item, node_id, "positive" if item.get("signal") in accepted else "negative" if item.get("signal") in {"positive", "negative"} else "unknown", "decision", {"accepted": sorted(accepted)}) for item in incoming]

    @staticmethod
    def _final_decision(item: dict[str, Any]) -> str:
        evidence = item.get("evidence", [])
        evaluated = [entry for entry in evidence if entry.get("status") not in {"omitted", "decision", "merge_all", "merge_any", "merge_output"}]
        if not evaluated or item.get("signal") == "unknown":
            return "indeterminate"
        environmental = any(entry.get("data", {}).get("contextual_class") == "environmental_anomaly" for entry in evaluated)
        fault = any(entry.get("data", {}).get("status") == "compatible" for entry in evaluated)
        if environmental and fault:
            return "indeterminate"
        if environmental and item.get("signal") == "negative":
            return "environmental"
        diagnosis = next((entry.get("data", {}) for entry in reversed(evidence) if entry.get("status") in {"compatible", "no_fault_evidence", "indeterminate"}), None)
        if isinstance(diagnosis, dict) and diagnosis.get("status") == "compatible" and item.get("signal") == "positive":
            return "fault_compatible"
        if isinstance(diagnosis, dict) and diagnosis.get("status") == "no_fault_evidence" and item.get("signal") == "negative":
            return "normal"
        if item.get("signal") == "positive":
            return "suspicion"
        if item.get("signal") == "negative":
            return "no_alert"
        return "indeterminate"
