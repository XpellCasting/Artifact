"""Configuración y ejecución reproducible de flujos de filtros.

El módulo no sustituye al pipeline histórico.  Ofrece un laboratorio aislado
para componer los mismos filtros sin compartir su estado ni sus resultados con
la reproducción convencional.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from typing import Any

import pandas as pd

from app.services.estados_validos import valid_state


NODE_TYPES = {"input", "statistical_focus", "contextual", "classifier", "merge", "decision", "output"}
SIGNALS = {"positive", "negative", "unknown"}
CONTINUOUS_METHODS = {"residual_focus_v1"}


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


def _node(key, kind, x, y=180, **config):
    return {"id": key, "type": kind, "position": {"x": x, "y": y},
            "config": {"evidence_policy": "independent", **config}}


def _edge(source, target, *signals):
    edge = {"id": f"{source}-{target}", "source": source, "target": target}
    if signals:
        edge["when"] = list(signals)
    return edge


def _statistical_node(x, y=180):
    return _node("statistical", "statistical_focus", x, y, suspicion_threshold=0.6,
                 continuous_method="residual_focus_v1", signal_policy="detection_before_suppression")


def default_flow() -> dict[str, Any]:
    return {
        "name": "Tesis · Secuencial",
        "nodes": [_node("input", "input", 0), _statistical_node(260),
                  _node("contextual", "contextual", 520), _node("classifier", "classifier", 780),
                  _node("output", "output", 1040, verdict_policy="sequential_v1")],
        "edges": [_edge("input", "statistical"), _edge("statistical", "contextual", "positive"),
                  _edge("statistical", "output", "negative", "unknown"),
                  _edge("contextual", "output", "negative"),
                  _edge("contextual", "classifier", "positive", "unknown"), _edge("classifier", "output")],
    }


def thesis_flows():
    parallel = {
        "name": "Tesis · Paralelo",
        "nodes": [_node("input", "input", 0), _statistical_node(260, 50),
                  _node("contextual", "contextual", 260, 310),
                  _node("scores", "merge", 520, mode="any", label="Reunir puntajes"),
                  _node("classifier", "classifier", 780, verdict_policy="parallel_v1"),
                  _node("output", "output", 1040, verdict_policy="classifier_v1")],
        "edges": [_edge("input", "statistical"), _edge("input", "contextual"),
                  _edge("statistical", "scores"), _edge("contextual", "scores"),
                  _edge("scores", "classifier"), _edge("classifier", "output")],
    }
    hybrid = {
        "name": "Tesis · Híbrido",
        "nodes": [_node("input", "input", 0), _statistical_node(260),
                  _node("contextual", "contextual", 520, 50), _node("classifier", "classifier", 520, 310),
                  _node("output", "output", 780, verdict_policy="hybrid_v1", label="Fusión de veredictos")],
        "edges": [_edge("input", "statistical"), _edge("statistical", "output", "negative", "unknown"),
                  _edge("statistical", "contextual", "positive"), _edge("statistical", "classifier", "positive"),
                  _edge("contextual", "output"), _edge("classifier", "output")],
    }
    return [default_flow(), parallel, hybrid]


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
        if "verdict_policy" in config:
            policies = {"output": ("sequential_v1", "classifier_v1", "hybrid_v1"), "classifier": ("parallel_v1",)}
            if config["verdict_policy"] not in policies.get(node["type"], ()):
                errors.append("La regla de diagnóstico no corresponde al tipo de nodo.")
            if config.get("evidence_policy") != "independent":
                errors.append("La regla de diagnóstico requiere la evaluación propia de cada filtro.")
        if config.get("evidence_policy", "legacy") not in {"legacy", "independent", "combine_v1"}:
            errors.append("La política de evidencias debe ser independent o combine_v1.")
        if "enabled" in config and not isinstance(config["enabled"], bool):
            errors.append("La activación de un nodo debe ser booleana.")
        if node["type"] == "classifier" and config.get("model_id") is not None and (not isinstance(config["model_id"], str) or not config["model_id"].strip()):
            errors.append("El modelo del clasificador debe ser un identificador válido o quedar sin seleccionar.")
        if node["type"] in {"statistical_focus"}:
            threshold = config.get("suspicion_threshold", .6)
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not .01 <= threshold <= .99:
                errors.append("El umbral estadístico debe estar entre 0,01 y 0,99.")
            method = config.get("continuous_method", "residual_focus_v1")
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


