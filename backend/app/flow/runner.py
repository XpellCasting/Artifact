from __future__ import annotations
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
from time import perf_counter
from typing import Any
import pandas as pd
from app.flow.definitions import _clean, _now, _observation_key, _time, observation_valid, validate_flow
from app.nodos.handlers import NodeHandlers
from app.flow.definitions import SIGNALS

@dataclass
class FlowRunner(NodeHandlers):
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
        self._evidence_nodes = nodes
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
            elif node["type"] in {"statistical_focus", "contextual", "classifier"} and node.get("config", {}).get("evidence_policy", "legacy") != "legacy":
                produced = []
                groups = defaultdict(list)
                for item in incoming:
                    groups[item["timestamp"]].append(item)
                for timestamp in sorted(groups):
                    candidates_at = groups[timestamp]
                    history = ordered[ordered["timestamp_dt"] <= pd.Timestamp(timestamp)]
                    if node["type"] == "statistical_focus":
                        produced.extend(self._statistical(node_id, candidates_at, history, entities, node.get("config", {})))
                    elif node["type"] == "contextual":
                        produced.extend(self._contextual(node_id, candidates_at, history, entities, context_map))
                    else:
                        produced.extend(self._classifier(node_id, candidates_at, history))
            elif node["type"] in {"statistical_focus"}:
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


