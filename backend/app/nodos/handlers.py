from __future__ import annotations
from collections import defaultdict
from copy import deepcopy
from typing import Any
from uuid import uuid4
import pandas as pd
from app.flow.definitions import _time, observation_valid
from app.flow.evidence import append_evidence, merge_evidence
from app.modulos.estadistico.disponibilidad import availability_detection

class NodeHandlers:
    def _with_evidence(self, item: dict[str, Any], node_id: str, signal: str, status: str, data: dict[str, Any]) -> dict[str, Any]:
        node = getattr(self, "_evidence_nodes", {}).get(node_id, {})
        policy = node.get("config", {}).get("evidence_policy", "legacy")
        if policy != "legacy":
            item = deepcopy(item)
            for entry in item.get("evidence", []):
                entry.setdefault("module_type", getattr(self, "_evidence_nodes", {}).get(entry.get("node_id"), {}).get("type", "unknown"))
            return append_evidence(item, node_id, node.get("type", "unknown"), signal, status, data,
                                   policy, getattr(self, "execution_id", ""))
        result = deepcopy(item)
        result["signal"] = signal
        result["evidence"] = [*result.get("evidence", []), {"node_id": node_id, "signal": signal, "status": status, "data": data}]
        return result
    def _statistical(self, node_id: str, incoming: list[dict[str, Any]], events: pd.DataFrame, entities: pd.DataFrame, config: dict[str, Any]) -> list[dict[str, Any]]:
        if not incoming:
            return []
        if any(item.get("sensor_kind") == "continuous" for item in incoming):
            raise ValueError("FOCuS requiere Desarrollo y eventos posteriores")
        threshold = float(config.get("suspicion_threshold", 0.6))
        result = deepcopy(self.statistical).analizar_historial(events, entities, suspicion_threshold=threshold)
        suspects = result.get("all_results", result.get("results", []))
        by_raw = {str(item.get("source_raw_index")): item for item in suspects if item.get("source_raw_index") is not None}
        by_point = {(str(item.get("entity_id")), _time(item.get("timestamp"))): item for item in suspects if item.get("source_raw_index") is None}
        duplicate_points = set(events.loc[events.duplicated(["entity_id", "timestamp_dt"], keep=False)].apply(lambda row: (str(row["entity_id"]), _time(row["timestamp_dt"])), axis=1))
        output = []
        for item in incoming:
            availability = availability_detection(item)
            if availability is not None:
                output.append(self._with_evidence(item, node_id, "positive", "suspicion", availability))
                continue
            if not observation_valid(item):
                output.append(self._with_evidence(item, node_id, "unknown", "not_evaluable", {"reason": "invalid_state"}))
                continue
            entity_rows = events[events["entity_id"].astype(str).eq(str(item["entity_id"]))]
            valid_rows = sum(observation_valid(row) for row in entity_rows.to_dict("records"))
            kind = item.get("sensor_kind")
            minimum = (
                getattr(getattr(self.statistical, "continuos", None), "min_history_events", 12)
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
        independent = getattr(self, "_evidence_nodes", {}).get(node_id, {}).get("config", {}).get("evidence_policy", "legacy") != "legacy"
        for item in incoming:
            has_statistical = not independent and any(e.get("node_id") and e.get("status") == "suspicion" and e.get("signal") == "positive" for e in item.get("evidence", []))
            requests.append({**item, "is_suspicious": True, "passed_to_contextual_filter": True, "suspicion_score": next((e["data"].get("suspicion_score") for e in reversed(item.get("evidence", [])) if isinstance(e.get("data"), dict) and e["data"].get("suspicion_score") is not None), None), "_flow_has_statistical": has_statistical})
            if independent:
                requests[-1]["suspicion_score"] = None
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
                # Only top-level presentation fields change here. The evidence
                # envelope below still makes its own isolated deep copy.
                evidence = dict(evidence)
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
            if status == "environmental_anomaly" and getattr(self, "_evidence_nodes", {}).get(node_id, {}).get("config", {}).get("evidence_policy", "legacy") != "legacy":
                signal = "negative"
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
            base["evidence"] = merge_evidence(items)
            output.append(self._with_evidence(base, node_id, signal, f"merge_{mode}", {"signals": signals, "missing_branches": missing}))
        return output
    def _decision(self, node_id: str, incoming: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
        accepted = set(config.get("accept", ["positive"]))
        return [self._with_evidence(item, node_id, "positive" if item.get("signal") in accepted else "negative" if item.get("signal") in {"positive", "negative"} else "unknown", "decision", {"accepted": sorted(accepted)}) for item in incoming]
    @staticmethod
    def _final_decision(item: dict[str, Any]) -> str:
        if item.get("evidence_policy") == "combine_v1":
            label = item.get("combined_evaluation", {}).get("label", "indeterminate")
            return {"sensor_fault": "fault_compatible", "environmental_anomaly": "environmental",
                    "normal": "normal", "no_alert": "no_alert", "suspicion": "suspicion"}.get(label, "indeterminate")
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
