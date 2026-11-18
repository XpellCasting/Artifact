"""Execution metadata alongside, rather than in place of, diagnostic evidence."""
from copy import deepcopy
import hashlib
import json
import math

from app.flow.definitions import _clean


TRACE_VERSION = "filter-execution-v1"
FILTER_TYPES = frozenset({"statistical_focus", "contextual", "classifier"})


def execution_context(flow, execution_id):
    """Hash the frozen graph/configuration, excluding presentation and storage IDs.

    Runtime model/baseline identities remain in the existing run snapshot and
    evidence. This hash identifies configuration, not model weights or input data.
    Array order is retained; object key order does not affect the digest.
    """
    configuration = {
        "nodes": [{"id": node["id"], "type": node["type"],
                   "config": {key: value for key, value in node.get("config", {}).items() if key != "label"}}
                  for node in flow["nodes"]],
        "edges": [{key: edge[key] for key in ("source", "target", "when") if key in edge}
                  for edge in flow["edges"]],
    }
    encoded = json.dumps(_clean(configuration), sort_keys=True, ensure_ascii=False,
                         default=str, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {"schema_version": TRACE_VERSION, "execution_id": execution_id,
            "flow_id": flow.get("flow_id"), "config_hash": hashlib.sha256(encoded).hexdigest()}


def execution_status(entry):
    """Abstaining with valid class scores is distinct from lacking input data."""
    data = entry.get("data") or {}
    status = entry.get("status")
    observation = data.get("statistical_observation") or {}
    scores = data.get("class_scores")
    has_scores = isinstance(scores, dict) and bool(scores) and all(
        type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1 for value in scores.values())
    if observation.get("evaluable") is False:
        return "insufficient_data", observation.get("reason") or "not_evaluable"
    if status in {"not_evaluable", "insufficient_data"} or (
            entry.get("signal") == "unknown" and not has_scores):
        reasons = data.get("reasons") or []
        return "insufficient_data", data.get("reason") or (reasons[0] if reasons else status or "not_evaluable")
    return "completed", None


class FilterExecutionTrace:
    """Bounded to the current graph invocation (one timestamp in causal runs)."""

    def __init__(self, nodes, candidates, context, first_group=0):
        self.all_nodes = nodes
        self.nodes = {key: node for key, node in nodes.items() if node["type"] in FILTER_TYPES}
        self.context = context
        self.groups = {}
        self.next_group = first_group
        self.observations = {}
        for item in candidates:
            self.observations[item["observation_id"]] = {
                "filters": {key: {"module_type": node["type"],
                                  "module_version": None,
                                  "enabled": node.get("config", {}).get("enabled") is not False,
                                  "execution_status": "disabled" if node.get("config", {}).get("enabled") is False else "pending",
                                  "reason": "disabled_by_configuration" if node.get("config", {}).get("enabled") is False else None,
                                  "sequence": None, "group_id": None, "evidence_ids": [],
                                  "input_evidence_ids": []} for key, node in self.nodes.items()},
                "order": [], "evidence": {},
            }

    def retain(self, observation_id, entry):
        identity = entry.get("evidence_id") or hashlib.sha256(
            json.dumps([self.context["execution_id"], observation_id,
                        {key: entry.get(key) for key in ("node_id", "signal", "status", "data")}],
                       sort_keys=True, default=str).encode()).hexdigest()
        # Modern envelopes are immutable and can be shared until finalization.
        # Enrich only the trace copy of older envelopes, keeping diagnostics intact.
        traced = entry
        if not entry.get("evidence_id"):
            traced = {**entry, "evidence_id": identity,
                      "module_type": self.all_nodes.get(entry.get("node_id"), {}).get("type", "unknown"),
                      "observation_id": observation_id, "execution_id": self.context["execution_id"]}
        self.observations[observation_id]["evidence"].setdefault(identity, traced)
        return identity

    def start(self, node_id, incoming):
        group_id = f"{self.context['execution_id']}:{self.next_group}"
        self.next_group += 1
        self.groups[group_id] = {"group_id": group_id, "node_id": node_id,
                                 "observation_count": len(incoming), "timing_scope": "filter_call",
                                 "timestamps": sorted({item["timestamp"] for item in incoming}),
                                 "duration_ms": None, "execution_status": "pending"}
        for item in incoming:
            observation = self.observations[item["observation_id"]]
            observation["order"].append(node_id)
            observation["filters"][node_id].update(
                sequence=len(observation["order"]) - 1, group_id=group_id,
                input_evidence_ids=[self.retain(item["observation_id"], entry)
                                    for entry in item.get("evidence", []) if entry.get("status") != "omitted"])
        return group_id

    def complete(self, node_id, incoming, produced, group_id, duration_ms, preparation=None):
        group = self.groups[group_id]
        group.update(duration_ms=duration_ms, execution_status="completed")
        if preparation is not None:
            group["preparation"] = preparation
        returned = {item["observation_id"]: item for item in produced}
        for original in incoming:
            observation = self.observations[original["observation_id"]]
            record = observation["filters"][node_id]
            item = returned.get(original["observation_id"])
            entries = [entry for entry in (item or {}).get("evidence", []) if entry.get("node_id") == node_id]
            if not entries:
                record.update(execution_status="insufficient_data", reason="missing_filter_result")
                continue
            own = entries[-1]
            status, reason = execution_status(own)
            record.update(execution_status=status, reason=reason,
                          module_version=own.get("module_version", (own.get("data") or {}).get(
                              "model_version", (own.get("data") or {}).get("version", "1"))),
                          evidence_ids=[self.retain(original["observation_id"], entry) for entry in entries])

    def failed(self, node_id, group_id, duration_ms, error):
        self.groups[group_id].update(duration_ms=duration_ms, execution_status="error")
        for observation in self.observations.values():
            record = observation["filters"][node_id]
            if record["group_id"] == group_id:
                record.update(execution_status="error", reason="filter_execution_error", error_type=type(error).__name__)

    def finish(self, item, *, aborted=False):
        observation = self.observations[item["observation_id"]]
        filters = deepcopy(observation["filters"])
        for record in filters.values():
            if record["execution_status"] == "pending":
                record.update(execution_status="skipped",
                              reason="execution_aborted" if aborted else "not_routed_to_filter")
        group_ids = [filters[key]["group_id"] for key in observation["order"]]
        item.update({key: value for key, value in self.context.items() if key != "schema_version"})
        item["execution_trace"] = {
            "schema_version": TRACE_VERSION, "filters": filters,
            "absent_filter_types": sorted(FILTER_TYPES - {node["type"] for node in self.nodes.values()}),
            "availability_mask": {key: record["execution_status"] == "completed" for key, record in filters.items()},
            "execution_order": list(observation["order"]),
            "groups": [deepcopy(self.groups[key]) for key in group_ids],
            "evidence": list(deepcopy(observation["evidence"]).values()),
        }
        return item
