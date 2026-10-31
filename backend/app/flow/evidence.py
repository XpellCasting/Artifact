"""Versioned, immutable evidence shared by every processing node."""
from copy import deepcopy
import hashlib
import json

VERSION = "module-envelope-v1"
POLICIES = {"independent", "combine_v1"}


def own_assessment(signal, status, data):
    label = data.get("predicted_label")
    if status == "not_evaluable" or status == "omitted":
        label = "indeterminate"
    elif label not in {"normal", "sensor_fault", "environmental_anomaly"}:
        if status == "compatible":
            label = "sensor_fault"
        elif status == "environmental_anomaly" or data.get("contextual_class") == "environmental_anomaly":
            label = "environmental_anomaly"
        elif status == "no_fault_evidence":
            label = "normal"
        elif status == "supported":
            label = "contextual_support"
        else:
            label = "suspicion" if signal == "positive" else "no_alert" if signal == "negative" else "indeterminate"
    return {"label": label, "signal": signal, "status": status}


def combine(entries, observation_id=None, timestamp=None, execution_id=None):
    # Only original assessments count. Combined results are never new votes.
    assessments = [e for e in entries if e.get("module_type") in {"statistical_focus", "contextual", "classifier"}
                   and (observation_id is None or e.get("observation_id", observation_id) == observation_id)
                   and (timestamp is None or e.get("timestamp", timestamp) == timestamp)
                   and (execution_id is None or e.get("execution_id", execution_id) == execution_id)]
    claims = [(e, e.get("own_evaluation", own_assessment(e.get("signal"), e.get("status"), e.get("data", {})))) for e in assessments]
    integrated = [(e,a) for e,a in claims if e.get("module_type") == "classifier"
                  and e.get("data",{}).get("decision_policy") == "integrated_classifier_v1"
                  and e.get("data",{}).get("classifier_evaluable") is True]
    if integrated:
        # An executed abstention is authoritative too. Earlier filters are audit
        # inputs, never a second round of votes over their consumer's decision.
        entry,own = integrated[-1]
        return {"label":own["label"],"signal":own["signal"],"conflict":False,
                "used_evidence_ids":[entry.get("evidence_id")],
                "unused_evidence_ids":[e.get("evidence_id") for e in entries if e is not entry],
                "policy":"integrated_classifier_v1"}
    labels = {a["label"] for _, a in claims} - {"indeterminate", "contextual_support"}
    conflict = ({"sensor_fault", "environmental_anomaly"} <= labels or
                ("normal" in labels and bool(labels & {"suspicion", "sensor_fault", "environmental_anomaly"})))
    label = ("indeterminate" if conflict else "sensor_fault" if "sensor_fault" in labels else
             "environmental_anomaly" if "environmental_anomaly" in labels else
             "normal" if "normal" in labels else "suspicion" if "suspicion" in labels else
             "no_alert" if "no_alert" in labels else "indeterminate")
    signal = "positive" if label in {"sensor_fault", "suspicion"} else "negative" if label in {"normal", "environmental_anomaly", "no_alert"} else "unknown"
    return {"label": label, "signal": signal, "conflict": conflict,
            "used_evidence_ids": [e.get("evidence_id") for e, a in claims if a["label"] != "indeterminate"],
            "unused_evidence_ids": [e.get("evidence_id") for e in entries if e not in assessments],
            "policy": "combine_v1"}


def append_evidence(item, node_id, module_type, signal, status, data, policy, execution_id="", filter_config=None):
    result = deepcopy(item)
    identity = [execution_id, item["observation_id"], node_id, item.get("timestamp"), status, data]
    if filter_config is not None:
        identity.append(filter_config)
    identifier = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    own = own_assessment(signal, status, data)
    entry = {"evidence_id": identifier, "node_id": node_id, "module_type": module_type,
             "schema_version": VERSION, "module_version": data.get("model_version", data.get("version", "1")),
             "execution_id": execution_id, "observation_id": item["observation_id"],
             "timestamp": item.get("timestamp"), "signal": signal, "status": status,
             "data": deepcopy(data), "own_evaluation": own}
    if module_type in {"statistical_focus", "contextual"}:
        from app.modulos.clasificador.evidence_features import normalized, VERSION as FILTER_VERSION, MEASURES
        structured = normalized(entry)
        entry["filter_contract_version"] = FILTER_VERSION
        entry["filter_configuration_sha256"] = hashlib.sha256(json.dumps(filter_config or {}, sort_keys=True, default=str).encode()).hexdigest()
        entry["filter_configuration"] = deepcopy(filter_config or {})
        entry["filter_features"] = {"version":FILTER_VERSION,"execution_state":structured["state"],
            "measures":structured["measures"],"masks":{key:key in structured["measures"] for key in MEASURES[module_type]}}
    entries = [*result.get("evidence", []), entry]
    combined = combine(entries, item["observation_id"], item.get("timestamp"), execution_id)
    entry["used_evidence_ids"] = [key for key in combined["used_evidence_ids"] if key != identifier]
    result.update(contract_version=VERSION, evidence=entries, own_evaluation=own,
                  combined_evaluation=combined, evidence_policy=policy,
                  signal=combined["signal"] if policy == "combine_v1" else signal)
    return result


def merge_evidence(items):
    unique = {}
    for item in items:
        for entry in item.get("evidence", []):
            identifier = entry.get("evidence_id") or hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
            unique.setdefault(identifier, deepcopy(entry))
    return list(unique.values())
