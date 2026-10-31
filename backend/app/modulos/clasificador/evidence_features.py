"""Whitelisted, order-independent evidence contract shared by replay and runtime.

Identifiers are audit metadata only. Missing, negative and unevaluable sources
have distinct status channels; zero measurements always have a separate mask.
"""
import hashlib
import json
import math

VERSION = "optional-filter-evidence-v1"
FAMILIES = ("statistical_focus", "contextual")
MEASURES = {
    "statistical_focus": ("suspicion_score", "normalized_residual", "score_ratio",
                          "persistence", "coverage", "P", "D", "N"),
    "contextual": ("support", "contradiction", "score_ratio", "coverage",
                   "evaluable_neighbors", "map_fresh"),
}
STATES = ("absent", "negative", "positive", "not_evaluable", "omitted", "error")
FEATURE_NAMES = tuple(f"{family}:{key}" for family in FAMILIES
                     for key in (*STATES, *(f"{m}:{k}" for m in MEASURES[family] for k in ("value", "mask"))))


def normalized(entry):
    if entry.get("schema_version", "module-envelope-v1") not in {"module-envelope-v1", "module-envelope-v2"}:
        return None
    family = entry.get("module_type")
    if family not in FAMILIES:
        return None
    data = entry.get("data", {})
    status = entry.get("status")
    state = (status if status in {"omitted", "error", "not_evaluable"} else
             "not_evaluable" if entry.get("signal") not in {"positive", "negative"} else
             "positive" if entry["signal"] == "positive" else "negative")
    measures = {k: data[k] for k in MEASURES[family] if k in data}
    if family == "statistical_focus":
        if "z" in data: measures["normalized_residual"] = data["z"]
        if "joint_score" in data: measures["score_ratio"] = data["joint_score"]
        measures.update({k:v for k,v in data.get("channels",{}).items() if k in {"P","D","N"}})
        if isinstance(data.get("persistence"),dict):
            measures["persistence"]=data["persistence"].get("score")
    # New producers can provide an explicitly versioned whitelist block.
    block = data.get("filter_features") or entry.get("filter_features", {})
    if block and block.get("version") != VERSION:
        state,measures="not_evaluable",{}
    if block.get("version") == VERSION:
        measures.update({k: v for k, v in block.get("measures", {}).items() if k in MEASURES[family]})
    if family == "contextual":
        coverage=data.get("coverage",{})
        if isinstance(coverage,dict):
            measures["evaluable_neighbors"]=coverage.get("observed_neighbors")
            measures["coverage"]=coverage.get("ratio")
        score,threshold=data.get("decision_score"),data.get("decision_threshold")
        if isinstance(score,(int,float)) and isinstance(threshold,(int,float)) and threshold>0:
            measures["score_ratio"]=score/threshold
        outcome = data.get("context_outcome")
        if outcome in {"supported", "contradicted"}:
            measures.update(support=float(outcome == "supported"), contradiction=float(outcome == "contradicted"))
        stale = data.get("map_quality", {}).get("stale")
        if isinstance(stale, bool):
            measures["map_fresh"] = float(not stale)
            if stale:
                state = "not_evaluable"
    clean = {}
    for key, value in measures.items():
        if isinstance(value, (float, int)) and math.isfinite(value):
            clean[key] = float(value)
    return {"family": family, "state": state, "measures": clean,
            "id": entry.get("evidence_id") or hashlib.sha256(json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()}


def encode_evidence(entries=(), *, observation_id=None, timestamp=None, execution_id=None, mode="both"):
    if mode not in {"none", "statistical", "contextual", "both"}:
        raise ValueError("Modalidad de evidencia inválida")
    unique = {}
    for entry in entries:
        if any(expected is not None and entry.get(key) != expected
               for key, expected in (("observation_id", observation_id), ("timestamp", timestamp), ("execution_id", execution_id))):
            continue
        row = normalized(entry)
        if row and (mode == "both" or mode == "statistical" and row["family"] == FAMILIES[0]
                    or mode == "contextual" and row["family"] == FAMILIES[1]):
            # Sort canonical content too: malformed duplicate IDs cannot pick the last writer.
            unique.setdefault(row["id"], []).append(row)
    rows = [min(values, key=lambda r: json.dumps(r, sort_keys=True)) for _, values in sorted(unique.items())]
    vector, consumed, valid_families = [], [], []
    for family in FAMILIES:
        group = [r for r in rows if r["family"] == family]
        vector.extend([float(not group), *(float(any(r["state"] == state for r in group)) for state in STATES[1:])])
        usable = [r for r in group if r["state"] in {"positive", "negative"}]
        if usable:
            valid_families.append(family)
            consumed.extend(r["id"] for r in usable)
        for key in MEASURES[family]:
            values = [r["measures"][key] for r in usable if key in r["measures"]]
            vector.extend([math.fsum(sorted(values))/len(values) if values else 0., float(bool(values))])
    actual_mode = "both" if len(valid_families) == 2 else "statistical" if FAMILIES[0] in valid_families else "contextual" if valid_families else "none"
    fingerprint = hashlib.sha256(json.dumps([VERSION, vector, sorted(consumed)], separators=(",", ":")).encode()).hexdigest()
    return {"version": VERSION, "values": vector, "usable": bool(valid_families),
            "mode": actual_mode, "consumed_ids": sorted(consumed), "fingerprint": fingerprint}
