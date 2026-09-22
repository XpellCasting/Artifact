"""Causal, inspectable F3 features, pattern catalogue and supervised prototypes.

Catalogue scores describe compatibility with an observable pattern, not a fault's
physical cause or a calibrated probability. Thresholds are engineering defaults
to validate, not learned medical/physical limits. Only reviewed normal observations
belong in ``build_reference``; caller owns that provenance and its temporal split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from statistics import median
from typing import Any
from .Estados_Validos import valid_state

ENGINE_VERSION = "diagnostic-patterns-1.0"
FEATURE_VERSION = "causal-window-1.0"
CATALOG = [
    {"fault_type": "spike", "label": "Pico", "description": "Cambio abrupto seguido de recuperación observada.", "entity_kinds": ["continuous"]},
    {"fault_type": "bias", "label": "Desplazamiento", "description": "Desviación sostenida del nivel de referencia, sin tendencia dominante.", "entity_kinds": ["continuous"]},
    {"fault_type": "drift", "label": "Deriva", "description": "Separación progresiva y consistente en tiempo real.", "entity_kinds": ["continuous"]},
    {"fault_type": "noise", "label": "Ruido", "description": "Fluctuaciones y cambios de dirección superiores a la referencia.", "entity_kinds": ["continuous"]},
    {"fault_type": "stuck", "label": "Valor congelado", "description": "Lecturas repetidas durante una constancia anormalmente prolongada.", "entity_kinds": ["continuous", "binary", "multistate"]},
    {"fault_type": "unavailable", "label": "Indisponibilidad", "description": "Estados explícitos no válidos; el silencio por sí solo no confirma indisponibilidad.", "entity_kinds": ["continuous", "binary", "multistate"]},
    {"fault_type": "state_replace", "label": "Estado sustituido", "description": "Persistencia de un estado desconocido respecto de la referencia revisada.", "entity_kinds": ["binary", "multistate"]},
]
KINDS = {"continuous", "binary", "multistate"}
INVALID_STATES = {"", "unknown", "unavailable", "none", "null", "nan", "inf", "-inf", "+inf", "infinity", "-infinity"}
# This allowlist is deliberately independent of labels, IDs, injection controls
# and original-vs-injected comparisons. None of those are available at inference.
FEATURE_NAMES = (
    "sample_count", "numeric_count", "duration_seconds", "invalid_ratio", "last_invalid",
    "invalid_run_count", "cadence_ratio", "latest_gap_ratio", "level_z", "latest_z",
    "max_residual_z", "shifted_fraction", "delta_ratio", "rate_ratio", "noise_ratio",
    "variability_ratio", "direction_reversal_ratio", "trend_z", "trend_fit_r2",
    "direction_consistency", "constant_duration_seconds", "constant_duration_ratio",
    "constant_run_count", "spike_strength", "spike_recovery_ratio", "unseen_state_ratio",
    "transition_ratio", "f1_s_H", "f1_s_R", "f1_s_C", "f1_s_D_const", "f1_s_M",
    "f1_s_DSPOT", "f2_support", "f2_contradiction", "f2_coverage",
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _time(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Cada observación requiere timestamp ISO con zona horaria.")
    # datetime otherwise silently truncates sub-microsecond precision.
    import re
    fraction = re.search(r"[.,](\d+)", value)
    if fraction and any(c != "0" for c in fraction.group(1)[6:]):
        raise ValueError("La precisión temporal máxima es de microsegundos.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Timestamp ISO no válido.") from exc
    if parsed.tzinfo is None:
        raise ValueError("El timestamp requiere zona horaria explícita.")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _rows(rows: list[dict], entity_id: str | None = None) -> list[dict]:
    parsed = []
    for index, row in enumerate(rows):
        if entity_id and row.get("entity_id") not in (None, entity_id):
            continue
        state = str(row.get("state", row.get("value", "")))
        parsed.append({"time": _time(row.get("timestamp", row.get("last_changed"))),
                       "state": state, "value": _number(state),
                       "valid": valid_state(state),
                       "order": _number(row.get("source_row")) or index})
    return sorted(parsed, key=lambda item: (item["time"], item["order"]))


def _mad(values: list[float]) -> float:
    center = median(values) if values else 0.0
    return median([abs(value - center) for value in values]) if values else 0.0


def _quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _same(a: dict, b: dict, kind: str) -> bool:
    if not a["valid"] or not b["valid"]:
        return False
    if kind == "continuous":
        return a["value"] is not None and a["value"] == b["value"]
    return a["state"] == b["state"]


def _runs(rows: list[dict], kind: str, cadence: float) -> list[tuple[float, int]]:
    runs: list[tuple[float, int]] = []
    start = 0
    for index in range(1, len(rows) + 1):
        continuing = index < len(rows) and _same(rows[index - 1], rows[index], kind)
        if continuing and cadence > 0:
            continuing = (rows[index]["time"] - rows[index - 1]["time"]).total_seconds() <= cadence * 5
        if not continuing:
            runs.append(((rows[index - 1]["time"] - rows[start]["time"]).total_seconds(), index - start))
            start = index
    return runs


def build_reference(rows: list[dict], entity_id: str, kind: str) -> dict:
    """Materialize robust statistics from reviewed NORMAL rows, without mutation."""
    if kind not in KINDS:
        raise ValueError("Tipo de entidad desconocido.")
    parsed = _rows(rows, entity_id)
    valid = [row for row in parsed if row["valid"] and (kind != "continuous" or row["value"] is not None)]
    if len(valid) < 3:
        raise ValueError("La referencia requiere al menos tres observaciones válidas.")
    gaps = [(b["time"] - a["time"]).total_seconds() for a, b in zip(parsed, parsed[1:])]
    positive_gaps = [value for value in gaps if value > 0]
    cadence = median(positive_gaps) if positive_gaps else 0.0
    if cadence <= 0:
        raise ValueError("La referencia requiere al menos dos instantes distintos.")
    values = [row["value"] for row in valid] if kind == "continuous" else []
    deltas, rates = [], []
    transitions = 0
    for a, b in zip(parsed, parsed[1:]):
        elapsed = (b["time"] - a["time"]).total_seconds()
        if elapsed <= 0 or not a["valid"] or not b["valid"]:
            continue
        transitions += int(not _same(a, b, kind))
        if kind == "continuous" and a["value"] is not None and b["value"] is not None:
            delta = b["value"] - a["value"]
            deltas.append(delta)
            rates.append(delta / elapsed)
    center = median(values) if values else None
    floor = max(abs(center or 0) * 1e-6, 1e-9)
    robust_spread = 1.4826 * _mad(values)
    if robust_spread <= floor:
        robust_spread = (_quantile(values, .75) - _quantile(values, .25)) / 1.349
    scale = max(robust_spread, floor) if values else None
    delta_scale = max(median([abs(v) for v in deltas]) if deltas else 0, 1.4826 * _mad(deltas), (scale or floor) * .01, floor)
    rate_scale = max(median([abs(v) for v in rates]) if rates else 0, 1.4826 * _mad(rates), delta_scale / cadence)
    runs = _runs(parsed, kind, cadence)
    duration = (parsed[-1]["time"] - parsed[0]["time"]).total_seconds()
    return {"entity_id": entity_id, "sensor_kind": kind, "feature_version": FEATURE_VERSION,
            "start": _iso(parsed[0]["time"]), "end": _iso(parsed[-1]["time"]),
            "rows": len(parsed), "valid_rows": len(valid), "duration_seconds": duration,
            "stats": {"center": center, "scale": scale, "mad": _mad(values) if values else None,
                      "flat_reference": bool(values and max(values) == min(values)),
                      "delta_scale": delta_scale, "rate_scale": rate_scale,
                      "cadence_seconds": cadence, "max_constant_seconds": max(run[0] for run in runs),
                      "max_constant_reports": max(run[1] for run in runs),
                      "transition_rate": transitions / duration if duration else 0,
                      "states": dict(Counter(row["state"] for row in valid))},
            "limitations": ["Referencia limitada a los períodos normales revisados; no representa todas las condiciones de operación.",
                            "La cadencia observada no demuestra que el sensor deba emitir reportes periódicos."]}


def extract_features(rows: list[dict], reference: dict, timestamp: str | None = None,
                     window_seconds: int = 3600, *, allow_retrospective_reference: bool = False) -> dict:
    """Use only observations available by timestamp, in the actual preceding time window."""
    if not isinstance(window_seconds, (int, float)) or not math.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("La ventana debe tener una duración positiva y finita.")
    all_rows = _rows(rows, reference.get("entity_id"))
    cutoff = _time(timestamp) if timestamp else (all_rows[-1]["time"] if all_rows else None)
    if not allow_retrospective_reference and cutoff and reference.get("end") and _time(reference["end"]) >= cutoff:
        raise ValueError("La referencia debe finalizar antes del instante diagnosticado.")
    features = dict.fromkeys(FEATURE_NAMES)
    if cutoff is None:
        return features
    parsed = [row for row in all_rows if 0 <= (cutoff - row["time"]).total_seconds() <= window_seconds]
    if not parsed:
        features["sample_count"] = 0
        return features
    kind = reference.get("sensor_kind", "continuous")
    stats = reference.get("stats") or {}
    cadence = max(_number(stats.get("cadence_seconds")) or 0, 1e-9)
    valid = [row for row in parsed if row["valid"] and (kind != "continuous" or row["value"] is not None)]
    is_valid = lambda row: row["valid"] and (kind != "continuous" or row["value"] is not None)
    invalid_run = 0
    for row in reversed(parsed):
        if is_valid(row):
            break
        invalid_run += 1
    duration = (parsed[-1]["time"] - parsed[0]["time"]).total_seconds()
    gaps = [(b["time"] - a["time"]).total_seconds() for a, b in zip(parsed, parsed[1:]) if b["time"] > a["time"]]
    runs = _runs(parsed, kind, cadence)
    last_duration, last_reports = runs[-1] if runs else (0, 0)
    constant_reference = max(_number(stats.get("max_constant_seconds")) or 0, cadence)
    features.update(sample_count=len(parsed), numeric_count=sum(row["value"] is not None for row in valid),
                    duration_seconds=duration, invalid_ratio=1 - len(valid) / len(parsed),
                    last_invalid=int(not is_valid(parsed[-1])), invalid_run_count=invalid_run,
                    cadence_ratio=max(gaps, default=0) / cadence,
                    latest_gap_ratio=(cutoff - parsed[-1]["time"]).total_seconds() / cadence,
                    constant_duration_seconds=last_duration if is_valid(parsed[-1]) else 0,
                    constant_duration_ratio=last_duration / constant_reference if is_valid(parsed[-1]) else 0,
                    constant_run_count=last_reports if is_valid(parsed[-1]) else 0)
    transitions = sum(not _same(a, b, kind) for a, b in zip(parsed, parsed[1:]) if is_valid(a) and is_valid(b))
    ref_transition = _number(stats.get("transition_rate")) or 0
    features["transition_ratio"] = (transitions / duration) / ref_transition if ref_transition > 0 and duration > 0 else None
    if kind != "continuous":
        known = stats.get("states") or {}
        features["unseen_state_ratio"] = sum(row["state"] not in known for row in valid) / len(valid) if valid else None
        return features
    if not valid:
        return features
    scale, center = _number(stats.get("scale")), _number(stats.get("center"))
    if scale is None or scale <= 0 or center is None:
        return features
    values = [row["value"] for row in valid]
    residuals = [(value - center) / scale for value in values]
    features.update(level_z=abs(median(residuals)), latest_z=abs(residuals[-1]),
                    max_residual_z=max(abs(value) for value in residuals),
                    shifted_fraction=sum(abs(value) >= 3 for value in residuals) / len(residuals),
                    variability_ratio=1.4826 * _mad(values) / scale)
    deltas, rates = [], []
    for a, b in zip(parsed, parsed[1:]):
        dt = (b["time"] - a["time"]).total_seconds()
        if dt > 0 and is_valid(a) and is_valid(b):
            deltas.append(b["value"] - a["value"])
            rates.append((b["value"] - a["value"]) / dt)
    delta_scale = max(_number(stats.get("delta_scale")) or scale, 1e-9)
    rate_scale = max(_number(stats.get("rate_scale")) or delta_scale / cadence, 1e-9)
    signs = [1 if value > 0 else -1 for value in deltas if abs(value) > delta_scale * .1]
    features.update(delta_ratio=max([abs(v) for v in deltas], default=0) / delta_scale,
                    rate_ratio=max([abs(v) for v in rates], default=0) / rate_scale,
                    noise_ratio=1.4826 * _mad(deltas) / delta_scale,
                    direction_reversal_ratio=sum(a != b for a, b in zip(signs, signs[1:])) / max(len(signs) - 1, 1),
                    direction_consistency=max(signs.count(1), signs.count(-1)) / len(signs) if signs else 0)
    x = [(row["time"] - valid[0]["time"]).total_seconds() for row in valid]
    mean_x, mean_y = sum(x) / len(x), sum(values) / len(values)
    xx = sum((value - mean_x) ** 2 for value in x)
    yy = sum((value - mean_y) ** 2 for value in values)
    xy = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, values))
    features["trend_z"] = abs(xy / xx) * (x[-1] - x[0]) / scale if xx else 0
    features["trend_fit_r2"] = min(1, max(0, xy * xy / (xx * yy))) if xx and yy else 0
    strength, recovery = 0.0, None
    for a, b, c in zip(parsed, parsed[1:], parsed[2:]):
        if not all(is_valid(row) for row in (a, b, c)):
            continue
        before = (b["time"] - a["time"]).total_seconds()
        after = (c["time"] - b["time"]).total_seconds()
        if not (0 < before <= 3 * cadence and 0 < after <= 3 * cadence):
            continue
        up, down = b["value"] - a["value"], c["value"] - b["value"]
        candidate = min(abs(up), abs(down)) / delta_scale
        return_ratio = abs(c["value"] - a["value"]) / max(abs(up), delta_scale)
        if up * down < 0 and return_ratio <= .35 and candidate > strength:
            strength, recovery = candidate, return_ratio
    features["spike_strength"], features["spike_recovery_ratio"] = strength, recovery
    return features


def _reference_summary(reference: dict | None) -> dict:
    return {key: reference.get(key) for key in ("entity_id", "sensor_kind", "start", "end", "rows", "feature_version")} if reference else {}


def _attach_filter_features(features: dict, statistical: dict, contextual: dict | None) -> None:
    reading = statistical.get("reading") or statistical
    components = reading.get("components") or statistical.get("components") or {}
    for name in ("s_H", "s_R", "s_C", "s_D_const", "s_M", "s_DSPOT"):
        features["f1_" + name] = _number(components.get(name))
    if contextual:
        features["f2_support"] = _number(contextual.get("support_score"))
        features["f2_contradiction"] = _number(contextual.get("contradiction_score"))
        features["f2_coverage"] = _number(contextual.get("coverage_ratio", (contextual.get("coverage") or {}).get("ratio")))


def _catalogue(features: dict, reference: dict, kind: str) -> list[dict]:
    f = lambda key: _number(features.get(key)) or 0
    matches = []
    def add(label: str, score: float, evidence: list[str]) -> None:
        matches.append({"fault_type": label, "score": round(min(.98, max(0, score)), 6), "evidence": evidence})
    n = f("sample_count")
    if f("last_invalid") and f("invalid_run_count") >= 2:
        add("unavailable", .85 + min(.1, f("invalid_ratio") * .1), ["Se observan al menos dos reportes consecutivos con estado explícito no válido."])
    if n < 4 or f("invalid_ratio") > .5:
        return sorted(matches, key=lambda item: -item["score"])
    if (f("constant_run_count") >= 5 and f("constant_duration_ratio") >= 3
            and not reference.get("stats", {}).get("flat_reference")
            and f("cadence_ratio") <= 5
            and (kind == "continuous" or f("unseen_state_ratio") == 0)):
        add("stuck", .78 + min(.17, (f("constant_duration_ratio") - 3) * .025),
            ["Hay reportes repetidos sin cambio durante al menos tres veces la constancia normal de referencia.",
             "La duración se mide entre reportes observados; no se interpreta el silencio como constancia."])
    if kind != "continuous":
        if f("unseen_state_ratio") >= .7 and f("constant_run_count") >= 3:
            add("state_replace", .83, ["Un estado ausente de la referencia revisada persiste en varios reportes.",
                                         "Un estado legítimo no representado en la referencia puede producir el mismo patrón."])
        return sorted(matches, key=lambda item: -item["score"])
    if f("spike_strength") >= 4 and f("noise_ratio") < 3 and f("shifted_fraction") < .5:
        add("spike", .83 + min(.1, (f("spike_strength") - 4) * .015),
            ["Se observa un salto y una recuperación breve al nivel previo.", "La recuperación ya ocurrió antes de emitir este diagnóstico."])
    if (f("noise_ratio") >= 3 and f("direction_reversal_ratio") >= .5
            and n >= 8 and f("trend_fit_r2") < .65):
        add("noise", .8 + min(.13, (f("noise_ratio") - 3) * .015),
            ["Los incrementos fluctúan al menos tres veces más que en la referencia y cambian repetidamente de dirección."])
    if (f("trend_z") >= 3 and f("trend_fit_r2") >= .8
            and f("direction_consistency") >= .75 and n >= 6):
        add("drift", .83 + min(.1, (f("trend_z") - 3) * .015),
            ["La separación evoluciona de forma progresiva y consistente usando el tiempo real de las observaciones."])
    if (f("level_z") >= 3 and f("shifted_fraction") >= .7 and n >= 6
            and f("trend_fit_r2") < .75 and f("noise_ratio") < 3
            and f("constant_run_count") < 5):
        add("bias", .82 + min(.11, (f("level_z") - 3) * .015),
            ["La mayor parte de la ventana mantiene un nivel separado de la referencia, sin tendencia ni ruido dominantes."])
    return sorted(matches, key=lambda item: -item["score"])


def diagnose(statistical: dict, contextual: dict | None, reference: dict | None = None,
             rows: list[dict] | None = None, model: dict | None = None) -> dict:
    """Diagnose pending F2 cases. Never overrides a terminal F2 decision."""
    kind = statistical.get("sensor_kind") or (reference or {}).get("sensor_kind", "continuous")
    result = {"entity_id": statistical.get("entity_id") or (reference or {}).get("entity_id"),
              "timestamp": statistical.get("timestamp"), "sensor_kind": kind,
              "status": "indeterminate", "fault_type": None, "predicted_label": None, "compatibility": None,
              "confidence_kind": "compatibility_not_probability", "pattern_matches": [], "reasons": [],
              "features": dict.fromkeys(FEATURE_NAMES), "reference": _reference_summary(reference),
              "model_version": (model or {}).get("model_version", "catalogue:" + ENGINE_VERSION),
              "catalog_version": ENGINE_VERSION, "needs_review": True}
    if contextual and ((contextual.get("routing") or {}).get("terminal") is True):
        result.update(status="no_fault_evidence", needs_review=False,
                      reasons=["F2 cerró el episodio; F3 no se ejecuta ni modifica ese resultado."])
        return result
    if reference is None or not rows:
        result["reasons"].append("Falta una referencia normal anterior o una ventana de observaciones para comparar patrones.")
        return result
    if reference.get("entity_id") != result["entity_id"] or reference.get("sensor_kind") != kind:
        result["reasons"].append("La referencia no corresponde a la entidad y tipo diagnosticados.")
        return result
    try:
        features = extract_features(rows, reference, statistical.get("timestamp"), statistical.get("diagnostic_window_seconds", 3600))
    except ValueError as exc:
        result["reasons"].append(str(exc))
        return result
    _attach_filter_features(features, statistical, contextual)
    result["features"] = features
    if (_number(features.get("sample_count")) or 0) < 3:
        result["reasons"].append("La ventana todavía no reúne tres reportes para comparar la evolución.")
        return result
    if model is not None:
        prediction = predict_features(features, kind, model)
        result.update({key: prediction[key] for key in ("status", "fault_type", "predicted_label", "compatibility", "pattern_matches", "reasons", "needs_review")})
    else:
        matches = _catalogue(features, reference, kind)
        result["pattern_matches"] = matches
        if matches and (len(matches) == 1 or matches[0]["score"] - matches[1]["score"] >= .1):
            result.update(status="compatible", fault_type=matches[0]["fault_type"], predicted_label=matches[0]["fault_type"], compatibility=matches[0]["score"], needs_review=False)
            result["reasons"].extend(matches[0]["evidence"])
        elif matches:
            result["reasons"].append("Hay patrones competidores sin separación suficiente; se conserva el tipo indeterminado.")
        elif statistical.get("is_suspicious") is False and (_number(features.get("invalid_ratio")) or 0) == 0:
            result.update(status="no_fault_evidence", predicted_label="normal", needs_review=False)
            result["reasons"].append("No se reconoce evidencia de los patrones del catálogo en esta ventana.")
        else:
            result["reasons"].append("La evolución no satisface un patrón suficientemente definido del catálogo.")
    result["reasons"].append("La semejanza del patrón no confirma una causa física ni constituye una probabilidad calibrada.")
    if contextual:
        coverage = _number(features.get("f2_coverage")) or 0
        if coverage < .5:
            result["reasons"].append("La cobertura contextual es limitada; la ausencia de respaldo no se interpreta como prueba de falla.")
        elif (_number(features.get("f2_contradiction")) or 0) > (_number(features.get("f2_support")) or 0):
            result["reasons"].append("F2 aporta contradicción contextual observada como evidencia complementaria.")
    return result


def _feature_values(features: dict) -> dict[str, float]:
    return {name: value for name in FEATURE_NAMES if (value := _number(features.get(name))) is not None}


def fit_model(examples: list[dict]) -> dict:
    """Fit inspectable nearest-centroid prototypes using DEVELOPMENT only.

    Training radius and separation are conservative engineering rejection rules,
    never a validation-set accuracy claim. Callers must evaluate the candidate on
    temporally separate validation data before activation.
    """
    if any(row.get("partition") != "development" for row in examples):
        raise ValueError("El ajuste solo admite ejemplos de desarrollo; validación y prueba no pueden entrenar el modelo.")
    valid_labels = {item["fault_type"] for item in CATALOG} | {"normal", "environmental_anomaly"}
    grouped = defaultdict(list)
    excluded = []
    for index, row in enumerate(examples):
        label, kind = row.get("label"), row.get("sensor_kind")
        values = _feature_values(row.get("features") or {})
        applicable = label in {"normal", "environmental_anomaly"} or any(item["fault_type"] == label and kind in item["entity_kinds"] for item in CATALOG)
        if kind not in KINDS or label not in valid_labels or not applicable or len(values) < 3 or values.get("sample_count", 3) < 3:
            excluded.append({"example_id": row.get("example_id", str(index)), "reason": "Clase, tipo o características insuficientes/incompatibles."})
            continue
        grouped[kind].append((row, values))
    model = {"engine_version": ENGINE_VERSION, "feature_version": FEATURE_VERSION,
             "algorithm": "robust_scaled_nearest_centroid", "confidence_kind": "compatibility_not_probability",
             "scalers_by_kind": {}, "prototypes_by_kind": {}, "feature_names": {},
             "classes": [], "support": {}, "training_groups": sorted({str(row.get("group_id", row.get("example_id", ""))) for row in examples}),
             "excluded_examples": excluded, "requires_validation": True,
             "rejection": {"min_margin": .15, "radius_multiplier": 1.5, "radius_floor": 1.5},
             "limitations": ["Los puntajes de distancia son compatibilidad, no probabilidades.",
                             "La aceptación exige evaluación independiente antes de activar esta versión."]}
    for kind, entries in sorted(grouped.items()):
        labels = sorted({row["label"] for row, _ in entries})
        if "normal" not in labels or not set(labels).intersection({item["fault_type"] for item in CATALOG}):
            excluded.extend({"example_id": row.get("example_id"), "reason": "Este tipo necesita controles normales y al menos una clase de falla."} for row, _ in entries)
            continue
        names = [name for name in FEATURE_NAMES if all(name in values for _, values in entries)]
        if len(names) < 3:
            continue
        scalers = {}
        for name in names:
            values = [features[name] for _, features in entries]
            center = median(values)
            scalers[name] = {"center": center, "scale": max(1.4826 * _mad(values), (_quantile(values, .9) - _quantile(values, .1)) / 2.563, abs(center) * .01, .01)}
        normalized = [(row, [(values[name] - scalers[name]["center"]) / scalers[name]["scale"] for name in names]) for row, values in entries]
        prototypes = []
        for label in labels:
            members = [vector for row, vector in normalized if row["label"] == label]
            center = [sum(vector[index] for vector in members) / len(members) for index in range(len(names))]
            distances = [_distance(vector, center) for vector in members]
            prototypes.append({"label": label, "centroid": center, "support": len(members),
                               "group_support": len({str(row.get("group_id", row.get("example_id", ""))) for row, _ in normalized if row["label"] == label}),
                               "radius": max(1.5, _quantile(distances, .95) * 1.5)})
            model["support"][kind + ":" + label] = len(members)
        model["scalers_by_kind"][kind], model["feature_names"][kind] = scalers, names
        model["prototypes_by_kind"][kind] = prototypes
        model["classes"] = sorted(set(model["classes"]) | set(labels))
    if not model["prototypes_by_kind"]:
        raise ValueError("Se necesitan ejemplos normales y de falla del mismo tipo con al menos tres características comunes. Cada ventana requiere tres reportes; incluye contexto previo al preparar casos de pico.")
    digest = hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()[:16]
    model["model_version"] = "prototypes:" + digest
    return model


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(a))


def predict_features(features: dict, sensor_kind: str, model: dict) -> dict:
    """Predict from an already extracted feature vector, including held-out rows."""
    result = {"status": "indeterminate", "fault_type": None, "predicted_label": None, "compatibility": None,
              "pattern_matches": [], "reasons": [], "needs_review": True}
    names = model.get("feature_names", {}).get(sensor_kind, [])
    scalers = model.get("scalers_by_kind", {}).get(sensor_kind, {})
    prototypes = model.get("prototypes_by_kind", {}).get(sensor_kind, [])
    if model.get("feature_version") != FEATURE_VERSION or not names or len(prototypes) < 2:
        result["reasons"].append("No existe un modelo compatible entrenado para este tipo de entidad.")
        return result
    values = _feature_values(features)
    if values.get("sample_count", 3) < 3:
        result["reasons"].append("La ventana no reúne tres reportes; se conserva el tipo indeterminado.")
        return result
    if any(name not in values or name not in FEATURE_NAMES for name in names):
        result["reasons"].append("Faltan características utilizadas en el entrenamiento; el modelo se abstiene.")
        return result
    vector = [(values[name] - scalers[name]["center"]) / scalers[name]["scale"] for name in names]
    distances = sorted([(_distance(vector, proto["centroid"]), proto) for proto in prototypes], key=lambda pair: pair[0])
    for distance, proto in distances:
        result["pattern_matches"].append({"fault_type": proto["label"], "score": round(1 / (1 + distance), 6),
                                          "evidence": [f"Distancia normalizada al prototipo: {distance:.3f}; soporte de entrenamiento: {proto['support']}."]})
    distance, closest = distances[0]
    separation = (distances[1][0] - distance) / max(distances[1][0], 1e-9)
    if distance > closest["radius"]:
        result["reasons"].append("El episodio está fuera de la región de patrones aprendidos.")
    elif separation < model.get("rejection", {}).get("min_margin", .15):
        result["reasons"].append("Dos clases tienen distancias demasiado similares; se conserva el tipo indeterminado.")
    else:
        normal = closest["label"] in {"normal", "environmental_anomaly"}
        result.update(status="no_fault_evidence" if normal else "compatible", fault_type=None if normal else closest["label"],
                      predicted_label=closest["label"], compatibility=round(1 / (1 + distance), 6), needs_review=False)
        result["reasons"].append("El episodio es compatible con el prototipo supervisado de " + closest["label"] + ".")
    return result
