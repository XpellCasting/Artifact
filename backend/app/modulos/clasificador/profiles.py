"""Per-run input calibration for unseen sensors; never changes model weights."""
from datetime import timedelta

from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.features import _finite_number, fit_preprocessor
from app.modulos.clasificador.segments import fit_hourly_profile
from app.modulos.clasificador.temporal import CausalInputError
from app.services.estados_validos import valid_state
from app.services.tipos_entidad import infer_kind


MIN_REFERENCE_SECONDS = 3600
MIN_REFERENCE_READINGS = 3
AUTOMATIC_POLICY = "automatic_historical_reference_v1"


def build_sensor_profile(rows, entity_id, before, config, trained_kinds, *,
                         sensor_kind=None, origin="historical_baseline"):
    """Calibrate on earlier observations, without treating them as normal labels."""
    eligible = sorted((row for row in rows if row["entity_id"] == entity_id
                       and parse_utc(row["timestamp"]) < before),
                      key=lambda row: parse_utc(row["timestamp"]))
    valid = [row for row in eligible if valid_state(row.get("state"))]
    if not valid:
        raise CausalInputError("insufficient_sensor_reference")
    declared = sensor_kind or next((row.get("sensor_kind") for row in valid
                                    if row.get("sensor_kind")), "")
    kind = infer_kind(entity_id, [row["state"] for row in valid], declared=declared)
    input_kind = kind
    # Existing segment snapshots encode binary states in the categorical branch.
    if kind == "binary" and "binary" not in trained_kinds and "multistate" in trained_kinds:
        input_kind = "multistate"
    if input_kind not in trained_kinds:
        raise CausalInputError("unsupported_sensor_kind")
    if input_kind == "continuous":
        valid = [row for row in valid if _finite_number(row["state"])]
    times = {parse_utc(row["timestamp"]) for row in valid}
    if (len(times) < MIN_REFERENCE_READINGS
            or (max(times) - min(times)).total_seconds() < MIN_REFERENCE_SECONDS):
        raise CausalInputError("insufficient_sensor_reference")
    prep = fit_preprocessor(eligible, [], config, {entity_id: input_kind})
    info = prep["entities"][entity_id]
    start, end = parse_utc(eligible[0]["timestamp"]), parse_utc(eligible[-1]["timestamp"])
    info["hourly_profile"] = fit_hourly_profile(eligible, start, before, config.local_timezone)
    return {"entity_id": entity_id, "status": "available", "sensor_kind": kind,
            "kind": input_kind, "entity": info, "reference_start": start.isoformat(),
            "reference_end": end.isoformat(), "reference_rows": len(valid),
            "available_from": before.isoformat(),
            "onboarding_policy": AUTOMATIC_POLICY, "reference_origin": origin,
            "normality_verified": False}


def profile_from_past(provider, entity_id, target, config, trained_kinds, sensor_kind=None):
    # Keep the most recent hour out of calibration so the current episode does
    # not set its own scale. Successful profiles are frozen by the run's caller.
    start = target - timedelta(seconds=config.context_seconds)
    end = target - timedelta(seconds=MIN_REFERENCE_SECONDS)
    rows = [row for row in provider(start, end, [entity_id])
            if parse_utc(row["timestamp"]) >= start]
    return build_sensor_profile(rows, entity_id, end, config, trained_kinds,
                                sensor_kind=sensor_kind, origin="causal_warmup")
