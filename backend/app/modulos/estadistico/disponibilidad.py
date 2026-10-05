"""Detecta ausencia explícita de disponibilidad, sin inferir causas físicas."""
from typing import Any

import pandas as pd


AVAILABILITY_SOURCE = "explicit_availability_state"


def availability_detection(row: dict[str, Any]) -> dict[str, Any] | None:
    state = row.get("state")
    if not isinstance(state, str) or state.strip().lower() not in {"unavailable", "unknown"}:
        return None
    timestamp = pd.to_datetime(row.get("timestamp_dt", row.get("timestamp")), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return None
    return {
        "entity_id": str(row.get("entity_id", "")),
        "timestamp": timestamp.isoformat(),
        "source_raw_index": row.get("_raw_index", row.get("source_raw_index")),
        "sensor_kind": row.get("sensor_kind"),
        "observed_state": state,
        "value": None,
        "pattern": "unavailable",
        "suspected_pattern": "unavailable",
        "fault_type": "unavailable",
        "detection_sources": [AVAILABILITY_SOURCE],
        "suspicion_score": 1.0,
        "is_suspicious": True,
    }


def availability_detections(events: pd.DataFrame) -> list[dict[str, Any]]:
    if "state" not in events:
        return []
    explicit = events["state"].astype(str).str.strip().str.lower().isin({"unavailable", "unknown"})
    return [detection for row in events.loc[explicit].to_dict("records")
            if (detection := availability_detection(row)) is not None]
