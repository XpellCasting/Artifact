from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .domain.entities import canonical_entities_to_internal as domain_entities_to_internal
from .domain.schema import GROUND_TRUTH_COLUMNS, SCHEMA_VERSION, sensor_kind_to_tipo
from .stat_utils import mad


@dataclass(frozen=True)
class AdaptedFrames:
    """Resultado comun de los adaptadores de entrada del artefacto."""

    entities: dict[str, dict[str, Any]]
    states: pd.DataFrame
    reference_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    ground_truth_lookup: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    ground_truth_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict[str, Any] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)


class CanonicalFrameAdapter:
    """Adapter base para convertir frames canonicos al modelo interno."""

    def entities_to_internal(self, entities: pd.DataFrame) -> dict[str, dict[str, Any]]:
        return domain_entities_to_internal(entities.to_dict("records"))

    def events_to_internal_states(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "entity_id",
                    "domain",
                    "state",
                    "last_changed",
                    "last_updated",
                    "unit_of_measurement",
                    "device_class",
                    "tipo_interno",
                ]
            )
        states = events.copy()
        states["last_changed"] = states["timestamp"]
        states["last_updated"] = states["timestamp"]
        return states

    def truth_to_lookup(self, ground_truth: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
        return {
            (str(row["timestamp"]), str(row["entity_id"])): row
            for row in ground_truth.to_dict("records")
        }


class StaticHistoryAdapter(CanonicalFrameAdapter):
    """Adapta CSV canonicos persistidos como historico estatico."""

    def __init__(
        self,
        entities: pd.DataFrame,
        events: pd.DataFrame,
        ground_truth: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> None:
        self.entities_frame = entities
        self.events_frame = events
        self.ground_truth_frame = ground_truth
        self.metadata_frame = metadata

    def adapt(self) -> AdaptedFrames:
        states = self.events_to_internal_states(self.events_frame)
        return AdaptedFrames(
            entities=self.entities_to_internal(self.entities_frame),
            states=states,
            reference_states=states.copy(),
            event_states=pd.DataFrame(),
            ground_truth_lookup=self.truth_to_lookup(self.ground_truth_frame) if not self.ground_truth_frame.empty else {},
            ground_truth_frame=self.ground_truth_frame.copy(),
            metadata=self.metadata_frame.copy(),
        )


class SimulationResultAdapter(CanonicalFrameAdapter):
    """Adapta la simulacion controlada al mismo modelo interno."""

    def __init__(self, simulation: Any) -> None:
        self.simulation = simulation

    def adapt(self) -> AdaptedFrames:
        entities_frame = self.simulation.entities.copy()
        entities_frame["selected"] = True
        entities = {
            str(row["entity_id"]): row.to_dict()
            for _, row in entities_frame.iterrows()
            if str(row.get("entity_id", "")).strip()
        }
        ground_truth = self.simulation.ground_truth.copy()
        return AdaptedFrames(
            entities=entities,
            states=self.simulation.states.copy(),
            reference_states=self.simulation.reference_states.copy(),
            event_states=self.simulation.event_states.copy(),
            ground_truth_lookup=self.truth_to_lookup(ground_truth),
            ground_truth_frame=ground_truth,
            config=dict(self.simulation.config),
            faults=[dict(fault) for fault in self.simulation.faults],
        )


class RealtimeEventsAdapter(CanonicalFrameAdapter):
    """Adapta eventos nuevos acumulados y su ground truth realtime."""

    def __init__(self, events: pd.DataFrame, ground_truth: pd.DataFrame) -> None:
        self.events = events
        self.ground_truth = ground_truth

    def adapt(self) -> AdaptedFrames:
        return AdaptedFrames(
            entities={},
            states=self.events_to_internal_states(self.events),
            ground_truth_lookup=self.truth_to_lookup(self.ground_truth) if not self.ground_truth.empty else {},
            ground_truth_frame=self.ground_truth.copy(),
        )


class RealHistoryAdapter(CanonicalFrameAdapter):
    """Adapta un history.csv real exportado desde Home Assistant.

    El archivo real no trae metadata de entidades ni etiquetas de falla. Este
    adapter infiere el tipo desde entity_id y deja ground_truth vacio para no
    asumir que todo el historico observado es normal.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def adapt(self) -> AdaptedFrames:
        raw = pd.read_csv(self.path, dtype=str, keep_default_na=False).fillna("")
        required = {"entity_id", "state", "last_changed"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise ValueError(f"Real history CSV is missing columns: {', '.join(missing)}")

        raw["timestamp_dt"] = pd.to_datetime(raw["last_changed"], errors="coerce", utc=True)
        raw["state_text"] = raw["state"].astype(str).str.strip()
        raw = raw[raw["timestamp_dt"].notna() & raw["state_text"].ne("")].copy()
        raw["state_lower"] = raw["state_text"].str.lower()
        raw = raw[~raw["state_lower"].isin({"unknown", "unavailable", "none", "null", "nan"})].copy()

        entity_rows = []
        event_frames = []
        for entity_id, group in raw.sort_values("timestamp_dt").groupby("entity_id", sort=True):
            entity = infer_real_entity(str(entity_id), group)
            if entity is None:
                continue
            entity_rows.append(entity)
            events = pd.DataFrame(
                {
                    "entity_id": str(entity_id),
                    "timestamp": group["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "state": group["state_text"].astype(str),
                    "unit_of_measurement": entity["unit_of_measurement"],
                    "device_class": entity["device_class"],
                    "domain": entity["domain"],
                    "tipo_interno": entity["tipo_interno"],
                }
            )
            event_frames.append(events)

        if not entity_rows:
            raise ValueError("Real history CSV does not contain supported PIR, magnetic, temperature or humidity entities")

        entities = pd.DataFrame(entity_rows)
        events = pd.concat(event_frames, ignore_index=True).sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
        metadata = pd.DataFrame(
            [
                {
                    "seed": "",
                    "start_timestamp": str(events["timestamp"].min()),
                    "end_timestamp": str(events["timestamp"].max()),
                    "duration_days": str(real_duration_days(events)),
                    "base_sample_seconds": "",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "schema_version": f"{SCHEMA_VERSION}-real-history",
                    "source": str(self.path),
                    "source_type": "real_history",
                }
            ]
        )
        ground_truth = pd.DataFrame(columns=GROUND_TRUTH_COLUMNS)
        states = self.events_to_internal_states(events)
        return AdaptedFrames(
            entities=self.entities_to_internal(entities),
            states=states,
            reference_states=states.copy(),
            # En datos reales sin etiquetas no basta usar el historico solo como
            # referencia: tambien debe evaluarse para detectar sospechas
            # historicas. La base se limpiara luego con reglas robustas.
            event_states=states.copy(),
            ground_truth_lookup={},
            ground_truth_frame=ground_truth,
            metadata=metadata,
            config={
                "source_type": "real_history",
                "source": str(self.path),
                "entities": int(len(entities)),
                "rows": int(len(events)),
                "duration_days": real_duration_days(events),
            },
        )


def infer_real_entity(entity_id: str, group: pd.DataFrame) -> dict[str, Any] | None:
    text = entity_id.lower()
    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    if domain == "binary_sensor" and "motion" in text:
        device_class = "motion"
        sensor_kind = "binary"
        unit = ""
    elif domain == "binary_sensor" and any(token in text for token in ["door", "contact", "magnetic", "window"]):
        device_class = "door"
        sensor_kind = "binary"
        unit = ""
    elif domain == "sensor" and any(token in text for token in ["temperature", "temp"]):
        device_class = "temperature"
        sensor_kind = "continuous"
        unit = "C"
    elif domain == "sensor" and "humidity" in text:
        device_class = "humidity"
        sensor_kind = "continuous"
        unit = "%"
    else:
        return None

    numeric = pd.to_numeric(group["state_text"], errors="coerce").dropna().astype(float)
    last_state = str(group.sort_values("timestamp_dt")["state_text"].iloc[-1])
    area_id = infer_area_id(entity_id)
    entity = {
        "entity_id": entity_id,
        "unit_of_measurement": unit,
        "device_class": device_class,
        "domain": domain,
        "tipo_interno": sensor_kind_to_tipo(sensor_kind),
        "area_id": area_id,
        "device_id": entity_id.replace(".", "_"),
        "integration": "real_history",
        "sensor_kind": sensor_kind,
        "last_state": last_state,
    }
    if not numeric.empty:
        values = numeric.tolist()
        entity["numeric_median"] = str(float(numeric.median()))
        entity["numeric_mad"] = str(float(mad(values)))
        entity["numeric_min"] = str(float(numeric.min()))
        entity["numeric_max"] = str(float(numeric.max()))
    return entity


def infer_area_id(entity_id: str) -> str:
    text = entity_id.lower()
    for area in ["bathroom", "bedroom", "kitchen", "foyer", "hall", "office", "entertainment", "front"]:
        if area in text:
            return area
    return "real"


def real_duration_days(events: pd.DataFrame) -> int:
    timestamps = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
    if timestamps.empty or timestamps.isna().all():
        return 0
    duration = timestamps.max() - timestamps.min()
    return int(max(1, round(duration.total_seconds() / 86400)))
