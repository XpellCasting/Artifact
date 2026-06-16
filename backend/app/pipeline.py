from __future__ import annotations

import os
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .canonical import (
    DEFAULT_HISTORY_DAYS,
    DEFAULT_SAMPLE_SECONDS,
    DEFAULT_SEED,
    ensure_static_history,
    load_static_history,
    next_realtime_events,
    validate_static_history_domain,
)
from .data_adapters import CanonicalFrameAdapter, RealHistoryAdapter, RealtimeEventsAdapter, SimulationResultAdapter, StaticHistoryAdapter
from .domain.schema import tipo_to_sensor_kind
from .estadistico import run_statistical_filter as execute_statistical_filter
from .features import build_binary_temporal_features
from .normalidad import clean_segment_record, select_clean_segments
from .domain.fault_types import DEFAULT_FAULTS
from .exporting import export_artifact_csv
from .simulacion import SimulationResult, generate_simulation
from .stat_utils import summarize_numeric, transition_counts


INVALID_STATES = {"", "unknown", "unavailable", "none", "null", "nan"}


@dataclass
class ArtifactState:
    dataset_dir: Path
    fixture_dir: Path
    static_history_dir: Path = field(default_factory=lambda: Path("data/static_history"))
    real_history_path: Path = field(default_factory=lambda: Path("data/datosreales/history.csv"))
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    states: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_states: pd.DataFrame = field(default_factory=pd.DataFrame)
    ground_truth: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    selected_entities: set[str] = field(default_factory=set)
    validation_report: dict[str, Any] = field(default_factory=dict)
    normalized: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_normalized: pd.DataFrame = field(default_factory=pd.DataFrame)
    event_normalized: pd.DataFrame = field(default_factory=pd.DataFrame)
    reference_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    event_features: dict[str, pd.DataFrame] = field(default_factory=dict)
    baseline: dict[str, dict[str, Any]] = field(default_factory=dict)
    results: list[dict[str, Any]] = field(default_factory=list)
    historical_results: list[dict[str, Any]] = field(default_factory=list)
    historical_summary: dict[str, Any] = field(default_factory=dict)
    simulation_config: dict[str, Any] = field(default_factory=dict)
    simulation_faults: list[dict[str, Any]] = field(default_factory=lambda: [dict(fault) for fault in DEFAULT_FAULTS])
    simulation_ground_truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    static_entities: pd.DataFrame = field(default_factory=pd.DataFrame)
    static_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    static_ground_truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    static_metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_ground_truth: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_current_timestamp: str = ""
    realtime_tick_index: int = 0
    realtime_fault_windows: list[dict[str, Any]] = field(default_factory=list)

    def load_simulation(self, simulation: SimulationResult) -> dict[str, Any]:
        # El simulador reemplaza al historico externo como fuente principal del
        # artefacto. Se cargan DataFrames con el mismo formato HA para reutilizar
        # validacion, normalizacion, baseline y filtro.
        adapted = SimulationResultAdapter(simulation).adapt()
        self.entities = adapted.entities
        self.states = adapted.states
        self.reference_states = adapted.reference_states
        self.event_states = adapted.event_states
        self.simulation_ground_truth = adapted.ground_truth_frame
        self.simulation_config = adapted.config
        self.simulation_faults = adapted.faults
        self.ground_truth = adapted.ground_truth_lookup
        self.selected_entities = set(self.entities)
        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.results = []
        self.historical_results = []
        self.historical_summary = {}
        self.historical_results = []
        self.historical_summary = {}
        return self.current_simulation()

    def generate_simulation(
        self,
        duration_hours: int = 72,
        sample_seconds: int = 10,
        seed: int = 20260614,
        faults: list[dict[str, Any]] | None = None,
        selected_entities: list[str] | None = None,
    ) -> dict[str, Any]:
        simulation = generate_simulation(
            duration_hours=duration_hours,
            sample_seconds=sample_seconds,
            seed=seed,
            faults=faults if faults is not None else self.simulation_faults,
            selected_entities=selected_entities,
        )
        return self.load_simulation(simulation)

    def set_simulation_faults(self, faults: list[dict[str, Any]]) -> dict[str, Any]:
        self.simulation_faults = faults
        return {"faults": self.simulation_faults}

    def load_real_history(self, path: str | None = None) -> dict[str, Any]:
        source_path = Path(path) if path else self.real_history_path
        if not source_path.exists():
            raise ValueError(f"Real history CSV not found: {source_path}")

        adapted = RealHistoryAdapter(source_path).adapt()
        self.static_entities = pd.DataFrame(adapted.entities.values())
        self.static_events = adapted.states[
            ["entity_id", "timestamp", "state", "unit_of_measurement", "device_class", "domain", "tipo_interno"]
        ].copy()
        self.static_ground_truth = adapted.ground_truth_frame
        self.static_metadata = adapted.metadata
        self.entities = adapted.entities
        self.states = adapted.states
        self.reference_states = adapted.reference_states
        self.event_states = adapted.event_states
        self.selected_entities = set(self.entities)
        self.ground_truth = {}
        self.simulation_ground_truth = pd.DataFrame()
        self.simulation_config = adapted.config
        self.validation_report = {}
        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.realtime_events = pd.DataFrame()
        self.realtime_ground_truth = pd.DataFrame()
        self.realtime_current_timestamp = ""
        self.realtime_tick_index = 0
        self.realtime_fault_windows = []
        self.results = []
        self.historical_summary = self.run_historical_filter()
        status = self.static_history_status()
        status["source_type"] = "real_history"
        status["loaded"] = True
        status["historical_summary"] = self.historical_summary
        status["historical_results"] = self.historical_results
        return status

    def ensure_static_history(self) -> dict[str, Any]:
        history = ensure_static_history(
            self.static_history_dir,
            duration_days=DEFAULT_HISTORY_DAYS,
            base_sample_seconds=DEFAULT_SAMPLE_SECONDS,
            seed=DEFAULT_SEED,
        )
        self.load_static_history_frames(history.entities, history.events, history.ground_truth, history.metadata)
        status = self.static_history_status()
        status["created"] = history.created
        return status

    def static_history_status(self) -> dict[str, Any]:
        if self.static_events.empty and (self.static_history_dir / "events.csv").exists():
            try:
                history = load_static_history(self.static_history_dir, load_ground_truth=False)
                validate_static_history_domain(history, self.static_history_dir)
            except ValueError as error:
                return {
                    "exists": True,
                    "valid": False,
                    "directory": str(self.static_history_dir),
                    "error": str(error),
                    "entities": [],
                    "entity_count": 0,
                    "rows": 0,
                    "ground_truth_rows": 0,
                    "start": None,
                    "end": None,
                    "metadata": {},
                }
            self.load_static_history_frames(history.entities, history.events, history.ground_truth, history.metadata)
        exists = not self.static_events.empty
        metadata = self.static_metadata.to_dict("records")[0] if not self.static_metadata.empty else {}
        return {
            "exists": exists,
            "valid": exists,
            "directory": str(self.static_history_dir),
            "entities": self.list_entities() if exists else [],
            "entity_count": int(len(self.static_entities)) if exists else 0,
            "rows": int(len(self.static_events)) if exists else 0,
            "ground_truth_rows": int(len(self.static_ground_truth)) if exists else 0,
            "start": str(self.static_events["timestamp"].min()) if exists else None,
            "end": str(self.static_events["timestamp"].max()) if exists else None,
            "metadata": metadata,
            "source_type": str(metadata.get("source_type", "static_history")) if metadata else "static_history",
        }

    def load_static_history_frames(
        self,
        entities: pd.DataFrame,
        events: pd.DataFrame,
        ground_truth: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> None:
        self.static_entities = entities.copy()
        self.static_events = events.copy()
        self.static_ground_truth = ground_truth.copy()
        self.static_metadata = metadata.copy()
        adapted = StaticHistoryAdapter(entities, events, ground_truth, metadata).adapt()
        self.entities = adapted.entities
        self.states = adapted.states
        self.reference_states = adapted.reference_states
        self.event_states = adapted.event_states
        self.selected_entities = set(self.entities)
        self.ground_truth = adapted.ground_truth_lookup
        self.simulation_ground_truth = pd.DataFrame()
        self.simulation_config = {}
        self.validation_report = {}
        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.reset_realtime()
        self.historical_results = []
        self.historical_summary = {}

    def reset_realtime(self) -> dict[str, Any]:
        self.realtime_events = pd.DataFrame()
        self.realtime_ground_truth = pd.DataFrame()
        self.realtime_current_timestamp = ""
        self.realtime_tick_index = 0
        self.realtime_fault_windows = []
        self.event_states = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.event_features = {}
        self.results = []
        return self.current_realtime()

    def realtime_tick(
        self,
        selected_entities: list[str],
        tick_seconds: int = 60,
        faults: list[dict[str, Any]] | None = None,
        suspicion_threshold: float = 0.5,
    ) -> dict[str, Any]:
        if self.static_events.empty:
            self.ensure_static_history()
        known = set(self.entities)
        selected = [entity_id for entity_id in selected_entities if entity_id in known]
        if not selected:
            raise ValueError("No valid entities selected for realtime tick")

        tick_seconds = max(10, int(tick_seconds))
        next_timestamp = self.next_realtime_timestamp(tick_seconds)
        new_events, new_truth, active_faults = next_realtime_events(
            self.static_entities,
            next_timestamp,
            selected,
            faults or [],
            self.realtime_tick_index,
        )
        self.realtime_tick_index += 1
        self.realtime_current_timestamp = next_timestamp.isoformat()
        for fault in active_faults:
            fault["end"] = (next_timestamp + pd.Timedelta(seconds=tick_seconds)).isoformat()
        self.realtime_fault_windows.extend(active_faults)
        if not new_events.empty:
            self.realtime_events = pd.concat([self.realtime_events, new_events], ignore_index=True)
        if not new_truth.empty:
            self.realtime_ground_truth = pd.concat([self.realtime_ground_truth, new_truth], ignore_index=True)

        self.selected_entities = set(selected)
        adapted_realtime = RealtimeEventsAdapter(self.realtime_events, self.realtime_ground_truth).adapt()
        self.event_states = adapted_realtime.states
        self.ground_truth = adapted_realtime.ground_truth_lookup
        self.normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.event_features = {}
        self.normalize()
        missing_baseline = [entity_id for entity_id in selected if entity_id not in self.baseline]
        if self.baseline == {} or missing_baseline:
            self.build_baseline()
        summary = self.run_statistical_filter(suspicion_threshold=suspicion_threshold)
        missing_added = self.add_realtime_missing_results(selected, next_timestamp, tick_seconds, suspicion_threshold)
        summary["missing_results_added"] = missing_added
        summary = self.refresh_filter_summary(summary)
        return self.current_realtime(summary)

    def next_realtime_timestamp(self, tick_seconds: int) -> pd.Timestamp:
        if self.realtime_current_timestamp:
            return pd.to_datetime(self.realtime_current_timestamp) + pd.Timedelta(seconds=tick_seconds)
        last_static = pd.to_datetime(self.static_events["timestamp"], errors="coerce").max()
        return last_static + pd.Timedelta(seconds=tick_seconds)

    def current_realtime(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        labels = {}
        if not self.realtime_ground_truth.empty and "label" in self.realtime_ground_truth:
            labels = self.realtime_ground_truth["label"].value_counts().to_dict()
        return {
            "current_timestamp": self.realtime_current_timestamp,
            "tick_index": self.realtime_tick_index,
            "events": int(len(self.realtime_events)),
            "ground_truth_rows": int(len(self.realtime_ground_truth)),
            "ground_truth_labels": {str(key): int(value) for key, value in labels.items()},
            "results": self.results,
            "historical_results": self.historical_results,
            "result_count": len(self.results),
            "historical_result_count": len(self.historical_results),
            "summary": summary or {},
            "historical_summary": self.historical_summary,
            "faults": self.group_realtime_fault_windows(),
            "fault_ticks": self.realtime_fault_windows,
        }

    def realtime_timeline(
        self,
        max_points: int = 900,
        context_hours: int = 24,
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        if self.static_events.empty:
            return {"entities": [], "faults": []}
        context = self.timeline_range_frame(self.static_events, start=start, end=end, context_hours=context_hours)
        if self.selected_entities:
            context = context[context["entity_id"].isin(self.selected_entities)]
        combined = context
        if not self.realtime_events.empty:
            realtime_context = self.timeline_range_frame(self.realtime_events, start=start, end=end, context_hours=context_hours)
            combined = pd.concat([context, realtime_context], ignore_index=True)
        state_frame = CanonicalFrameAdapter().events_to_internal_states(combined)
        payload = self.timeline_from_states(state_frame, max_points=max_points, faults=self.group_realtime_fault_windows())
        if not self.realtime_events.empty:
            payload["realtime_start"] = str(self.realtime_events["timestamp"].min())
            payload["realtime_end"] = str(self.realtime_events["timestamp"].max())
        else:
            payload["realtime_start"] = None
            payload["realtime_end"] = None
        return payload

    def timeline_range_frame(
        self,
        frame: pd.DataFrame,
        start: str | None = None,
        end: str | None = None,
        context_hours: int = 24,
    ) -> pd.DataFrame:
        working = frame.copy()
        if working.empty:
            return working
        working["timestamp_dt"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
        working = working[working["timestamp_dt"].notna()]
        start_dt = parse_timeline_timestamp(start)
        end_dt = parse_timeline_timestamp(end)
        if pd.notna(start_dt) and pd.notna(end_dt) and start_dt > end_dt:
            start_dt, end_dt = end_dt, start_dt

        if pd.notna(start_dt) or pd.notna(end_dt):
            if pd.notna(start_dt):
                working = working[working["timestamp_dt"] >= start_dt]
            if pd.notna(end_dt):
                working = working[working["timestamp_dt"] <= end_dt]
        else:
            if self.realtime_current_timestamp:
                cutoff = parse_timeline_timestamp(self.realtime_current_timestamp) - pd.Timedelta(hours=context_hours)
            else:
                cutoff = working["timestamp_dt"].max() - pd.Timedelta(hours=context_hours)
            working = working[working["timestamp_dt"] >= cutoff]
        return working.drop(columns=["timestamp_dt"])

    def add_realtime_missing_results(
        self,
        selected_entities: list[str],
        timestamp: pd.Timestamp,
        tick_seconds: int,
        suspicion_threshold: float,
    ) -> int:
        if self.realtime_ground_truth.empty:
            return 0
        added = 0
        missing_truth = self.realtime_ground_truth[
            self.realtime_ground_truth["timestamp"].eq(timestamp.isoformat())
            & self.realtime_ground_truth["fault_type"].eq("missing")
        ]
        required_missing_ticks = max(2, int(round(1 / max(0.01, suspicion_threshold))))
        for entity_id in selected_entities:
            entity_missing = missing_truth[missing_truth["entity_id"].eq(entity_id)]
            if entity_missing.empty:
                continue
            consecutive = self.consecutive_missing_ticks(entity_id)
            if consecutive < required_missing_ticks:
                continue
            raw_score = min(1.0, 0.45 + (consecutive - required_missing_ticks + 1) * 0.15)
            probability_tau = 0.35
            suspicion_score = raw_score / (raw_score + probability_tau)
            result = {
                "entity_id": entity_id,
                "timestamp": timestamp.isoformat(),
                "window_start": timestamp.isoformat(),
                "window_end": timestamp.isoformat(),
                "sensor_kind": self.entities.get(entity_id, {}).get("sensor_kind", "unknown"),
                "tipo_interno": self.entities.get(entity_id, {}).get("tipo_interno", ""),
                "observed_state": "",
                "value": None,
                "reading": {"missing_ticks": consecutive, "gap_seconds": consecutive * tick_seconds},
                "raw_score": round(raw_score, 4),
                "suspicion_score": round(suspicion_score, 4),
                "is_suspicious": True,
                "pattern": "missing",
                "suspected_pattern": "missing",
                "pattern_detail": "missing_row_gap",
                "filter": "statistical_temporal",
                "passed_to_contextual_filter": True,
                "triggered_indicators": ["missing_row", "gap"],
                "baseline_summary": {},
                "ground_truth_label": "sensor_fault",
            }
            self.results.insert(0, result)
            added += 1
        return added

    def refresh_filter_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        self.results.sort(key=lambda item: item.get("suspicion_score", 0), reverse=True)
        self.results = self.results[:1000]
        by_kind = Counter(str(item.get("sensor_kind", "unknown")) for item in self.results)
        summary["results"] = len(self.results)
        summary["suspicious"] = sum(1 for item in self.results if item.get("is_suspicious"))
        summary["by_kind"] = {str(kind): int(count) for kind, count in by_kind.items()}
        summary["top_results"] = self.results[:50]
        return summary

    def group_realtime_fault_windows(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], dict[str, Any]] = {}
        for fault in sorted(self.realtime_fault_windows, key=lambda item: (item.get("entity_id", ""), item.get("fault_type", ""), item.get("start", ""))):
            key = (str(fault.get("entity_id", "")), str(fault.get("fault_type", "")))
            start = pd.to_datetime(fault.get("start"), errors="coerce")
            end = pd.to_datetime(fault.get("end"), errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue
            current = grouped.get(key)
            if not current:
                grouped[key] = {
                    "entity_id": key[0],
                    "fault_type": key[1],
                    "enabled": True,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "ticks": 1,
                }
                continue

            current_end = pd.to_datetime(current["end"], errors="coerce")
            if pd.notna(current_end) and start <= current_end + pd.Timedelta(seconds=1):
                current["end"] = max(current_end, end).isoformat()
                current["ticks"] = int(current["ticks"]) + 1
            else:
                # Si hay una pausa, se conserva el tramo mas reciente porque es
                # el que importa para el chart de tiempo real actual.
                grouped[key] = {
                    "entity_id": key[0],
                    "fault_type": key[1],
                    "enabled": True,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "ticks": 1,
                }
        return list(grouped.values())

    def consecutive_missing_ticks(self, entity_id: str) -> int:
        if self.realtime_ground_truth.empty:
            return 0
        ordered = self.realtime_ground_truth[
            self.realtime_ground_truth["entity_id"].eq(entity_id)
        ].sort_values("timestamp", ascending=False)
        count = 0
        for row in ordered.to_dict("records"):
            if str(row.get("fault_type", "")) == "missing":
                count += 1
            else:
                break
        return count

    def current_simulation(self) -> dict[str, Any]:
        labels = {}
        if not self.simulation_ground_truth.empty:
            labels = self.simulation_ground_truth["label"].value_counts().to_dict()
        return {
            "config": self.simulation_config,
            "faults": self.simulation_faults,
            "entities": self.list_entities(),
            "rows": int(len(self.states)),
            "ground_truth_labels": {str(key): int(value) for key, value in labels.items()},
        }

    def simulation_timeline(self, max_points: int = 300) -> dict[str, Any]:
        if self.states.empty:
            return {"entities": []}
        return self.timeline_from_states(self.states, max_points=max_points, faults=self.simulation_faults)

    def timeline_from_states(
        self,
        states: pd.DataFrame,
        max_points: int = 300,
        faults: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        entity_lookup = {entity_id: dict(entity) for entity_id, entity in self.entities.items()}
        timelines = []
        if states.empty:
            return {"entities": [], "faults": faults or []}
        for entity_id, group in states.sort_values("timestamp").groupby("entity_id"):
            entity = entity_lookup.get(str(entity_id), {})
            kind = str(entity.get("sensor_kind", "multistate"))
            categories: dict[str, int] = {}
            if kind == "binary":
                points, bucket_seconds = binary_percentage_timeline(group.reset_index(drop=True), max_points)
                chart_kind = "active_percentage"
                scale_unit = "%"
            else:
                sampled = downsample_frame(group.reset_index(drop=True), max_points)
                points = []
                for row in sampled.to_dict("records"):
                    state = str(row["state"])
                    if kind == "continuous":
                        value = float(state)
                    else:
                        if state not in categories:
                            categories[state] = len(categories)
                        value = float(categories[state])
                    points.append({"timestamp": str(row["timestamp"]), "state": state, "value": value})
                bucket_seconds = None
                chart_kind = "state_line"
                scale_unit = "state"
            timelines.append(
                {
                    "entity_id": str(entity_id),
                    "sensor_kind": kind,
                    "area_id": str(entity.get("area_id", "")),
                    "chart_kind": chart_kind,
                    "scale_unit": scale_unit,
                    "bucket_seconds": bucket_seconds,
                    "points": points,
                    "categories": {str(value): key for key, value in categories.items()},
                }
            )
        return {"entities": timelines, "faults": faults or []}

    def export_csv(self, name: str) -> str:
        return export_artifact_csv(name, self.states, self.entities, self.simulation_ground_truth)

    def load_dataset(self, dataset_dir: str | None = None) -> dict[str, Any]:
        if dataset_dir:
            self.dataset_dir = Path(dataset_dir)
        required = ["ha_entities.csv", "ha_states.csv"]
        missing = [name for name in required if not (self.dataset_dir / name).exists()]
        if missing:
            raise ValueError(f"Missing required dataset files: {', '.join(missing)}")

        # Pandas queda a cargo de leer los CSV grandes de Home Assistant. Se usan
        # strings para no perder estados categoricos como "off", "heat" o "unknown".
        entities_df = read_csv_frame(self.dataset_dir / "ha_entities.csv")
        entities_df["sensor_kind"] = entities_df.apply(lambda row: classify_entity(row.to_dict()), axis=1)
        entities_df["selected"] = True
        self.entities = {
            str(row["entity_id"]): row.to_dict()
            for _, row in entities_df.iterrows()
            if str(row.get("entity_id", "")).strip()
        }

        self.states = read_csv_frame(self.dataset_dir / "ha_states.csv")
        self.reference_states = pd.DataFrame()
        self.event_states = pd.DataFrame()
        self.selected_entities = set(self.entities)

        truth_path = self.dataset_dir / "ground_truth.csv"
        self.ground_truth = {}
        if truth_path.exists():
            truth_df = read_csv_frame(truth_path)
            for row in truth_df.to_dict("records"):
                timestamp = str(row.get("timestamp", ""))
                entity_id = str(row.get("entity_id", ""))
                if timestamp and entity_id:
                    self.ground_truth[(timestamp, entity_id)] = row

        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.results = []
        self.historical_results = []
        self.historical_summary = {}

        return {
            "dataset_dir": str(self.dataset_dir),
            "entities": len(self.entities),
            "state_rows": int(len(self.states)),
            "ground_truth_rows": len(self.ground_truth),
        }

    def load_multistate_fixture(self) -> dict[str, Any]:
        entities_path = self.fixture_dir / "multistate_entities.csv"
        states_path = self.fixture_dir / "multistate_states.csv"
        if not entities_path.exists() or not states_path.exists():
            raise ValueError("Multistate fixture files are missing")

        entities_df = read_csv_frame(entities_path)
        entities_df["sensor_kind"] = "multistate"
        entities_df["selected"] = True

        added_entities = 0
        for row in entities_df.to_dict("records"):
            entity_id = str(row.get("entity_id", "")).strip()
            if not entity_id:
                continue
            self.entities[entity_id] = row
            self.selected_entities.add(entity_id)
            added_entities += 1

        fixture_states = read_csv_frame(states_path)
        self.states = pd.concat([self.states, fixture_states], ignore_index=True)

        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.results = []

        return {"entities_added": added_entities, "state_rows_added": int(len(fixture_states))}

    def list_entities(self) -> list[dict[str, Any]]:
        counts = {}
        if not self.states.empty and "entity_id" in self.states:
            counts = self.states["entity_id"].value_counts().to_dict()

        output = []
        for entity_id, entity in sorted(self.entities.items()):
            item = dict(entity)
            item["selected"] = entity_id in self.selected_entities
            item["rows"] = int(counts.get(entity_id, 0))
            output.append(item)
        return output

    def select_entities(self, entity_ids: list[str]) -> dict[str, Any]:
        unknown = sorted(set(entity_ids) - set(self.entities))
        if unknown:
            raise ValueError(f"Unknown entities: {', '.join(unknown)}")
        self.selected_entities = set(entity_ids)
        self.normalized = pd.DataFrame()
        self.reference_normalized = pd.DataFrame()
        self.event_normalized = pd.DataFrame()
        self.reference_features = {}
        self.event_features = {}
        self.baseline = {}
        self.results = []
        self.historical_results = []
        self.historical_summary = {}
        return {"selected": len(self.selected_entities)}

    def validate(self) -> dict[str, Any]:
        required_state_cols = {
            "timestamp",
            "entity_id",
            "domain",
            "state",
            "last_changed",
            "last_updated",
        }
        missing_cols = sorted(required_state_cols - set(self.states.columns))
        if self.states.empty or missing_cols:
            self.validation_report = {
                "missing_columns": missing_cols,
                "selected_entities": len(self.selected_entities),
                "checked_rows": 0,
                "invalid_rows": 0,
                "timestamp_errors": 0,
                "entities": [],
            }
            return self.validation_report

        # La validacion trabaja solo sobre entidades seleccionadas. Esto permite
        # excluir sensores irrelevantes sin alterar el historico original.
        df = self.states[self.states["entity_id"].isin(self.selected_entities)].copy()
        df["state_text"] = df["state"].astype(str).str.strip().str.lower()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")

        invalid_mask = df["state_text"].isin(INVALID_STATES)
        timestamp_error_mask = df["timestamp_dt"].isna()

        rows_by_entity = df.groupby("entity_id").size()
        invalid_by_entity = df[invalid_mask].groupby("entity_id").size()
        parse_errors = df[timestamp_error_mask].groupby("entity_id").size()

        continuity = {}
        valid_ts = df[~timestamp_error_mask].sort_values(["entity_id", "timestamp_dt"])
        for entity_id, group in valid_ts.groupby("entity_id"):
            gaps = group["timestamp_dt"].diff().dt.total_seconds()
            large_gaps = gaps[gaps > 600]
            continuity[str(entity_id)] = {
                "first": group["timestamp"].iloc[0],
                "last": group["timestamp"].iloc[-1],
                "large_gaps": int(large_gaps.count()),
                "max_gap_seconds": float(large_gaps.max()) if not large_gaps.empty else 0,
            }

        entity_reports = []
        for entity_id in sorted(self.selected_entities):
            rows = int(rows_by_entity.get(entity_id, 0))
            entity_reports.append(
                {
                    "entity_id": entity_id,
                    "sensor_kind": self.entities.get(entity_id, {}).get("sensor_kind", "unknown"),
                    "rows": rows,
                    "invalid_rows": int(invalid_by_entity.get(entity_id, 0)),
                    "timestamp_errors": int(parse_errors.get(entity_id, 0)),
                    "sufficient_data": rows >= 30,
                    "continuity": continuity.get(entity_id, {}),
                }
            )

        self.validation_report = {
            "missing_columns": missing_cols,
            "selected_entities": len(self.selected_entities),
            "checked_rows": int(len(df)),
            "invalid_rows": int(invalid_mask.sum()),
            "timestamp_errors": int(timestamp_error_mask.sum()),
            "entities": entity_reports,
        }
        return self.validation_report

    def normalize(self) -> dict[str, Any]:
        if self.states.empty:
            self.normalized = pd.DataFrame()
            self.reference_normalized = pd.DataFrame()
            self.event_normalized = pd.DataFrame()
            self.reference_features = {}
            self.event_features = {}
            return {"entities": 0, "rows": 0, "by_kind": {}}

        self.reference_features = {}
        self.event_features = {}
        if not self.reference_states.empty:
            self.reference_normalized = self.normalize_frame(self.reference_states)
            self.event_normalized = self.normalize_frame(self.event_states) if not self.event_states.empty else pd.DataFrame()
            self.normalized = self.event_normalized if not self.event_normalized.empty else self.reference_normalized
            by_kind = self.normalized.groupby("sensor_kind").size().to_dict()
            return {
                "entities": int(self.normalized["entity_id"].nunique()),
                "rows": int(len(self.normalized)),
                "reference_rows": int(len(self.reference_normalized)),
                "event_rows": int(len(self.event_normalized)),
                "by_kind": {str(key): int(value) for key, value in by_kind.items()},
            }

        self.normalized = self.normalize_frame(self.states)
        by_kind = self.normalized.groupby("sensor_kind").size().to_dict()
        return {
            "entities": int(self.normalized["entity_id"].nunique()),
            "rows": int(len(self.normalized)),
            "by_kind": {str(key): int(value) for key, value in by_kind.items()},
        }

    def build_feature_tables(self) -> dict[str, Any]:
        if self.normalized.empty and self.reference_normalized.empty:
            self.normalize()

        self.reference_features = self.build_binary_feature_tables(self.reference_normalized)
        self.event_features = self.build_binary_feature_tables(self.event_normalized)
        if not self.event_features and not self.normalized.empty:
            self.event_features = self.build_binary_feature_tables(self.normalized)

        return {
            "reference_entities": len(self.reference_features),
            "event_entities": len(self.event_features),
            "reference_rows": int(sum(len(frame) for frame in self.reference_features.values())),
            "event_rows": int(sum(len(frame) for frame in self.event_features.values())),
        }

    def build_binary_feature_tables(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        if frame.empty:
            return {}

        feature_tables: dict[str, pd.DataFrame] = {}
        binary_frame = frame[frame["sensor_kind"].eq("binary")]
        for entity_id, rows in binary_frame.groupby("entity_id", sort=True):
            # La informacion binaria se compacta por ventanas temporales: A_W,
            # C_W, T_on,W y P_on,W. Esta tabla es la entrada real del filtro.
            step_seconds = 60 if len(rows) > 10000 else 10
            feature_tables[str(entity_id)] = build_binary_temporal_features(rows, window_seconds=60, step_seconds=step_seconds)
        return feature_tables

    def normalize_frame(self, states: pd.DataFrame) -> pd.DataFrame:
        # Normalizacion: se crea una tabla comun con timestamp parseado, estado
        # textual, tipo de sensor y valor numerico cuando corresponde.
        df = states[states["entity_id"].isin(self.selected_entities)].copy()
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["state"] = df["state"].astype(str).str.strip().str.lower()
        df = df[~df["state"].isin(INVALID_STATES) & df["timestamp_dt"].notna()].copy()

        kind_map = {entity_id: entity.get("sensor_kind", "multistate") for entity_id, entity in self.entities.items()}
        if "tipo_interno" in df.columns:
            df["sensor_kind"] = df["tipo_interno"].map(lambda value: tipo_to_sensor_kind(str(value)))
        else:
            df["sensor_kind"] = df["entity_id"].map(kind_map).fillna("multistate")
        df["timestamp_iso"] = df["timestamp"].astype(str)
        df["value"] = pd.NA

        continuous_mask = df["sensor_kind"].eq("continuous")
        df.loc[continuous_mask, "value"] = pd.to_numeric(df.loc[continuous_mask, "state"], errors="coerce")
        df = df[~(continuous_mask & df["value"].isna())].copy()

        binary_mask = df["sensor_kind"].eq("binary")
        active_states = {"on", "open", "home", "active", "detected"}
        df.loc[binary_mask, "value"] = df.loc[binary_mask, "state"].isin(active_states).astype(float)

        columns = ["timestamp_dt", "timestamp_iso", "entity_id", "sensor_kind", "state", "value"]
        return df[columns].sort_values(["entity_id", "timestamp_dt"]).reset_index(drop=True)

    def build_baseline(self, train_ratio: float = 0.35) -> dict[str, Any]:
        if self.normalized.empty and self.reference_normalized.empty:
            self.normalize()

        raw_baseline_source = self.reference_normalized if not self.reference_normalized.empty else self.normalized
        baseline_source = select_clean_segments(raw_baseline_source, self.ground_truth)
        event_mode = not self.reference_normalized.empty and not self.event_normalized.empty
        reference_mode = not self.reference_normalized.empty
        clean_simulation_normalized = self.clean_simulation_normalized_frame()
        self.build_feature_tables()
        baseline: dict[str, dict[str, Any]] = {}
        segments = []
        for entity_id, rows in baseline_source.groupby("entity_id", sort=True):
            rows = rows.sort_values("timestamp_dt").reset_index(drop=True)
            kind = str(rows["sensor_kind"].iloc[0])
            if reference_mode:
                train_count = len(rows)
            elif kind == "multistate":
                train_count = multistate_cycle_train_count(rows)
            else:
                train_count = min(len(rows), max(10, int(len(rows) * train_ratio)))
            training = rows.iloc[:train_count].copy()

            # La base de normalidad se calcula solo con el tramo inicial. Mas
            # adelante HU-08/HU-09 pueden reemplazar esta seleccion por segmentos
            # aprobados por el usuario.
            if kind == "continuous":
                values = pd.to_numeric(training["value"], errors="coerce").dropna()
                summary = summarize_numeric(values.astype(float).tolist())
                summary["rate"] = summarize_numeric(rate_changes_frame(training))
                summary["window_stdev"] = summarize_numeric(
                    values.rolling(12).std().dropna().astype(float).tolist()
                )
                summary["hourly"] = build_hourly_numeric_baseline(training)
                summary["residual_slope"] = summarize_numeric(
                    residual_slopes_frame(training, window_minutes=120)
                )
                summary["value_slope"] = summarize_numeric(
                    value_slopes_frame(training, window_minutes=120)
                )
                summary["cusum_window_minutes"] = 180
                summary["drift_window_minutes"] = 120
                # tau_i de P_i^sos(t)=S_i(t)/(S_i(t)+tau_i). S_i(t) ya viene
                # acotado en 0..1 desde los componentes de la tesis.
                summary["threshold"] = 0.40
                baseline[str(entity_id)] = build_baseline_item(str(entity_id), kind, training, rows, summary, event_mode)
            elif kind == "binary":
                if "entity_id" in clean_simulation_normalized.columns:
                    clean_training = clean_simulation_normalized[
                        clean_simulation_normalized["entity_id"].eq(str(entity_id))
                    ].copy()
                else:
                    clean_training = pd.DataFrame()
                if not clean_training.empty:
                    # En simulacion controlada existe una serie limpia. Para el
                    # baseline binario conviene usarla completa, porque el
                    # primer tramo puede no incluir rutinas normales de otras
                    # horas, como actividad nocturna del PIR.
                    training = clean_training.sort_values("timestamp_dt").reset_index(drop=True)
                    metrics = build_binary_temporal_features(training, window_seconds=60, step_seconds=10)
                else:
                    metrics = self.reference_features.get(str(entity_id))
                    if metrics is None or metrics.empty or not event_mode:
                        metrics = build_binary_temporal_features(training, window_seconds=60, step_seconds=10)
                summary = {
                    "activations": summarize_numeric(metrics["activations"].tolist()),
                    "state_changes": summarize_numeric(metrics["state_changes"].tolist()),
                    "active_seconds": summarize_numeric(metrics["active_seconds"].tolist()),
                    "active_ratio": summarize_numeric(metrics["active_ratio"].tolist()),
                    "max_active_seconds": summarize_numeric(metrics["max_active_seconds"].tolist()),
                    "gap": summarize_numeric(metrics["gap"].tolist()),
                    "hourly": build_hourly_binary_baseline(metrics),
                    "mean_activation_interval": summarize_numeric(metrics["mean_activation_interval"].dropna().tolist()),
                    "percentiles": {
                        "activations_p95": float(metrics["activations"].quantile(0.95)) if not metrics.empty else 0.0,
                        "state_changes_p95": float(metrics["state_changes"].quantile(0.95)) if not metrics.empty else 0.0,
                        "active_seconds_p95": float(metrics["active_seconds"].quantile(0.95)) if not metrics.empty else 0.0,
                        "active_ratio_p95": float(metrics["active_ratio"].quantile(0.95)) if not metrics.empty else 0.0,
                        "max_active_seconds_p95": float(metrics["max_active_seconds"].quantile(0.95)) if not metrics.empty else 0.0,
                        "gap_p95": float(metrics["gap"].quantile(0.95)) if not metrics.empty else 0.0,
                    },
                    "window_seconds": 60,
                    "step_seconds": 10,
                    "threshold": 0.35,
                }
                baseline[str(entity_id)] = build_baseline_item(str(entity_id), kind, training, rows, summary, event_mode)
            else:
                states = training["state"].astype(str)
                durations = state_durations_frame(training)
                summary = {
                    "states": {str(key): int(value) for key, value in states.value_counts().to_dict().items()},
                    "state_set": sorted(states.unique().tolist()),
                    "transitions": {key: dict(value) for key, value in transition_counts(states.tolist()).items()},
                    "durations": {state: summarize_numeric(values) for state, values in durations.items()},
                    "threshold": 0.30,
                }
                baseline[str(entity_id)] = build_baseline_item(str(entity_id), "multistate", training, rows, summary, event_mode)

            segment = clean_segment_record(str(entity_id), training)
            if segment:
                segments.append(segment)

        self.baseline = baseline
        return {"entities": len(self.baseline), "segments": segments, "baseline": self.baseline}

    def run_statistical_filter(self, suspicion_threshold: float = 0.5) -> dict[str, Any]:
        if self.baseline == {}:
            self.build_baseline()

        if not self.reference_normalized.empty:
            evaluation = self.event_normalized
            feature_tables = self.event_features
        else:
            evaluation = self.normalized
            feature_tables = None
        summary = execute_statistical_filter(
            evaluation,
            self.baseline,
            self.ground_truth,
            feature_tables=feature_tables,
            suspicion_threshold=suspicion_threshold,
        )
        self.results = summary.pop("all_results")
        return summary

    def run_historical_filter(self, suspicion_threshold: float = 0.5) -> dict[str, Any]:
        if self.reference_normalized.empty:
            self.normalize()
        if self.baseline == {}:
            self.build_baseline()
        if not self.reference_features and not self.reference_normalized.empty:
            self.build_feature_tables()

        summary = execute_statistical_filter(
            self.reference_normalized,
            self.baseline,
            {},
            feature_tables=self.reference_features,
            suspicion_threshold=suspicion_threshold,
        )
        self.historical_results = [
            dict(item, result_source="historical")
            for item in summary.pop("all_results")
        ]
        self.historical_summary = summary
        return summary

    def clean_simulation_normalized_frame(self) -> pd.DataFrame:
        if self.simulation_ground_truth.empty or "clean_state" not in self.simulation_ground_truth.columns:
            return pd.DataFrame()

        clean_states = self.simulation_ground_truth[
            ["timestamp", "entity_id", "sensor_kind", "clean_state"]
        ].copy()
        clean_states = clean_states[clean_states["entity_id"].isin(self.selected_entities)]
        if clean_states.empty:
            return pd.DataFrame()

        clean_states["domain"] = clean_states["entity_id"].astype(str).str.split(".").str[0]
        clean_states["state"] = clean_states["clean_state"]
        clean_states["last_changed"] = clean_states["timestamp"]
        clean_states["last_updated"] = clean_states["timestamp"]
        return self.normalize_frame(clean_states)


def read_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False).fillna("")


def parse_timeline_timestamp(value: str | None) -> pd.Timestamp:
    if not value:
        return pd.NaT
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return pd.NaT
    return timestamp.tz_convert(None)


def downsample_frame(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].head(max_points)


def binary_percentage_timeline(group: pd.DataFrame, max_points: int) -> tuple[list[dict[str, Any]], int]:
    working = group.copy()
    working["timestamp_dt"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working[working["timestamp_dt"].notna()].sort_values("timestamp_dt").reset_index(drop=True)
    if working.empty:
        return [], 60

    active_states = {"on", "open", "home", "active", "detected"}
    working["value"] = working["state"].astype(str).str.lower().isin(active_states).astype(float)
    span_seconds = max((working["timestamp_dt"].iloc[-1] - working["timestamp_dt"].iloc[0]).total_seconds(), 60.0)
    bucket_seconds = choose_timeline_bucket_seconds(span_seconds, max_points)
    features = build_binary_temporal_features(working, window_seconds=bucket_seconds, step_seconds=bucket_seconds)

    points = []
    for row in features.to_dict("records"):
        percentage = round(float(row["active_ratio"]) * 100.0, 2)
        points.append(
            {
                "timestamp": str(row["window_start"]),
                "state": f"{percentage:.1f}%",
                "value": percentage,
                "activations": int(row["activations"]),
                "active_seconds": round(float(row["active_seconds"]), 3),
                "active_ratio": round(float(row["active_ratio"]), 4),
            }
        )
    return points[:max_points], bucket_seconds


def choose_timeline_bucket_seconds(span_seconds: float, max_points: int) -> int:
    target_points = max(max_points, 1)
    raw_bucket = max(60, int(span_seconds / target_points) + 1)
    # Redondear a minutos completos mantiene etiquetas legibles y ventanas
    # comparables entre entidades.
    return int(((raw_bucket + 59) // 60) * 60)


def classify_entity(entity: dict[str, Any]) -> str:
    labels = set(str(entity.get("labels", "")).split("|"))
    domain = str(entity.get("domain", ""))
    state_class = str(entity.get("state_class", ""))
    device_class = str(entity.get("device_class", ""))
    if "continuous" in labels or (domain == "sensor" and state_class == "measurement"):
        return "continuous"
    if "binary" in labels or domain == "binary_sensor" or device_class in {"motion", "door", "window"}:
        return "binary"
    return "multistate"


def build_baseline_item(
    entity_id: str,
    kind: str,
    training: pd.DataFrame,
    rows: pd.DataFrame,
    summary: dict[str, Any],
    event_mode: bool = False,
) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "sensor_kind": kind,
        "train_rows": int(len(training)),
        "test_rows": int(max(0, len(rows) - len(training))),
        "evaluation_mode": "new_events" if event_mode else "historical_split",
        "summary": summary,
    }


def rate_changes_frame(rows: pd.DataFrame) -> list[float]:
    ordered = rows.sort_values("timestamp_dt")
    values = pd.to_numeric(ordered["value"], errors="coerce")
    minutes = ordered["timestamp_dt"].diff().dt.total_seconds() / 60.0
    rates = values.diff().abs() / minutes
    rates = rates[(minutes > 0) & rates.notna()]
    return rates.astype(float).tolist()


def build_hourly_numeric_baseline(rows: pd.DataFrame) -> dict[int, dict[str, Any]]:
    hourly: dict[int, dict[str, Any]] = {}
    working = rows.copy()
    working["hour"] = working["timestamp_dt"].dt.hour
    working["numeric_value"] = pd.to_numeric(working["value"], errors="coerce")
    for hour, group in working.groupby("hour"):
        values = group["numeric_value"].dropna().astype(float).tolist()
        if len(values) >= 6:
            hourly[int(hour)] = summarize_numeric(values)
    return hourly


def residual_slopes_frame(rows: pd.DataFrame, window_minutes: int = 120) -> list[float]:
    ordered = rows.sort_values("timestamp_dt").copy()
    ordered["numeric_value"] = pd.to_numeric(ordered["value"], errors="coerce")
    ordered = ordered[ordered["numeric_value"].notna()].reset_index(drop=True)
    if len(ordered) < 6:
        return []

    hourly = build_hourly_numeric_baseline(ordered)
    residuals = []
    for _, row in ordered.iterrows():
        hour = int(row["timestamp_dt"].hour)
        reference = hourly.get(hour)
        if not reference:
            continue
        residuals.append(
            {
                "timestamp_dt": row["timestamp_dt"],
                "residual": float(row["numeric_value"]) - float(reference["median"]),
            }
        )

    residual_df = pd.DataFrame(residuals)
    if residual_df.empty:
        return []

    first_timestamp = residual_df["timestamp_dt"].iloc[0]
    seconds = [
        float((timestamp - first_timestamp).total_seconds())
        for timestamp in residual_df["timestamp_dt"]
    ]
    residual_values = residual_df["residual"].astype(float).tolist()
    step = max(1, len(residual_df) // 5000)
    window_seconds = float(window_minutes * 60)
    slopes = []
    for index in range(0, len(residual_df), step):
        start_index = bisect_left(seconds, seconds[index] - window_seconds)
        if index - start_index + 1 < 6:
            continue

        x = [(item - seconds[start_index]) / 3600.0 for item in seconds[start_index : index + 1]]
        y = residual_values[start_index : index + 1]
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        denominator = sum((item - x_mean) ** 2 for item in x)
        if denominator <= 0:
            continue

        slope = sum((x[item] - x_mean) * (y[item] - y_mean) for item in range(len(x))) / denominator
        slopes.append(abs(slope))

    return slopes


def value_slopes_frame(rows: pd.DataFrame, window_minutes: int = 120) -> list[float]:
    ordered = rows.sort_values("timestamp_dt").copy()
    ordered["numeric_value"] = pd.to_numeric(ordered["value"], errors="coerce")
    ordered = ordered[ordered["numeric_value"].notna()].reset_index(drop=True)
    if len(ordered) < 6:
        return []

    first_timestamp = ordered["timestamp_dt"].iloc[0]
    seconds = [
        float((timestamp - first_timestamp).total_seconds())
        for timestamp in ordered["timestamp_dt"]
    ]
    values = ordered["numeric_value"].astype(float).tolist()
    step = max(1, len(ordered) // 5000)
    window_seconds = float(window_minutes * 60)
    slopes = []
    for index in range(0, len(ordered), step):
        start_index = bisect_left(seconds, seconds[index] - window_seconds)
        if index - start_index + 1 < 6:
            continue

        x = [(item - seconds[start_index]) / 3600.0 for item in seconds[start_index : index + 1]]
        y = values[start_index : index + 1]
        x_mean = sum(x) / len(x)
        y_mean = sum(y) / len(y)
        denominator = sum((item - x_mean) ** 2 for item in x)
        if denominator <= 0:
            continue

        slope = sum((x[item] - x_mean) * (y[item] - y_mean) for item in range(len(x))) / denominator
        slopes.append(abs(slope))

    return slopes


def build_hourly_binary_baseline(metrics: pd.DataFrame) -> dict[int, dict[str, Any]]:
    if metrics.empty or "window_start" not in metrics.columns:
        return {}

    working = metrics.copy()
    working["window_start_dt"] = pd.to_datetime(working["window_start"], errors="coerce")
    working = working[working["window_start_dt"].notna()].copy()
    if working.empty:
        return {}

    working["hour"] = working["window_start_dt"].dt.hour
    hourly: dict[int, dict[str, Any]] = {}
    for hour, group in working.groupby("hour"):
        hour_summary: dict[str, Any] = {}
        for column in ["activations", "state_changes", "active_seconds", "active_ratio", "max_active_seconds", "gap"]:
            values = pd.to_numeric(group[column], errors="coerce").dropna().astype(float).tolist()
            if not values:
                continue
            column_summary = summarize_numeric(values)
            column_summary["p90"] = float(group[column].quantile(0.90))
            column_summary["p95"] = float(group[column].quantile(0.95))
            hour_summary[column] = column_summary
        if hour_summary:
            hourly[int(hour)] = hour_summary
    return hourly


def state_durations_frame(rows: pd.DataFrame) -> dict[str, list[float]]:
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    durations: dict[str, list[float]] = {}
    if ordered.empty:
        return durations

    current_state = str(ordered["state"].iloc[0])
    current_start = ordered["timestamp_dt"].iloc[0]
    previous_timestamp = current_start
    for _, row in ordered.iloc[1:].iterrows():
        if str(row["state"]) != current_state:
            minutes = max((previous_timestamp - current_start).total_seconds() / 60.0, 1.0)
            durations.setdefault(current_state, []).append(float(minutes))
            current_state = str(row["state"])
            current_start = row["timestamp_dt"]
        previous_timestamp = row["timestamp_dt"]

    minutes = max((previous_timestamp - current_start).total_seconds() / 60.0, 1.0)
    durations.setdefault(current_state, []).append(float(minutes))
    return durations


def multistate_cycle_train_count(rows: pd.DataFrame, minimum_cycle_hours: int = 48) -> int:
    ordered = rows.sort_values("timestamp_dt").reset_index(drop=True)
    if ordered.empty:
        return 0

    first_timestamp = ordered["timestamp_dt"].iloc[0]
    last_timestamp = ordered["timestamp_dt"].iloc[-1]
    available_hours = max((last_timestamp - first_timestamp).total_seconds() / 3600.0, 0.0)
    if available_hours < 24:
        return len(ordered)

    target_hours = minimum_cycle_hours if available_hours >= minimum_cycle_hours + 1 else 24
    cutoff = first_timestamp + pd.Timedelta(hours=target_hours)
    train_count = int(ordered["timestamp_dt"].le(cutoff).sum())
    return min(len(ordered), max(10, train_count))


def create_state() -> ArtifactState:
    root = Path(__file__).resolve().parents[2]
    dataset_dir = Path(os.getenv("DATASET_DIR", root.parent / "dataset_simulado" / "output"))
    fixture_dir = Path(os.getenv("MULTISTATE_FIXTURE_DIR", root / "data" / "fixtures"))
    static_history_dir = Path(os.getenv("STATIC_HISTORY_DIR", root / "data" / "static_history"))
    real_history_path = Path(os.getenv("REAL_HISTORY_PATH", root / "data" / "datosreales" / "history.csv"))
    return ArtifactState(
        dataset_dir=dataset_dir,
        fixture_dir=fixture_dir,
        static_history_dir=static_history_dir,
        real_history_path=real_history_path,
    )
