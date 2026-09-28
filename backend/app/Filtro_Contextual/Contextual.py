from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd


ESTADOS_ACTIVOS = {"on", "open", "home", "active", "detected", "true", "1", "heat", "cool"}


@dataclass(frozen=True)
class ConfiguracionContextual:
    """Reglas versionadas y reproducibles del segundo filtro."""

    top_k: int = 5
    window_seconds: int = 3600
    max_lag_windows: int = 2
    association_threshold: float = 0.35
    min_observations: int = 30
    min_pair_coverage: float = 0.60
    stability_slices: int = 4
    min_stable_slices: int = 3
    stability_association_threshold: float = 0.20
    min_sign_consistency: float = 0.75
    runtime_coverage_threshold: float = 0.50
    evidence_threshold: float = 0.20
    decision_threshold: float = 0.35
    common_cause_min_entities: int = 3
    max_map_age_days: int = 7
    association_drift_threshold: float = 0.20
    coverage_drift_threshold: float = 0.20
    continuous_max_age_seconds: float = 3600.0
    continuous_max_age_seconds_by_entity: dict[str, float] = field(default_factory=dict)
    window_candidates_seconds: tuple[int, ...] = (300, 900, 3600)
    common_cause_window_seconds: int = 300
    cadence_ttl_multiplier: float = 2.0
    residual_quantile: float = 0.99
    temporal_feature_min_observations: int = 2
    temporal_model_min_observations: int = 30
    temporal_decision_min_observations: int = 30

    def __post_init__(self) -> None:
        if (self.window_seconds <= 0 or self.max_lag_windows < 0
                or self.common_cause_window_seconds <= 0
                or self.cadence_ttl_multiplier < 1
                or not 0.5 < self.residual_quantile < 1
                or self.temporal_feature_min_observations < 2
                or self.temporal_model_min_observations < 1
                or self.temporal_decision_min_observations < 1
                or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
                       for v in self.window_candidates_seconds)):
            raise ValueError("Configuracion temporal o de residuos invalida.")
        if not isinstance(self.continuous_max_age_seconds_by_entity, dict):
            raise ValueError("La antiguedad por entidad debe ser un objeto de segundos.")
        limits = [("default", self.continuous_max_age_seconds),
                  *self.continuous_max_age_seconds_by_entity.items()]
        for entity_id, seconds in limits:
            if (not isinstance(entity_id, str) or not entity_id
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds < 0):
                raise ValueError("La antiguedad contextual debe ser finita y no negativa.")

    def antiguedad_maxima(self, entity_id: str) -> float:
        return float(self.continuous_max_age_seconds_by_entity.get(
            entity_id, self.continuous_max_age_seconds
        ))

    def como_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["window_candidates_seconds"] = list(self.window_candidates_seconds)
        payload["window_selection"] = "smallest_eligible_else_smallest"
        payload["lag_candidates_seconds"] = [
            lag * self.window_seconds for lag in range(self.max_lag_windows + 1)
        ]
        payload["lag_candidates_by_window"] = {
            str(seconds): self.desfases(seconds) for seconds in self.ventanas()
        }
        payload["acceptance_rule"] = (
            "n_observations >= min_observations AND "
            "coverage_ratio >= min_pair_coverage AND "
            "|association_signed| >= association_threshold AND "
            "stable_slices >= min_stable_slices AND "
            "sign_consistency >= min_sign_consistency; despues se conserva top_k por entidad"
        )
        return payload


    def ventanas(self) -> list[int]:
        return sorted({self.window_seconds, *(
            value for value in self.window_candidates_seconds
            if value <= self.window_seconds and self.window_seconds % value == 0
        )})

    def desfases(self, seconds: int) -> list[int]:
        maximum = self.max_lag_windows * self.window_seconds
        return sorted(offset for offset in {0, seconds, 2 * seconds, 3 * seconds,
                      6 * seconds, self.window_seconds, maximum // 2, maximum}
                      if offset <= maximum and offset % seconds == 0)


class PreparedCausalEvents:
    """Prepared per-entity event indexes updated in chronological batches."""
    def __init__(self, filtro, entidades: pd.DataFrame, eventos: pd.DataFrame):
        self.filtro = filtro
        prepared = filtro._preparar_eventos(eventos)
        self.catalog = filtro._catalogo(entidades, prepared)
        self.observations = filtro._indexar_observaciones(prepared, self.catalog)

    def advance(self, events: pd.DataFrame, cutoff: pd.Timestamp) -> None:
        if not isinstance(events, pd.DataFrame) or events.empty:
            return
        prepared = self.filtro._preparar_eventos(events)
        for entity_id, rows in prepared.groupby("entity_id", sort=False):
            entity_id = str(entity_id)
            if entity_id not in self.catalog:
                # Future values do not define metadata. Only declared type/domain
                # fields and the stable entity id are consulted for new entities.
                first = rows.iloc[0]
                metadata = {key: first.get(key) for key in ("sensor_kind", "domain")}
                self.catalog[entity_id] = {
                    "entity_id": entity_id,
                    **metadata,
                    "sensor_kind": self.filtro._tipo_entidad(metadata, entity_id),
                }
            current = self.observations.get(entity_id)
            incoming = rows.set_index("timestamp_dt")
            if current is None:
                current = incoming
            elif not incoming.empty:
                current = pd.concat([current, incoming]).sort_index(kind="stable")
            self.observations[entity_id] = current
        for entity_id, current in self.observations.items():
            boundary = int(current.index.searchsorted(cutoff, side="left"))
            if boundary:
                self.observations[entity_id] = pd.concat(
                    [current.iloc[boundary - 1:boundary], current.iloc[boundary:]]
                )

    @property
    def buffered_event_rows(self) -> int:
        return sum(len(frame) for frame in self.observations.values())


class FiltroContextualEspacioTemporal:
    """Construye el mapa histórico y contrasta sospechas en línea.

    El mapa se aprende solo desde el histórico normalizado. La inferencia usa
    relaciones dirigidas hacia contexto presente o pasado; nunca consulta
    observaciones futuras para explicar una sospecha.
    """

    # v4 replaces the contradictory sign heuristic for continuous pairs with
    # conditional level residuals.  Maps from earlier implementations stay
    # intact and are rebuilt through the existing version check.
    MAP_VERSION = "contextual-spatiotemporal-v5"

    def __init__(self, config: ConfiguracionContextual | None = None) -> None:
        self.config = config or ConfiguracionContextual()

    def preparar_eventos_causales(self, entidades: pd.DataFrame, eventos: pd.DataFrame) -> PreparedCausalEvents:
        return PreparedCausalEvents(self, entidades, eventos)

    def implementation_sha256(self) -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def fingerprint(self, eventos: pd.DataFrame, entidades: pd.DataFrame) -> dict[str, Any]:
        trabajo = self._preparar_eventos(eventos)
        catalogo = self._catalogo(entidades, trabajo)
        catalog_text = "|".join(
            f"{entity_id}:{meta['sensor_kind']}" for entity_id, meta in sorted(catalogo.items())
        )
        catalog_hash = hashlib.sha256(catalog_text.encode("utf-8")).hexdigest()
        if trabajo.empty:
            data_hash = hashlib.sha256(b"empty").hexdigest()
            inicio = None
            fin = None
        else:
            columnas = [
                columna
                for columna in ["entity_id", "timestamp", "state", "sensor_kind", "is_valid_value"]
                if columna in trabajo.columns
            ]
            hashed = pd.util.hash_pandas_object(trabajo[columnas], index=False)
            data_hash = hashlib.sha256(hashed.values.tobytes()).hexdigest()
            inicio = self._timestamp_o_none(trabajo["timestamp_dt"].min())
            fin = self._timestamp_o_none(trabajo["timestamp_dt"].max())
        return {
            "data_hash": data_hash,
            "catalog_hash": catalog_hash,
            "rows": int(len(trabajo)),
            "entity_count": int(len(catalogo)),
            "start": inicio,
            "end": fin,
        }

    def razones_actualizacion(
        self,
        mapa: dict[str, Any] | None,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
    ) -> list[str]:
        if not mapa:
            return ["missing_map"]
        razones: list[str] = []
        if mapa.get("map_version") != self.MAP_VERSION:
            razones.append("map_version_changed")
        if (
            mapa.get("implementation_sha256")
            != self.implementation_sha256()
        ):
            razones.append("implementation_changed")
        if mapa.get("config") != self.config.como_dict():
            razones.append("configuration_changed")
        actual = self.fingerprint(eventos, entidades)
        anterior = mapa.get("source_fingerprint") or {}
        if anterior.get("catalog_hash") != actual.get("catalog_hash"):
            razones.append("entity_catalog_changed")
        if anterior.get("data_hash") != actual.get("data_hash"):
            razones.append("source_history_changed")
        if self._cruzo_estacion(anterior.get("end"), actual.get("end")):
            razones.append("season_boundary")
        built_at = pd.to_datetime(mapa.get("built_at"), errors="coerce", utc=True)
        if pd.isna(built_at):
            razones.append("invalid_build_timestamp")
        else:
            age_days = (pd.Timestamp.now(tz="UTC") - built_at).total_seconds() / 86400.0
            if age_days > self.config.max_map_age_days:
                razones.append("max_age_exceeded")
        return list(dict.fromkeys(razones))

    def construir_mapa(
        self,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        *,
        trigger: str = "missing_map",
        mapa_anterior: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        inicio_calculo = perf_counter()
        trabajo = self._preparar_eventos(eventos)
        catalogo = self._catalogo(entidades, trabajo)
        partitions = self._particiones_desarrollo(trabajo)
        ajuste = self._filtrar_particion(trabajo, partitions, "fit")
        signals, availability, stats, full_index = self._construir_senales(ajuste, catalogo)
        entity_ids = sorted(signals)
        temporal = self._calibrar_tiempos(ajuste, catalogo)
        resolutions = set(self.config.ventanas())
        grids = {self.config.window_seconds: (signals, availability, stats, full_index)}
        for seconds in sorted(resolutions - grids.keys()):
            grids[seconds] = self._construir_senales(ajuste, catalogo, seconds)

        edges: list[dict[str, Any]] = []
        for source in entity_ids:
            for target in entity_ids:
                if source == target:
                    continue
                relation = self._seleccionar_ventana(source, target, catalogo, grids)
                _, _, pair_stats, _ = grids[relation["window_seconds"]]
                relation["source_stats"] = pair_stats[source]
                relation["target_stats"] = pair_stats[target]
                relation["max_age_seconds"] = temporal[target]["max_age_seconds"]
                edges.append(relation)

        self._aplicar_top_k(edges)
        observaciones = self._indexar_observaciones(trabajo, catalogo)
        for edge in edges:
            if (edge.get("accepted") and edge.get("relation_family") == "continuous_continuous"):
                edge["temporal_models"] = self._calibrar_modelos_temporales(
                    edge, observaciones, partitions
                )
        decision_calibrations = self._calibrar_decisiones_temporales(
            edges, observaciones, partitions
        )
        # Solo las relaciones utilizables necesitan payload de inferencia y
        # auditoria de candidatos; no multiplicarlo por todos los pares rechazados.
        for edge in edges:
            if not edge["accepted"]:
                for key in ("source_stats", "target_stats", "expected_response", "lag_candidates"):
                    edge.pop(key, None)
        accepted_edges = [edge for edge in edges if edge["accepted"]]
        accepted_by_source: dict[str, int] = {}
        for edge in accepted_edges:
            accepted_by_source[edge["source"]] = accepted_by_source.get(edge["source"], 0) + 1

        nodes = []
        for entity_id in entity_ids:
            meta = catalogo[entity_id]
            node_stats = stats[entity_id]
            nodes.append(
                {
                    "entity_id": entity_id,
                    "sensor_kind": meta["sensor_kind"],
                    "device_class": meta.get("device_class", ""),
                    "domain": meta.get("domain", ""),
                    "unit_of_measurement": meta.get("unit_of_measurement", ""),
                    "observations": int(node_stats.get("observations", 0)),
                    "coverage_ratio": self._redondear(node_stats.get("coverage_ratio", 0.0)),
                    "center": self._redondear_o_none(node_stats.get("center")),
                    "scale": self._redondear_o_none(node_stats.get("scale")),
                    "state_probabilities": node_stats.get("state_probabilities", {}),
                    "accepted_relations": int(accepted_by_source.get(entity_id, 0)),
                    "covered": bool(accepted_by_source.get(entity_id, 0)),
                    "context_status": "connected" if accepted_by_source.get(entity_id) else "no_learned_relations",
                    "temporal_profile": temporal[entity_id],
                }
            )

        matrices = self._construir_matrices(entity_ids, edges)
        fingerprint = self.fingerprint(trabajo, entidades)
        drift = self._comparar_mapas(mapa_anterior, edges, nodes)
        rebuild_reasons = [trigger] if trigger else []
        if drift["association_drift_detected"]:
            rebuild_reasons.append("association_drift")
        if drift["coverage_drift_detected"]:
            rebuild_reasons.append("coverage_drift")

        rejected_counts: dict[str, int] = {}
        for edge in edges:
            for reason in edge["rejection_reasons"]:
                rejected_counts[reason] = rejected_counts.get(reason, 0) + 1

        result = {
            "map_version": self.MAP_VERSION,
            "implementation_sha256": self.implementation_sha256(),
            "built_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "source_range": {"start": fingerprint["start"], "end": fingerprint["end"]},
            "source_fingerprint": fingerprint,
            "config": self.config.como_dict(),
            "calibration_partition": "development",
            "development_partitions": partitions,
            "decision_calibrations": decision_calibrations,
            "refresh": {
                "stale": False,
                "stale_reasons": [],
                "trigger": trigger,
                "rebuild_reasons": list(dict.fromkeys(rebuild_reasons)),
                "policy": {
                    "scheduled_max_age_days": self.config.max_map_age_days,
                    "entity_catalog_change": "automatic",
                    "source_history_change": "automatic",
                    "season_boundary": "automatic_on_new_history",
                    "association_drift_threshold": self.config.association_drift_threshold,
                    "coverage_drift_threshold": self.config.coverage_drift_threshold,
                },
                "drift": drift,
            },
            "summary": {
                "nodes": len(nodes),
                "candidate_edges": len(edges),
                "eligible_edges_before_top_k": sum(bool(edge["threshold_eligible"]) for edge in edges),
                "accepted_edges": len(accepted_edges),
                "covered_entities": len(accepted_by_source),
                "uncovered_entities": max(0, len(nodes) - len(accepted_by_source)),
                "rejected_by_reason": rejected_counts,
                "window_count": len(full_index),
                "elapsed_ms": self._redondear((perf_counter() - inicio_calculo) * 1000.0, 3),
            },
            "nodes": nodes,
            "edges": edges,
            "matrices": matrices,
            "limitations": {
                "causal": "Las asociaciones no demuestran causalidad.",
                "common_cause": (
                    "Una causa comun de red, alimentacion o integracion no es identificable "
                    "sin evidencia independiente."
                ),
                "spatial": (
                    "Espacial significa dependencia entre entidades; no implica coordenadas "
                    "fisicas cuando el historico no contiene areas."
                ),
            },
        }
        return result

    def analizar(
        self,
        sospechas: list[dict[str, Any]],
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        mapa: dict[str, Any],
    ) -> dict[str, Any]:
        return self.analizar_causal(sospechas, eventos, entidades, mapa)

    def analizar_causal(
        self,
        sospechas: list[dict[str, Any]],
        eventos_disponibles: pd.DataFrame,
        entidades: pd.DataFrame,
        mapa: dict[str, Any],
        concurrence_state: dict[tuple[str, str], pd.Timestamp] | None = None,
        accept_all: bool = False,
        prepared_state: PreparedCausalEvents | None = None,
    ) -> dict[str, Any]:
        """Evalua cada sospecha usando solo observaciones con fecha no futura."""
        inicio = perf_counter()
        if prepared_state is None:
            trabajo = self._preparar_eventos(eventos_disponibles)
            catalogo = self._catalogo(entidades, trabajo)
            observaciones = self._indexar_observaciones(trabajo, catalogo)
        else:
            catalogo = prepared_state.catalog
            observaciones = prepared_state.observations
        edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in mapa.get("edges", []):
            if edge.get("accepted"):
                edges_by_source.setdefault(str(edge["source"]), []).append(edge)
        stats_mapa = self._estadisticas_mapa(mapa)
        concurrence_state = {} if concurrence_state is None else concurrence_state
        candidatas = sorted(
            (
                dict(item)
                for item in sospechas
                if (accept_all or item.get("is_suspicious") or item.get("passed_to_contextual_filter"))
                and (accept_all or not item.get("suppressed_by_episode"))
            ),
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
        )
        resultados: list[dict[str, Any]] = []
        for sospecha in candidatas:
            timestamp = pd.to_datetime(
                sospecha.get("timestamp"), errors="coerce", utc=True
            )
            entity_id = str(sospecha.get("entity_id", ""))
            edges = edges_by_source.get(entity_id, [])
            signals = {}
            stats = stats_mapa
            concurrentes = self._concurrencia_relacionada(sospecha, edges, concurrence_state)
            resultado = self._evaluar_sospecha(
                sospecha=sospecha,
                signals=signals,
                stats=stats,
                edges=edges,
                map_quality={
                    "version": mapa.get("map_version"),
                    **mapa.get("refresh", {}),
                },
                concurrent_entities=concurrentes,
                continuous_observations=observaciones,
                decision_calibrations=mapa.get("decision_calibrations", {}),
            )
            resultado["causal_runtime"] = {
                "causal": True,
                "uses_future_rows": False,
                "available_through": (
                    None if pd.isna(timestamp) else timestamp.isoformat()
                ),
                "baseline_partition": "development",
            }
            resultados.append(resultado)

        outcomes: dict[str, int] = {}
        for item in resultados:
            outcome = str(item["context_outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        return {
            "results": resultados,
            "summary": {
                "evaluated": len(resultados),
                **self.resumen_enrutamiento(resultados),
                "by_outcome": outcomes,
                "supported": outcomes.get("supported", 0),
                "contradicted": outcomes.get("contradicted", 0),
                "unexplained": outcomes.get("unexplained", 0),
                "insufficient_coverage": outcomes.get(
                    "insufficient_coverage", 0
                ),
                "no_learned_relations": outcomes.get("no_learned_relations", 0),
                "contextual_coincidences": sum(
                    bool(item.get("common_cause", {}).get("contextual_coincidence")) for item in resultados
                ),
                "possible_common_cause": sum(
                    bool(item.get("common_cause", {}).get("candidate"))
                    for item in resultados
                ),
                "elapsed_ms": self._redondear(
                    (perf_counter() - inicio) * 1000.0, 3
                ),
                "causal": True,
                "uses_future_rows": False,
            },
            "map_quality": mapa.get("refresh", {}),
        }

    def _preparar_eventos(self, eventos: pd.DataFrame) -> pd.DataFrame:
        if eventos is None or eventos.empty:
            return pd.DataFrame(
                columns=[
                    "entity_id",
                    "timestamp",
                    "timestamp_dt",
                    "state",
                    "value",
                    "sensor_kind",
                    "is_valid_value",
                ]
            )
        trabajo = eventos.copy()
        if "timestamp_dt" not in trabajo.columns:
            trabajo["timestamp_dt"] = pd.to_datetime(
                trabajo.get("timestamp"), format="ISO8601", errors="coerce", utc=True
            )
        else:
            trabajo["timestamp_dt"] = pd.to_datetime(
                trabajo["timestamp_dt"], errors="coerce", utc=True
            )
        trabajo = trabajo[trabajo["timestamp_dt"].notna()].copy()
        trabajo["entity_id"] = trabajo["entity_id"].astype(str)
        trabajo["state"] = trabajo.get("state", "").astype(str)
        if "sensor_kind" not in trabajo.columns:
            trabajo["sensor_kind"] = ""
        if "is_valid_value" not in trabajo.columns:
            trabajo["is_valid_value"] = True
        trabajo["is_valid_value"] = trabajo["is_valid_value"].astype(bool)
        # ``mergesort`` preserves the producer order of simultaneous readings;
        # causal feature extraction never uses a later row at that timestamp.
        return trabajo.sort_values(["entity_id", "timestamp_dt"], kind="stable").reset_index(drop=True)

    def _catalogo(
        self, entidades: pd.DataFrame, eventos: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        catalogo: dict[str, dict[str, Any]] = {}
        if entidades is not None and not entidades.empty:
            for row in entidades.to_dict("records"):
                entity_id = str(row.get("entity_id", "")).strip()
                if entity_id:
                    catalogo[entity_id] = {
                        **row,
                        "sensor_kind": self._tipo_entidad(row, entity_id),
                    }
        for entity_id, grupo in eventos.groupby("entity_id", sort=False):
            entity_id = str(entity_id)
            if entity_id not in catalogo:
                row = grupo.iloc[0].to_dict()
                catalogo[entity_id] = {
                    **row,
                    "sensor_kind": self._tipo_entidad(row, entity_id),
                }
        return catalogo

    def _tipo_entidad(self, row: dict[str, Any], entity_id: str) -> str:
        declared = str(row.get("sensor_kind", "")).strip()
        if declared in {"continuous", "binary", "multistate"}:
            return declared
        domain = str(row.get("domain", "")).lower() or entity_id.split(".")[0].lower()
        if domain == "binary_sensor":
            return "binary"
        if domain in {"select", "input_select", "climate"}:
            return "multistate"
        return "continuous"

    def _construir_senales(
        self,
        eventos: pd.DataFrame,
        catalogo: dict[str, dict[str, Any]],
        window_seconds: int | None = None,
    ) -> tuple[
        dict[str, pd.Series],
        dict[str, pd.Series],
        dict[str, dict[str, Any]],
        pd.DatetimeIndex,
    ]:
        if eventos.empty or not catalogo:
            return {}, {}, {}, pd.DatetimeIndex([])
        freq = f"{window_seconds or self.config.window_seconds}s"
        start = eventos["timestamp_dt"].min().floor(freq)
        end = eventos["timestamp_dt"].max().floor(freq)
        full_index = pd.date_range(start, end, freq=freq, tz="UTC")
        signals: dict[str, pd.Series] = {}
        availability: dict[str, pd.Series] = {}
        stats: dict[str, dict[str, Any]] = {}

        for entity_id, meta in sorted(catalogo.items()):
            grupo = eventos[eventos["entity_id"].eq(entity_id)].copy()
            if grupo.empty:
                continue
            grupo["bucket"] = grupo["timestamp_dt"].dt.floor(freq)
            kind = str(meta["sensor_kind"])
            validity = (
                grupo.groupby("bucket", sort=True)["is_valid_value"]
                .last()
                .reindex(full_index)
            )
            if kind in {"binary", "multistate"}:
                validity = validity.ffill().fillna(False).astype(bool)
            else:
                validity = validity.fillna(False).astype(bool)

            if kind == "continuous":
                numeric = pd.to_numeric(grupo.get("value", grupo["state"]), errors="coerce")
                numeric = numeric.replace([math.inf, -math.inf], float("nan"))
                grupo["_feature"] = numeric.where(grupo["is_valid_value"])
                signal = (
                    grupo.groupby("bucket", sort=True)["_feature"]
                    .median()
                    .reindex(full_index)
                )
            elif kind == "binary":
                if "value" in grupo.columns:
                    numeric = pd.to_numeric(grupo["value"], errors="coerce")
                else:
                    numeric = pd.Series(float("nan"), index=grupo.index)
                fallback = grupo["state"].str.lower().isin(ESTADOS_ACTIVOS).astype(float)
                grupo["_feature"] = numeric.where(numeric.notna(), fallback)
                grupo.loc[~grupo["is_valid_value"], "_feature"] = float("nan")
                active_in_bucket = (
                    grupo.groupby("bucket", sort=True)["_feature"]
                    .max()
                    .reindex(full_index)
                )
                state_at_end = (
                    grupo.groupby("bucket", sort=True)["_feature"]
                    .last()
                    .reindex(full_index)
                    .ffill()
                )
                signal = active_in_bucket.where(active_in_bucket.notna(), state_at_end)
                signal = signal.where(validity)
            else:
                grupo["_feature"] = grupo["state"].where(grupo["is_valid_value"])
                signal = (
                    grupo.groupby("bucket", sort=True)["_feature"]
                    .last()
                    .reindex(full_index)
                    .ffill()
                    .where(validity)
                )

            signal.name = entity_id
            signals[entity_id] = signal
            availability[entity_id] = validity
            valid_signal = signal.dropna()
            node_stats: dict[str, Any] = {
                "observations": int(len(valid_signal)),
                "coverage_ratio": float(len(valid_signal) / len(full_index)) if len(full_index) else 0.0,
            }
            if kind == "multistate":
                probabilities = valid_signal.astype(str).value_counts(normalize=True)
                node_stats["state_probabilities"] = {
                    str(state): self._redondear(probability)
                    for state, probability in probabilities.items()
                }
                node_stats["center"] = None
                node_stats["scale"] = None
            else:
                values = pd.to_numeric(valid_signal, errors="coerce").dropna()
                center = float(values.median()) if not values.empty else 0.0
                mad = float((values - center).abs().median()) if not values.empty else 0.0
                std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
                scale = max(1.4826 * mad, std, 1e-6)
                node_stats["center"] = center
                node_stats["scale"] = scale
                node_stats["state_probabilities"] = {}
            stats[entity_id] = node_stats
        return signals, availability, stats, full_index

    def _calibrar_tiempos(self, eventos, catalogo) -> dict[str, dict[str, Any]]:
        """Politica acotada de cadencia, aprendida exclusivamente en Desarrollo.

        Un intervalo entre cambios no prueba la frecuencia de muestreo. Por eso
        el TTL inferido nunca supera el limite explicito y no se aplica a estados.
        """
        windows = self.config.ventanas()
        profiles = {}
        for entity, group in eventos.groupby("entity_id", sort=False):
            gaps = group.loc[group["is_valid_value"], "timestamp_dt"].sort_values().diff().dt.total_seconds()
            gaps = gaps[gaps.gt(0)]
            calibrated = len(gaps) >= self.config.min_observations
            median = float(gaps.median()) if calibrated else None
            p95 = float(gaps.quantile(.95)) if calibrated else None
            window = next((value for value in windows if median is not None and value >= median),
                          self.config.window_seconds)
            limit = self.config.antiguedad_maxima(entity)
            explicit = entity in self.config.continuous_max_age_seconds_by_entity
            continuous = catalogo[entity]["sensor_kind"] == "continuous"
            ttl = limit if explicit or not calibrated else min(limit, p95 * self.config.cadence_ttl_multiplier)
            profiles[entity] = {
                "window_seconds": window,
                "max_age_seconds": ttl if continuous else None,
                "median_interval_seconds": median, "p95_interval_seconds": p95,
                "interval_count": len(gaps), "partition": "development",
                "ttl_source": ("not_applicable_to_state" if not continuous else
                               "explicit_entity_limit" if explicit else
                               "development_cadence_capped" if calibrated else "configured_fallback"),
            }
        return profiles

    def _particiones_desarrollo(self, eventos: pd.DataFrame) -> dict[str, Any]:
        """Fronteras temporales comunes; nunca se infieren de etiquetas."""
        if eventos.empty:
            return {"fit": {"start": None, "end": None}, "tolerance": {"start": None, "end": None},
                    "decision": {"start": None, "end": None}}
        start, end = eventos["timestamp_dt"].min(), eventos["timestamp_dt"].max()
        span = end - start
        fit_end, tolerance_end = start + span * .5, start + span * .75
        iso = lambda value: None if pd.isna(value) else value.isoformat()
        return {
            "fit": {"start": iso(start), "end": iso(fit_end)},
            "tolerance": {"start": iso(fit_end), "end": iso(tolerance_end)},
            "decision": {"start": iso(tolerance_end), "end": iso(end)},
        }

    def _filtrar_particion(self, eventos: pd.DataFrame, partitions: dict[str, Any], name: str) -> pd.DataFrame:
        bounds = partitions.get(name, {})
        start, end = (pd.to_datetime(bounds.get(key), errors="coerce", utc=True) for key in ("start", "end"))
        if eventos.empty or pd.isna(start) or pd.isna(end):
            return eventos.iloc[:0].copy()
        if name == "fit":
            return eventos[eventos["timestamp_dt"].between(start, end, inclusive="both")].copy()
        return eventos[(eventos["timestamp_dt"] > start) & (eventos["timestamp_dt"] <= end)].copy()

    def _caracteristicas_continuas(
        self,
        events: pd.DataFrame,
        reference: pd.Timestamp,
        entity_id: str,
        *,
        window_seconds: int,
        max_age_seconds: float | None = None,
        current_value: float | None = None,
        partition_start: pd.Timestamp | None = None,
    ) -> dict[str, Any]:
        """Nivel, innovacion y tendencia causal sobre lecturas originales.

        La lectura actual se excluye de la referencia de innovacion. Las
        medianas no se rellenan: una interrupcion invalida toda caracteristica
        que atraviese el reporte invalido.
        """
        reference = pd.to_datetime(reference, errors="coerce", utc=True)
        result: dict[str, Any] = {
            "reference_timestamp": None if pd.isna(reference) else reference.isoformat(),
            "window_seconds": window_seconds, "original_value": None,
            "level": None, "innovation": None, "trend": None,
            "feature_windows": {}, "feature_counts": {}, "absence_reasons": {},
            "observed_timestamp": None, "observation_age_seconds": None,
            "max_age_seconds": max_age_seconds, "carried_forward": False,
        }
        if events.empty or pd.isna(reference):
            result["absence_reasons"] = {key: "no_previous_observation" for key in ("level", "innovation", "trend")}
            return result
        limit = self.config.antiguedad_maxima(entity_id) if max_age_seconds is None else float(max_age_seconds)
        end = int(events.index.searchsorted(reference, side="right"))
        latest = events.iloc[end - 1] if end else None
        observed_at = events.index[end - 1] if end else None
        actual = self._numeric_or_none(current_value)
        if actual is None:
            if latest is None or not bool(latest.get("is_valid_value")):
                reason = "no_previous_observation" if latest is None else "invalid_latest_observation"
                result["absence_reasons"] = {key: reason for key in ("level", "innovation", "trend")}
                return result
            actual = self._numeric_or_none(latest.get("value", latest.get("state")))
        if actual is None:
            result["absence_reasons"] = {key: "invalid_latest_observation" for key in ("level", "innovation", "trend")}
            return result
        if observed_at is not None:
            age = (reference - observed_at).total_seconds()
            result.update(observed_timestamp=observed_at.isoformat(), observation_age_seconds=round(age, 6),
                          carried_forward=observed_at < reference)
            if current_value is None and age > limit:
                result["absence_reasons"] = {key: "expired_observation" for key in ("level", "innovation", "trend")}
                return result
        result["original_value"] = actual
        result["level"] = actual
        result["absence_reasons"]["level"] = None

        def median_window(start: pd.Timestamp, finish: pd.Timestamp, *, left_open: bool, right_closed: bool, key: str):
            if partition_start is not None and start < partition_start:
                result["absence_reasons"][key] = "partition_history_unavailable"
                result["feature_counts"][key] = 0
                return None
            left = int(events.index.searchsorted(start, side="right" if left_open else "left"))
            right = int(events.index.searchsorted(finish, side="right" if right_closed else "left"))
            window = events.iloc[left:right]
            result["feature_windows"][key] = {"start": start.isoformat(), "end": finish.isoformat(),
                                               "left_open": left_open, "right_closed": right_closed}
            if window.empty:
                result["absence_reasons"][key] = "empty_window"
                result["feature_counts"][key] = 0
                return None
            if not window["is_valid_value"].all():
                result["absence_reasons"][key] = "invalid_intermediate_observation"
                result["feature_counts"][key] = int(window["is_valid_value"].sum())
                return None
            values = pd.to_numeric(window.get("value", window["state"]), errors="coerce").replace([math.inf, -math.inf], float("nan"))
            values = values.dropna()
            # En ejecucion la sospecha puede llegar antes de quedar anexada al
            # buffer. Para (t-W, t] esa lectura x(t) debe participar en la
            # mediana reciente; nunca se agrega a [t-W, t).
            if (right_closed and current_value is not None and finish == reference
                    and (latest is None or observed_at != reference)):
                values = pd.concat([values, pd.Series([actual])], ignore_index=True)
            result["feature_counts"][key] = int(len(values))
            if len(values) < self.config.temporal_feature_min_observations:
                result["absence_reasons"][key] = "insufficient_window_observations"
                return None
            return float(values.median())

        previous = median_window(reference - pd.Timedelta(seconds=window_seconds), reference,
                                 left_open=False, right_closed=False, key="innovation")
        if previous is not None:
            result["innovation"] = actual - previous
            result["absence_reasons"]["innovation"] = None
        recent = median_window(reference - pd.Timedelta(seconds=window_seconds), reference,
                               left_open=True, right_closed=True, key="trend_recent")
        earlier = median_window(reference - pd.Timedelta(seconds=2 * window_seconds), reference - pd.Timedelta(seconds=window_seconds),
                                left_open=True, right_closed=True, key="trend_previous")
        result["feature_counts"]["trend"] = min(result["feature_counts"].get("trend_recent", 0), result["feature_counts"].get("trend_previous", 0))
        if recent is not None and earlier is not None:
            result["trend"] = recent - earlier
            result["absence_reasons"]["trend"] = None
        else:
            result["absence_reasons"]["trend"] = (
                result["absence_reasons"].get("trend_recent") or result["absence_reasons"].get("trend_previous")
            )
        return result

    def _muestras_modelo_temporal(self, edge, observations, bounds: dict[str, Any]) -> dict[str, pd.DataFrame]:
        source, target = str(edge["source"]), str(edge["target"])
        source_events, target_events = observations.get(source), observations.get(target)
        rows = {name: [] for name in ("level", "innovation", "trend")}
        if source_events is None or target_events is None or source_events.empty or target_events.empty:
            return {name: pd.DataFrame(columns=["target", "context", "timestamp"]) for name in rows}
        start, end = (pd.to_datetime(bounds.get(key), errors="coerce", utc=True) for key in ("start", "end"))
        if pd.isna(start) or pd.isna(end):
            return {name: pd.DataFrame(columns=["target", "context", "timestamp"]) for name in rows}
        candidates = source_events[(source_events.index > start) & (source_events.index <= end)]
        # Solo la ultima lectura estable de un timestamp puede ser x(t).
        candidates = candidates[~candidates.index.duplicated(keep="last")]
        for timestamp, row in candidates.iterrows():
            if not bool(row.get("is_valid_value")):
                continue
            actual = self._numeric_or_none(row.get("value", row.get("state")))
            if actual is None:
                continue
            source_features = self._caracteristicas_continuas(
                source_events, timestamp, source, window_seconds=int(edge["window_seconds"]),
                max_age_seconds=edge.get("max_age_seconds"), current_value=actual, partition_start=start,
            )
            target_features = self._caracteristicas_continuas(
                target_events, timestamp - pd.Timedelta(seconds=int(edge.get("lag_seconds", 0))), target,
                window_seconds=int(edge["window_seconds"]), max_age_seconds=edge.get("max_age_seconds"), partition_start=start,
            )
            for name in rows:
                response, predictor = source_features.get(name), target_features.get(name)
                if response is not None and predictor is not None:
                    rows[name].append({"timestamp": timestamp, "target": response, "context": predictor})
        return {name: pd.DataFrame(values, columns=["timestamp", "target", "context"]) for name, values in rows.items()}

    def _ajustar_modelo_temporal(self, feature: str, fit: pd.DataFrame, tolerance: pd.DataFrame) -> dict[str, Any]:
        metadata = {"feature": feature, "fit_observations": int(len(fit)), "tolerance_observations": int(len(tolerance))}
        if len(fit) < self.config.temporal_model_min_observations or len(tolerance) < self.config.temporal_model_min_observations:
            return {**metadata, "status": "rejected", "reason": "insufficient_model_samples"}
        variance = float(fit["context"].var(ddof=0))
        if not math.isfinite(variance) or variance <= 0:
            return {**metadata, "status": "rejected", "reason": "constant_predictor"}
        slope = float(fit["target"].cov(fit["context"]) / fit["context"].var())
        intercept = float(fit["target"].mean() - slope * fit["context"].mean())
        if not (math.isfinite(slope) and math.isfinite(intercept)):
            return {**metadata, "status": "rejected", "reason": "non_finite_parameters"}
        residuals = (tolerance["target"] - (intercept + slope * tolerance["context"])).abs()
        floor = max(float(fit["target"].std(ddof=0)) * .05, 1e-6)
        threshold = max(float(residuals.quantile(self.config.residual_quantile)), floor)
        return {**metadata, "status": "available", "method": "linear_conditional_residual",
                "slope": slope, "intercept": intercept, "tolerance": threshold,
                "residual_quantile": self.config.residual_quantile,
                "context_min": float(fit["context"].min()), "context_max": float(fit["context"].max())}

    def _calibrar_modelos_temporales(self, edge, observations, partitions) -> dict[str, Any]:
        fit = self._muestras_modelo_temporal(edge, observations, partitions["fit"])
        tolerance = self._muestras_modelo_temporal(edge, observations, partitions["tolerance"])
        models = {name: self._ajustar_modelo_temporal(name, fit[name], tolerance[name])
                  for name in ("level", "innovation", "trend")}
        return {"partitioning": partitions, "models": models,
                "sample_counts": {name: {"fit": len(fit[name]), "tolerance": len(tolerance[name])} for name in models}}

    def _calibrar_decisiones_temporales(self, edges, observations, partitions) -> dict[str, Any]:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in edges:
            if edge.get("accepted") and edge.get("relation_family") == "continuous_continuous":
                by_source.setdefault(str(edge["source"]), []).append(edge)
        results: dict[str, Any] = {}
        for source, source_edges in by_source.items():
            events = observations.get(source)
            bounds = partitions["decision"]
            start, end = (pd.to_datetime(bounds.get(key), errors="coerce", utc=True) for key in ("start", "end"))
            groups: dict[str, list[dict[str, Any]]] = {}
            if events is not None and not pd.isna(start) and not pd.isna(end):
                candidates = events[(events.index > start) & (events.index <= end)]
                for timestamp, row in candidates[~candidates.index.duplicated(keep="last")].iterrows():
                    actual = self._numeric_or_none(row.get("value", row.get("state"))) if bool(row.get("is_valid_value")) else None
                    scores = self._puntajes_familiares(source, timestamp, actual, source_edges, observations, partition_start=start)
                    families = scores["available_families"]
                    if families:
                        key = "+".join(families)
                        groups.setdefault(key, []).append({"score": scores["decision_score"], "neighbors": scores["evaluable_neighbors"]})
            calibrated = {}
            for key, values in groups.items():
                if len(values) < self.config.temporal_decision_min_observations:
                    calibrated[key] = {"status": "insufficient_calibration", "samples": len(values)}
                    continue
                series = pd.Series([item["score"] for item in values])
                calibrated[key] = {"status": "available", "samples": len(values),
                                   "threshold": float(series.quantile(self.config.residual_quantile, interpolation="higher")),
                                   "neighbor_counts": sorted({int(item["neighbors"]) for item in values}),
                                   "partition": "decision", "quantile": self.config.residual_quantile}
            results[source] = {"partition": bounds, "combinations": calibrated}
        return results

    def _calibrar_respuesta(self, pair: pd.DataFrame) -> dict[str, Any] | None:
        """Recta contextual, no clasificador de fallas; residuos fuera del ajuste.

        Ajuste con el primer 70 % de Desarrollo y tolerancia empirica con el
        ultimo 30 %. No es un intervalo de confianza ni una prueba causal.
        """
        split = int(len(pair) * .7)
        train, calibration = pair.iloc[:split], pair.iloc[split:]
        if len(train) < 20 or len(calibration) < 10 or train["target"].nunique() < 2:
            return None
        slope = float(train["source"].cov(train["target"]) / train["target"].var())
        intercept = float(train["source"].mean() - slope * train["target"].mean())
        residuals = (calibration["source"] - (intercept + slope * calibration["target"])).abs()
        tolerance = max(float(residuals.quantile(self.config.residual_quantile)),
                        float(train["source"].std(ddof=0)) * .05, 1e-6)
        return {
            "method": "linear_conditional_residual", "partition": "development",
            "slope": slope, "intercept": intercept, "tolerance": tolerance,
            "residual_quantile": self.config.residual_quantile,
            "fit_observations": len(train), "calibration_observations": len(calibration),
            "context_min": float(train["target"].min()), "context_max": float(train["target"].max()),
        }

    def _seleccionar_ventana(self, source, target, catalogo, grids) -> dict[str, Any]:
        candidates = []
        for seconds in sorted(grids):
            signals, _, _, index = grids[seconds]
            candidates.append(self._mejor_relacion(
                source, target, signals[source], signals[target],
                catalogo[source]["sensor_kind"], catalogo[target]["sensor_kind"],
                len(index), window_seconds=seconds,
            ))
        # La resolucion mas fina elegible prevalece sobre una correlacion mayor.
        # Sin candidatas elegibles, mostrar la mas fina para auditar el rechazo.
        selected = next((item for item in candidates if item["threshold_eligible"]), candidates[0])
        selected["window_selection"] = "smallest_eligible_else_smallest"
        selected["window_candidates"] = [
            {key: item[key] for key in (
                "window_seconds", "lag_seconds", "n_observations", "coverage_ratio",
                "association_signed", "stable_slices", "sign_consistency",
                "threshold_eligible", "rejection_reasons",
            )} for item in candidates
        ]
        # Copiar motivos: top_k se aplica despues y no es un control de ventana.
        for item in selected["window_candidates"]:
            item["rejection_reasons"] = list(item["rejection_reasons"])
        return selected

    def _mejor_relacion(
        self,
        source: str,
        target: str,
        source_signal: pd.Series,
        target_signal: pd.Series,
        source_kind: str,
        target_kind: str,
        total_windows: int,
        *,
        window_seconds: int | None = None,
    ) -> dict[str, Any]:
        seconds = window_seconds or self.config.window_seconds
        candidates = []
        # Como maximo ocho desfases: se conserva el horizonte anterior sin
        # multiplicar la busqueda por cada segundo de una grilla mas fina.
        for offset in self.config.desfases(seconds):
            lag = offset // seconds
            aligned_source = source_signal
            aligned_target = target_signal.shift(lag)
            encoded = self._codificar_par(
                aligned_source,
                aligned_target,
                source_kind,
                target_kind,
            )
            pair = encoded["pair"]
            association = self._correlacion(pair)
            stability = self._estabilidad(pair, association or 0.0)
            coverage = float(len(pair) / total_windows) if total_windows else 0.0
            checks = {
                "insufficient_observations": len(pair) < self.config.min_observations,
                "insufficient_pair_coverage": coverage < self.config.min_pair_coverage,
                "weak_association": abs(association or 0.0) < self.config.association_threshold,
                "temporal_instability": stability["stable_slices"] < self.config.min_stable_slices,
                "sign_instability": stability["sign_consistency"] < self.config.min_sign_consistency,
            }
            candidates.append(
                {
                    **encoded,
                    "association": association,
                    "lag_windows": lag,
                    "n_observations": int(len(pair)),
                    "coverage_ratio": coverage,
                    "stability": stability,
                    "rejection_reasons": [key for key, failed in checks.items() if failed],
                }
            )
        best = max(
            candidates,
            key=lambda item: (
                not item["rejection_reasons"],
                -1.0 if item["association"] is None else abs(item["association"]),
                item["n_observations"],
                -item["lag_windows"],
            ),
        )
        association = best["association"] or 0.0
        stability = best["stability"]
        rejection_reasons = best["rejection_reasons"]
        threshold_eligible = not rejection_reasons
        family, metric = self._familia_metrica(
            source_kind,
            target_kind,
            best.get("source_state"),
            best.get("target_state"),
        )
        conditional_probability, cooccurrence = self._coocurrencia(best["pair"])
        return {
            "edge_id": f"{source}->{target}",
            "source": source,
            "target": target,
            "source_kind": source_kind,
            "target_kind": target_kind,
            "relation_family": family,
            "metric": metric,
            "association_signed": self._redondear(association),
            "strength": self._redondear(abs(association)),
            "expected_sign": "positive" if association >= 0 else "negative",
            "lag_windows": int(best["lag_windows"]),
            "lag_seconds": int(best["lag_windows"] * seconds),
            "window_seconds": seconds,
            "lag_selection": "quality_before_strength",
            "lag_candidates": [
                {"lag_seconds": item["lag_windows"] * seconds,
                 "association": self._redondear_o_none(item["association"]),
                 "n_observations": item["n_observations"],
                 "coverage_ratio": self._redondear(item["coverage_ratio"]),
                 "rejection_reasons": item["rejection_reasons"]}
                for item in candidates
            ],
            "expected_response": (
                self._calibrar_respuesta(best["pair"])
                if threshold_eligible and source_kind == target_kind == "continuous" else None
            ),
            "n_observations": int(best["n_observations"]),
            "coverage_ratio": self._redondear(best["coverage_ratio"]),
            "stability_score": self._redondear(stability["stability_score"]),
            "stable_slices": int(stability["stable_slices"]),
            "valid_slices": int(stability["valid_slices"]),
            "slice_associations": stability["slice_associations"],
            "sign_consistency": self._redondear(stability["sign_consistency"]),
            "source_state": best.get("source_state"),
            "target_state": best.get("target_state"),
            "conditional_probability": self._redondear_o_none(conditional_probability),
            "cooccurrence_count": int(cooccurrence),
            "threshold_eligible": threshold_eligible,
            "accepted": False,
            "rejection_reasons": rejection_reasons,
            "acceptance_checks": {
                "observations": best["n_observations"] >= self.config.min_observations,
                "coverage": best["coverage_ratio"] >= self.config.min_pair_coverage,
                "association": abs(association) >= self.config.association_threshold,
                "stable_slices": stability["stable_slices"] >= self.config.min_stable_slices,
                "sign_consistency": stability["sign_consistency"] >= self.config.min_sign_consistency,
                "top_k": False,
            },
        }

    def _codificar_par(
        self,
        source: pd.Series,
        target: pd.Series,
        source_kind: str,
        target_kind: str,
    ) -> dict[str, Any]:
        base = pd.concat([source.rename("source"), target.rename("target")], axis=1).dropna()
        if base.empty:
            return {"pair": pd.DataFrame(columns=["source", "target"]), "source_state": None, "target_state": None}
        if source_kind != "multistate" and target_kind != "multistate":
            pair = base.apply(pd.to_numeric, errors="coerce").dropna()
            return {"pair": pair, "source_state": None, "target_state": None}

        source_states = (
            [None]
            if source_kind != "multistate"
            else base["source"].astype(str).value_counts().head(8).index.tolist()
        )
        target_states = (
            [None]
            if target_kind != "multistate"
            else base["target"].astype(str).value_counts().head(8).index.tolist()
        )
        candidates = []
        for source_state in source_states:
            for target_state in target_states:
                encoded = pd.DataFrame(index=base.index)
                if source_state is None:
                    encoded["source"] = pd.to_numeric(base["source"], errors="coerce")
                else:
                    encoded["source"] = base["source"].astype(str).eq(str(source_state)).astype(float)
                if target_state is None:
                    encoded["target"] = pd.to_numeric(base["target"], errors="coerce")
                else:
                    encoded["target"] = base["target"].astype(str).eq(str(target_state)).astype(float)
                encoded = encoded.dropna()
                association = self._correlacion(encoded)
                candidates.append(
                    {
                        "pair": encoded,
                        "source_state": source_state,
                        "target_state": target_state,
                        "association": association,
                    }
                )
        if not candidates:
            return {"pair": pd.DataFrame(columns=["source", "target"]), "source_state": None, "target_state": None}
        return max(
            candidates,
            key=lambda item: -1.0 if item["association"] is None else abs(item["association"]),
        )

    def _correlacion(self, pair: pd.DataFrame) -> float | None:
        if len(pair) < 2:
            return None
        if pair["source"].nunique(dropna=True) < 2 or pair["target"].nunique(dropna=True) < 2:
            return None
        value = pair["source"].corr(pair["target"], method="pearson")
        return None if pd.isna(value) or not math.isfinite(float(value)) else float(value)

    def _estabilidad(self, pair: pd.DataFrame, full_association: float) -> dict[str, Any]:
        if pair.empty:
            return {
                "stable_slices": 0,
                "valid_slices": 0,
                "slice_associations": [],
                "sign_consistency": 0.0,
                "stability_score": 0.0,
            }
        slice_values: list[float | None] = []
        total = len(pair)
        for index in range(self.config.stability_slices):
            start = (index * total) // self.config.stability_slices
            end = ((index + 1) * total) // self.config.stability_slices
            value = self._correlacion(pair.iloc[start:end])
            slice_values.append(value)
        valid = [value for value in slice_values if value is not None]
        full_sign = 1 if full_association >= 0 else -1
        same_sign = [
            value
            for value in valid
            if (1 if value >= 0 else -1) == full_sign
        ]
        stable = [
            value
            for value in same_sign
            if abs(value) >= self.config.stability_association_threshold
        ]
        sign_consistency = len(same_sign) / len(valid) if valid else 0.0
        stability_score = len(stable) / self.config.stability_slices
        return {
            "stable_slices": len(stable),
            "valid_slices": len(valid),
            "slice_associations": [self._redondear_o_none(value) for value in slice_values],
            "sign_consistency": sign_consistency,
            "stability_score": stability_score,
        }

    def _familia_metrica(
        self,
        source_kind: str,
        target_kind: str,
        source_state: str | None,
        target_state: str | None,
    ) -> tuple[str, str]:
        if source_state is not None or target_state is not None:
            return "mixed" if source_kind != target_kind else "state_state", "one_vs_rest_phi"
        if source_kind == "continuous" and target_kind == "continuous":
            return "continuous_continuous", "pearson"
        if source_kind == "binary" and target_kind == "binary":
            return "state_state", "phi_cooccurrence"
        return "mixed", "point_biserial"

    def _coocurrencia(self, pair: pd.DataFrame) -> tuple[float | None, int]:
        if pair.empty:
            return None, 0
        source_active = pair["source"] > pair["source"].median()
        target_active = pair["target"] > pair["target"].median()
        denominator = int(source_active.sum())
        cooccurrence = int((source_active & target_active).sum())
        probability = cooccurrence / denominator if denominator else None
        return probability, cooccurrence

    def _aplicar_top_k(self, edges: list[dict[str, Any]]) -> None:
        sources = sorted({str(edge["source"]) for edge in edges})
        for source in sources:
            candidates = [
                edge
                for edge in edges
                if edge["source"] == source and edge["threshold_eligible"]
            ]
            candidates.sort(
                key=lambda edge: (
                    edge["strength"] * edge["stability_score"],
                    edge["n_observations"],
                    -edge["lag_windows"],
                    edge["target"],
                ),
                reverse=True,
            )
            selected = {edge["edge_id"] for edge in candidates[: self.config.top_k]}
            for edge in edges:
                if edge["source"] != source:
                    continue
                edge["accepted"] = edge["edge_id"] in selected
                edge["acceptance_checks"]["top_k"] = edge["accepted"]
                if edge["threshold_eligible"] and not edge["accepted"]:
                    edge["rejection_reasons"].append("outside_top_k")

    def _construir_matrices(
        self,
        entity_ids: list[str],
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        edge_lookup = {(edge["source"], edge["target"]): edge for edge in edges}
        association_values = []
        accepted_values = []
        for source in entity_ids:
            association_row = []
            accepted_row = []
            for target in entity_ids:
                if source == target:
                    association_row.append(1.0)
                    accepted_row.append(True)
                    continue
                edge = edge_lookup.get((source, target))
                association_row.append(edge.get("association_signed") if edge else None)
                accepted_row.append(bool(edge and edge.get("accepted")))
            association_values.append(association_row)
            accepted_values.append(accepted_row)
        return {
            "entities": entity_ids,
            "association": association_values,
            "accepted": accepted_values,
        }

    def _puntajes_familiares(self, source, timestamp, current_value, edges, observations, *, partition_start=None):
        families = ("level", "innovation", "trend")
        source_events = observations.get(source)
        family_rows = {name: [] for name in families}
        details: dict[str, dict[str, Any]] = {}
        total_weights = {name: 0.0 for name in families}
        evaluated_weights = {name: 0.0 for name in families}
        source_feature_cache: dict[int, dict[str, Any]] = {}
        for edge in edges:
            weight = abs(float(edge.get("association_signed", 0.0))) * float(edge.get("stability_score", 0.0))
            model_bundle = edge.get("temporal_models", {}).get("models", {})
            target = str(edge["target"])
            target_events = observations.get(target)
            if source_events is None or target_events is None:
                continue
            window = int(edge.get("window_seconds") or self.config.window_seconds)
            if window not in source_feature_cache:
                source_feature_cache[window] = self._caracteristicas_continuas(
                    source_events, timestamp, source, window_seconds=window,
                    max_age_seconds=edge.get("max_age_seconds"), current_value=current_value,
                    partition_start=partition_start,
                )
            source_features = source_feature_cache[window]
            target_features = self._caracteristicas_continuas(
                target_events, timestamp - pd.Timedelta(seconds=int(edge.get("lag_seconds", 0))), target,
                window_seconds=window, max_age_seconds=edge.get("max_age_seconds"), partition_start=partition_start,
            )
            neighbor = details.setdefault(target, {"neighbor_id": target, "weight": self._redondear(weight),
                                                    "features": {"source": source_features, "context": target_features}})
            for family in families:
                model = model_bundle.get(family, {})
                check: dict[str, Any] = {"status": "unavailable", "reason": None, "ratio": None,
                                         "residual": None, "tolerance": None, "extrapolated": False}
                if model.get("status") != "available":
                    check["reason"] = model.get("reason", "model_unavailable")
                elif source_features.get(family) is None or target_features.get(family) is None:
                    check["reason"] = (source_features.get("absence_reasons", {}).get(family)
                                       or target_features.get("absence_reasons", {}).get(family)
                                       or "feature_unavailable")
                else:
                    total_weights[family] += weight
                    predictor, response = target_features[family], source_features[family]
                    expected = model["intercept"] + model["slope"] * predictor
                    if predictor < model["context_min"] or predictor > model["context_max"]:
                        check.update(status="excluded", reason="predictor_extrapolated", extrapolated=True,
                                     expected_value=expected, predictor_value=predictor)
                    else:
                        residual = abs(response - expected)
                        ratio = residual / max(model["tolerance"], 1e-6)
                        check.update(status="evaluated", ratio=ratio, residual=residual,
                                     tolerance=model["tolerance"], expected_value=expected,
                                     predictor_value=predictor)
                        evaluated_weights[family] += weight
                        family_rows[family].append((ratio, weight))
                neighbor[f"{family}_check"] = check
        family_scores = {}
        available = []
        for family in families:
            denominator = total_weights[family]
            coverage = evaluated_weights[family] / denominator if denominator else 0.0
            score = (sum(ratio * weight for ratio, weight in family_rows[family]) / evaluated_weights[family]
                     if evaluated_weights[family] else None)
            family_scores[family] = {"score": self._redondear_o_none(score), "coverage": self._redondear(coverage),
                                      "eligible_weight": self._redondear(denominator),
                                      "evaluated_weight": self._redondear(evaluated_weights[family]),
                                      "evaluable_neighbors": len(family_rows[family]),
                                      "status": "available" if score is not None and coverage >= self.config.runtime_coverage_threshold else "insufficient_coverage"}
            if family_scores[family]["status"] == "available":
                available.append(family)
        return {"family_scores": family_scores, "available_families": available,
                "decision_score": max((family_scores[name]["score"] for name in available), default=None),
                "evaluable_neighbors": max((family_scores[name]["evaluable_neighbors"] for name in available), default=0),
                "neighbors": details}

    def _evaluar_sospecha(
        self, *, sospecha, signals, stats, edges, map_quality, concurrent_entities,
        continuous_observations=None, decision_calibrations=None,
    ) -> dict[str, Any]:
        continuous_edges = [edge for edge in edges if edge.get("relation_family") == "continuous_continuous"]
        if map_quality.get("version") == self.MAP_VERSION and continuous_edges and continuous_observations is not None:
            return self._evaluar_sospecha_v5(
                sospecha=sospecha, signals=signals, stats=stats, edges=edges, map_quality=map_quality,
                concurrent_entities=concurrent_entities, continuous_observations=continuous_observations,
                decision_calibrations=decision_calibrations or {},
            )
        return self._evaluar_sospecha_v4(
            sospecha=sospecha, signals=signals, stats=stats, edges=edges, map_quality=map_quality,
            concurrent_entities=concurrent_entities, continuous_observations=continuous_observations,
        )

    def _evaluar_sospecha_v5(
        self, *, sospecha, signals, stats, edges, map_quality, concurrent_entities,
        continuous_observations, decision_calibrations,
    ) -> dict[str, Any]:
        # Se reutiliza el envoltorio y las protecciones de enrutamiento v4;
        # sus votos continuos se sustituyen completamente por la decision v5.
        base = self._evaluar_sospecha_v4(
            sospecha=sospecha, signals=signals, stats=stats, edges=edges, map_quality=map_quality,
            concurrent_entities=concurrent_entities, continuous_observations=continuous_observations,
        )
        source = str(sospecha.get("entity_id", ""))
        timestamp = pd.to_datetime(sospecha.get("timestamp"), errors="coerce", utc=True)
        current = self._numeric_or_none(sospecha.get("value", (sospecha.get("reading") or {}).get("value")))
        if current is None and source in continuous_observations and not pd.isna(timestamp):
            current, _ = self._observacion_continua(continuous_observations[source], timestamp, source)
        continuous_edges = [edge for edge in edges if edge.get("relation_family") == "continuous_continuous"]
        scores = self._puntajes_familiares(source, timestamp, current, continuous_edges, continuous_observations)
        families = scores["available_families"]
        combination = "+".join(families)
        calibration = (decision_calibrations.get(source, {}).get("combinations", {}).get(combination) if combination else None)
        exclusions = []
        for item in scores["neighbors"].values():
            for family in ("level", "innovation", "trend"):
                check = item.get(f"{family}_check", {})
                if check.get("reason"):
                    exclusions.append({"neighbor_id": item["neighbor_id"], "family": family, "reason": check["reason"]})
        if not families:
            outcome, threshold = "insufficient_coverage", None
        elif not calibration or calibration.get("status") != "available":
            outcome, threshold = "unexplained", None
        else:
            threshold = float(calibration["threshold"])
            outcome = "contradicted" if scores["decision_score"] > threshold else "supported"
        base["context_outcome"] = outcome
        base["context_score"] = -1.0 if outcome == "contradicted" else 1.0 if outcome == "supported" else None
        base["coherence_score"] = 0.0 if outcome == "contradicted" else 1.0 if outcome == "supported" else None
        base["context_evaluable"] = bool(families and calibration and calibration.get("status") == "available")
        base["support_score"] = 1.0 if outcome == "supported" else 0.0
        base["contradiction_score"] = 1.0 if outcome == "contradicted" else 0.0
        base["score_semantics"] = "v5_outcome_indicator_not_probability"
        base["decision_method"] = "calibrated_max_weighted_family_ratio"
        base["family_scores"] = scores["family_scores"]
        base["available_families"] = families
        base["decision_score"] = self._redondear_o_none(scores["decision_score"])
        base["decision_threshold"] = self._redondear_o_none(threshold)
        base["decision_exclusions"] = exclusions
        base["decision_calibration"] = calibration
        base["coverage"]["observed_neighbors"] = scores["evaluable_neighbors"]
        base["coverage"]["ratio"] = max((item["coverage"] for item in scores["family_scores"].values()), default=0.0)
        base["coverage"]["status"] = "sufficient" if families else "insufficient"
        for evidence in base["evidence"]:
            detail = scores["neighbors"].get(evidence.get("neighbor_id"))
            if detail:
                evidence.update({key: detail[key] for key in ("level_check", "innovation_check", "trend_check")})
                evidence["evidence_method"] = "v5_temporal_feature_models"
                evidence["observed_delta"] = None
        base["supports"] = []
        base["contradictions"] = []
        base["unknowns"] = list(base["evidence"])
        routing = self._decidir_ruta(outcome, map_quality)
        base["routing"] = routing
        base["contextual_class"] = "environmental_anomaly" if routing["terminal"] else None
        base["suspicion_effect"] = "decrease" if outcome == "supported" else "increase" if outcome == "contradicted" else "maintain"
        return base

    def _evaluar_sospecha_v4(
        self,
        *,
        sospecha: dict[str, Any],
        signals: dict[str, pd.Series],
        stats: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        map_quality: dict[str, Any],
        concurrent_entities: list[str],
        continuous_observations: dict[str, pd.DataFrame] | None = None,
    ) -> dict[str, Any]:
        entity_id = str(sospecha.get("entity_id", ""))
        bucket = self._bucket_timestamp(sospecha.get("timestamp"))
        target_signal = signals.get(entity_id)
        target_stats = stats.get(entity_id, {})
        target_value = self._signal_value(target_signal, bucket)
        timestamp = pd.to_datetime(sospecha.get("timestamp"), errors="coerce", utc=True)
        if continuous_observations is not None and entity_id in continuous_observations:
            target_value, _ = self._observacion_contextual(
                continuous_observations[entity_id], timestamp, entity_id
            )
        # La magnitud sospechosa es la lectura puntual, no una mediana que
        # podria diluir precisamente el salto que detecto F1.
        reading_value = self._numeric_or_none(sospecha.get("value", (sospecha.get("reading") or {}).get("value")))
        if reading_value is not None and sospecha.get("sensor_kind") == "continuous":
            target_value = reading_value
        if target_value is None:
            target_value = self._numeric_or_none(
                sospecha.get("value", (sospecha.get("reading") or {}).get("value"))
            )
        target_z = self._standardized_value(target_value, target_stats)
        pattern = sospecha.get("suspected_pattern") or sospecha.get("pattern")
        directional_applicable = target_value is not None and pattern != "missing_gap"
        # ``delta_value`` remains a presentation aid for the legacy
        # directional evidence used by state and mixed relations.  It cannot
        # enter the continuous conditional-level calculation.
        has_directional_relation = any(
            edge.get("relation_family") != "continuous_continuous" for edge in edges
        )
        if has_directional_relation and abs(target_z) < 0.05:
            delta = self._numeric_or_none((sospecha.get("reading") or {}).get("delta_value"))
            if delta is not None and delta != 0:
                target_z = 1.0 if delta > 0 else -1.0

        evidence = []
        eligible_weight = 0.0
        observed_weight = 0.0
        support_mass = 0.0
        contradiction_mass = 0.0
        for edge in edges:
            weight = abs(float(edge.get("association_signed", 0.0))) * float(
                edge.get("stability_score", 0.0)
            )
            eligible_weight += weight
            context_id = str(edge["target"])
            context_bucket = (
                bucket - pd.Timedelta(seconds=int(edge.get("lag_seconds", 0)))
                if bucket is not None
                else None
            )
            context_signal = signals.get(context_id)
            context_value = self._signal_value(context_signal, context_bucket)
            observation_details: dict[str, Any] = {}
            if continuous_observations is not None and context_id in continuous_observations:
                # El desfase es una referencia temporal, no permiso para mirar adelante.
                reference = timestamp - pd.Timedelta(seconds=int(edge.get("lag_seconds", 0)))
                context_value, observation_details = self._observacion_contextual(
                    continuous_observations[context_id], reference, context_id, edge
                )
            if context_value is None:
                is_continuous_pair = edge.get("relation_family") == "continuous_continuous"
                evidence.append(
                    {
                        "neighbor_id": context_id,
                        "weight": self._redondear(weight),
                        "association_signed": edge.get("association_signed"),
                        "expected_sign": edge.get("expected_sign"),
                        "lag_seconds": edge.get("lag_seconds", 0),
                        "lag_error_seconds": None,
                        "observed_value": None,
                        "observed_delta": None,
                        "evidence_method": (
                            "conditional_level_residual" if is_continuous_pair else "directional_association"
                        ),
                        "residual": None,
                        "tolerance": None,
                        "ratio": None,
                        "contribution": 0.0,
                        "expected_direction_or_state": (
                            self._expected_level_text(edge)
                            if is_continuous_pair else self._expected_text(edge, target_z)
                        ),
                        "c_j_raw": None,
                        "c_j_normalized": None,
                        "weighted_contribution": 0.0,
                        "observation_status": "missing",
                        "temporal_match": False,
                        "n_observations": edge.get("n_observations"),
                        "coverage_ratio": edge.get("coverage_ratio"),
                        "stability_score": edge.get("stability_score"),
                        **observation_details,
                    }
                )
                continue

            # A continuous-continuous relation is evidence about conditional
            # level, not about which side of either sensor's historical centre
            # the readings fall on.  In particular, do not use source_z,
            # context_z, association sign, or delta_value as a proxy here.
            if edge.get("relation_family") == "continuous_continuous":
                model = edge.get("expected_response")
                actual = self._numeric_or_none(target_value)
                context_numeric = self._numeric_or_none(context_value)
                unknown_reason = None
                if not model:
                    unknown_reason = "missing_conditional_level_model"
                elif not directional_applicable or actual is None:
                    unknown_reason = "missing_evaluable_target_level"
                elif context_numeric is None:
                    unknown_reason = "invalid_context_level"

                if unknown_reason is not None:
                    evidence.append(
                        {
                            "neighbor_id": context_id,
                            "weight": self._redondear(weight),
                            "association_signed": edge.get("association_signed"),
                            "expected_sign": edge.get("expected_sign"),
                            "lag_seconds": edge.get("lag_seconds", 0),
                            "lag_error_seconds": 0,
                            "observed_value": self._serializable_value(context_value),
                            "observed_delta": None,
                            "expected_direction_or_state": self._expected_level_text(edge),
                            "evidence_method": "conditional_level_residual",
                            "residual": None,
                            "tolerance": None,
                            "ratio": None,
                            "contribution": 0.0,
                            "unknown_reason": unknown_reason,
                            "c_j_raw": None,
                            "c_j_normalized": None,
                            "weighted_contribution": 0.0,
                            "magnitude_check": {"status": "not_calibrated" if not model else "not_evaluable"},
                            "observation_status": "unknown",
                            "temporal_match": True,
                            "n_observations": edge.get("n_observations"),
                            "coverage_ratio": edge.get("coverage_ratio"),
                            "stability_score": edge.get("stability_score"),
                            **observation_details,
                        }
                    )
                    # Unknown conditional evidence must not pretend to cover
                    # this neighbor, even when a contextual observation exists.
                    continue

                expected = model["intercept"] + model["slope"] * context_numeric
                residual = abs(actual - expected)
                tolerance = max(model["tolerance"], 1e-6)
                ratio = residual / tolerance
                c_raw = 1.0 - ratio
                c_normalized = max(-1.0, min(1.0, c_raw))
                contribution = weight * c_normalized
                magnitude = {
                    "status": "compatible" if residual <= tolerance else "incompatible",
                    "expected_value": expected,
                    "actual_value": actual,
                    "lower": expected - tolerance,
                    "upper": expected + tolerance,
                    "absolute_residual": residual,
                    "tolerance": tolerance,
                    "ratio": ratio,
                    "extrapolated": not model["context_min"] <= context_numeric <= model["context_max"],
                    "partition": "development",
                    "method": model["method"],
                }
                observed_weight += weight
                # Los limites son inclusivos. Un epsilon evita que la
                # representacion binaria de, por ejemplo, 1.20 convierta el
                # umbral documentado en 1.2000000000000002.
                threshold_epsilon = 1e-12
                if c_normalized >= self.config.evidence_threshold - threshold_epsilon:
                    status = "supports"
                    support_mass += contribution
                elif c_normalized <= -self.config.evidence_threshold + threshold_epsilon:
                    status = "contradicts"
                    contradiction_mass += abs(contribution)
                else:
                    status = "observed_neutral"
                evidence.append(
                    {
                        "neighbor_id": context_id,
                        "weight": self._redondear(weight),
                        "association_signed": edge.get("association_signed"),
                        "expected_sign": edge.get("expected_sign"),
                        "lag_seconds": edge.get("lag_seconds", 0),
                        "lag_error_seconds": 0,
                        "observed_value": self._serializable_value(context_value),
                        "observed_delta": None,
                        "expected_direction_or_state": self._expected_level_text(edge),
                        "evidence_method": "conditional_level_residual",
                        "residual": self._redondear(residual),
                        "tolerance": self._redondear(tolerance),
                        "ratio": self._redondear(ratio),
                        "contribution": self._redondear(contribution),
                        "c_j_raw": self._redondear(c_raw),
                        "c_j_normalized": self._redondear(c_normalized),
                        "weighted_contribution": self._redondear(contribution),
                        "magnitude_check": magnitude,
                        "observation_status": status,
                        "temporal_match": True,
                        "n_observations": edge.get("n_observations"),
                        "coverage_ratio": edge.get("coverage_ratio"),
                        "stability_score": edge.get("stability_score"),
                        **observation_details,
                    }
                )
                continue

            observed_weight += weight
            context_stats = edge.get("target_stats") or stats.get(context_id, {})
            if edge.get("target_state") is not None:
                state = str(edge["target_state"])
                context_component = 1.0 if str(context_value) == state else 0.0
                probability = float(context_stats.get("state_probabilities", {}).get(state, 0.0))
                scale = max(math.sqrt(max(probability * (1.0 - probability), 0.0)), 1e-6)
                context_z = (context_component - probability) / scale
            else:
                context_z = self._standardized_value(context_value, context_stats)

            source_stats = edge.get("source_stats") or target_stats
            source_z = self._standardized_value(target_value, source_stats)
            if edge.get("source_state") is not None:
                state = str(edge["source_state"])
                source_z = float(str(target_value) == state) - float(source_stats.get("state_probabilities", {}).get(state, 0))
            elif abs(source_z) < .05:
                source_z = target_z
            target_direction = 1.0 if source_z >= 0 else -1.0
            relation_direction = 1.0 if float(edge.get("association_signed", 0.0)) >= 0 else -1.0
            c_raw = relation_direction * target_direction * context_z
            if not directional_applicable:
                c_raw = 0.0
            c_normalized = max(-1.0, min(1.0, c_raw / 2.0))
            magnitude = {"status": "not_calibrated"}
            model = edge.get("expected_response")
            actual, context_numeric = self._numeric_or_none(target_value), self._numeric_or_none(context_value)
            if model and directional_applicable and actual is not None and context_numeric is not None:
                expected = model["intercept"] + model["slope"] * context_numeric
                residual = abs(actual - expected)
                tolerance = max(model["tolerance"], 1e-6)
                compatible = residual <= tolerance
                magnitude = {
                    "status": "compatible" if compatible else "incompatible",
                    "expected_value": expected, "actual_value": actual,
                    "lower": expected - tolerance, "upper": expected + tolerance,
                    "absolute_residual": residual, "tolerance": tolerance,
                    "extrapolated": not model["context_min"] <= context_numeric <= model["context_max"],
                    "partition": "development", "method": model["method"],
                }
                if not compatible:
                    c_normalized = min(c_normalized, -min(1.0, (residual / tolerance - 1.0) / 2.0))
            contribution = weight * c_normalized
            if c_normalized >= self.config.evidence_threshold:
                status = "supports"
                support_mass += contribution
            elif c_normalized <= -self.config.evidence_threshold:
                status = "contradicts"
                contradiction_mass += abs(contribution)
            else:
                status = "observed_neutral"
            evidence.append(
                {
                    "neighbor_id": context_id,
                    "weight": self._redondear(weight),
                    "association_signed": edge.get("association_signed"),
                    "expected_sign": edge.get("expected_sign"),
                    "lag_seconds": edge.get("lag_seconds", 0),
                    "lag_error_seconds": 0,
                    "observed_value": self._serializable_value(context_value),
                    "observed_delta": self._redondear(context_z),
                    "evidence_method": "directional_association",
                    "residual": None,
                    "tolerance": None,
                    "ratio": None,
                    "contribution": self._redondear(contribution),
                    "expected_direction_or_state": self._expected_text(edge, target_z),
                    "c_j_raw": self._redondear(c_raw),
                    "c_j_normalized": self._redondear(c_normalized),
                    "weighted_contribution": self._redondear(contribution),
                    "magnitude_check": magnitude,
                    "observation_status": status,
                    "temporal_match": True,
                    "n_observations": edge.get("n_observations"),
                    "coverage_ratio": edge.get("coverage_ratio"),
                    "stability_score": edge.get("stability_score"),
                    **observation_details,
                }
            )

        coverage_ratio = observed_weight / eligible_weight if eligible_weight > 0 else 0.0
        support_score = support_mass / observed_weight if observed_weight > 0 else 0.0
        contradiction_score = (
            contradiction_mass / observed_weight if observed_weight > 0 else 0.0
        )
        context_score = (
            (support_mass - contradiction_mass) / observed_weight
            if observed_weight > 0
            else 0.0
        )
        if not edges:
            outcome = "no_learned_relations"
            effect = "maintain"
        elif (
            coverage_ratio < self.config.runtime_coverage_threshold
            or observed_weight <= 0
        ):
            outcome = "insufficient_coverage"
            effect = "maintain"
        elif (
            contradiction_score >= self.config.decision_threshold
            and contradiction_score > support_score
        ):
            outcome = "contradicted"
            effect = "increase"
        elif (
            support_score >= self.config.decision_threshold
            and support_score > contradiction_score
        ):
            outcome = "supported"
            effect = "decrease"
        else:
            outcome = "unexplained"
            effect = "maintain"

        common_candidate = len(concurrent_entities) >= self.config.common_cause_min_entities
        statistical_score = float(sospecha.get("suspicion_score") or 0.0)
        if effect == "decrease":
            adjusted = statistical_score * (1.0 - 0.35 * support_score)
        elif effect == "increase":
            adjusted = statistical_score + (1.0 - statistical_score) * 0.35 * contradiction_score
        else:
            adjusted = statistical_score

        # No atribuir ni vetar por coincidencia. La deteccion de infraestructura
        # requiere una fuente independiente que este filtro aun no recibe.
        common_cause = {
            "candidate": False,
            "status": "not_evaluated",
            "identifiable": False,
            "contextual_coincidence": common_candidate,
            "concurrent_entities": concurrent_entities,
            "window_seconds": self.config.common_cause_window_seconds,
            "grouping_rule": "accepted_neighbors_same_pattern_past_window",
            "independent_evidence_available": False,
            "independent_domains_required": ["network", "power", "integration"],
            "interpretation": (
                "Coincidencia en entidades relacionadas; no demuestra una causa comun ni bloquea la salida ambiental."
                if common_candidate
                else "Sin coincidencia contextual suficiente. La causa comun no se evalua sin telemetria independiente."
            ),
        }
        covered_neighbors = sum(
            item["observation_status"] not in {"missing", "unknown"} for item in evidence
        )
        routing = self._decidir_ruta(outcome, map_quality)
        environmental = routing["terminal"]
        result = {
            "observation_id": sospecha.get("observation_id"),
            "event_id": f"{entity_id}@{sospecha.get('timestamp')}",
            "episode": sospecha.get("episode"),
            "entity_id": entity_id,
            "timestamp": sospecha.get("timestamp"),
            "sensor_kind": sospecha.get("sensor_kind"),
            "suspected_pattern": sospecha.get("suspected_pattern", sospecha.get("pattern")),
            "statistical_score": self._redondear(statistical_score),
            "context_score": self._redondear(context_score) if observed_weight > 0 else None,
            "coherence_score": (
                self._redondear((context_score + 1.0) / 2.0) if observed_weight > 0 else None
            ),
            "context_evaluable": observed_weight > 0,
            "context_applicability": "applicable" if edges else "no_learned_relations",
            "directional_evidence_applicable": directional_applicable,
            "support_score": self._redondear(support_score),
            "contradiction_score": self._redondear(contradiction_score),
            "adjusted_suspicion_score": self._redondear(max(0.0, min(1.0, adjusted))),
            "context_outcome": outcome,
            "contextual_class": "environmental_anomaly" if environmental else None,
            "routing": routing,
            "alert": (
                {
                    "type": "environmental_anomaly",
                    "level": "informational",
                    "message": "Cambio respaldado por el contexto; anomalia ambiental.",
                    "delivery": "recorded_only",
                }
                if environmental
                else None
            ),
            "suspicion_effect": effect,
            "coverage": {
                "eligible_neighbors": len(edges),
                "observed_neighbors": covered_neighbors,
                "carried_neighbors": sum(bool(item.get("carried_forward")) for item in evidence),
                "ratio": self._redondear(coverage_ratio),
                "status": (
                    "sufficient"
                    if coverage_ratio >= self.config.runtime_coverage_threshold and edges
                    else "not_applicable" if not edges else "insufficient"
                ),
            },
            "evidence": evidence,
            "supports": [item for item in evidence if item["observation_status"] == "supports"],
            "contradictions": [
                item for item in evidence if item["observation_status"] == "contradicts"
            ],
            "unknowns": [
                item
                for item in evidence
                if item["observation_status"] in {"missing", "unknown", "observed_neutral"}
            ],
            "common_cause": common_cause,
            "adjustment_blocked_by_common_cause": False,
            "map_quality": map_quality,
            "diagnostic_guard": "context_is_associative_not_causal",
        }
        return result

    def _estadisticas_mapa(self, mapa: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {
            str(node["entity_id"]): {
                "observations": int(node.get("observations") or 0),
                "coverage_ratio": float(node.get("coverage_ratio") or 0.0),
                "center": node.get("center"),
                "scale": node.get("scale"),
                "state_probabilities": dict(node.get("state_probabilities") or {}),
            }
            for node in mapa.get("nodes", []) if node.get("entity_id")
        }

    def _indexar_observaciones(
        self, eventos: pd.DataFrame, catalogo: dict[str, dict[str, Any]]
    ) -> dict[str, pd.DataFrame]:
        """Indice de eventos originales, sin rellenar ni cambiar sus fechas."""
        groups = dict(tuple(eventos.groupby("entity_id", sort=False)))
        return {
            entity_id: groups.get(entity_id, eventos.iloc[:0]).set_index("timestamp_dt")
            for entity_id, meta in catalogo.items()
        }

    def _observacion_continua(
        self, events: pd.DataFrame, reference: pd.Timestamp, entity_id: str,
        *, window_seconds: int | None = None, max_age_seconds: float | None = None,
    ) -> tuple[float | None, dict[str, Any]]:
        """Mediana de la ventana parcial o ultimo valor anterior con caducidad.

        Nunca salta por encima de un reporte invalido para rescatar un valor
        antiguo. El TTL se mide en la fecha consultada (t - desfase).
        """
        max_age = (self.config.antiguedad_maxima(entity_id)
                   if max_age_seconds is None else float(max_age_seconds))
        seconds = window_seconds or self.config.window_seconds
        details = {
            "reference_timestamp": None if pd.isna(reference) else reference.isoformat(),
            "observed_timestamp": None,
            "observation_age_seconds": None,
            "max_age_seconds": max_age,
            "window_seconds": seconds,
            "observation_method": "unavailable",
            "carried_forward": False,
            "missing_reason": "no_previous_observation",
        }
        if events.empty or pd.isna(reference):
            return None, details
        end = int(events.index.searchsorted(reference, side="right"))
        if not end:
            return None, details
        latest = events.iloc[end - 1]
        observed_at = events.index[end - 1]
        elapsed = reference - observed_at
        age = elapsed.total_seconds()
        details.update({
            "observed_timestamp": observed_at.isoformat(),
            "observation_age_seconds": round(age, 6),
        })
        value = self._numeric_or_none(latest.get("value", latest.get("state")))
        if not bool(latest.get("is_valid_value")) or value is None:
            details["missing_reason"] = "invalid_latest_observation"
            return None, details
        if elapsed > pd.Timedelta(seconds=max_age):
            details["missing_reason"] = "expired_observation"
            return None, details
        bucket = reference.floor(f"{seconds}s")
        if observed_at < bucket:
            details.update({"observation_method": "last_known", "carried_forward": True})
        else:
            # No incluir en la mediana muestras caducadas ni anteriores a una
            # interrupcion explicita de disponibilidad dentro de la ventana.
            start = max(bucket, reference - pd.Timedelta(seconds=max_age))
            window = events.iloc[int(events.index.searchsorted(start)):end]
            invalid_positions = [i for i, valid in enumerate(window["is_valid_value"]) if not valid]
            if invalid_positions:
                window = window.iloc[invalid_positions[-1] + 1:]
            values = pd.to_numeric(window.get("value", window["state"]), errors="coerce")
            values = values.replace([math.inf, -math.inf], float("nan"))
            value = self._numeric_or_none(values.median())
            details["observation_method"] = "bucket_median"
        details["missing_reason"] = None if value is not None else "invalid_latest_observation"
        if value is not None:
            details["lag_error_seconds"] = round(age, 6)
        return value, details

    def _observacion_contextual(self, events, reference, entity_id, edge=None):
        edge = edge or {}
        if events.empty or events.iloc[-1].get("sensor_kind") == "continuous":
            return self._observacion_continua(
                events, reference, entity_id, window_seconds=edge.get("window_seconds"),
                max_age_seconds=edge.get("max_age_seconds"),
            )
        seconds = edge.get("window_seconds") or self.config.window_seconds
        details = {"reference_timestamp": None if pd.isna(reference) else reference.isoformat(),
                   "observed_timestamp": None, "max_age_seconds": None,
                   "window_seconds": seconds, "observation_method": "state_as_of",
                   "missing_reason": "no_previous_observation", "carried_forward": False}
        end = 0 if pd.isna(reference) else int(events.index.searchsorted(reference, side="right"))
        if not end:
            return None, details
        latest = events.iloc[end - 1]
        observed_at = events.index[end - 1]
        details.update(observed_timestamp=observed_at.isoformat(),
                       observation_age_seconds=(reference - observed_at).total_seconds())
        if not bool(latest.get("is_valid_value")):
            details["missing_reason"] = "invalid_latest_observation"
            return None, details
        value = latest.get("state")
        bucket = reference.floor(f"{seconds}s")
        details["carried_forward"] = observed_at < bucket
        if latest.get("sensor_kind") == "binary":
            start = int(events.index.searchsorted(bucket))
            window = events.iloc[start:end]
            # Descartar estados anteriores a una interrupcion explicita.
            invalid = [i for i, valid in enumerate(window["is_valid_value"]) if not valid]
            if invalid:
                window = window.iloc[invalid[-1] + 1:]
            value = self._numeric_or_none(latest.get("value"))
            if value is None:
                value = float(str(latest.get("state")).lower() in ESTADOS_ACTIVOS)
            if not window.empty:
                values = pd.to_numeric(window.get("value", window["state"]), errors="coerce")
                fallback = window["state"].str.lower().isin(ESTADOS_ACTIVOS).astype(float)
                value = float(values.where(values.notna(), fallback).max())
        details["missing_reason"] = None
        return value, details

    def _decidir_ruta(
        self,
        outcome: str,
        map_quality: dict[str, Any],
    ) -> dict[str, Any]:
        """Cierra los ambientales antes del tercer filtro, sin diagnosticar fallas.

        ``supported`` ya exige cobertura y el unico umbral de decision
        contextual. El puntaje ajustado se conserva como evidencia, no como
        compuerta. Una ruta al clasificador indica elegibilidad, no ejecucion.
        """
        if map_quality.get("stale") is not False:
            reason = "stale_or_unknown_context_map"
        elif outcome == "supported":
            reason = "context_supported"
        else:
            reason = outcome
        environmental = reason == "context_supported"
        return {
            "terminal": environmental,
            "destination": "environmental_alert" if environmental else "classifier",
            "send_to_classifier": not environmental,
            "reason": reason,
        }

    def resumen_enrutamiento(
        self, resultados: list[dict[str, Any]]
    ) -> dict[str, int]:
        ambientales = sum(
            item.get("routing", {}).get("terminal") is True
            and item.get("routing", {}).get("send_to_classifier") is False
            and item.get("contextual_class") == "environmental_anomaly"
            for item in resultados
        )
        return {
            "environmental_anomalies": ambientales,
            "requires_classification": len(resultados) - ambientales,
        }

    def _concurrencia_relacionada(self, sospecha, edges, state) -> list[str]:
        """Ultimo evento por entidad/patron; memoria acotada, sin consultar futuro.

        El llamador conserva state entre lotes del mismo replay y lo reinicia
        al reiniciar la evaluacion. Coincidencia no significa causa identificada.
        """
        timestamp = pd.to_datetime(sospecha.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            return []
        entity = str(sospecha.get("entity_id", ""))
        pattern = str(sospecha.get("suspected_pattern") or sospecha.get("pattern") or "unspecified")
        since = timestamp - pd.Timedelta(seconds=self.config.common_cause_window_seconds)
        for key in list(state):
            if state[key] < since:
                del state[key]
        state[(entity, pattern)] = timestamp
        related = {entity, *(str(edge["target"]) for edge in edges)}
        return sorted({other for (other, other_pattern), when in state.items()
                       if other in related and other_pattern == pattern and since <= when <= timestamp})

    def _bucket_timestamp(self, value: Any) -> pd.Timestamp | None:
        timestamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(timestamp):
            return None
        return timestamp.floor(f"{self.config.window_seconds}s")

    def _bucket_key(self, value: Any) -> str | None:
        bucket = self._bucket_timestamp(value)
        return None if bucket is None else bucket.isoformat()

    def _signal_value(
        self, signal: pd.Series | None, timestamp: pd.Timestamp | None
    ) -> Any:
        if signal is None or timestamp is None or timestamp not in signal.index:
            return None
        value = signal.loc[timestamp]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        return None if pd.isna(value) else value

    def _standardized_value(self, value: Any, stats: dict[str, Any]) -> float:
        numeric = self._numeric_or_none(value)
        center = self._numeric_or_none(stats.get("center"))
        scale = self._numeric_or_none(stats.get("scale"))
        if numeric is None or center is None or scale is None or scale <= 0:
            return 0.0
        return (numeric - center) / scale

    def _expected_text(self, edge: dict[str, Any], target_z: float) -> str:
        if edge.get("target_state") is not None:
            return f"estado {edge['target_state']} con desfase {edge.get('lag_seconds', 0)} s"
        target_direction = "sube" if target_z >= 0 else "baja"
        relation = "mismo sentido" if edge.get("expected_sign") == "positive" else "sentido opuesto"
        return f"{relation} cuando la entidad evaluada {target_direction}"

    def _expected_level_text(self, edge: dict[str, Any]) -> str:
        return f"nivel condicional segun modelo calibrado con desfase {edge.get('lag_seconds', 0)} s"

    def _comparar_mapas(
        self,
        previous: dict[str, Any] | None,
        current_edges: list[dict[str, Any]],
        current_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not previous:
            return {
                "association_mean_delta": None,
                "association_changed_fraction": None,
                "coverage_mean_delta": None,
                "association_drift_detected": False,
                "coverage_drift_detected": False,
            }
        old_edges = {
            str(edge.get("edge_id")): float(edge.get("association_signed", 0.0))
            for edge in previous.get("edges", [])
            if edge.get("edge_id")
        }
        deltas = [
            abs(float(edge.get("association_signed", 0.0)) - old_edges[edge["edge_id"]])
            for edge in current_edges
            if edge.get("edge_id") in old_edges
        ]
        mean_delta = sum(deltas) / len(deltas) if deltas else 0.0
        changed_fraction = (
            sum(delta >= self.config.association_drift_threshold for delta in deltas)
            / len(deltas)
            if deltas
            else 0.0
        )
        old_nodes = {
            str(node.get("entity_id")): float(node.get("coverage_ratio", 0.0))
            for node in previous.get("nodes", [])
            if node.get("entity_id")
        }
        coverage_deltas = [
            abs(float(node.get("coverage_ratio", 0.0)) - old_nodes[node["entity_id"]])
            for node in current_nodes
            if node.get("entity_id") in old_nodes
        ]
        coverage_mean_delta = (
            sum(coverage_deltas) / len(coverage_deltas) if coverage_deltas else 0.0
        )
        return {
            "association_mean_delta": self._redondear(mean_delta),
            "association_changed_fraction": self._redondear(changed_fraction),
            "coverage_mean_delta": self._redondear(coverage_mean_delta),
            "association_drift_detected": bool(
                mean_delta >= self.config.association_drift_threshold
                or changed_fraction >= 0.25
            ),
            "coverage_drift_detected": bool(
                coverage_mean_delta >= self.config.coverage_drift_threshold
            ),
        }

    def _cruzo_estacion(self, old_end: Any, new_end: Any) -> bool:
        old = pd.to_datetime(old_end, errors="coerce", utc=True)
        new = pd.to_datetime(new_end, errors="coerce", utc=True)
        if pd.isna(old) or pd.isna(new) or new <= old:
            return False
        return ((old.month - 1) // 3) != ((new.month - 1) // 3) or old.year != new.year

    def _timestamp_o_none(self, value: Any) -> str | None:
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()

    def _numeric_or_none(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if math.isfinite(numeric) else None

    def _serializable_value(self, value: Any) -> Any:
        numeric = self._numeric_or_none(value)
        return numeric if numeric is not None else str(value)

    def _redondear(self, value: Any, digits: int = 6) -> float:
        numeric = self._numeric_or_none(value)
        return 0.0 if numeric is None else round(numeric, digits)

    def _redondear_o_none(self, value: Any, digits: int = 6) -> float | None:
        numeric = self._numeric_or_none(value)
        return None if numeric is None else round(numeric, digits)
