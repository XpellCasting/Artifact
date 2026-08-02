from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
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

    def como_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["lag_candidates_seconds"] = [
            lag * self.window_seconds for lag in range(self.max_lag_windows + 1)
        ]
        payload["acceptance_rule"] = (
            "n_observations >= min_observations AND "
            "coverage_ratio >= min_pair_coverage AND "
            "|association_signed| >= association_threshold AND "
            "stable_slices >= min_stable_slices AND "
            "sign_consistency >= min_sign_consistency; despues se conserva top_k por entidad"
        )
        return payload


class FiltroContextualEspacioTemporal:
    """Construye el mapa histórico y contrasta sospechas en línea.

    El mapa se aprende solo desde el histórico normalizado. La inferencia usa
    relaciones dirigidas hacia contexto presente o pasado; nunca consulta
    observaciones futuras para explicar una sospecha.
    """

    MAP_VERSION = "contextual-spatiotemporal-v1"

    def __init__(self, config: ConfiguracionContextual | None = None) -> None:
        self.config = config or ConfiguracionContextual()

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
        signals, availability, stats, full_index = self._construir_senales(trabajo, catalogo)
        entity_ids = sorted(signals)
        covariance = self._matriz_covarianza(signals, catalogo, entity_ids)

        edges: list[dict[str, Any]] = []
        for source in entity_ids:
            for target in entity_ids:
                if source == target:
                    continue
                relation = self._mejor_relacion(
                    source,
                    target,
                    signals[source],
                    signals[target],
                    catalogo[source]["sensor_kind"],
                    catalogo[target]["sensor_kind"],
                    len(full_index),
                )
                relation["raw_covariance"] = self._valor_matriz(covariance, source, target)
                edges.append(relation)

        self._aplicar_top_k(edges)
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
                }
            )

        matrices = self._construir_matrices(entity_ids, covariance, edges)
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
        inicio = perf_counter()
        trabajo = self._preparar_eventos(eventos)
        catalogo = self._catalogo(entidades, trabajo)
        signals, _, stats, _ = self._construir_senales(trabajo, catalogo)
        edges_by_source: dict[str, list[dict[str, Any]]] = {}
        for edge in mapa.get("edges", []):
            if edge.get("accepted"):
                edges_by_source.setdefault(str(edge["source"]), []).append(edge)

        candidatas = [
            dict(item)
            for item in sospechas
            if item.get("is_suspicious") or item.get("passed_to_contextual_filter")
        ]
        concurrent_by_window = self._concurrencia(candidatas)
        resultados = []
        for sospecha in candidatas:
            resultados.append(
                self._evaluar_sospecha(
                    sospecha=sospecha,
                    signals=signals,
                    stats=stats,
                    edges=edges_by_source.get(str(sospecha.get("entity_id")), []),
                    map_quality={
                        "version": mapa.get("map_version"),
                        **mapa.get("refresh", {}),
                    },
                    concurrent_entities=concurrent_by_window.get(
                        self._bucket_key(sospecha.get("timestamp")), []
                    ),
                )
            )

        outcomes: dict[str, int] = {}
        for item in resultados:
            outcome = str(item["context_outcome"])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        return {
            "results": resultados,
            "summary": {
                "evaluated": len(resultados),
                "by_outcome": outcomes,
                "supported": outcomes.get("supported", 0),
                "contradicted": outcomes.get("contradicted", 0),
                "unexplained": outcomes.get("unexplained", 0),
                "insufficient_coverage": outcomes.get("insufficient_coverage", 0),
                "possible_common_cause": sum(
                    bool(item.get("common_cause", {}).get("candidate")) for item in resultados
                ),
                "elapsed_ms": self._redondear((perf_counter() - inicio) * 1000.0, 3),
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
                trabajo.get("timestamp"), errors="coerce", utc=True
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
        return trabajo.sort_values(["entity_id", "timestamp_dt"]).reset_index(drop=True)

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
    ) -> tuple[
        dict[str, pd.Series],
        dict[str, pd.Series],
        dict[str, dict[str, Any]],
        pd.DatetimeIndex,
    ]:
        if eventos.empty or not catalogo:
            return {}, {}, {}, pd.DatetimeIndex([])
        freq = f"{self.config.window_seconds}s"
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

    def _matriz_covarianza(
        self,
        signals: dict[str, pd.Series],
        catalogo: dict[str, dict[str, Any]],
        entity_ids: list[str],
    ) -> pd.DataFrame:
        continuous = [
            entity_id
            for entity_id in entity_ids
            if catalogo[entity_id]["sensor_kind"] == "continuous"
        ]
        if not continuous:
            return pd.DataFrame()
        frame = pd.concat([pd.to_numeric(signals[entity_id], errors="coerce") for entity_id in continuous], axis=1)
        frame.columns = continuous
        return frame.cov(min_periods=self.config.min_observations)

    def _mejor_relacion(
        self,
        source: str,
        target: str,
        source_signal: pd.Series,
        target_signal: pd.Series,
        source_kind: str,
        target_kind: str,
        total_windows: int,
    ) -> dict[str, Any]:
        candidates = []
        for lag in range(self.config.max_lag_windows + 1):
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
            candidates.append(
                {
                    **encoded,
                    "association": association,
                    "lag_windows": lag,
                    "n_observations": int(len(pair)),
                    "coverage_ratio": float(len(pair) / total_windows) if total_windows else 0.0,
                }
            )
        best = max(
            candidates,
            key=lambda item: (
                -1.0 if item["association"] is None else abs(item["association"]),
                item["n_observations"],
                -item["lag_windows"],
            ),
        )
        association = best["association"] or 0.0
        stability = self._estabilidad(best["pair"], association)
        rejection_reasons = []
        if best["n_observations"] < self.config.min_observations:
            rejection_reasons.append("insufficient_observations")
        if best["coverage_ratio"] < self.config.min_pair_coverage:
            rejection_reasons.append("insufficient_pair_coverage")
        if abs(association) < self.config.association_threshold:
            rejection_reasons.append("weak_association")
        if stability["stable_slices"] < self.config.min_stable_slices:
            rejection_reasons.append("temporal_instability")
        if stability["sign_consistency"] < self.config.min_sign_consistency:
            rejection_reasons.append("sign_instability")
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
            "lag_seconds": int(best["lag_windows"] * self.config.window_seconds),
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
            "raw_covariance": None,
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
        covariance: pd.DataFrame,
        edges: list[dict[str, Any]],
    ) -> dict[str, Any]:
        edge_lookup = {(edge["source"], edge["target"]): edge for edge in edges}
        covariance_values = []
        association_values = []
        accepted_values = []
        for source in entity_ids:
            covariance_row = []
            association_row = []
            accepted_row = []
            for target in entity_ids:
                if source == target:
                    covariance_row.append(self._valor_matriz(covariance, source, target))
                    association_row.append(1.0)
                    accepted_row.append(True)
                    continue
                edge = edge_lookup.get((source, target))
                covariance_row.append(self._valor_matriz(covariance, source, target))
                association_row.append(edge.get("association_signed") if edge else None)
                accepted_row.append(bool(edge and edge.get("accepted")))
            covariance_values.append(covariance_row)
            association_values.append(association_row)
            accepted_values.append(accepted_row)
        return {
            "entities": entity_ids,
            "covariance": covariance_values,
            "association": association_values,
            "accepted": accepted_values,
        }

    def _evaluar_sospecha(
        self,
        *,
        sospecha: dict[str, Any],
        signals: dict[str, pd.Series],
        stats: dict[str, dict[str, Any]],
        edges: list[dict[str, Any]],
        map_quality: dict[str, Any],
        concurrent_entities: list[str],
    ) -> dict[str, Any]:
        entity_id = str(sospecha.get("entity_id", ""))
        bucket = self._bucket_timestamp(sospecha.get("timestamp"))
        target_signal = signals.get(entity_id)
        target_stats = stats.get(entity_id, {})
        target_value = self._signal_value(target_signal, bucket)
        if target_value is None:
            target_value = self._numeric_or_none(
                sospecha.get("value", (sospecha.get("reading") or {}).get("value"))
            )
        target_z = self._standardized_value(target_value, target_stats)
        if abs(target_z) < 0.05:
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
            if context_value is None:
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
                        "expected_direction_or_state": self._expected_text(edge, target_z),
                        "c_j_raw": None,
                        "c_j_normalized": None,
                        "weighted_contribution": 0.0,
                        "observation_status": "missing",
                        "temporal_match": False,
                        "n_observations": edge.get("n_observations"),
                        "coverage_ratio": edge.get("coverage_ratio"),
                        "stability_score": edge.get("stability_score"),
                    }
                )
                continue

            observed_weight += weight
            context_stats = stats.get(context_id, {})
            if edge.get("target_state") is not None:
                state = str(edge["target_state"])
                context_component = 1.0 if str(context_value) == state else 0.0
                probability = float(context_stats.get("state_probabilities", {}).get(state, 0.0))
                scale = max(math.sqrt(max(probability * (1.0 - probability), 0.0)), 1e-6)
                context_z = (context_component - probability) / scale
            else:
                context_z = self._standardized_value(context_value, context_stats)

            target_direction = 1.0 if target_z >= 0 else -1.0
            relation_direction = 1.0 if float(edge.get("association_signed", 0.0)) >= 0 else -1.0
            c_raw = relation_direction * target_direction * context_z
            c_normalized = max(-1.0, min(1.0, c_raw / 2.0))
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
                    "expected_direction_or_state": self._expected_text(edge, target_z),
                    "c_j_raw": self._redondear(c_raw),
                    "c_j_normalized": self._redondear(c_normalized),
                    "weighted_contribution": self._redondear(contribution),
                    "observation_status": status,
                    "temporal_match": True,
                    "n_observations": edge.get("n_observations"),
                    "coverage_ratio": edge.get("coverage_ratio"),
                    "stability_score": edge.get("stability_score"),
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
        if (
            not edges
            or coverage_ratio < self.config.runtime_coverage_threshold
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

        # El mismo patrón coherente puede provenir del ambiente o de una falla
        # compartida. Sin telemetría independiente no se permite que el
        # contexto modifique por sí solo la sospecha estadística.
        if common_candidate:
            adjusted = statistical_score
            effect = "maintain"
        common_cause = {
            "candidate": common_candidate,
            "status": (
                "unidentifiable_without_independent_evidence"
                if common_candidate
                else "not_indicated"
            ),
            "concurrent_entities": concurrent_entities,
            "independent_evidence_available": False,
            "independent_domains_required": ["network", "power", "integration"],
            "interpretation": (
                "El historico no permite separar una causa comun de una anomalia ambiental."
                if common_candidate
                else "No se observo un patron multientidad suficiente; esto no descarta una causa comun."
            ),
        }
        covered_neighbors = sum(
            item["observation_status"] != "missing" for item in evidence
        )
        result = {
            "event_id": f"{entity_id}@{sospecha.get('timestamp')}",
            "entity_id": entity_id,
            "timestamp": sospecha.get("timestamp"),
            "sensor_kind": sospecha.get("sensor_kind"),
            "suspected_pattern": sospecha.get("suspected_pattern", sospecha.get("pattern")),
            "statistical_score": self._redondear(statistical_score),
            "context_score": self._redondear(context_score),
            "coherence_score": self._redondear((context_score + 1.0) / 2.0),
            "support_score": self._redondear(support_score),
            "contradiction_score": self._redondear(contradiction_score),
            "adjusted_suspicion_score": self._redondear(max(0.0, min(1.0, adjusted))),
            "context_outcome": outcome,
            "suspicion_effect": effect,
            "coverage": {
                "eligible_neighbors": len(edges),
                "observed_neighbors": covered_neighbors,
                "ratio": self._redondear(coverage_ratio),
                "status": (
                    "sufficient"
                    if coverage_ratio >= self.config.runtime_coverage_threshold and edges
                    else "insufficient"
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
                if item["observation_status"] in {"missing", "observed_neutral"}
            ],
            "common_cause": common_cause,
            "adjustment_blocked_by_common_cause": common_candidate,
            "map_quality": map_quality,
            "diagnostic_guard": (
                "do_not_label_environmental_without_independent_evidence"
                if common_candidate
                else "context_is_associative_not_causal"
            ),
        }
        return result

    def _concurrencia(
        self, sospechas: list[dict[str, Any]]
    ) -> dict[str | None, list[str]]:
        groups: dict[str | None, set[str]] = {}
        for item in sospechas:
            key = self._bucket_key(item.get("timestamp"))
            groups.setdefault(key, set()).add(str(item.get("entity_id", "")))
        return {key: sorted(entity_ids) for key, entity_ids in groups.items()}

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

    def _valor_matriz(
        self, matrix: pd.DataFrame, source: str, target: str
    ) -> float | None:
        if matrix.empty or source not in matrix.index or target not in matrix.columns:
            return None
        return self._redondear_o_none(matrix.loc[source, target])

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
