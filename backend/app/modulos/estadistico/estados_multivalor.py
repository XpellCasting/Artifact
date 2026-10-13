from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


INDICADORES_MULTIESTADO = {
    "state_change_count": "transition_burst",
    "state_probability": "rare_state",
    "transition_probability": "rare_transition",
    "state_duration": "state_persistence",
    "report_gap": "missing_gap",
}

DETALLE_INDICADOR = {
    "state_change_count": "Cambios multivalor C_i(W)",
    "state_probability": "Rareza del estado P(s)",
    "transition_probability": "Rareza de la transicion P(s_j|s_i)",
    "state_duration": "Duracion del estado D_s(t)",
    "report_gap": "Brecha temporal de reporte M_i(t)",
}


class FiltroEstadoMultivalorEstadistico:
    """Analiza transiciones categoricas observadas en el historial real.

    El filtro no asigna un orden artificial a los estados. Usa frecuencia de
    estados y transiciones, cambios dentro de una ventana y duraciones. Las
    series de alta cardinalidad (por ejemplo, un timestamp diferente por fila)
    se declaran no elegibles para evitar tratarlas como estados discretos.
    """

    score_version = "statistical-multistate-score-v1.0.0"
    score_kind = "statistical_evidence_strength_not_probability"
    window_seconds = 60 * 60
    min_history_events = 24
    min_unique_states = 2
    max_unique_states = 32
    high_cardinality_min_states = 8
    max_unique_ratio = 0.50
    rarity_state_floor = 0.02
    rarity_transition_floor = 0.05
    min_transition_support = 10
    min_duration_alert_seconds = 6 * 60 * 60
    min_report_gap_seconds = 6 * 60 * 60
    tau_indicator = 3.5
    component_minimum = 0.60
    suppression_seconds = 60 * 60

    def __init__(self) -> None:
        self.last_eligibility: dict[str, Any] = self._empty_eligibility()

    def especificacion_puntaje(self) -> dict[str, Any]:
        return {
            "version": self.score_version,
            "kind": self.score_kind,
            "uses_simulated_dataset": False,
            "uses_observed_states_only": True,
            "window_seconds": self.window_seconds,
            "min_history_events": self.min_history_events,
            "min_unique_states": self.min_unique_states,
            "max_unique_states": self.max_unique_states,
            "max_unique_ratio": self.max_unique_ratio,
            "indicators": sorted(INDICADORES_MULTIESTADO),
        }

    def analizar(
        self,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        if eventos.empty or "sensor_kind" not in eventos.columns:
            self.last_eligibility = self._empty_eligibility()
            return []
        trabajo = eventos[eventos["sensor_kind"].eq("multistate")].copy()
        if trabajo.empty:
            self.last_eligibility = self._empty_eligibility()
            return []

        resultados: list[dict[str, Any]] = []
        diagnosticos: list[dict[str, Any]] = []
        for entity_id, grupo in trabajo.groupby("entity_id", sort=True):
            serie = self._serie_valida(grupo)
            diagnostico = self._diagnosticar(str(entity_id), grupo, serie)
            diagnosticos.append(diagnostico)
            if diagnostico["status"] != "eligible":
                continue
            resultados.extend(
                self._analizar_entidad(
                    entity_id=str(entity_id),
                    serie=serie,
                    suspicion_threshold=suspicion_threshold,
                )
            )

        self.last_eligibility = self._resumir_diagnosticos(diagnosticos)
        return self._agrupar_resultados(resultados)

    def _serie_valida(self, grupo: pd.DataFrame) -> pd.DataFrame:
        serie = grupo.sort_values("timestamp_dt").copy()
        if "is_valid_value" in serie.columns:
            serie = serie[serie["is_valid_value"].astype(bool)].copy()
        serie = serie[serie["timestamp_dt"].notna()].copy()
        serie["state"] = serie["state"].astype(str).str.strip()
        serie = serie[serie["state"].ne("")].copy()
        serie["state_key"] = serie["state"].str.casefold()
        return serie

    def _diagnosticar(
        self,
        entity_id: str,
        grupo: pd.DataFrame,
        serie: pd.DataFrame,
    ) -> dict[str, Any]:
        valid_rows = int(len(serie))
        unique_states = int(serie["state_key"].nunique()) if valid_rows else 0
        unique_ratio = float(unique_states / valid_rows) if valid_rows else 0.0
        if valid_rows == 0:
            status = "invalid_only"
        elif valid_rows < self.min_history_events:
            status = "insufficient_history"
        elif unique_states < self.min_unique_states:
            status = "insufficient_states"
        elif unique_states > self.max_unique_states or (
            unique_states >= self.high_cardinality_min_states
            and unique_ratio > self.max_unique_ratio
        ):
            status = "high_cardinality"
        else:
            status = "eligible"
        return {
            "entity_id": entity_id,
            "status": status,
            "rows": int(len(grupo)),
            "valid_rows": valid_rows,
            "unique_states": unique_states,
            "unique_ratio": round(unique_ratio, 6),
            "states": sorted(serie["state"].drop_duplicates().tolist())[: self.max_unique_states],
        }

    def _resumir_diagnosticos(self, diagnosticos: list[dict[str, Any]]) -> dict[str, Any]:
        counts = Counter(item["status"] for item in diagnosticos)
        return {
            "entities": len(diagnosticos),
            "eligible_entities": int(counts.get("eligible", 0)),
            "by_status": dict(sorted(counts.items())),
            "details": diagnosticos,
        }

    def _empty_eligibility(self) -> dict[str, Any]:
        return {
            "entities": 0,
            "eligible_entities": 0,
            "by_status": {},
            "details": [],
        }

    def _analizar_entidad(
        self,
        entity_id: str,
        serie: pd.DataFrame,
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        features = self._extraer_features(serie)
        if features.empty:
            return []
        base = self._construir_base(features)
        resultados: list[dict[str, Any]] = []
        for fila in features.to_dict("records"):
            raw, componentes = self._puntuar(fila, base)
            indicadores = [
                indicador
                for indicador, componente in componentes.items()
                if componente >= self.component_minimum
            ]
            evidencia = max(componentes.values(), default=0.0)
            if not indicadores or evidencia < float(suspicion_threshold):
                continue
            principal = max(indicadores, key=lambda item: componentes[item])
            patron = INDICADORES_MULTIESTADO[principal]
            resultados.append(
                {
                    "entity_id": entity_id,
                    "source_raw_index": fila.get("_raw_index"),
                    "timestamp": str(fila["timestamp"]),
                    "window_start": str(fila["window_start"]),
                    "window_end": str(fila["timestamp"]),
                    "sensor_kind": "multistate",
                    "tipo_interno": "estado_multivalor",
                    "observed_state": str(fila["state"]),
                    "value": None,
                    "raw_score": round(float(raw[principal]), 6),
                    "evidence_score": round(float(evidencia), 6),
                    "suspicion_score": round(float(evidencia), 6),
                    "is_suspicious": True,
                    "pattern": patron,
                    "suspected_pattern": patron,
                    "pattern_detail": ", ".join(DETALLE_INDICADOR[item] for item in indicadores),
                    "filter": "statistical_temporal",
                    "passed_to_contextual_filter": True,
                    "triggered_indicators": indicadores,
                    "reading": {
                        "state": str(fila["state"]),
                        "previous_state": fila.get("previous_state"),
                        "state_changes": int(fila["state_changes"]),
                        "state_duration_seconds": round(float(fila["state_duration_seconds"]), 3),
                        "report_gap_seconds": round(float(fila["report_gap_seconds"]), 3),
                        "state_probability": round(float(fila["state_probability"]), 6),
                        "transition_probability": round(float(fila["transition_probability"]), 6),
                        "window_seconds": self.window_seconds,
                        "components": {key: round(float(value), 6) for key, value in componentes.items()},
                        "raw_components": {key: round(float(value), 6) for key, value in raw.items()},
                    },
                    "baseline_summary": self._base_para_salida(base),
                    "ground_truth_label": None,
                    "severity": self._severidad(evidencia),
                    "score_version": self.score_version,
                    "score_kind": self.score_kind,
                }
            )
        return resultados

    def _extraer_features(self, serie: pd.DataFrame) -> pd.DataFrame:
        trabajo = serie.reset_index(drop=True).copy()
        trabajo["previous_state_key"] = trabajo["state_key"].shift(1)
        trabajo["previous_state"] = trabajo["state"].shift(1)
        trabajo["state_change_event"] = (
            trabajo["previous_state_key"].notna()
            & trabajo["state_key"].ne(trabajo["previous_state_key"])
        ).astype(int)
        trabajo["report_gap_seconds"] = (
            trabajo["timestamp_dt"].diff().dt.total_seconds().fillna(0.0).clip(lower=0.0)
        )

        tiempos = self._timestamps_a_segundos(trabajo["timestamp_dt"])
        cambios = trabajo["state_change_event"].to_numpy(dtype="int64")
        acumulado = np.concatenate([[0], np.cumsum(cambios)])
        duraciones: list[float] = []
        inicio_estado = tiempos[0] if len(tiempos) else 0.0
        cambios_ventana: list[int] = []
        for indice, tiempo_actual in enumerate(tiempos):
            if indice == 0 or int(cambios[indice]) == 1:
                inicio_estado = tiempo_actual
            duraciones.append(max(0.0, tiempo_actual - inicio_estado))
            izquierda = int(np.searchsorted(tiempos, tiempo_actual - self.window_seconds, side="right"))
            cambios_ventana.append(int(acumulado[indice + 1] - acumulado[izquierda]))
        trabajo["state_duration_seconds"] = duraciones
        trabajo["state_changes"] = cambios_ventana

        state_counts = trabajo["state_key"].value_counts().to_dict()
        total = max(len(trabajo), 1)
        trabajo["state_probability"] = trabajo["state_key"].map(
            lambda state: max(0.0, (state_counts.get(state, 0) - 1) / max(total - 1, 1))
        )

        transition_rows = trabajo[trabajo["state_change_event"].eq(1)]
        transition_counts = Counter(
            zip(transition_rows["previous_state_key"], transition_rows["state_key"])
        )
        source_counts = Counter(transition_rows["previous_state_key"])
        probabilidades = []
        soportes = []
        for fila in trabajo.to_dict("records"):
            source = fila.get("previous_state_key")
            target = fila.get("state_key")
            if not fila.get("state_change_event") or source is None:
                probabilidades.append(1.0)
                soportes.append(0)
                continue
            source_support = int(source_counts.get(source, 0))
            transition_support = int(transition_counts.get((source, target), 0))
            probabilidades.append(
                max(0.0, (transition_support - 1) / max(source_support - 1, 1))
            )
            soportes.append(source_support)
        trabajo["transition_probability"] = probabilidades
        trabajo["transition_source_support"] = soportes
        trabajo["window_start"] = (
            trabajo["timestamp_dt"] - pd.Timedelta(seconds=self.window_seconds)
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        return trabajo

    def _construir_base(self, features: pd.DataFrame) -> dict[str, Any]:
        cambios = self._resumen_robusto(features["state_changes"], piso_mad=1.0)
        duraciones = self._resumen_robusto(
            features.loc[features["state_duration_seconds"].gt(0), "state_duration_seconds"],
            piso_mad=60.0,
        )
        brechas = self._resumen_robusto(
            features.loc[features["report_gap_seconds"].gt(0), "report_gap_seconds"],
            piso_mad=60.0,
        )
        return {
            "rows": int(len(features)),
            "states": features["state_key"].value_counts().to_dict(),
            "unique_states": int(features["state_key"].nunique()),
            "state_changes": cambios,
            "state_duration": duraciones,
            "report_gap": brechas,
            "state_change_alert": cambios["median"] + 6.0 * cambios["scaled_mad"],
            "state_duration_alert_seconds": max(
                self.min_duration_alert_seconds,
                duraciones["median"] + 8.0 * duraciones["scaled_mad"],
            ),
            "report_gap_alert_seconds": max(
                self.min_report_gap_seconds,
                brechas["median"] + 8.0 * brechas["scaled_mad"],
            ),
        }

    def _puntuar(
        self,
        fila: dict[str, Any],
        base: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float]]:
        raw_change, score_change = self._score_alto(
            float(fila["state_changes"]),
            base["state_changes"],
            float(base["state_change_alert"]),
        )
        raw_duration, score_duration = self._score_alto(
            float(fila["state_duration_seconds"]),
            base["state_duration"],
            float(base["state_duration_alert_seconds"]),
        )
        raw_gap, score_gap = self._score_alto(
            float(fila["report_gap_seconds"]),
            base["report_gap"],
            float(base["report_gap_alert_seconds"]),
        )
        state_probability = float(fila["state_probability"])
        score_state = self._score_rareza(state_probability, self.rarity_state_floor)
        transition_probability = float(fila["transition_probability"])
        score_transition = 0.0
        if int(fila["transition_source_support"]) >= self.min_transition_support:
            score_transition = self._score_rareza(
                transition_probability,
                self.rarity_transition_floor,
            )
        return (
            {
                "state_change_count": raw_change,
                "state_probability": -math.log10(max(state_probability, 1e-9)),
                "transition_probability": -math.log10(max(transition_probability, 1e-9)),
                "state_duration": raw_duration,
                "report_gap": raw_gap,
            },
            {
                "state_change_count": score_change,
                "state_probability": score_state,
                "transition_probability": score_transition,
                "state_duration": score_duration,
                "report_gap": score_gap,
            },
        )

    def _score_alto(
        self,
        valor: float,
        resumen: dict[str, float],
        alerta: float,
    ) -> tuple[float, float]:
        z_value = max(
            0.0,
            (float(valor) - resumen["median"]) / (resumen["scaled_mad"] + 1e-9),
        )
        if float(valor) < float(alerta):
            return z_value, 0.0
        return z_value, max(0.0, min(1.0, z_value / self.tau_indicator))

    def _score_rareza(self, probability: float, floor: float) -> float:
        if probability >= floor:
            return 0.0
        return max(0.0, min(1.0, (floor - probability) / floor))

    def _resumen_robusto(self, valores: pd.Series, piso_mad: float) -> dict[str, float]:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        if limpio.empty:
            limpio = pd.Series([0.0], dtype="float64")
        mediana = float(limpio.median())
        mad = max(float((limpio - mediana).abs().median()), piso_mad)
        return {
            "median": mediana,
            "mad": mad,
            "scaled_mad": 1.4826 * mad,
            "p95": float(limpio.quantile(0.95)),
        }

    def _base_para_salida(self, base: dict[str, Any]) -> dict[str, Any]:
        return {
            "rows": int(base["rows"]),
            "unique_states": int(base["unique_states"]),
            "state_counts": base["states"],
            "window_seconds": self.window_seconds,
            "state_changes_median": round(base["state_changes"]["median"], 6),
            "state_change_alert": round(base["state_change_alert"], 6),
            "state_duration_alert_seconds": round(base["state_duration_alert_seconds"], 3),
            "report_gap_alert_seconds": round(base["report_gap_alert_seconds"], 3),
            "rarity_state_floor": self.rarity_state_floor,
            "rarity_transition_floor": self.rarity_transition_floor,
        }

    def _agrupar_resultados(self, resultados: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not resultados:
            return []
        ordenados = sorted(
            resultados,
            key=lambda item: (item["entity_id"], item["pattern"], item["timestamp"]),
        )
        grupos: list[list[dict[str, Any]]] = []
        for resultado in ordenados:
            ultimo_grupo = grupos[-1] if grupos else None
            if ultimo_grupo:
                ultimo = ultimo_grupo[-1]
                actual_ts = pd.to_datetime(resultado["timestamp"], errors="coerce", utc=True)
                ultimo_ts = pd.to_datetime(ultimo["timestamp"], errors="coerce", utc=True)
                misma_serie = (
                    resultado["entity_id"] == ultimo["entity_id"]
                    and resultado["pattern"] == ultimo["pattern"]
                )
                cerca = pd.notna(actual_ts) and pd.notna(ultimo_ts) and (
                    actual_ts - ultimo_ts
                ).total_seconds() <= self.suppression_seconds
                if misma_serie and cerca:
                    ultimo_grupo.append(resultado)
                    continue
            grupos.append([resultado])
        representativos = []
        for grupo in grupos:
            mejor = max(grupo, key=lambda item: item["suspicion_score"])
            if len(grupo) > 1:
                mejor = dict(mejor)
                mejor["suppressed_events"] = len(grupo) - 1
            representativos.append(mejor)
        return sorted(representativos, key=lambda item: item["suspicion_score"], reverse=True)

    def _timestamps_a_segundos(self, timestamps: pd.Series) -> np.ndarray:
        epoch = pd.Timestamp("1970-01-01T00:00:00Z")
        return (timestamps - epoch).dt.total_seconds().to_numpy(dtype="float64")

    def _severidad(self, suspicion_score: float) -> str:
        if suspicion_score >= 0.85:
            return "alta"
        if suspicion_score >= 0.70:
            return "media"
        return "baja"
