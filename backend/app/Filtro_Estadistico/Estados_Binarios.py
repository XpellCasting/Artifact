from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


INDICADORES_BINARIOS = {
    "activation_count": "activation_burst",
    "state_change_count": "activation_burst",
    "active_time_ratio": "state_persistence",
    "time_since_last_activation": "absence_gap",
    "report_gap": "missing_gap",
    "time_since_last_change": "state_persistence",
    "z_duration_score": "state_persistence",
}

DETALLE_INDICADOR = {
    "activation_count": "Activaciones A_i(W)",
    "state_change_count": "Cambios de estado C_i(W)",
    "active_time_ratio": "Proporcion activa P_on,i(W)",
    "time_since_last_activation": "Tiempo desde ultima activacion G_i(t)",
    "report_gap": "Brecha temporal de reporte M_i(t)",
    "time_since_last_change": "Tiempo desde ultimo cambio TLC_i(t)",
    "z_duration_score": "Z-score de duracion Z_D_i",
}


@dataclass(frozen=True)
class PerfilBinario:
    """Parametros del filtro segun la naturaleza del sensor de estado."""

    window_seconds: int
    min_report_gap_seconds: float
    min_absence_gap_seconds: float
    ratio_floor: float


class FiltroEstadoBinarioEstadistico:
    """Filtro estadistico-temporal para entidades de estado binario.

    Usa las variables de la tesis: A_i(W), C_i(W), P_on,i(W), G_i(t) y M_i(t).
    El historico real se evalua por ventanas hacia atras, sin simulacion.
    """

    tau_probabilidad = 0.10
    tau_indicador = 3.5
    zona_muerta_z = 2.5
    componente_minimo = 0.60
    supresion_segundos = 60 * 60
    min_history_events = 24
    recent_window_days = 14

    def analizar(self, eventos: pd.DataFrame, entidades: pd.DataFrame, suspicion_threshold: float) -> list[dict[str, Any]]:
        if eventos.empty or "sensor_kind" not in eventos.columns:
            return []
        trabajo = eventos[eventos["sensor_kind"].eq("binary")].copy()
        if trabajo.empty:
            return []

        lookup_entidades = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
        resultados: list[dict[str, Any]] = []
        for entity_id, grupo in trabajo.groupby("entity_id", sort=True):
            entidad = lookup_entidades.get(str(entity_id), {})
            resultados.extend(
                self._analizar_entidad(
                    entity_id=str(entity_id),
                    grupo=grupo,
                    entidad=entidad,
                    suspicion_threshold=suspicion_threshold,
                )
            )
        return self._agrupar_resultados(resultados)

    def _analizar_entidad(
        self,
        entity_id: str,
        grupo: pd.DataFrame,
        entidad: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        serie = grupo.sort_values("timestamp_dt").copy()
        if "is_valid_value" in serie.columns:
            serie = serie[serie["is_valid_value"].astype(bool)].copy()
        serie["value"] = pd.to_numeric(serie["value"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        serie = serie[serie["timestamp_dt"].notna()].copy()
        if len(serie) < self.min_history_events:
            return []

        perfil = self._perfil_binario(entity_id, str(entidad.get("device_class", "")))
        features = self._extraer_features(serie, perfil)
        if features.empty:
            return []
        base = self._construir_base_normalidad(features, perfil)
        raw_threshold = self._probabilidad_a_evidencia(suspicion_threshold)
        resultados: list[dict[str, Any]] = []

        for fila in features.to_dict("records"):
            raw_componentes, componentes, evidence_score = self._puntuar_doble_horizonte(fila, base)
            suspicion_score = self._score_a_probabilidad(evidence_score)
            indicadores = self._indicadores_activados(componentes)

            if indicadores and evidence_score >= raw_threshold and suspicion_score >= suspicion_threshold:
                indicador_principal = self._indicador_principal(componentes, indicadores)
                patron = INDICADORES_BINARIOS[indicador_principal]
                resultados.append(
                    {
                        "entity_id": entity_id,
                        "timestamp": str(fila["timestamp"]),
                        "window_start": str(fila["window_start"]),
                        "window_end": str(fila["timestamp"]),
                        "sensor_kind": "binary",
                        "tipo_interno": "estado_binario",
                        "observed_state": str(fila["state"]),
                        "value": float(fila["value"]),
                        "raw_score": self._raw_dominante(raw_componentes, indicador_principal),
                        "evidence_score": round(evidence_score, 6),
                        "suspicion_score": round(suspicion_score, 6),
                        "is_suspicious": True,
                        "pattern": patron,
                        "suspected_pattern": patron,
                        "pattern_detail": ", ".join(DETALLE_INDICADOR[indicador] for indicador in indicadores),
                        "filter": "statistical_temporal",
                        "passed_to_contextual_filter": True,
                        "triggered_indicators": indicadores,
                        "reading": {
                            "activations": int(fila["activations"]),
                            "state_changes": int(fila["state_changes"]),
                            "active_seconds": round(float(fila["active_seconds"]), 3),
                            "inactive_seconds": round(float(fila["inactive_seconds"]), 3),
                            "active_ratio": round(float(fila["active_ratio"]), 6),
                            "gap_seconds": round(float(fila["gap_seconds"]), 3),
                            "time_since_last_change_seconds": round(float(fila["time_since_last_change_seconds"]), 3),
                            "report_gap_seconds": round(float(fila["report_gap_seconds"]), 3),
                            "window_seconds": perfil.window_seconds,
                            "components": componentes,
                            "raw_components": raw_componentes,
                        },
                        "baseline_summary": self._resumen_base_para_salida(base),
                        "ground_truth_label": None,
                        "severity": self._severidad(suspicion_score),
                    }
                )

        return resultados

    def _extraer_features(self, serie: pd.DataFrame, perfil: PerfilBinario) -> pd.DataFrame:
        trabajo = serie.reset_index(drop=True).copy()
        trabajo["prev_value"] = trabajo["value"].shift(1)
        trabajo["report_gap_seconds"] = trabajo["timestamp_dt"].diff().dt.total_seconds().fillna(0.0)
        trabajo["activation_event"] = ((trabajo["prev_value"].fillna(0.0).eq(0.0)) & trabajo["value"].eq(1.0)).astype(int)
        trabajo["state_change_event"] = (
            trabajo["prev_value"].notna() & trabajo["prev_value"].ne(trabajo["value"])
        ).astype(int)

        tiempos = self._timestamps_a_segundos(trabajo["timestamp_dt"])
        valores = trabajo["value"].to_numpy(dtype="float64")
        activaciones_array = trabajo["activation_event"].to_numpy(dtype="int64")
        cambios_array = trabajo["state_change_event"].to_numpy(dtype="int64")
        acumulado_activaciones = np.concatenate([[0], np.cumsum(activaciones_array)])
        acumulado_cambios = np.concatenate([[0], np.cumsum(cambios_array)])
        activo_acumulado = self._activo_acumulado_en_eventos(tiempos, valores)

        if len(trabajo) == 0:
            return pd.DataFrame()

        indices = np.arange(len(trabajo), dtype="int64")
        inicios = tiempos - float(perfil.window_seconds)
        izquierdas = np.searchsorted(tiempos, inicios, side="right")
        activaciones = acumulado_activaciones[indices + 1] - acumulado_activaciones[izquierdas]
        cambios = acumulado_cambios[indices + 1] - acumulado_cambios[izquierdas]

        indices_inicio = np.searchsorted(tiempos, inicios, side="right") - 1
        activo_en_inicio = np.zeros(len(trabajo), dtype="float64")
        con_previo = indices_inicio >= 0
        previos = indices_inicio[con_previo]
        activo_en_inicio[con_previo] = (
            activo_acumulado[previos]
            + np.maximum(0.0, inicios[con_previo] - tiempos[previos])
            * valores[previos]
        )
        activos_segundos = np.maximum(0.0, activo_acumulado - activo_en_inicio)
        inactivos_segundos = np.maximum(
            0.0, float(perfil.window_seconds) - activos_segundos
        )

        activacion_en = np.where(activaciones_array == 1, tiempos, -np.inf)
        cambio_en = np.where(cambios_array == 1, tiempos, -np.inf)
        ultima_activacion = np.maximum.accumulate(activacion_en)
        ultimo_cambio = np.maximum.accumulate(cambio_en)
        gaps_activacion = np.where(
            np.isfinite(ultima_activacion),
            np.maximum(0.0, tiempos - ultima_activacion),
            0.0,
        )
        gaps_cambio = np.where(
            np.isfinite(ultimo_cambio),
            np.maximum(0.0, tiempos - ultimo_cambio),
            0.0,
        )

        salida = trabajo[
            ["entity_id", "timestamp", "timestamp_dt", "state", "value"]
        ].copy()
        salida["window_start"] = (
            trabajo["timestamp_dt"]
            - pd.Timedelta(seconds=perfil.window_seconds)
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        salida["value"] = valores
        salida["activations"] = activaciones.astype("int64")
        salida["state_changes"] = cambios.astype("int64")
        salida["active_seconds"] = activos_segundos
        salida["inactive_seconds"] = inactivos_segundos
        salida["active_ratio"] = np.clip(
            activos_segundos / float(perfil.window_seconds), 0.0, 1.0
        )
        salida["gap_seconds"] = gaps_activacion
        salida["time_since_last_change_seconds"] = gaps_cambio
        salida["report_gap_seconds"] = np.maximum(
            0.0,
            pd.to_numeric(
                trabajo["report_gap_seconds"], errors="coerce"
            ).fillna(0.0).to_numpy(dtype="float64"),
        )
        return salida

    def _activo_acumulado_en_eventos(self, tiempos: np.ndarray, valores: np.ndarray) -> np.ndarray:
        acumulado = np.zeros(len(tiempos), dtype="float64")
        if len(tiempos) <= 1:
            return acumulado
        duraciones = np.diff(tiempos)
        duraciones = np.maximum(duraciones, 0.0)
        acumulado[1:] = np.cumsum(duraciones * valores[:-1])
        return acumulado

    def _segundos_activos_rapido(
        self,
        tiempos: np.ndarray,
        valores: np.ndarray,
        activo_acumulado: np.ndarray,
        inicio: float,
        fin: float,
        indice_fin: int,
    ) -> float:
        if len(tiempos) == 0 or fin <= inicio:
            return 0.0
        activo_fin = activo_acumulado[indice_fin]
        indice_inicio = int(np.searchsorted(tiempos, inicio, side="right")) - 1
        if indice_inicio < 0:
            activo_inicio = 0.0
        else:
            activo_inicio = activo_acumulado[indice_inicio]
            activo_inicio += max(0.0, inicio - tiempos[indice_inicio]) * valores[indice_inicio]
        return max(0.0, activo_fin - activo_inicio)

    def _timestamps_a_segundos(self, timestamps: pd.Series) -> np.ndarray:
        epoch = pd.Timestamp("1970-01-01T00:00:00Z")
        return (timestamps - epoch).dt.total_seconds().to_numpy(dtype="float64")

    def _construir_base_normalidad(self, features: pd.DataFrame, perfil: PerfilBinario) -> dict[str, Any]:
        base_larga = self._construir_base_horizonte(features, perfil)
        recientes = self._features_recientes(features)
        reciente_usable = len(recientes) >= self.min_history_events
        base_reciente = self._construir_base_horizonte(recientes if reciente_usable else features, perfil)
        base = dict(base_larga)
        base.update(
            {
                "long_horizon": base_larga,
                "recent_horizon": base_reciente,
                "recent_horizon_usable": reciente_usable,
                "recent_horizon_events": int(len(recientes)),
                "recent_window_days": self.recent_window_days,
            }
        )
        return base

    def _construir_base_horizonte(self, features: pd.DataFrame, perfil: PerfilBinario) -> dict[str, Any]:
        activaciones = self._resumen_robusto(features["activations"], piso_mad=1.0)
        cambios = self._resumen_robusto(features["state_changes"], piso_mad=1.0)
        proporcion = self._resumen_robusto(features["active_ratio"], piso_mad=perfil.ratio_floor)
        duracion_activa = self._resumen_robusto(features["active_seconds"], piso_mad=60.0)
        duracion_inactiva = self._resumen_robusto(features["inactive_seconds"], piso_mad=60.0)
        gaps_activacion = features.loc[features["gap_seconds"].gt(0), "gap_seconds"]
        gaps_cambio = features.loc[features["time_since_last_change_seconds"].gt(0), "time_since_last_change_seconds"]
        gaps_reporte = features.loc[features["report_gap_seconds"].gt(0), "report_gap_seconds"]
        gap = self._resumen_robusto(gaps_activacion, piso_mad=60.0)
        last_change_gap = self._resumen_robusto(gaps_cambio, piso_mad=60.0)
        report_gap = self._resumen_robusto(gaps_reporte, piso_mad=60.0)
        absence_alert = max(
            perfil.min_absence_gap_seconds,
            gap["median"] + 8.0 * gap["scaled_mad"],
        )
        report_alert = max(
            perfil.min_report_gap_seconds,
            report_gap["median"] + 8.0 * report_gap["scaled_mad"],
        )
        last_change_alert = max(
            perfil.min_absence_gap_seconds,
            last_change_gap["median"] + 8.0 * last_change_gap["scaled_mad"],
        )
        return {
            "activations": activaciones,
            "state_changes": cambios,
            "active_ratio": proporcion,
            "active_duration": duracion_activa,
            "inactive_duration": duracion_inactiva,
            "gap": gap,
            "last_change_gap": last_change_gap,
            "report_gap": report_gap,
            "absence_alert_seconds": absence_alert,
            "last_change_alert_seconds": last_change_alert,
            "report_gap_alert_seconds": report_alert,
            "window_seconds": perfil.window_seconds,
        }

    def _features_recientes(self, features: pd.DataFrame) -> pd.DataFrame:
        if "timestamp_dt" not in features.columns:
            return features
        ultimo = features["timestamp_dt"].max()
        if pd.isna(ultimo):
            return features.iloc[0:0]
        return features[features["timestamp_dt"].ge(ultimo - pd.Timedelta(days=self.recent_window_days))]

    def _puntuar_doble_horizonte(self, fila: dict[str, Any], base: dict[str, Any]) -> tuple[dict[str, float], dict[str, float], float]:
        raw_largo, comp_largo = self._puntuar_horizonte(fila, base["long_horizon"])
        if not base["recent_horizon_usable"]:
            return (
                {clave: round(valor, 6) for clave, valor in raw_largo.items()},
                {clave: round(valor, 6) for clave, valor in comp_largo.items()},
                self._evidencia_componentes(comp_largo),
            )

        raw_reciente, comp_reciente = self._puntuar_horizonte(fila, base["recent_horizon"])
        raw = {clave: min(raw_largo[clave], raw_reciente[clave]) for clave in raw_largo}
        componentes = {clave: min(comp_largo[clave], comp_reciente[clave]) for clave in comp_largo}
        return (
            {clave: round(valor, 6) for clave, valor in raw.items()},
            {clave: round(valor, 6) for clave, valor in componentes.items()},
            self._evidencia_componentes(componentes),
        )

    def _puntuar_horizonte(self, fila: dict[str, Any], base: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        raw_a, s_a = self._score_alto(fila["activations"], base["activations"])
        raw_c, s_c = self._score_alto(fila["state_changes"], base["state_changes"])
        raw_p, s_p = self._score_alto(fila["active_ratio"], base["active_ratio"])
        raw_g, s_g = self._score_gap(fila["gap_seconds"], base["gap"], base["absence_alert_seconds"])
        raw_m, s_m = self._score_gap(fila["report_gap_seconds"], base["report_gap"], base["report_gap_alert_seconds"])
        raw_off, s_off = self._score_alto(fila["inactive_seconds"], base["inactive_duration"])
        raw_tlc, s_tlc = self._score_gap(fila["time_since_last_change_seconds"], base["last_change_gap"], base["last_change_alert_seconds"])
        raw_d, s_d = self._score_duracion(fila["active_seconds"], base)
        return (
            {
                "Z_A_i": raw_a,
                "Z_C_i": raw_c,
                "Z_P_on_i": raw_p,
                "Z_G_i": raw_g,
                "Z_M_i": raw_m,
                "Z_D_off_i": raw_off,
                "Z_TLC_i": raw_tlc,
                "Z_D_i": raw_d,
            },
            {
                "s_A": s_a,
                "s_C": s_c,
                "s_P": s_p,
                "s_G": s_g,
                "s_M": s_m,
                "s_D_off": s_off,
                "s_TLC": s_tlc,
                "s_D": s_d,
            },
        )

    def _evidencia_componentes(self, componentes: dict[str, float]) -> float:
        evidencia_base = sum(componentes[clave] for clave in ["s_A", "s_C", "s_P", "s_G", "s_M"]) / 5.0
        return min(1.0, max(evidencia_base, componentes["s_TLC"], componentes["s_D"]))

    def _score_alto(self, valor: float, resumen: dict[str, float]) -> tuple[float, float]:
        z = max(0.0, (float(valor) - resumen["median"]) / (resumen["scaled_mad"] + 1e-9))
        return z, self._score_desde_z(z)

    def _score_gap(self, valor: float, resumen: dict[str, float], alerta: float) -> tuple[float, float]:
        if float(valor) < alerta:
            return 0.0, 0.0
        z = max(0.0, (float(valor) - resumen["median"]) / (resumen["scaled_mad"] + 1e-9))
        score = max(self._score_desde_z(z), min(1.0, float(valor) / alerta))
        return z, score

    def _score_duracion(self, active_seconds: float, base: dict[str, Any]) -> tuple[float, float]:
        z_active = max(0.0, (float(active_seconds) - base["active_duration"]["median"]) / (base["active_duration"]["scaled_mad"] + 1e-9))
        return z_active, self._score_desde_z(z_active)

    def _score_desde_z(self, z_value: float) -> float:
        if not math.isfinite(z_value) or z_value <= self.zona_muerta_z:
            return 0.0
        return max(0.0, min(1.0, z_value / self.tau_indicador))

    def _indicadores_activados(self, componentes: dict[str, float]) -> list[str]:
        mapa = {
            "s_A": "activation_count",
            "s_C": "state_change_count",
            "s_P": "active_time_ratio",
            "s_G": "time_since_last_activation",
            "s_M": "report_gap",
            "s_TLC": "time_since_last_change",
            "s_D": "z_duration_score",
        }
        return [indicador for clave, indicador in mapa.items() if componentes.get(clave, 0.0) >= self.componente_minimo]

    def _indicador_principal(self, componentes: dict[str, float], indicadores: list[str]) -> str:
        componente_por_indicador = {
            "activation_count": componentes.get("s_A", 0.0),
            "state_change_count": componentes.get("s_C", 0.0),
            "active_time_ratio": componentes.get("s_P", 0.0),
            "time_since_last_activation": componentes.get("s_G", 0.0),
            "report_gap": componentes.get("s_M", 0.0),
            "time_since_last_change": componentes.get("s_TLC", 0.0),
            "z_duration_score": componentes.get("s_D", 0.0),
        }
        return max(indicadores, key=lambda indicador: componente_por_indicador.get(indicador, 0.0))

    def _raw_dominante(self, raw_componentes: dict[str, float], indicador: str) -> float:
        mapa = {
            "activation_count": "Z_A_i",
            "state_change_count": "Z_C_i",
            "active_time_ratio": "Z_P_on_i",
            "time_since_last_activation": "Z_G_i",
            "report_gap": "Z_M_i",
            "time_since_last_change": "Z_TLC_i",
            "z_duration_score": "Z_D_i",
        }
        return round(float(raw_componentes.get(mapa[indicador], 0.0)), 6)

    def _agrupar_resultados(self, resultados: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not resultados:
            return []
        ordenados = sorted(resultados, key=lambda item: (item["entity_id"], item["pattern"], item["timestamp"]))
        grupos: list[list[dict[str, Any]]] = []
        for resultado in ordenados:
            timestamp = pd.to_datetime(resultado["timestamp"], errors="coerce", utc=True)
            ultimo_grupo = grupos[-1] if grupos else None
            if ultimo_grupo:
                ultimo = ultimo_grupo[-1]
                ultimo_timestamp = pd.to_datetime(ultimo["timestamp"], errors="coerce", utc=True)
                misma_serie = resultado["entity_id"] == ultimo["entity_id"] and resultado["pattern"] == ultimo["pattern"]
                cerca = pd.notna(timestamp) and pd.notna(ultimo_timestamp) and (
                    timestamp - ultimo_timestamp
                ).total_seconds() <= self.supresion_segundos
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

    def _resumen_robusto(self, valores: pd.Series, piso_mad: float) -> dict[str, float]:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        if limpio.empty:
            limpio = pd.Series([0.0], dtype="float64")
        mediana = float(limpio.median())
        mad = max(self._mad(limpio), piso_mad)
        return {
            "median": mediana,
            "mad": mad,
            "scaled_mad": 1.4826 * mad,
            "p90": float(limpio.quantile(0.90)),
            "p95": float(limpio.quantile(0.95)),
            "p99": float(limpio.quantile(0.99)),
        }

    def _resumen_base_para_salida(self, base: dict[str, Any]) -> dict[str, Any]:
        return {
            "window_seconds": int(base["window_seconds"]),
            "activations_median": round(base["activations"]["median"], 6),
            "state_changes_median": round(base["state_changes"]["median"], 6),
            "active_ratio_median": round(base["active_ratio"]["median"], 6),
            "active_duration_median": round(base["active_duration"]["median"], 6),
            "inactive_duration_median": round(base["inactive_duration"]["median"], 6),
            "absence_alert_seconds": round(base["absence_alert_seconds"], 3),
            "last_change_alert_seconds": round(base["last_change_alert_seconds"], 3),
            "report_gap_alert_seconds": round(base["report_gap_alert_seconds"], 3),
            "horizon_model": "long_and_recent",
            "recent_window_days": int(base["recent_window_days"]),
            "recent_horizon_events": int(base["recent_horizon_events"]),
            "recent_horizon_usable": bool(base["recent_horizon_usable"]),
            "long_report_gap_alert_seconds": round(base["long_horizon"]["report_gap_alert_seconds"], 3),
            "recent_report_gap_alert_seconds": round(base["recent_horizon"]["report_gap_alert_seconds"], 3),
        }

    def _perfil_binario(self, entity_id: str, device_class: str) -> PerfilBinario:
        texto = f"{entity_id} {device_class}".lower()
        if "motion" in texto or "pir" in texto:
            return PerfilBinario(
                window_seconds=60 * 60,
                min_report_gap_seconds=12 * 60 * 60,
                min_absence_gap_seconds=24 * 60 * 60,
                ratio_floor=0.05,
            )
        if "door" in texto or "contact" in texto or "window" in texto:
            return PerfilBinario(
                window_seconds=60 * 60,
                min_report_gap_seconds=24 * 60 * 60,
                min_absence_gap_seconds=48 * 60 * 60,
                ratio_floor=0.05,
            )
        return PerfilBinario(
            window_seconds=60 * 60,
            min_report_gap_seconds=24 * 60 * 60,
            min_absence_gap_seconds=48 * 60 * 60,
            ratio_floor=0.05,
        )

    def _score_a_probabilidad(self, evidence_score: float) -> float:
        evidencia = max(0.0, min(1.0, float(evidence_score)))
        return evidencia / (evidencia + self.tau_probabilidad) if evidencia > 0 else 0.0

    def _probabilidad_a_evidencia(self, suspicion_threshold: float) -> float:
        umbral = max(0.01, min(0.99, float(suspicion_threshold)))
        return min(1.0, (umbral * self.tau_probabilidad) / (1.0 - umbral))

    def _severidad(self, suspicion_score: float) -> str:
        if suspicion_score >= 0.85:
            return "alta"
        if suspicion_score >= 0.70:
            return "media"
        return "baja"

    def _mad(self, valores: pd.Series) -> float:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        if limpio.empty:
            return 0.0
        mediana = limpio.median()
        return float((limpio - mediana).abs().median())
