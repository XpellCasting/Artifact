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
}

DETALLE_INDICADOR = {
    "activation_count": "Activaciones A_i(W)",
    "state_change_count": "Cambios de estado C_i(W)",
    "active_time_ratio": "Proporcion activa P_on,i(W)",
    "time_since_last_activation": "Tiempo desde ultima activacion G_i(t)",
    "report_gap": "Brecha temporal de reporte M_i(t)",
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
        serie["value"] = pd.to_numeric(serie["value"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
        serie = serie[serie["timestamp_dt"].notna()].copy()
        if len(serie) < 3:
            return []

        perfil = self._perfil_binario(entity_id, str(entidad.get("device_class", "")))
        features = self._extraer_features(serie, perfil)
        if features.empty:
            return []
        base = self._construir_base_normalidad(features, perfil)
        raw_threshold = self._probabilidad_a_evidencia(suspicion_threshold)
        resultados: list[dict[str, Any]] = []

        for fila in features.to_dict("records"):
            raw_a, s_a = self._score_alto(fila["activations"], base["activations"])
            raw_c, s_c = self._score_alto(fila["state_changes"], base["state_changes"])
            raw_p, s_p = self._score_alto(fila["active_ratio"], base["active_ratio"])
            raw_g, s_g = self._score_gap(fila["gap_seconds"], base["gap"], base["absence_alert_seconds"])
            raw_m, s_m = self._score_gap(fila["report_gap_seconds"], base["report_gap"], base["report_gap_alert_seconds"])

            componentes = {
                "s_A": round(s_a, 6),
                "s_C": round(s_c, 6),
                "s_P": round(s_p, 6),
                "s_G": round(s_g, 6),
                "s_M": round(s_m, 6),
            }
            raw_componentes = {
                "Z_A_i": round(raw_a, 6),
                "Z_C_i": round(raw_c, 6),
                "Z_P_on_i": round(raw_p, 6),
                "Z_G_i": round(raw_g, 6),
                "Z_M_i": round(raw_m, 6),
            }
            evidence_score = min(1.0, sum(componentes.values()) / 5.0)
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
                            "active_ratio": round(float(fila["active_ratio"]), 6),
                            "gap_seconds": round(float(fila["gap_seconds"]), 3),
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

        ultima_activacion: pd.Timestamp | None = None
        filas: list[dict[str, Any]] = []
        for indice, fila in trabajo.iterrows():
            timestamp = fila["timestamp_dt"]
            tiempo_actual = tiempos[indice]
            if int(fila["activation_event"]) == 1:
                ultima_activacion = timestamp
            gap_activacion = 0.0
            if ultima_activacion is not None:
                gap_activacion = max(0.0, (timestamp - ultima_activacion).total_seconds())

            window_start = timestamp - pd.Timedelta(seconds=perfil.window_seconds)
            inicio_segundos = tiempo_actual - perfil.window_seconds
            izquierda = int(np.searchsorted(tiempos, inicio_segundos, side="right"))
            activaciones = int(acumulado_activaciones[indice + 1] - acumulado_activaciones[izquierda])
            cambios = int(acumulado_cambios[indice + 1] - acumulado_cambios[izquierda])
            activo_segundos = self._segundos_activos_rapido(
                tiempos=tiempos,
                valores=valores,
                activo_acumulado=activo_acumulado,
                inicio=inicio_segundos,
                fin=tiempo_actual,
                indice_fin=indice,
            )
            filas.append(
                {
                    "entity_id": fila["entity_id"],
                    "timestamp": fila["timestamp"],
                    "timestamp_dt": timestamp,
                    "window_start": window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "state": fila["state"],
                    "value": float(fila["value"]),
                    "activations": activaciones,
                    "state_changes": cambios,
                    "active_seconds": activo_segundos,
                    "active_ratio": max(0.0, min(1.0, activo_segundos / perfil.window_seconds)),
                    "gap_seconds": gap_activacion,
                    "report_gap_seconds": max(0.0, float(fila["report_gap_seconds"])),
                }
            )
        return pd.DataFrame(filas)

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
        activaciones = self._resumen_robusto(features["activations"], piso_mad=1.0)
        cambios = self._resumen_robusto(features["state_changes"], piso_mad=1.0)
        proporcion = self._resumen_robusto(features["active_ratio"], piso_mad=perfil.ratio_floor)
        gaps_activacion = features.loc[features["gap_seconds"].gt(0), "gap_seconds"]
        gaps_reporte = features.loc[features["report_gap_seconds"].gt(0), "report_gap_seconds"]
        gap = self._resumen_robusto(gaps_activacion, piso_mad=60.0)
        report_gap = self._resumen_robusto(gaps_reporte, piso_mad=60.0)
        absence_alert = max(
            perfil.min_absence_gap_seconds,
            gap["median"] + 8.0 * gap["scaled_mad"],
            gap["p99"] * 1.5,
        )
        report_alert = max(
            perfil.min_report_gap_seconds,
            report_gap["median"] + 8.0 * report_gap["scaled_mad"],
            report_gap["p99"] * 2.0,
        )
        return {
            "activations": activaciones,
            "state_changes": cambios,
            "active_ratio": proporcion,
            "gap": gap,
            "report_gap": report_gap,
            "absence_alert_seconds": absence_alert,
            "report_gap_alert_seconds": report_alert,
            "window_seconds": perfil.window_seconds,
        }

    def _score_alto(self, valor: float, resumen: dict[str, float]) -> tuple[float, float]:
        z = max(0.0, (float(valor) - resumen["median"]) / (resumen["scaled_mad"] + 1e-9))
        return z, self._score_desde_z(z)

    def _score_gap(self, valor: float, resumen: dict[str, float], alerta: float) -> tuple[float, float]:
        if float(valor) < alerta:
            return 0.0, 0.0
        z = max(0.0, (float(valor) - resumen["median"]) / (resumen["scaled_mad"] + 1e-9))
        score = max(self._score_desde_z(z), min(1.0, float(valor) / alerta))
        return z, score

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
        }
        return [indicador for clave, indicador in mapa.items() if componentes.get(clave, 0.0) >= self.componente_minimo]

    def _indicador_principal(self, componentes: dict[str, float], indicadores: list[str]) -> str:
        componente_por_indicador = {
            "activation_count": componentes.get("s_A", 0.0),
            "state_change_count": componentes.get("s_C", 0.0),
            "active_time_ratio": componentes.get("s_P", 0.0),
            "time_since_last_activation": componentes.get("s_G", 0.0),
            "report_gap": componentes.get("s_M", 0.0),
        }
        return max(indicadores, key=lambda indicador: componente_por_indicador.get(indicador, 0.0))

    def _raw_dominante(self, raw_componentes: dict[str, float], indicador: str) -> float:
        mapa = {
            "activation_count": "Z_A_i",
            "state_change_count": "Z_C_i",
            "active_time_ratio": "Z_P_on_i",
            "time_since_last_activation": "Z_G_i",
            "report_gap": "Z_M_i",
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
            "absence_alert_seconds": round(base["absence_alert_seconds"], 3),
            "report_gap_alert_seconds": round(base["report_gap_alert_seconds"], 3),
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
