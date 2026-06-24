from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd


INDICADORES_CONTINUOS = {
    "hampel": "isolated_spike",
    "robust_rate_change": "abrupt_change",
    "cusum_residual": "persistent_shift",
    "constant_persistence": "stuck_value",
    "reading_gap": "missing_gap",
}

DETALLE_INDICADOR = {
    "hampel": "Filtro Hampel",
    "robust_rate_change": "Tasa de cambio robusta",
    "cusum_residual": "CUSUM sobre residuales",
    "constant_persistence": "Persistencia constante",
    "reading_gap": "Brecha temporal de lectura",
}


@dataclass(frozen=True)
class PerfilContinuo:
    """Parametros conservadores segun el tipo fisico del sensor."""

    value_floor: float
    rate_floor_per_minute: float
    const_epsilon_floor: float
    min_constant_seconds: float
    min_gap_seconds: float
    physical_low: float | None = None
    physical_high: float | None = None
    extreme_low: float | None = None
    extreme_high: float | None = None


class FiltroContinuoEstadistico:
    """Filtro estadistico-temporal para entidades continuas.

    Implementa solo los indicadores definidos para entidades continuas en la
    tesis: Hampel, tasa de cambio robusta, CUSUM sobre residuales, persistencia
    constante y brecha temporal de lectura.
    """

    window_hampel = 12
    alpha_ewma = 0.2
    tau_probabilidad = 0.4
    umbral_z = 3.5
    zona_muerta_z = 2.5
    supresion_segundos = 15 * 60

    def analizar(self, eventos: pd.DataFrame, entidades: pd.DataFrame, suspicion_threshold: float) -> list[dict[str, Any]]:
        if eventos.empty:
            return []
        trabajo = eventos.copy()
        if "sensor_kind" not in trabajo.columns:
            return []
        trabajo = trabajo[trabajo["sensor_kind"].eq("continuous")].copy()
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
        serie["value"] = pd.to_numeric(serie["value"], errors="coerce")
        serie = serie[serie["value"].notna() & serie["timestamp_dt"].notna()].copy()
        if len(serie) < self.window_hampel:
            return []

        perfil = self._perfil_continuo(entity_id, str(entidad.get("device_class", "")))
        base = self._construir_base_normalidad(serie, entity_id, perfil)
        if not base:
            return []

        serie = self._preparar_variables_temporales(serie, base)
        raw_threshold = self._probabilidad_a_raw(suspicion_threshold)
        resultados: list[dict[str, Any]] = []

        c_pos = 0.0
        c_neg = 0.0
        ewma = None
        duracion_constante = 0.0
        valor_previo = None

        for fila in serie.to_dict("records"):
            valor = float(fila["value"])
            gap_seconds = self._numero(fila.get("gap_seconds"), 0.0)
            delta_value = self._numero(fila.get("delta_value"), 0.0)
            rate_per_minute = self._numero(fila.get("rate_per_minute"), 0.0)

            if ewma is None:
                ewma = valor
            prediccion = ewma
            residual = valor - prediccion

            if gap_seconds > base["cusum_reset_gap_seconds"]:
                c_pos = 0.0
                c_neg = 0.0
                duracion_constante = 0.0

            fisico = self._score_fisico(valor, perfil)
            dentro_banda = base["normal_lower"] <= valor <= base["normal_upper"]

            raw_h, s_h = self._score_hampel(fila, base)
            if dentro_banda and fisico <= 0 and s_h < 1.0:
                s_h = 0.0
            if fisico > s_h:
                raw_h = max(raw_h, fisico * self.umbral_z)
                s_h = fisico

            raw_r, s_r = self._score_tasa_robusta(rate_per_minute, base)
            if dentro_banda and fisico <= 0 and abs(delta_value) < base["normal_delta_for_rate"]:
                s_r = 0.0

            z_residual = residual / base["residual_scale"]
            if dentro_banda and abs(z_residual) < 1.5:
                c_pos = 0.0
                c_neg = 0.0
            else:
                c_pos = max(0.0, c_pos + z_residual - base["cusum_k"])
                c_neg = max(0.0, c_neg - z_residual - base["cusum_k"])
            raw_c = max(c_pos, c_neg)
            s_c = min(1.0, raw_c / base["cusum_theta"])
            if dentro_banda and fisico <= 0:
                s_c = 0.0

            if valor_previo is not None and abs(valor - valor_previo) <= base["epsilon_const"] and gap_seconds <= base["gap_alert_seconds"]:
                duracion_constante += max(gap_seconds, 0.0)
            else:
                duracion_constante = 0.0
            raw_d_const, s_d_const = self._score_persistencia_constante(duracion_constante, base)

            raw_m, s_m = self._score_brecha_temporal(gap_seconds, base)

            componentes = {
                "s_H": round(s_h, 6),
                "s_R": round(s_r, 6),
                "s_C": round(s_c, 6),
                "s_D_const": round(s_d_const, 6),
                "s_M": round(s_m, 6),
            }
            raw_componentes = {
                "H_i": round(raw_h, 6),
                "Z_R_i": round(raw_r, 6),
                "C_residual": round(raw_c, 6),
                "Z_D_const": round(raw_d_const, 6),
                "Z_M": round(raw_m, 6),
            }
            indicadores = self._indicadores_activados(componentes, raw_threshold)
            evidence_score = max(componentes.values()) if componentes else 0.0
            suspicion_score = self._score_a_probabilidad(evidence_score)

            if indicadores and suspicion_score >= suspicion_threshold:
                indicador_principal = self._indicador_principal(componentes, indicadores)
                patron = INDICADORES_CONTINUOS[indicador_principal]
                resultados.append(
                    {
                        "entity_id": entity_id,
                        "timestamp": str(fila["timestamp"]),
                        "window_start": str(fila["timestamp"]),
                        "window_end": str(fila["timestamp"]),
                        "sensor_kind": "continuous",
                        "tipo_interno": "continua",
                        "observed_state": str(fila.get("state", valor)),
                        "value": round(valor, 6),
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
                            "value": round(valor, 6),
                            "delta_value": round(delta_value, 6),
                            "rate_per_minute": round(rate_per_minute, 6),
                            "gap_seconds": round(gap_seconds, 3),
                            "constant_duration_seconds": round(duracion_constante, 3),
                            "residual": round(residual, 6),
                            "components": componentes,
                            "raw_components": raw_componentes,
                        },
                        "baseline_summary": self._resumen_base_para_salida(base),
                        "ground_truth_label": None,
                        "severity": self._severidad(suspicion_score),
                    }
                )

            ewma = self.alpha_ewma * valor + (1.0 - self.alpha_ewma) * ewma
            valor_previo = valor

        return resultados

    def _preparar_variables_temporales(self, serie: pd.DataFrame, base: dict[str, Any]) -> pd.DataFrame:
        trabajo = serie.copy()
        trabajo["prev_value"] = trabajo["value"].shift(1)
        trabajo["delta_value"] = trabajo["value"] - trabajo["prev_value"]
        trabajo["gap_seconds"] = trabajo["timestamp_dt"].diff().dt.total_seconds()
        trabajo["rate_per_minute"] = (
            trabajo["delta_value"].abs()
            / (trabajo["gap_seconds"].clip(lower=1.0) / 60.0)
        )
        trabajo[["delta_value", "gap_seconds", "rate_per_minute"]] = trabajo[
            ["delta_value", "gap_seconds", "rate_per_minute"]
        ].fillna(0.0)

        rolling = trabajo["value"].rolling(self.window_hampel, min_periods=max(6, self.window_hampel // 2))
        trabajo["local_median"] = rolling.median()
        trabajo["local_mad"] = rolling.apply(lambda valores: self._mad(pd.Series(valores)), raw=False)
        trabajo["local_median"] = trabajo["local_median"].fillna(base["median"])
        trabajo["local_mad"] = trabajo["local_mad"].fillna(base["mad"])
        return trabajo

    def _construir_base_normalidad(
        self,
        serie: pd.DataFrame,
        entity_id: str,
        perfil: PerfilContinuo,
    ) -> dict[str, Any]:
        valores = pd.to_numeric(serie["value"], errors="coerce").dropna()
        if len(valores) < self.window_hampel:
            return {}

        q01 = float(valores.quantile(0.01))
        q05 = float(valores.quantile(0.05))
        q25 = float(valores.quantile(0.25))
        q50 = float(valores.quantile(0.50))
        q75 = float(valores.quantile(0.75))
        q95 = float(valores.quantile(0.95))
        q99 = float(valores.quantile(0.99))
        iqr = max(q75 - q25, perfil.value_floor)

        # La normalidad se estima desde el cuerpo robusto del historico, no
        # desde el minimo y maximo observados, porque esos extremos pueden ser
        # justamente lecturas atipicas.
        cuerpo_lower = max(q01, q25 - 3.0 * iqr)
        cuerpo_upper = min(q99, q75 + 3.0 * iqr)
        mascara_cuerpo = valores.between(cuerpo_lower, cuerpo_upper)
        if perfil.physical_low is not None:
            mascara_cuerpo &= valores.ge(perfil.physical_low)
        if perfil.physical_high is not None:
            mascara_cuerpo &= valores.le(perfil.physical_high)
        valores_limpios = valores[mascara_cuerpo]
        if len(valores_limpios) < max(30, len(valores) * 0.20):
            valores_limpios = valores

        mediana = float(valores_limpios.median())
        mad = max(self._mad(valores_limpios), perfil.value_floor)
        q05_limpio = float(valores_limpios.quantile(0.05))
        q25_limpio = float(valores_limpios.quantile(0.25))
        q75_limpio = float(valores_limpios.quantile(0.75))
        q95_limpio = float(valores_limpios.quantile(0.95))
        iqr_limpio = max(q75_limpio - q25_limpio, perfil.value_floor)
        normal_lower = q05_limpio - 1.5 * iqr_limpio
        normal_upper = q95_limpio + 1.5 * iqr_limpio
        if perfil.physical_low is not None:
            normal_lower = max(normal_lower, perfil.physical_low)
        if perfil.physical_high is not None:
            normal_upper = min(normal_upper, perfil.physical_high)

        temporal = serie[["timestamp_dt", "value"]].copy()
        temporal["gap_seconds"] = temporal["timestamp_dt"].diff().dt.total_seconds()
        temporal["delta_value"] = temporal["value"].diff()
        temporal["rate_per_minute"] = temporal["delta_value"].abs() / (temporal["gap_seconds"].clip(lower=1.0) / 60.0)
        gaps = temporal["gap_seconds"].dropna()
        rates = temporal.loc[temporal["gap_seconds"].fillna(0).gt(0), "rate_per_minute"].dropna()
        rate_summary = self._resumen_robusto(rates, perfil.rate_floor_per_minute)
        gap_summary = self._resumen_robusto(gaps, 60.0)

        ewma_previa = temporal["value"].ewm(alpha=self.alpha_ewma, adjust=False).mean().shift(1)
        residuales = (temporal["value"] - ewma_previa).dropna()
        residual_mad = max(self._mad(residuales), perfil.value_floor)

        epsilon_const = max(perfil.const_epsilon_floor, min(mad * 0.03, perfil.value_floor))
        duraciones_constantes = self._duraciones_constantes(temporal, epsilon_const, perfil.min_gap_seconds)
        const_summary = self._resumen_robusto(pd.Series(duraciones_constantes, dtype="float64"), 60.0)
        gap_alert = max(
            perfil.min_gap_seconds,
            gap_summary["median"] + 8.0 * gap_summary["scaled_mad"],
            gap_summary["p99"] * 2.0,
        )
        const_alert = max(
            perfil.min_constant_seconds,
            const_summary["median"] + 8.0 * const_summary["scaled_mad"],
            const_summary["p99"] * 2.0,
        )

        return {
            "entity_id": entity_id,
            "median": mediana,
            "mad": mad,
            "scaled_mad": 1.4826 * mad,
            "q05": q05_limpio,
            "q95": q95_limpio,
            "normal_lower": normal_lower,
            "normal_upper": normal_upper,
            "rate_summary": rate_summary,
            "gap_summary": gap_summary,
            "const_summary": const_summary,
            "residual_scale": max(1.4826 * residual_mad, perfil.value_floor),
            "cusum_k": 0.5,
            "cusum_theta": 8.0,
            "cusum_reset_gap_seconds": max(gap_alert, 6 * 60 * 60),
            "epsilon_const": epsilon_const,
            "gap_alert_seconds": gap_alert,
            "constant_alert_seconds": const_alert,
            "normal_delta_for_rate": max(1.4826 * mad, perfil.value_floor * 2.0),
            "perfil": perfil,
        }

    def _score_hampel(self, fila: dict[str, Any], base: dict[str, Any]) -> tuple[float, float]:
        local_median = self._numero(fila.get("local_median"), base["median"])
        local_mad = max(self._numero(fila.get("local_mad"), base["mad"]), base["perfil"].value_floor)
        h = abs(float(fila["value"]) - local_median) / (1.4826 * local_mad + 1e-9)
        return h, self._score_desde_z(h)

    def _score_tasa_robusta(self, rate_per_minute: float, base: dict[str, Any]) -> tuple[float, float]:
        resumen = base["rate_summary"]
        z_rate = abs(rate_per_minute - resumen["median"]) / (resumen["scaled_mad"] + 1e-9)
        return z_rate, self._score_desde_z(z_rate)

    def _score_persistencia_constante(self, duracion_segundos: float, base: dict[str, Any]) -> tuple[float, float]:
        if duracion_segundos < base["constant_alert_seconds"]:
            return 0.0, 0.0
        resumen = base["const_summary"]
        z_const = abs(duracion_segundos - resumen["median"]) / (resumen["scaled_mad"] + 1e-9)
        score = max(self._score_desde_z(z_const), min(1.0, duracion_segundos / base["constant_alert_seconds"]))
        return z_const, score

    def _score_brecha_temporal(self, gap_seconds: float, base: dict[str, Any]) -> tuple[float, float]:
        if gap_seconds < base["gap_alert_seconds"]:
            return 0.0, 0.0
        resumen = base["gap_summary"]
        z_gap = abs(gap_seconds - resumen["median"]) / (resumen["scaled_mad"] + 1e-9)
        score = max(self._score_desde_z(z_gap), min(1.0, gap_seconds / base["gap_alert_seconds"]))
        return z_gap, score

    def _score_desde_z(self, z_value: float) -> float:
        if not math.isfinite(z_value) or z_value <= self.zona_muerta_z:
            return 0.0
        return max(0.0, min(1.0, (z_value - self.zona_muerta_z) / (self.umbral_z - self.zona_muerta_z)))

    def _score_fisico(self, valor: float, perfil: PerfilContinuo) -> float:
        if perfil.extreme_low is not None and valor <= perfil.extreme_low:
            return 1.0
        if perfil.extreme_high is not None and valor >= perfil.extreme_high:
            return 1.0
        if perfil.physical_low is not None and valor < perfil.physical_low:
            return min(1.0, 0.75 + abs(valor - perfil.physical_low) / max(abs(perfil.physical_low), 10.0))
        if perfil.physical_high is not None and valor > perfil.physical_high:
            return min(1.0, 0.75 + abs(valor - perfil.physical_high) / max(abs(perfil.physical_high), 10.0))
        return 0.0

    def _perfil_continuo(self, entity_id: str, device_class: str) -> PerfilContinuo:
        texto = f"{entity_id} {device_class}".lower()
        if "humidity" in texto:
            return PerfilContinuo(
                value_floor=2.0,
                rate_floor_per_minute=2.0,
                const_epsilon_floor=0.10,
                min_constant_seconds=24 * 60 * 60,
                min_gap_seconds=2 * 60 * 60,
                physical_low=0.0,
                physical_high=100.0,
                extreme_low=5.0,
                extreme_high=98.0,
            )
        if "temperature" in texto or "temp" in texto:
            return PerfilContinuo(
                value_floor=0.25,
                rate_floor_per_minute=0.35,
                const_epsilon_floor=0.03,
                min_constant_seconds=24 * 60 * 60,
                min_gap_seconds=2 * 60 * 60,
                physical_low=-20.0,
                physical_high=60.0,
                extreme_low=0.0,
                extreme_high=45.0,
            )
        return PerfilContinuo(
            value_floor=0.05,
            rate_floor_per_minute=0.10,
            const_epsilon_floor=0.01,
            min_constant_seconds=24 * 60 * 60,
            min_gap_seconds=2 * 60 * 60,
        )

    def _indicadores_activados(self, componentes: dict[str, float], raw_threshold: float) -> list[str]:
        mapa = {
            "s_H": "hampel",
            "s_R": "robust_rate_change",
            "s_C": "cusum_residual",
            "s_D_const": "constant_persistence",
            "s_M": "reading_gap",
        }
        return [indicador for clave, indicador in mapa.items() if componentes.get(clave, 0.0) >= raw_threshold]

    def _indicador_principal(self, componentes: dict[str, float], indicadores: list[str]) -> str:
        componente_por_indicador = {
            "hampel": componentes.get("s_H", 0.0),
            "robust_rate_change": componentes.get("s_R", 0.0),
            "cusum_residual": componentes.get("s_C", 0.0),
            "constant_persistence": componentes.get("s_D_const", 0.0),
            "reading_gap": componentes.get("s_M", 0.0),
        }
        return max(indicadores, key=lambda indicador: componente_por_indicador.get(indicador, 0.0))

    def _raw_dominante(self, raw_componentes: dict[str, float], indicador: str) -> float:
        mapa = {
            "hampel": "H_i",
            "robust_rate_change": "Z_R_i",
            "cusum_residual": "C_residual",
            "constant_persistence": "Z_D_const",
            "reading_gap": "Z_M",
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

    def _duraciones_constantes(self, temporal: pd.DataFrame, epsilon: float, gap_max: float) -> list[float]:
        duraciones: list[float] = []
        duracion_actual = 0.0
        valor_anterior: float | None = None
        for fila in temporal.to_dict("records"):
            valor = float(fila["value"])
            gap = self._numero(fila.get("gap_seconds"), 0.0)
            if valor_anterior is not None and abs(valor - valor_anterior) <= epsilon and gap <= gap_max:
                duracion_actual += max(gap, 0.0)
            else:
                if duracion_actual > 0:
                    duraciones.append(duracion_actual)
                duracion_actual = 0.0
            valor_anterior = valor
        if duracion_actual > 0:
            duraciones.append(duracion_actual)
        return duraciones or [0.0]

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
            "median": round(base["median"], 6),
            "mad": round(base["mad"], 6),
            "normal_lower": round(base["normal_lower"], 6),
            "normal_upper": round(base["normal_upper"], 6),
            "rate_median": round(base["rate_summary"]["median"], 6),
            "rate_mad": round(base["rate_summary"]["mad"], 6),
            "gap_alert_seconds": round(base["gap_alert_seconds"], 3),
            "constant_alert_seconds": round(base["constant_alert_seconds"], 3),
            "epsilon_const": round(base["epsilon_const"], 6),
            "cusum_theta": round(base["cusum_theta"], 6),
        }

    def _score_a_probabilidad(self, raw_score: float) -> float:
        raw = max(0.0, min(1.0, float(raw_score)))
        return raw / (raw + self.tau_probabilidad) if raw > 0 else 0.0

    def _probabilidad_a_raw(self, suspicion_threshold: float) -> float:
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

    def _numero(self, valor: Any, defecto: float) -> float:
        try:
            numero = float(valor)
        except (TypeError, ValueError):
            return defecto
        return numero if math.isfinite(numero) else defecto
