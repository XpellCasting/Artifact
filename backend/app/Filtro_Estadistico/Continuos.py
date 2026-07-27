from __future__ import annotations

import math
import os
from configparser import ConfigParser, Error
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd


INDICADORES_CONTINUOS = {
    "hampel": "isolated_spike",
    "robust_rate_change": "abrupt_change",
    "cusum_residual": "persistent_shift",
    "constant_persistence": "stuck_value",
    "reading_gap": "missing_gap",
    "dspot_extreme": "extreme_value",
}

DETALLE_INDICADOR = {
    "hampel": "Filtro Hampel",
    "robust_rate_change": "Tasa de cambio robusta",
    "cusum_residual": "CUSUM sobre residuales",
    "constant_persistence": "Persistencia constante",
    "reading_gap": "Brecha temporal de lectura",
    "dspot_extreme": "DSPOT extremo",
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

    Implementa los indicadores de tesis y suma DSPOT solo para extremos.
    """

    window_hampel = 12
    alpha_ewma = 0.2
    tau_probabilidad = 0.4
    umbral_z = 3.5
    zona_muerta_z = 2.5
    supresion_segundos = 15 * 60

    def __init__(self, conf_path: Path | None = None) -> None:
        self.conf_path = conf_path or Path(os.getenv("CONTINUOUS_CONF_PATH", Path(__file__).resolve().parents[1] / "conf.txt"))
        self.config = ConfigParser()
        self.config.optionxform = str
        if self.conf_path.exists():
            self.config.read(self.conf_path, encoding="utf-8")

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

        base = self._construir_base_normalidad(serie, entity_id)
        if not base:
            return []
        perfil = base["perfil"]

        serie = self._preparar_variables_temporales(serie, base)
        raw_threshold = self._probabilidad_a_raw(suspicion_threshold)
        resultados: list[dict[str, Any]] = []

        c_pos = 0.0
        c_neg = 0.0
        ewma = None
        duracion_constante = 0.0
        valor_previo = None
        brecha_reportada = False

        for fila in serie.to_dict("records"):
            valor = float(fila["value"])
            gap_seconds = self._numero(fila.get("gap_seconds"), 0.0)
            delta_value = self._numero(fila.get("delta_value"), 0.0)
            rate_per_minute = self._numero(fila.get("rate_per_minute"), 0.0)
            base_fila = self._base_horaria(base, fila)

            if ewma is None or gap_seconds > base_fila["cusum_reset_gap_seconds"]:
                ewma = valor
                c_pos = 0.0
                c_neg = 0.0
                duracion_constante = 0.0
            prediccion = ewma
            residual = valor - prediccion

            fisico = self._score_fisico(valor, perfil)
            dentro_banda = base_fila["normal_lower"] <= valor <= base_fila["normal_upper"]

            raw_r, s_r = self._score_tasa_robusta(rate_per_minute, base_fila)
            if dentro_banda and fisico <= 0 and abs(delta_value) < base_fila["normal_delta_for_rate"]:
                s_r = 0.0

            z_residual = residual / base_fila["residual_scale"]
            if dentro_banda and abs(z_residual) < 1.5:
                c_pos = 0.0
                c_neg = 0.0
            else:
                c_pos = max(0.0, c_pos + z_residual - base_fila["cusum_k"])
                c_neg = max(0.0, c_neg - z_residual - base_fila["cusum_k"])
            raw_c = max(c_pos, c_neg)
            s_c = min(1.0, raw_c / base_fila["cusum_theta"])
            if dentro_banda and fisico <= 0:
                s_c = 0.0

            if valor_previo is not None and abs(valor - valor_previo) <= base_fila["epsilon_const"] and gap_seconds <= base_fila["gap_alert_seconds"]:
                duracion_constante += max(gap_seconds, 0.0)
            else:
                duracion_constante = 0.0
            raw_d_const, s_d_const = self._score_persistencia_constante(duracion_constante, base_fila)

            raw_m, s_m = self._score_brecha_temporal(gap_seconds, base_fila)
            if s_m > 0.0:
                if brecha_reportada:
                    raw_m, s_m = 0.0, 0.0
                brecha_reportada = True
            else:
                brecha_reportada = False
            raw_dspot, s_dspot = self._score_dspot_extremo(valor, base_fila)
            raw_h, s_h = self._score_hampel(fila, base_fila)
            if dentro_banda and fisico <= 0 and s_r <= 0.0 and s_dspot <= 0.0:
                s_h = 0.0
            if fisico > s_h:
                raw_h = max(raw_h, fisico * self.umbral_z)
                s_h = fisico

            componentes = {
                "s_H": round(s_h, 6),
                "s_R": round(s_r, 6),
                "s_C": round(s_c, 6),
                "s_D_const": round(s_d_const, 6),
                "s_M": round(s_m, 6),
                "s_DSPOT": round(s_dspot, 6),
            }
            raw_componentes = {
                "H_i": round(raw_h, 6),
                "Z_R_i": round(raw_r, 6),
                "C_residual": round(raw_c, 6),
                "Z_D_const": round(raw_d_const, 6),
                "Z_M": round(raw_m, 6),
                "Z_DSPOT": round(raw_dspot, 6),
            }
            indicadores = self._indicadores_activados(componentes, raw_threshold)
            evidence_score = max(componentes.values()) if componentes else 0.0
            if indicadores == ["reading_gap"]:
                evidence_score = min(evidence_score, 0.5)
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
                        "baseline_summary": self._resumen_base_para_salida(base_fila),
                        "ground_truth_label": None,
                        "severity": self._severidad(suspicion_score),
                    }
                )

            peso_pewma = self._peso_pewma(residual, base_fila)
            ewma = peso_pewma * valor + (1.0 - peso_pewma) * ewma
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

        historico_previo = trabajo["value"].shift(1)
        rolling = historico_previo.rolling(self.window_hampel, min_periods=max(6, self.window_hampel // 2))
        trabajo["local_median"] = rolling.median()
        trabajo["local_mad"] = rolling.apply(lambda valores: self._mad(pd.Series(valores)), raw=False)
        trabajo["local_median"] = trabajo["local_median"].fillna(base["median"])
        trabajo["local_mad"] = trabajo["local_mad"].fillna(base["mad"])
        return trabajo

    def _construir_base_normalidad(
        self,
        serie: pd.DataFrame,
        entity_id: str,
    ) -> dict[str, Any]:
        valores = pd.to_numeric(serie["value"], errors="coerce").dropna()
        if len(valores) < self.window_hampel:
            return {}

        temporal = serie[["timestamp_dt", "value"]].copy()
        temporal["hour"] = temporal["timestamp_dt"].dt.hour
        temporal["gap_seconds"] = temporal["timestamp_dt"].diff().dt.total_seconds()
        temporal["delta_value"] = temporal["value"].diff()
        temporal["rate_per_minute"] = temporal["delta_value"].abs() / (temporal["gap_seconds"].clip(lower=1.0) / 60.0)
        perfil = self._perfil_desde_historial(entity_id, valores, temporal)

        q01 = float(valores.quantile(0.01))
        q05 = float(valores.quantile(0.05))
        q25 = float(valores.quantile(0.25))
        q50 = float(valores.quantile(0.50))
        q75 = float(valores.quantile(0.75))
        q95 = float(valores.quantile(0.95))
        q99 = float(valores.quantile(0.99))
        iqr = max(q75 - q25, perfil.value_floor)
        perfil = replace(perfil, **self._limites_fisicos(entity_id, valores, iqr))

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

        gaps = temporal["gap_seconds"].dropna()
        rates = temporal.loc[temporal["gap_seconds"].fillna(0).gt(0), "rate_per_minute"].dropna()
        rate_summary = self._resumen_robusto(rates, perfil.rate_floor_per_minute)
        gap_summary = self._resumen_robusto(gaps, self._piso_robusto(gaps))

        ewma_previa = temporal["value"].ewm(alpha=self.alpha_ewma, adjust=False).mean().shift(1)
        residuales = (temporal["value"] - ewma_previa).dropna()
        residual_mad = max(self._mad(residuales), perfil.value_floor)
        hourly_baseline = self._baseline_por_hora(temporal, perfil)

        epsilon_const = max(perfil.const_epsilon_floor, min(mad * 0.03, perfil.value_floor))
        duraciones_constantes = self._duraciones_constantes(temporal, epsilon_const, perfil.min_gap_seconds)
        const_summary = self._resumen_robusto(
            pd.Series(duraciones_constantes, dtype="float64"),
            self._piso_robusto(pd.Series(duraciones_constantes, dtype="float64")),
        )
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
            "dspot_lower": normal_lower - 1.5 * iqr_limpio,
            "dspot_upper": normal_upper + 1.5 * iqr_limpio,
            "hourly_baseline": hourly_baseline,
            "rate_summary": rate_summary,
            "gap_summary": gap_summary,
            "const_summary": const_summary,
            "residual_scale": max(1.4826 * residual_mad, perfil.value_floor),
            "cusum_k": 0.5,
            "cusum_theta": 8.0,
            "cusum_reset_gap_seconds": gap_alert,
            "epsilon_const": epsilon_const,
            "gap_alert_seconds": gap_alert,
            "constant_alert_seconds": const_alert,
            "normal_delta_for_rate": max(1.4826 * mad, perfil.value_floor * 2.0),
            "perfil": perfil,
        }

    def _baseline_por_hora(self, temporal: pd.DataFrame, perfil: PerfilContinuo) -> dict[int, dict[str, float]]:
        salida: dict[int, dict[str, float]] = {}
        for hour, grupo in temporal.dropna(subset=["value", "hour"]).groupby("hour", sort=False):
            valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
            if len(valores) < self.window_hampel:
                continue
            q05 = float(valores.quantile(0.05))
            q25 = float(valores.quantile(0.25))
            q50 = float(valores.quantile(0.50))
            q75 = float(valores.quantile(0.75))
            q95 = float(valores.quantile(0.95))
            mad = max(self._mad(valores), perfil.value_floor)
            iqr = max(q75 - q25, perfil.value_floor)
            lower = q05 - 1.5 * iqr
            upper = q95 + 1.5 * iqr
            if perfil.physical_low is not None:
                lower = max(lower, perfil.physical_low)
            if perfil.physical_high is not None:
                upper = min(upper, perfil.physical_high)
            salida[int(hour)] = {
                "median": q50,
                "mad": mad,
                "scaled_mad": 1.4826 * mad,
                "normal_lower": lower,
                "normal_upper": upper,
                "dspot_lower": lower - 1.5 * iqr,
                "dspot_upper": upper + 1.5 * iqr,
                "normal_delta_for_rate": max(1.4826 * mad, perfil.value_floor * 2.0),
            }
        return salida

    def _base_horaria(self, base: dict[str, Any], fila: dict[str, Any]) -> dict[str, Any]:
        timestamp = pd.to_datetime(fila.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            return base
        hora = int(timestamp.hour)
        resumen_hora = base.get("hourly_baseline", {}).get(hora)
        if not resumen_hora:
            return base
        mezclada = dict(base)
        mezclada.update(resumen_hora)
        mezclada["hour"] = hora
        return mezclada

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

    def _score_dspot_extremo(self, valor: float, base: dict[str, Any]) -> tuple[float, float]:
        lower = float(base["dspot_lower"])
        upper = float(base["dspot_upper"])
        escala = max(float(base["scaled_mad"]), base["perfil"].value_floor)
        if valor < lower:
            z = (lower - valor) / escala
            return z, self._score_desde_z(z)
        if valor > upper:
            z = (valor - upper) / escala
            return z, self._score_desde_z(z)
        return 0.0, 0.0

    def _peso_pewma(self, residual: float, base: dict[str, Any]) -> float:
        z = abs(residual) / (float(base["residual_scale"]) + 1e-9)
        probabilidad = math.exp(-0.5 * min(z, 8.0) ** 2)
        return self.alpha_ewma * max(0.02, min(1.0, probabilidad))

    def _score_desde_z(self, z_value: float) -> float:
        if not math.isfinite(z_value) or z_value <= self.zona_muerta_z:
            return 0.0
        return max(0.0, min(1.0, (z_value - self.zona_muerta_z) / (self.umbral_z - self.zona_muerta_z)))

    def _score_fisico(self, valor: float, perfil: PerfilContinuo) -> float:
        if perfil.extreme_low is not None and valor <= perfil.extreme_low:
            return 1.0
        if perfil.extreme_high is not None and valor >= perfil.extreme_high:
            return 1.0
        limite_bajo = perfil.physical_low if perfil.physical_low is not None else valor
        limite_alto = perfil.physical_high if perfil.physical_high is not None else valor
        amplitud = max(limite_alto - limite_bajo, perfil.value_floor)
        if perfil.physical_low is not None and valor < perfil.physical_low:
            return min(1.0, 0.75 + abs(valor - perfil.physical_low) / amplitud)
        if perfil.physical_high is not None and valor > perfil.physical_high:
            return min(1.0, 0.75 + abs(valor - perfil.physical_high) / amplitud)
        return 0.0

    def _perfil_desde_historial(self, entity_id: str, valores: pd.Series, temporal: pd.DataFrame) -> PerfilContinuo:
        q25 = float(valores.quantile(0.25))
        q50 = float(valores.quantile(0.50))
        q75 = float(valores.quantile(0.75))
        iqr = abs(q75 - q25)
        deltas = pd.to_numeric(temporal["delta_value"], errors="coerce").abs().dropna()
        limite_delta_central = self._percentil_o_cero(deltas, 0.95)
        deltas_no_cero = deltas[deltas.gt(0) & deltas.le(limite_delta_central)]
        gaps = pd.to_numeric(temporal["gap_seconds"], errors="coerce").dropna()
        gaps = gaps[gaps.gt(0)]
        rates = pd.to_numeric(temporal["rate_per_minute"], errors="coerce").dropna()
        rates = rates[rates.gt(0)]

        value_floor = max(
            self._percentil_o_cero(deltas_no_cero, 0.10),
            iqr * 0.01,
            abs(q50) * 0.001,
            1e-9,
        )
        rate_floor = max(
            self._mad(rates),
            self._percentil_o_cero(rates, 0.10),
            1e-9,
        )
        const_epsilon = max(min(self._percentil_o_cero(deltas_no_cero, 0.05), value_floor) * 0.25, 1e-9)
        gap_median = self._percentil_o_cero(gaps, 0.50)
        gap_p95 = self._percentil_o_cero(gaps, 0.95)
        gap_p99 = self._percentil_o_cero(gaps, 0.99)

        perfil = PerfilContinuo(
            value_floor=value_floor,
            rate_floor_per_minute=rate_floor,
            const_epsilon_floor=const_epsilon,
            min_constant_seconds=max(gap_p95 * 10.0, gap_median),
            min_gap_seconds=max(gap_p99, gap_median),
        )
        return self._perfil_configurado(entity_id, perfil)

    def _perfil_configurado(self, entity_id: str, perfil: PerfilContinuo) -> PerfilContinuo:
        seccion = f"continuous.{entity_id}"
        claves = [
            "value_floor",
            "rate_floor_per_minute",
            "const_epsilon_floor",
            "min_constant_seconds",
            "min_gap_seconds",
        ]
        if self.config.has_section(seccion) and self.config.get(seccion, "origen", fallback="") == "editado_por_usuario":
            try:
                valores = {clave: self.config.getfloat(seccion, clave) for clave in claves}
                return replace(perfil, **valores)
            except (Error, TypeError, ValueError):
                pass
        if not self.config.has_section(seccion):
            self.config.add_section(seccion)
            self.config.set(seccion, "origen", "calculado_desde_historial")
        for clave in claves:
            self.config.set(seccion, clave, f"{getattr(perfil, clave):.6f}")
        self._guardar_config()
        return perfil

    def _limites_fisicos(self, entity_id: str, valores: pd.Series, iqr: float) -> dict[str, float]:
        seccion = f"continuous.{entity_id}"
        if self.config.has_section(seccion):
            try:
                return {
                    "physical_low": self.config.getfloat(seccion, "physical_low"),
                    "physical_high": self.config.getfloat(seccion, "physical_high"),
                    "extreme_low": self.config.getfloat(seccion, "extreme_low"),
                    "extreme_high": self.config.getfloat(seccion, "extreme_high"),
                }
            except (Error, TypeError, ValueError):
                self.config.remove_section(seccion)

        q01 = float(valores.quantile(0.01))
        q05 = float(valores.quantile(0.05))
        q95 = float(valores.quantile(0.95))
        q99 = float(valores.quantile(0.99))
        limites = {
            "physical_low": q01 - 3.0 * iqr,
            "physical_high": q99 + 3.0 * iqr,
            "extreme_low": q05 - 1.5 * iqr,
            "extreme_high": q95 + 1.5 * iqr,
        }
        if not self.config.has_section(seccion):
            self.config.add_section(seccion)
        self.config.set(seccion, "origen", "calculado_desde_historial")
        for clave, valor in limites.items():
            self.config.set(seccion, clave, f"{valor:.6f}")
        self._guardar_config()
        return limites

    def configuracion_continua(self, eventos: pd.DataFrame, entidades: pd.DataFrame) -> list[dict[str, Any]]:
        if eventos.empty or "sensor_kind" not in eventos.columns:
            return []
        trabajo = eventos[eventos["sensor_kind"].eq("continuous")].copy()
        lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
        salida = []
        for entity_id, grupo in trabajo.groupby("entity_id", sort=True):
            serie = grupo.sort_values("timestamp_dt").copy()
            serie["value"] = pd.to_numeric(serie["value"], errors="coerce")
            serie = serie[serie["value"].notna() & serie["timestamp_dt"].notna()].copy()
            if len(serie) < self.window_hampel:
                continue
            base = self._construir_base_normalidad(serie, str(entity_id))
            entidad = lookup.get(str(entity_id), {})
            salida.append(
                {
                    "entity_id": str(entity_id),
                    "device_class": str(entidad.get("device_class") or ""),
                    "unit_of_measurement": str(entidad.get("unit_of_measurement") or ""),
                    "values": self._resumen_base_para_salida(base),
                }
            )
        return salida

    def guardar_configuracion(self, configuraciones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        claves = [
            "value_floor",
            "rate_floor_per_minute",
            "const_epsilon_floor",
            "min_constant_seconds",
            "min_gap_seconds",
            "physical_low",
            "physical_high",
            "extreme_low",
            "extreme_high",
        ]
        for item in configuraciones:
            entity_id = str(item.get("entity_id") or "").strip()
            if not entity_id:
                continue
            seccion = f"continuous.{entity_id}"
            if not self.config.has_section(seccion):
                self.config.add_section(seccion)
            self.config.set(seccion, "origen", "editado_por_usuario")
            valores = item.get("values") or {}
            for clave in claves:
                if clave in valores:
                    self.config.set(seccion, clave, f"{float(valores[clave]):.6f}")
        self._guardar_config()
        return configuraciones

    def _guardar_config(self) -> None:
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conf_path.open("w", encoding="utf-8") as archivo:
            self.config.write(archivo)

    def _indicadores_activados(self, componentes: dict[str, float], raw_threshold: float) -> list[str]:
        mapa = {
            "s_H": "hampel",
            "s_R": "robust_rate_change",
            "s_C": "cusum_residual",
            "s_D_const": "constant_persistence",
            "s_M": "reading_gap",
            "s_DSPOT": "dspot_extreme",
        }
        return [indicador for clave, indicador in mapa.items() if componentes.get(clave, 0.0) >= raw_threshold]

    def _indicador_principal(self, componentes: dict[str, float], indicadores: list[str]) -> str:
        componente_por_indicador = {
            "hampel": componentes.get("s_H", 0.0),
            "robust_rate_change": componentes.get("s_R", 0.0),
            "cusum_residual": componentes.get("s_C", 0.0),
            "constant_persistence": componentes.get("s_D_const", 0.0),
            "reading_gap": componentes.get("s_M", 0.0),
            "dspot_extreme": componentes.get("s_DSPOT", 0.0),
        }
        return max(indicadores, key=lambda indicador: componente_por_indicador.get(indicador, 0.0))

    def _raw_dominante(self, raw_componentes: dict[str, float], indicador: str) -> float:
        mapa = {
            "hampel": "H_i",
            "robust_rate_change": "Z_R_i",
            "cusum_residual": "C_residual",
            "constant_persistence": "Z_D_const",
            "reading_gap": "Z_M",
            "dspot_extreme": "Z_DSPOT",
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
            mejor = max(grupo, key=self._ranking_sospecha)
            if len(grupo) > 1:
                mejor = dict(mejor)
                mejor["suppressed_events"] = len(grupo) - 1
            representativos.append(mejor)
        return sorted(representativos, key=lambda item: item["suspicion_score"], reverse=True)

    def _ranking_sospecha(self, item: dict[str, Any]) -> tuple[float, float, float, float]:
        lectura = item.get("reading") or {}
        raw_componentes = lectura.get("raw_components") or {}
        raw_maximo = max([float(valor) for valor in raw_componentes.values()] or [0.0])
        return (
            float(item.get("suspicion_score") or 0.0),
            float(item.get("evidence_score") or 0.0),
            raw_maximo,
            abs(float(lectura.get("delta_value") or 0.0)),
        )

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

    def _piso_robusto(self, valores: pd.Series) -> float:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        limpio = limpio[limpio.gt(0)]
        if limpio.empty:
            return 1e-9
        return max(self._mad(limpio), float(limpio.quantile(0.10)), 1e-9)

    def _percentil_o_cero(self, valores: pd.Series, percentil: float) -> float:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        return float(limpio.quantile(percentil)) if not limpio.empty else 0.0

    def _resumen_base_para_salida(self, base: dict[str, Any]) -> dict[str, Any]:
        return {
            "median": round(base["median"], 6),
            "mad": round(base["mad"], 6),
            "normal_lower": round(base["normal_lower"], 6),
            "normal_upper": round(base["normal_upper"], 6),
            "dspot_lower": round(base["dspot_lower"], 6),
            "dspot_upper": round(base["dspot_upper"], 6),
            "physical_low": round(base["perfil"].physical_low, 6),
            "physical_high": round(base["perfil"].physical_high, 6),
            "extreme_low": round(base["perfil"].extreme_low, 6),
            "extreme_high": round(base["perfil"].extreme_high, 6),
            "hour": base.get("hour", "global"),
            "rate_median": round(base["rate_summary"]["median"], 6),
            "rate_mad": round(base["rate_summary"]["mad"], 6),
            "gap_alert_seconds": round(base["gap_alert_seconds"], 3),
            "constant_alert_seconds": round(base["constant_alert_seconds"], 3),
            "epsilon_const": round(base["epsilon_const"], 6),
            "value_floor": round(base["perfil"].value_floor, 6),
            "rate_floor_per_minute": round(base["perfil"].rate_floor_per_minute, 6),
            "const_epsilon_floor": round(base["perfil"].const_epsilon_floor, 6),
            "min_gap_seconds": round(base["perfil"].min_gap_seconds, 3),
            "min_constant_seconds": round(base["perfil"].min_constant_seconds, 3),
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
