from __future__ import annotations
from app.config import setting
from app.config import CACHE_DIR, DATA_DIR

import math
import os
import sys
from configparser import ConfigParser, Error
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

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
    """Configuración y comprobación de constancia que acompañan a FOCuS."""

    SCORE_VERSION = "statistical-continuous-score-v2.0.0"
    SCORE_KIND = "statistical_evidence_strength_not_probability"
    min_history_events = 12
    zona_muerta_z = 2.5
    ancla_activacion_z = 3.1
    score_ancla_activacion = 0.6
    lambda_log = (
        math.log(2.0)
        * (1.0 - score_ancla_activacion)
        / score_ancla_activacion
    )
    supresion_segundos = 15 * 60

    def __init__(self, conf_path: Path | None = None) -> None:
        self.conf_path = conf_path or Path(setting('CONTINUOUS_CONF_PATH'))
        if conf_path is None and not self.conf_path.exists() and (DATA_DIR / "defaults/conf.txt").is_file():
            self.conf_path.parent.mkdir(parents=True, exist_ok=True)
            self.conf_path.write_bytes((DATA_DIR / "defaults/conf.txt").read_bytes())
        self.config = ConfigParser()
        self.config.optionxform = str
        self._config_loaded_path = self.conf_path
        if self.conf_path.exists():
            self.config.read(self.conf_path, encoding="utf-8")

    def _sincronizar_config(self) -> None:
        if self._config_loaded_path == self.conf_path:
            return
        self.config = ConfigParser()
        self.config.optionxform = str
        if self.conf_path.exists():
            self.config.read(self.conf_path, encoding="utf-8")
        self._config_loaded_path = self.conf_path

    def ajustes_manuales(self, entity_ids: set[str] | None = None) -> dict[str, dict[str, float]]:
        self._sincronizar_config()
        salida = {}
        for seccion in self.config.sections():
            if not seccion.startswith("continuous.") or self.config.get(seccion, "origen", fallback="") != "editado_por_usuario":
                continue
            entity_id = seccion.removeprefix("continuous.")
            if entity_ids is not None and entity_id not in entity_ids:
                continue
            valores = {}
            for clave in self._claves_configurables():
                if not self.config.has_option(seccion, clave):
                    continue
                try:
                    valor = self.config.getfloat(seccion, clave)
                except (Error, TypeError, ValueError) as exc:
                    raise ValueError(f"Ajuste manual inválido para {entity_id}.{clave}") from exc
                if not math.isfinite(valor):
                    raise ValueError(f"Ajuste manual no finito para {entity_id}.{clave}")
                if clave in {"value_floor", "rate_floor_per_minute", "const_epsilon_floor", "min_constant_seconds", "min_gap_seconds"} and valor <= 0:
                    raise ValueError(f"Ajuste manual debe ser positivo: {entity_id}.{clave}")
                valores[clave] = valor
            for bajo, alto in (("physical_low", "physical_high"), ("extreme_low", "extreme_high")):
                if bajo in valores and alto in valores and valores[bajo] >= valores[alto]:
                    raise ValueError(f"Límites incompatibles para {entity_id}: {bajo} debe ser menor que {alto}")
            salida[entity_id] = valores
        return salida

    @staticmethod
    def _claves_configurables() -> tuple[str, ...]:
        return ("value_floor", "rate_floor_per_minute", "const_epsilon_floor", "min_constant_seconds", "min_gap_seconds", "physical_low", "physical_high", "extreme_low", "extreme_high")

    def especificacion_puntaje(self) -> dict[str, Any]:
        return {"version": "focus-residual-v1", "kind": "calibrated_evidence_not_probability",
                "continuous_method": "residual_focus_v1", "library": "focus-cpt==0.1.10",
                "channels": ["point_residual", "focus", "residual_energy"],
                "independent_check": "constant_persistence"}

    def _construir_base_normalidad(
        self,
        serie: pd.DataFrame,
        entity_id: str,
        *,
        persistir_perfil_calculado: bool = True,
    ) -> dict[str, Any]:
        valores = pd.to_numeric(serie["value"], errors="coerce").dropna()
        if len(valores) < self.min_history_events:
            return {}

        temporal = serie[["timestamp_dt", "value"]].copy()
        temporal["hour"] = temporal["timestamp_dt"].dt.hour
        temporal["gap_seconds"] = temporal["timestamp_dt"].diff().dt.total_seconds()
        temporal["delta_value"] = temporal["value"].diff()
        temporal["rate_per_minute"] = temporal["delta_value"].abs() / (temporal["gap_seconds"].clip(lower=1.0) / 60.0)
        perfil = self._perfil_desde_historial(
            entity_id,
            valores,
            temporal,
            persistir_perfil_calculado=persistir_perfil_calculado,
        )

        q01 = float(valores.quantile(0.01))
        q05 = float(valores.quantile(0.05))
        q25 = float(valores.quantile(0.25))
        q50 = float(valores.quantile(0.50))
        q75 = float(valores.quantile(0.75))
        q95 = float(valores.quantile(0.95))
        q99 = float(valores.quantile(0.99))
        iqr = max(q75 - q25, perfil.value_floor)
        perfil = replace(
            perfil,
            **self._limites_fisicos(
                entity_id,
                valores,
                iqr,
                persistir_perfil_calculado=persistir_perfil_calculado,
            ),
        )

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

        gap_alert = max(
            perfil.min_gap_seconds,
            gap_summary["median"] + 8.0 * gap_summary["scaled_mad"],
            gap_summary["p99"] * 2.0,
        )
        hourly_baseline = self._baseline_por_hora(temporal, perfil)

        epsilon_const = max(perfil.const_epsilon_floor, min(mad * 0.03, perfil.value_floor))
        duraciones_constantes = self._duraciones_constantes(temporal, epsilon_const, perfil.min_gap_seconds)
        const_summary = self._resumen_robusto(
            pd.Series(duraciones_constantes, dtype="float64"),
            self._piso_robusto(pd.Series(duraciones_constantes, dtype="float64")),
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
            "hourly_baseline": hourly_baseline,
            "rate_summary": rate_summary,
            "gap_summary": gap_summary,
            "const_summary": const_summary,
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
            if len(valores) < self.min_history_events:
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

    def _estado_constancia_vacio(self) -> dict[str, float | None]:
        return {
            "min_value": None,
            "max_value": None,
            "duration_seconds": 0.0,
        }

    def _actualizar_estado_constancia(
        self,
        estado: dict[str, float | None],
        valor: float,
        gap_seconds: float,
        epsilon: float,
        gap_max: float,
    ) -> tuple[dict[str, float | None], float | None]:
        min_anterior = estado.get("min_value")
        max_anterior = estado.get("max_value")
        duracion_anterior = float(estado.get("duration_seconds") or 0.0)
        estado_inicial = {
            "min_value": float(valor),
            "max_value": float(valor),
            "duration_seconds": 0.0,
        }
        if min_anterior is None or max_anterior is None:
            return estado_inicial, None
        if not math.isfinite(gap_seconds) or gap_seconds < 0.0 or gap_seconds > gap_max:
            return estado_inicial, duracion_anterior

        nuevo_minimo = min(float(min_anterior), float(valor))
        nuevo_maximo = max(float(max_anterior), float(valor))
        if nuevo_maximo - nuevo_minimo <= epsilon:
            return {
                "min_value": nuevo_minimo,
                "max_value": nuevo_maximo,
                "duration_seconds": duracion_anterior + gap_seconds,
            }, None
        return estado_inicial, duracion_anterior

    def _score_persistencia_constante(self, duracion_segundos: float, base: dict[str, Any]) -> tuple[float, float]:
        if duracion_segundos < base["constant_alert_seconds"]:
            return 0.0, 0.0
        resumen = base["const_summary"]
        z_const = abs(duracion_segundos - resumen["median"]) / (resumen["scaled_mad"] + 1e-9)
        score = max(
            self._score_desde_z(z_const),
            self._score_desde_ratio(
                duracion_segundos / base["constant_alert_seconds"]
            ),
        )
        return z_const, score

    def _score_desde_z(self, z_value: float) -> float:
        if not math.isfinite(z_value) or z_value <= self.zona_muerta_z:
            return 0.0
        ratio = (
            (z_value - self.zona_muerta_z)
            / (self.ancla_activacion_z - self.zona_muerta_z)
        )
        return self._score_desde_ratio(ratio)

    def _score_desde_ratio(self, ratio_value: float) -> float:
        """Curva logaritmica suave: ratio=1 produce evidencia 0.60."""
        try:
            ratio = float(ratio_value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(ratio) or ratio <= 0.0:
            return 0.0
        if math.isinf(ratio):
            ratio = sys.float_info.max
        log_exceso = math.log1p(ratio)
        score = log_exceso / (log_exceso + self.lambda_log)
        return max(0.0, min(math.nextafter(1.0, 0.0), score))

    def _perfil_desde_historial(
        self,
        entity_id: str,
        valores: pd.Series,
        temporal: pd.DataFrame,
        *,
        persistir_perfil_calculado: bool = True,
    ) -> PerfilContinuo:
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
        return self._perfil_configurado(
            entity_id,
            perfil,
            persistir_perfil_calculado=persistir_perfil_calculado,
        )

    def _perfil_configurado(
        self,
        entity_id: str,
        perfil: PerfilContinuo,
        *,
        persistir_perfil_calculado: bool = True,
    ) -> PerfilContinuo:
        seccion = f"continuous.{entity_id}"
        self._sincronizar_config()
        claves = [
            "value_floor",
            "rate_floor_per_minute",
            "const_epsilon_floor",
            "min_constant_seconds",
            "min_gap_seconds",
        ]
        manuales = self.ajustes_manuales({entity_id}).get(entity_id, {})
        return replace(perfil, **{clave: manuales[clave] for clave in claves if clave in manuales})

    def _limites_fisicos(
        self,
        entity_id: str,
        valores: pd.Series,
        iqr: float,
        *,
        persistir_perfil_calculado: bool = True,
    ) -> dict[str, float]:
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
        manuales = self.ajustes_manuales({entity_id}).get(entity_id, {})
        for clave in limites:
            if clave in manuales:
                limites[clave] = manuales[clave]
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
            if len(serie) < self.min_history_events:
                continue
            base = self._construir_base_normalidad(serie, str(entity_id), persistir_perfil_calculado=False)
            entidad = lookup.get(str(entity_id), {})
            manuales = self.ajustes_manuales({str(entity_id)}).get(str(entity_id), {})
            salida.append(
                {
                    "entity_id": str(entity_id),
                    "device_class": str(entidad.get("device_class") or ""),
                    "unit_of_measurement": str(entidad.get("unit_of_measurement") or ""),
                    "values": self._resumen_base_para_salida(base),
                    "provenance": {clave: ("manual" if clave in manuales else "automatic") for clave in self._claves_configurables()},
                }
            )
        return salida

    def guardar_configuracion(self, configuraciones: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._sincronizar_config()
        claves = self._claves_configurables()
        existentes = self.ajustes_manuales()
        preparados = []
        for item in configuraciones:
            entity_id = str(item.get("entity_id") or "").strip()
            if not entity_id:
                raise ValueError("Cada configuración requiere entity_id")
            valores = item.get("values") or {}
            normalizados = {}
            for clave in claves:
                if clave in valores:
                    try:
                        valor = float(valores[clave])
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Ajuste manual inválido para {entity_id}.{clave}") from exc
                    if not math.isfinite(valor):
                        raise ValueError(f"Ajuste manual no finito para {entity_id}.{clave}")
                    if clave in {"value_floor", "rate_floor_per_minute", "const_epsilon_floor", "min_constant_seconds", "min_gap_seconds"} and valor <= 0:
                        raise ValueError(f"Ajuste manual debe ser positivo: {entity_id}.{clave}")
                    normalizados[clave] = valor
            efectivos = {**existentes.get(entity_id, {}), **normalizados}
            for bajo, alto in (("physical_low", "physical_high"), ("extreme_low", "extreme_high")):
                if bajo in efectivos and alto in efectivos and efectivos[bajo] >= efectivos[alto]:
                    raise ValueError(f"Límites incompatibles para {entity_id}: {bajo} debe ser menor que {alto}")
            preparados.append((entity_id, normalizados))
        for entity_id, valores in preparados:
            seccion = f"continuous.{entity_id}"
            if not self.config.has_section(seccion):
                self.config.add_section(seccion)
            self.config.set(seccion, "origen", "editado_por_usuario")
            for clave, valor in valores.items():
                self.config.set(seccion, clave, format(valor, ".17g"))
        self._guardar_config()
        return configuraciones

    def _guardar_config(self) -> None:
        self.conf_path.parent.mkdir(parents=True, exist_ok=True)
        with self.conf_path.open("w", encoding="utf-8") as archivo:
            self.config.write(archivo)

    def _duraciones_constantes(self, temporal: pd.DataFrame, epsilon: float, gap_max: float) -> list[float]:
        duraciones: list[float] = []
        estado = self._estado_constancia_vacio()
        for fila in temporal.to_dict("records"):
            valor = float(fila["value"])
            try:
                gap = float(fila.get("gap_seconds"))
            except (TypeError, ValueError):
                gap = math.nan
            estado, duracion_cerrada = self._actualizar_estado_constancia(
                estado, valor, gap, epsilon, gap_max
            )
            if duracion_cerrada is not None and duracion_cerrada > 0.0:
                duraciones.append(duracion_cerrada)
        duracion_final = float(estado["duration_seconds"] or 0.0)
        if duracion_final > 0.0:
            duraciones.append(duracion_final)
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
        }

    def _mad(self, valores: pd.Series) -> float:
        limpio = pd.to_numeric(valores, errors="coerce").dropna()
        if limpio.empty:
            return 0.0
        mediana = limpio.median()
        return float((limpio - mediana).abs().median())

