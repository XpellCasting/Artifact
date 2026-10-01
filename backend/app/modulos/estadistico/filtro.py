from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from app.modulos.estadistico.continuos import FiltroContinuoEstadistico
from app.modulos.estadistico.continuos_temporales import FiltroContinuoTemporal
from app.modulos.estadistico.estados_binarios import FiltroEstadoBinarioEstadistico
from app.modulos.estadistico.estados_multivalor import FiltroEstadoMultivalorEstadistico


class FiltroEstadistico:
    """Orquesta los filtros estadisticos por tipo de entidad.

    Ejecuta entidades continuas, binarias y multivaloricas con reglas propias
    para cada naturaleza de dato.
    """

    def __init__(self) -> None:
        self.continuos = FiltroContinuoEstadistico()
        self.continuos_temporales = FiltroContinuoTemporal()
        self.binarios = FiltroEstadoBinarioEstadistico()
        self.multivalor = FiltroEstadoMultivalorEstadistico()
        self.umbral_activacion_lapsos = 0.35

    def analizar_historial(
        self,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        suspicion_threshold: float = 0.6,
    ) -> dict[str, Any]:
        from app.services.reproduccion import MotorReproduccionValidacion
        inicio = perf_counter()
        umbral_local = min(float(suspicion_threshold), self.umbral_activacion_lapsos)
        discrete = self.binarios.analizar(eventos, entidades, umbral_local) + self.multivalor.analizar(eventos, entidades, umbral_local)
        discrete = self._aplicar_lapsos_temporales(eventos, discrete, suspicion_threshold)
        continuous = eventos[eventos["sensor_kind"].eq("continuous")].copy()
        results = list(discrete)
        if not continuous.empty:
            continuous["timestamp_dt"] = pd.to_datetime(continuous["timestamp_dt"], errors="coerce", utc=True)
            times = continuous["timestamp_dt"].dropna().sort_values().unique()
            if len(times) >= 2:
                boundary = times[min(len(times)-1, max(1, int(len(times)*.8)))]
                development = continuous[continuous["timestamp_dt"] < boundary]
                validation = continuous[continuous["timestamp_dt"] >= boundary]
                motor = MotorReproduccionValidacion(self)
                baseline = motor.construir_baseline(development, entidades)
                results.extend(motor.analizar_lote(validation, baseline, suspicion_threshold)["all_results"])
        results = sorted(results, key=self._clave_fecha_resultado, reverse=True)
        summary = {"results": len(results), "suspicious": len(results),
                   "by_kind": {kind: sum(item.get("sensor_kind") == kind for item in results) for kind in ("continuous", "binary", "multistate")},
                   "elapsed_ms": round((perf_counter()-inicio)*1000, 3), "top_results": results[:1000],
                   "continuous_method": "residual_focus_v1", "continuous_evaluated_partition": "last_20_percent",
                   "multistate_eligibility": self.multivalor.last_eligibility,
                   "continuous_score_specification": self.continuos.especificacion_puntaje(),
                   "multistate_score_specification": self.multivalor.especificacion_puntaje()}
        return {**summary, "result_count": len(results), "results": results[:1000], "all_results": results, "summary": summary}

    def _clave_fecha_resultado(self, item: dict[str, Any]) -> tuple[pd.Timestamp, float]:
        timestamp = pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.min.tz_localize("UTC")
        return timestamp, float(item.get("suspicion_score") or 0.0)

    def _aplicar_lapsos_temporales(
        self,
        eventos: pd.DataFrame,
        resultados: list[dict[str, Any]],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        if not resultados or eventos.empty:
            return []
        trabajo = eventos.copy()
        trabajo["timestamp_dt"] = pd.to_datetime(trabajo["timestamp_dt"], errors="coerce", utc=True)
        trabajo["hour"] = trabajo["timestamp_dt"].dt.hour
        continuos = self._baseline_horario_continuo(trabajo)
        binarios = self._baseline_horario_binario(trabajo)
        repetidos_continuos = self._excursiones_repetidas_continuas(trabajo)

        filtrados = []
        for resultado in resultados:
            enriquecido = dict(resultado)
            score_contexto = self._score_contexto_temporal(enriquecido, continuos, binarios)
            repeticion = self._repeticion_continua(enriquecido, repetidos_continuos)
            local = float(enriquecido.get("suspicion_score") or 0.0)
            enriquecido["suspicion_score"] = round(local, 6)
            enriquecido["is_suspicious"] = local >= suspicion_threshold
            enriquecido["time_window"] = "month_repetition" if repeticion else ("local" if score_contexto <= local else "hour_of_day")
            enriquecido["temporal_context"] = {
                "activation_threshold": self.umbral_activacion_lapsos,
                "local_score": round(local, 6),
                "hour_of_day_score": round(score_contexto, 6),
                "temporal_consistency": self._consistencia_temporal(score_contexto, repeticion),
                **repeticion,
            }
            if enriquecido["is_suspicious"]:
                filtrados.append(enriquecido)
        return filtrados

    def _consistencia_temporal(self, score_contexto: float, repeticion: dict[str, Any]) -> str:
        if repeticion:
            return "alta"
        if score_contexto >= self.umbral_activacion_lapsos:
            return "media"
        return "baja"

    def _baseline_horario_continuo(self, eventos: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
        if "sensor_kind" not in eventos.columns:
            return {}
        trabajo = eventos[eventos["sensor_kind"].eq("continuous")].copy()
        if trabajo.empty:
            return {}
        trabajo["value"] = pd.to_numeric(trabajo["value"], errors="coerce")
        salida = {}
        for clave, grupo in trabajo.dropna(subset=["value", "hour"]).groupby(["entity_id", "hour"], sort=False):
            valores = grupo["value"]
            mediana = float(valores.median())
            mad = float((valores - mediana).abs().median())
            salida[(str(clave[0]), int(clave[1]))] = {"median": mediana, "scaled_mad": max(1.4826 * mad, 0.25)}
        return salida

    def _baseline_horario_binario(self, eventos: pd.DataFrame) -> dict[tuple[str, int], dict[str, float]]:
        if "sensor_kind" not in eventos.columns:
            return {}
        trabajo = eventos[eventos["sensor_kind"].eq("binary")].copy()
        if trabajo.empty:
            return {}
        trabajo["value"] = pd.to_numeric(trabajo["value"], errors="coerce").fillna(0.0)
        salida = {}
        for clave, grupo in trabajo.dropna(subset=["hour"]).groupby(["entity_id", "hour"], sort=False):
            promedio = float(grupo["value"].mean())
            desviacion = float(grupo["value"].std()) if len(grupo) > 1 else 0.0
            salida[(str(clave[0]), int(clave[1]))] = {"mean": promedio, "std": max(desviacion, 0.10)}
        return salida

    def _excursiones_repetidas_continuas(self, eventos: pd.DataFrame) -> dict[str, dict[str, float]]:
        if "sensor_kind" not in eventos.columns:
            return {}
        trabajo = eventos[eventos["sensor_kind"].eq("continuous")].copy()
        if trabajo.empty:
            return {}
        trabajo["value"] = pd.to_numeric(trabajo["value"], errors="coerce")
        salida = {}
        for entity_id, grupo in trabajo.dropna(subset=["value"]).sort_values("timestamp_dt").groupby("entity_id", sort=False):
            valores = grupo["value"]
            mediana = float(valores.median())
            mad = float((valores - mediana).abs().median())
            escala = max(1.4826 * mad, 0.25)
            upper = mediana + 2.5 * escala
            lower = mediana - 2.5 * escala
            high = valores.ge(upper)
            low = valores.le(lower)
            fechas = grupo["timestamp_dt"].dt.date
            # ponytail: usa el historico cargado como proxy mensual; si luego hay particiones mensuales reales, agrupar aqui por mes.
            salida[str(entity_id)] = {
                "upper": upper,
                "lower": lower,
                "high_events": float((high & ~high.shift(fill_value=False)).sum()),
                "low_events": float((low & ~low.shift(fill_value=False)).sum()),
                "high_days": float(fechas[high].nunique()),
                "low_days": float(fechas[low].nunique()),
                "scale": escala,
                "points": list(zip(fechas.astype(str), valores.astype(float))),
                "high_points": list(zip(fechas[high].astype(str), valores[high].astype(float))),
                "low_points": list(zip(fechas[low].astype(str), valores[low].astype(float))),
            }
        return salida

    def _repeticion_continua(
        self,
        resultado: dict[str, Any],
        repetidos: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        if resultado.get("sensor_kind") != "continuous":
            return {}
        base = repetidos.get(str(resultado.get("entity_id")))
        if not base:
            return {}
        valor = float(resultado.get("value") or 0.0)
        if resultado.get("suspected_pattern") == "missing_gap":
            return {}
        tolerancia_general = max(float(base.get("scale") or 0.0) * 0.5, abs(valor) * 0.01, 0.1)
        dias_generales = {fecha for fecha, punto in base.get("points", []) if abs(float(punto) - valor) <= tolerancia_general}
        if len(dias_generales) >= 3:
            return {
                "repeated_pattern": True,
                "similar_days_in_month": len(dias_generales),
                "consistency_direction": "similar_value",
            }
        if valor >= base["upper"]:
            tolerancia = max(float(base.get("scale") or 0.0) * 0.5, abs(valor) * 0.01, 0.1)
            dias = {fecha for fecha, punto in base["high_points"] if punto >= valor - tolerancia}
            if len(dias) >= 2:
                return {"repeated_pattern": True, "similar_days_in_month": len(dias), "consistency_direction": "high"}
        if valor <= base["lower"]:
            tolerancia = max(float(base.get("scale") or 0.0) * 0.5, abs(valor) * 0.01, 0.1)
            dias = {fecha for fecha, punto in base["low_points"] if punto <= valor + tolerancia}
            if len(dias) >= 2:
                return {"repeated_pattern": True, "similar_days_in_month": len(dias), "consistency_direction": "low"}
        return {}

    def _score_contexto_temporal(
        self,
        resultado: dict[str, Any],
        continuos: dict[tuple[str, int], dict[str, float]],
        binarios: dict[tuple[str, int], dict[str, float]],
    ) -> float:
        timestamp = pd.to_datetime(resultado.get("timestamp"), errors="coerce", utc=True)
        if pd.isna(timestamp):
            return 0.0
        clave = (str(resultado.get("entity_id")), int(timestamp.hour))
        if resultado.get("sensor_kind") == "continuous" and clave in continuos:
            base = continuos[clave]
            valor = float(resultado.get("value") or 0.0)
            z = abs(valor - base["median"]) / (base["scaled_mad"] + 1e-9)
            return self.continuos._score_desde_z(z)
        if resultado.get("sensor_kind") == "binary" and clave in binarios:
            base = binarios[clave]
            ratio = float((resultado.get("reading") or {}).get("active_ratio") or resultado.get("value") or 0.0)
            z = abs(ratio - base["mean"]) / (base["std"] + 1e-9)
            return self._score_contexto_binario_desde_z(z)
        return 0.0

    def _score_contexto_binario_desde_z(self, z_value: float) -> float:
        if z_value <= 2.5:
            return 0.0
        return max(0.0, min(1.0, (z_value - 2.5) / 1.0))
