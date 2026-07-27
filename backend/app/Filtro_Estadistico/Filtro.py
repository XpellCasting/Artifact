from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd

from .Continuos import DETALLE_INDICADOR as DETALLE_CONTINUO
from .Continuos import INDICADORES_CONTINUOS, FiltroContinuoEstadistico
from .Estados_Binarios import DETALLE_INDICADOR as DETALLE_BINARIO
from .Estados_Binarios import INDICADORES_BINARIOS, FiltroEstadoBinarioEstadistico


class FiltroEstadistico:
    """Orquesta los filtros estadisticos por tipo de entidad.

    En esta etapa se ejecutan entidades continuas y de estado binario. El filtro
    multivalorico queda fuera del alcance del paso actual.
    """

    def __init__(self) -> None:
        self.continuos = FiltroContinuoEstadistico()
        self.binarios = FiltroEstadoBinarioEstadistico()
        self.umbral_activacion_lapsos = 0.35

    def analizar_historial(
        self,
        eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        suspicion_threshold: float = 0.6,
    ) -> dict[str, Any]:
        inicio_total = perf_counter()
        umbral_local = min(float(suspicion_threshold), self.umbral_activacion_lapsos)
        inicio_continuos = perf_counter()
        resultados_continuos = self.continuos.analizar(
            eventos=eventos,
            entidades=entidades,
            suspicion_threshold=umbral_local,
        )
        tiempo_continuos_ms = (perf_counter() - inicio_continuos) * 1000.0
        inicio_binarios = perf_counter()
        resultados_binarios = self.binarios.analizar(
            eventos=eventos,
            entidades=entidades,
            suspicion_threshold=umbral_local,
        )
        tiempo_binarios_ms = (perf_counter() - inicio_binarios) * 1000.0
        elapsed_ms = (perf_counter() - inicio_total) * 1000.0

        resultados = self._aplicar_lapsos_temporales(
            eventos=eventos,
            resultados=resultados_continuos + resultados_binarios,
            suspicion_threshold=suspicion_threshold,
        )
        resultados = sorted(resultados, key=self._clave_fecha_resultado, reverse=True)
        top_results = resultados[:1000]
        resumen = {
            "results": len(resultados),
            "suspicious": sum(1 for resultado in resultados if resultado.get("is_suspicious")),
            "by_kind": {
                "continuous": sum(1 for resultado in resultados if resultado.get("sensor_kind") == "continuous"),
                "binary": sum(1 for resultado in resultados if resultado.get("sensor_kind") == "binary"),
            },
            "top_results": top_results,
            "elapsed_ms": round(elapsed_ms, 3),
            "elapsed_seconds": round(elapsed_ms / 1000.0, 6),
            "timings_by_kind_ms": {
                "continuous": round(tiempo_continuos_ms, 3),
                "binary": round(tiempo_binarios_ms, 3),
            },
            "first_detection_delay_seconds": None,
            "implemented_kinds": ["continuous", "binary"],
            "pending_kinds": ["multistate"],
            "temporal_windows": ["local", "hour_of_day", "month_repetition"],
            "temporal_activation_threshold": self.umbral_activacion_lapsos,
        }
        return {
            **resumen,
            "result_count": len(resultados),
            "results": top_results,
            "top_results": top_results,
            "all_results": resultados,
            "summary": resumen,
        }

    def construir_baseline_realtime(self, eventos: pd.DataFrame, entidades: pd.DataFrame) -> dict[str, Any]:
        if eventos.empty or "sensor_kind" not in eventos.columns:
            return {"continuous": {}, "binary": {}, "context": {}}
        lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
        baseline: dict[str, Any] = {"continuous": {}, "binary": {}, "context": {}}
        trabajo = eventos.copy()
        trabajo["timestamp_dt"] = pd.to_datetime(trabajo["timestamp_dt"], errors="coerce", utc=True)
        trabajo["hour"] = trabajo["timestamp_dt"].dt.hour

        for entity_id, grupo in trabajo[trabajo["sensor_kind"].eq("continuous")].groupby("entity_id", sort=False):
            serie = grupo.sort_values("timestamp_dt").copy()
            serie["value"] = pd.to_numeric(serie["value"], errors="coerce")
            serie = serie[serie["value"].notna() & serie["timestamp_dt"].notna()].copy()
            base = self.continuos._construir_base_normalidad(serie, str(entity_id))
            if base:
                baseline["continuous"][str(entity_id)] = base

        for entity_id, grupo in trabajo[trabajo["sensor_kind"].eq("binary")].groupby("entity_id", sort=False):
            entidad = lookup.get(str(entity_id), {})
            serie = grupo.sort_values("timestamp_dt").copy()
            serie["value"] = pd.to_numeric(serie["value"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
            serie = serie[serie["timestamp_dt"].notna()].copy()
            if len(serie) < self.binarios.min_history_events:
                continue
            perfil = self.binarios._perfil_binario(str(entity_id), str(entidad.get("device_class", "")))
            features = self.binarios._extraer_features(serie, perfil)
            if not features.empty:
                baseline["binary"][str(entity_id)] = {
                    "perfil": perfil,
                    "base": self.binarios._construir_base_normalidad(features, perfil),
                }

        baseline["context"] = {
            "continuous": self._baseline_horario_continuo(trabajo),
            "binary": self._baseline_horario_binario(trabajo),
            "repeated_continuous": self._excursiones_repetidas_continuas(trabajo),
        }
        return baseline

    def analizar_realtime(
        self,
        eventos_historial: pd.DataFrame,
        eventos_realtime: pd.DataFrame,
        nuevos_eventos: pd.DataFrame,
        entidades: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float = 0.6,
    ) -> dict[str, Any]:
        inicio_total = perf_counter()
        if nuevos_eventos.empty:
            return self._resultado_realtime_vacio(inicio_total, 0, 0)

        eventos = pd.concat([eventos_historial, eventos_realtime], ignore_index=True) if not eventos_realtime.empty else eventos_historial
        lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
        umbral_local = min(float(suspicion_threshold), self.umbral_activacion_lapsos)
        inicio_continuos = perf_counter()
        resultados_continuos = self._analizar_continuos_realtime(eventos, nuevos_eventos, baseline, umbral_local)
        tiempo_continuos_ms = (perf_counter() - inicio_continuos) * 1000.0
        inicio_binarios = perf_counter()
        resultados_binarios = self._analizar_binarios_realtime(eventos, nuevos_eventos, baseline, lookup, umbral_local)
        tiempo_binarios_ms = (perf_counter() - inicio_binarios) * 1000.0
        resultados = self._aplicar_lapsos_temporales_realtime(
            resultados_continuos + resultados_binarios,
            baseline.get("context", {}),
            suspicion_threshold,
        )
        resultados = sorted(resultados, key=self._clave_fecha_resultado, reverse=True)
        elapsed_ms = (perf_counter() - inicio_total) * 1000.0
        resumen = {
            "results": len(resultados),
            "suspicious": sum(1 for resultado in resultados if resultado.get("is_suspicious")),
            "by_kind": {
                "continuous": sum(1 for resultado in resultados if resultado.get("sensor_kind") == "continuous"),
                "binary": sum(1 for resultado in resultados if resultado.get("sensor_kind") == "binary"),
            },
            "top_results": resultados[:1000],
            "elapsed_ms": round(elapsed_ms, 3),
            "elapsed_seconds": round(elapsed_ms / 1000.0, 6),
            "timings_by_kind_ms": {
                "continuous": round(tiempo_continuos_ms, 3),
                "binary": round(tiempo_binarios_ms, 3),
            },
            "first_detection_delay_seconds": None,
            "implemented_kinds": ["continuous", "binary"],
            "pending_kinds": ["multistate"],
            "temporal_windows": ["local", "hour_of_day", "month_repetition"],
            "temporal_activation_threshold": self.umbral_activacion_lapsos,
            "incremental": True,
        }
        return {
            **resumen,
            "result_count": len(resultados),
            "results": resultados[:1000],
            "top_results": resultados[:1000],
            "all_results": resultados,
            "summary": resumen,
        }

    def _resultado_realtime_vacio(self, inicio_total: float, continuos_ms: float, binarios_ms: float) -> dict[str, Any]:
        elapsed_ms = (perf_counter() - inicio_total) * 1000.0
        resumen = {
            "results": 0,
            "suspicious": 0,
            "by_kind": {"continuous": 0, "binary": 0},
            "top_results": [],
            "elapsed_ms": round(elapsed_ms, 3),
            "elapsed_seconds": round(elapsed_ms / 1000.0, 6),
            "timings_by_kind_ms": {"continuous": round(continuos_ms, 3), "binary": round(binarios_ms, 3)},
            "first_detection_delay_seconds": None,
            "implemented_kinds": ["continuous", "binary"],
            "pending_kinds": ["multistate"],
            "incremental": True,
            "skipped_analysis": True,
        }
        return {**resumen, "result_count": 0, "results": [], "top_results": [], "all_results": [], "summary": resumen}

    def _analizar_continuos_realtime(
        self,
        eventos: pd.DataFrame,
        nuevos_eventos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        nuevos = nuevos_eventos[nuevos_eventos.get("sensor_kind", "").eq("continuous")] if "sensor_kind" in nuevos_eventos else pd.DataFrame()
        if nuevos.empty:
            return []
        raw_threshold = self.continuos._probabilidad_a_raw(suspicion_threshold)
        resultados: list[dict[str, Any]] = []
        bases = baseline.get("continuous", {})
        for entity_id, filas_nuevas in nuevos.groupby("entity_id", sort=False):
            base = bases.get(str(entity_id))
            if not base:
                continue
            grupo = eventos[eventos["entity_id"].eq(entity_id)].sort_values("timestamp_dt").copy()
            grupo["value"] = pd.to_numeric(grupo["value"], errors="coerce")
            grupo = grupo[grupo["value"].notna() & grupo["timestamp_dt"].notna()]
            for fila in filas_nuevas.sort_values("timestamp_dt").to_dict("records"):
                timestamp = pd.to_datetime(fila.get("timestamp_dt"), errors="coerce", utc=True)
                if pd.isna(timestamp):
                    continue
                valor = float(pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0])
                previos = grupo[grupo["timestamp_dt"].lt(timestamp)]
                previo = previos.tail(1)
                gap_seconds = 0.0 if previo.empty else max(0.0, (timestamp - previo.iloc[-1]["timestamp_dt"]).total_seconds())
                delta_value = 0.0 if previo.empty else valor - float(previo.iloc[-1]["value"])
                rate_per_minute = abs(delta_value) / max(gap_seconds / 60.0, 1.0 / 60.0)
                locales = previos.tail(self.continuos.window_hampel)["value"]
                local_median = float(locales.median()) if not locales.empty else float(base["median"])
                local_mad = max(self.continuos._mad(locales), float(base["mad"])) if not locales.empty else float(base["mad"])
                fila_score = {
                    **fila,
                    "value": valor,
                    "timestamp": str(fila.get("timestamp") or timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")),
                    "local_median": local_median,
                    "local_mad": local_mad,
                }
                base_fila = self.continuos._base_horaria(base, fila_score)
                fisico = self.continuos._score_fisico(valor, base_fila["perfil"])
                dentro_banda = base_fila["normal_lower"] <= valor <= base_fila["normal_upper"]
                raw_h, s_h = self.continuos._score_hampel(fila_score, base_fila)
                raw_r, s_r = self.continuos._score_tasa_robusta(rate_per_minute, base_fila)
                raw_m, s_m = self.continuos._score_brecha_temporal(gap_seconds, base_fila)
                raw_dspot, s_dspot = self.continuos._score_dspot_extremo(valor, base_fila)
                residual = valor - float(base_fila["median"])
                raw_c = abs(residual) / (float(base_fila["residual_scale"]) + 1e-9)
                s_c = 0.0 if dentro_banda else self.continuos._score_desde_z(raw_c)
                if dentro_banda and fisico <= 0 and abs(delta_value) < base_fila["normal_delta_for_rate"]:
                    s_r = 0.0
                if dentro_banda and fisico <= 0 and s_r <= 0.0 and s_dspot <= 0.0:
                    s_h = 0.0
                if fisico > s_h:
                    raw_h = max(raw_h, fisico * self.continuos.umbral_z)
                    s_h = fisico
                componentes = {
                    "s_H": round(s_h, 6),
                    "s_R": round(s_r, 6),
                    "s_C": round(s_c, 6),
                    "s_D_const": 0.0,
                    "s_M": round(s_m, 6),
                    "s_DSPOT": round(s_dspot, 6),
                }
                raw_componentes = {
                    "H_i": round(raw_h, 6),
                    "Z_R_i": round(raw_r, 6),
                    "C_residual": round(raw_c, 6),
                    "Z_D_const": 0.0,
                    "Z_M": round(raw_m, 6),
                    "Z_DSPOT": round(raw_dspot, 6),
                }
                indicadores = self.continuos._indicadores_activados(componentes, raw_threshold)
                evidence_score = max(componentes.values()) if componentes else 0.0
                suspicion_score = self.continuos._score_a_probabilidad(evidence_score)
                if indicadores and suspicion_score >= suspicion_threshold:
                    indicador_principal = self.continuos._indicador_principal(componentes, indicadores)
                    patron = INDICADORES_CONTINUOS[indicador_principal]
                    resultados.append(
                        {
                            "entity_id": str(entity_id),
                            "timestamp": fila_score["timestamp"],
                            "window_start": fila_score["timestamp"],
                            "window_end": fila_score["timestamp"],
                            "sensor_kind": "continuous",
                            "tipo_interno": "continua",
                            "observed_state": str(fila.get("state", valor)),
                            "value": round(valor, 6),
                            "raw_score": self.continuos._raw_dominante(raw_componentes, indicador_principal),
                            "evidence_score": round(evidence_score, 6),
                            "suspicion_score": round(suspicion_score, 6),
                            "is_suspicious": True,
                            "pattern": patron,
                            "suspected_pattern": patron,
                            "pattern_detail": ", ".join(DETALLE_CONTINUO[indicador] for indicador in indicadores),
                            "filter": "statistical_temporal",
                            "passed_to_contextual_filter": True,
                            "triggered_indicators": indicadores,
                            "reading": {
                                "value": round(valor, 6),
                                "delta_value": round(delta_value, 6),
                                "rate_per_minute": round(rate_per_minute, 6),
                                "gap_seconds": round(gap_seconds, 3),
                                "constant_duration_seconds": 0.0,
                                "residual": round(residual, 6),
                                "components": componentes,
                                "raw_components": raw_componentes,
                            },
                            "baseline_summary": self.continuos._resumen_base_para_salida(base_fila),
                            "ground_truth_label": None,
                            "severity": self.continuos._severidad(suspicion_score),
                        }
                    )
        return resultados

    def _analizar_binarios_realtime(
        self,
        eventos: pd.DataFrame,
        nuevos_eventos: pd.DataFrame,
        baseline: dict[str, Any],
        lookup_entidades: dict[str, dict[str, Any]],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        nuevos = nuevos_eventos[nuevos_eventos.get("sensor_kind", "").eq("binary")] if "sensor_kind" in nuevos_eventos else pd.DataFrame()
        if nuevos.empty:
            return []
        raw_threshold = self.binarios._probabilidad_a_evidencia(suspicion_threshold)
        resultados: list[dict[str, Any]] = []
        bases = baseline.get("binary", {})
        for entity_id, filas_nuevas in nuevos.groupby("entity_id", sort=False):
            cached = bases.get(str(entity_id))
            entidad = lookup_entidades.get(str(entity_id), {})
            perfil = cached["perfil"] if cached else self.binarios._perfil_binario(str(entity_id), str(entidad.get("device_class", "")))
            base = cached["base"] if cached else None
            if not base:
                continue
            min_ts = pd.to_datetime(filas_nuevas["timestamp_dt"], errors="coerce", utc=True).min()
            desde = min_ts - pd.Timedelta(seconds=perfil.window_seconds * 2) if pd.notna(min_ts) else None
            grupo = eventos[eventos["entity_id"].eq(entity_id)].sort_values("timestamp_dt").copy()
            if desde is not None:
                grupo = grupo[grupo["timestamp_dt"].ge(desde)]
            grupo["value"] = pd.to_numeric(grupo["value"], errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
            features = self.binarios._extraer_features(grupo, perfil)
            if features.empty:
                continue
            nuevos_ts = {str(pd.to_datetime(valor, errors="coerce", utc=True)) for valor in filas_nuevas["timestamp_dt"]}
            for fila in features[features["timestamp_dt"].map(lambda valor: str(valor) in nuevos_ts)].to_dict("records"):
                raw_componentes, componentes, evidence_score = self.binarios._puntuar_doble_horizonte(fila, base)
                suspicion_score = self.binarios._score_a_probabilidad(evidence_score)
                indicadores = self.binarios._indicadores_activados(componentes)
                if indicadores and evidence_score >= raw_threshold and suspicion_score >= suspicion_threshold:
                    indicador_principal = self.binarios._indicador_principal(componentes, indicadores)
                    patron = INDICADORES_BINARIOS[indicador_principal]
                    resultados.append(
                        {
                            "entity_id": str(entity_id),
                            "timestamp": str(fila["timestamp"]),
                            "window_start": str(fila["window_start"]),
                            "window_end": str(fila["timestamp"]),
                            "sensor_kind": "binary",
                            "tipo_interno": "estado_binario",
                            "observed_state": str(fila["state"]),
                            "value": float(fila["value"]),
                            "raw_score": self.binarios._raw_dominante(raw_componentes, indicador_principal),
                            "evidence_score": round(evidence_score, 6),
                            "suspicion_score": round(suspicion_score, 6),
                            "is_suspicious": True,
                            "pattern": patron,
                            "suspected_pattern": patron,
                            "pattern_detail": ", ".join(DETALLE_BINARIO[indicador] for indicador in indicadores),
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
                            "baseline_summary": self.binarios._resumen_base_para_salida(base),
                            "ground_truth_label": None,
                            "severity": self.binarios._severidad(suspicion_score),
                        }
                    )
        return resultados

    def _aplicar_lapsos_temporales_realtime(
        self,
        resultados: list[dict[str, Any]],
        context: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        if not resultados:
            return []
        filtrados = []
        for resultado in resultados:
            enriquecido = dict(resultado)
            score_contexto = self._score_contexto_temporal(
                enriquecido,
                context.get("continuous", {}),
                context.get("binary", {}),
            )
            repeticion = self._repeticion_continua(enriquecido, context.get("repeated_continuous", {}))
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
            return self._score_desde_z(z)
        if resultado.get("sensor_kind") == "binary" and clave in binarios:
            base = binarios[clave]
            ratio = float((resultado.get("reading") or {}).get("active_ratio") or resultado.get("value") or 0.0)
            z = abs(ratio - base["mean"]) / (base["std"] + 1e-9)
            return self._score_desde_z(z)
        return 0.0

    def _score_desde_z(self, z_value: float) -> float:
        if z_value <= 2.5:
            return 0.0
        return max(0.0, min(1.0, (z_value - 2.5) / 1.0))
