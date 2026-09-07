from __future__ import annotations

from collections import Counter, deque
from time import perf_counter
from typing import Any

import pandas as pd

from .Filtro_Estadistico.Continuos import (
    DETALLE_INDICADOR as DETALLE_CONTINUO,
    INDICADORES_CONTINUOS,
)
from .Filtro_Estadistico.Estados_Binarios import (
    DETALLE_INDICADOR as DETALLE_BINARIO,
    INDICADORES_BINARIOS,
)
from .Filtro_Estadistico.Estados_Multivalor import (
    DETALLE_INDICADOR as DETALLE_MULTIVALOR,
    INDICADORES_MULTIESTADO,
)


class MotorReproduccionValidacion:
    """Evalua lotes cronologicos contra una base fija de Desarrollo.

    El motor no genera eventos. Mantiene solamente el estado causal necesario
    entre lotes para que cada lectura de Validacion se evalue sin observar filas
    futuras del mismo archivo.
    """

    VERSION = "validation-causal-replay-v1"

    def __init__(self, filtro_estadistico: Any) -> None:
        self.filtro = filtro_estadistico

    def construir_baseline(
        self,
        eventos_desarrollo: pd.DataFrame,
        entidades_desarrollo: pd.DataFrame,
        entidades_objetivo: set[str] | None = None,
    ) -> dict[str, Any]:
        if eventos_desarrollo.empty or "sensor_kind" not in eventos_desarrollo.columns:
            raise ValueError("Desarrollo no contiene eventos normalizados para construir la base.")
        trabajo = eventos_desarrollo.copy()
        trabajo["timestamp_dt"] = pd.to_datetime(
            trabajo["timestamp_dt"], errors="coerce", utc=True
        )
        if entidades_objetivo:
            trabajo = trabajo[trabajo["entity_id"].isin(entidades_objetivo)].copy()
        lookup = (
            entidades_desarrollo.set_index("entity_id").to_dict("index")
            if not entidades_desarrollo.empty
            else {}
        )
        baseline: dict[str, Any] = {
            "version": self.VERSION,
            "continuous": {},
            "binary": {},
            "multistate": {},
            "context": {},
            "last_emitted": {},
        }

        for entity_id, grupo in trabajo[
            trabajo["sensor_kind"].eq("continuous")
        ].groupby("entity_id", sort=False):
            serie = grupo.sort_values("timestamp_dt").copy()
            if "is_valid_value" in serie.columns:
                serie = serie[serie["is_valid_value"].astype(bool)].copy()
            serie["value"] = pd.to_numeric(serie["value"], errors="coerce")
            serie = serie[serie["value"].notna() & serie["timestamp_dt"].notna()]
            base = self.filtro.continuos._construir_base_normalidad(
                serie,
                str(entity_id),
                persistir_perfil_calculado=False,
            )
            if not base:
                continue
            ultima = serie.tail(1)
            ultimo_valor = None if ultima.empty else float(ultima.iloc[0]["value"])
            estado_cusum = self.filtro.continuos._estado_cusum_vacio()
            estado_cusum["ewma"] = ultimo_valor
            baseline["continuous"][str(entity_id)] = {
                "base": base,
                "runtime": {
                    "recent_values": deque(
                        serie["value"].tail(self.filtro.continuos.window_hampel).astype(float),
                        maxlen=self.filtro.continuos.window_hampel,
                    ),
                    "last_timestamp": None if ultima.empty else ultima.iloc[0]["timestamp_dt"],
                    "last_value": ultimo_valor,
                    "cusum": estado_cusum,
                    "constant_duration_seconds": 0.0,
                    "gap_reported": False,
                },
            }

        for entity_id, grupo in trabajo[
            trabajo["sensor_kind"].eq("binary")
        ].groupby("entity_id", sort=False):
            serie = grupo.sort_values("timestamp_dt").copy()
            if "is_valid_value" in serie.columns:
                serie = serie[serie["is_valid_value"].astype(bool)].copy()
            serie["value"] = (
                pd.to_numeric(serie["value"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
            serie = serie[serie["timestamp_dt"].notna()].copy()
            if len(serie) < self.filtro.binarios.min_history_events:
                continue
            entidad = lookup.get(str(entity_id), {})
            perfil = self.filtro.binarios._perfil_binario(
                str(entity_id), str(entidad.get("device_class", ""))
            )
            features = self.filtro.binarios._extraer_features(serie, perfil)
            if features.empty:
                continue
            ultima_fecha = serie["timestamp_dt"].max()
            tail = serie[
                serie["timestamp_dt"].ge(
                    ultima_fecha - pd.Timedelta(seconds=perfil.window_seconds * 2)
                )
            ].copy()
            baseline["binary"][str(entity_id)] = {
                "perfil": perfil,
                "base": self.filtro.binarios._construir_base_normalidad(
                    features, perfil
                ),
                "runtime": {"tail": tail},
            }

        multivalor = self.filtro.multivalor
        for entity_id, grupo in trabajo[
            trabajo["sensor_kind"].eq("multistate")
        ].groupby("entity_id", sort=False):
            serie = multivalor._serie_valida(grupo)
            diagnostico = multivalor._diagnosticar(str(entity_id), grupo, serie)
            if diagnostico["status"] != "eligible":
                continue
            features = multivalor._extraer_features(serie)
            base = multivalor._construir_base(features)
            transiciones = features[features["state_change_event"].eq(1)]
            transition_counts = Counter(
                zip(transiciones["previous_state_key"], transiciones["state_key"])
            )
            source_counts = Counter(transiciones["previous_state_key"])
            ultima = serie.tail(1).iloc[0]
            ultimo_estado = str(ultima["state_key"])
            mascara_cambio = serie["state_key"].ne(serie["state_key"].shift(1))
            inicio_estado = serie.loc[mascara_cambio, "timestamp_dt"].iloc[-1]
            cambios_recientes = features[
                features["state_change_event"].eq(1)
                & features["timestamp_dt"].ge(
                    ultima["timestamp_dt"]
                    - pd.Timedelta(seconds=multivalor.window_seconds)
                )
            ]["timestamp_dt"].tolist()
            baseline["multistate"][str(entity_id)] = {
                "base": base,
                "state_counts": Counter(features["state_key"]),
                "transition_counts": transition_counts,
                "source_counts": source_counts,
                "runtime": {
                    "last_timestamp": ultima["timestamp_dt"],
                    "last_state": str(ultima["state"]),
                    "last_state_key": ultimo_estado,
                    "state_started_at": inicio_estado,
                    "change_times": deque(cambios_recientes),
                },
            }

        trabajo["hour"] = trabajo["timestamp_dt"].dt.hour
        baseline["context"] = {
            "continuous": self.filtro._baseline_horario_continuo(trabajo),
            "binary": self.filtro._baseline_horario_binario(trabajo),
            "repeated_continuous": self.filtro._excursiones_repetidas_continuas(
                trabajo
            ),
        }
        baseline["summary"] = {
            "continuous": len(baseline["continuous"]),
            "binary": len(baseline["binary"]),
            "multistate": len(baseline["multistate"]),
        }
        return baseline

    def analizar_lote(
        self,
        nuevos_eventos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> dict[str, Any]:
        inicio = perf_counter()
        if nuevos_eventos.empty:
            return self._resultado([], inicio, {})
        umbral_local = min(float(suspicion_threshold), self.filtro.umbral_activacion_lapsos)

        inicio_tipo = perf_counter()
        continuos = self._analizar_continuos(
            nuevos_eventos, baseline, umbral_local
        )
        tiempo_continuos = (perf_counter() - inicio_tipo) * 1000.0
        inicio_tipo = perf_counter()
        binarios = self._analizar_binarios(nuevos_eventos, baseline, umbral_local)
        tiempo_binarios = (perf_counter() - inicio_tipo) * 1000.0
        inicio_tipo = perf_counter()
        multivalor = self._analizar_multivalor(
            nuevos_eventos, baseline, umbral_local
        )
        tiempo_multivalor = (perf_counter() - inicio_tipo) * 1000.0

        resultados = self._aplicar_contexto(
            continuos + binarios + multivalor,
            baseline.get("context", {}),
            suspicion_threshold,
        )
        resultados = self._suprimir_repetidos(resultados, baseline)
        return self._resultado(
            resultados,
            inicio,
            {
                "continuous": tiempo_continuos,
                "binary": tiempo_binarios,
                "multistate": tiempo_multivalor,
            },
        )

    def _analizar_continuos(
        self,
        nuevos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        filtro = self.filtro.continuos
        trabajo = nuevos[nuevos["sensor_kind"].eq("continuous")].copy()
        if "is_valid_value" in trabajo.columns:
            trabajo = trabajo[trabajo["is_valid_value"].astype(bool)].copy()
        resultados: list[dict[str, Any]] = []
        evidence_threshold = filtro._umbral_evidencia(suspicion_threshold)
        for entity_id, filas in trabajo.groupby("entity_id", sort=False):
            cached = baseline.get("continuous", {}).get(str(entity_id))
            if not cached:
                continue
            base = cached["base"]
            runtime = cached["runtime"]
            for fila in filas.sort_values("timestamp_dt").to_dict("records"):
                timestamp = pd.to_datetime(
                    fila.get("timestamp_dt"), errors="coerce", utc=True
                )
                valor = pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").iloc[0]
                if pd.isna(timestamp) or pd.isna(valor):
                    continue
                valor = float(valor)
                previo_ts = runtime.get("last_timestamp")
                previo_valor = runtime.get("last_value")
                gap_seconds = (
                    0.0
                    if previo_ts is None
                    else max(0.0, (timestamp - previo_ts).total_seconds())
                )
                delta_value = 0.0 if previo_valor is None else valor - float(previo_valor)
                rate_per_minute = abs(delta_value) / max(
                    gap_seconds / 60.0, 1.0 / 60.0
                )
                recientes = pd.Series(list(runtime["recent_values"]), dtype="float64")
                local_median = float(recientes.median()) if not recientes.empty else float(base["median"])
                local_mad = (
                    max(filtro._mad(recientes), float(base["mad"]))
                    if not recientes.empty
                    else float(base["mad"])
                )
                fila_score = {
                    **fila,
                    "value": valor,
                    "timestamp": str(
                        fila.get("timestamp")
                        or timestamp.isoformat()
                    ),
                    "local_median": local_median,
                    "local_mad": local_mad,
                }
                base_fila = filtro._base_horaria(base, fila_score)
                if (
                    runtime["cusum"].get("ewma") is None
                    or gap_seconds > base_fila["cusum_reset_gap_seconds"]
                ):
                    runtime["constant_duration_seconds"] = 0.0
                raw_fisico, s_fisico = filtro._score_fisico(valor, base_fila["perfil"])
                dentro_banda = base_fila["normal_lower"] <= valor <= base_fila["normal_upper"]
                raw_h, s_h = filtro._score_hampel(fila_score, base_fila)
                raw_r, s_r = filtro._score_tasa_robusta(rate_per_minute, base_fila)
                raw_m, s_m = filtro._score_brecha_temporal(gap_seconds, base_fila)
                raw_dspot, s_dspot = filtro._score_dspot_extremo(valor, base_fila)
                runtime["cusum"], raw_c, s_c, residual = filtro._actualizar_cusum(
                    valor=valor,
                    gap_seconds=gap_seconds,
                    base=base_fila,
                    dentro_banda=dentro_banda,
                    score_limite_operacional=s_fisico,
                    estado=runtime["cusum"],
                )
                if (
                    previo_valor is not None
                    and abs(valor - float(previo_valor)) <= base_fila["epsilon_const"]
                    and gap_seconds <= base_fila["gap_alert_seconds"]
                ):
                    runtime["constant_duration_seconds"] += gap_seconds
                else:
                    runtime["constant_duration_seconds"] = 0.0
                raw_const, s_const = filtro._score_persistencia_constante(
                    runtime["constant_duration_seconds"], base_fila
                )
                if s_m > 0.0:
                    if runtime["gap_reported"]:
                        raw_m, s_m = 0.0, 0.0
                    runtime["gap_reported"] = True
                else:
                    runtime["gap_reported"] = False
                if dentro_banda and s_fisico <= 0 and abs(delta_value) < base_fila["normal_delta_for_rate"]:
                    s_r = 0.0
                if dentro_banda and s_fisico <= 0 and s_r <= 0.0 and s_dspot <= 0.0:
                    s_h = 0.0
                fuente_h = "hampel"
                if s_fisico > s_h:
                    s_h = s_fisico
                    fuente_h = "operational_limit"
                componentes = {
                    "s_H": round(s_h, 6),
                    "s_R": round(s_r, 6),
                    "s_C": round(s_c, 6),
                    "s_D_const": round(s_const, 6),
                    "s_M": round(s_m, 6),
                    "s_DSPOT": round(s_dspot, 6),
                }
                raw_componentes = {
                    "H_i": round(raw_h, 6),
                    "Z_R_i": round(raw_r, 6),
                    "C_residual": round(raw_c, 6),
                    "Z_D_const": round(raw_const, 6),
                    "Z_M": round(raw_m, 6),
                    "Z_DSPOT": round(raw_dspot, 6),
                    "Z_F_i": round(raw_fisico, 6),
                }
                indicadores = filtro._indicadores_activados(
                    componentes, evidence_threshold
                )
                evidencia = filtro._evidencia_continua(componentes, indicadores)
                score = filtro._score_sospecha(evidencia)
                if indicadores and score >= suspicion_threshold:
                    principal = filtro._indicador_principal(componentes, indicadores)
                    patron = INDICADORES_CONTINUOS[principal]
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
                            "raw_score": filtro._raw_dominante(raw_componentes, principal),
                            "evidence_score": round(evidencia, 6),
                            "suspicion_score": round(score, 6),
                            "score_version": filtro.SCORE_VERSION,
                            "score_kind": filtro.SCORE_KIND,
                            "is_suspicious": True,
                            "pattern": patron,
                            "suspected_pattern": patron,
                            "pattern_detail": ", ".join(DETALLE_CONTINUO[item] for item in indicadores),
                            "filter": "statistical_temporal_validation_replay",
                            "passed_to_contextual_filter": True,
                            "triggered_indicators": indicadores,
                            "reading": {
                                "value": round(valor, 6),
                                "delta_value": round(delta_value, 6),
                                "rate_per_minute": round(rate_per_minute, 6),
                                "gap_seconds": round(gap_seconds, 3),
                                "constant_duration_seconds": round(runtime["constant_duration_seconds"], 3),
                                "residual": round(residual, 6),
                                "components": componentes,
                                "raw_components": raw_componentes,
                                "component_sources": {"s_H": fuente_h},
                                "operational_limit_score": round(s_fisico, 6),
                            },
                            "baseline_summary": filtro._resumen_base_para_salida(base_fila),
                            "ground_truth_label": None,
                            "severity": filtro._severidad(score),
                        }
                    )
                runtime["recent_values"].append(valor)
                runtime["last_timestamp"] = timestamp
                runtime["last_value"] = valor
        return resultados

    def _analizar_binarios(
        self,
        nuevos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        filtro = self.filtro.binarios
        trabajo = nuevos[nuevos["sensor_kind"].eq("binary")].copy()
        if "is_valid_value" in trabajo.columns:
            trabajo = trabajo[trabajo["is_valid_value"].astype(bool)].copy()
        resultados: list[dict[str, Any]] = []
        raw_threshold = filtro._probabilidad_a_evidencia(suspicion_threshold)
        for entity_id, filas in trabajo.groupby("entity_id", sort=False):
            cached = baseline.get("binary", {}).get(str(entity_id))
            if not cached:
                continue
            perfil = cached["perfil"]
            runtime = cached["runtime"]
            tail = runtime["tail"].assign(_replay_new=False)
            actuales = filas.assign(_replay_new=True)
            combinado = pd.concat([tail, actuales], ignore_index=True).sort_values("timestamp_dt")
            combinado["value"] = (
                pd.to_numeric(combinado["value"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0, upper=1.0)
            )
            features = filtro._extraer_features(combinado, perfil)
            timestamps_nuevos = set(
                pd.to_datetime(
                    actuales["timestamp_dt"], errors="coerce", utc=True
                ).dropna()
            )
            features_nuevos = features[
                features["timestamp_dt"].isin(timestamps_nuevos)
            ]
            for fila in features_nuevos.to_dict("records"):
                raw, componentes, evidencia = filtro._puntuar_doble_horizonte(
                    fila, cached["base"]
                )
                score = filtro._score_a_probabilidad(evidencia)
                indicadores = filtro._indicadores_activados(componentes)
                if not indicadores or evidencia < raw_threshold or score < suspicion_threshold:
                    continue
                principal = filtro._indicador_principal(componentes, indicadores)
                patron = INDICADORES_BINARIOS[principal]
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
                        "raw_score": filtro._raw_dominante(raw, principal),
                        "evidence_score": round(evidencia, 6),
                        "suspicion_score": round(score, 6),
                        "is_suspicious": True,
                        "pattern": patron,
                        "suspected_pattern": patron,
                        "pattern_detail": ", ".join(DETALLE_BINARIO[item] for item in indicadores),
                        "filter": "statistical_temporal_validation_replay",
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
                            "raw_components": raw,
                        },
                        "baseline_summary": filtro._resumen_base_para_salida(cached["base"]),
                        "ground_truth_label": None,
                        "severity": filtro._severidad(score),
                    }
                )
            ultima_fecha = combinado["timestamp_dt"].max()
            runtime["tail"] = combinado[
                combinado["timestamp_dt"].ge(
                    ultima_fecha - pd.Timedelta(seconds=perfil.window_seconds * 2)
                )
            ].drop(columns=["_replay_new"], errors="ignore")
        return resultados

    def _analizar_multivalor(
        self,
        nuevos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        filtro = self.filtro.multivalor
        trabajo = nuevos[nuevos["sensor_kind"].eq("multistate")].copy()
        if "is_valid_value" in trabajo.columns:
            trabajo = trabajo[trabajo["is_valid_value"].astype(bool)].copy()
        resultados: list[dict[str, Any]] = []
        for entity_id, filas in trabajo.groupby("entity_id", sort=False):
            cached = baseline.get("multistate", {}).get(str(entity_id))
            if not cached:
                continue
            runtime = cached["runtime"]
            base = cached["base"]
            for fila in filas.sort_values("timestamp_dt").to_dict("records"):
                timestamp = pd.to_datetime(
                    fila.get("timestamp_dt"), errors="coerce", utc=True
                )
                if pd.isna(timestamp):
                    continue
                state = str(fila.get("state", "")).strip()
                state_key = state.casefold()
                previous_state = runtime.get("last_state")
                previous_key = runtime.get("last_state_key")
                previous_timestamp = runtime.get("last_timestamp")
                changed = previous_key is not None and state_key != previous_key
                gap_seconds = (
                    0.0
                    if previous_timestamp is None
                    else max(0.0, (timestamp - previous_timestamp).total_seconds())
                )
                if changed or runtime.get("state_started_at") is None:
                    runtime["state_started_at"] = timestamp
                    if changed:
                        runtime["change_times"].append(timestamp)
                limite = timestamp - pd.Timedelta(seconds=filtro.window_seconds)
                while runtime["change_times"] and runtime["change_times"][0] <= limite:
                    runtime["change_times"].popleft()
                state_probability = cached["state_counts"].get(state_key, 0) / max(
                    int(base["rows"]), 1
                )
                source_support = int(cached["source_counts"].get(previous_key, 0))
                transition_probability = 1.0
                if changed and previous_key is not None:
                    transition_probability = cached["transition_counts"].get(
                        (previous_key, state_key), 0
                    ) / max(source_support, 1)
                duracion = max(
                    0.0, (timestamp - runtime["state_started_at"]).total_seconds()
                )
                feature = {
                    "state_changes": len(runtime["change_times"]),
                    "state_duration_seconds": duracion,
                    "report_gap_seconds": gap_seconds,
                    "state_probability": state_probability,
                    "transition_probability": transition_probability,
                    "transition_source_support": source_support,
                }
                raw, componentes = filtro._puntuar(feature, base)
                indicadores = [
                    item
                    for item, componente in componentes.items()
                    if componente >= filtro.component_minimum
                ]
                evidencia = max(componentes.values(), default=0.0)
                if indicadores and evidencia >= suspicion_threshold:
                    principal = max(indicadores, key=lambda item: componentes[item])
                    patron = INDICADORES_MULTIESTADO[principal]
                    resultados.append(
                        {
                            "entity_id": str(entity_id),
                            "timestamp": str(fila.get("timestamp")),
                            "window_start": (
                                timestamp - pd.Timedelta(seconds=filtro.window_seconds)
                            ).isoformat(),
                            "window_end": str(fila.get("timestamp")),
                            "sensor_kind": "multistate",
                            "tipo_interno": "estado_multivalor",
                            "observed_state": state,
                            "value": None,
                            "raw_score": round(float(raw[principal]), 6),
                            "evidence_score": round(float(evidencia), 6),
                            "suspicion_score": round(float(evidencia), 6),
                            "is_suspicious": True,
                            "pattern": patron,
                            "suspected_pattern": patron,
                            "pattern_detail": ", ".join(DETALLE_MULTIVALOR[item] for item in indicadores),
                            "filter": "statistical_temporal_validation_replay",
                            "passed_to_contextual_filter": True,
                            "triggered_indicators": indicadores,
                            "reading": {
                                "state": state,
                                "previous_state": previous_state,
                                "state_changes": len(runtime["change_times"]),
                                "state_duration_seconds": round(duracion, 3),
                                "report_gap_seconds": round(gap_seconds, 3),
                                "state_probability": round(state_probability, 6),
                                "transition_probability": round(transition_probability, 6),
                                "window_seconds": filtro.window_seconds,
                                "components": {key: round(float(value), 6) for key, value in componentes.items()},
                                "raw_components": {key: round(float(value), 6) for key, value in raw.items()},
                            },
                            "baseline_summary": filtro._base_para_salida(base),
                            "ground_truth_label": None,
                            "severity": filtro._severidad(evidencia),
                            "score_version": filtro.score_version,
                            "score_kind": filtro.score_kind,
                        }
                    )
                runtime["last_timestamp"] = timestamp
                runtime["last_state"] = state
                runtime["last_state_key"] = state_key
        return resultados

    def _aplicar_contexto(
        self,
        resultados: list[dict[str, Any]],
        context: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        filtrados = []
        for resultado in resultados:
            enriquecido = dict(resultado)
            score_contexto = self.filtro._score_contexto_temporal(
                enriquecido,
                context.get("continuous", {}),
                context.get("binary", {}),
            )
            repeticion = self.filtro._repeticion_continua(
                enriquecido, context.get("repeated_continuous", {})
            )
            local = float(enriquecido.get("suspicion_score") or 0.0)
            enriquecido["is_suspicious"] = local >= suspicion_threshold
            enriquecido["time_window"] = (
                "month_repetition"
                if repeticion
                else ("local" if score_contexto <= local else "hour_of_day")
            )
            enriquecido["temporal_context"] = {
                "activation_threshold": self.filtro.umbral_activacion_lapsos,
                "local_score": round(local, 6),
                "hour_of_day_score": round(score_contexto, 6),
                "temporal_consistency": self.filtro._consistencia_temporal(
                    score_contexto, repeticion
                ),
                **repeticion,
            }
            if enriquecido["is_suspicious"]:
                filtrados.append(enriquecido)
        return filtrados

    def _suprimir_repetidos(
        self,
        resultados: list[dict[str, Any]],
        baseline: dict[str, Any],
    ) -> list[dict[str, Any]]:
        salida = []
        ultimo = baseline.setdefault("last_emitted", {})
        for resultado in sorted(resultados, key=self.filtro._clave_fecha_resultado):
            clave = (str(resultado["entity_id"]), str(resultado["pattern"]))
            timestamp = pd.to_datetime(resultado["timestamp"], errors="coerce", utc=True)
            previo = ultimo.get(clave)
            supresion = (
                self.filtro.continuos.supresion_segundos
                if resultado["sensor_kind"] == "continuous"
                else 60 * 60
            )
            if previo is not None and pd.notna(timestamp) and (
                timestamp - previo
            ).total_seconds() <= supresion:
                continue
            ultimo[clave] = timestamp
            salida.append(resultado)
        return sorted(salida, key=self.filtro._clave_fecha_resultado, reverse=True)

    def _resultado(
        self,
        resultados: list[dict[str, Any]],
        inicio: float,
        tiempos: dict[str, float],
    ) -> dict[str, Any]:
        elapsed_ms = (perf_counter() - inicio) * 1000.0
        resumen = {
            "results": len(resultados),
            "suspicious": len(resultados),
            "by_kind": {
                kind: sum(1 for item in resultados if item.get("sensor_kind") == kind)
                for kind in ("continuous", "binary", "multistate")
            },
            "elapsed_ms": round(elapsed_ms, 3),
            "timings_by_kind_ms": {
                kind: round(float(tiempos.get(kind, 0.0)), 3)
                for kind in ("continuous", "binary", "multistate")
            },
            "causal": True,
            "baseline_partition": "development",
            "evaluated_partition": "validation",
            "uses_future_validation_rows": False,
            "engine_version": self.VERSION,
        }
        return {
            "results": resultados,
            "all_results": resultados,
            "summary": resumen,
        }
