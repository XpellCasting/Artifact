from __future__ import annotations

from collections import Counter, deque
import math
from time import perf_counter
from typing import Any
from copy import deepcopy

import pandas as pd

from app.services.entrenamiento_paralelo import train_sensors

from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_FOCUS_METHOD as CONTINUOUS_METHOD
from app.modulos.estadistico.continuos_temporales import causal_features
from app.modulos.estadistico.continuos_temporales import utc_timestamp
from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_METHODS
from app.modulos.estadistico.residuos_secuenciales import RESIDUAL_FOCUS_METHOD
from app.modulos.estadistico.residuos_secuenciales import update_channels
from app.modulos.estadistico.bernoulli import METHOD as BINARY_METHOD
from app.modulos.estadistico.disponibilidad import availability_detections
from app.modulos.estadistico.persistencia import prepare_persistence, evaluate_persistence
from app.modulos.estadistico.estados_multivalor import DETALLE_INDICADOR as DETALLE_MULTIVALOR
from app.modulos.estadistico.estados_multivalor import INDICADORES_MULTIESTADO


class MotorReproduccionValidacion:
    """Evalua lotes cronologicos contra una base fija de Desarrollo.

    El motor no genera eventos. Mantiene solamente el estado causal necesario
    entre lotes para que cada lectura de Validacion se evalue sin observar filas
    futuras del mismo archivo.
    """

    VERSION = "validation-focus-replay-v3-daily"

    def __init__(self, filtro_estadistico: Any) -> None:
        self.filtro = filtro_estadistico

    def construir_baseline(
        self,
        eventos_desarrollo: pd.DataFrame,
        entidades_desarrollo: pd.DataFrame,
        entidades_objetivo: set[str] | None = None,
        continuous_method: str = CONTINUOUS_METHOD,
        progress_callback=None,
    ) -> dict[str, Any]:
        def report(phase, progress, message):
            if progress_callback is not None:
                progress_callback(phase, progress, message)

        def groups(kind, phase, label, start, span):
            grouped = trabajo[trabajo["sensor_kind"].eq(kind)].groupby("entity_id", sort=False)
            total = len(grouped)
            for index, (entity_id, group) in enumerate(grouped):
                report(phase, start + span * index / max(1, total),
                       f"{label}: {entity_id} ({index}/{total} sensores completados)")
                yield entity_id, group
            report(phase, start + span, f"{label}: {total}/{total} sensores completados")

        report("baseline_prepare", 0.0, "Ordenando eventos de la base estadística")
        if continuous_method not in RESIDUAL_METHODS:
            raise ValueError(f"Método continuo desconocido: {continuous_method}")
        if eventos_desarrollo.empty or "sensor_kind" not in eventos_desarrollo.columns:
            raise ValueError("Desarrollo no contiene eventos normalizados para construir la base.")
        trabajo = eventos_desarrollo.copy()
        trabajo["__baseline_order"] = range(len(trabajo))
        trabajo["timestamp_dt"] = pd.to_datetime(
            trabajo["timestamp_dt"], errors="coerce", utc=True
        )
        if entidades_objetivo:
            trabajo = trabajo[trabajo["entity_id"].isin(entidades_objetivo)].copy()
        trabajo = trabajo.sort_values(
            ["entity_id", "timestamp_dt", "__baseline_order"], kind="stable"
        ).drop(columns=["__baseline_order"])
        lookup = (
            entidades_desarrollo.set_index("entity_id").to_dict("index")
            if not entidades_desarrollo.empty
            else {}
        )
        baseline: dict[str, Any] = {
            "version": self.VERSION,
            "continuous_method": continuous_method,
            "continuous": {},
            "binary": {},
            "multistate": {},
            "context": {},
            "last_emitted": {},
        }

        temporal_models: dict[str, Any] = {}
        temporal_entries: dict[str, Any] = {}
        continuous_groups = trabajo[trabajo["sensor_kind"].eq("continuous")].groupby("entity_id", sort=False)
        for entity_id, learned, runtime in train_sensors(
            self.filtro.continuos_temporales, continuous_groups, continuous_method,
            lambda progress, message: report("baseline_training", 25 + progress * .45, message),
        ):
            if learned.get("eligible"):
                serie = trabajo[trabajo["entity_id"].astype(str).eq(str(entity_id))].copy()
                if "is_valid_value" in serie:
                    serie = serie[serie["is_valid_value"].astype(bool)]
                serie["value"] = pd.to_numeric(serie["value"], errors="coerce")
                serie = serie.dropna(subset=["value", "timestamp_dt"])
                learned["constant_base"] = self.filtro.continuos._construir_base_normalidad(
                    serie, str(entity_id), persistir_perfil_calculado=False,
                )
                learned["persistence_reference"] = prepare_persistence(
                    trabajo[trabajo["entity_id"].astype(str).eq(str(entity_id))])
                temporal_entries[str(entity_id)] = learned
            temporal_models[str(entity_id)] = {key: value for key, value in learned.items() if key not in {"predictor", "runtime", "constant_base", "persistence_reference"}}
        # El orden de finalización de los procesos no cambia el orden del baseline.
        temporal_entries = dict(sorted(temporal_entries.items()))
        temporal_models = dict(sorted(temporal_models.items()))
        baseline["continuous"] = temporal_entries
        baseline["continuous_model_manifest"] = temporal_models

        for entity_id, grupo in groups("binary", "baseline_binary", "Preparando sensores binarios", 70, 10):
            cached = self.filtro.binarios.prepare(grupo)
            if cached:
                cached["persistence_reference"] = prepare_persistence(grupo)
                baseline["binary"][str(entity_id)] = cached

        multivalor = self.filtro.multivalor
        for entity_id, grupo in groups("multistate", "baseline_multistate", "Preparando sensores multivalor", 80, 10):
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

        report("baseline_hourly", 90.0, "Calculando referencias horarias y excursiones repetidas")
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
        report("baseline_ready", 100.0, "Base estadística preparada")
        return baseline

    def analizar_lote(
        self,
        nuevos_eventos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
        continuous_method: str | None = None,
    ) -> dict[str, Any]:
        inicio = perf_counter()
        if nuevos_eventos.empty:
            return self._resultado([], inicio, {}, detections=[])
        umbral_local = min(float(suspicion_threshold), self.filtro.umbral_activacion_lapsos)

        inicio_tipo = perf_counter()
        method = continuous_method or baseline.get("continuous_method", CONTINUOUS_METHOD)
        if method not in RESIDUAL_METHODS:
            raise ValueError(f"Método continuo desconocido: {method}")
        continuos = self._analizar_continuos_residuos(nuevos_eventos, baseline, method)
        tiempo_continuos = (perf_counter() - inicio_tipo) * 1000.0
        inicio_tipo = perf_counter()
        binarios = self._analizar_binarios(nuevos_eventos, baseline, umbral_local)
        tiempo_binarios = (perf_counter() - inicio_tipo) * 1000.0
        inicio_tipo = perf_counter()
        multivalor = self._analizar_multivalor(
            nuevos_eventos, baseline, umbral_local
        )
        tiempo_multivalor = (perf_counter() - inicio_tipo) * 1000.0

        resultados_umbral = self._aplicar_contexto(
            continuos + binarios + multivalor,
            baseline.get("context", {}),
            suspicion_threshold,
        )
        disponibilidad = availability_detections(nuevos_eventos)
        resultados_umbral.extend(disponibilidad)
        observations = baseline.setdefault("last_observations", {})
        for detection in disponibilidad:
            observations[str(detection["source_raw_index"])] = {
                "evaluable": True, "reason": None, "detected_before_suppression": True,
                "observed_state": detection["observed_state"],
                "detection_sources": detection["detection_sources"],
            }
        detections = deepcopy(resultados_umbral)
        resultados = self._suprimir_repetidos(deepcopy(resultados_umbral), baseline)
        return self._resultado(
            resultados,
            inicio,
            {
                "continuous": tiempo_continuos,
                "binary": tiempo_binarios,
                "multistate": tiempo_multivalor,
            },
            detections=detections,
        )



    def _analizar_continuos_residuos(self, nuevos: pd.DataFrame, baseline: dict[str, Any], method: str) -> list[dict[str, Any]]:
        """Causal residual scoring; a timestamp group shares its sequential update."""
        filtro = self.filtro.continuos
        work = nuevos[nuevos["sensor_kind"].eq("continuous")].copy()
        work["__order"] = range(len(work))
        numeric_values = "value" in work and pd.api.types.is_numeric_dtype(work["value"].dtype)
        observations = baseline.setdefault("last_observations", {})
        observations.clear(); results: list[dict[str, Any]] = []
        for entity_id, rows in work.groupby("entity_id", sort=False):
            cached = baseline.get("continuous", {}).get(str(entity_id))
            if not cached:
                continue
            runtime, predictor = cached["runtime"], cached["predictor"]
            ordered = rows.sort_values(["timestamp_dt", "__order"], kind="stable")
            for timestamp, simultaneous in ordered.groupby("timestamp_dt", sort=True):
                timestamp = utc_timestamp(timestamp)
                policy = cached.get("reporting_policy", {})
                if runtime.history and policy.get("mode") == "periodic":
                    limit = policy.get("max_report_gap_seconds")
                    if limit is None or (timestamp - runtime.history[-1][0]).total_seconds() > limit:
                        runtime.history.clear()
                        runtime.reset_sequential(method)
                        runtime.last_timestamp = None
                schema = getattr(predictor, "metadata", {}).get("feature_schema_version")
                vector = causal_features(runtime.history, timestamp, **({"feature_schema_version": schema} if schema else {}))
                valid_rows, pending = [], []
                for row in simultaneous.to_dict("records"):
                    raw = row.get("_raw_index"); key = str(raw) if raw is not None else f"{entity_id}:{timestamp.isoformat()}:{row['__order']}"
                    number = (row.get("value") if numeric_values else
                              pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0])
                    if pd.isna(timestamp) or not bool(row.get("is_valid_value", True)) or pd.isna(number) or not math.isfinite(float(number)):
                        runtime.history.clear()
                        runtime.reset_sequential(method)
                        runtime.last_timestamp = None
                        if cached.get("persistence_reference"):
                            evaluate_persistence(cached["persistence_reference"], row)
                        observations[key] = {"evaluable": False, "reason": "invalid_state", "continuous_method": method}; continue
                    value = float(number); pending.append((timestamp, value))
                    if vector is None:
                        observations[key] = {"evaluable": False, "reason": "insufficient_history", "continuous_method": method}; continue
                    expected = predictor.predict_median(vector) if hasattr(predictor, "predict_median") else predictor.predict_raw(vector)[1]
                    valid_rows.append((row, key, value, expected, (value - expected - cached["centre"]) / cached["scale"]))
                if valid_rows:
                    reset_after_group = False
                    # One sequential update per simultaneous group, while P stays per reading.
                    mean_z = sum(row[4] for row in valid_rows) / len(valid_rows)
                    channels = update_channels(runtime, mean_z, timestamp, method, cached["half_life_seconds"])
                    for row, key, value, expected, z in valid_rows:
                        p = abs(z); d, n = channels["D"], channels["N"]
                        qs = cached["channel_q99"]
                        ratios = {"P": p / qs["P"], "D": d / qs["D"], "N": n / qs["N"]}
                        limits = cached.get("channel_thresholds")
                        channel_scores = {key: ratios[key] / limits[key] for key in ratios} if limits else ratios
                        score = max(channel_scores.values()); dominant = max(channel_scores, key=channel_scores.get)
                        detected_numeric = score >= cached["threshold"]
                        reset_after_group |= detected_numeric
                        constant_score = 0.0
                        persistence = evaluate_persistence(cached["persistence_reference"], row) if cached.get("persistence_reference") else {}
                        constant_detected = bool(persistence.get("detected"))
                        constant_score = float(constant_detected)
                        sources = (["residual_sequential"] if detected_numeric else []) + (["constant_persistence"] if constant_detected else [])
                        detected = bool(sources)
                        change = channels.get("focus_changepoint_index")
                        observation = {
                            "evaluable": True, "reason": None, "detected_before_suppression": detected,
                            "continuous_method": method, "expected_value": expected, "residual": value - expected, "z": z,
                            "channels": {"P": p, "D": d, "N": n}, "dominant_channel": dominant,
                            "joint_score": score, "threshold": cached["threshold"], "detection_sources": sources,
                            "channel_q99": cached["channel_q99"], "constant_score": constant_score,
                            "channel_thresholds": limits, "channel_scores": channel_scores,
                            "persistence": persistence,
                            "focus_statistic": d if method == RESIDUAL_FOCUS_METHOD else None,
                            "focus_changepoint_index": change,
                        }
                        observations[key] = observation
                        if detected:
                            results.append({
                                "entity_id": str(entity_id), "source_raw_index": row.get("_raw_index"), "timestamp": str(row.get("timestamp") or timestamp.isoformat()),
                                "window_start": str(row.get("timestamp") or timestamp.isoformat()), "window_end": str(row.get("timestamp") or timestamp.isoformat()),
                                "sensor_kind": "continuous", "tipo_interno": "continua", "observed_state": str(row.get("state", value)), "value": value,
                                "continuous_method": method, "expected_value": expected, "residual": value - expected, "z": z,
                                "channels": observation["channels"], "dominant_channel": dominant, "joint_score": score, "threshold": cached["threshold"],
                                "suspicion_score": score, "is_suspicious": True, "pattern": "stuck_value" if constant_detected and not detected_numeric else "residual_sequential_deviation",
                                "suspected_pattern": "stuck_value" if constant_detected and not detected_numeric else "residual_sequential_deviation", "detection_sources": sources,
                                "triggered_indicators": sources, "filter": method, "passed_to_contextual_filter": True,
                                "reading": observation, "focus_changepoint_index": change,
                            })
                    if reset_after_group:
                        runtime.reset_sequential(method)
                for item in pending:
                    runtime.history.append(item)
        return results

    def _analizar_binarios(
        self,
        nuevos: pd.DataFrame,
        baseline: dict[str, Any],
        suspicion_threshold: float,
    ) -> list[dict[str, Any]]:
        resultados = []
        trabajo = nuevos[nuevos["sensor_kind"].eq("binary")]
        for entity_id, filas in trabajo.groupby("entity_id", sort=False):
            cached = baseline.get("binary", {}).get(str(entity_id))
            if cached:
                numeric = self.filtro.binarios.evaluate(filas, cached, entity_id)
                by_raw = {str(item.get("source_raw_index")): item for item in numeric}
                reference = cached.get("persistence_reference")
                if reference:
                    for row in filas.sort_values("timestamp_dt", kind="stable").to_dict("records"):
                        persistence = evaluate_persistence(reference, row)
                        key = str(row.get("_raw_index"))
                        detected = bool(persistence["detected"] or key in by_raw)
                        baseline.setdefault("last_observations", {})[key] = {
                            "evaluable": bool(row.get("is_valid_value", True)) and pd.notna(row.get("value")),
                            "reason": None if bool(row.get("is_valid_value", True)) and pd.notna(row.get("value")) else "invalid_state",
                            "detected_before_suppression": detected, "persistence": persistence}
                        if persistence["detected"] and key not in by_raw:
                            numeric.append({"entity_id": str(entity_id), "source_raw_index": row.get("_raw_index"),
                                "timestamp": row["timestamp_dt"].isoformat(), "window_start": persistence["started_at"],
                                "window_end": row["timestamp_dt"].isoformat(), "sensor_kind": "binary", "tipo_interno": "estado_binario",
                                "observed_state": str(row.get("state")), "value": float(row["value"]),
                                "binary_method": BINARY_METHOD, "is_suspicious": True, "suspicion_score": 1.,
                                "pattern": "state_persistence", "suspected_pattern": "stuck_value",
                                "detection_sources": ["state_and_hour_persistence"], "triggered_indicators": ["state_duration"],
                                "reading": persistence, "filter": "state_and_hour_persistence_v1", "passed_to_contextual_filter": True})
                resultados.extend(numeric)
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
                timestamp = utc_timestamp(fila.get("timestamp_dt"))
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
                            "source_raw_index": fila.get("_raw_index"),
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
            calibrated = (enriquecido.get("continuous_method") == RESIDUAL_FOCUS_METHOD
                          or enriquecido.get("binary_method") == BINARY_METHOD)
            enriquecido["is_suspicious"] = bool(enriquecido.get("is_suspicious")) if calibrated else local >= suspicion_threshold
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
        *,
        detections: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        detecciones = list(detections or [])
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
            "detection_count": len(detecciones),
            "emitted_alert_count": len(resultados),
            "suppressed_detection_count": max(0, len(detecciones) - len(resultados)),
            "statistical_evaluation_version": "pre_suppression_v1",
        }
        return {
            "detections": detecciones,
            "results": resultados,
            "all_results": resultados,
            "summary": resumen,
        }
