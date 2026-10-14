from __future__ import annotations
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
import pandas as pd
from app.services.historial.cargar_historial import ResultadoHistorial
from app.modulos.estadistico.episodios import DEFAULT_QUIET_SECONDS
from app.modulos.estadistico.episodios import agrupar_sospechas
from app.modulos.estadistico.episodios import enviar_a_contextual
from app.modulos.estadistico.episodios import resumen_episodios
from app.services.application_common import serializable_json, filtrar_rango, construir_timeline, reducir_puntos, valor_para_timeline

class ServicesAnalysisMixin:
    def analizar_historial(
        self, suspicion_threshold: float, suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")

        cached = self._leer_cache_analisis_real(suspicion_threshold, suspicion_quiet_seconds)
        if cached is not None:
            self.resultados = list(cached.get("all_results", cached.get("results", [])))
            contextual = cached.get("contextual")
            if not isinstance(contextual, dict):
                contextual = self._analizar_contextual(self.resultados, self.historial.eventos)
                cached["contextual"] = contextual
            self.resultados_contextuales = list(contextual.get("results", []))
            self.resultados_contextuales = self._diagnosticar_contextuales(
                self.resultados, self.resultados_contextuales, self.historial.eventos,
            )
            cached["contextual"] = {**contextual, "results": self.resultados_contextuales}
            cached.update(self._salidas_pipeline(self.resultados, self.resultados_contextuales))
            summary = dict(cached.get("summary") or {})
            summary["suspicion_threshold"] = float(suspicion_threshold)
            cached["summary"] = summary
            self.analysis_completed = True
            return cached

        resultado = self.filtro_estadistico.analizar_historial(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
            suspicion_threshold=suspicion_threshold,
        )
        agrupados, _ = agrupar_sospechas(
            list(resultado.get("all_results", resultado.get("results", []))), suspicion_quiet_seconds
        )
        resultado["all_results"] = agrupados
        resultado["results"] = agrupados[:1000]
        resultado["top_results"] = agrupados[:1000]
        summary = dict(resultado.get("summary") or {})
        summary["suspicion_threshold"] = float(suspicion_threshold)
        summary["suspicion_quiet_seconds"] = suspicion_quiet_seconds
        summary["grouping"] = resumen_episodios(agrupados)
        summary["top_results"] = agrupados[:1000]
        summary["cache_hit"] = False
        summary["cache_path"] = str(self.analysis_cache_path) if self._es_historial_real_activo() else ""
        resultado["summary"] = summary
        # Los limites de visualizacion no deben recortar las entradas de F2.
        self.resultados = list(resultado.get("all_results", resultado.get("results", [])))
        contextual = self._analizar_contextual(self.resultados, self.historial.eventos)
        resultado["contextual"] = contextual
        summary["contextual_evaluated"] = int(contextual.get("summary", {}).get("evaluated", 0))
        summary["contextual_elapsed_ms"] = float(contextual.get("summary", {}).get("elapsed_ms", 0.0))
        self.resultados_contextuales = list(contextual.get("results", []))
        self.resultados_contextuales = self._diagnosticar_contextuales(
            self.resultados, self.resultados_contextuales, self.historial.eventos,
        )
        resultado["contextual"] = {**contextual, "results": self.resultados_contextuales}
        resultado.update(self._salidas_pipeline(self.resultados, self.resultados_contextuales))
        self.analysis_completed = True
        self._guardar_cache_analisis_real(suspicion_threshold, resultado, suspicion_quiet_seconds)
        return resultado
    def vista_normalizacion(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        normalidad = self._asegurar_normalidad_persistida()
        return {
            "source_type": self.historial.metadatos.get("source_type", ""),
            "source_path": self.historial.metadatos.get("source_path", ""),
            "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
            "entity_count": int(self.historial.metadatos.get("entity_count", len(self.historial.entidades))),
            "samples_per_entity": 5,
            "samples": self.historial.normalizacion_muestras,
            "fragments": self._fragmentos_normalizacion(),
            "normality": self._leer_normalidad_persistida(normalidad),
        }
    def configuracion_continua(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        configs = self.filtro_estadistico.continuos.configuracion_continua(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
        )
        return {"config_path": str(self.filtro_estadistico.continuos.conf_path), "baseline_id": (self.future_baseline_template or {}).get("provenance", {}).get("baseline_id"), "configs": configs}
    def guardar_configuracion_continua(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        guardadas = self.filtro_estadistico.continuos.guardar_configuracion(configs)
        if self.analysis_cache_path.exists():
            self.analysis_cache_path.unlink()
        # La configuracion cambia la base estadistica. Se vuelve a congelar el
        # estado desde el historico aqui, nunca durante la carga de Evaluacion.
        self._preparar_continuidad_historica()
        return {"config_path": str(self.filtro_estadistico.continuos.conf_path), "baseline_id": (self.future_baseline_template or {}).get("provenance", {}).get("baseline_id"), "configs": guardadas}
    def mapa_contextual(self, force: bool = False) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        if (
            force
            and self.validation_replay_state
            and int(self.validation_replay_state.get("cursor", 0)) > 0
        ):
            raise ValueError(
                "Reinicia Evaluacion antes de reconstruir el mapa contextual; "
                "el pipeline debe conservar la misma referencia durante toda la ejecucion."
            )
        mapa = self._asegurar_mapa_contextual(force=force)
        self.future_context_map_template = deepcopy(mapa)
        if self.validation_context_map is not None:
            self.validation_context_map = deepcopy(mapa)
        return mapa
    def analizar_contexto_actual(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        payload = self._analizar_contextual(self.resultados, self.historial.eventos)
        self.resultados_contextuales = list(payload.get("results", []))
        self.resultados_contextuales = self._diagnosticar_contextuales(
            self.resultados, self.resultados_contextuales, self.historial.eventos,
        )
        payload["results"] = self.resultados_contextuales
        payload.update(self._salidas_pipeline(self.resultados, self.resultados_contextuales))
        return payload
    def contexto_actual(self) -> dict[str, Any]:
        if self.validation_replay_state:
            status = self.estado_reproduccion_validacion()
            contextuales = self.resultados_reproduccion_validacion(
                "contextual", 0, 500
            )
            ambientales = self.resultados_reproduccion_validacion(
                "environmental", 0, 500
            )
            pendientes = self.resultados_reproduccion_validacion(
                "pending", 0, 500
            )
            clasificados = self.resultados_reproduccion_validacion("classifier", 0, 500)
            return {
                "results": contextuales["items"],
                "summary": status.get("contextual_summary", {}),
                "environmental_count": status.get("environmental_count", 0),
                "pending_classification_count": status.get(
                    "pending_classification_count", 0
                ),
                "environmental_results": ambientales["items"],
                "pending_classification_results": pendientes["items"],
                "classifier_results": clasificados["items"],
                "classification_count": status.get("classification_count", 0),
                "indeterminate_count": status.get("indeterminate_count", 0),
                "no_fault_count": status.get("no_fault_count", 0),
                "pipeline_summary": status.get("pipeline_summary", {}),
                "source": "validation_replay",
                "map_quality": (self.validation_context_map or {}).get(
                    "refresh", {}
                ),
            }

        resultados_origen = self.resultados_contextuales
        resultados = sorted(
            resultados_origen,
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
            reverse=True,
        )
        summary: dict[str, Any] = {
            "evaluated": len(resultados),
            **self.filtro_contextual.resumen_enrutamiento(resultados),
            "by_outcome": {},
            "possible_common_cause": 0,
        }
        for item in resultados:
            outcome = str(item.get("context_outcome", "unexplained"))
            summary["by_outcome"][outcome] = summary["by_outcome"].get(outcome, 0) + 1
            summary["possible_common_cause"] += int(
                bool(item.get("common_cause", {}).get("candidate"))
            )
        summary["contextual_coincidences"] = sum(
            bool(item.get("common_cause", {}).get("contextual_coincidence")) for item in resultados
        )
        return {
            "results": resultados[:500],
            "summary": summary,
            **self._salidas_pipeline(
                self.resultados,
                resultados,
                limit=500,
            ),
            "source": "historical",
            "map_quality": (self.mapa_contextual_cache or {}).get("refresh", {}),
        }
    def _analizar_contextual(
        self,
        resultados_estadisticos: list[dict[str, Any]],
        eventos: pd.DataFrame,
    ) -> dict[str, Any]:
        mapa = self._asegurar_mapa_contextual()
        return self.filtro_contextual.analizar(
            sospechas=[item for item in resultados_estadisticos if enviar_a_contextual(item)],
            eventos=eventos,
            entidades=self.historial.entidades,
            mapa=mapa,
        )
    def _asegurar_mapa_contextual(self, force: bool = False, progress_callback=None) -> dict[str, Any]:
        if progress_callback is not None:
            progress_callback("context_cache", 0.0, "Comprobando el mapa contextual guardado")
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        previous = self.mapa_contextual_cache or self._leer_mapa_contextual_cache()
        reasons = self.filtro_contextual.razones_actualizacion(
            previous,
            self.historial.eventos,
            self.historial.entidades,
        )
        if force:
            reasons = ["manual_rebuild", *reasons]
        if previous is not None and not reasons:
            previous = deepcopy(previous)
            previous.setdefault("refresh", {})["trigger"] = "cache_reused"
            previous["refresh"]["stale"] = False
            previous["refresh"]["stale_reasons"] = []
            self.mapa_contextual_cache = previous
            return previous

        trigger = reasons[0] if reasons else "manual_rebuild"
        mapa = self.filtro_contextual.construir_mapa(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
            trigger=trigger,
            mapa_anterior=previous,
            **({"progress_callback": progress_callback} if progress_callback is not None else {}),
        )
        mapa["refresh"]["rebuild_reasons"] = list(
            dict.fromkeys([*reasons, *mapa["refresh"].get("rebuild_reasons", [])])
        )
        self.mapa_contextual_cache = mapa
        if progress_callback is not None:
            progress_callback("context_save", 99.0, "Guardando el mapa contextual")
        self._guardar_mapa_contextual_cache(mapa)
        return mapa
    def _ruta_cache_contextual(self) -> Path:
        return self.context_map_cache_path or self.analysis_cache_path.parent / "contextual_map.json"
    def _leer_mapa_contextual_cache(self) -> dict[str, Any] | None:
        path = self._ruta_cache_contextual()
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as archivo:
                payload = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None
    def _guardar_mapa_contextual_cache(self, mapa: dict[str, Any]) -> None:
        path = self._ruta_cache_contextual()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as archivo:
                json.dump(serializable_json(mapa), archivo, ensure_ascii=True)
        except OSError:
            return
    def _fragmentos_normalizacion(self, max_points_per_entity: int = 80) -> list[dict[str, Any]]:
        if self.historial is None or self.historial.eventos.empty:
            return []
        lookup = self.historial.entidades.set_index("entity_id").to_dict("index") if not self.historial.entidades.empty else {}
        fragments = []
        for entity_id, grupo in self.historial.eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
            entidad = lookup.get(str(entity_id), {})
            tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
            sampleado = reducir_puntos(grupo.head(max_points_per_entity * 3), max_points_per_entity)
            points = []
            for fila in sampleado.to_dict("records"):
                if tipo == "binary":
                    value = pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0]
                    value = 100.0 if float(value) >= 0.5 else 0.0
                else:
                    value = valor_para_timeline(fila, tipo)
                points.append(
                    {
                        "timestamp": str(fila["timestamp"]),
                        "state": str(fila["state"]),
                        "value": value,
                    }
                )
            fragments.append(
                {
                    "entity_id": str(entity_id),
                    "sensor_kind": tipo,
                    "area_id": "",
                    "chart_kind": "normalization_fragment",
                    "scale_unit": "%" if tipo == "binary" else str(entidad.get("unit_of_measurement") or "state"),
                    "points": points,
                }
            )
        return fragments
    def timeline_historial(
        self,
        start: str | None,
        end: str | None,
        max_points: int,
    ) -> dict[str, Any]:
        """Construye una linea temporal exclusivamente desde el conjunto real cargado."""
        if self.historial is None:
            self.status()
        if self.historial is None:
            return {"entities": []}

        if not start and not end:
            clave = json.dumps(
                {"historical_only": True, "max_points": int(max_points)},
                sort_keys=True,
            )
            if clave not in self.timeline_base_cache:
                self.timeline_base_cache[clave] = construir_timeline(
                    self.historial.eventos,
                    self.historial.entidades,
                    max_points=max_points,
                )
            return deepcopy(self.timeline_base_cache[clave])

        eventos = filtrar_rango(self.historial.eventos, start, end)
        return construir_timeline(
            eventos,
            self.historial.entidades,
            max_points=max_points,
        )
    def _status_desde_historial(self, historial: ResultadoHistorial) -> dict[str, Any]:
        metadatos = historial.metadatos
        normalidad = self._asegurar_normalidad_persistida()
        return {
            "exists": True,
            "valid": True,
            "rows": int(metadatos.get("rows", len(historial.eventos))),
            "entity_count": int(metadatos.get("entity_count", len(historial.entidades))),
            "entities": historial.entidades.to_dict("records"),
            "source_type": metadatos.get("source_type", ""),
            "source_path": metadatos.get("source_path", ""),
            "history_origin": metadatos.get("history_origin", "csv"),
            "start": metadatos.get("start"),
            "end": metadatos.get("end"),
            "normality_history": normalidad,
            "future_evaluation_ready": bool(
                self.future_baseline_template is not None
                and self.future_context_map_template is not None
                and self.future_context_events_template is not None
            ),
            "future_evaluation_prepared_at": self.future_evaluation_prepared_at,
            "future_preparation_cache": dict(self.future_preparation_cache),
            "future_baseline_summary": (
                {}
                if self.future_baseline_template is None
                else self.future_baseline_template.get("summary", {})
            ),
            "continuous_models": (self.future_baseline_template or {}).get("continuous_model_manifest", {}),
            "baseline_id": (self.future_baseline_template or {}).get("provenance", {}).get("baseline_id"),
        }
    def _reset_analysis(self) -> None:
        self.resultados = []
        self.resultados_contextuales = []
        self.analysis_completed = False
    def _es_historial_real_activo(self) -> bool:
        return self.historial is not None and self.historial.metadatos.get("source_type") == "real"
