from __future__ import annotations

import json
import os
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .Cargar_Historial.Cargar_Historial import CargadorHistorial, ResultadoHistorial
from .Filtro_Clasificacion import ClasificadorDiagnostico, EvaluadorExtremoAExtremo
from .Filtro_Contextual import FiltroContextualEspacioTemporal
from .Filtro_Estadistico.Filtro import FiltroEstadistico
from .Patrones_Diseno.Unidad_Trabajo_Iteraciones import UnidadTrabajoIteraciones
from .simulacion.Inyeccion_Fallas import InyectorFallasSensornets


class TickRequest(BaseModel):
    selected_entities: list[str] = Field(default_factory=list)
    tick_seconds: int = 60
    faults: list[dict[str, Any]] = Field(default_factory=list)
    suspicion_threshold: float = 0.6


class IterationsRequest(TickRequest):
    iteration_count: int = 1


class AnalyzeRequest(BaseModel):
    suspicion_threshold: float = 0.6


class ClassifierEvaluationRequest(BaseModel):
    partition: str = "test"


class ContinuousConfigRequest(BaseModel):
    configs: list[dict[str, Any]] = Field(default_factory=list)


class ManualTickRequest(BaseModel):
    selected_entities: list[str] = Field(default_factory=list)
    sensor_states: dict[str, Any] = Field(default_factory=dict)
    tick_seconds: int = 60
    suspicion_threshold: float = 0.6


@dataclass
class EstadoAplicacion:
    cargador: CargadorHistorial
    filtro_estadistico: FiltroEstadistico
    static_history_dir: Path
    real_history_path: Path
    analysis_cache_path: Path
    filtro_contextual: FiltroContextualEspacioTemporal = field(
        default_factory=FiltroContextualEspacioTemporal
    )
    filtro_clasificacion: ClasificadorDiagnostico = field(
        default_factory=ClasificadorDiagnostico
    )
    evaluador_extremo_a_extremo: EvaluadorExtremoAExtremo = field(
        default_factory=EvaluadorExtremoAExtremo
    )
    context_map_cache_path: Path | None = None
    ground_truth_path: Path | None = None
    inyector_fallas: InyectorFallasSensornets = field(default_factory=InyectorFallasSensornets)
    historial: ResultadoHistorial | None = None
    resultados: list[dict[str, Any]] = field(default_factory=list)
    realtime_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_tick_index: int = 0
    realtime_current_timestamp: str | None = None
    realtime_faults: list[dict[str, Any]] = field(default_factory=list)
    realtime_fault_ticks: list[dict[str, Any]] = field(default_factory=list)
    selected_entities: set[str] = field(default_factory=set)
    realtime_baseline: dict[str, Any] | None = None
    timeline_base_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    mapa_contextual_cache: dict[str, Any] | None = None
    resultados_contextuales: list[dict[str, Any]] = field(default_factory=list)
    resultados_clasificacion: list[dict[str, Any]] = field(default_factory=list)
    ultimo_payload_clasificacion: dict[str, Any] | None = None
    ultima_evaluacion_clasificacion: dict[str, Any] | None = None
    analysis_completed: bool = False

    def status(self) -> dict[str, Any]:
        if self.historial is None:
            if self.real_history_path.exists():
                try:
                    self.historial = self.cargador.cargar_real(self.real_history_path)
                except Exception:
                    pass
            elif (self.static_history_dir / "events.csv").exists():
                try:
                    self.historial = self.cargador.cargar_canonico(self.static_history_dir)
                except Exception:
                    pass

        if self.historial is None:
            return {
                "exists": False,
                "valid": False,
                "rows": 0,
                "entity_count": 0,
                "entities": [],
                "source_type": "",
                "start": None,
                "end": None,
                "normality_history": {"exists": False, "rows": 0},
            }
        return self._status_desde_historial(self.historial)

    def cargar_real(self) -> dict[str, Any]:
        self.historial = self.cargador.cargar_real(self.real_history_path)
        self._reset_realtime()
        self.realtime_baseline = None
        self.timeline_base_cache = {}
        self.mapa_contextual_cache = None
        self._asegurar_normalidad_persistida()
        self._baseline_realtime()
        return self._status_desde_historial(self.historial)

    def subir_real(self, contenido: bytes) -> dict[str, Any]:
        if not contenido.strip():
            raise ValueError("El archivo de historial esta vacio.")
        self.real_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.real_history_path.write_bytes(contenido)
        return self.cargar_real()

    def asegurar_canonico(self) -> dict[str, Any]:
        self.historial = self.cargador.crear_canonico_minimo(self.static_history_dir)
        self._reset_realtime()
        self.realtime_baseline = None
        self.timeline_base_cache = {}
        self.mapa_contextual_cache = None
        self._baseline_realtime()
        return self._status_desde_historial(self.historial)

    def analizar_historial(self, suspicion_threshold: float) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")

        cached = self._leer_cache_analisis_real(suspicion_threshold)
        if cached is not None:
            self.resultados = list(cached.get("results", []))
            contextual = cached.get("contextual")
            if not isinstance(contextual, dict):
                contextual = self._analizar_contextual(
                    self.resultados,
                    self.historial.eventos,
                )
                cached["contextual"] = contextual
                self._guardar_cache_analisis_real(suspicion_threshold, cached)
            self.resultados_contextuales = list(contextual.get("results", []))
            classification = cached.get("classification")
            if not isinstance(classification, dict):
                classification = self._analizar_clasificacion(
                    self.resultados,
                    self.resultados_contextuales,
                    source="historical",
                )
                cached["classification"] = classification
                self._guardar_cache_analisis_real(suspicion_threshold, cached)
            self._guardar_estado_clasificacion(classification)
            self.analysis_completed = True
            return cached

        resultado = self.filtro_estadistico.analizar_historial(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
            suspicion_threshold=suspicion_threshold,
        )
        summary = dict(resultado.get("summary") or {})
        summary["cache_hit"] = False
        summary["cache_path"] = str(self.analysis_cache_path) if self._es_historial_real_activo() else ""
        resultado["summary"] = summary
        self.resultados = list(resultado.get("results", []))
        contextual = self._analizar_contextual(
            self.resultados,
            self.historial.eventos,
        )
        resultado["contextual"] = contextual
        summary["contextual_evaluated"] = int(contextual.get("summary", {}).get("evaluated", 0))
        summary["contextual_elapsed_ms"] = float(
            contextual.get("summary", {}).get("elapsed_ms", 0.0)
        )
        self.resultados_contextuales = list(contextual.get("results", []))
        classification = self._analizar_clasificacion(
            self.resultados,
            self.resultados_contextuales,
            source="historical",
        )
        resultado["classification"] = classification
        summary["classification_evaluated"] = int(
            classification.get("summary", {}).get("evaluated", 0)
        )
        summary["classification_elapsed_ms"] = float(
            classification.get("summary", {}).get("elapsed_ms", 0.0)
        )
        self._guardar_estado_clasificacion(classification)
        self.analysis_completed = True
        self._guardar_cache_analisis_real(suspicion_threshold, resultado)
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
        return {"config_path": str(self.filtro_estadistico.continuos.conf_path), "configs": configs}

    def guardar_configuracion_continua(self, configs: list[dict[str, Any]]) -> dict[str, Any]:
        guardadas = self.filtro_estadistico.continuos.guardar_configuracion(configs)
        if self.analysis_cache_path.exists():
            self.analysis_cache_path.unlink()
        self.realtime_baseline = None
        return {"config_path": str(self.filtro_estadistico.continuos.conf_path), "configs": guardadas}

    def mapa_contextual(self, force: bool = False) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        return self._asegurar_mapa_contextual(force=force)

    def analizar_contexto_actual(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        payload = self._analizar_contextual(self.resultados, self._eventos_combinados())
        self.resultados_contextuales = list(payload.get("results", []))
        classification = self._analizar_clasificacion(
            self.resultados,
            self.resultados_contextuales,
            source="mixed" if not self.realtime_events.empty else "historical",
        )
        self._guardar_estado_clasificacion(classification)
        payload["classification"] = classification
        return payload

    def contexto_actual(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "evaluated": len(self.resultados_contextuales),
            "by_outcome": {},
            "possible_common_cause": 0,
        }
        for item in self.resultados_contextuales:
            outcome = str(item.get("context_outcome", "unexplained"))
            summary["by_outcome"][outcome] = summary["by_outcome"].get(outcome, 0) + 1
            summary["possible_common_cause"] += int(
                bool(item.get("common_cause", {}).get("candidate"))
            )
        return {
            "results": self.resultados_contextuales,
            "summary": summary,
            "map_quality": (
                (self.mapa_contextual_cache or {}).get("refresh", {})
            ),
        }

    def especificacion_clasificacion(self) -> dict[str, Any]:
        specification = deepcopy(self.filtro_clasificacion.especificacion())
        specification["runtime_validation_context"] = self._contexto_validacion_runtime()
        return specification

    def _contexto_validacion_runtime(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        start = None if self.historial is None else self.historial.metadatos.get("start")
        end = None if self.historial is None else self.historial.metadatos.get("end")
        start_dt = pd.to_datetime(start, errors="coerce", utc=True)
        end_dt = pd.to_datetime(end, errors="coerce", utc=True)
        observed_days = (
            float((end_dt - start_dt).total_seconds() / 86400.0)
            if pd.notna(start_dt) and pd.notna(end_dt) and end_dt >= start_dt
            else 0.0
        )
        ground_truth = self.ground_truth_path
        labels_available = bool(ground_truth and ground_truth.exists())
        return {
            "history_source": (
                None
                if self.historial is None
                else self.historial.metadatos.get("source_type")
            ),
            "history_start": start,
            "history_end": end,
            "observed_days": round(observed_days, 3),
            "planned_history_days": 150,
            "history_requirement_met": observed_days >= 150.0,
            "uses_simulated_dataset": False,
            "ground_truth_path": str(ground_truth) if ground_truth else None,
            "real_labels_available": labels_available,
            "operational_status": "provisional_real_month",
            "validation_status": (
                "ready_for_temporal_validation"
                if observed_days >= 150.0 and labels_available
                else "pre_registered_pending_real_validation"
            ),
            "warning": (
                "El clasificador puede operar sobre el mes real, pero sus metricas "
                "finales no se estiman hasta disponer de cobertura temporal y "
                "etiquetas reales suficientes."
            ),
        }

    def analizar_clasificacion_actual(self) -> dict[str, Any]:
        payload = self._analizar_clasificacion(
            self.resultados,
            self.resultados_contextuales,
            source="mixed" if not self.realtime_events.empty else "historical",
        )
        self._guardar_estado_clasificacion(payload)
        return payload

    def clasificacion_actual(self) -> dict[str, Any]:
        if self.ultimo_payload_clasificacion is not None:
            return self.ultimo_payload_clasificacion
        if self.resultados or self.resultados_contextuales:
            return self.analizar_clasificacion_actual()
        specification = self.especificacion_clasificacion()
        return {
            "schema_version": "classifier-results-v1",
            "model": {
                "version": specification.get("model_version"),
                "type": specification.get("model_type"),
                "confidence_kind": specification.get("confidence_kind"),
                "selection_status": specification.get("selection_status"),
            },
            "summary": self._resumen_resultados_clasificacion([]),
            "results": [],
        }

    def _analizar_clasificacion(
        self,
        resultados_estadisticos: list[dict[str, Any]],
        resultados_contextuales: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        return self.filtro_clasificacion.analizar(
            resultados_estadisticos,
            resultados_contextuales,
            source=source,
        )

    def _guardar_estado_clasificacion(self, payload: dict[str, Any]) -> None:
        self.resultados_clasificacion = list(payload.get("results", []))
        self.ultimo_payload_clasificacion = payload
        self.ultima_evaluacion_clasificacion = None

    def _agregar_estado_clasificacion(self, payload: dict[str, Any]) -> None:
        anteriores = self.ultimo_payload_clasificacion or {}
        self.resultados_clasificacion.extend(payload.get("results", []))
        elapsed_ms = float(anteriores.get("summary", {}).get("elapsed_ms", 0.0))
        elapsed_ms += float(payload.get("summary", {}).get("elapsed_ms", 0.0))
        self.ultimo_payload_clasificacion = {
            "schema_version": payload.get(
                "schema_version",
                anteriores.get("schema_version", "classifier-results-v1"),
            ),
            "model": payload.get("model") or anteriores.get("model") or {},
            "summary": self._resumen_resultados_clasificacion(
                self.resultados_clasificacion,
                elapsed_ms=elapsed_ms,
            ),
            "results": self.resultados_clasificacion,
        }
        self.ultima_evaluacion_clasificacion = None

    def _resumen_resultados_clasificacion(
        self,
        resultados: list[dict[str, Any]],
        elapsed_ms: float = 0.0,
    ) -> dict[str, Any]:
        by_class: dict[str, int] = {}
        confidences: list[float] = []
        low_confidence = 0
        for item in resultados:
            diagnostic_class = str(item.get("diagnostic_class", "indeterminate"))
            by_class[diagnostic_class] = by_class.get(diagnostic_class, 0) + 1
            confidence = pd.to_numeric(
                pd.Series([item.get("confidence")]),
                errors="coerce",
            ).iloc[0]
            if pd.notna(confidence):
                numeric_confidence = float(confidence)
                confidences.append(numeric_confidence)
                low_confidence += int(numeric_confidence < 0.70)
        return {
            "evaluated": len(resultados),
            "by_class": by_class,
            "abstained": by_class.get("indeterminate", 0),
            "low_confidence": low_confidence,
            "confidence_mean": (
                round(sum(confidences) / len(confidences), 6)
                if confidences
                else None
            ),
            "elapsed_ms": round(float(elapsed_ms), 3),
        }

    def evaluacion_clasificacion(
        self,
        partition: str = "test",
        force: bool = False,
    ) -> dict[str, Any]:
        requested_partition = str(partition or "test").lower()
        cached_partition = str(
            (self.ultima_evaluacion_clasificacion or {})
            .get("dataset", {})
            .get("split", "")
        ).lower()
        if (
            not force
            and self.ultima_evaluacion_clasificacion is not None
            and cached_partition == requested_partition
        ):
            return self.ultima_evaluacion_clasificacion

        if self.realtime_faults:
            ground_truth = pd.DataFrame(
                [
                    {
                        "entity_id": item.get("entity_id"),
                        "start": item.get("start"),
                        "end": item.get("end"),
                        "label": item.get("label", "sensor_fault"),
                        "root_cause": item.get("fault_type"),
                    }
                    for item in self.realtime_faults
                ]
            )
            payload = self.evaluador_extremo_a_extremo.evaluar(
                ground_truth=ground_truth,
                resultados_estadisticos=self._resultados_realtime(),
                resultados_contextuales=self._resultados_contextuales_realtime(),
                resultados_clasificacion=self._resultados_clasificacion_realtime(),
                partition="all",
                dataset_id="controlled-realtime-real-history",
            )
            payload.setdefault("dataset", {})["requested_split"] = requested_partition
            payload["dataset"]["source"] = "controlled_realtime_faults"
            payload.setdefault("warnings", []).append(
                "La evaluacion controlada solo cubre las clases inyectadas en "
                "tiempo real y no sustituye la validacion final con verdad real."
            )
            self.ultima_evaluacion_clasificacion = payload
            return payload

        ground_truth_path = self.ground_truth_path
        if ground_truth_path is None or not ground_truth_path.exists():
            return self._evaluacion_no_disponible(
                requested_partition,
                (
                    "No existe una base de verdad real asociada al mes cargado. "
                    "Se requieren etiquetas reales o fallas controladas en tiempo real."
                ),
            )
        if not self.analysis_completed:
            return self._evaluacion_no_disponible(
                requested_partition,
                "Primero debe ejecutarse el flujo completo sobre el historial real.",
            )

        ground_truth = pd.read_csv(
            ground_truth_path,
            dtype=str,
            keep_default_na=False,
        )
        correspondence = self._correspondencia_ground_truth(ground_truth)
        if not correspondence["matches"]:
            return self._evaluacion_no_disponible(
                requested_partition,
                correspondence["warning"],
            )
        payload = self.evaluador_extremo_a_extremo.evaluar(
            ground_truth=ground_truth,
            resultados_estadisticos=self.resultados,
            resultados_contextuales=self.resultados_contextuales,
            resultados_clasificacion=self.resultados_clasificacion,
            partition=requested_partition,
            dataset_id="real-month-ground-truth",
            timeline_start=self.historial.metadatos.get("start"),
            timeline_end=self.historial.metadatos.get("end"),
        )
        payload.setdefault("dataset", {})["source"] = "real_labeled_history"
        payload["dataset"]["path"] = str(ground_truth_path)
        payload["dataset"]["correspondence"] = correspondence
        payload["runtime_validation_context"] = self._contexto_validacion_runtime()
        self.ultima_evaluacion_clasificacion = payload
        return payload

    def _correspondencia_ground_truth(
        self,
        ground_truth: pd.DataFrame,
    ) -> dict[str, Any]:
        if self.historial is None or ground_truth.empty:
            return {
                "matches": False,
                "warning": "No hay historial activo o la base de verdad esta vacia.",
            }
        if "entity_id" not in ground_truth:
            return {
                "matches": False,
                "warning": "La base de verdad real no contiene entity_id.",
            }
        timestamp_column = (
            "start"
            if "start" in ground_truth
            else "timestamp"
            if "timestamp" in ground_truth
            else None
        )
        if timestamp_column is None:
            return {
                "matches": False,
                "warning": "La base de verdad real no contiene timestamp ni start.",
            }
        active_entities = set(
            self.historial.entidades["entity_id"].astype(str)
        )
        truth_entities = set(ground_truth["entity_id"].astype(str))
        common_entities = active_entities.intersection(truth_entities)
        truth_times = pd.to_datetime(
            ground_truth[timestamp_column],
            errors="coerce",
            utc=True,
        ).dropna()
        history_times = pd.to_datetime(
            self.historial.eventos["timestamp_dt"],
            errors="coerce",
            utc=True,
        ).dropna()
        temporal_overlap = bool(
            not truth_times.empty
            and not history_times.empty
            and truth_times.max() >= history_times.min()
            and truth_times.min() <= history_times.max()
        )
        matches = bool(common_entities and temporal_overlap)
        return {
            "matches": matches,
            "common_entity_count": len(common_entities),
            "temporal_overlap": temporal_overlap,
            "warning": (
                ""
                if matches
                else "La base de verdad no corresponde en entidades o tiempo "
                "al mes real actualmente cargado."
            ),
        }

    def _evaluacion_no_disponible(
        self,
        partition: str,
        warning: str,
    ) -> dict[str, Any]:
        runtime = self._contexto_validacion_runtime()
        payload = {
            "schema_version": "classifier-evaluation-v1",
            "available": False,
            "status": "not_evaluated",
            "dataset": {
                "id": "real-month-pending-ground-truth",
                "source": "real_history",
                "split": partition,
                "labeled_cases": None,
                "months_available": round(
                    float(runtime.get("observed_days", 0.0)) / 30.4375,
                    3,
                ),
                "months_required": 5,
                "uses_simulated_dataset": False,
            },
            "metric_policy": {
                "gate_rejection_policy": "not_reached",
                "indeterminate_policy": "abstention_and_non_true_positive",
                "fixed_thresholds": True,
            },
            "runtime_validation_context": runtime,
            "warnings": [warning, runtime["warning"]],
        }
        self.ultima_evaluacion_clasificacion = payload
        return payload

    def _analizar_contextual(
        self,
        resultados_estadisticos: list[dict[str, Any]],
        eventos: pd.DataFrame,
    ) -> dict[str, Any]:
        mapa = self._asegurar_mapa_contextual()
        return self.filtro_contextual.analizar(
            sospechas=resultados_estadisticos,
            eventos=eventos,
            entidades=self.historial.entidades,
            mapa=mapa,
        )

    def _asegurar_mapa_contextual(self, force: bool = False) -> dict[str, Any]:
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
        )
        mapa["refresh"]["rebuild_reasons"] = list(
            dict.fromkeys([*reasons, *mapa["refresh"].get("rebuild_reasons", [])])
        )
        self.mapa_contextual_cache = mapa
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

    def reset_realtime(self) -> dict[str, Any]:
        self._reset_realtime()
        return self.current_realtime({})

    def tick(self, request: TickRequest) -> dict[str, Any]:
        nuevos_eventos = self._insertar_tick_realtime(request)
        summary = self._analizar_realtime(request.suspicion_threshold, nuevos_eventos)
        return self.current_realtime(summary)

    def iterations(self, request: IterationsRequest) -> dict[str, Any]:
        nuevos: list[pd.DataFrame] = []
        unidad = UnidadTrabajoIteraciones(request.iteration_count)
        resultado_unidad = unidad.ejecutar(lambda: nuevos.append(self._insertar_tick_realtime(request)))
        nuevos_validos = [frame for frame in nuevos if not frame.empty]
        nuevos_eventos = pd.concat(nuevos_validos, ignore_index=True) if nuevos_validos else pd.DataFrame()
        summary = self._analizar_realtime(request.suspicion_threshold, nuevos_eventos)
        summary["iterations_requested"] = resultado_unidad.solicitadas
        summary["iterations_done"] = resultado_unidad.completadas
        return self.current_realtime(summary)

    def manual_tick(self, request: ManualTickRequest) -> dict[str, Any]:
        nuevos_eventos, skipped = self._insertar_tick_manual(request)
        summary = self._analizar_realtime(request.suspicion_threshold, nuevos_eventos)
        inserted = int(len(nuevos_eventos))
        summary["manual_inserted"] = inserted
        summary["manual_skipped_by_cadence"] = skipped
        return self.current_realtime(summary)

    def _insertar_tick_realtime(self, request: TickRequest) -> pd.DataFrame:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        entidades_validas = set(self.historial.entidades["entity_id"].astype(str))
        seleccion = [entity_id for entity_id in request.selected_entities if entity_id in entidades_validas]
        if not seleccion:
            seleccion = self._entidades_analizables()
        if not seleccion:
            raise ValueError("Selecciona al menos una entidad valida para generar un tick.")

        eventos_actuales = self._eventos_combinados()
        base = eventos_actuales[eventos_actuales["entity_id"].isin(seleccion)].copy()
        if base.empty:
            raise ValueError("No hay eventos historicos para las entidades seleccionadas.")
        ultimo = pd.to_datetime(eventos_actuales["timestamp_dt"], errors="coerce", utc=True).max()
        if self.realtime_current_timestamp:
            ultimo = pd.to_datetime(self.realtime_current_timestamp, errors="coerce", utc=True)
        tick_seconds = max(10, int(request.tick_seconds))
        nuevo_timestamp = ultimo + pd.Timedelta(seconds=tick_seconds)
        self.realtime_tick_index += 1
        self.selected_entities = set(seleccion)
        siguiente = self._generar_tick_simulado(
            eventos=eventos_actuales,
            seleccion=seleccion,
            timestamp=nuevo_timestamp,
            tick_seconds=tick_seconds,
            faults=request.faults,
        )
        if not siguiente.empty:
            self.realtime_events = pd.concat([self.realtime_events, siguiente], ignore_index=True)
            timestamp_real = pd.to_datetime(siguiente["timestamp_dt"], errors="coerce", utc=True).max()
            if pd.notna(timestamp_real):
                nuevo_timestamp = max(nuevo_timestamp, timestamp_real)
        self.realtime_current_timestamp = nuevo_timestamp.isoformat()
        return siguiente

    def _analizar_realtime(self, suspicion_threshold: float, nuevos_eventos: pd.DataFrame) -> dict[str, Any]:
        inicio = perf_counter()
        if nuevos_eventos.empty:
            summary = self._summary_vacio((perf_counter() - inicio) * 1000.0)
            summary["realtime_events"] = int(len(self.realtime_events))
            summary["active_faults"] = len(self.realtime_faults)
            summary["incremental"] = True
            summary["skipped_analysis"] = True
            return summary
        resultado = self.filtro_estadistico.analizar_realtime(
            eventos_historial=self.historial.eventos,
            eventos_realtime=self.realtime_events,
            nuevos_eventos=nuevos_eventos,
            entidades=self.historial.entidades,
            baseline=self._baseline_realtime(),
            suspicion_threshold=suspicion_threshold,
        )
        summary = dict(resultado.get("summary") or {})
        summary["elapsed_ms"] = round((perf_counter() - inicio) * 1000.0, 3)
        summary["realtime_events"] = int(len(self.realtime_events))
        summary["active_faults"] = len(self.realtime_faults)
        nuevos_resultados = list(resultado.get("results", []))
        self.resultados.extend(nuevos_resultados)
        contextual = self._analizar_contextual(
            nuevos_resultados,
            self._eventos_combinados(),
        )
        self.resultados_contextuales.extend(contextual.get("results", []))
        summary["contextual"] = contextual.get("summary", {})
        summary["contextual_elapsed_ms"] = float(
            contextual.get("summary", {}).get("elapsed_ms", 0.0)
        )
        classification = self._analizar_clasificacion(
            nuevos_resultados,
            list(contextual.get("results", [])),
            source="realtime",
        )
        self._agregar_estado_clasificacion(classification)
        self.analysis_completed = True
        summary["classification"] = classification.get("summary", {})
        summary["classification_elapsed_ms"] = float(
            classification.get("summary", {}).get("elapsed_ms", 0.0)
        )
        return summary

    def current_realtime(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        realtime_results = self._resultados_realtime()
        realtime_contextual_results = self._resultados_contextuales_realtime()
        realtime_classification_results = self._resultados_clasificacion_realtime()
        ground_truth_labels: dict[str, int] = {}
        for item in self.realtime_fault_ticks:
            label = str(item.get("label", "unknown"))
            ground_truth_labels[label] = ground_truth_labels.get(label, 0) + 1
        return {
            "current_timestamp": self.realtime_current_timestamp,
            "tick_index": self.realtime_tick_index,
            "events": int(len(self.realtime_events)),
            "ground_truth_rows": len(self.realtime_fault_ticks),
            "ground_truth_labels": ground_truth_labels,
            "results": realtime_results,
            "historical_results": self.resultados,
            "result_count": len(realtime_results),
            "historical_result_count": len(self.resultados),
            "summary": summary or {},
            "historical_summary": summary or {},
            "faults": self.realtime_faults,
            "fault_ticks": self.realtime_fault_ticks,
            "contextual_results": realtime_contextual_results,
            "historical_contextual_results": self.resultados_contextuales,
            "contextual_result_count": len(realtime_contextual_results),
            "classification_results": realtime_classification_results,
            "historical_classification_results": self.resultados_clasificacion,
            "classification_result_count": len(realtime_classification_results),
            "classification_model": self.especificacion_clasificacion(),
        }

    def timeline(self, start: str | None, end: str | None, max_points: int) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            return {"entities": [], "faults": []}

        if not start and not end:
            payload = self._timeline_desde_cache(max_points=max_points)
            payload["faults"] = self.realtime_faults
            return payload
        eventos = self._eventos_combinados()
        if self.selected_entities:
            eventos = eventos[eventos["entity_id"].isin(self.selected_entities)]
        eventos = filtrar_rango(eventos, start, end)
        payload = construir_timeline(eventos, self.historial.entidades, max_points=max_points)
        payload["faults"] = self.realtime_faults
        return payload

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
            "start": metadatos.get("start"),
            "end": metadatos.get("end"),
            "normality_history": normalidad,
        }

    def _summary_vacio(self, elapsed_ms: float) -> dict[str, Any]:
        return {
            "results": 0,
            "suspicious": 0,
            "by_kind": {},
            "top_results": [],
            "elapsed_ms": round(float(elapsed_ms), 3),
            "elapsed_seconds": round(float(elapsed_ms) / 1000.0, 6),
            "timings_by_kind_ms": {},
            "first_detection_delay_seconds": None,
        }

    def _reset_realtime(self) -> None:
        self.realtime_events = pd.DataFrame()
        self.realtime_tick_index = 0
        self.realtime_current_timestamp = None
        self.realtime_faults = []
        self.realtime_fault_ticks = []
        self.selected_entities = set()
        self.resultados = []
        self.resultados_contextuales = []
        self.resultados_clasificacion = []
        self.ultimo_payload_clasificacion = None
        self.ultima_evaluacion_clasificacion = None
        self.analysis_completed = False

    def _timeline_desde_cache(self, max_points: int) -> dict[str, Any]:
        seleccion = tuple(sorted(self.selected_entities))
        clave = json.dumps({"selected": seleccion, "max_points": int(max_points)}, sort_keys=True)
        if clave not in self.timeline_base_cache:
            clave_todos = json.dumps({"selected": (), "max_points": int(max_points)}, sort_keys=True)
            if self.selected_entities and clave_todos in self.timeline_base_cache:
                base_todos = deepcopy(self.timeline_base_cache[clave_todos])
                base_todos["entities"] = [
                    item for item in base_todos.get("entities", [])
                    if item.get("entity_id") in self.selected_entities
                ]
                self.timeline_base_cache[clave] = base_todos
            else:
                eventos_base = self.historial.eventos
                if self.selected_entities:
                    eventos_base = eventos_base[eventos_base["entity_id"].isin(self.selected_entities)]
                self.timeline_base_cache[clave] = construir_timeline(eventos_base, self.historial.entidades, max_points=max_points)
        payload = deepcopy(self.timeline_base_cache[clave])
        if self.realtime_events.empty:
            return payload
        timelines = {item["entity_id"]: item for item in payload.get("entities", [])}
        entidades = self.historial.entidades.set_index("entity_id").to_dict("index") if not self.historial.entidades.empty else {}
        realtime = self.realtime_events
        if self.selected_entities:
            realtime = realtime[realtime["entity_id"].isin(self.selected_entities)]
        for fila in realtime.sort_values("timestamp_dt").to_dict("records"):
            entity_id = str(fila.get("entity_id"))
            timeline = timelines.get(entity_id)
            if not timeline:
                entidad = entidades.get(entity_id, {})
                tipo = str(entidad.get("sensor_kind") or fila.get("sensor_kind") or "multistate")
                timeline = {
                    "entity_id": entity_id,
                    "sensor_kind": tipo,
                    "area_id": "",
                    "chart_kind": "active_percentage" if tipo == "binary" else "state_line",
                    "scale_unit": "%" if tipo == "binary" else "state",
                    "bucket_seconds": None,
                    "points": [],
                    "invalid_gaps": [],
                }
                payload.setdefault("entities", []).append(timeline)
                timelines[entity_id] = timeline
            tipo = str(timeline.get("sensor_kind") or fila.get("sensor_kind") or "")
            valor = pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0]
            point_value = 100.0 if tipo == "binary" and float(valor) >= 0.5 else (0.0 if tipo == "binary" else valor_para_timeline(fila, tipo))
            timeline.setdefault("points", []).append(
                {
                    "timestamp": str(fila.get("timestamp")),
                    "state": str(fila.get("state")),
                    "value": point_value,
                    "is_valid_value": bool(fila.get("is_valid_value", True)),
                }
            )
        for timeline in payload.get("entities", []):
            timeline["points"] = sorted(timeline.get("points", []), key=lambda item: item.get("timestamp") or "")
        return payload

    def _eventos_combinados(self) -> pd.DataFrame:
        if self.historial is None:
            return pd.DataFrame()
        if self.realtime_events.empty:
            return self.historial.eventos.copy()
        return pd.concat([self.historial.eventos, self.realtime_events], ignore_index=True)

    def _baseline_realtime(self) -> dict[str, Any]:
        if self.realtime_baseline is None:
            self.realtime_baseline = self.filtro_estadistico.construir_baseline_realtime(
                eventos=self.historial.eventos,
                entidades=self.historial.entidades,
            )
        return self.realtime_baseline

    def _entidades_analizables(self) -> list[str]:
        if self.historial is None or self.historial.entidades.empty:
            return []
        entidades = self.historial.entidades
        return entidades[entidades["sensor_kind"].isin(["continuous", "binary"])]["entity_id"].astype(str).tolist()

    def _resultados_realtime(self) -> list[dict[str, Any]]:
        if self.realtime_events.empty:
            return []
        inicio = pd.to_datetime(self.realtime_events["timestamp_dt"], errors="coerce", utc=True).min()
        if pd.isna(inicio):
            return []
        resultados = []
        for resultado in self.resultados:
            timestamp = pd.to_datetime(resultado.get("timestamp"), errors="coerce", utc=True)
            if pd.notna(timestamp) and timestamp >= inicio:
                resultados.append(resultado)
        return resultados

    def _resultados_contextuales_realtime(self) -> list[dict[str, Any]]:
        if self.realtime_events.empty:
            return []
        inicio = pd.to_datetime(self.realtime_events["timestamp_dt"], errors="coerce", utc=True).min()
        if pd.isna(inicio):
            return []
        resultados = []
        for resultado in self.resultados_contextuales:
            timestamp = pd.to_datetime(resultado.get("timestamp"), errors="coerce", utc=True)
            if pd.notna(timestamp) and timestamp >= inicio:
                resultados.append(resultado)
        return resultados

    def _resultados_clasificacion_realtime(self) -> list[dict[str, Any]]:
        if self.realtime_events.empty:
            return []
        inicio = pd.to_datetime(
            self.realtime_events["timestamp_dt"],
            errors="coerce",
            utc=True,
        ).min()
        if pd.isna(inicio):
            return []
        resultados = []
        for resultado in self.resultados_clasificacion:
            timestamp = pd.to_datetime(
                resultado.get("timestamp"),
                errors="coerce",
                utc=True,
            )
            if pd.notna(timestamp) and timestamp >= inicio:
                resultados.append(resultado)
        return resultados

    def _generar_tick_simulado(
        self,
        eventos: pd.DataFrame,
        seleccion: list[str],
        timestamp: pd.Timestamp,
        tick_seconds: int,
        faults: list[dict[str, Any]],
    ) -> pd.DataFrame:
        entidades = self.historial.entidades.set_index("entity_id").to_dict("index") if self.historial is not None else {}
        normalidades = self._normalidad_por_entidad()
        fallas = {
            str(fault.get("entity_id")): fault
            for fault in faults
            if fault.get("enabled") and str(fault.get("entity_id")) in seleccion
        }
        filas: list[dict[str, Any]] = []
        for entity_id in seleccion:
            grupo = eventos[eventos["entity_id"].eq(entity_id)].sort_values("timestamp_dt")
            if grupo.empty:
                continue
            entidad = entidades.get(entity_id, {})
            historico = self._grupo_historico_entidad(entity_id, grupo)
            fila = self._proyectar_siguiente_fila(historico, entidad, timestamp, normalidades.get(entity_id, {}))
            fault = fallas.get(entity_id)
            if fault:
                resultado = self.inyector_fallas.aplicar(
                    fila=fila,
                    grupo=historico,
                    entidad=entidad,
                    fault=fault,
                    tick_index=self.realtime_tick_index,
                    tick_seconds=tick_seconds,
                )
                fila = resultado.fila
                if resultado.aplicada:
                    self._registrar_falla_tick(
                        entity_id=entity_id,
                        fault_type=resultado.fault_type,
                        timestamp=fila["timestamp_dt"],
                        tick_seconds=tick_seconds,
                        clean_state=resultado.clean_state,
                        observed_state=resultado.observed_state,
                        metadata=resultado.metadata,
                    )
            filas.append(fila)
        if not filas:
            return pd.DataFrame(columns=eventos.columns)
        return pd.DataFrame(filas).reindex(columns=eventos.columns)

    def _insertar_tick_manual(self, request: ManualTickRequest) -> tuple[pd.DataFrame, int]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        entidades_validas = set(self.historial.entidades["entity_id"].astype(str))
        seleccion = [entity_id for entity_id in request.selected_entities if entity_id in entidades_validas]
        if not seleccion:
            seleccion = [
                entity_id
                for entity_id in request.sensor_states
                if entity_id in entidades_validas
            ] or self._entidades_analizables()
        if not seleccion:
            raise ValueError("Selecciona al menos una entidad valida para simular.")

        eventos_actuales = self._eventos_combinados()
        ultimo = pd.to_datetime(eventos_actuales["timestamp_dt"], errors="coerce", utc=True).max()
        if self.realtime_current_timestamp:
            ultimo = pd.to_datetime(self.realtime_current_timestamp, errors="coerce", utc=True)
        tick_seconds = max(10, int(request.tick_seconds))
        nuevo_timestamp = ultimo + pd.Timedelta(seconds=tick_seconds)
        self.realtime_tick_index += 1
        self.selected_entities = set(seleccion)

        entidades = self.historial.entidades.set_index("entity_id").to_dict("index")
        filas = []
        omitidas = 0
        for entity_id in seleccion:
            grupo = eventos_actuales[eventos_actuales["entity_id"].eq(entity_id)].sort_values("timestamp_dt")
            if grupo.empty or entity_id not in request.sensor_states:
                omitidas += 1
                continue
            cadencia = self._cadencia_entidad_segundos(entity_id)
            ultimo_reporte = pd.to_datetime(grupo["timestamp_dt"], errors="coerce", utc=True).max()
            if pd.notna(ultimo_reporte) and (nuevo_timestamp - ultimo_reporte).total_seconds() + 1e-9 < cadencia:
                omitidas += 1
                continue
            fila = grupo.tail(1).iloc[0].to_dict()
            fila["timestamp_dt"] = nuevo_timestamp
            fila["timestamp"] = nuevo_timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            filas.append(self._aplicar_estado_manual(fila, entidades.get(entity_id, {}), request.sensor_states[entity_id]))

        nuevos_eventos = pd.DataFrame(filas).reindex(columns=eventos_actuales.columns) if filas else pd.DataFrame(columns=eventos_actuales.columns)
        if not nuevos_eventos.empty:
            self.realtime_events = pd.concat([self.realtime_events, nuevos_eventos], ignore_index=True)
        self.realtime_current_timestamp = nuevo_timestamp.isoformat()
        return nuevos_eventos, omitidas

    def _aplicar_estado_manual(self, fila: dict[str, Any], entidad: dict[str, Any], estado: Any) -> dict[str, Any]:
        tipo = str(entidad.get("sensor_kind") or fila.get("sensor_kind") or "")
        fila["is_valid_value"] = True
        if tipo == "binary":
            activo = str(estado).strip().lower() in {"on", "open", "home", "active", "detected", "true", "1"}
            fila["value"] = 1.0 if activo else 0.0
            fila["state"] = "on" if activo else "off"
            return fila
        valor = float(pd.to_numeric(pd.Series([estado]), errors="coerce").fillna(0.0).iloc[0])
        fila["value"] = valor
        fila["state"] = f"{valor:.6f}"
        return fila

    def _cadencia_entidad_segundos(self, entity_id: str) -> float:
        grupo = self._grupo_historico_entidad(entity_id, pd.DataFrame()).sort_values("timestamp_dt")
        if grupo.empty:
            return 60.0
        gaps = pd.to_datetime(grupo["timestamp_dt"], errors="coerce", utc=True).diff().dt.total_seconds().dropna()
        gaps = gaps[gaps.gt(0)]
        if gaps.empty:
            return 60.0
        return max(10.0, float(gaps.median()))

    def _proyectar_siguiente_fila(
        self,
        grupo: pd.DataFrame,
        entidad: dict[str, Any],
        timestamp: pd.Timestamp,
        normalidad: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ultima = grupo.tail(1).iloc[0].to_dict()
        fila = dict(ultima)
        fila["timestamp_dt"] = timestamp
        fila["timestamp"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        tipo = str(entidad.get("sensor_kind") or ultima.get("sensor_kind") or "")
        if tipo == "continuous":
            valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
            normalidad = normalidad or {}
            lower = float(normalidad.get("p05", valores.quantile(0.05) if not valores.empty else 0.0))
            upper = float(normalidad.get("p95", valores.quantile(0.95) if not valores.empty else 0.0))
            normales = valores[valores.between(min(lower, upper), max(lower, upper))]
            fuente = normales if not normales.empty else valores
            valor = float(fuente.iloc[(self.realtime_tick_index - 1) % len(fuente)]) if not fuente.empty else float(normalidad.get("median", 0.0))
            fila["value"] = valor
            fila["state"] = f"{valor:.6f}"
        elif tipo == "binary":
            normalidad = normalidad or {}
            valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
            ratio = normalidad.get("active_ratio")
            if ratio is None:
                ratio = float((valores >= 0.5).mean()) if not valores.empty else 0.0
            ratio = max(0.0, min(1.0, float(ratio)))
            anterior = int((self.realtime_tick_index - 1) * ratio)
            actual = int(self.realtime_tick_index * ratio)
            valor = 1.0 if actual > anterior else 0.0
            fila["value"] = 1.0 if valor >= 0.5 else 0.0
            fila["state"] = self._estado_binario(fila["value"])
        return fila

    def _registrar_falla_tick(
        self,
        entity_id: str,
        fault_type: str,
        timestamp: pd.Timestamp,
        tick_seconds: int,
        clean_state: str,
        observed_state: str,
        metadata: dict[str, Any],
    ) -> None:
        start = pd.to_datetime(timestamp, errors="coerce", utc=True)
        end = start + pd.Timedelta(seconds=max(10, tick_seconds))
        evento = {
            "entity_id": entity_id,
            "fault_type": fault_type,
            "label": "sensor_fault",
            "is_reliable": False,
            "clean_state": clean_state,
            "observed_state": observed_state,
            "metadata": metadata,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tick_index": self.realtime_tick_index,
        }
        self.realtime_fault_ticks.append(evento)
        ultima = self.realtime_faults[-1] if self.realtime_faults else None
        if ultima and ultima["entity_id"] == entity_id and ultima["fault_type"] == fault_type:
            ultima["end"] = evento["end"]
            ultima["observed_state"] = observed_state
        else:
            self.realtime_faults.append(
                {k: evento[k] for k in ("entity_id", "fault_type", "label", "start", "end", "clean_state", "observed_state")}
            )

    def _escala_continua(self, grupo: pd.DataFrame) -> float:
        valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
        if valores.empty:
            return 1.0
        iqr = float(valores.quantile(0.75) - valores.quantile(0.25))
        std = float(valores.std()) if len(valores) > 1 else 0.0
        return max(abs(iqr), abs(std), abs(float(valores.median())) * 0.05, 1.0)

    def _grupo_historico_entidad(self, entity_id: str, fallback: pd.DataFrame) -> pd.DataFrame:
        if self.historial is None or self.historial.eventos.empty or not entity_id:
            return fallback
        grupo = self.historial.eventos[self.historial.eventos["entity_id"].eq(entity_id)]
        return grupo if not grupo.empty else fallback

    def _estado_binario(self, value: float) -> str:
        return "on" if float(value) >= 0.5 else "off"

    def _es_historial_real_activo(self) -> bool:
        return self.historial is not None and self.historial.metadatos.get("source_type") == "real"

    def _identidad_historial_real(self, suspicion_threshold: float) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        stat = self.real_history_path.stat()
        conf_path = Path(os.getenv("CONTINUOUS_CONF_PATH", ""))
        classifier_specification = self.filtro_clasificacion.especificacion()
        return {
            "cache_version": 15,
            "source_path": str(self.real_history_path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_sha256": self._sha256_archivo(self.real_history_path),
            "continuous_conf_sha256": self._sha256_archivo(conf_path) if conf_path.is_file() else "",
            "suspicion_threshold": round(float(suspicion_threshold), 6),
            "classifier_version": classifier_specification.get("model_version"),
            "classifier_implementation_sha256": (
                classifier_specification.get("artifacts", {}).get(
                    "implementation_sha256"
                )
            ),
            "classifier_configuration_sha256": (
                classifier_specification.get("artifacts", {}).get(
                    "configuration_sha256"
                )
            ),
        }

    def _identidad_normalidad_real(self) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        stat = self.real_history_path.stat()
        return {
            "cache_version": 2,
            "source_path": str(self.real_history_path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_sha256": self._sha256_archivo(self.real_history_path),
        }

    def _normality_cache_path(self, identidad: dict[str, Any]) -> Path:
        nombre = f"normality_{identidad['source_sha256'][:16]}.json"
        return self.analysis_cache_path.parent / nombre

    def _sha256_archivo(self, ruta: Path) -> str:
        return hashlib.sha256(ruta.read_bytes()).hexdigest()

    def _asegurar_normalidad_persistida(self) -> dict[str, Any]:
        if self.historial is None:
            return {"exists": False, "rows": 0}
        identidad = self._identidad_normalidad_real()
        if identidad is None:
            return {
                "exists": True,
                "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
                "mode": "memory",
            }
        path = self._normality_cache_path(identidad)
        if not self._normalidad_cache_valida(path, identidad):
            payload = {
                "identity": identidad,
                "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
                "entity_count": int(self.historial.metadatos.get("entity_count", len(self.historial.entidades))),
                "by_kind": self.historial.entidades["sensor_kind"].value_counts().to_dict(),
                "entities": self._normalidad_por_entidad(),
            }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as archivo:
                    json.dump(serializable_json(payload), archivo, ensure_ascii=True)
            except OSError:
                return {"exists": False, "rows": 0, "mode": "disk_error"}
        return {
            "exists": True,
            "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
            "mode": "persisted",
            "path": str(path),
        }

    def _normalidad_cache_valida(self, path: Path, identidad: dict[str, Any]) -> bool:
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as archivo:
                payload = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("identity") == identidad and isinstance(payload.get("entities"), dict)

    def _leer_normalidad_persistida(self, status: dict[str, Any]) -> dict[str, Any]:
        path = status.get("path")
        if not path:
            return status
        try:
            with Path(path).open("r", encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return status

    def _normalidad_por_entidad(self) -> dict[str, Any]:
        if self.historial is None or self.historial.eventos.empty:
            return {}
        entidades = self.historial.entidades.set_index("entity_id").to_dict("index") if not self.historial.entidades.empty else {}
        salida: dict[str, Any] = {}
        for entity_id, grupo in self.historial.eventos.groupby("entity_id", sort=True):
            entity_id = str(entity_id)
            entidad = entidades.get(entity_id, {})
            tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
            base: dict[str, Any] = {
                "sensor_kind": tipo,
                "tipo_interno": str(entidad.get("tipo_interno", "")),
                "rows": int(len(grupo)),
            }
            if tipo == "continuous":
                valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
                if valores.empty:
                    base.update({"median": 0.0, "mad": 0.0, "scaled_mad": 1.0})
                else:
                    mediana = float(valores.median())
                    mad = float((valores - mediana).abs().median())
                    escala = max(1.4826 * mad, 0.01)
                    base.update(
                        {
                            "median": mediana,
                            "mad": mad,
                            "scaled_mad": escala,
                            "p05": float(valores.quantile(0.05)),
                            "p95": float(valores.quantile(0.95)),
                        }
                    )
            elif tipo == "binary":
                trabajo = grupo.sort_values("timestamp_dt").copy()
                valores = pd.to_numeric(trabajo["value"], errors="coerce").fillna(0.0)
                activos = int((valores >= 0.5).sum())
                tiempos = pd.to_datetime(trabajo["timestamp_dt"], errors="coerce", utc=True)
                duraciones = tiempos.shift(-1).sub(tiempos).dt.total_seconds()
                mediana_gap = duraciones[duraciones.gt(0)].median()
                duraciones = duraciones.fillna(float(mediana_gap) if pd.notna(mediana_gap) else 0.0).clip(lower=0.0)
                total = float(duraciones.sum())
                activos_segundos = float(duraciones[valores >= 0.5].sum())
                base.update(
                    {
                        "active_ratio": activos_segundos / total if total > 0 else float(activos / len(valores)) if len(valores) else 0.0,
                        "on_count": activos,
                        "off_count": int(len(valores) - activos),
                    }
                )
            else:
                estados = grupo["state"].astype(str)
                base.update({"states": estados.value_counts().to_dict(), "unique_states": int(estados.nunique())})
            salida[entity_id] = base
        return salida

    def _leer_cache_analisis_real(self, suspicion_threshold: float) -> dict[str, Any] | None:
        identidad = self._identidad_historial_real(suspicion_threshold)
        if identidad is None or not self.analysis_cache_path.exists():
            return None
        try:
            with self.analysis_cache_path.open("r", encoding="utf-8") as archivo:
                cache = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return None
        if cache.get("identity") != identidad:
            return None
        payload = cache.get("payload")
        if not isinstance(payload, dict):
            return None
        payload = dict(payload)
        payload["cache"] = {"hit": True, "path": str(self.analysis_cache_path)}
        summary = dict(payload.get("summary") or {})
        summary["cache_hit"] = True
        summary["cache_path"] = str(self.analysis_cache_path)
        payload["summary"] = summary
        return payload

    def _guardar_cache_analisis_real(self, suspicion_threshold: float, resultado: dict[str, Any]) -> None:
        identidad = self._identidad_historial_real(suspicion_threshold)
        if identidad is None:
            return
        payload = serializable_json(resultado)
        payload["cache"] = {"hit": False, "path": str(self.analysis_cache_path)}
        try:
            self.analysis_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.analysis_cache_path.open("w", encoding="utf-8") as archivo:
                json.dump({"identity": identidad, "payload": payload}, archivo, ensure_ascii=True)
        except OSError:
            return


def serializable_json(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {str(clave): serializable_json(contenido) for clave, contenido in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [serializable_json(item) for item in valor]
    if isinstance(valor, pd.Timestamp):
        return None if pd.isna(valor) else valor.isoformat()
    if valor is None or isinstance(valor, (str, int, float, bool)):
        return valor
    if hasattr(valor, "item"):
        try:
            return serializable_json(valor.item())
        except (TypeError, ValueError):
            pass
    if hasattr(valor, "isoformat"):
        try:
            return valor.isoformat()
        except (TypeError, ValueError):
            pass
    return str(valor)


def crear_app() -> FastAPI:
    api = FastAPI(
        title="Artefacto de tesis",
        description=(
            "API para carga de historial, filtros estadistico-temporal y "
            "contextual espacio-temporal, y clasificacion diagnostica reproducible."
        ),
        version="0.4.0",
    )
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return api


app = crear_app()
ARTIFACT_ROOT = Path(__file__).resolve().parents[2]
state = EstadoAplicacion(
    cargador=CargadorHistorial(),
    filtro_estadistico=FiltroEstadistico(),
    static_history_dir=Path(
        os.getenv("STATIC_HISTORY_DIR", str(ARTIFACT_ROOT / "data/static_history"))
    ),
    real_history_path=Path(
        os.getenv(
            "REAL_HISTORY_PATH",
            str(ARTIFACT_ROOT / "data/datosreales/history.csv"),
        )
    ),
    analysis_cache_path=Path(
        os.getenv(
            "ANALYSIS_CACHE_PATH",
            str(ARTIFACT_ROOT / "data/cache/real_history_analysis.json"),
        )
    ),
    context_map_cache_path=Path(
        os.getenv(
            "CONTEXT_MAP_CACHE_PATH",
            str(ARTIFACT_ROOT / "data/cache/contextual_map.json"),
        )
    ),
    ground_truth_path=Path(
        os.getenv(
            "GROUND_TRUTH_PATH",
            str(ARTIFACT_ROOT / "data/datosreales/ground_truth.csv"),
        )
    ),
)


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mensaje": "API operativa",
        "static_history_dir": str(state.static_history_dir),
        "real_history_path": str(state.real_history_path),
    }


@app.get("/history/static/status", tags=["history"])
def history_static_status() -> dict[str, Any]:
    return state.status()


@app.post("/history/static/ensure", tags=["history"])
def history_static_ensure() -> dict[str, Any]:
    try:
        return state.asegurar_canonico()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/history/real/load", tags=["history"])
def history_real_load() -> dict[str, Any]:
    try:
        return state.cargar_real()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/history/real/upload", tags=["history"])
async def history_real_upload(request: Request) -> dict[str, Any]:
    try:
        return state.subir_real(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/history/analyze", tags=["history"])
def history_analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        return state.analizar_historial(request.suspicion_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/history/normalization/preview", tags=["history"])
def history_normalization_preview() -> dict[str, Any]:
    try:
        return state.vista_normalizacion()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/config/continuous", tags=["config"])
def continuous_config() -> dict[str, Any]:
    try:
        return state.configuracion_continua()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/config/continuous", tags=["config"])
def save_continuous_config(request: ContinuousConfigRequest) -> dict[str, Any]:
    try:
        return state.guardar_configuracion_continua(request.configs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/context/map", tags=["contextual"])
def contextual_map() -> dict[str, Any]:
    try:
        return state.mapa_contextual()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/context/map/rebuild", tags=["contextual"])
def rebuild_contextual_map() -> dict[str, Any]:
    try:
        return state.mapa_contextual(force=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/context/analyze", tags=["contextual"])
def analyze_context() -> dict[str, Any]:
    try:
        return state.analizar_contexto_actual()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/context/results", tags=["contextual"])
def contextual_results() -> dict[str, Any]:
    return state.contexto_actual()


@app.get("/classifier/specification", tags=["classifier"])
def classifier_specification() -> dict[str, Any]:
    return state.especificacion_clasificacion()


@app.post("/classifier/analyze", tags=["classifier"])
def analyze_classifier() -> dict[str, Any]:
    try:
        return state.analizar_clasificacion_actual()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/classifier/results", tags=["classifier"])
def classifier_results() -> dict[str, Any]:
    return state.clasificacion_actual()


@app.get("/classifier/evaluation", tags=["classifier"])
def classifier_evaluation(
    partition: str = Query(default="test"),
) -> dict[str, Any]:
    try:
        return state.evaluacion_clasificacion(partition=partition)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/classifier/evaluate", tags=["classifier"])
def evaluate_classifier(
    request: ClassifierEvaluationRequest,
) -> dict[str, Any]:
    try:
        return state.evaluacion_clasificacion(
            partition=request.partition,
            force=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/realtime/reset", tags=["realtime"])
def realtime_reset() -> dict[str, Any]:
    return state.reset_realtime()


@app.post("/realtime/tick", tags=["realtime"])
def realtime_tick(request: TickRequest) -> dict[str, Any]:
    try:
        return state.tick(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/realtime/iterations", tags=["realtime"])
def realtime_iterations(request: IterationsRequest) -> dict[str, Any]:
    try:
        return state.iterations(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/manual/tick", tags=["manual"])
def manual_tick(request: ManualTickRequest) -> dict[str, Any]:
    try:
        return state.manual_tick(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/realtime/current", tags=["realtime"])
def realtime_current() -> dict[str, Any]:
    return state.current_realtime()


@app.get("/realtime/timeline", tags=["realtime"])
def realtime_timeline(
    start: str | None = None,
    end: str | None = None,
    max_points: int = Query(default=1000, ge=10, le=10000),
) -> dict[str, Any]:
    return state.timeline(start=start, end=end, max_points=max_points)


def filtrar_rango(eventos: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if eventos.empty:
        return eventos
    trabajo = eventos.copy()
    if start:
        start_dt = pd.to_datetime(start, errors="coerce", utc=True)
        if pd.notna(start_dt):
            trabajo = trabajo[trabajo["timestamp_dt"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.notna(end_dt):
            trabajo = trabajo[trabajo["timestamp_dt"] <= end_dt]
    if not start and not end:
        limite = trabajo["timestamp_dt"].max() - pd.Timedelta(days=30)
        trabajo = trabajo[trabajo["timestamp_dt"] >= limite]
    return trabajo


def construir_timeline(eventos: pd.DataFrame, entidades: pd.DataFrame, max_points: int) -> dict[str, Any]:
    if eventos.empty:
        return {"entities": [], "faults": []}
    lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
    timelines = []
    for entity_id, grupo in eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
        entidad = lookup.get(str(entity_id), {})
        tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
        if tipo == "binary":
            points, bucket_seconds = timeline_binario(grupo, max_points=max_points)
            chart_kind = "active_percentage"
            scale_unit = "%"
            gaps = []
        else:
            sampleado = muestrear_timeline_con_cortes(grupo, max_points)
            points = [
                {
                    "timestamp": str(fila["timestamp"]),
                    "state": str(fila["state"]),
                    "value": valor_para_timeline(fila, tipo),
                    "is_valid_value": bool(fila.get("is_valid_value", True)),
                }
                for fila in sampleado.to_dict("records")
            ]
            bucket_seconds = None
            chart_kind = "state_line"
            scale_unit = "state"
            gaps = timeline_gaps_invalidos(grupo)
        timelines.append(
            {
                "entity_id": str(entity_id),
                "sensor_kind": tipo,
                "area_id": "",
                "chart_kind": chart_kind,
                "scale_unit": scale_unit,
                "bucket_seconds": bucket_seconds,
                "points": points,
                "gaps": gaps,
                "categories": {},
            }
        )
    return {"entities": timelines, "faults": []}


def reducir_puntos(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step].head(max_points)


def muestrear_timeline_con_cortes(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    trabajo = frame.sort_values("timestamp_dt").copy()
    validos = trabajo[trabajo.get("is_valid_value", True).astype(bool)] if "is_valid_value" in trabajo else trabajo
    base = reducir_puntos(validos, max_points)
    if "is_valid_value" not in trabajo.columns or trabajo["is_valid_value"].all():
        return base.sort_values("timestamp_dt")
    posiciones = set()
    for posicion, valido in enumerate(trabajo["is_valid_value"].astype(bool).tolist()):
        if not valido:
            posiciones.update([max(0, posicion - 1), posicion, min(len(trabajo) - 1, posicion + 1)])
    vecinos = trabajo.iloc[sorted(posiciones)]
    return pd.concat([base, vecinos]).drop_duplicates(["entity_id", "timestamp"]).sort_values("timestamp_dt")


def timeline_gaps_invalidos(frame: pd.DataFrame) -> list[dict[str, str]]:
    if "is_valid_value" not in frame.columns or frame.empty:
        return []
    gaps = []
    inicio = None
    for fila in frame.sort_values("timestamp_dt").to_dict("records"):
        valido = bool(fila.get("is_valid_value", True))
        if not valido and inicio is None:
            inicio = str(fila["timestamp"])
        elif valido and inicio is not None:
            gaps.append({"start": inicio, "end": str(fila["timestamp"]), "label": "sin lectura valida"})
            inicio = None
    return gaps


def timeline_binario(frame: pd.DataFrame, max_points: int) -> tuple[list[dict[str, Any]], int]:
    trabajo = frame.sort_values("timestamp_dt").copy()
    if trabajo.empty:
        return [], 60
    inicio = trabajo["timestamp_dt"].min()
    fin = trabajo["timestamp_dt"].max()
    duracion = max((fin - inicio).total_seconds(), 60.0)
    bucket_seconds = max(60, int(duracion / max(max_points, 1)))
    trabajo["bucket"] = ((trabajo["timestamp_dt"] - inicio).dt.total_seconds() // bucket_seconds).astype(int)
    points = []
    for _, grupo in trabajo.groupby("bucket", sort=True):
        activos = pd.to_numeric(grupo["value"], errors="coerce").fillna(0.0)
        porcentaje = float(activos.mean() * 100.0)
        points.append(
            {
                "timestamp": str(grupo["timestamp"].iloc[-1]),
                "state": str(grupo["state"].iloc[-1]),
                "value": round(porcentaje, 4),
                "activations": int(((activos.shift(1).fillna(0) == 0) & (activos == 1)).sum()),
            }
        )
    return points[:max_points], bucket_seconds


def valor_para_timeline(fila: dict[str, Any], tipo: str) -> float:
    if not bool(fila.get("is_valid_value", True)):
        return None
    if tipo == "continuous":
        valor = fila.get("value")
        if valor is not None and not pd.isna(valor):
            return float(valor)
        return float(pd.to_numeric(pd.Series([fila.get("state")]), errors="coerce").fillna(0).iloc[0])
    return 0.0
