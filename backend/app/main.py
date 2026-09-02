from __future__ import annotations

import json
import os
import hashlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from time import time
from typing import Any, Callable
from urllib.parse import unquote
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .Cargar_Historial.Cargar_Historial import CargadorHistorial, ResultadoHistorial
from .Filtro_Clasificacion import ClasificadorDiagnostico, EvaluadorExtremoAExtremo
from .Filtro_Contextual import FiltroContextualEspacioTemporal
from .Filtro_Estadistico.Filtro import FiltroEstadistico
from .Reproduccion_Validacion import MotorReproduccionValidacion
DEFAULT_HISTORY_LOAD_SECONDS_PER_MIB = float(
    os.getenv("HISTORY_LOAD_SECONDS_PER_MIB", "0.22")
)


class AnalyzeRequest(BaseModel):
    suspicion_threshold: float = 0.6


class ClassifierEvaluationRequest(BaseModel):
    partition: str = "test"


class ContinuousConfigRequest(BaseModel):
    configs: list[dict[str, Any]] = Field(default_factory=list)


class ValidationReplayAdvanceRequest(BaseModel):
    batch_size: int = Field(default=1000, ge=1, le=20000)
    suspicion_threshold: float = Field(default=0.6, ge=0.01, le=0.99)


@dataclass
class EstadoAplicacion:
    cargador: CargadorHistorial
    filtro_estadistico: FiltroEstadistico
    static_history_dir: Path
    real_history_path: Path
    analysis_cache_path: Path
    validation_history_path: Path | None = None
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
    historial: ResultadoHistorial | None = None
    resultados: list[dict[str, Any]] = field(default_factory=list)
    timeline_base_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    mapa_contextual_cache: dict[str, Any] | None = None
    resultados_contextuales: list[dict[str, Any]] = field(default_factory=list)
    resultados_clasificacion: list[dict[str, Any]] = field(default_factory=list)
    ultimo_payload_clasificacion: dict[str, Any] | None = None
    ultima_evaluacion_clasificacion: dict[str, Any] | None = None
    analysis_completed: bool = False
    future_baseline_template: dict[str, Any] | None = field(
        default=None, repr=False
    )
    future_context_map_template: dict[str, Any] | None = field(
        default=None, repr=False
    )
    future_context_events_template: pd.DataFrame | None = field(
        default=None, repr=False
    )
    future_evaluation_prepared_at: str | None = field(
        default=None, repr=False
    )
    history_load_jobs: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    history_load_jobs_lock: Any = field(default_factory=Lock, repr=False)
    history_load_execution_lock: Any = field(default_factory=Lock, repr=False)
    history_load_seconds_per_byte: float | None = field(
        default_factory=lambda: DEFAULT_HISTORY_LOAD_SECONDS_PER_MIB / (1024 * 1024),
        repr=False,
    )
    historial_validacion: ResultadoHistorial | None = field(default=None, repr=False)
    eventos_validacion_ordenados: pd.DataFrame | None = field(default=None, repr=False)
    validation_baseline: dict[str, Any] | None = field(default=None, repr=False)
    validation_context_map: dict[str, Any] | None = field(default=None, repr=False)
    validation_context_events: pd.DataFrame | None = field(default=None, repr=False)
    validation_results: list[dict[str, Any]] = field(default_factory=list, repr=False)
    validation_contextual_results: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    validation_classification_results: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    validation_replay_state: dict[str, Any] = field(default_factory=dict, repr=False)
    validation_replay_lock: Any = field(default_factory=Lock, repr=False)

    def status(self, autoload: bool = True) -> dict[str, Any]:
        if self.historial is None and autoload:
            if self.real_history_path.exists():
                try:
                    self.historial = self.cargador.cargar_real(self.real_history_path)
                    self._preparar_continuidad_historica()
                except Exception:
                    pass
            elif (self.static_history_dir / "events.csv").exists():
                try:
                    self.historial = self.cargador.cargar_canonico(self.static_history_dir)
                    self._preparar_continuidad_historica()
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
                "future_evaluation_ready": False,
                "real_history_available": self.real_history_path.exists(),
                "real_history_size_bytes": (
                    self.real_history_path.stat().st_size
                    if self.real_history_path.exists()
                    else 0
                ),
            }
        return self._status_desde_historial(self.historial)

    def cargar_real(
        self,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, Any]:
        def report_loader(phase: str, progress: float, message: str) -> None:
            if progress_callback is not None:
                progress_callback(phase, 2.0 + progress * 0.88, message)

        if progress_callback is not None:
            progress_callback("preparing", 1.0, "Preparando la carga del historico")
        self.historial = self.cargador.cargar_real(
            self.real_history_path,
            progress_callback=report_loader,
        )
        self.future_baseline_template = None
        self.future_context_map_template = None
        self.future_context_events_template = None
        self.future_evaluation_prepared_at = None
        if progress_callback is not None:
            progress_callback("resetting_state", 92.0, "Reiniciando el estado de analisis")
        self._reset_analysis()
        self._limpiar_reproduccion_validacion()
        self.timeline_base_cache = {}
        self.mapa_contextual_cache = None
        if progress_callback is not None:
            progress_callback("persisting_normality", 94.0, "Persistiendo la base de normalidad")
        self._asegurar_normalidad_persistida()
        if progress_callback is not None:
            progress_callback(
                "preparing_future_evaluation",
                96.0,
                "Preparando el estado causal al final del historico",
            )
        self._preparar_continuidad_historica()
        if progress_callback is not None:
            progress_callback("finalizing", 99.0, "Preparando el estado para la interfaz")
        return self._status_desde_historial(self.historial)

    def subir_real(self, contenido: bytes) -> dict[str, Any]:
        if not contenido:
            raise ValueError("El archivo de historial esta vacio.")
        self.real_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.real_history_path.write_bytes(contenido)
        return self.cargar_real()

    def iniciar_carga_real(self) -> dict[str, Any]:
        if not self.real_history_path.exists():
            raise FileNotFoundError(f"No se encontro el historial real: {self.real_history_path}")
        job_id = self._crear_job_carga(self.real_history_path.stat().st_size)
        self._iniciar_hilo_carga(job_id)
        return self.estado_carga_real(job_id)

    def iniciar_subida_real(self, contenido: bytes) -> dict[str, Any]:
        if not contenido:
            raise ValueError("El archivo de historial esta vacio.")
        job_id = self._crear_job_carga(len(contenido), phase="saving_upload")
        temporary_path = self.real_history_path.with_suffix(
            f"{self.real_history_path.suffix}.{job_id}.uploading"
        )
        try:
            self.real_history_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path.write_bytes(contenido)
            temporary_path.replace(self.real_history_path)
        except Exception as exc:
            temporary_path.unlink(missing_ok=True)
            self._fallar_job_carga(job_id, exc)
            raise
        self._actualizar_job_carga(
            job_id,
            phase="upload_saved",
            progress=1.0,
            message="Archivo recibido; iniciando lectura y normalizacion",
        )
        self._iniciar_hilo_carga(job_id)
        return self.estado_carga_real(job_id)

    def estado_carga_real(self, job_id: str) -> dict[str, Any]:
        with self.history_load_jobs_lock:
            job = self.history_load_jobs.get(job_id)
            if job is None:
                raise KeyError(f"No existe el trabajo de carga: {job_id}")
            payload = dict(job)

        now = time()
        started_epoch = payload.pop("_started_epoch", None)
        completed_epoch = payload.pop("_completed_epoch", None)
        if started_epoch is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, (completed_epoch or now) - started_epoch)
        payload["elapsed_seconds"] = round(elapsed, 3)

        progress = float(payload.get("progress", 0.0))
        estimated_total = payload.get("estimated_total_seconds")
        if payload.get("status") == "completed":
            payload["eta_seconds"] = 0.0
        else:
            dynamic_total = elapsed * 100.0 / progress if progress >= 2.0 else None
            candidates = [
                float(value)
                for value in (estimated_total, dynamic_total)
                if value is not None
            ]
            effective_total = max(candidates) if candidates else None
            payload["estimated_total_seconds"] = (
                None if effective_total is None else round(effective_total, 3)
            )
            payload["eta_seconds"] = (
                None
                if effective_total is None
                else round(max(0.0, effective_total - elapsed), 3)
            )
        return payload

    def _crear_job_carga(
        self,
        file_size_bytes: int,
        phase: str = "queued",
    ) -> str:
        with self.history_load_jobs_lock:
            active = next(
                (
                    job
                    for job in self.history_load_jobs.values()
                    if job.get("status") in {"queued", "running"}
                ),
                None,
            )
            if active is not None:
                raise ValueError("Ya existe una carga de historial en ejecucion.")

            job_id = uuid4().hex
            estimated_total = None
            if self.history_load_seconds_per_byte is not None:
                estimated_total = round(
                    max(1.0, file_size_bytes * self.history_load_seconds_per_byte),
                    3,
                )
            self.history_load_jobs[job_id] = {
                "job_id": job_id,
                "status": "queued",
                "phase": phase,
                "progress": 0.0,
                "message": "Carga en espera",
                "file_size_bytes": int(file_size_bytes),
                "estimated_total_seconds": estimated_total,
                "started_at": None,
                "completed_at": None,
                "history": None,
                "error": None,
                "_started_epoch": None,
                "_completed_epoch": None,
            }
            while len(self.history_load_jobs) > 10:
                oldest_id = next(iter(self.history_load_jobs))
                if oldest_id == job_id:
                    break
                self.history_load_jobs.pop(oldest_id, None)
        return job_id

    def _iniciar_hilo_carga(self, job_id: str) -> None:
        Thread(
            target=self._ejecutar_job_carga,
            args=(job_id,),
            name=f"history-load-{job_id[:8]}",
            daemon=True,
        ).start()

    def _ejecutar_job_carga(self, job_id: str) -> None:
        with self.history_load_execution_lock:
            started_epoch = time()
            self._actualizar_job_carga(
                job_id,
                status="running",
                phase="preparing",
                progress=1.0,
                message="Preparando la carga del historico",
                started_at=datetime.now(timezone.utc).isoformat(),
                _started_epoch=started_epoch,
            )
            try:
                history_status = self.cargar_real(
                    progress_callback=lambda phase, progress, message: self._actualizar_job_carga(
                        job_id,
                        status="running",
                        phase=phase,
                        progress=progress,
                        message=message,
                    )
                )
            except Exception as exc:
                self._fallar_job_carga(job_id, exc)
                return

            completed_epoch = time()
            elapsed = max(0.001, completed_epoch - started_epoch)
            with self.history_load_jobs_lock:
                file_size = int(
                    self.history_load_jobs.get(job_id, {}).get("file_size_bytes", 0)
                )
            if file_size > 0:
                self.history_load_seconds_per_byte = elapsed / file_size
            self._actualizar_job_carga(
                job_id,
                status="completed",
                phase="ready",
                progress=100.0,
                message="Historico listo para utilizar",
                history=history_status,
                completed_at=datetime.now(timezone.utc).isoformat(),
                _completed_epoch=completed_epoch,
                estimated_total_seconds=round(elapsed, 3),
            )

    def _actualizar_job_carga(self, job_id: str, **changes: Any) -> None:
        with self.history_load_jobs_lock:
            job = self.history_load_jobs.get(job_id)
            if job is not None:
                job.update(changes)

    def _fallar_job_carga(self, job_id: str, exc: Exception) -> None:
        self._actualizar_job_carga(
            job_id,
            status="failed",
            phase="failed",
            message="La carga del historico fallo",
            error=str(exc),
            completed_at=datetime.now(timezone.utc).isoformat(),
            _completed_epoch=time(),
        )

    def asegurar_canonico(self) -> dict[str, Any]:
        self.historial = self.cargador.crear_canonico_minimo(self.static_history_dir)
        self._reset_analysis()
        self._limpiar_reproduccion_validacion()
        self.timeline_base_cache = {}
        self.mapa_contextual_cache = None
        self._preparar_continuidad_historica()
        return self._status_desde_historial(self.historial)

    def _preparar_continuidad_historica(self) -> None:
        if self.historial is None or self.historial.eventos.empty:
            self.future_baseline_template = None
            self.future_context_map_template = None
            self.future_context_events_template = None
            self.future_evaluation_prepared_at = None
            return
        baseline = MotorReproduccionValidacion(
            self.filtro_estadistico
        ).construir_baseline(
            eventos_desarrollo=self.historial.eventos,
            entidades_desarrollo=self.historial.entidades,
        )
        mapa = self._asegurar_mapa_contextual()
        fin_historico = pd.to_datetime(
            self.historial.metadatos.get("end"), errors="coerce", utc=True
        )
        if pd.isna(fin_historico):
            fin_historico = pd.to_datetime(
                self.historial.eventos["timestamp_dt"],
                errors="coerce",
                utc=True,
            ).max()
        self.future_baseline_template = deepcopy(baseline)
        self.future_context_map_template = deepcopy(mapa)
        self.future_context_events_template = self._recortar_eventos_contextuales(
            self.historial.eventos,
            fin_historico,
        )
        self.future_evaluation_prepared_at = datetime.now(
            timezone.utc
        ).isoformat()

    def subir_validacion(
        self,
        contenido: bytes,
        filename: str = "historico_validacion.csv",
    ) -> dict[str, Any]:
        if not contenido:
            raise ValueError("El archivo de Evaluacion esta vacio.")
        if self.historial is None:
            raise ValueError(
                "Primero debes cargar el conjunto Desarrollo y esperar a que "
                "termine de preparar el estado del detector."
            )
        if (
            self.future_baseline_template is None
            or self.future_context_map_template is None
            or self.future_context_events_template is None
        ):
            raise ValueError(
                "El historico no tiene preparado su estado para Evaluacion; "
                "vuelve a cargar el historico real."
            )

        with self.validation_replay_lock:
            inicio = time()
            ruta = self.validation_history_path or (
                self.analysis_cache_path.parent / "historico_validacion_replay.csv"
            )
            ruta.parent.mkdir(parents=True, exist_ok=True)
            ruta.write_bytes(contenido)
            validacion = self.cargador.cargar_real(ruta)
            if validacion.eventos.empty:
                raise ValueError("Evaluacion no contiene eventos con entidad y fecha validas.")

            inicio_desarrollo = pd.to_datetime(
                self.historial.metadatos.get("start"), errors="coerce", utc=True
            )
            fin_desarrollo = pd.to_datetime(
                self.historial.metadatos.get("end"), errors="coerce", utc=True
            )
            inicio_validacion = validacion.eventos["timestamp_dt"].min()
            fin_validacion = validacion.eventos["timestamp_dt"].max()
            if pd.notna(fin_desarrollo) and inicio_validacion < fin_desarrollo:
                raise ValueError(
                    "Evaluacion comienza antes de terminar el historico cargado; revisa la continuidad temporal."
                )

            eventos = validacion.eventos.sort_values(
                ["timestamp_dt", "entity_id"]
            ).reset_index(drop=True)
            eventos["_replay_index"] = eventos.index
            entidades_objetivo = set(eventos["entity_id"].astype(str).unique())
            baseline_historico = self.future_baseline_template
            mapa_contextual_historico = self.future_context_map_template
            entidades_con_base = set().union(
                baseline_historico.get("continuous", {}),
                baseline_historico.get("binary", {}),
                baseline_historico.get("multistate", {}),
            )
            self.historial_validacion = validacion
            self.eventos_validacion_ordenados = eventos
            # Cargar Evaluacion no inicia detectores. El primer avance toma una
            # copia del estado que ya quedo congelado al cargar el historico.
            self.validation_baseline = None
            self.validation_context_map = None
            self.validation_context_events = None
            self.validation_results = []
            self.validation_contextual_results = []
            self.validation_classification_results = []
            self.validation_replay_state = {
                "status": "ready",
                "filename": Path(filename).name or ruta.name,
                "evaluation_upload_mode": "load_only",
                "state_source": "historical_template",
                "historical_state_prepared_at": self.future_evaluation_prepared_at,
                "detector_state": "pending_until_advance",
                "rows": int(len(eventos)),
                "cursor": 0,
                "start": inicio_validacion.isoformat(),
                "end": fin_validacion.isoformat(),
                "current_timestamp": inicio_validacion.isoformat(),
                "development_start": None if pd.isna(inicio_desarrollo) else inicio_desarrollo.isoformat(),
                "development_end": None if pd.isna(fin_desarrollo) else fin_desarrollo.isoformat(),
                "entity_count": int(eventos["entity_id"].nunique()),
                "baseline_summary": baseline_historico.get("summary", {}),
                "context_map_version": mapa_contextual_historico.get("map_version"),
                "context_map_stale": bool(
                    mapa_contextual_historico.get("refresh", {}).get("stale")
                ),
                "entities_with_baseline": len(entidades_con_base),
                "entities_without_baseline": int(
                    len(entidades_objetivo - entidades_con_base)
                ),
                "load_elapsed_seconds": round(max(0.0, time() - inicio), 3),
                "processing_elapsed_seconds": 0.0,
                "contextual_elapsed_ms": 0.0,
                "classification_elapsed_ms": 0.0,
                "last_batch": None,
            }
            return self._payload_reproduccion_validacion()

    def _limpiar_reproduccion_validacion(self) -> None:
        self.historial_validacion = None
        self.eventos_validacion_ordenados = None
        self.validation_baseline = None
        self.validation_context_map = None
        self.validation_context_events = None
        self.validation_results = []
        self.validation_contextual_results = []
        self.validation_classification_results = []
        self.validation_replay_state = {}

    def _recortar_eventos_contextuales(
        self,
        eventos: pd.DataFrame,
        disponible_hasta: pd.Timestamp,
    ) -> pd.DataFrame:
        if eventos is None or eventos.empty or pd.isna(disponible_hasta):
            return pd.DataFrame(columns=[] if eventos is None else eventos.columns)
        trabajo = eventos.copy()
        trabajo["timestamp_dt"] = pd.to_datetime(
            trabajo["timestamp_dt"], errors="coerce", utc=True
        )
        trabajo = trabajo[
            trabajo["timestamp_dt"].notna()
            & trabajo["timestamp_dt"].le(disponible_hasta)
        ].copy()
        ventana = self.filtro_contextual.config.window_seconds * (
            self.filtro_contextual.config.max_lag_windows + 1
        )
        desde = disponible_hasta.floor(
            f"{self.filtro_contextual.config.window_seconds}s"
        ) - pd.Timedelta(seconds=ventana)
        recientes = trabajo[trabajo["timestamp_dt"].ge(desde)].copy()
        anteriores_estado = trabajo[
            trabajo["timestamp_dt"].lt(desde)
            & trabajo["sensor_kind"].isin({"binary", "multistate"})
        ]
        if anteriores_estado.empty:
            return recientes.sort_values(
                ["timestamp_dt", "entity_id"]
            ).reset_index(drop=True)
        arrastre = (
            anteriores_estado.sort_values("timestamp_dt")
            .groupby("entity_id", sort=False)
            .tail(1)
        )
        return (
            pd.concat([arrastre, recientes], ignore_index=True)
            .drop_duplicates(
                subset=["entity_id", "timestamp_dt"], keep="last"
            )
            .sort_values(["timestamp_dt", "entity_id"])
            .reset_index(drop=True)
        )

    def estado_reproduccion_validacion(self) -> dict[str, Any]:
        with self.validation_replay_lock:
            return self._payload_reproduccion_validacion()

    def avanzar_reproduccion_validacion(
        self,
        batch_size: int,
        suspicion_threshold: float,
    ) -> dict[str, Any]:
        with self.validation_replay_lock:
            if (
                self.historial_validacion is None
                or self.eventos_validacion_ordenados is None
            ):
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
            total = len(self.eventos_validacion_ordenados)
            cursor = int(self.validation_replay_state.get("cursor", 0))
            if cursor >= total:
                self.validation_replay_state["status"] = "completed"
                return self._payload_reproduccion_validacion(new_results=[])
            self._inicializar_estado_reproduccion_validacion()

            fin = min(total, cursor + int(batch_size))
            lote = self.eventos_validacion_ordenados.iloc[cursor:fin].copy()
            reloj_anterior = pd.to_datetime(
                self.validation_replay_state.get("current_timestamp"),
                errors="coerce",
                utc=True,
            )
            inicio_proceso = time()
            resultado = MotorReproduccionValidacion(
                self.filtro_estadistico
            ).analizar_lote(
                nuevos_eventos=lote,
                baseline=self.validation_baseline,
                suspicion_threshold=suspicion_threshold,
            )
            reloj = lote["timestamp_dt"].max()
            nuevos_resultados = []
            for item in resultado.get("results", []):
                enriquecido = dict(item)
                timestamp = pd.to_datetime(
                    enriquecido.get("timestamp"), errors="coerce", utc=True
                )
                delay = (
                    0.0
                    if pd.isna(timestamp)
                    else max(0.0, (reloj - timestamp).total_seconds())
                )
                enriquecido["replay"] = {
                    "detected_at_simulated": reloj.isoformat(),
                    "detection_delay_seconds": round(delay, 3),
                    "batch_start_cursor": cursor,
                    "batch_end_cursor": fin,
                    "causal": True,
                }
                enriquecido["source"] = "validation_replay"
                enriquecido["partition"] = "evaluation"
                nuevos_resultados.append(enriquecido)
            self.validation_results.extend(nuevos_resultados)

            eventos_contextuales = pd.concat(
                [self.validation_context_events, lote],
                ignore_index=True,
            ).sort_values(["timestamp_dt", "entity_id"])
            contextual = self.filtro_contextual.analizar_causal(
                sospechas=nuevos_resultados,
                eventos_disponibles=eventos_contextuales,
                entidades=self.historial.entidades,
                mapa=self.validation_context_map or {},
            )
            replay_por_evento = {
                (
                    str(item.get("entity_id")),
                    pd.to_datetime(
                        item.get("timestamp"), errors="coerce", utc=True
                    ).isoformat(),
                ): item.get("replay", {})
                for item in nuevos_resultados
            }
            nuevos_contextuales = []
            for item in contextual.get("results", []):
                enriquecido = dict(item)
                timestamp = pd.to_datetime(
                    enriquecido.get("timestamp"), errors="coerce", utc=True
                )
                clave = (
                    str(enriquecido.get("entity_id")),
                    None if pd.isna(timestamp) else timestamp.isoformat(),
                )
                enriquecido["replay"] = deepcopy(
                    replay_por_evento.get(clave, {})
                )
                enriquecido["source"] = "validation_replay"
                enriquecido["partition"] = "evaluation"
                nuevos_contextuales.append(enriquecido)
            self.validation_contextual_results.extend(nuevos_contextuales)

            clasificacion = self._analizar_clasificacion(
                nuevos_resultados,
                nuevos_contextuales,
                source="validation_replay",
            )
            nuevos_diagnosticos = []
            for item in clasificacion.get("results", []):
                enriquecido = dict(item)
                timestamp = pd.to_datetime(
                    enriquecido.get("timestamp"), errors="coerce", utc=True
                )
                clave = (
                    str(enriquecido.get("entity_id")),
                    None if pd.isna(timestamp) else timestamp.isoformat(),
                )
                enriquecido["replay"] = deepcopy(
                    replay_por_evento.get(clave, {})
                )
                enriquecido["partition"] = "evaluation"
                nuevos_diagnosticos.append(enriquecido)
            self.validation_classification_results.extend(nuevos_diagnosticos)
            self.validation_context_events = self._recortar_eventos_contextuales(
                eventos_contextuales,
                reloj,
            )
            elapsed = max(0.0, time() - inicio_proceso)

            simulated_span = (
                0.0
                if pd.isna(reloj_anterior)
                else max(0.0, (reloj - reloj_anterior).total_seconds())
            )
            acumulado = float(
                self.validation_replay_state.get(
                    "processing_elapsed_seconds", 0.0
                )
            ) + elapsed
            contextual_elapsed = float(
                contextual.get("summary", {}).get("elapsed_ms", 0.0)
            )
            classification_elapsed = float(
                clasificacion.get("summary", {}).get("elapsed_ms", 0.0)
            )
            self.validation_replay_state.update(
                {
                    "status": "completed" if fin >= total else "running",
                    "detector_state": "active",
                    "cursor": fin,
                    "current_timestamp": reloj.isoformat(),
                    "processing_elapsed_seconds": round(acumulado, 6),
                    "contextual_elapsed_ms": round(
                        float(
                            self.validation_replay_state.get(
                                "contextual_elapsed_ms", 0.0
                            )
                        )
                        + contextual_elapsed,
                        3,
                    ),
                    "classification_elapsed_ms": round(
                        float(
                            self.validation_replay_state.get(
                                "classification_elapsed_ms", 0.0
                            )
                        )
                        + classification_elapsed,
                        3,
                    ),
                    "last_batch": {
                        "events": int(len(lote)),
                        "processing_elapsed_ms": round(elapsed * 1000.0, 3),
                        "simulated_span_seconds": round(simulated_span, 3),
                        "events_per_second": round(len(lote) / max(elapsed, 1e-9), 3),
                        "keeps_up_with_event_time": elapsed <= max(simulated_span, 0.001),
                        "new_detections": len(nuevos_resultados),
                        "contextual_evaluated": len(nuevos_contextuales),
                        "classified": len(nuevos_diagnosticos),
                        "timings_by_kind_ms": resultado.get("summary", {}).get(
                            "timings_by_kind_ms", {}
                        ),
                        "contextual_elapsed_ms": contextual_elapsed,
                        "classification_elapsed_ms": classification_elapsed,
                        "classification_by_class": clasificacion.get(
                            "summary", {}
                        ).get("by_class", {}),
                    },
                }
            )
            return self._payload_reproduccion_validacion(
                new_results=nuevos_resultados,
                new_contextual_results=nuevos_contextuales,
                new_classification_results=nuevos_diagnosticos,
            )

    def reiniciar_reproduccion_validacion(self) -> dict[str, Any]:
        with self.validation_replay_lock:
            if (
                self.historial_validacion is None
                or self.eventos_validacion_ordenados is None
            ):
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
            self.validation_baseline = None
            self.validation_context_map = None
            self.validation_context_events = None
            self.validation_results = []
            self.validation_contextual_results = []
            self.validation_classification_results = []
            self.validation_replay_state.update(
                {
                    "status": "ready",
                    "detector_state": "pending_until_advance",
                    "cursor": 0,
                    "current_timestamp": self.validation_replay_state.get("start"),
                    "processing_elapsed_seconds": 0.0,
                    "contextual_elapsed_ms": 0.0,
                    "classification_elapsed_ms": 0.0,
                    "last_batch": None,
                }
            )
            return self._payload_reproduccion_validacion(new_results=[])

    def _inicializar_estado_reproduccion_validacion(self) -> None:
        if (
            self.validation_baseline is not None
            and self.validation_context_map is not None
            and self.validation_context_events is not None
        ):
            return
        if (
            self.future_baseline_template is None
            or self.future_context_map_template is None
            or self.future_context_events_template is None
        ):
            raise ValueError(
                "El estado historico para Evaluacion ya no esta disponible; "
                "vuelve a cargar el historico real."
            )
        self.validation_baseline = deepcopy(self.future_baseline_template)
        self.validation_context_map = deepcopy(self.future_context_map_template)
        self.validation_context_events = self.future_context_events_template.copy()

    def _payload_reproduccion_validacion(
        self,
        new_results: list[dict[str, Any]] | None = None,
        new_contextual_results: list[dict[str, Any]] | None = None,
        new_classification_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.validation_replay_state:
            return {
                "loaded": False,
                "status": "empty",
                "rows": 0,
                "processed_rows": 0,
                "progress": 0.0,
                "results": [],
                "new_results": [],
                "contextual_results": [],
                "new_contextual_results": [],
                "classification_results": [],
                "new_classification_results": [],
            }
        payload = dict(self.validation_replay_state)
        rows = int(payload.get("rows", 0))
        cursor = int(payload.get("cursor", 0))
        resultados = sorted(
            self.validation_results,
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
            reverse=True,
        )
        delays = [
            float(item.get("replay", {}).get("detection_delay_seconds", 0.0))
            for item in resultados
        ]
        contextuales = sorted(
            self.validation_contextual_results,
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
            reverse=True,
        )
        diagnosticos = sorted(
            self.validation_classification_results,
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
            reverse=True,
        )
        outcomes: dict[str, int] = {}
        for item in contextuales:
            outcome = str(item.get("context_outcome", "unexplained"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        resumen_contextual = {
            "evaluated": len(contextuales),
            "by_outcome": outcomes,
            "possible_common_cause": sum(
                bool(item.get("common_cause", {}).get("candidate"))
                for item in contextuales
            ),
            "elapsed_ms": round(
                float(payload.get("contextual_elapsed_ms", 0.0)), 3
            ),
            "causal": True,
        }
        resumen_clasificacion = self._resumen_resultados_clasificacion(
            diagnosticos,
            elapsed_ms=float(
                payload.get("classification_elapsed_ms", 0.0)
            ),
        )
        return {
            "loaded": True,
            **payload,
            "processed_rows": cursor,
            "remaining_rows": max(0, rows - cursor),
            "progress": round(cursor * 100.0 / max(rows, 1), 4),
            "detection_count": len(resultados),
            "average_detection_delay_seconds": round(
                sum(delays) / len(delays), 3
            ) if delays else None,
            "max_detection_delay_seconds": round(max(delays), 3) if delays else None,
            "results": resultados[:500],
            "new_results": new_results or [],
            "contextual_count": len(contextuales),
            "contextual_summary": resumen_contextual,
            "contextual_results": contextuales[:500],
            "new_contextual_results": new_contextual_results or [],
            "classification_count": len(diagnosticos),
            "classification_summary": resumen_clasificacion,
            "classification_results": diagnosticos[:500],
            "new_classification_results": new_classification_results or [],
            "pipeline_summary": {
                "statistical": len(resultados),
                "contextual": len(contextuales),
                "classifier": len(diagnosticos),
                "causal": True,
                "uses_future_evaluation_rows": False,
            },
        }

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
        # La configuracion cambia la base estadistica. Se vuelve a congelar el
        # estado desde el historico aqui, nunca durante la carga de Evaluacion.
        self._preparar_continuidad_historica()
        return {"config_path": str(self.filtro_estadistico.continuos.conf_path), "configs": guardadas}

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
        classification = self._analizar_clasificacion(
            self.resultados,
            self.resultados_contextuales,
            source="historical",
        )
        self._guardar_estado_clasificacion(classification)
        payload["classification"] = classification
        return payload

    def contexto_actual(self) -> dict[str, Any]:
        resultados_origen = (
            self.validation_contextual_results
            if self.validation_replay_state
            else self.resultados_contextuales
        )
        resultados = sorted(
            resultados_origen,
            key=lambda item: pd.to_datetime(
                item.get("timestamp"), errors="coerce", utc=True
            ),
            reverse=True,
        )
        summary: dict[str, Any] = {
            "evaluated": len(resultados),
            "by_outcome": {},
            "possible_common_cause": 0,
        }
        for item in resultados:
            outcome = str(item.get("context_outcome", "unexplained"))
            summary["by_outcome"][outcome] = summary["by_outcome"].get(outcome, 0) + 1
            summary["possible_common_cause"] += int(
                bool(item.get("common_cause", {}).get("candidate"))
            )
        return {
            "results": resultados,
            "summary": summary,
            "source": (
                "validation_replay"
                if self.validation_replay_state
                else "historical"
            ),
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
            source="historical",
        )
        self._guardar_estado_clasificacion(payload)
        return payload

    def clasificacion_actual(self) -> dict[str, Any]:
        if self.validation_replay_state:
            specification = self.especificacion_clasificacion()
            resultados = sorted(
                self.validation_classification_results,
                key=lambda item: pd.to_datetime(
                    item.get("timestamp"), errors="coerce", utc=True
                ),
                reverse=True,
            )
            return {
                "schema_version": "classifier-results-v1",
                "model": {
                    "version": specification.get("model_version"),
                    "type": specification.get("model_type"),
                    "confidence_kind": specification.get("confidence_kind"),
                    "selection_status": specification.get("selection_status"),
                },
                "summary": self._resumen_resultados_clasificacion(
                    resultados,
                    elapsed_ms=float(
                        self.validation_replay_state.get(
                            "classification_elapsed_ms", 0.0
                        )
                    ),
                ),
                "results": resultados,
                "source": "validation_replay",
            }
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

        ground_truth_path = self.ground_truth_path
        if ground_truth_path is None or not ground_truth_path.exists():
            return self._evaluacion_no_disponible(
                requested_partition,
                (
                    "No existe una base de verdad real asociada al conjunto cargado. "
                    "Se requieren etiquetas reales para calcular las metricas."
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
            "start": metadatos.get("start"),
            "end": metadatos.get("end"),
            "normality_history": normalidad,
            "future_evaluation_ready": bool(
                self.future_baseline_template is not None
                and self.future_context_map_template is not None
                and self.future_context_events_template is not None
            ),
            "future_evaluation_prepared_at": self.future_evaluation_prepared_at,
            "future_baseline_summary": (
                {}
                if self.future_baseline_template is None
                else self.future_baseline_template.get("summary", {})
            ),
        }

    def _reset_analysis(self) -> None:
        self.resultados = []
        self.resultados_contextuales = []
        self.resultados_clasificacion = []
        self.ultimo_payload_clasificacion = None
        self.ultima_evaluacion_clasificacion = None
        self.analysis_completed = False

    def _es_historial_real_activo(self) -> bool:
        return self.historial is not None and self.historial.metadatos.get("source_type") == "real"

    def _identidad_historial_real(self, suspicion_threshold: float) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        stat = self.real_history_path.stat()
        conf_path = self.filtro_estadistico.continuos.conf_path
        classifier_specification = self.filtro_clasificacion.especificacion()
        contextual_map_path = self._ruta_cache_contextual()
        contextual_map_expired: bool | None = None
        if (
            contextual_map_path is not None
            and contextual_map_path.is_file()
        ):
            try:
                with contextual_map_path.open("r", encoding="utf-8") as archivo:
                    contextual_map_payload = json.load(archivo)
                built_at = pd.to_datetime(
                    contextual_map_payload.get("built_at"),
                    errors="coerce",
                    utc=True,
                )
                contextual_map_expired = bool(
                    pd.isna(built_at)
                    or (
                        pd.Timestamp.now(tz="UTC") - built_at
                    ).total_seconds()
                    > (
                        self.filtro_contextual.config.max_map_age_days
                        * 86400.0
                    )
                )
            except (OSError, json.JSONDecodeError):
                contextual_map_expired = True
        return {
            "cache_version": 17,
            "source_path": str(self.real_history_path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "source_sha256": self._sha256_archivo(self.real_history_path),
            "continuous_conf_sha256": self._sha256_archivo(conf_path) if conf_path.is_file() else "",
            "suspicion_threshold": round(float(suspicion_threshold), 6),
            "continuous_score_specification": (
                self.filtro_estadistico.continuos.especificacion_puntaje()
            ),
            "multistate_score_specification": (
                self.filtro_estadistico.multivalor.especificacion_puntaje()
            ),
            "contextual_filter_version": self.filtro_contextual.MAP_VERSION,
            "contextual_configuration": (
                self.filtro_contextual.config.como_dict()
            ),
            "contextual_implementation_sha256": (
                self.filtro_contextual.implementation_sha256()
            ),
            "contextual_map_sha256": (
                self._sha256_archivo(contextual_map_path)
                if contextual_map_path is not None
                and contextual_map_path.is_file()
                else ""
            ),
            "contextual_map_expired": contextual_map_expired,
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
        digest = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for chunk in iter(lambda: archivo.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

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
    validation_history_path=Path(
        os.getenv(
            "VALIDATION_HISTORY_PATH",
            str(ARTIFACT_ROOT / "data/datosreales/historico_validacion.csv"),
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
def history_static_status(
    autoload: bool = Query(default=True),
) -> dict[str, Any]:
    return state.status(autoload=autoload)


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


@app.post("/history/real/load/start", tags=["history"])
def history_real_load_start() -> dict[str, Any]:
    try:
        return state.iniciar_carga_real()
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


@app.post("/history/real/upload/start", tags=["history"])
async def history_real_upload_start(request: Request) -> dict[str, Any]:
    try:
        return state.iniciar_subida_real(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/history/real/load/jobs/{job_id}", tags=["history"])
def history_real_load_job(job_id: str) -> dict[str, Any]:
    try:
        return state.estado_carga_real(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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


@app.post("/validation/replay/upload", tags=["validation-replay"])
async def validation_replay_upload(request: Request) -> dict[str, Any]:
    try:
        return state.subir_validacion(
            await request.body(),
            filename=unquote(
                request.headers.get("X-Filename", "historico_validacion.csv")
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/validation/replay/status", tags=["validation-replay"])
def validation_replay_status() -> dict[str, Any]:
    return state.estado_reproduccion_validacion()


@app.post("/validation/replay/advance", tags=["validation-replay"])
def validation_replay_advance(
    request: ValidationReplayAdvanceRequest,
) -> dict[str, Any]:
    try:
        return state.avanzar_reproduccion_validacion(
            batch_size=request.batch_size,
            suspicion_threshold=request.suspicion_threshold,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/validation/replay/reset", tags=["validation-replay"])
def validation_replay_reset() -> dict[str, Any]:
    try:
        return state.reiniciar_reproduccion_validacion()
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


@app.get("/history/timeline", tags=["history"])
def history_timeline(
    start: str | None = None,
    end: str | None = None,
    max_points: int = Query(default=1000, ge=10, le=10000),
) -> dict[str, Any]:
    return state.timeline_historial(start=start, end=end, max_points=max_points)


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
        return {"entities": []}
    lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
    timelines = []
    for entity_id, grupo in eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
        entidad = lookup.get(str(entity_id), {})
        tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
        categories: dict[str, int] = {}
        analysis_status = "eligible"
        if tipo == "binary":
            points, bucket_seconds = timeline_binario(grupo, max_points=max_points)
            chart_kind = "active_percentage"
            scale_unit = "%"
            gaps = []
        elif tipo == "multistate":
            validos = grupo[
                grupo["is_valid_value"].astype(bool)
            ] if "is_valid_value" in grupo.columns else grupo
            estados = validos["state"].astype(str).str.strip()
            estados_unicos = estados.drop_duplicates().tolist()
            unique_ratio = len(estados_unicos) / max(len(estados), 1)
            if len(estados) < 24:
                analysis_status = "insufficient_history"
            elif len(estados_unicos) < 2:
                analysis_status = "insufficient_states"
            elif len(estados_unicos) > 32 or (
                len(estados_unicos) >= 8 and unique_ratio > 0.50
            ):
                analysis_status = "high_cardinality"
            if analysis_status != "high_cardinality":
                categories = {
                    estado: indice
                    for indice, estado in enumerate(estados_unicos)
                }
            sampleado = muestrear_timeline_con_cortes(grupo, max_points)
            points = [
                {
                    "timestamp": str(fila["timestamp"]),
                    "state": str(fila["state"]),
                    "value": (
                        categories.get(str(fila["state"]))
                        if bool(fila.get("is_valid_value", True))
                        else None
                    ),
                    "is_valid_value": bool(fila.get("is_valid_value", True)),
                }
                for fila in sampleado.to_dict("records")
            ]
            bucket_seconds = None
            chart_kind = "categorical_state"
            scale_unit = "estado"
            gaps = timeline_gaps_invalidos(grupo)
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
                "categories": categories,
                "analysis_status": analysis_status,
            }
        )
    return {"entities": timelines}


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
