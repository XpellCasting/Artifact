from __future__ import annotations
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from time import time
from typing import Any, Callable
from uuid import uuid4
import pandas as pd
from app.services.reproduccion import MotorReproduccionValidacion
import app.services.cache_causal as causal_cache
from app.services.procedencia_normalidad import SCHEMA_VERSION as NORMALITY_PROVENANCE_VERSION
from app.services.procedencia_normalidad import build_identity as build_baseline_identity
from app.services.procedencia_normalidad import effective_parameters as baseline_effective_parameters
from app.services.procedencia_normalidad import write_manifest as write_baseline_manifest

class ServicesHistoryMixin:
    def status(self, autoload: bool = True) -> dict[str, Any]:
        with self.history_load_jobs_lock:
            has_load_job = bool(self.history_load_jobs)
        if self.historial is None and autoload and not has_load_job:
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
        self._preparar_continuidad_historica(progress_callback=progress_callback)
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

        end = completed_epoch or now
        phase_started = payload.pop("_phase_started_epoch", None)
        last_progress = payload.pop("_last_progress_epoch", None)
        payload["phase_elapsed_seconds"] = round(max(0.0, end - phase_started), 3) if phase_started else 0.0
        payload["seconds_since_progress"] = round(max(0.0, end - last_progress), 3) if last_progress else 0.0
        # Los porcentajes representan etapas, no una fracción del tiempo de cálculo.
        payload["estimated_total_seconds"] = elapsed if payload.get("status") == "completed" else None
        payload["eta_seconds"] = 0.0 if payload.get("status") == "completed" else None
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
                now = time()
                if changes.get("phase", job.get("phase")) != job.get("phase") or "_phase_started_epoch" not in job:
                    job["_phase_started_epoch"] = now
                if any(changes.get(key, job.get(key)) != job.get(key) for key in ("phase", "progress", "message")):
                    job["_last_progress_epoch"] = now
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
    def _preparar_continuidad_historica(self, progress_callback=None) -> None:
        def report(phase, progress, message):
            if progress_callback is not None:
                progress_callback(phase, progress, message)

        report("causal_identity", 96.0, "Calculando la identidad del histórico")
        self.future_preparation_cache = {"reused": False, "persisted": False}
        if self.historial is None or self.historial.eventos.empty:
            self.future_baseline_template = None
            self.future_context_map_template = None
            self.future_context_events_template = None
            self.future_evaluation_prepared_at = None
            return
        eventos_capturados = self.historial.eventos
        entidades_capturadas = self.historial.entidades
        entity_ids = set(eventos_capturados["entity_id"].astype(str))
        ajustes = self.filtro_estadistico.continuos.ajustes_manuales(entity_ids)
        identidad = build_baseline_identity(eventos_capturados, entidades_capturadas, ajustes)
        self.future_baseline_template = None
        self.future_context_map_template = None
        self.future_context_events_template = None
        self.future_evaluation_prepared_at = None
        cache_path = self.analysis_cache_path.parent / "causal_preparation.pkl"
        report("causal_cache", 96.05, "Comprobando la preparación causal guardada")
        key = causal_cache.cache_key(
            identidad, self.historial, self.filtro_contextual.config,
            self._ruta_cache_contextual(),
        )
        cached = causal_cache.load(cache_path, key)
        if cached is not None:
            self.future_baseline_template = cached["baseline"]
            self.future_context_map_template = cached["context_map"]
            self.future_context_events_template = cached["context_events"]
            self.future_evaluation_prepared_at = cached["prepared_at"]
            self.mapa_contextual_cache = deepcopy(cached["context_map"])
            self.future_preparation_cache = {"reused": True, "persisted": True}
            return
        report("baseline_prepare", 96.1, "Preparando eventos para construir la base estadística")
        eventos_capturados = eventos_capturados.copy(deep=True)
        entidades_capturadas = entidades_capturadas.copy(deep=True)
        baseline = MotorReproduccionValidacion(
            self.filtro_estadistico
        ).construir_baseline(
            eventos_desarrollo=eventos_capturados,
            entidades_desarrollo=entidades_capturadas,
            progress_callback=lambda phase, progress, message: report(phase, 96.1 + progress * 0.009, message),
        )
        report("baseline_manifest", 97.0, "Guardando los parámetros de la base estadística")
        parametros = baseline_effective_parameters(baseline)
        source = {
            "name": Path(str(self.historial.metadatos.get("source_path") or "")).name,
            "path": self.historial.metadatos.get("source_path"),
            "rows": len(eventos_capturados),
            "entities": len(entity_ids),
            "start": self.historial.metadatos.get("start"),
            "end": self.historial.metadatos.get("end"),
        }
        manifest_path, manifest = write_baseline_manifest(
            self.analysis_cache_path.parent, identidad, source, ajustes, parametros
        )
        baseline["provenance"] = {
            "baseline_id": identidad["baseline_id"],
            "schema_version": NORMALITY_PROVENANCE_VERSION,
            "identity": identidad,
            "effective_parameters_sha256": manifest["effective_parameters_sha256"],
            "manifest_path": str(manifest_path),
        }
        mapa = self._asegurar_mapa_contextual(
            progress_callback=lambda phase, progress, message: report(phase, 97.1 + progress * 0.014, message)
        )
        report("causal_snapshot", 98.5, "Preparando el estado y los eventos del final del histórico")
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
        report("causal_save", 98.8, "Guardando la preparación causal para próximas cargas")
        # La construcción puede actualizar el mapa: guardar su huella definitiva.
        key = causal_cache.cache_key(
            identidad, self.historial, self.filtro_contextual.config,
            self._ruta_cache_contextual(),
        )
        self.future_preparation_cache["persisted"] = causal_cache.save(cache_path, key, {
            "baseline": self.future_baseline_template,
            "context_map": self.future_context_map_template,
            "context_events": self.future_context_events_template,
            "prepared_at": self.future_evaluation_prepared_at,
        })
