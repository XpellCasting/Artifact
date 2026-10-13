from __future__ import annotations
from collections import deque
from copy import deepcopy
from itertools import islice
from pathlib import Path
from threading import Event, Thread, current_thread
from time import time
from typing import Any
from uuid import uuid4
import pandas as pd
from app.modulos.estadistico.episodios import DEFAULT_QUIET_SECONDS
from app.modulos.estadistico.episodios import agrupar_sospechas
from app.modulos.estadistico.episodios import enviar_a_contextual
from app.modulos.estadistico.episodios import resumen_episodios
from app.services.reproduccion import MotorReproduccionValidacion
import app.database.evaluacion_etiquetada as labelled_evaluation
from app.services.application_common import VALIDATION_REPLAY_MEMORY_RESULTS, VALIDATION_REPLAY_STATUS_RESULTS, MIN_SUSPICION_THRESHOLD, MAX_SUSPICION_THRESHOLD, DEFAULT_SUSPICION_THRESHOLD

class ServicesReplayMixin:
    def subir_validacion(
        self,
        contenido: bytes,
        filename: str = "historico_validacion.csv",
        labelled: bool = False,
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

        truth = []
        if labelled:
            contenido, truth = labelled_evaluation.separate_csv(contenido)

        # Una carga nueva invalida cualquier ejecucion anterior. La señal se
        # establece antes de esperar el lock para que el worker no tome otro lote.
        self.validation_replay_pause.set()
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
            # Evita conservar simultaneamente el orden por entidad creado por
            # el cargador y la copia cronologica usada por el replay.
            validacion.eventos = eventos
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
            self.validation_diagnostic_events = None
            self.validation_context_concurrence.clear()
            self.validation_suspicion_episodes.clear()
            self.validation_results = deque(maxlen=VALIDATION_REPLAY_MEMORY_RESULTS)
            self.validation_contextual_results = deque(
                maxlen=VALIDATION_REPLAY_MEMORY_RESULTS
            )
            self.validation_last_results = []
            self.validation_last_contextual_results = []
            self._reiniciar_almacen_resultados_validacion()
            self.validation_replay_generation += 1
            self.validation_replay_pause = Event()
            self.validation_replay_worker = None
            labelled_evaluation.install(self._ruta_almacen_resultados_validacion(), truth, eventos)
            self.validation_replay_state = {
                "labelled": labelled, "unmatched_label_detections": 0,
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
                "baseline_id": baseline_historico.get("provenance", {}).get("baseline_id"),
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
                "detection_count": 0,
                "contextual_count": 0,
                "environmental_count": 0,
                "pending_classification_count": 0,
                "classification_count": 0,
                "indeterminate_count": 0,
                "no_fault_count": 0,
                "classifier_status": "ready",
                "diagnostic_run_id": uuid4().hex,
                "suspicion_threshold": DEFAULT_SUSPICION_THRESHOLD,
                "suspicion_quiet_seconds": DEFAULT_QUIET_SECONDS,
                "suppressed_count": 0,
                "forwarded_count": 0,
                "contextual_outcomes": {},
                "contextual_coincidences": 0,
                "possible_common_cause": 0,
                "detection_delay_sum_seconds": 0.0,
                "max_detection_delay_seconds": None,
                "last_error": None,
                "last_batch": None,
            }
            return self._publicar_snapshot_reproduccion()
    def _limpiar_reproduccion_validacion(self) -> None:
        self.validation_replay_pause.set()
        with self.validation_replay_lock:
            self.validation_replay_generation += 1
            self.historial_validacion = None
            self.eventos_validacion_ordenados = None
            self.validation_baseline = None
            self.validation_context_map = None
            self.validation_context_events = None
            self.validation_diagnostic_events = None
            self.validation_context_concurrence.clear()
            self.validation_suspicion_episodes.clear()
            self.validation_results = deque(maxlen=VALIDATION_REPLAY_MEMORY_RESULTS)
            self.validation_contextual_results = deque(
                maxlen=VALIDATION_REPLAY_MEMORY_RESULTS
            )
            self.validation_last_results = []
            self.validation_last_contextual_results = []
            self.validation_replay_state = {}
            self.validation_replay_snapshot = {}
            self.validation_results_store_ready = False
            self.validation_replay_worker = None
    @staticmethod
    def _recortar_eventos_diagnosticos(
        eventos: pd.DataFrame, hasta: pd.Timestamp, window_seconds: int,
    ) -> pd.DataFrame:
        """Buffer propio de F3: su ventana puede superar la requerida por F2."""
        if eventos.empty:
            return eventos.copy()
        stamps = pd.to_datetime(eventos["timestamp_dt"], utc=True, errors="coerce")
        lower = hasta - pd.Timedelta(seconds=window_seconds)
        return eventos.loc[stamps.between(lower, hasta, inclusive="both")].copy()
    def _eventos_diagnosticos_del_lote(self, lote: pd.DataFrame, run_id: str) -> pd.DataFrame:
        if "diagnostic_window_seconds" not in self.validation_replay_state:
            window = (self.diagnostic_workbench.window_seconds_for_run("validation_replay", run_id)
                      if self.diagnostic_workbench is not None else 3600)
            self.validation_replay_state["diagnostic_window_seconds"] = int(window)
        window = self.validation_replay_state["diagnostic_window_seconds"]
        previous = self.validation_diagnostic_events
        if previous is None:
            historical = getattr(self.historial, "eventos", None)
            if not isinstance(historical, pd.DataFrame):
                historical = self.validation_context_events
            previous = self._recortar_eventos_diagnosticos(
                historical if historical is not None else lote.iloc[:0],
                lote["timestamp_dt"].min(), window,
            )
        return pd.concat([previous, lote], ignore_index=True).sort_values(["timestamp_dt", "entity_id"])
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
        # Conservar tambien el ultimo reporte continuo, incluso si es invalido:
        # F2 decide si sigue vigente y nunca rejuvenece su fecha.
        anteriores = trabajo[
            trabajo["timestamp_dt"].lt(desde)
        ]
        if anteriores.empty:
            return recientes.sort_values(
                ["timestamp_dt", "entity_id"]
            ).reset_index(drop=True)
        arrastre = (
            anteriores.sort_values("timestamp_dt")
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
    def _publicar_snapshot_reproduccion(self) -> dict[str, Any]:
        snapshot = self._payload_reproduccion_validacion(include_recent=True)
        self.validation_replay_snapshot = snapshot
        return snapshot
    def estado_reproduccion_validacion(self) -> dict[str, Any]:
        snapshot = self.validation_replay_snapshot
        if snapshot:
            return snapshot
        with self.validation_replay_lock:
            return self._publicar_snapshot_reproduccion()
    def iniciar_reproduccion_validacion(
        self,
        batch_size: int,
        suspicion_threshold: float,
        suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> dict[str, Any]:
        worker_anterior = self.validation_replay_worker
        if (
            worker_anterior is not None
            and worker_anterior.is_alive()
            and self.validation_replay_pause.is_set()
        ):
            worker_anterior.join(timeout=2.0)

        with self.validation_replay_lock:
            if (
                self.historial_validacion is None
                or self.eventos_validacion_ordenados is None
            ):
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
            if int(self.validation_replay_state.get("cursor", 0)) >= len(
                self.eventos_validacion_ordenados
            ):
                self.validation_replay_state["status"] = "completed"
                return self._publicar_snapshot_reproduccion()
            if (
                self.validation_replay_worker is not None
                and self.validation_replay_worker.is_alive()
                and not self.validation_replay_pause.is_set()
            ):
                return self._publicar_snapshot_reproduccion()

            cursor = int(self.validation_replay_state.get("cursor", 0))
            active_baseline_id = self.validation_replay_state.get("baseline_id")
            prepared_baseline_id = (self.future_baseline_template or {}).get("provenance", {}).get("baseline_id")
            if cursor > 0 and active_baseline_id != prepared_baseline_id:
                raise ValueError("La base de normalidad cambió durante una reproducción parcial; reiníciala antes de continuar.")
            if cursor > 0 and self.validation_replay_state.get("statistical_evaluation_version") != "pre_suppression_v1":
                raise ValueError("Esta reproducción parcial usa la semántica anterior; reiníciala antes de continuar.")
            self.validation_replay_state["statistical_evaluation_version"] = "pre_suppression_v1"
            umbral_anterior = self.validation_replay_state.get("suspicion_threshold")
            umbral_activo = (
                float(umbral_anterior)
                if cursor > 0
                and isinstance(umbral_anterior, (int, float))
                and MIN_SUSPICION_THRESHOLD <= float(umbral_anterior) <= MAX_SUSPICION_THRESHOLD
                else float(suspicion_threshold)
            )
            quiet_seconds = (int(self.validation_replay_state.get("suspicion_quiet_seconds", 0))
                             if cursor > 0 else suspicion_quiet_seconds)
            self.validation_replay_generation += 1
            generation = self.validation_replay_generation
            pause = Event()
            self.validation_replay_pause = pause
            self.validation_replay_state.update(
                {
                    "status": "running",
                    "last_error": None,
                    "automatic": True,
                    "batch_size": int(batch_size),
                    "suspicion_threshold": umbral_activo,
                    "suspicion_quiet_seconds": quiet_seconds,
                }
            )
            worker = Thread(
                target=self._ejecutar_reproduccion_validacion,
                args=(generation, pause, int(batch_size), umbral_activo, quiet_seconds),
                name=f"validation-replay-{generation}",
                daemon=True,
            )
            self.validation_replay_worker = worker
            worker.start()
            return self._publicar_snapshot_reproduccion()
    def _ejecutar_reproduccion_validacion(
        self,
        generation: int,
        pause: Event,
        batch_size: int,
        suspicion_threshold: float,
        suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> None:
        try:
            while not pause.is_set():
                payload = self.avanzar_reproduccion_validacion(
                    batch_size=batch_size,
                    suspicion_threshold=suspicion_threshold,
                    suspicion_quiet_seconds=suspicion_quiet_seconds,
                    _automatic=True,
                    _generation=generation,
                )
                if payload.get("status") in {"completed", "failed", "paused"}:
                    break
                # Cede brevemente el lock entre lotes para que status y pause
                # respondan aun durante una reproduccion de larga duracion.
                if pause.wait(0.01):
                    break
        except Exception as exc:
            with self.validation_replay_lock:
                if generation == self.validation_replay_generation:
                    self.validation_replay_state.update(
                        {
                            "status": "failed",
                            "last_error": str(exc),
                            "automatic": False,
                        }
                    )
                    self._publicar_snapshot_reproduccion()
        finally:
            with self.validation_replay_lock:
                if generation == self.validation_replay_generation:
                    if (
                        pause.is_set()
                        and self.validation_replay_state.get("status")
                        not in {"completed", "failed"}
                    ):
                        self.validation_replay_state["status"] = "paused"
                    self.validation_replay_state["automatic"] = False
                    self._publicar_snapshot_reproduccion()
    def pausar_reproduccion_validacion(self) -> dict[str, Any]:
        self.validation_replay_pause.set()
        with self.validation_replay_lock:
            if not self.validation_replay_state:
                return self._publicar_snapshot_reproduccion()
            if self.validation_replay_state.get("status") not in {
                "completed",
                "failed",
            }:
                self.validation_replay_state["status"] = "paused"
            self.validation_replay_state["automatic"] = False
            return self._publicar_snapshot_reproduccion()
    def avanzar_reproduccion_validacion(
        self,
        batch_size: int,
        suspicion_threshold: float,
        _automatic: bool = False,
        _generation: int | None = None,
        suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> dict[str, Any]:
        with self.validation_replay_lock:
            if (
                self.historial_validacion is None
                or self.eventos_validacion_ordenados is None
            ):
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
            if _automatic:
                if (
                    _generation != self.validation_replay_generation
                    or self.validation_replay_pause.is_set()
                ):
                    self.validation_replay_state["status"] = "paused"
                    return self._publicar_snapshot_reproduccion()
            elif (
                self.validation_replay_worker is not None
                and self.validation_replay_worker.is_alive()
                and current_thread() is not self.validation_replay_worker
                and not self.validation_replay_pause.is_set()
            ):
                raise ValueError(
                    "La reproduccion automatica esta activa; pausala antes de avanzar manualmente."
                )
            total = len(self.eventos_validacion_ordenados)
            cursor = int(self.validation_replay_state.get("cursor", 0))
            if cursor >= total:
                self.validation_replay_state["status"] = "completed"
                return self._publicar_snapshot_reproduccion()
            umbral_guardado = self.validation_replay_state.get("suspicion_threshold")
            if (
                cursor > 0
                and isinstance(umbral_guardado, (int, float))
                and MIN_SUSPICION_THRESHOLD <= float(umbral_guardado) <= MAX_SUSPICION_THRESHOLD
            ):
                # Una evaluación es reproducible: al reanudar no se cambia el
                # umbral a mitad de la secuencia aunque la interfaz tenga otro.
                suspicion_threshold = float(umbral_guardado)
            else:
                self.validation_replay_state["suspicion_threshold"] = float(suspicion_threshold)
            quiet_seconds = (int(self.validation_replay_state.get("suspicion_quiet_seconds", 0))
                             if cursor > 0 else suspicion_quiet_seconds)
            self.validation_replay_state["suspicion_quiet_seconds"] = quiet_seconds
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
            nuevas_detecciones = resultado.get("detections", resultado.get("results", []))
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

            nuevos_resultados, siguientes_episodios = agrupar_sospechas(
                nuevos_resultados, quiet_seconds, self.validation_suspicion_episodes
            )
            agrupacion_lote = resumen_episodios(nuevos_resultados)

            eventos_contextuales = pd.concat(
                [self.validation_context_events, lote],
                ignore_index=True,
            ).sort_values(["timestamp_dt", "entity_id"])
            siguiente_concurrencia = dict(self.validation_context_concurrence)
            contextual = self.filtro_contextual.analizar_causal(
                sospechas=[item for item in nuevos_resultados if enviar_a_contextual(item)],
                eventos_disponibles=eventos_contextuales,
                entidades=self.historial.entidades,
                mapa=self.validation_context_map or {},
                concurrence_state=siguiente_concurrencia,
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

            diagnostic_run_id = self.validation_replay_state.setdefault("diagnostic_run_id", uuid4().hex)
            eventos_diagnosticos = self._eventos_diagnosticos_del_lote(lote, diagnostic_run_id)
            nuevos_contextuales = self._diagnosticar_contextuales(
                nuevos_resultados,
                nuevos_contextuales,
                eventos_diagnosticos,
                source="validation_replay",
                run_id=diagnostic_run_id,
            )

            siguientes_eventos_contextuales = self._recortar_eventos_contextuales(
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
            rutas_lote = self._salidas_pipeline(nuevos_resultados, nuevos_contextuales)
            self._guardar_resultados_validacion(
                nuevos_resultados,
                nuevos_contextuales,
                rutas_lote,
            )
            if self.validation_replay_state.get("labelled"):
                unmatched = labelled_evaluation.record(self._ruta_almacen_resultados_validacion(), nuevas_detecciones, lote, self.validation_baseline)
                self.validation_replay_state["unmatched_label_detections"] += unmatched
            self.validation_context_events = siguientes_eventos_contextuales
            self.validation_diagnostic_events = self._recortar_eventos_diagnosticos(
                eventos_diagnosticos, reloj, self.validation_replay_state["diagnostic_window_seconds"],
            )
            self.validation_context_concurrence = siguiente_concurrencia
            self.validation_suspicion_episodes = siguientes_episodios
            self.validation_results.extend(nuevos_resultados)
            self.validation_contextual_results.extend(nuevos_contextuales)
            self.validation_last_results = nuevos_resultados[
                -VALIDATION_REPLAY_STATUS_RESULTS:
            ]
            self.validation_last_contextual_results = nuevos_contextuales[
                -VALIDATION_REPLAY_STATUS_RESULTS:
            ]
            delays_lote = [
                max(
                    0.0,
                    (
                        reloj
                        - pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True)
                    ).total_seconds(),
                )
                for item in nuevas_detecciones
                if pd.notna(pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True))
            ]
            outcomes = dict(
                self.validation_replay_state.get("contextual_outcomes", {})
            )
            for item in nuevos_contextuales:
                outcome = str(item.get("context_outcome", "unexplained"))
                outcomes[outcome] = int(outcomes.get(outcome, 0)) + 1
            delay_sum = float(
                self.validation_replay_state.get(
                    "detection_delay_sum_seconds", 0.0
                )
            ) + sum(delays_lote)
            max_delay_anterior = self.validation_replay_state.get(
                "max_detection_delay_seconds"
            )
            max_delay_lote = max(delays_lote) if delays_lote else None
            if max_delay_anterior is None:
                max_delay = max_delay_lote
            elif max_delay_lote is None:
                max_delay = float(max_delay_anterior)
            else:
                max_delay = max(float(max_delay_anterior), max_delay_lote)
            self.validation_replay_state.update(
                {
                    "status": (
                        "completed"
                        if fin >= total
                        else "running" if _automatic else "paused"
                    ),
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
                    "detection_count": int(
                        self.validation_replay_state.get("detection_count", 0)
                    )
                    + len(nuevas_detecciones),
                    "emitted_alert_count": int(self.validation_replay_state.get("emitted_alert_count", 0))
                    + len(nuevos_resultados),
                    "forwarded_count": int(self.validation_replay_state.get(
                        "forwarded_count", self.validation_replay_state.get("contextual_count", 0)))
                    + agrupacion_lote["forwarded"],
                    "suppressed_count": int(self.validation_replay_state.get("suppressed_count", 0))
                    + agrupacion_lote["suppressed"],
                    "contextual_count": int(
                        self.validation_replay_state.get("contextual_count", 0)
                    )
                    + len(nuevos_contextuales),
                    "environmental_count": int(
                        self.validation_replay_state.get("environmental_count", 0)
                    )
                    + rutas_lote["environmental_count"],
                    "pending_classification_count": int(
                        self.validation_replay_state.get(
                            "pending_classification_count", 0
                        )
                    )
                    + rutas_lote["pending_classification_count"],
                    "classification_count": int(self.validation_replay_state.get("classification_count", 0))
                    + rutas_lote["classification_count"],
                    "indeterminate_count": int(self.validation_replay_state.get("indeterminate_count", 0))
                    + rutas_lote["indeterminate_count"],
                    "no_fault_count": int(self.validation_replay_state.get("no_fault_count", 0))
                    + rutas_lote["no_fault_count"],
                    "classifier_status": (
                        rutas_lote["pipeline_summary"]["classifier_status"]
                        if rutas_lote["classification_count"]
                        else self.validation_replay_state.get("classifier_status", "ready")
                    ),
                    "contextual_outcomes": outcomes,
                    "contextual_coincidences": int(self.validation_replay_state.get("contextual_coincidences", 0))
                    + sum(bool(item.get("common_cause", {}).get("contextual_coincidence")) for item in nuevos_contextuales),
                    "possible_common_cause": int(
                        self.validation_replay_state.get(
                            "possible_common_cause", 0
                        )
                    )
                    + sum(
                        bool(item.get("common_cause", {}).get("candidate"))
                        for item in nuevos_contextuales
                    ),
                    "detection_delay_sum_seconds": delay_sum,
                    "max_detection_delay_seconds": max_delay,
                    "last_batch": {
                        "events": int(len(lote)),
                        "processing_elapsed_ms": round(elapsed * 1000.0, 3),
                        "simulated_span_seconds": round(simulated_span, 3),
                        "events_per_second": round(len(lote) / max(elapsed, 1e-9), 3),
                        "keeps_up_with_event_time": elapsed <= max(simulated_span, 0.001),
                        "new_detections": len(nuevas_detecciones),
                        "new_emitted_alerts": len(nuevos_resultados),
                        "forwarded": agrupacion_lote["forwarded"],
                        "suppressed": agrupacion_lote["suppressed"],
                        "contextual_evaluated": len(nuevos_contextuales),
                        "classified": rutas_lote["classification_count"],
                        "indeterminate": rutas_lote["indeterminate_count"],
                        "no_fault": rutas_lote["no_fault_count"],
                        "environmental": rutas_lote["environmental_count"],
                        "pending_classification": rutas_lote["pending_classification_count"],
                        "timings_by_kind_ms": resultado.get("summary", {}).get(
                            "timings_by_kind_ms", {}
                        ),
                        "contextual_elapsed_ms": contextual_elapsed,
                    },
                }
            )
            respuesta = self._payload_reproduccion_validacion(
                new_results=nuevos_resultados[-VALIDATION_REPLAY_MEMORY_RESULTS:],
                new_contextual_results=nuevos_contextuales[
                    -VALIDATION_REPLAY_MEMORY_RESULTS:
                ],
            )
            self._publicar_snapshot_reproduccion()
            return respuesta
    def reiniciar_reproduccion_validacion(self) -> dict[str, Any]:
        self.validation_replay_pause.set()
        with self.validation_replay_lock:
            if (
                self.historial_validacion is None
                or self.eventos_validacion_ordenados is None
            ):
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
            self.validation_baseline = None
            self.validation_context_map = None
            self.validation_context_events = None
            self.validation_diagnostic_events = None
            self.validation_context_concurrence.clear()
            self.validation_suspicion_episodes.clear()
            self.validation_results = deque(maxlen=VALIDATION_REPLAY_MEMORY_RESULTS)
            self.validation_contextual_results = deque(
                maxlen=VALIDATION_REPLAY_MEMORY_RESULTS
            )
            self.validation_last_results = []
            self.validation_last_contextual_results = []
            self._reiniciar_almacen_resultados_validacion()
            labelled_evaluation.reset(self._ruta_almacen_resultados_validacion())
            self.validation_replay_state["unmatched_label_detections"] = 0
            self.validation_replay_generation += 1
            self.validation_replay_worker = None
            self.validation_replay_pause = Event()
            self.validation_replay_state.update(
                {
                    "status": "ready",
                    "detector_state": "pending_until_advance",
                    "cursor": 0,
                    "current_timestamp": self.validation_replay_state.get("start"),
                    "processing_elapsed_seconds": 0.0,
                    "contextual_elapsed_ms": 0.0,
                    "detection_count": 0,
                    "forwarded_count": 0,
                    "suppressed_count": 0,
                    "contextual_count": 0,
                    "environmental_count": 0,
                    "pending_classification_count": 0,
                    "classification_count": 0,
                    "indeterminate_count": 0,
                    "no_fault_count": 0,
                    "classifier_status": "ready",
                    "diagnostic_run_id": uuid4().hex,
                    "contextual_outcomes": {},
                    "contextual_coincidences": 0,
                    "possible_common_cause": 0,
                    "detection_delay_sum_seconds": 0.0,
                    "max_detection_delay_seconds": None,
                    "last_error": None,
                    "automatic": False,
                    "last_batch": None,
                }
            )
            self.validation_replay_state.pop("diagnostic_window_seconds", None)
            return self._publicar_snapshot_reproduccion()
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
        self.validation_context_concurrence.clear()
        self.validation_suspicion_episodes.clear()
    @staticmethod
    def _clave_evento(item: dict[str, Any]) -> tuple[str, str]:
        timestamp = pd.to_datetime(item.get("timestamp"), errors="coerce", utc=True)
        return (
            str(item.get("entity_id", "")),
            str(item.get("timestamp")) if pd.isna(timestamp) else timestamp.isoformat(),
        )
    def _diagnosticar_contextuales(
        self,
        estadisticos: list[dict[str, Any]],
        contextuales: list[dict[str, Any]],
        eventos: pd.DataFrame | None = None,
        source: str = "history",
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Ejecuta F3 una vez en el productor; las consultas leen estos snapshots."""
        if self.diagnostic_workbench is not None:
            return self.diagnostic_workbench.enrich_contextual(
                estadisticos, contextuales, events=eventos,
                source=source, run_id=run_id or uuid4().hex,
            )
        from app.modulos.clasificador.disponibilidad import unavailable_results
        return unavailable_results(contextuales)
    @staticmethod
    def _es_ambiental_terminal(contextual: dict[str, Any]) -> bool:
        route = contextual.get("routing", {})
        return (
            contextual.get("contextual_class") == "environmental_anomaly"
            and route.get("terminal") is True
            and route.get("send_to_classifier") is False
        )
    @staticmethod
    def _estado_clasificador(resultados: list[dict[str, Any]]) -> str:
        if not resultados:
            return "ready"
        modes = {item.get("diagnosis_mode") for item in resultados}
        if "active_model" in modes:
            return "active_model"
        return "unavailable"
    def _salidas_pipeline(
        self,
        estadisticos: list[dict[str, Any]],
        contextuales: list[dict[str, Any]],
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Resume snapshots F1/F2/F3; no ejecuta inferencia al consultar resultados."""
        lookup = {self._clave_evento(item): item for item in contextuales}
        ambientales: list[dict[str, Any]] = []
        pendientes: list[dict[str, Any]] = []
        clasificados: list[dict[str, Any]] = []
        for statistical in estadisticos:
            if not enviar_a_contextual(statistical):
                continue
            contextual = lookup.get(self._clave_evento(statistical))
            if contextual is None:
                # La ausencia de contexto nunca descarta una sospecha.
                contextual = {
                    "event_id": f"{statistical.get('entity_id')}@{statistical.get('timestamp')}",
                    "entity_id": statistical.get("entity_id"),
                    "timestamp": statistical.get("timestamp"),
                    "context_outcome": "insufficient_coverage",
                    "contextual_class": None,
                    "alert": None,
                    "routing": {
                        "terminal": False,
                        "destination": "classifier",
                        "send_to_classifier": True,
                        "reason": "missing_contextual_result",
                    },
                }
            contextual = {**contextual, **({"episode": statistical["episode"]} if statistical.get("episode") else {})}
            if self._es_ambiental_terminal(contextual):
                ambientales.append({**contextual, "resolved_by": "contextual"})
            else:
                diagnosis = contextual.get("diagnosis") or {}
                routed = {
                    **contextual,
                    "classification_status": diagnosis.get("status", "pending"),
                }
                if diagnosis:
                    clasificados.append({**routed, "resolved_by": "classifier"})
                if not diagnosis or diagnosis.get("status") == "indeterminate":
                    pendientes.append(routed)
        indeterminados = sum(item["diagnosis"].get("status") == "indeterminate" for item in clasificados)
        sin_falla = sum(item["diagnosis"].get("status") == "no_fault_evidence" for item in clasificados)
        return {
            "environmental_count": len(ambientales),
            "pending_classification_count": len(pendientes),
            "classification_count": len(clasificados),
            "indeterminate_count": indeterminados,
            "no_fault_count": sin_falla,
            "environmental_results": ambientales[:limit],
            "pending_classification_results": pendientes[:limit],
            "classifier_results": clasificados[:limit],
            "pipeline_summary": {
                "version": "sequential-diagnostic-v2",
                "statistical": resumen_episodios(estadisticos)["detected"],
                "grouping": resumen_episodios(estadisticos),
                "contextual": len(contextuales),
                "environmental": len(ambientales),
                "pending_classification": len(pendientes),
                "classifier": len(clasificados),
                "indeterminate": indeterminados,
                "no_fault": sin_falla,
                "classifier_status": self._estado_clasificador(clasificados),
            },
        }
    def _payload_reproduccion_validacion(
        self,
        new_results: list[dict[str, Any]] | None = None,
        new_contextual_results: list[dict[str, Any]] | None = None,
        include_recent: bool = False,
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
                "recent_results": [],
                "recent_contextual_results": [],
                **self._salidas_pipeline([], []),
            }
        payload = dict(self.validation_replay_state)
        rows = int(payload.get("rows", 0))
        cursor = int(payload.get("cursor", 0))
        detection_count = int(payload.get("detection_count", len(self.validation_results)))
        contextual_count = int(
            payload.get("contextual_count", len(self.validation_contextual_results))
        )
        environmental_count = int(payload.get("environmental_count", 0))
        pending_count = int(payload.get("pending_classification_count", 0))
        classification_count = int(payload.get("classification_count", 0))
        indeterminate_count = int(payload.get("indeterminate_count", 0))
        no_fault_count = int(payload.get("no_fault_count", 0))
        delay_sum = float(payload.pop("detection_delay_sum_seconds", 0.0))
        outcomes = dict(payload.pop("contextual_outcomes", {}))
        response_results = (
            new_results
            if new_results is not None
            else self.validation_last_results if include_recent else []
        )
        response_contextual = (
            new_contextual_results
            if new_contextual_results is not None
            else self.validation_last_contextual_results if include_recent else []
        )
        resumen_contextual = {
            "evaluated": contextual_count,
            "by_outcome": outcomes,
            "environmental_anomalies": environmental_count,
            "requires_classification": max(0, contextual_count - environmental_count),
            "possible_common_cause": int(payload.get("possible_common_cause", 0)),
            "contextual_coincidences": int(payload.get("contextual_coincidences", 0)),
            "elapsed_ms": round(float(payload.get("contextual_elapsed_ms", 0.0)), 3),
            "causal": True,
        }
        salidas_respuesta = self._salidas_pipeline(
            response_results,
            response_contextual,
        )
        nuevas_salidas = self._salidas_pipeline(
            new_results or [],
            new_contextual_results or [],
        )
        pipeline_summary = {
            "version": "sequential-diagnostic-v2",
            "statistical": detection_count,
            "grouping": {
                "detected": detection_count,
                "forwarded": int(payload.get("forwarded_count", contextual_count)),
                "suppressed": int(payload.get("suppressed_count", 0)),
            },
            "contextual": contextual_count,
            "environmental": environmental_count,
            "pending_classification": pending_count,
            "classifier": classification_count,
            "indeterminate": indeterminate_count,
            "no_fault": no_fault_count,
            "classifier_status": payload.get("classifier_status", "ready"),
            "causal": True,
            "uses_future_evaluation_rows": False,
        }
        return {
            "loaded": True,
            **payload,
            "processed_rows": cursor,
            "remaining_rows": max(0, rows - cursor),
            "progress": round(cursor * 100.0 / max(rows, 1), 4),
            "detection_count": detection_count,
            "average_detection_delay_seconds": (
                round(delay_sum / detection_count, 3) if detection_count else None
            ),
            # Compatibilidad: en una respuesta de avance estos nombres contienen
            # solamente el lote nuevo, nunca el acumulado completo.
            "results": response_results,
            "new_results": new_results or [],
            "recent_results": [],
            "contextual_count": contextual_count,
            "contextual_summary": resumen_contextual,
            "contextual_results": response_contextual,
            "new_contextual_results": new_contextual_results or [],
            "recent_contextual_results": [],
            "environmental_count": environmental_count,
            "pending_classification_count": pending_count,
            "classification_count": classification_count,
            "indeterminate_count": indeterminate_count,
            "no_fault_count": no_fault_count,
            "classifier_results": salidas_respuesta["classifier_results"],
            "new_classifier_results": nuevas_salidas["classifier_results"],
            "environmental_results": salidas_respuesta["environmental_results"],
            "pending_classification_results": salidas_respuesta[
                "pending_classification_results"
            ],
            "new_environmental_results": nuevas_salidas["environmental_results"],
            "new_pending_classification_results": nuevas_salidas["pending_classification_results"],
            "pipeline_summary": pipeline_summary,
        }
    def resultados_reproduccion_validacion(
        self,
        kind: str,
        offset: int,
        limit: int,
    ) -> dict[str, Any]:
        if kind not in {"statistical", "contextual", "environmental", "pending", "classifier"}:
            raise ValueError("Tipo de resultado de Evaluacion no valido.")
        if not self.validation_replay_state:
            raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")
        if self.validation_results_store_ready:
            total, items = self._leer_resultados_validacion(kind, offset, limit)
            return {
                "kind": kind,
                "offset": offset,
                "limit": limit,
                "total": total,
                "next_offset": (
                    offset + len(items) if offset + len(items) < total else None
                ),
                "items": items,
            }

        with self.validation_replay_lock:
            if not self.validation_replay_state:
                raise ValueError("Primero debes subir los eventos futuros de Evaluacion.")

            if kind == "statistical":
                total = len(self.validation_results)
                items = list(
                    islice(reversed(self.validation_results), offset, offset + limit)
                )
            elif kind == "contextual":
                total = len(self.validation_contextual_results)
                items = list(
                    islice(
                        reversed(self.validation_contextual_results),
                        offset,
                        offset + limit,
                    )
                )
            else:
                items = []
                vistos = 0
                for contextual in reversed(self.validation_contextual_results):
                    es_ambiental = self._es_ambiental_terminal(contextual)
                    diagnosis = contextual.get("diagnosis") or {}
                    matches = (
                        es_ambiental if kind == "environmental"
                        else bool(diagnosis) and not es_ambiental if kind == "classifier"
                        else not es_ambiental and (not diagnosis or diagnosis.get("status") == "indeterminate")
                    )
                    if not matches:
                        continue
                    if vistos >= offset:
                        items.append(
                            {
                                **contextual,
                                **(
                                    {"resolved_by": "contextual"}
                                    if es_ambiental
                                    else {
                                        "classification_status": diagnosis.get("status", "pending"),
                                        **({"resolved_by": "classifier"} if kind == "classifier" else {}),
                                    }
                                ),
                            }
                        )
                    vistos += 1
                    if len(items) >= limit:
                        break
                total = int(
                    self.validation_replay_state.get(
                        "environmental_count"
                        if kind == "environmental"
                        else "classification_count" if kind == "classifier"
                        else "pending_classification_count",
                        vistos,
                    )
                )

            return {
                "kind": kind,
                "offset": offset,
                "limit": limit,
                "total": total,
                "next_offset": offset + len(items) if offset + len(items) < total else None,
                "items": items,
            }
