from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any
import pandas as pd
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.historial.cargar_historial import ResultadoHistorial
from app.modulos.contextual import FiltroContextualEspacioTemporal
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.application_common import DEFAULT_HISTORY_LOAD_SECONDS_PER_MIB, VALIDATION_REPLAY_MEMORY_RESULTS
from app.services.history import ServicesHistoryMixin
from app.services.replay import ServicesReplayMixin
from app.database.replay import DatabaseReplayMixin
from app.services.analysis import ServicesAnalysisMixin
from app.services.normality import ServicesNormalityMixin

@dataclass
class EstadoAplicacion(ServicesHistoryMixin, ServicesReplayMixin, DatabaseReplayMixin, ServicesAnalysisMixin, ServicesNormalityMixin):
    cargador: CargadorHistorial
    filtro_estadistico: FiltroEstadistico
    static_history_dir: Path
    real_history_path: Path
    analysis_cache_path: Path
    validation_history_path: Path | None = None
    filtro_contextual: FiltroContextualEspacioTemporal = field(
        default_factory=FiltroContextualEspacioTemporal
    )
    context_map_cache_path: Path | None = None
    diagnostic_workbench: Any = field(default=None, repr=False)
    historial: ResultadoHistorial | None = None
    resultados: list[dict[str, Any]] = field(default_factory=list)
    timeline_base_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    mapa_contextual_cache: dict[str, Any] | None = None
    resultados_contextuales: list[dict[str, Any]] = field(default_factory=list)
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
    future_preparation_cache: dict[str, Any] = field(default_factory=dict, repr=False)
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
    validation_diagnostic_events: pd.DataFrame | None = field(default=None, repr=False)
    validation_context_concurrence: dict = field(default_factory=dict, repr=False)
    validation_suspicion_episodes: dict = field(default_factory=dict, repr=False)
    validation_results: Any = field(
        default_factory=lambda: deque(maxlen=VALIDATION_REPLAY_MEMORY_RESULTS),
        repr=False,
    )
    validation_contextual_results: Any = field(
        default_factory=lambda: deque(maxlen=VALIDATION_REPLAY_MEMORY_RESULTS),
        repr=False,
    )
    validation_replay_state: dict[str, Any] = field(default_factory=dict, repr=False)
    validation_replay_lock: Any = field(default_factory=Lock, repr=False)
    validation_replay_pause: Any = field(default_factory=Event, repr=False)
    validation_replay_worker: Thread | None = field(default=None, repr=False)
    validation_replay_generation: int = field(default=0, repr=False)
    validation_last_results: list[dict[str, Any]] = field(default_factory=list, repr=False)
    validation_last_contextual_results: list[dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    validation_replay_snapshot: dict[str, Any] = field(default_factory=dict, repr=False)
    validation_results_store_ready: bool = field(default=False, repr=False)


