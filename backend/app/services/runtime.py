from __future__ import annotations
from app.config import setting
import json
import os
from functools import partial
from pathlib import Path
from app.services.historial.cargar_historial import CargadorHistorial
from app.modulos.contextual import ConfiguracionContextual
from app.modulos.contextual import FiltroContextualEspacioTemporal
from app.modulos.estadistico.filtro import FiltroEstadistico
from app.services.diagnostico import DiagnosticWorkbench
from app.database.flows import FlowStore
from app.database.confidence_entities import ConfidenceEntityStore
from app.flow.evaluacion import FlowEvaluationJobs
from app.services.aplicacion import EstadoAplicacion
from app.config import ARTIFACT_ROOT
from app.config import STORAGE_DIR
from app.services.home_assistant_bridge import HomeAssistantBridge
from app.services.history_selection import HistoryCandidates
REVIEW_DATA_DIRECTORY = Path(
    setting('REVIEW_DATA_DIR')
)
state = EstadoAplicacion(
    cargador=CargadorHistorial(),
    filtro_estadistico=FiltroEstadistico(),
    filtro_contextual=FiltroContextualEspacioTemporal(ConfiguracionContextual(
        continuous_max_age_seconds=float(setting('CONTEXT_CONTINUOUS_MAX_AGE_SECONDS')),
        continuous_max_age_seconds_by_entity=json.loads(
            setting('CONTEXT_CONTINUOUS_MAX_AGE_BY_ENTITY')
        ),
    )),
    static_history_dir=Path(
        setting('STATIC_HISTORY_DIR')
    ),
    real_history_path=Path(
        setting('REAL_HISTORY_PATH')
    ),
    analysis_cache_path=Path(
        setting('ANALYSIS_CACHE_PATH')
    ),
    validation_history_path=Path(
        setting('VALIDATION_HISTORY_PATH')
    ),
    context_map_cache_path=Path(
        setting('CONTEXT_MAP_CACHE_PATH')
    ),
    diagnostic_workbench=DiagnosticWorkbench(REVIEW_DATA_DIRECTORY),
)
flow_store = FlowStore(Path(setting('FLOW_LAB_DB_PATH')))
home_assistant_bridge = HomeAssistantBridge(STORAGE_DIR / "home_assistant_bridge.json")
confidence_entities = ConfidenceEntityStore(flow_store.path)
home_assistant_bridge.set_confidence_provider(partial(confidence_entities.snapshot, include_deleted=True))
flow_evaluation_jobs = FlowEvaluationJobs(flow_store, confidence_store=confidence_entities,
                                         confidence_publish=home_assistant_bridge.publish_confidence)
history_candidates = HistoryCandidates(STORAGE_DIR / "history_candidates")


