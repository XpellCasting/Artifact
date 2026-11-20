from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from threading import Event
from urllib.parse import unquote
from uuid import uuid4
from typing import Any, Literal
from fastapi import HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
import app.database.evaluacion_etiquetada as labelled_evaluation
from app.flow.context_replay import ContextReplay
from app.flow.definitions import thesis_flows, validate_flow
from app.flow.evaluacion import CausalFlowRunner
from fastapi import APIRouter
from app.api.schemas import FlowRunRequest, HomeAssistantLiveRequest
from app.services.home_assistant_live import HomeAssistantEventSource, live_entity_catalog, reference_entities
from app.services.historial.cargar_historial import CargadorHistorial
from app.services.runtime import state, flow_store, flow_evaluation_jobs, home_assistant_bridge
router = APIRouter()


def require_classifier_models(flow):
    for node in flow.get("nodes", []):
        config = node.get("config", {})
        if node.get("type") == "classifier" and config.get("enabled") is not False:
            state.diagnostic_workbench.require_classifier_model(config.get("model_id"))

@router.get("/flows", tags=["flow-lab"])
def flows() -> dict[str, Any]:
    presets = thesis_flows()
    return {"items": flow_store.list(), "default": presets[0], "thesis_templates": presets,
            "evidence_contract": {"version": "module-envelope-v1", "policies": ["independent", "combine_v1"],
                                  "modules": ["statistical_focus", "contextual", "classifier"], "cycles": False}}

@router.post("/flows/validate", tags=["flow-lab"])
async def validate_flow_payload(request: Request) -> dict[str, Any]:
    try:
        return validate_flow(await request.json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/flows", tags=["flow-lab"])
async def create_flow(request: Request) -> dict[str, Any]:
    try:
        return flow_store.create(await request.json())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/flows/{flow_id}", tags=["flow-lab"])
def get_flow(flow_id: str) -> dict[str, Any]:
    try:
        return flow_store.get(flow_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc

@router.put("/flows/{flow_id}", tags=["flow-lab"])
async def update_flow(flow_id: str, request: Request) -> dict[str, Any]:
    try:
        return flow_store.update(flow_id, await request.json())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/flows/{flow_id}/runs", tags=["flow-lab"])
def run_flow(flow_id: str, request: FlowRunRequest) -> dict[str, Any]:
    return _start_csv_flow(flow_id)


@router.post("/flows/{flow_id}/runs/context", tags=["flow-lab"])
async def run_flow_with_context(flow_id: str, request: Request) -> dict[str, Any]:
    filename = Path(unquote(request.headers.get("X-Filename", "contexto.csv"))).name or "contexto.csv"
    return await run_in_threadpool(_start_csv_flow, flow_id, await request.body(), filename)


def _start_csv_flow(flow_id, context_csv=None, context_filename=None):
    try:
        flow = flow_store.get(flow_id)
        require_classifier_models(flow)
        with state.validation_replay_lock:
            if state.historial is None or state.future_baseline_template is None:
                raise ValueError("Carga el histórico de Desarrollo y espera a que se prepare el detector.")
            if state.eventos_validacion_ordenados is None or not state.validation_replay_state.get("labelled"):
                raise ValueError("Carga un conjunto CSV etiquetado en Evaluación de flujos.")
            development = state.historial.eventos.copy(deep=True)
            future = state.eventos_validacion_ordenados.copy(deep=True)
            entities = state.historial.entidades.copy(deep=True)
            baseline = deepcopy(state.future_baseline_template)
            context_map = deepcopy(state.future_context_map_template or {})
            truth = labelled_evaluation.read_truth(state._ruta_almacen_resultados_validacion())
            if not truth:
                raise ValueError("No hay etiquetas de referencia disponibles; vuelve a cargar el conjunto etiquetado.")
            filename = state.validation_replay_state.get("filename")
            runner = CausalFlowRunner(deepcopy(state.filtro_estadistico), deepcopy(state.filtro_contextual), state.diagnostic_workbench)
        context_source = None
        if context_csv is not None:
            if not any(node["type"] == "contextual" and node.get("config", {}).get("enabled") is not False
                       for node in flow["nodes"]):
                raise ValueError("Activa un filtro contextual en el flujo para usar el CSV de contexto.")
            if not context_csv.strip():
                raise ValueError("El CSV de contexto está vacío.")
            events = CargadorHistorial().cargar_real_csv(context_csv, context_filename).eventos
            context_source = ContextReplay(events, future, development["timestamp_dt"].max(), context_filename)

        cancel_event = Event()

        def evaluate(progress, result_sink):
            summary, results, snapshot = runner.evaluate(flow, development, future, entities, context_map, baseline, truth, progress,
                                                         result_sink=result_sink, context_source=context_source, cancel_event=cancel_event)
            snapshot["source"]["filename"] = filename
            return summary, results, snapshot
        return flow_evaluation_jobs.start(flow_id, evaluate, flow=flow, context_source=context_source,
                                         cancel_event=cancel_event, test_sources=future["entity_id"].unique())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/flow-evaluations", tags=["flow-evaluation"])
def shared_flow_evaluation():
    return {"job": flow_evaluation_jobs.latest(include_run=False), "latest_run_id": flow_store.latest_run_id()}


@router.get("/flow-evaluations/{job_id}", tags=["flow-evaluation"])
def flow_evaluation_status(job_id: str, compact: bool = False):
    try:
        return flow_evaluation_jobs.get(job_id, compact=compact)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="La evaluación no está disponible; el servicio pudo reiniciarse. Las ejecuciones terminadas siguen guardadas.") from exc

@router.get("/flows/live/entities", tags=["flow-evaluation"])
def live_flow_entities():
    try:
        with state.validation_replay_lock:
            reference = reference_entities(state.historial.entidades if state.historial is not None else None)
        catalog = live_entity_catalog(home_assistant_bridge, reference)
        return {"entities": [catalog[entity_id] for entity_id in sorted(catalog)]}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/flows/{flow_id}/live/start", tags=["flow-evaluation"])
def start_live_flow(flow_id: str, request: HomeAssistantLiveRequest):
    try:
        home_assistant_bridge.require_connected()
        flow = flow_store.get(flow_id)
        require_classifier_models(flow)
        with state.validation_replay_lock:
            if state.historial is None or state.future_baseline_template is None:
                raise ValueError("Carga un historial en el Selector de Historial y espera a que se prepare el detector.")
            development = state.historial.eventos.copy(deep=True)
            entities = state.historial.entidades.copy(deep=True)
            baseline = deepcopy(state.future_baseline_template)
            context_map = deepcopy(state.future_context_map_template or {})
            runner = CausalFlowRunner(deepcopy(state.filtro_estadistico), deepcopy(state.filtro_contextual), state.diagnostic_workbench)
        source = HomeAssistantEventSource(entities, development["timestamp_dt"].max(), request.entity_ids,
                                          bridge=home_assistant_bridge)
        def evaluate(progress, result_sink):
            summary, results, snapshot = runner.evaluate(
                flow, development, development.iloc[:0].copy(), entities, context_map, baseline,
                progress=progress, result_sink=result_sink, batch_source=source.batches(), source_identity=uuid4().hex,
            )
            if source.error and not summary["candidate_count"]:
                raise ValueError(source.error)
            snapshot["source"].update(filename="Eventos entrantes de Home Assistant", type="home_assistant",
                                      entity_ids=sorted(source.selected),
                                      connection_error=source.error,
                                      reception={**source.status(), "status": "failed" if source.error else "stopped"})
            return summary, results, snapshot
        return flow_evaluation_jobs.start(flow_id, evaluate, live_source=source, flow=flow)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/flow-evaluations/{job_id}/stop", tags=["flow-evaluation"])
def stop_flow_evaluation(job_id: str):
    try:
        return flow_evaluation_jobs.stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="La ejecución ya no está disponible.") from exc

@router.get("/flow-runs", tags=["flow-lab"])
def flow_runs(flow_id: str | None = None, compact: bool = False) -> dict[str, Any]:
    return {"items": flow_store.list_runs(flow_id, compact=compact)}

@router.get("/flow-alerts", tags=["flow-lab"])
def flow_alerts(
    before_id: int | None = Query(default=None, ge=1),
    limit: int = Query(default=50, ge=1, le=100),
    severity: Literal["fault", "anomaly"] | None = Query(default=None),
) -> dict[str, Any]:
    return flow_store.alerts(before_id, limit, severity)

@router.get("/flow-runs/{run_id}/results", tags=["flow-lab"])
def flow_run_results(
    run_id: str,
    node_id: str = "output",
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        return flow_store.results(run_id, node_id, offset, limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada.") from exc


@router.get("/flow-runs/{run_id}/metrics", tags=["flow-lab"])
def flow_run_metrics(run_id: str) -> dict[str, Any]:
    try:
        return flow_store.run_metrics(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada.") from exc


@router.get("/flow-runs/{run_id}/confidence", tags=["flow-lab"])
def flow_run_confidence(run_id: str, entity_id: str | None = None,
                        offset: int = Query(default=0, ge=0), limit: int = Query(default=1000, ge=1, le=5000)):
    try:
        return flow_store.confidence_series(run_id, entity_id, offset, limit)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Ejecución no encontrada.") from exc
