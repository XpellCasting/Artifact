from __future__ import annotations
from copy import deepcopy
from uuid import uuid4
from typing import Any, Literal
from fastapi import HTTPException, Query, Request
import app.database.evaluacion_etiquetada as labelled_evaluation
from app.flow.definitions import default_flow
from app.flow.definitions import validate_flow
from app.flow.evaluacion import CausalFlowRunner
from fastapi import APIRouter
from app.api.schemas import FlowRunRequest, HomeAssistantLiveRequest
from app.services.home_assistant_live import HomeAssistantEventSource
from app.services.runtime import state, flow_store, flow_evaluation_jobs, home_assistant_bridge
router = APIRouter()


def require_classifier_models(flow):
    for node in flow.get("nodes", []):
        config = node.get("config", {})
        if node.get("type") == "classifier" and config.get("enabled") is not False:
            state.diagnostic_workbench.require_classifier_model(config.get("model_id"))

@router.get("/flows", tags=["flow-lab"])
def flows() -> dict[str, Any]:
    return {"items": flow_store.list(), "default": default_flow(),
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
        def evaluate(progress, result_sink):
            summary, results, snapshot = runner.evaluate(flow, development, future, entities, context_map, baseline, truth, progress,
                                                         result_sink=result_sink)
            snapshot["source"]["filename"] = filename
            return summary, results, snapshot
        return flow_evaluation_jobs.start(flow_id, evaluate, flow=flow)
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
                                      connection_error=source.error,
                                      reception={**source.status(), "status": "failed" if source.error else "stopped"})
            return summary, results, snapshot
        return flow_evaluation_jobs.start(flow_id, evaluate, live_source=source, flow=flow)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/flow-evaluations/{job_id}/stop", tags=["flow-evaluation"])
def stop_live_flow(job_id: str):
    try:
        return flow_evaluation_jobs.stop(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="La recepción ya no está disponible.") from exc

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
