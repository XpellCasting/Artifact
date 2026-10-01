from __future__ import annotations
from copy import deepcopy
from typing import Any
from fastapi import HTTPException, Query, Request
import app.database.evaluacion_etiquetada as labelled_evaluation
from app.flow.definitions import default_flow
from app.flow.definitions import validate_flow
from app.flow.evaluacion import CausalFlowRunner
from fastapi import APIRouter
from app.api.schemas import FlowRunRequest
from app.services.runtime import state, flow_store, flow_evaluation_jobs
router = APIRouter()

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
        for node in flow.get("nodes", []):
            identifier = node.get("config", {}).get("model_id")
            if node.get("type") == "classifier" and identifier:
                state.diagnostic_workbench._read("models", identifier)
        with state.validation_replay_lock:
            if state.historial is None or state.future_baseline_template is None:
                raise ValueError("Carga el histórico de Desarrollo y espera a que se prepare el detector.")
            if state.eventos_validacion_ordenados is None:
                raise ValueError("Carga los eventos futuros o un conjunto etiquetado en Evaluación de flujos.")
            development = state.historial.eventos.copy(deep=True)
            future = state.eventos_validacion_ordenados.copy(deep=True)
            entities = state.historial.entidades.copy(deep=True)
            baseline = deepcopy(state.future_baseline_template)
            context_map = deepcopy(state.future_context_map_template or {})
            truth = labelled_evaluation.read_truth(state._ruta_almacen_resultados_validacion()) if state.validation_replay_state.get("labelled") else []
            filename = state.validation_replay_state.get("filename")
            runner = CausalFlowRunner(deepcopy(state.filtro_estadistico), deepcopy(state.filtro_contextual), state.diagnostic_workbench)
        def evaluate(progress, result_sink):
            summary, results, snapshot = runner.evaluate(flow, development, future, entities, context_map, baseline, truth, progress,
                                                         result_sink=result_sink)
            snapshot["source"]["filename"] = filename
            return summary, results, snapshot
        return flow_evaluation_jobs.start(flow_id, evaluate)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Flujo no encontrado.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/flow-evaluations/{job_id}", tags=["flow-evaluation"])
def flow_evaluation_status(job_id: str):
    try:
        return flow_evaluation_jobs.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="La evaluación no está disponible; el servicio pudo reiniciarse. Las ejecuciones terminadas siguen guardadas.") from exc

@router.get("/flow-runs", tags=["flow-lab"])
def flow_runs(flow_id: str | None = None) -> dict[str, Any]:
    return {"items": flow_store.list_runs(flow_id)}

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
