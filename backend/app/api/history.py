from __future__ import annotations
from typing import Any
from urllib.parse import unquote
from fastapi import HTTPException, Query, Request
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool
from app.api.schemas import AnalyzeRequest, HomeAssistantHistoryRequest, HistorySelectionRequest
from app.services.home_assistant_history import HomeAssistantHistorySource
from app.services.runtime import state, home_assistant_bridge, history_candidates
router = APIRouter()

@router.get("/history/home-assistant/config", tags=["history"])
def home_assistant_config() -> dict[str, Any]:
    return home_assistant_bridge.status()

@router.post("/history/home-assistant/import/start", tags=["history"])
def home_assistant_import(request: HomeAssistantHistoryRequest) -> dict[str, Any]:
    try:
        if not request.entity_ids:
            raise ValueError("Activa al menos un sensor antes de preparar los filtros.")
        home_assistant_bridge.require_connected()
        source = HomeAssistantHistorySource(
            start=request.start, end=request.end, entity_ids=request.entity_ids,
            transport=home_assistant_bridge.history_transport,
        )
        return state.iniciar_importacion_home_assistant(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/home-assistant/entities", tags=["history"])
def home_assistant_entities(request: HomeAssistantHistoryRequest):
    try:
        home_assistant_bridge.require_connected()
        source = HomeAssistantHistorySource(request.start, request.end,
            transport=home_assistant_bridge.history_transport, catalog_transport=home_assistant_bridge.entity_catalog)
        return source.discover_entities()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/home-assistant/prepare", tags=["history"])
def prepare_home_assistant(request: HomeAssistantHistoryRequest):
    if not request.entity_ids:
        raise HTTPException(status_code=400, detail="Activa al menos un sensor antes de preparar los filtros.")
    return home_assistant_import(request)

@router.post("/history/selection/csv", tags=["history"])
async def inspect_history_csv(request: Request):
    identifier, destination = history_candidates.create()
    try:
        with destination.open("wb") as handle:
            async for chunk in request.stream():
                await run_in_threadpool(handle.write, chunk)
        filename = unquote(request.headers.get("X-Filename", "history.csv")).replace("\\", "/").rsplit("/", 1)[-1]
        return await run_in_threadpool(history_candidates.inspect, identifier, filename)
    except BaseException as exc:
        history_candidates.discard(identifier)
        if isinstance(exc, ValueError):
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise

@router.post("/history/selection/saved", tags=["history"])
def inspect_saved_csv():
    try:
        return history_candidates.from_saved(state.real_history_path)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/selection/{candidate_id}/prepare", tags=["history"])
def prepare_selected_csv(candidate_id: str, request: HistorySelectionRequest):
    try:
        return state.iniciar_seleccion_csv(history_candidates, candidate_id, request.entity_ids)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/history/static/status", tags=["history"])
def history_static_status(
    autoload: bool = Query(default=True),
) -> dict[str, Any]:
    payload = state.status(autoload=autoload)
    with state.history_load_jobs_lock:
        job_id = next(reversed(state.history_load_jobs), None)
    payload["history_load"] = state.estado_carga_real(job_id) if job_id else None
    return payload

@router.post("/history/static/ensure", tags=["history"])
def history_static_ensure() -> dict[str, Any]:
    try:
        return state.asegurar_canonico()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@router.post("/history/real/load", tags=["history"])
def history_real_load() -> dict[str, Any]:
    try:
        return state.cargar_real()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/real/load/start", tags=["history"])
def history_real_load_start() -> dict[str, Any]:
    try:
        return state.iniciar_carga_real()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/real/upload", tags=["history"])
async def history_real_upload(request: Request) -> dict[str, Any]:
    try:
        return state.subir_real(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/history/real/upload/start", tags=["history"])
async def history_real_upload_start(request: Request) -> dict[str, Any]:
    try:
        return state.iniciar_subida_real(await request.body())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/history/real/load/jobs/{job_id}", tags=["history"])
def history_real_load_job(job_id: str) -> dict[str, Any]:
    try:
        return state.estado_carga_real(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

@router.post("/history/analyze", tags=["history"])
def history_analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        return state.analizar_historial(request.suspicion_threshold, request.suspicion_quiet_seconds)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/history/normalization/preview", tags=["history"])
def history_normalization_preview() -> dict[str, Any]:
    try:
        return state.vista_normalizacion()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/history/timeline", tags=["history"])
def history_timeline(
    start: str | None = None,
    end: str | None = None,
    max_points: int = Query(default=1000, ge=10, le=10000),
) -> dict[str, Any]:
    return state.timeline_historial(start=start, end=end, max_points=max_points)
