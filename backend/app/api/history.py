from __future__ import annotations
from typing import Any
from fastapi import HTTPException, Query, Request
from fastapi import APIRouter
from app.api.schemas import AnalyzeRequest
from app.services.runtime import state
router = APIRouter()

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
