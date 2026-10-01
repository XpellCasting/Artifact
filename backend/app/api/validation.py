from __future__ import annotations
from typing import Any
from urllib.parse import unquote
from fastapi import HTTPException, Query, Request
import app.database.evaluacion_etiquetada as labelled_evaluation
from fastapi import APIRouter
from app.api.schemas import ValidationReplayAdvanceRequest
from app.services.runtime import state
router = APIRouter()

@router.post("/validation/replay/upload", tags=["validation-replay"])
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

@router.post("/validation/labelled/upload", tags=["validation-replay"])
async def labelled_upload(request: Request):
    from starlette.concurrency import run_in_threadpool
    try:
        return await run_in_threadpool(state.subir_validacion, await request.body(),
            unquote(request.headers.get("X-Filename", "ground_truth.csv")), True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/validation/labelled/metrics", tags=["validation-replay"])
def labelled_metrics():
    with state.validation_replay_lock:
        replay = state.validation_replay_state
        if not replay.get("labelled"):
            return {"loaded": False, "entities": []}
        return {"loaded": True, "filename": replay["filename"], "run_id": replay.get("diagnostic_run_id"), "status": replay["status"],
            "statistical_evaluation_version": replay.get("statistical_evaluation_version"),
            "unmatched_detections": replay.get("unmatched_label_detections", 0),
            "entities": labelled_evaluation.summary(state._ruta_almacen_resultados_validacion(), int(replay.get("cursor", 0)))}

@router.get("/validation/replay/status", tags=["validation-replay"])
def validation_replay_status() -> dict[str, Any]:
    return state.estado_reproduccion_validacion()

@router.get("/validation/replay/results", tags=["validation-replay"])
def validation_replay_results(
    kind: str = Query(
        default="statistical",
        pattern="^(statistical|contextual|environmental|pending|classifier)$",
    ),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=500),
) -> dict[str, Any]:
    try:
        return state.resultados_reproduccion_validacion(kind, offset, limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/validation/replay/start", tags=["validation-replay"])
def validation_replay_start(
    request: ValidationReplayAdvanceRequest,
) -> dict[str, Any]:
    try:
        return state.iniciar_reproduccion_validacion(
            batch_size=request.batch_size,
            suspicion_threshold=request.suspicion_threshold,
            suspicion_quiet_seconds=request.suspicion_quiet_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/validation/replay/pause", tags=["validation-replay"])
def validation_replay_pause() -> dict[str, Any]:
    try:
        return state.pausar_reproduccion_validacion()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/validation/replay/advance", tags=["validation-replay"])
def validation_replay_advance(
    request: ValidationReplayAdvanceRequest,
) -> dict[str, Any]:
    try:
        return state.avanzar_reproduccion_validacion(
            batch_size=request.batch_size,
            suspicion_threshold=request.suspicion_threshold,
            suspicion_quiet_seconds=request.suspicion_quiet_seconds,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/validation/replay/reset", tags=["validation-replay"])
def validation_replay_reset() -> dict[str, Any]:
    try:
        return state.reiniciar_reproduccion_validacion()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
