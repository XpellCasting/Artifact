from __future__ import annotations
from typing import Any
from fastapi import HTTPException
from fastapi import APIRouter
from app.api.schemas import ContinuousConfigRequest
from app.services.runtime import state
router = APIRouter()

@router.get("/config/continuous", tags=["config"])
def continuous_config() -> dict[str, Any]:
    try:
        return state.configuracion_continua()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/config/continuous", tags=["config"])
def save_continuous_config(request: ContinuousConfigRequest) -> dict[str, Any]:
    try:
        return state.guardar_configuracion_continua(request.configs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
