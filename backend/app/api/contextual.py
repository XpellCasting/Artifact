from __future__ import annotations
from typing import Any
from fastapi import HTTPException
from fastapi import APIRouter
from app.services.runtime import state
router = APIRouter()

@router.get("/context/map", tags=["contextual"])
def contextual_map() -> dict[str, Any]:
    try:
        return state.mapa_contextual()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/context/map/rebuild", tags=["contextual"])
def rebuild_contextual_map() -> dict[str, Any]:
    try:
        return state.mapa_contextual(force=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.post("/context/analyze", tags=["contextual"])
def analyze_context() -> dict[str, Any]:
    try:
        return state.analizar_contexto_actual()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@router.get("/context/results", tags=["contextual"])
def contextual_results() -> dict[str, Any]:
    return state.contexto_actual()
