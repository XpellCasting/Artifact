from __future__ import annotations
from typing import Any
from fastapi import HTTPException
from fastapi import APIRouter
import pandas as pd
from app.services.runtime import state, flow_evaluation_jobs
router = APIRouter()


def map_view(mapa, compact=False):
    """Report age without changing the frozen detector's calibration."""
    refresh = dict(mapa.get("refresh", {}))
    built_at = pd.to_datetime(mapa.get("built_at"), utc=True, errors="coerce")
    max_age = mapa.get("config", {}).get("max_map_age_days", 7)
    if pd.notna(built_at) and (pd.Timestamp.now(tz="UTC") - built_at).total_seconds() > max_age * 86400:
        refresh["stale"] = True
        refresh["stale_reasons"] = list(dict.fromkeys([*refresh.get("stale_reasons", []), "max_age_exceeded"]))
    if not compact:
        return {**mapa, "refresh": refresh}
    edge_fields = {"edge_id", "source", "target", "source_kind", "target_kind", "relation_family",
                   "relation_view", "metric", "strength", "association_signed", "accepted", "window_seconds"}
    return {**{key: value for key, value in mapa.items() if key not in {"decision_calibrations", "nodes", "edges"}},
            "compact": True, "refresh": refresh,
            "nodes": [{key: value for key, value in node.items() if key != "temporal_profile"}
                      for node in mapa.get("nodes", [])],
            "edges": [{key: value for key, value in edge.items() if key in edge_fields}
                      for edge in mapa.get("edges", [])]}


@router.get("/context/map", tags=["contextual"])
def contextual_map(compact: bool = False) -> dict[str, Any]:
    try:
        return map_view(state.mapa_contextual(), compact)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/context/map/edges/{edge_id}", tags=["contextual"])
def contextual_edge(edge_id: str) -> dict[str, Any]:
    try:
        mapa = state.mapa_contextual()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    for edge in mapa.get("edges", []):
        if edge.get("edge_id") == edge_id:
            return edge
    raise HTTPException(status_code=404, detail="La relación ya no está disponible. Actualiza el mapa.")


@router.post("/context/map/rebuild", tags=["contextual"])
def rebuild_contextual_map(compact: bool = False) -> dict[str, Any]:
    job = flow_evaluation_jobs.latest(include_run=False)
    if job and job.get("status") == "running":
        raise HTTPException(status_code=409, detail="Espera a que termine la evaluación antes de reconstruir el mapa contextual.")
    try:
        return map_view(state.mapa_contextual(force=True), compact)
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
