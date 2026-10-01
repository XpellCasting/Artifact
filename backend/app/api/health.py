from __future__ import annotations
from typing import Any
from fastapi import APIRouter
from app.services.runtime import state
router = APIRouter()

@router.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mensaje": "API operativa",
        "static_history_dir": str(state.static_history_dir),
        "real_history_path": str(state.real_history_path),
    }
