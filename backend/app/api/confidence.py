from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.database.confidence_entities import ConfidenceSelectionChanged, confidence_entity_id
from app.services.home_assistant_bridge import BridgeOperationUnsupported, BridgeUnavailable


class ConfidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_entity_id: str = Field(max_length=255)
    source_mode: Literal["live", "test"]
    revision: int = Field(ge=0)


class DeleteConfidenceEntities(BaseModel):
    entities: list[ConfidenceSelection] = Field(min_length=1, max_length=10000)


def create_confidence_router(store, bridge):
    router = APIRouter()

    @router.get("/confidence/entities", tags=["confidence"])
    def confidence_entities(compact: bool = False):
        return store.snapshot(compact=compact)

    @router.delete("/confidence/entities", tags=["confidence"])
    def delete_confidence_entities(request: DeleteConfidenceEntities):
        selection = {}
        try:
            for item in request.entities:
                confidence_entity_id(item.source_entity_id, test=item.source_mode == "test")
                selection[item.source_entity_id, item.source_mode] = item.revision
            removed = store.delete(selection)
        except ConfidenceSelectionChanged as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        synchronization = "completed"
        try:
            bridge.publish_confidence(removed)
        except BridgeOperationUnsupported:
            synchronization = "integration_update_required"
        except BridgeUnavailable:
            synchronization = "pending"
        return {"removed": len(removed), "synchronization": synchronization}

    return router
