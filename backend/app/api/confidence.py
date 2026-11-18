from fastapi import APIRouter


def create_confidence_router(store):
    router = APIRouter()

    @router.get("/confidence/entities", tags=["confidence"])
    def confidence_entities():
        return store.snapshot()

    return router
