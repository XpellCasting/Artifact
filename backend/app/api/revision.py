from __future__ import annotations
import csv
import os
import tempfile
from pathlib import Path
from typing import Any
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from app.database.review import ReviewError, ReviewStore, PARTITIONS

def crear_router_revision(directory: Path) -> APIRouter:
    store = ReviewStore(directory)
    router = APIRouter(prefix="/review", tags=["Revisión manual de fragmentos"])

    def translate(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (UnicodeError, csv.Error) as exc:
            raise HTTPException(status_code=400, detail="CSV ilegible: use UTF-8 y una estructura tabular válida.") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/datasets")
    def datasets():
        return {"datasets": store.list_datasets()}

    @router.post("/datasets/upload")
    async def upload(request: Request, partition: str = "development"):
        if partition not in PARTITIONS:
            raise HTTPException(status_code=400, detail="Partición desconocida.")
        filename = request.headers.get("x-filename", "history.csv")
        handle, temporary_name = tempfile.mkstemp(prefix="upload-", suffix=".csv", dir=store.temporary)
        path = Path(temporary_name)
        try:
            with os.fdopen(handle, "wb") as stream:
                async for chunk in request.stream():
                    await run_in_threadpool(stream.write, chunk)
            return await run_in_threadpool(translate, store.import_csv, path, filename, partition)
        finally:
            path.unlink(missing_ok=True)

    @router.get("/datasets/{dataset_id}/fragment")
    def fragment(dataset_id: str, entity_ids: str, start: str, end: str,
                 padding_minutes: float = 30, max_points: int = 600):
        return translate(store.fragment, dataset_id, [value.strip() for value in entity_ids.split(",") if value.strip()],
                         start, end, padding_minutes, max_points)

    from app.services.exploracion import overview
    from app.services.exploracion import entity_detail

    @router.get("/datasets/{dataset_id}/exploration")
    def exploration(dataset_id: str):
        return translate(overview, store, dataset_id)

    @router.get("/datasets/{dataset_id}/exploration/entity")
    def exploration_entity(dataset_id: str, entity_id: str, start: str, end: str):
        return translate(entity_detail, store, dataset_id, entity_id, start, end)

    @router.get("/annotations")
    def annotations(dataset_id: str | None = None):
        return {"annotations": translate(store.list_annotations, dataset_id)}

    @router.post("/annotations")
    def create_annotation(payload: dict[str, Any]):
        return translate(store.save_annotation, payload)

    @router.put("/annotations/{annotation_id}")
    def update_annotation(annotation_id: str, payload: dict[str, Any]):
        return translate(store.save_annotation, payload, annotation_id)

    @router.get("/annotations/{annotation_id}/export")
    def export(annotation_id: str):
        path = translate(store.export_annotation, annotation_id)
        return FileResponse(path, media_type="application/zip", filename=f"fragment-{annotation_id}.zip",
                            background=BackgroundTask(path.unlink, missing_ok=True))

    from app.api.inyecciones import crear_router_inyecciones

    router.include_router(crear_router_inyecciones(store))
    return router
