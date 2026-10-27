from __future__ import annotations
import os
from pathlib import Path
import tempfile
from typing import Literal
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from app.database.review import ReviewError
from app.services.diagnostico import DiagnosticWorkbench
from app.api.schemas import BuildInput

def crear_router_diagnostico(workbench: DiagnosticWorkbench):
    router = APIRouter(prefix="/diagnostics", tags=["Diagnóstico GAT-LSTM v2"])

    def translate(call, *args, **kwargs):
        try:
            return call(*args, **kwargs)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/status")
    def status():
        return translate(workbench.status)

    @router.get("/classifier-models")
    def classifier_models():
        return translate(workbench.classifier_models)

    @router.post("/models/import")
    async def import_model(request: Request):
        from app.modulos.clasificador.transfer import MAX_MODEL_ZIP_BYTES
        handle, name = tempfile.mkstemp(prefix="model-upload-", suffix=".zip", dir=workbench.review.temporary)
        path = Path(name)
        try:
            size = 0
            with os.fdopen(handle, "wb") as stream:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > MAX_MODEL_ZIP_BYTES:
                        raise HTTPException(413, "El ZIP supera el límite de 256 MiB.")
                    await run_in_threadpool(stream.write, chunk)
            return await run_in_threadpool(translate, workbench.import_model, path)
        finally:
            path.unlink(missing_ok=True)

    @router.get("/sources")
    def sources(method: Literal["gat_lstm"] = Query("gat_lstm")):
        return translate(workbench.sources, method == "gat_lstm")

    @router.get("/configurations")
    def configurations():
        return translate(workbench.configurations)

    @router.post("/configurations")
    def save_configuration(payload: dict):
        if set(payload) != {"name", "config"}:
            raise HTTPException(400, "Indica name y config para guardar una configuración GAT-LSTM.")
        return translate(workbench.save_gat_configuration, payload)

    @router.post("/sensors/onboard")
    def onboard_sensor(payload: dict):
        return translate(workbench.register_gat_sensor, payload)

    @router.post("/build")
    def build(payload: BuildInput):
        if payload.injection_ids or payload.include_drafts:
            raise HTTPException(400, "GAT-LSTM admite fuentes de Desarrollo y anotaciones revisadas.")
        return translate(workbench.start_job, workbench.build_gat_dataset,
                         payload.source_ids, payload.annotation_ids, payload.gat_config)

    @router.get("/jobs/{job_id}")
    def job(job_id: str):
        return translate(workbench.job, job_id)

    @router.get("/datasets/{dataset_id}")
    def dataset(dataset_id: str):
        return translate(lambda: workbench._dataset_summary(workbench._read("datasets", dataset_id)))

    @router.get("/datasets/{dataset_id}/export")
    def export(dataset_id: str):
        path = translate(workbench.export_dataset, dataset_id)
        return FileResponse(path, media_type="application/zip", filename=f"diagnostic-examples-{dataset_id}.zip", background=BackgroundTask(path.unlink, missing_ok=True))

    @router.post("/train")
    def train(payload: dict):
        if set(payload) - {"dataset_id", "method", "gat_config"} or "dataset_id" not in payload or payload.get("method", "gat_lstm") != "gat_lstm":
            raise HTTPException(400, "Indica dataset_id y opcionalmente method=gat_lstm y gat_config.")
        translate(workbench._read, "datasets", payload["dataset_id"])
        return translate(workbench.start_job, workbench.train_gat, payload["dataset_id"], None, payload.get("gat_config"))


    @router.post("/models/{model_id}/evaluate")
    def evaluate(model_id: str, payload: dict):
        if set(payload) != {"dataset_id", "partition"}:
            raise HTTPException(400, "Indica dataset_id y partition para evaluar.")
        trained = translate(workbench._read, "models", model_id)
        if trained.get("method") == "gat_lstm":
            return translate(workbench.start_job, workbench.evaluate_gat, model_id,
                             payload["dataset_id"], payload["partition"])
        raise HTTPException(409, "Solo se admite GAT-LSTM v2.")

    @router.post("/models/{model_id}/activate")
    def activate(model_id: str, payload: dict):
        if set(payload) - {"allow_exploratory"} or not isinstance(payload.get("allow_exploratory", False), bool):
            raise HTTPException(400, "allow_exploratory debe ser booleano.")
        return translate(workbench.activate, model_id, payload.get("allow_exploratory", False))

    @router.post("/models/{model_id}/deactivate")
    def deactivate(model_id: str):
        return translate(workbench.deactivate, model_id)

    @router.get("/results")
    def results(source: Literal["history", "validation_replay"] | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
        return translate(workbench.results, source, offset, limit)

    return router
