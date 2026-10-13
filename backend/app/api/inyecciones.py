from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.database.review import ReviewError
from app.database.review import ReviewStore
from app.services.inyecciones import InjectionStore

def crear_router_inyecciones(store: ReviewStore) -> APIRouter:
    from app.services.escenarios import ScenarioStore
    injections = InjectionStore(store)
    scenarios = ScenarioStore(injections)
    router = APIRouter(tags=["Inyección controlada de fragmentos"])

    def translate(call, *args):
        try:
            return call(*args)
        except ReviewError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    @router.get("/injections")
    def list_injections(dataset_id: str | None = None):
        return {"injections": translate(injections.list_injections, dataset_id)}

    @router.post("/injections")
    def create_injection(payload: dict[str, Any]):
        return translate(injections.create, payload)

    @router.get("/injections/{injection_id}")
    def get_injection(injection_id: str):
        return translate(injections.injection, injection_id)

    @router.get("/injections/{injection_id}/export")
    def export_injection(injection_id: str):
        path = translate(injections.export_injection, injection_id)
        return FileResponse(path, media_type="application/zip", filename=f"injection-{injection_id}.zip")

    @router.get("/scenarios")
    def list_scenarios(dataset_id: str):
        return {"scenarios": translate(scenarios.list, dataset_id)}

    @router.post("/scenarios")
    def create_scenario(payload: dict[str, Any]):
        return translate(scenarios.create, payload)

    @router.get("/scenarios/{scenario_id}/export")
    def export_scenario(scenario_id: str):
        translate(scenarios.get, scenario_id)
        return FileResponse(scenarios.path(scenario_id) / "scenario.zip", media_type="application/zip", filename=f"historico-escenario-{scenario_id}.zip")

    @router.get("/scenarios/{scenario_id}/csv")
    def export_labelled_csv(scenario_id: str):
        snapshot = translate(scenarios.get, scenario_id)
        if not snapshot.get("labelled_csv"):
            raise HTTPException(409, "Este escenario es anterior al CSV etiquetado. Carga su selección y guarda una nueva versión.")
        return FileResponse(scenarios.path(scenario_id) / "labelled.csv", media_type="text/csv", filename=f"historico-etiquetado-{scenario_id}.csv")

    return router
