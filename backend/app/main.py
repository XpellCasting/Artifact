from __future__ import annotations

import os
import re
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from .pipeline import create_state


class DatasetLoadRequest(BaseModel):
    dataset_dir: str | None = None


class EntitySelectionRequest(BaseModel):
    entity_ids: list[str]


class SimulationGenerateRequest(BaseModel):
    duration_hours: int = 72
    sample_seconds: int = 10
    seed: int = 20260614
    faults: list[dict[str, Any]] | None = None
    selected_entities: list[str] | None = None


class SimulationFaultsRequest(BaseModel):
    faults: list[dict[str, Any]]


class StatisticalFilterRequest(BaseModel):
    suspicion_threshold: float = 0.5


class RealtimeTickRequest(BaseModel):
    selected_entities: list[str]
    tick_seconds: int = 60
    faults: list[dict[str, Any]] = []
    suspicion_threshold: float = 0.5


class RealHistoryLoadRequest(BaseModel):
    path: str | None = None


openapi_tags = [
    {
        "name": "health",
        "description": "Control operativo basico del servicio API.",
    },
    {
        "name": "datasets",
        "description": "Carga de historicos Home Assistant y fixtures de evaluacion.",
    },
    {
        "name": "simulation",
        "description": "Historico canonico estatico, ticks de tiempo real y fallas sinteticas.",
    },
    {
        "name": "entities",
        "description": "Listado, clasificacion y seleccion de entidades diagnosticables.",
    },
    {
        "name": "preprocess",
        "description": "Validacion, limpieza, normalizacion y alineacion temporal.",
    },
    {
        "name": "baseline",
        "description": "Construccion de base de normalidad para cada entidad.",
    },
    {
        "name": "filters",
        "description": "Ejecucion del filtro estadistico-temporal.",
    },
    {
        "name": "results",
        "description": "Consulta de sospechas estadisticas calculadas.",
    },
]

app = FastAPI(
    title="Artifacto estadistico HA",
    version="0.1.0",
    description=(
        "API del prototipo para crear historico canonico estatico, emitir "
        "ticks de tiempo real y ejecutar el filtro estadistico-temporal."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=openapi_tags,
    swagger_ui_parameters={
        "defaultModelsExpandDepth": -1,
        "displayRequestDuration": True,
        "docExpansion": "none",
        "operationsSorter": "method",
        "tagsSorter": "alpha",
    },
)
state = create_state()

DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
origins = [item.strip() for item in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",") if item.strip()]
LOCAL_ORIGIN_RE = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    origin = request.headers.get("origin")
    headers = {}
    if origin and (origin in origins or LOCAL_ORIGIN_RE.match(origin)):
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
        headers["Vary"] = "Origin"
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "error_type": type(exc).__name__},
        headers=headers,
    )


@app.get("/health", tags=["health"], summary="Verifica que la API este operativa")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "dataset_dir": str(state.dataset_dir),
        "fixture_dir": str(state.fixture_dir),
        "static_history_dir": str(state.static_history_dir),
        "real_history_path": str(state.real_history_path),
    }


@app.post("/history/static/ensure", tags=["simulation"], summary="Crea o carga historico canonico estatico")
def ensure_static_history_endpoint() -> dict[str, Any]:
    try:
        return state.ensure_static_history()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/history/static/status", tags=["simulation"], summary="Estado del historico canonico estatico")
def static_history_status() -> dict[str, Any]:
    return state.static_history_status()


@app.post("/history/real/load", tags=["simulation"], summary="Carga historico real desde CSV")
def load_real_history(request: RealHistoryLoadRequest | None = None) -> dict[str, Any]:
    try:
        return state.load_real_history(request.path if request else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/history/analyze", tags=["simulation"], summary="Analiza sospechas en el historico cargado")
def analyze_history(request: StatisticalFilterRequest | None = None) -> dict[str, Any]:
    threshold = request.suspicion_threshold if request else 0.5
    summary = state.run_historical_filter(suspicion_threshold=threshold)
    return {
        "summary": summary,
        "results": state.historical_results,
        "count": len(state.historical_results),
    }


@app.post("/realtime/reset", tags=["simulation"], summary="Limpia eventos de tiempo real")
def realtime_reset() -> dict[str, Any]:
    return state.reset_realtime()


@app.post("/realtime/tick", tags=["simulation"], summary="Genera un tick de tiempo real y ejecuta filtro")
def realtime_tick(request: RealtimeTickRequest) -> dict[str, Any]:
    try:
        return state.realtime_tick(
            selected_entities=request.selected_entities,
            tick_seconds=request.tick_seconds,
            faults=request.faults,
            suspicion_threshold=request.suspicion_threshold,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/realtime/current", tags=["simulation"], summary="Estado actual de tiempo real")
def realtime_current() -> dict[str, Any]:
    return state.current_realtime()


@app.get("/realtime/timeline", tags=["simulation"], summary="Linea temporal de historico reciente y tiempo real")
def realtime_timeline(
    start: str | None = None,
    end: str | None = None,
    max_points: int = 900,
) -> dict[str, Any]:
    return state.realtime_timeline(start=start, end=end, max_points=max_points)


@app.post("/simulation/generate", tags=["simulation"], summary="Genera escenario sintetico")
def generate_simulation_endpoint(request: SimulationGenerateRequest) -> dict[str, Any]:
    return state.generate_simulation(
        duration_hours=request.duration_hours,
        sample_seconds=request.sample_seconds,
        seed=request.seed,
        faults=request.faults,
        selected_entities=request.selected_entities,
    )


@app.get("/simulation/current", tags=["simulation"], summary="Obtiene el escenario sintetico actual")
def current_simulation() -> dict[str, Any]:
    return state.current_simulation()


@app.get("/simulation/timeline", tags=["simulation"], summary="Obtiene series temporales simuladas")
def simulation_timeline() -> dict[str, Any]:
    return state.simulation_timeline()


@app.post("/simulation/faults", tags=["simulation"], summary="Actualiza fallas configuradas")
def simulation_faults(request: SimulationFaultsRequest) -> dict[str, Any]:
    return state.set_simulation_faults(request.faults)


@app.get("/simulation/export/states.csv", tags=["simulation"], summary="Exporta estados simulados")
def export_states() -> Response:
    return csv_response(state.export_csv("states"), "ha_states.csv")


@app.get("/simulation/export/entities.csv", tags=["simulation"], summary="Exporta entidades simuladas")
def export_entities() -> Response:
    return csv_response(state.export_csv("entities"), "ha_entities.csv")


@app.get("/simulation/export/ground_truth.csv", tags=["simulation"], summary="Exporta ground truth simulado")
def export_ground_truth() -> Response:
    return csv_response(state.export_csv("ground_truth"), "ground_truth.csv")


@app.post("/datasets/load", tags=["datasets"], summary="Carga el dataset historico")
def load_dataset(request: DatasetLoadRequest | None = None) -> dict[str, Any]:
    try:
        return state.load_dataset(request.dataset_dir if request else None)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/entities", tags=["entities"], summary="Lista entidades cargadas")
def entities() -> list[dict[str, Any]]:
    return state.list_entities()


@app.post("/entities/selection", tags=["entities"], summary="Guarda la seleccion de entidades")
def select_entities(request: EntitySelectionRequest) -> dict[str, Any]:
    try:
        return state.select_entities(request.entity_ids)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/preprocess/validate", tags=["preprocess"], summary="Valida historicos y calidad de datos")
def validate() -> dict[str, Any]:
    return state.validate()


@app.post("/preprocess/normalize", tags=["preprocess"], summary="Normaliza y alinea entidades")
def normalize() -> dict[str, Any]:
    return state.normalize()


@app.post("/preprocess/features", tags=["preprocess"], summary="Calcula features temporales")
def features() -> dict[str, Any]:
    return state.build_feature_tables()


@app.post("/baseline/build", tags=["baseline"], summary="Construye la base de normalidad")
def baseline() -> dict[str, Any]:
    return state.build_baseline()


@app.post("/fixtures/multistate/load", tags=["datasets"], summary="Carga fixture multivalorico")
def load_multistate_fixture() -> dict[str, Any]:
    try:
        return state.load_multistate_fixture()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/filters/statistical/run", tags=["filters"], summary="Ejecuta el filtro estadistico")
def run_statistical_filter(request: StatisticalFilterRequest | None = None) -> dict[str, Any]:
    threshold = request.suspicion_threshold if request else 0.5
    return state.run_statistical_filter(suspicion_threshold=threshold)


@app.get("/results/statistical", tags=["results"], summary="Obtiene sospechas estadisticas")
def statistical_results() -> dict[str, Any]:
    combined = state.historical_results + state.results
    combined = sorted(combined, key=lambda item: item.get("suspicion_score", 0), reverse=True)[:1000]
    return {
        "results": combined,
        "count": len(combined),
        "historical_results": state.historical_results,
        "historical_count": len(state.historical_results),
        "realtime_results": state.results,
        "realtime_count": len(state.results),
    }


def csv_response(content: str, filename: str) -> Response:
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
