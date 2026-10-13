"""Composición y ciclo de vida de la API."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import CORS_ORIGINS
from app.services.runtime import state, REVIEW_DATA_DIRECTORY
from app.api.revision import crear_router_revision
from app.api.diagnostico import crear_router_diagnostico
from app.api.health import router as health_router
from app.api.history import router as history_router
from app.api.validation import router as validation_router
from app.api.configuracion import router as configuracion_router
from app.api.contextual import router as contextual_router
from app.api.flows import router as flows_router

@asynccontextmanager
async def lifespan(api):
    if state.historial is None and state.real_history_path.is_file():
        state.iniciar_carga_real()
    yield

def crear_app():
    api = FastAPI(title="Artefacto de tesis", version="0.6.0", lifespan=lifespan,
                  description="Filtros estadístico y contextual con diagnóstico GAT-LSTM v2.")
    api.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    api.include_router(crear_router_revision(REVIEW_DATA_DIRECTORY))
    api.include_router(crear_router_diagnostico(state.diagnostic_workbench))
    api.include_router(health_router)
    api.include_router(history_router)
    api.include_router(validation_router)
    api.include_router(configuracion_router)
    api.include_router(contextual_router)
    api.include_router(flows_router)
    return api

app = crear_app()
