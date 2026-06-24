from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .Cargar_Historial.Cargar_Historial import CargadorHistorial, ResultadoHistorial
from .Filtro_Estadistico.Filtro import FiltroEstadistico


class TickRequest(BaseModel):
    selected_entities: list[str] = Field(default_factory=list)
    tick_seconds: int = 60
    faults: list[dict[str, Any]] = Field(default_factory=list)
    suspicion_threshold: float = 0.6


class AnalyzeRequest(BaseModel):
    suspicion_threshold: float = 0.6


@dataclass
class EstadoAplicacion:
    cargador: CargadorHistorial
    filtro_estadistico: FiltroEstadistico
    static_history_dir: Path
    real_history_path: Path
    historial: ResultadoHistorial | None = None
    resultados: list[dict[str, Any]] = field(default_factory=list)
    realtime_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_tick_index: int = 0
    realtime_current_timestamp: str | None = None
    selected_entities: set[str] = field(default_factory=set)

    def status(self) -> dict[str, Any]:
        if self.historial is None:
            if (self.static_history_dir / "events.csv").exists():
                try:
                    self.historial = self.cargador.cargar_canonico(self.static_history_dir)
                except Exception:
                    pass

        if self.historial is None:
            return {
                "exists": False,
                "valid": False,
                "rows": 0,
                "entity_count": 0,
                "entities": [],
                "source_type": "",
                "start": None,
                "end": None,
                "normality_history": {"exists": False, "rows": 0},
            }
        return self._status_desde_historial(self.historial)

    def cargar_real(self) -> dict[str, Any]:
        self.historial = self.cargador.cargar_real(self.real_history_path)
        self._reset_realtime()
        return self._status_desde_historial(self.historial)

    def asegurar_canonico(self) -> dict[str, Any]:
        self.historial = self.cargador.crear_canonico_minimo(self.static_history_dir)
        self._reset_realtime()
        return self._status_desde_historial(self.historial)

    def analizar_historial(self, suspicion_threshold: float) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")

        resultado = self.filtro_estadistico.analizar_historial(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
            suspicion_threshold=suspicion_threshold,
        )
        self.resultados = list(resultado.get("results", []))
        return resultado

    def reset_realtime(self) -> dict[str, Any]:
        self._reset_realtime()
        return self.current_realtime({})

    def tick(self, request: TickRequest) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        entidades_validas = set(self.historial.entidades["entity_id"].astype(str))
        seleccion = [entity_id for entity_id in request.selected_entities if entity_id in entidades_validas]
        if not seleccion:
            raise ValueError("Selecciona al menos una entidad valida para generar un tick.")

        base = self.historial.eventos[self.historial.eventos["entity_id"].isin(seleccion)].copy()
        if base.empty:
            raise ValueError("No hay eventos historicos para las entidades seleccionadas.")
        siguiente = base.groupby("entity_id", as_index=False).tail(1).copy()
        ultimo = pd.to_datetime(self.historial.eventos["timestamp_dt"], errors="coerce", utc=True).max()
        if self.realtime_current_timestamp:
            ultimo = pd.to_datetime(self.realtime_current_timestamp, errors="coerce", utc=True)
        nuevo_timestamp = ultimo + pd.Timedelta(seconds=max(10, int(request.tick_seconds)))
        siguiente["timestamp_dt"] = nuevo_timestamp
        siguiente["timestamp"] = nuevo_timestamp.isoformat()
        self.realtime_tick_index += 1
        self.realtime_current_timestamp = nuevo_timestamp.isoformat()
        self.selected_entities = set(seleccion)
        self.realtime_events = pd.concat([self.realtime_events, siguiente], ignore_index=True)
        summary = self._summary_vacio(0.0)
        return self.current_realtime(summary)

    def current_realtime(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "current_timestamp": self.realtime_current_timestamp,
            "tick_index": self.realtime_tick_index,
            "events": int(len(self.realtime_events)),
            "ground_truth_rows": 0,
            "ground_truth_labels": {},
            "results": self.resultados,
            "historical_results": self.resultados,
            "result_count": len(self.resultados),
            "historical_result_count": len(self.resultados),
            "summary": summary or {},
            "historical_summary": summary or {},
            "faults": [],
            "fault_ticks": [],
        }

    def timeline(self, start: str | None, end: str | None, max_points: int) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            return {"entities": [], "faults": []}

        eventos = self.historial.eventos.copy()
        if not self.realtime_events.empty:
            eventos = pd.concat([eventos, self.realtime_events], ignore_index=True)
        if self.selected_entities:
            eventos = eventos[eventos["entity_id"].isin(self.selected_entities)]
        eventos = filtrar_rango(eventos, start, end)
        return construir_timeline(eventos, self.historial.entidades, max_points=max_points)

    def _status_desde_historial(self, historial: ResultadoHistorial) -> dict[str, Any]:
        metadatos = historial.metadatos
        return {
            "exists": True,
            "valid": True,
            "rows": int(metadatos.get("rows", len(historial.eventos))),
            "entity_count": int(metadatos.get("entity_count", len(historial.entidades))),
            "entities": historial.entidades.to_dict("records"),
            "source_type": metadatos.get("source_type", ""),
            "source_path": metadatos.get("source_path", ""),
            "start": metadatos.get("start"),
            "end": metadatos.get("end"),
            "normality_history": {"exists": False, "rows": 0},
        }

    def _summary_vacio(self, elapsed_ms: float) -> dict[str, Any]:
        return {
            "results": 0,
            "suspicious": 0,
            "by_kind": {},
            "top_results": [],
            "elapsed_ms": round(float(elapsed_ms), 3),
            "elapsed_seconds": round(float(elapsed_ms) / 1000.0, 6),
            "timings_by_kind_ms": {},
            "first_detection_delay_seconds": None,
        }

    def _reset_realtime(self) -> None:
        self.realtime_events = pd.DataFrame()
        self.realtime_tick_index = 0
        self.realtime_current_timestamp = None
        self.selected_entities = set()
        self.resultados = []


def crear_app() -> FastAPI:
    api = FastAPI(
        title="Artefacto de tesis",
        description="API para carga de historial y filtros del artefacto.",
        version="0.2.0",
    )
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
        if origin.strip()
    ]
    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return api


app = crear_app()
state = EstadoAplicacion(
    cargador=CargadorHistorial(),
    filtro_estadistico=FiltroEstadistico(),
    static_history_dir=Path(os.getenv("STATIC_HISTORY_DIR", "data/static_history")),
    real_history_path=Path(os.getenv("REAL_HISTORY_PATH", "data/datosreales/history.csv")),
)


@app.get("/health", tags=["health"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "mensaje": "API operativa",
        "static_history_dir": str(state.static_history_dir),
        "real_history_path": str(state.real_history_path),
    }


@app.get("/history/static/status", tags=["history"])
def history_static_status() -> dict[str, Any]:
    return state.status()


@app.post("/history/static/ensure", tags=["history"])
def history_static_ensure() -> dict[str, Any]:
    try:
        return state.asegurar_canonico()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/history/real/load", tags=["history"])
def history_real_load() -> dict[str, Any]:
    try:
        return state.cargar_real()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/history/analyze", tags=["history"])
def history_analyze(request: AnalyzeRequest) -> dict[str, Any]:
    try:
        return state.analizar_historial(request.suspicion_threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/realtime/reset", tags=["realtime"])
def realtime_reset() -> dict[str, Any]:
    return state.reset_realtime()


@app.post("/realtime/tick", tags=["realtime"])
def realtime_tick(request: TickRequest) -> dict[str, Any]:
    try:
        return state.tick(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/realtime/current", tags=["realtime"])
def realtime_current() -> dict[str, Any]:
    return state.current_realtime()


@app.get("/realtime/timeline", tags=["realtime"])
def realtime_timeline(
    start: str | None = None,
    end: str | None = None,
    max_points: int = Query(default=1000, ge=10, le=10000),
) -> dict[str, Any]:
    return state.timeline(start=start, end=end, max_points=max_points)


def filtrar_rango(eventos: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if eventos.empty:
        return eventos
    trabajo = eventos.copy()
    if start:
        start_dt = pd.to_datetime(start, errors="coerce", utc=True)
        if pd.notna(start_dt):
            trabajo = trabajo[trabajo["timestamp_dt"] >= start_dt]
    if end:
        end_dt = pd.to_datetime(end, errors="coerce", utc=True)
        if pd.notna(end_dt):
            trabajo = trabajo[trabajo["timestamp_dt"] <= end_dt]
    if not start and not end:
        limite = trabajo["timestamp_dt"].max() - pd.Timedelta(days=30)
        trabajo = trabajo[trabajo["timestamp_dt"] >= limite]
    return trabajo


def construir_timeline(eventos: pd.DataFrame, entidades: pd.DataFrame, max_points: int) -> dict[str, Any]:
    if eventos.empty:
        return {"entities": [], "faults": []}
    lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
    timelines = []
    for entity_id, grupo in eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
        entidad = lookup.get(str(entity_id), {})
        tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
        if tipo == "binary":
            points, bucket_seconds = timeline_binario(grupo, max_points=max_points)
            chart_kind = "active_percentage"
            scale_unit = "%"
        else:
            sampleado = reducir_puntos(grupo, max_points)
            points = [
                {
                    "timestamp": str(fila["timestamp"]),
                    "state": str(fila["state"]),
                    "value": valor_para_timeline(fila, tipo),
                }
                for fila in sampleado.to_dict("records")
            ]
            bucket_seconds = None
            chart_kind = "state_line"
            scale_unit = "state"
        timelines.append(
            {
                "entity_id": str(entity_id),
                "sensor_kind": tipo,
                "area_id": "",
                "chart_kind": chart_kind,
                "scale_unit": scale_unit,
                "bucket_seconds": bucket_seconds,
                "points": points,
                "categories": {},
            }
        )
    return {"entities": timelines, "faults": []}


def reducir_puntos(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step].head(max_points)


def timeline_binario(frame: pd.DataFrame, max_points: int) -> tuple[list[dict[str, Any]], int]:
    trabajo = frame.sort_values("timestamp_dt").copy()
    if trabajo.empty:
        return [], 60
    inicio = trabajo["timestamp_dt"].min()
    fin = trabajo["timestamp_dt"].max()
    duracion = max((fin - inicio).total_seconds(), 60.0)
    bucket_seconds = max(60, int(duracion / max(max_points, 1)))
    trabajo["bucket"] = ((trabajo["timestamp_dt"] - inicio).dt.total_seconds() // bucket_seconds).astype(int)
    points = []
    for _, grupo in trabajo.groupby("bucket", sort=True):
        activos = pd.to_numeric(grupo["value"], errors="coerce").fillna(0.0)
        porcentaje = float(activos.mean() * 100.0)
        points.append(
            {
                "timestamp": str(grupo["timestamp"].iloc[-1]),
                "state": str(grupo["state"].iloc[-1]),
                "value": round(porcentaje, 4),
                "activations": int(((activos.shift(1).fillna(0) == 0) & (activos == 1)).sum()),
            }
        )
    return points[:max_points], bucket_seconds


def valor_para_timeline(fila: dict[str, Any], tipo: str) -> float:
    if tipo == "continuous":
        valor = fila.get("value")
        if valor is not None and not pd.isna(valor):
            return float(valor)
        return float(pd.to_numeric(pd.Series([fila.get("state")]), errors="coerce").fillna(0).iloc[0])
    return 0.0
