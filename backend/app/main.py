from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .Cargar_Historial.Cargar_Historial import CargadorHistorial, ResultadoHistorial
from .Filtro_Estadistico.Filtro import FiltroEstadistico
from .Patrones_Diseno.Unidad_Trabajo_Iteraciones import UnidadTrabajoIteraciones
from .simulacion.Inyeccion_Fallas import InyectorFallasSensornets


class TickRequest(BaseModel):
    selected_entities: list[str] = Field(default_factory=list)
    tick_seconds: int = 60
    faults: list[dict[str, Any]] = Field(default_factory=list)
    suspicion_threshold: float = 0.6


class IterationsRequest(TickRequest):
    iteration_count: int = 1


class AnalyzeRequest(BaseModel):
    suspicion_threshold: float = 0.6


@dataclass
class EstadoAplicacion:
    cargador: CargadorHistorial
    filtro_estadistico: FiltroEstadistico
    static_history_dir: Path
    real_history_path: Path
    analysis_cache_path: Path
    inyector_fallas: InyectorFallasSensornets = field(default_factory=InyectorFallasSensornets)
    historial: ResultadoHistorial | None = None
    resultados: list[dict[str, Any]] = field(default_factory=list)
    realtime_events: pd.DataFrame = field(default_factory=pd.DataFrame)
    realtime_tick_index: int = 0
    realtime_current_timestamp: str | None = None
    realtime_faults: list[dict[str, Any]] = field(default_factory=list)
    realtime_fault_ticks: list[dict[str, Any]] = field(default_factory=list)
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

        cached = self._leer_cache_analisis_real(suspicion_threshold)
        if cached is not None:
            self.resultados = list(cached.get("results", []))
            return cached

        resultado = self.filtro_estadistico.analizar_historial(
            eventos=self.historial.eventos,
            entidades=self.historial.entidades,
            suspicion_threshold=suspicion_threshold,
        )
        summary = dict(resultado.get("summary") or {})
        summary["cache_hit"] = False
        summary["cache_path"] = str(self.analysis_cache_path) if self._es_historial_real_activo() else ""
        resultado["summary"] = summary
        self.resultados = list(resultado.get("results", []))
        self._guardar_cache_analisis_real(suspicion_threshold, resultado)
        return resultado

    def vista_normalizacion(self) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        return {
            "source_type": self.historial.metadatos.get("source_type", ""),
            "source_path": self.historial.metadatos.get("source_path", ""),
            "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
            "entity_count": int(self.historial.metadatos.get("entity_count", len(self.historial.entidades))),
            "samples_per_entity": 5,
            "samples": self.historial.normalizacion_muestras,
            "fragments": self._fragmentos_normalizacion(),
        }

    def _fragmentos_normalizacion(self, max_points_per_entity: int = 80) -> list[dict[str, Any]]:
        if self.historial is None or self.historial.eventos.empty:
            return []
        lookup = self.historial.entidades.set_index("entity_id").to_dict("index") if not self.historial.entidades.empty else {}
        fragments = []
        for entity_id, grupo in self.historial.eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
            entidad = lookup.get(str(entity_id), {})
            tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
            sampleado = reducir_puntos(grupo.head(max_points_per_entity * 3), max_points_per_entity)
            points = []
            for fila in sampleado.to_dict("records"):
                if tipo == "binary":
                    value = pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0]
                    value = 100.0 if float(value) >= 0.5 else 0.0
                else:
                    value = valor_para_timeline(fila, tipo)
                points.append(
                    {
                        "timestamp": str(fila["timestamp"]),
                        "state": str(fila["state"]),
                        "value": value,
                    }
                )
            fragments.append(
                {
                    "entity_id": str(entity_id),
                    "sensor_kind": tipo,
                    "area_id": "",
                    "chart_kind": "normalization_fragment",
                    "scale_unit": "%" if tipo == "binary" else str(entidad.get("unit_of_measurement") or "state"),
                    "points": points,
                }
            )
        return fragments

    def reset_realtime(self) -> dict[str, Any]:
        self._reset_realtime()
        return self.current_realtime({})

    def tick(self, request: TickRequest) -> dict[str, Any]:
        self._insertar_tick_realtime(request)
        summary = self._analizar_realtime(request.suspicion_threshold)
        return self.current_realtime(summary)

    def iterations(self, request: IterationsRequest) -> dict[str, Any]:
        unidad = UnidadTrabajoIteraciones(request.iteration_count)
        resultado_unidad = unidad.ejecutar(lambda: self._insertar_tick_realtime(request))
        summary = self._analizar_realtime(request.suspicion_threshold)
        summary["iterations_requested"] = resultado_unidad.solicitadas
        summary["iterations_done"] = resultado_unidad.completadas
        return self.current_realtime(summary)

    def _insertar_tick_realtime(self, request: TickRequest) -> None:
        if self.historial is None:
            self.status()
        if self.historial is None:
            raise ValueError("Primero debes cargar un historial.")
        entidades_validas = set(self.historial.entidades["entity_id"].astype(str))
        seleccion = [entity_id for entity_id in request.selected_entities if entity_id in entidades_validas]
        if not seleccion:
            raise ValueError("Selecciona al menos una entidad valida para generar un tick.")

        eventos_actuales = self._eventos_combinados()
        base = eventos_actuales[eventos_actuales["entity_id"].isin(seleccion)].copy()
        if base.empty:
            raise ValueError("No hay eventos historicos para las entidades seleccionadas.")
        ultimo = pd.to_datetime(eventos_actuales["timestamp_dt"], errors="coerce", utc=True).max()
        if self.realtime_current_timestamp:
            ultimo = pd.to_datetime(self.realtime_current_timestamp, errors="coerce", utc=True)
        tick_seconds = max(10, int(request.tick_seconds))
        nuevo_timestamp = ultimo + pd.Timedelta(seconds=tick_seconds)
        self.realtime_tick_index += 1
        self.selected_entities = set(seleccion)
        siguiente = self._generar_tick_simulado(
            eventos=eventos_actuales,
            seleccion=seleccion,
            timestamp=nuevo_timestamp,
            tick_seconds=tick_seconds,
            faults=request.faults,
        )
        if not siguiente.empty:
            self.realtime_events = pd.concat([self.realtime_events, siguiente], ignore_index=True)
            timestamp_real = pd.to_datetime(siguiente["timestamp_dt"], errors="coerce", utc=True).max()
            if pd.notna(timestamp_real):
                nuevo_timestamp = max(nuevo_timestamp, timestamp_real)
        self.realtime_current_timestamp = nuevo_timestamp.isoformat()

    def _analizar_realtime(self, suspicion_threshold: float) -> dict[str, Any]:
        inicio = perf_counter()
        resultado = self.filtro_estadistico.analizar_historial(
            eventos=self._eventos_combinados(),
            entidades=self.historial.entidades,
            suspicion_threshold=suspicion_threshold,
        )
        summary = dict(resultado.get("summary") or {})
        summary["elapsed_ms"] = round((perf_counter() - inicio) * 1000.0, 3)
        summary["realtime_events"] = int(len(self.realtime_events))
        summary["active_faults"] = len(self.realtime_faults)
        self.resultados = list(resultado.get("results", []))
        return summary

    def current_realtime(self, summary: dict[str, Any] | None = None) -> dict[str, Any]:
        realtime_results = self._resultados_realtime()
        return {
            "current_timestamp": self.realtime_current_timestamp,
            "tick_index": self.realtime_tick_index,
            "events": int(len(self.realtime_events)),
            "ground_truth_rows": 0,
            "ground_truth_labels": {},
            "results": realtime_results,
            "historical_results": self.resultados,
            "result_count": len(realtime_results),
            "historical_result_count": len(self.resultados),
            "summary": summary or {},
            "historical_summary": summary or {},
            "faults": self.realtime_faults,
            "fault_ticks": self.realtime_fault_ticks,
        }

    def timeline(self, start: str | None, end: str | None, max_points: int) -> dict[str, Any]:
        if self.historial is None:
            self.status()
        if self.historial is None:
            return {"entities": [], "faults": []}

        eventos = self._eventos_combinados()
        if self.selected_entities:
            eventos = eventos[eventos["entity_id"].isin(self.selected_entities)]
        eventos = filtrar_rango(eventos, start, end)
        payload = construir_timeline(eventos, self.historial.entidades, max_points=max_points)
        payload["faults"] = self.realtime_faults
        return payload

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
        self.realtime_faults = []
        self.realtime_fault_ticks = []
        self.selected_entities = set()
        self.resultados = []

    def _eventos_combinados(self) -> pd.DataFrame:
        if self.historial is None:
            return pd.DataFrame()
        if self.realtime_events.empty:
            return self.historial.eventos.copy()
        return pd.concat([self.historial.eventos, self.realtime_events], ignore_index=True)

    def _resultados_realtime(self) -> list[dict[str, Any]]:
        if self.realtime_events.empty:
            return []
        inicio = pd.to_datetime(self.realtime_events["timestamp_dt"], errors="coerce", utc=True).min()
        if pd.isna(inicio):
            return []
        resultados = []
        for resultado in self.resultados:
            timestamp = pd.to_datetime(resultado.get("timestamp"), errors="coerce", utc=True)
            if pd.notna(timestamp) and timestamp >= inicio:
                resultados.append(resultado)
        return resultados

    def _generar_tick_simulado(
        self,
        eventos: pd.DataFrame,
        seleccion: list[str],
        timestamp: pd.Timestamp,
        tick_seconds: int,
        faults: list[dict[str, Any]],
    ) -> pd.DataFrame:
        entidades = self.historial.entidades.set_index("entity_id").to_dict("index") if self.historial is not None else {}
        fallas = {
            str(fault.get("entity_id")): fault
            for fault in faults
            if fault.get("enabled") and str(fault.get("entity_id")) in seleccion
        }
        filas: list[dict[str, Any]] = []
        source_timestamp = self._timestamp_fuente_simulacion(tick_seconds)
        for entity_id in seleccion:
            grupo = eventos[eventos["entity_id"].eq(entity_id)].sort_values("timestamp_dt")
            if grupo.empty:
                continue
            entidad = entidades.get(entity_id, {})
            fila = self._proyectar_siguiente_fila(grupo, entidad, timestamp, source_timestamp)
            fault = fallas.get(entity_id)
            if fault:
                fault_context = {
                    **fault,
                    "fault_tick_count": self._conteo_falla_entidad(entity_id, str(fault.get("fault_type", "fault"))) + 1,
                }
                resultado = self.inyector_fallas.aplicar(
                    fila=fila,
                    grupo=self._grupo_historico_entidad(entity_id, grupo),
                    entidad=entidad,
                    fault=fault_context,
                    tick_index=self.realtime_tick_index,
                    tick_seconds=tick_seconds,
                )
                fila = resultado.fila
                if resultado.aplicada:
                    self._registrar_falla_tick(
                        entity_id=entity_id,
                        fault_type=resultado.fault_type,
                        timestamp=fila["timestamp_dt"],
                        tick_seconds=tick_seconds,
                        clean_state=resultado.clean_state,
                        observed_state=resultado.observed_state,
                        metadata=resultado.metadata,
                    )
            filas.append(fila)
        if not filas:
            return pd.DataFrame(columns=eventos.columns)
        return pd.DataFrame(filas).reindex(columns=eventos.columns)

    def _proyectar_siguiente_fila(
        self,
        grupo: pd.DataFrame,
        entidad: dict[str, Any],
        timestamp: pd.Timestamp,
        source_timestamp: pd.Timestamp,
    ) -> dict[str, Any]:
        ultima = grupo.tail(1).iloc[0].to_dict()
        fila = dict(ultima)
        fila["timestamp_dt"] = timestamp
        fila["timestamp"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        tipo = str(entidad.get("sensor_kind") or ultima.get("sensor_kind") or "")
        historico = self._grupo_historico_entidad(str(ultima.get("entity_id")), grupo)
        fuente = self._fila_historica_cercana(historico, source_timestamp)
        if tipo == "continuous":
            valor = pd.to_numeric(pd.Series([fuente.get("value")]), errors="coerce").iloc[0]
            if pd.isna(valor):
                valor = pd.to_numeric(pd.Series([ultima.get("value")]), errors="coerce").fillna(0.0).iloc[0]
            valor = float(valor)
            fila["value"] = valor
            fila["state"] = f"{valor:.6f}"
        elif tipo == "binary":
            valor = pd.to_numeric(pd.Series([fuente.get("value")]), errors="coerce").fillna(0.0).iloc[0]
            fila["value"] = 1.0 if valor >= 0.5 else 0.0
            fila["state"] = self._estado_binario(fila["value"])
        else:
            fila["state"] = str(fuente.get("state", ultima.get("state", "")))
            fila["value"] = fuente.get("value")
        return fila

    def _registrar_falla_tick(
        self,
        entity_id: str,
        fault_type: str,
        timestamp: pd.Timestamp,
        tick_seconds: int,
        clean_state: str,
        observed_state: str,
        metadata: dict[str, Any],
    ) -> None:
        start = pd.to_datetime(timestamp, errors="coerce", utc=True)
        end = start + pd.Timedelta(seconds=max(10, tick_seconds))
        evento = {
            "entity_id": entity_id,
            "fault_type": fault_type,
            "label": "sensor_fault",
            "is_reliable": True,
            "clean_state": clean_state,
            "observed_state": observed_state,
            "metadata": metadata,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "tick_index": self.realtime_tick_index,
        }
        self.realtime_fault_ticks.append(evento)
        ultima = self.realtime_faults[-1] if self.realtime_faults else None
        if ultima and ultima["entity_id"] == entity_id and ultima["fault_type"] == fault_type:
            ultima["end"] = evento["end"]
            ultima["observed_state"] = observed_state
        else:
            self.realtime_faults.append(
                {k: evento[k] for k in ("entity_id", "fault_type", "label", "start", "end", "clean_state", "observed_state")}
            )

    def _escala_continua(self, grupo: pd.DataFrame) -> float:
        valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
        if valores.empty:
            return 1.0
        iqr = float(valores.quantile(0.75) - valores.quantile(0.25))
        std = float(valores.std()) if len(valores) > 1 else 0.0
        return max(abs(iqr), abs(std), abs(float(valores.median())) * 0.05, 1.0)

    def _grupo_historico_entidad(self, entity_id: str, fallback: pd.DataFrame) -> pd.DataFrame:
        if self.historial is None or self.historial.eventos.empty or not entity_id:
            return fallback
        grupo = self.historial.eventos[self.historial.eventos["entity_id"].eq(entity_id)]
        return grupo if not grupo.empty else fallback

    def _timestamp_fuente_simulacion(self, tick_seconds: int) -> pd.Timestamp:
        if self.historial is None or self.historial.eventos.empty:
            return pd.Timestamp.now(tz="UTC")
        tiempos = pd.to_datetime(self.historial.eventos["timestamp_dt"], errors="coerce", utc=True).dropna()
        if tiempos.empty:
            return pd.Timestamp.now(tz="UTC")
        inicio = tiempos.min()
        fin = tiempos.max()
        duracion = max((fin - inicio).total_seconds(), float(max(10, tick_seconds)))
        avance = ((max(1, self.realtime_tick_index) - 1) * max(10, int(tick_seconds))) % duracion
        return inicio + pd.Timedelta(seconds=avance)

    def _fila_historica_cercana(self, grupo: pd.DataFrame, source_timestamp: pd.Timestamp) -> dict[str, Any]:
        if grupo.empty:
            return {}
        trabajo = grupo.copy()
        tiempos = pd.to_datetime(trabajo["timestamp_dt"], errors="coerce", utc=True)
        if tiempos.notna().any():
            distancia = (tiempos - source_timestamp).abs()
            indice = distancia.idxmin()
            return trabajo.loc[indice].to_dict()
        return trabajo.tail(1).iloc[0].to_dict()

    def _valor_continuo_normal(self, grupo: pd.DataFrame, timestamp: pd.Timestamp) -> float:
        valores = pd.to_numeric(grupo["value"], errors="coerce")
        validos = grupo[valores.notna()].copy()
        if validos.empty:
            return 0.0
        validos["value_numeric"] = valores[valores.notna()].astype(float)
        ventana = self._ventana_por_hora_del_dia(validos, timestamp)
        if ventana.empty:
            ventana = validos.tail(min(48, len(validos)))
        return round(float(ventana["value_numeric"].median()), 6)

    def _valor_binario_normal(self, grupo: pd.DataFrame, timestamp: pd.Timestamp) -> float:
        valores = pd.to_numeric(grupo["value"], errors="coerce")
        validos = grupo[valores.notna()].copy()
        if validos.empty:
            return 0.0
        validos["value_numeric"] = valores[valores.notna()].astype(float)
        ventana = self._ventana_por_hora_del_dia(validos, timestamp)
        if ventana.empty:
            ventana = validos.tail(min(48, len(validos)))
        return 1.0 if float(ventana["value_numeric"].mean()) >= 0.5 else 0.0

    def _ventana_por_hora_del_dia(self, grupo: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
        tiempos = pd.to_datetime(grupo["timestamp_dt"], errors="coerce", utc=True)
        objetivo = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        segundos = tiempos.dt.hour * 3600 + tiempos.dt.minute * 60 + tiempos.dt.second
        distancia = (segundos - objetivo).abs()
        distancia = pd.concat([distancia, 86400 - distancia], axis=1).min(axis=1)
        for ventana_segundos in (30 * 60, 2 * 3600, 6 * 3600):
            candidatos = grupo[distancia <= ventana_segundos]
            if not candidatos.empty:
                return candidatos
        return grupo

    def _conteo_falla_entidad(self, entity_id: str, fault_type: str) -> int:
        return sum(
            1
            for evento in self.realtime_fault_ticks
            if evento.get("entity_id") == entity_id and evento.get("fault_type") == fault_type
        )

    def _estado_binario(self, value: float) -> str:
        return "on" if float(value) >= 0.5 else "off"

    def _es_historial_real_activo(self) -> bool:
        return self.historial is not None and self.historial.metadatos.get("source_type") == "real"

    def _identidad_historial_real(self, suspicion_threshold: float) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        stat = self.real_history_path.stat()
        return {
            "cache_version": 1,
            "source_path": str(self.real_history_path),
            "source_size": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "suspicion_threshold": round(float(suspicion_threshold), 6),
        }

    def _leer_cache_analisis_real(self, suspicion_threshold: float) -> dict[str, Any] | None:
        identidad = self._identidad_historial_real(suspicion_threshold)
        if identidad is None or not self.analysis_cache_path.exists():
            return None
        try:
            with self.analysis_cache_path.open("r", encoding="utf-8") as archivo:
                cache = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return None
        if cache.get("identity") != identidad:
            return None
        payload = cache.get("payload")
        if not isinstance(payload, dict):
            return None
        payload = dict(payload)
        payload["cache"] = {"hit": True, "path": str(self.analysis_cache_path)}
        summary = dict(payload.get("summary") or {})
        summary["cache_hit"] = True
        summary["cache_path"] = str(self.analysis_cache_path)
        payload["summary"] = summary
        return payload

    def _guardar_cache_analisis_real(self, suspicion_threshold: float, resultado: dict[str, Any]) -> None:
        identidad = self._identidad_historial_real(suspicion_threshold)
        if identidad is None:
            return
        payload = serializable_json(resultado)
        payload["cache"] = {"hit": False, "path": str(self.analysis_cache_path)}
        try:
            self.analysis_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.analysis_cache_path.open("w", encoding="utf-8") as archivo:
                json.dump({"identity": identidad, "payload": payload}, archivo, ensure_ascii=True)
        except OSError:
            return


def serializable_json(valor: Any) -> Any:
    if isinstance(valor, dict):
        return {str(clave): serializable_json(contenido) for clave, contenido in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [serializable_json(item) for item in valor]
    if isinstance(valor, pd.Timestamp):
        return None if pd.isna(valor) else valor.isoformat()
    if valor is None or isinstance(valor, (str, int, float, bool)):
        return valor
    if hasattr(valor, "item"):
        try:
            return serializable_json(valor.item())
        except (TypeError, ValueError):
            pass
    if hasattr(valor, "isoformat"):
        try:
            return valor.isoformat()
        except (TypeError, ValueError):
            pass
    return str(valor)


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
    analysis_cache_path=Path(os.getenv("ANALYSIS_CACHE_PATH", "data/cache/real_history_analysis.json")),
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


@app.get("/history/normalization/preview", tags=["history"])
def history_normalization_preview() -> dict[str, Any]:
    try:
        return state.vista_normalizacion()
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


@app.post("/realtime/iterations", tags=["realtime"])
def realtime_iterations(request: IterationsRequest) -> dict[str, Any]:
    try:
        return state.iterations(request)
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
