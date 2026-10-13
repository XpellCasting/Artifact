from __future__ import annotations
from app.config import setting
import os
from typing import Any
import pandas as pd
DEFAULT_HISTORY_LOAD_SECONDS_PER_MIB = float(
    setting('HISTORY_LOAD_SECONDS_PER_MIB')
)
VALIDATION_REPLAY_MEMORY_RESULTS = max(
    1, int(setting('VALIDATION_REPLAY_MEMORY_RESULTS'))
)
VALIDATION_REPLAY_STATUS_RESULTS = max(
    1, int(setting('VALIDATION_REPLAY_STATUS_RESULTS'))
)
MIN_SUSPICION_THRESHOLD = 0.01
MAX_SUSPICION_THRESHOLD = 0.99
DEFAULT_SUSPICION_THRESHOLD = 0.60
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
        return {"entities": []}
    lookup = entidades.set_index("entity_id").to_dict("index") if not entidades.empty else {}
    timelines = []
    for entity_id, grupo in eventos.sort_values("timestamp_dt").groupby("entity_id", sort=True):
        entidad = lookup.get(str(entity_id), {})
        tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
        categories: dict[str, int] = {}
        analysis_status = "eligible"
        if tipo == "binary":
            points, bucket_seconds = timeline_binario(grupo, max_points=max_points)
            chart_kind = "active_percentage"
            scale_unit = "%"
            gaps = []
        elif tipo == "multistate":
            validos = grupo[
                grupo["is_valid_value"].astype(bool)
            ] if "is_valid_value" in grupo.columns else grupo
            estados = validos["state"].astype(str).str.strip()
            estados_unicos = estados.drop_duplicates().tolist()
            unique_ratio = len(estados_unicos) / max(len(estados), 1)
            if len(estados) < 24:
                analysis_status = "insufficient_history"
            elif len(estados_unicos) < 2:
                analysis_status = "insufficient_states"
            elif len(estados_unicos) > 32 or (
                len(estados_unicos) >= 8 and unique_ratio > 0.50
            ):
                analysis_status = "high_cardinality"
            if analysis_status != "high_cardinality":
                categories = {
                    estado: indice
                    for indice, estado in enumerate(estados_unicos)
                }
            sampleado = muestrear_timeline_con_cortes(grupo, max_points)
            points = [
                {
                    "timestamp": str(fila["timestamp"]),
                    "state": str(fila["state"]),
                    "value": (
                        categories.get(str(fila["state"]))
                        if bool(fila.get("is_valid_value", True))
                        else None
                    ),
                    "is_valid_value": bool(fila.get("is_valid_value", True)),
                }
                for fila in sampleado.to_dict("records")
            ]
            bucket_seconds = None
            chart_kind = "categorical_state"
            scale_unit = "estado"
            gaps = timeline_gaps_invalidos(grupo)
        else:
            sampleado = muestrear_timeline_con_cortes(grupo, max_points)
            points = [
                {
                    "timestamp": str(fila["timestamp"]),
                    "state": str(fila["state"]),
                    "value": valor_para_timeline(fila, tipo),
                    "is_valid_value": bool(fila.get("is_valid_value", True)),
                }
                for fila in sampleado.to_dict("records")
            ]
            bucket_seconds = None
            chart_kind = "state_line"
            scale_unit = "state"
            gaps = timeline_gaps_invalidos(grupo)
        timelines.append(
            {
                "entity_id": str(entity_id),
                "sensor_kind": tipo,
                "area_id": "",
                "chart_kind": chart_kind,
                "scale_unit": scale_unit,
                "bucket_seconds": bucket_seconds,
                "points": points,
                "gaps": gaps,
                "categories": categories,
                "analysis_status": analysis_status,
            }
        )
    return {"entities": timelines}
def reducir_puntos(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step].head(max_points)
def muestrear_timeline_con_cortes(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    trabajo = frame.sort_values("timestamp_dt").copy()
    validos = trabajo[trabajo.get("is_valid_value", True).astype(bool)] if "is_valid_value" in trabajo else trabajo
    base = reducir_puntos(validos, max_points)
    if "is_valid_value" not in trabajo.columns or trabajo["is_valid_value"].all():
        return base.sort_values("timestamp_dt")
    posiciones = set()
    for posicion, valido in enumerate(trabajo["is_valid_value"].astype(bool).tolist()):
        if not valido:
            posiciones.update([max(0, posicion - 1), posicion, min(len(trabajo) - 1, posicion + 1)])
    vecinos = trabajo.iloc[sorted(posiciones)]
    return pd.concat([base, vecinos]).drop_duplicates(["entity_id", "timestamp"]).sort_values("timestamp_dt")
def timeline_gaps_invalidos(frame: pd.DataFrame) -> list[dict[str, str]]:
    if "is_valid_value" not in frame.columns or frame.empty:
        return []
    gaps = []
    inicio = None
    for fila in frame.sort_values("timestamp_dt").to_dict("records"):
        valido = bool(fila.get("is_valid_value", True))
        if not valido and inicio is None:
            inicio = str(fila["timestamp"])
        elif valido and inicio is not None:
            gaps.append({"start": inicio, "end": str(fila["timestamp"]), "label": "sin lectura valida"})
            inicio = None
    return gaps
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
    if not bool(fila.get("is_valid_value", True)):
        return None
    if tipo == "continuous":
        valor = fila.get("value")
        if valor is not None and not pd.isna(valor):
            return float(valor)
        return float(pd.to_numeric(pd.Series([fila.get("state")]), errors="coerce").fillna(0).iloc[0])
    return 0.0
