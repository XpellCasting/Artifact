from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


ESTADOS_INVALIDOS = {"", "unknown", "unavailable", "none", "null", "nan"}
ESTADOS_ACTIVOS = {"on", "open", "home", "active", "detected", "true", "1"}
ESTADOS_INACTIVOS = {"off", "closed", "not_home", "inactive", "clear", "false", "0"}


@dataclass
class ResultadoHistorial:
    """Resultado en memoria despues de cargar un historial."""

    eventos: pd.DataFrame
    entidades: pd.DataFrame
    metadatos: dict[str, Any] = field(default_factory=dict)


class CargadorHistorial:
    """Carga historiales reales o canonicos y los deja en un formato comun."""

    columnas_minimas_reales = {"entity_id", "state", "last_changed"}
    columnas_minimas_canonicas = {"entity_id", "timestamp", "state"}

    def cargar_real(self, ruta: Path) -> ResultadoHistorial:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el historial real: {ruta}")
        frame = pd.read_csv(ruta, dtype=str, keep_default_na=False).fillna("")
        faltantes = sorted(self.columnas_minimas_reales - set(frame.columns))
        if faltantes:
            raise ValueError(f"El historial real no contiene columnas requeridas: {', '.join(faltantes)}")
        frame = frame.rename(columns={"last_changed": "timestamp"})
        return self._normalizar(frame, origen="real", ruta=ruta)

    def cargar_canonico(self, directorio: Path) -> ResultadoHistorial:
        eventos_path = directorio / "events.csv"
        entidades_path = directorio / "entities.csv"
        metadata_path = directorio / "metadata.csv"
        if not eventos_path.exists():
            raise FileNotFoundError(f"No se encontro events.csv en: {directorio}")
        frame = pd.read_csv(eventos_path, dtype=str, keep_default_na=False).fillna("")
        faltantes = sorted(self.columnas_minimas_canonicas - set(frame.columns))
        if faltantes:
            raise ValueError(f"El historial canonico no contiene columnas requeridas: {', '.join(faltantes)}")

        resultado = self._normalizar(frame, origen="canonico", ruta=eventos_path)
        if entidades_path.exists():
            entidades = pd.read_csv(entidades_path, dtype=str, keep_default_na=False).fillna("")
            if "entity_id" in entidades.columns:
                resultado.entidades = self._mezclar_entidades(resultado.entidades, entidades)
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False).fillna("")
            if not metadata.empty:
                resultado.metadatos.update(metadata.iloc[0].to_dict())
        return resultado

    def crear_canonico_minimo(self, directorio: Path, dias: int = 60) -> ResultadoHistorial:
        """Crea un historial canonico simple si no existe.

        Este generador es solo el punto de partida de carga. No inyecta fallas y
        no reemplaza el simulador estadistico que se implementara despues.
        """
        directorio.mkdir(parents=True, exist_ok=True)
        eventos_path = directorio / "events.csv"
        entidades_path = directorio / "entities.csv"
        metadata_path = directorio / "metadata.csv"
        if eventos_path.exists() and entidades_path.exists():
            return self.cargar_canonico(directorio)

        inicio = pd.Timestamp("2026-01-01T00:00:00Z")
        timestamps = pd.date_range(inicio, periods=dias * 24, freq="1h")
        catalogo = [
            {
                "entity_id": "sensor.living_temperature",
                "unit_of_measurement": "C",
                "device_class": "temperature",
                "domain": "sensor",
                "sensor_kind": "continuous",
                "tipo_interno": "continua",
            },
            {
                "entity_id": "sensor.living_humidity",
                "unit_of_measurement": "%",
                "device_class": "humidity",
                "domain": "sensor",
                "sensor_kind": "continuous",
                "tipo_interno": "continua",
            },
            {
                "entity_id": "binary_sensor.living_motion",
                "unit_of_measurement": "",
                "device_class": "motion",
                "domain": "binary_sensor",
                "sensor_kind": "binary",
                "tipo_interno": "estado_binario",
            },
            {
                "entity_id": "binary_sensor.main_door_contact",
                "unit_of_measurement": "",
                "device_class": "door",
                "domain": "binary_sensor",
                "sensor_kind": "binary",
                "tipo_interno": "estado_binario",
            },
            {
                "entity_id": "select.home_mode",
                "unit_of_measurement": "",
                "device_class": "mode",
                "domain": "select",
                "sensor_kind": "multistate",
                "tipo_interno": "estado_multivalor",
            },
        ]
        eventos = []
        for indice, timestamp in enumerate(timestamps):
            hora = timestamp.hour
            eventos.extend(
                [
                    self._evento(catalogo[0], timestamp, f"{20 + 2.5 * (hora >= 12) - 1.5 * (hora < 7):.2f}"),
                    self._evento(catalogo[1], timestamp, f"{58 + 8 * (hora < 8):.2f}"),
                    self._evento(catalogo[2], timestamp, "on" if 18 <= hora <= 22 and indice % 3 == 0 else "off"),
                    self._evento(catalogo[3], timestamp, "open" if hora in {8, 19} else "closed"),
                    self._evento(catalogo[4], timestamp, self._modo_hogar(hora)),
                ]
            )

        entidades = pd.DataFrame(catalogo)
        eventos_frame = pd.DataFrame(eventos)
        metadata = pd.DataFrame(
            [
                {
                    "seed": "canonico-minimo",
                    "start_timestamp": timestamps.min().isoformat(),
                    "end_timestamp": timestamps.max().isoformat(),
                    "duration_days": str(dias),
                    "created_by": "CargadorHistorial",
                }
            ]
        )
        entidades.to_csv(entidades_path, index=False)
        eventos_frame.to_csv(eventos_path, index=False)
        metadata.to_csv(metadata_path, index=False)
        return self.cargar_canonico(directorio)

    def _normalizar(self, frame: pd.DataFrame, origen: str, ruta: Path) -> ResultadoHistorial:
        trabajo = frame.copy()
        trabajo["entity_id"] = trabajo["entity_id"].astype(str).str.strip()
        trabajo["state"] = trabajo["state"].astype(str).str.strip()
        trabajo["timestamp_dt"] = pd.to_datetime(trabajo["timestamp"], errors="coerce", utc=True)
        trabajo = trabajo[
            trabajo["entity_id"].ne("")
            & trabajo["timestamp_dt"].notna()
            & ~trabajo["state"].str.lower().isin(ESTADOS_INVALIDOS)
        ].copy()
        trabajo["timestamp"] = trabajo["timestamp_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        trabajo["domain"] = trabajo.get("domain", trabajo["entity_id"].str.split(".").str[0])
        trabajo["sensor_kind"] = trabajo.apply(self._clasificar_fila, axis=1)
        trabajo["tipo_interno"] = trabajo["sensor_kind"].map(
            {
                "continuous": "continua",
                "binary": "estado_binario",
                "multistate": "estado_multivalor",
            }
        )
        trabajo["value"] = trabajo.apply(self._valor_numerico, axis=1)
        trabajo["unit_of_measurement"] = trabajo.get("unit_of_measurement", "")
        trabajo["device_class"] = trabajo.apply(self._device_class, axis=1)
        trabajo = trabajo.sort_values(["entity_id", "timestamp_dt"]).reset_index(drop=True)

        entidades = self._construir_entidades(trabajo)
        metadatos = {
            "source_type": origen,
            "source_path": str(ruta),
            "rows": int(len(trabajo)),
            "entity_count": int(trabajo["entity_id"].nunique()) if not trabajo.empty else 0,
            "start": None if trabajo.empty else trabajo["timestamp_dt"].min().isoformat(),
            "end": None if trabajo.empty else trabajo["timestamp_dt"].max().isoformat(),
        }
        return ResultadoHistorial(eventos=trabajo, entidades=entidades, metadatos=metadatos)

    def _clasificar_fila(self, fila: pd.Series) -> str:
        if str(fila.get("sensor_kind", "")).strip() in {"continuous", "binary", "multistate"}:
            return str(fila["sensor_kind"]).strip()
        estado = str(fila.get("state", "")).strip().lower()
        entity_id = str(fila.get("entity_id", "")).lower()
        dominio = str(fila.get("domain", "")).lower() or entity_id.split(".")[0]
        if dominio == "binary_sensor":
            return "binary"
        if dominio in {"select", "input_select", "climate"}:
            return "multistate"
        if pd.to_numeric(pd.Series([estado]), errors="coerce").notna().iloc[0]:
            return "continuous"
        if estado in ESTADOS_ACTIVOS or estado in ESTADOS_INACTIVOS:
            return "binary"
        return "multistate"

    def _valor_numerico(self, fila: pd.Series) -> float | None:
        tipo = str(fila["sensor_kind"])
        estado = str(fila["state"]).lower()
        if tipo == "continuous":
            valor = pd.to_numeric(pd.Series([estado]), errors="coerce").iloc[0]
            return None if pd.isna(valor) else float(valor)
        if tipo == "binary":
            return 1.0 if estado in ESTADOS_ACTIVOS else 0.0
        return None

    def _device_class(self, fila: pd.Series) -> str:
        if str(fila.get("device_class", "")).strip():
            return str(fila["device_class"]).strip()
        entity_id = str(fila.get("entity_id", "")).lower()
        if "humidity" in entity_id:
            return "humidity"
        if "temperature" in entity_id or "temp" in entity_id:
            return "temperature"
        if "motion" in entity_id or "pir" in entity_id:
            return "motion"
        if "door" in entity_id or "contact" in entity_id or "window" in entity_id:
            return "door"
        return ""

    def _construir_entidades(self, eventos: pd.DataFrame) -> pd.DataFrame:
        columnas = ["entity_id", "unit_of_measurement", "device_class", "domain", "tipo_interno", "sensor_kind"]
        if eventos.empty:
            return pd.DataFrame(columns=columnas)
        entidades = eventos.groupby("entity_id", as_index=False).first()[columnas]
        return entidades.sort_values("entity_id").reset_index(drop=True)

    def _mezclar_entidades(self, inferidas: pd.DataFrame, declaradas: pd.DataFrame) -> pd.DataFrame:
        base = inferidas.set_index("entity_id")
        declaradas = declaradas.set_index("entity_id")
        for columna in declaradas.columns:
            if columna not in base.columns:
                base[columna] = ""
            base.loc[base.index.intersection(declaradas.index), columna] = declaradas.loc[
                base.index.intersection(declaradas.index), columna
            ]
        return base.reset_index()

    def _evento(self, entidad: dict[str, str], timestamp: pd.Timestamp, state: str) -> dict[str, str]:
        return {
            **entidad,
            "timestamp": timestamp.isoformat(),
            "state": state,
        }

    def _modo_hogar(self, hora: int) -> str:
        if 0 <= hora < 7:
            return "sleep"
        if 8 <= hora < 17:
            return "away"
        return "home"
