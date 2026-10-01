from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from app.services.tipos_entidad import infer_kind
from app.services.tipos_entidad import KINDS
from app.services.estados_validos import valid_state


ESTADOS_INVALIDOS = {"", "unknown", "unavailable", "none", "null", "nan"}
ESTADOS_ACTIVOS = {"on", "open", "home", "active", "detected", "true", "1"}
ESTADOS_INACTIVOS = {"off", "closed", "not_home", "inactive", "clear", "false", "0"}
ProgressCallback = Callable[[str, float, str], None]


@dataclass
class ResultadoHistorial:
    """Resultado en memoria despues de cargar un historial."""

    eventos: pd.DataFrame
    entidades: pd.DataFrame
    metadatos: dict[str, Any] = field(default_factory=dict)
    normalizacion_muestras: list[dict[str, Any]] = field(default_factory=list)


class CargadorHistorial:
    """Carga historiales reales o canonicos y los deja en un formato comun."""

    columnas_minimas_reales = {"entity_id", "state", "last_changed"}
    columnas_minimas_canonicas = {"entity_id", "timestamp", "state"}

    def cargar_real(
        self,
        ruta: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> ResultadoHistorial:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el historial real: {ruta}")
        self._reportar_progreso(
            progress_callback,
            "reading_csv",
            5.0,
            f"Leyendo {ruta.name}",
        )
        frame = pd.read_csv(ruta, dtype=str, keep_default_na=False).fillna("")
        self._reportar_progreso(
            progress_callback,
            "validating_columns",
            22.0,
            f"{len(frame):,} filas leidas; validando columnas",
        )
        faltantes = sorted(self.columnas_minimas_reales - set(frame.columns))
        if faltantes:
            raise ValueError(f"El historial real no contiene columnas requeridas: {', '.join(faltantes)}")
        frame = frame.rename(columns={"last_changed": "timestamp"})
        return self._normalizar(
            frame,
            origen="real",
            ruta=ruta,
            progress_callback=progress_callback,
        )

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
        """Crea un historial canonico simple si no existe."""
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

    def _normalizar(
        self,
        frame: pd.DataFrame,
        origen: str,
        ruta: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> ResultadoHistorial:
        self._reportar_progreso(
            progress_callback,
            "normalizing",
            28.0,
            "Preparando columnas de trabajo",
        )
        trabajo = frame.copy()
        trabajo["_raw_index"] = range(len(trabajo))
        trabajo["entity_id"] = trabajo["entity_id"].astype(str).str.strip()
        trabajo["state"] = trabajo["state"].astype(str).str.strip()
        trabajo["timestamp_dt"] = pd.to_datetime(
            trabajo["timestamp"], format="ISO8601", errors="coerce", utc=True
        )
        trabajo = trabajo[
            trabajo["entity_id"].ne("")
            & trabajo["timestamp_dt"].notna()
        ].copy()
        self._reportar_progreso(
            progress_callback,
            "normalizing_timestamps",
            42.0,
            f"{len(trabajo):,} filas con entidad y fecha validas",
        )
        trabajo["is_valid_value"] = trabajo["state"].map(valid_state)
        # La fecha serializada tambien es una clave y un corte causal: no truncar.
        trabajo["timestamp"] = trabajo["timestamp_dt"].map(
            lambda value: value.isoformat().replace("+00:00", "Z")
        )
        if "domain" in trabajo.columns:
            trabajo["domain"] = trabajo["domain"].astype(str).str.strip()
        else:
            trabajo["domain"] = trabajo["entity_id"].str.split(".").str[0]
        dominio = trabajo["domain"].str.lower()
        dominio = dominio.mask(dominio.eq(""), trabajo["entity_id"].str.split(".").str[0].str.lower())
        estado = trabajo["state"].str.lower()
        valores_numericos = pd.to_numeric(estado, errors="coerce")
        tipo_por_entidad = {}
        for entity_id, grupo in trabajo.groupby("entity_id", sort=False):
            def metadata(column):
                if column not in grupo:
                    return ""
                values = grupo[column].astype(str).str.strip()
                values = values[values.ne("")]
                return values.iloc[0] if len(values) else ""

            declarations = set(grupo["sensor_kind"].astype(str).str.strip().str.lower()) & KINDS if "sensor_kind" in grupo else set()
            if len(declarations) > 1:
                raise ValueError(f"Tipos declarados incompatibles para {entity_id}: {sorted(declarations)}")
            valid_states = grupo.loc[grupo["is_valid_value"], "state"].unique()
            tipo_por_entidad[entity_id] = infer_kind(
                str(entity_id), valid_states, declared=next(iter(declarations), ""),
                domain=metadata("domain"), unit=metadata("unit_of_measurement"),
                device_class=metadata("device_class"),
            )
        trabajo["sensor_kind"] = trabajo["entity_id"].map(tipo_por_entidad)
        self._reportar_progreso(
            progress_callback,
            "classifying_entities",
            62.0,
            "Clasificando entidades continuas, binarias y multivaloricas",
        )
        trabajo["tipo_interno"] = trabajo["sensor_kind"].map(
            {
                "continuous": "continua",
                "binary": "estado_binario",
                "multistate": "estado_multivalor",
            }
        )
        trabajo["value"] = None
        es_valido = trabajo["is_valid_value"].astype(bool)
        es_continuo = trabajo["sensor_kind"].eq("continuous") & es_valido
        es_binario = trabajo["sensor_kind"].eq("binary") & es_valido
        trabajo.loc[es_continuo, "value"] = valores_numericos.loc[es_continuo].astype(float)
        trabajo.loc[es_binario, "value"] = estado.loc[es_binario].isin(ESTADOS_ACTIVOS).astype(float)
        self._reportar_progreso(
            progress_callback,
            "normalizing_values",
            74.0,
            "Normalizando valores numericos y estados",
        )
        if "unit_of_measurement" in trabajo.columns:
            trabajo["unit_of_measurement"] = trabajo["unit_of_measurement"].astype(str).str.strip()
        else:
            trabajo["unit_of_measurement"] = ""
        if "device_class" in trabajo.columns:
            trabajo["device_class"] = trabajo["device_class"].astype(str).str.strip()
        else:
            trabajo["device_class"] = ""
        entity_lower = trabajo["entity_id"].str.lower()
        trabajo.loc[trabajo["device_class"].eq("") & entity_lower.str.contains("humidity", regex=False), "device_class"] = "humidity"
        trabajo.loc[trabajo["device_class"].eq("") & (entity_lower.str.contains("temperature", regex=False) | entity_lower.str.contains("temp", regex=False)), "device_class"] = "temperature"
        trabajo.loc[trabajo["device_class"].eq("") & (entity_lower.str.contains("motion", regex=False) | entity_lower.str.contains("pir", regex=False)), "device_class"] = "motion"
        trabajo.loc[trabajo["device_class"].eq("") & (entity_lower.str.contains("door", regex=False) | entity_lower.str.contains("contact", regex=False) | entity_lower.str.contains("window", regex=False)), "device_class"] = "door"
        self._reportar_progreso(
            progress_callback,
            "sorting_history",
            84.0,
            "Ordenando el historico por entidad y tiempo",
        )
        trabajo = trabajo.sort_values(["entity_id", "timestamp_dt"]).reset_index(drop=True)

        entidades = self._construir_entidades(trabajo)
        self._reportar_progreso(
            progress_callback,
            "building_catalog",
            93.0,
            f"Catalogo construido con {len(entidades):,} entidades",
        )
        muestras = self._construir_muestras_normalizacion(frame, trabajo)
        # El historial puede contener millones de filas. Estas columnas tienen
        # cardinalidad baja y no necesitan un objeto Python por celda. El valor
        # normalizado es siempre numerico; los estados multivalor se conservan
        # en `state`.
        trabajo["value"] = pd.to_numeric(trabajo["value"], errors="coerce")
        trabajo = trabajo.drop(columns=["_sensor_kind_candidate"], errors="ignore")
        for columna in (
            "entity_id",
            "domain",
            "sensor_kind",
            "tipo_interno",
            "unit_of_measurement",
            "device_class",
        ):
            if columna in trabajo.columns:
                trabajo[columna] = trabajo[columna].astype("category")
        if "_raw_index" in trabajo.columns and len(trabajo) < 2_147_483_647:
            trabajo["_raw_index"] = trabajo["_raw_index"].astype("int32")
        metadatos = {
            "source_type": origen,
            "source_path": str(ruta),
            "rows": int(len(trabajo)),
            "entity_count": int(trabajo["entity_id"].nunique()) if not trabajo.empty else 0,
            "start": None if trabajo.empty else trabajo["timestamp_dt"].min().isoformat(),
            "end": None if trabajo.empty else trabajo["timestamp_dt"].max().isoformat(),
        }
        self._reportar_progreso(
            progress_callback,
            "history_ready",
            100.0,
            "Lectura y normalizacion completadas",
        )
        return ResultadoHistorial(
            eventos=trabajo,
            entidades=entidades,
            metadatos=metadatos,
            normalizacion_muestras=muestras,
        )

    @staticmethod
    def _reportar_progreso(
        callback: ProgressCallback | None,
        phase: str,
        progress: float,
        message: str,
    ) -> None:
        if callback is not None:
            callback(phase, float(progress), message)

    def _clasificar_fila(self, fila: pd.Series) -> str:
        return infer_kind(str(fila.get("entity_id", "")), [fila.get("state", "")],
                          declared=fila.get("sensor_kind", ""), domain=fila.get("domain", ""),
                          unit=fila.get("unit_of_measurement", ""), device_class=fila.get("device_class", ""))

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

    def _construir_muestras_normalizacion(
        self,
        original: pd.DataFrame,
        normalizado: pd.DataFrame,
        filas_por_entidad: int = 5,
    ) -> list[dict[str, Any]]:
        if original.empty or normalizado.empty:
            return []

        columnas_originales = [
            columna
            for columna in ["entity_id", "timestamp", "state", "domain", "unit_of_measurement", "device_class", "sensor_kind"]
            if columna in original.columns
        ]
        columnas_normalizadas = [
            "entity_id",
            "timestamp",
            "state",
            "domain",
            "sensor_kind",
            "tipo_interno",
            "value",
            "unit_of_measurement",
            "device_class",
            "is_valid_value",
        ]
        muestras = []
        for entity_id, grupo in normalizado.groupby("entity_id", sort=True):
            for fila in grupo.head(filas_por_entidad).to_dict("records"):
                raw_index = fila.get("_raw_index")
                original_fila = original.iloc[int(raw_index)].to_dict() if raw_index is not None else {}
                muestras.append(
                    {
                        "entity_id": str(entity_id),
                        "raw_index": int(raw_index) if raw_index is not None else None,
                        "original": {
                            columna: self._serializar_valor(original_fila.get(columna))
                            for columna in columnas_originales
                        },
                        "normalizado": {
                            columna: self._serializar_valor(fila.get(columna))
                            for columna in columnas_normalizadas
                        },
                    }
                )
        return muestras

    def _serializar_valor(self, valor: Any) -> Any:
        if pd.isna(valor):
            return None
        if isinstance(valor, pd.Timestamp):
            return valor.isoformat()
        return valor

    def _mezclar_entidades(self, inferidas: pd.DataFrame, declaradas: pd.DataFrame) -> pd.DataFrame:
        base = inferidas.set_index("entity_id")
        declaradas = declaradas.set_index("entity_id")
        for columna in declaradas.columns:
            if columna not in base.columns:
                base[columna] = ""
            indices = base.index.intersection(declaradas.index)
            base.loc[indices, columna] = declaradas.loc[indices, columna]
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
