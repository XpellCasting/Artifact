from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ResultadoInyeccion:
    """Resultado de aplicar una falla sintetica a una lectura."""

    fila: dict[str, Any]
    aplicada: bool
    fault_type: str
    clean_state: str
    observed_state: str
    metadata: dict[str, Any]


class InyectorFallasSensornets:
    """Inyecta fallas siguiendo los modelos de SENSORNETS 2016.

    La seccion Fault Injection del paper trabaja sobre intervalos limpios S y
    anota los valores transformados con ground truth. En tiempo real cada tick
    activo se considera como una muestra perteneciente al intervalo configurado
    por la interfaz.
    """

    tipos_continuos = {"random", "random/spike", "malfunction", "bias", "drift", "missing"}
    tipos_binarios = {"random", "malfunction", "bias", "missing"}

    def aplicar(
        self,
        fila: dict[str, Any],
        grupo: pd.DataFrame,
        entidad: dict[str, Any],
        fault: dict[str, Any],
        tick_index: int,
        tick_seconds: int,
    ) -> ResultadoInyeccion:
        tipo = str(entidad.get("sensor_kind") or fila.get("sensor_kind") or "")
        fault_type = self._normalizar_tipo(str(fault.get("fault_type", "random")), tipo)
        clean_state = str(fila.get("state", ""))
        metadata: dict[str, Any] = {"modelo": "SENSORNETS_2016_fault_injection"}

        if fault_type == "missing":
            fila = self._aplicar_missing(fila, tick_seconds)
            return ResultadoInyeccion(
                fila=fila,
                aplicada=True,
                fault_type=fault_type,
                clean_state=clean_state,
                observed_state=str(fila.get("state", "")),
                metadata={**metadata, "formula": "brecha temporal no interpolada"},
            )

        if tipo == "continuous":
            return self._aplicar_continua(fila, grupo, fault, fault_type, tick_index, clean_state, metadata)
        if tipo == "binary":
            return self._aplicar_binaria(fila, fault_type, clean_state, metadata)

        return ResultadoInyeccion(
            fila=fila,
            aplicada=False,
            fault_type=fault_type,
            clean_state=clean_state,
            observed_state=str(fila.get("state", "")),
            metadata={**metadata, "motivo": "tipo de entidad no soportado para inyeccion"},
        )

    def _aplicar_continua(
        self,
        fila: dict[str, Any],
        grupo: pd.DataFrame,
        fault: dict[str, Any],
        fault_type: str,
        tick_index: int,
        clean_state: str,
        metadata: dict[str, Any],
    ) -> ResultadoInyeccion:
        valor = float(pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0])
        valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
        stats = self._stats_intervalo(valores)
        rng = random.Random(self._semilla(fila, fault_type, tick_index))
        aplicada = True

        if fault_type in {"random", "random/spike"}:
            density = float(fault.get("density", 1.0))
            aplicada = rng.random() <= max(0.0, min(1.0, density))
            intensity = self._elegir_intensidad(fault, "ir", [1.5, 2.5], rng)
            signo = -1.0 if rng.random() < 0.5 else 1.0
            nuevo_valor = valor * (1.0 + signo * intensity)
            formula = "I_k = S_k(1 + intensity)"
        elif fault_type == "malfunction":
            nm = float(fault.get("nm", 1.5))
            ruido = self._ruido_gaussiano(rng, stats["std"], stats["scale"]) * nm
            nuevo_valor = valor + ruido
            formula = "I_k = S_k + N(0, sigma_original^2) * n_m"
        elif fault_type == "drift":
            nd = float(fault.get("nd", 2.0))
            intensity = self._elegir_intensidad(fault, "id", [0.08, 0.12, -0.08, -0.12], rng)
            progreso = max(1, tick_index)
            offset = stats["scale"] * intensity * progreso
            ruido = self._ruido_gaussiano(rng, stats["std"], stats["scale"]) * nd * 0.25
            nuevo_valor = valor + offset + ruido
            formula = "I_k = S_k + N(0, sigma_original^2) * n_d + offset"
        elif fault_type == "bias":
            nb = float(fault.get("nb", 0.0))
            intensity = self._elegir_intensidad(fault, "ib", [1.5], rng)
            nuevo_promedio = stats["mean"] * intensity
            ruido = self._ruido_gaussiano(rng, stats["std"], stats["scale"]) * nb
            nuevo_valor = nuevo_promedio + ruido
            formula = "I_k = (media_original * i_b) + N(0, sigma_original^2) * n_b"
        else:
            return ResultadoInyeccion(fila, False, fault_type, clean_state, clean_state, metadata)

        if not aplicada:
            return ResultadoInyeccion(
                fila=fila,
                aplicada=False,
                fault_type=fault_type,
                clean_state=clean_state,
                observed_state=clean_state,
                metadata={**metadata, "formula": formula, "density_skip": True},
            )

        fila["value"] = round(float(nuevo_valor), 6)
        fila["state"] = f"{float(nuevo_valor):.6f}"
        return ResultadoInyeccion(
            fila=fila,
            aplicada=True,
            fault_type=fault_type,
            clean_state=clean_state,
            observed_state=str(fila["state"]),
            metadata={**metadata, "formula": formula, "stats": stats},
        )

    def _aplicar_binaria(
        self,
        fila: dict[str, Any],
        fault_type: str,
        clean_state: str,
        metadata: dict[str, Any],
    ) -> ResultadoInyeccion:
        valor = float(pd.to_numeric(pd.Series([fila.get("value")]), errors="coerce").fillna(0.0).iloc[0])
        if fault_type in {"random", "malfunction"}:
            valor = 0.0 if valor >= 0.5 else 1.0
            formula = "inversion discreta de estado para evento de falla"
        elif fault_type == "bias":
            valor = 1.0
            formula = "permanencia forzada en estado activo"
        else:
            return ResultadoInyeccion(fila, False, fault_type, clean_state, clean_state, metadata)

        fila["value"] = valor
        fila["state"] = "on" if valor >= 0.5 else "off"
        return ResultadoInyeccion(
            fila=fila,
            aplicada=True,
            fault_type=fault_type,
            clean_state=clean_state,
            observed_state=str(fila["state"]),
            metadata={**metadata, "formula": formula},
        )

    def _aplicar_missing(self, fila: dict[str, Any], tick_seconds: int) -> dict[str, Any]:
        missing_seconds = max(3600, tick_seconds * 10)
        timestamp = pd.to_datetime(fila["timestamp_dt"], errors="coerce", utc=True) + pd.Timedelta(seconds=missing_seconds)
        fila["timestamp_dt"] = timestamp
        fila["timestamp"] = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        return fila

    def _stats_intervalo(self, valores: pd.Series) -> dict[str, float]:
        if valores.empty:
            return {"mean": 1.0, "std": 1.0, "variance": 1.0, "scale": 1.0}
        mean = float(valores.mean())
        std = float(valores.std()) if len(valores) > 1 else 0.0
        variance = float(std**2)
        iqr = float(valores.quantile(0.75) - valores.quantile(0.25))
        scale = max(abs(iqr), abs(std), abs(float(valores.median())) * 0.05, 1.0)
        return {"mean": mean, "std": max(std, scale * 0.25), "variance": variance, "scale": scale}

    def _ruido_gaussiano(self, rng: random.Random, std: float, scale: float) -> float:
        ruido = rng.gauss(0.0, max(std, scale * 0.25))
        if abs(ruido) < scale * 0.5:
            ruido = (scale * 0.5) * (-1.0 if ruido < 0 else 1.0)
        return ruido

    def _elegir_intensidad(
        self,
        fault: dict[str, Any],
        clave: str,
        defecto: list[float],
        rng: random.Random,
    ) -> float:
        valores = fault.get(clave) or fault.get("intensities") or defecto
        if not isinstance(valores, list) or not valores:
            valores = defecto
        return float(valores[rng.randrange(0, len(valores))])

    def _normalizar_tipo(self, fault_type: str, sensor_kind: str) -> str:
        tipo = fault_type.strip().lower()
        if tipo == "spike":
            tipo = "random/spike"
        if sensor_kind == "continuous" and tipo in self.tipos_continuos:
            return tipo
        if sensor_kind == "binary" and tipo in self.tipos_binarios:
            return tipo
        return "random" if sensor_kind == "binary" else "random/spike"

    def _semilla(self, fila: dict[str, Any], fault_type: str, tick_index: int) -> int:
        base = f"{fila.get('entity_id')}|{fault_type}|{tick_index}|{fila.get('timestamp')}"
        return sum((indice + 1) * ord(caracter) for indice, caracter in enumerate(base))
