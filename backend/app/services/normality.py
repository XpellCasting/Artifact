from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any
import pandas as pd
from app.modulos.estadistico.episodios import DEFAULT_QUIET_SECONDS
from app.modulos.estadistico.episodios import EPISODE_POLICY_VERSION
from app.modulos.estadistico import episodios
from app.services.application_common import serializable_json

class ServicesNormalityMixin:
    def _identidad_historial_real(
        self, suspicion_threshold: float, suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        provenance = (self.future_baseline_template or {}).get("provenance", {})
        contextual_map_path = self._ruta_cache_contextual()
        contextual_map_expired: bool | None = None
        if (
            contextual_map_path is not None
            and contextual_map_path.is_file()
        ):
            try:
                with contextual_map_path.open("r", encoding="utf-8") as archivo:
                    contextual_map_payload = json.load(archivo)
                built_at = pd.to_datetime(
                    contextual_map_payload.get("built_at"),
                    errors="coerce",
                    utc=True,
                )
                contextual_map_expired = bool(
                    pd.isna(built_at)
                    or (
                        pd.Timestamp.now(tz="UTC") - built_at
                    ).total_seconds()
                    > (
                        self.filtro_contextual.config.max_map_age_days
                        * 86400.0
                    )
                )
            except (OSError, json.JSONDecodeError):
                contextual_map_expired = True
        return {
            "cache_version": 24,
            "baseline_id": provenance.get("baseline_id"),
            "effective_parameters_sha256": provenance.get("effective_parameters_sha256"),
            "suspicion_threshold": round(float(suspicion_threshold), 6),
            "episode_policy_version": EPISODE_POLICY_VERSION,
            "suspicion_quiet_seconds": suspicion_quiet_seconds,
            "episode_implementation_sha256": self._sha256_archivo(
                Path(episodios.__file__)
            ),
            "continuous_score_specification": (
                self.filtro_estadistico.continuos.especificacion_puntaje()
            ),
            "multistate_score_specification": (
                self.filtro_estadistico.multivalor.especificacion_puntaje()
            ),
            "contextual_filter_version": self.filtro_contextual.MAP_VERSION,
            "contextual_configuration": (
                self.filtro_contextual.config.como_dict()
            ),
            "contextual_implementation_sha256": (
                self.filtro_contextual.implementation_sha256()
            ),
            "contextual_map_sha256": (
                self._sha256_archivo(contextual_map_path)
                if contextual_map_path is not None
                and contextual_map_path.is_file()
                else ""
            ),
            "contextual_map_expired": contextual_map_expired,
        }
    def _identidad_normalidad_real(self) -> dict[str, Any] | None:
        if not self._es_historial_real_activo() or not self.real_history_path.exists():
            return None
        provenance = (self.future_baseline_template or {}).get("provenance", {})
        if not provenance.get("baseline_id"):
            return None
        return {
            "cache_version": 5,
            "baseline_id": provenance["baseline_id"],
            "effective_parameters_sha256": provenance.get("effective_parameters_sha256"),
        }
    def _normality_cache_path(self, identidad: dict[str, Any]) -> Path:
        nombre = f"normality_{identidad['baseline_id'][:16]}.json"
        return self.analysis_cache_path.parent / nombre
    def _sha256_archivo(self, ruta: Path) -> str:
        digest = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for chunk in iter(lambda: archivo.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    def _asegurar_normalidad_persistida(self) -> dict[str, Any]:
        if self.historial is None:
            return {"exists": False, "rows": 0}
        identidad = self._identidad_normalidad_real()
        if identidad is None:
            return {
                "exists": True,
                "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
                "mode": "memory",
            }
        path = self._normality_cache_path(identidad)
        if not self._normalidad_cache_valida(path, identidad):
            payload = {
                "identity": identidad,
                "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
                "entity_count": int(self.historial.metadatos.get("entity_count", len(self.historial.entidades))),
                "by_kind": self.historial.entidades["sensor_kind"].value_counts().to_dict(),
                "entities": self._normalidad_por_entidad(),
            }
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as archivo:
                    json.dump(serializable_json(payload), archivo, ensure_ascii=True)
            except OSError:
                return {"exists": False, "rows": 0, "mode": "disk_error"}
        return {
            "exists": True,
            "rows": int(self.historial.metadatos.get("rows", len(self.historial.eventos))),
            "mode": "persisted",
            "path": str(path),
        }
    def _normalidad_cache_valida(self, path: Path, identidad: dict[str, Any]) -> bool:
        if not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as archivo:
                payload = json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("identity") == identidad and isinstance(payload.get("entities"), dict)
    def _leer_normalidad_persistida(self, status: dict[str, Any]) -> dict[str, Any]:
        path = status.get("path")
        if not path:
            return status
        try:
            with Path(path).open("r", encoding="utf-8") as archivo:
                return json.load(archivo)
        except (OSError, json.JSONDecodeError):
            return status
    def _normalidad_por_entidad(self) -> dict[str, Any]:
        if self.historial is None or self.historial.eventos.empty:
            return {}
        entidades = self.historial.entidades.set_index("entity_id").to_dict("index") if not self.historial.entidades.empty else {}
        salida: dict[str, Any] = {}
        for entity_id, grupo in self.historial.eventos.groupby("entity_id", sort=True):
            entity_id = str(entity_id)
            entidad = entidades.get(entity_id, {})
            tipo = str(entidad.get("sensor_kind", grupo["sensor_kind"].iloc[0] if "sensor_kind" in grupo else "multistate"))
            base: dict[str, Any] = {
                "sensor_kind": tipo,
                "tipo_interno": str(entidad.get("tipo_interno", "")),
                "rows": int(len(grupo)),
            }
            if tipo == "continuous":
                valores = pd.to_numeric(grupo["value"], errors="coerce").dropna()
                if valores.empty:
                    base.update({"median": 0.0, "mad": 0.0, "scaled_mad": 1.0})
                else:
                    mediana = float(valores.median())
                    mad = float((valores - mediana).abs().median())
                    escala = max(1.4826 * mad, 0.01)
                    base.update(
                        {
                            "median": mediana,
                            "mad": mad,
                            "scaled_mad": escala,
                            "p05": float(valores.quantile(0.05)),
                            "p95": float(valores.quantile(0.95)),
                        }
                    )
            elif tipo == "binary":
                trabajo = grupo.sort_values("timestamp_dt").copy()
                valores = pd.to_numeric(trabajo["value"], errors="coerce").fillna(0.0)
                activos = int((valores >= 0.5).sum())
                tiempos = pd.to_datetime(trabajo["timestamp_dt"], errors="coerce", utc=True)
                duraciones = tiempos.shift(-1).sub(tiempos).dt.total_seconds()
                mediana_gap = duraciones[duraciones.gt(0)].median()
                duraciones = duraciones.fillna(float(mediana_gap) if pd.notna(mediana_gap) else 0.0).clip(lower=0.0)
                total = float(duraciones.sum())
                activos_segundos = float(duraciones[valores >= 0.5].sum())
                base.update(
                    {
                        "active_ratio": activos_segundos / total if total > 0 else float(activos / len(valores)) if len(valores) else 0.0,
                        "on_count": activos,
                        "off_count": int(len(valores) - activos),
                    }
                )
            else:
                estados = grupo["state"].astype(str)
                base.update({"states": estados.value_counts().to_dict(), "unique_states": int(estados.nunique())})
            salida[entity_id] = base
        return salida
    def _leer_cache_analisis_real(
        self, suspicion_threshold: float, suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> dict[str, Any] | None:
        identidad = self._identidad_historial_real(suspicion_threshold, suspicion_quiet_seconds)
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
    def _guardar_cache_analisis_real(
        self, suspicion_threshold: float, resultado: dict[str, Any],
        suspicion_quiet_seconds: int = DEFAULT_QUIET_SECONDS,
    ) -> None:
        identidad = self._identidad_historial_real(suspicion_threshold, suspicion_quiet_seconds)
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
