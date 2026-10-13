from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from importlib.metadata import version as package_version
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import sklearn


SCHEMA_VERSION = "normality_provenance_v1"
EVENT_COLUMNS = ("entity_id", "timestamp_dt", "state", "value", "is_valid_value", "sensor_kind", "reporting_mode", "max_report_gap_seconds")
ENTITY_COLUMNS = ("entity_id", "sensor_kind", "tipo_interno", "domain", "device_class", "unit_of_measurement", "reporting_mode", "max_report_gap_seconds")
IMPLEMENTATION_FILES = (
    "services/procedencia_normalidad.py",
    "services/historial/cargar_historial.py",
    "services/tipos_entidad.py",
    "services/estados_validos.py",
    "services/reproduccion.py",
    "modulos/estadistico/filtro.py",
    "modulos/estadistico/continuos.py",
    "modulos/estadistico/continuos_temporales.py",
    "modulos/estadistico/residuos_secuenciales.py",
    "modulos/estadistico/cobertura.py",
    "modulos/estadistico/persistencia.py",
    "services/entrenamiento_paralelo.py",
    "modulos/estadistico/estados_binarios.py",
    "modulos/estadistico/bernoulli.py",
    "modulos/estadistico/estados_multivalor.py",
)


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.tz_convert("UTC").isoformat() if value.tzinfo else value.tz_localize("UTC").isoformat()
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if hasattr(value, "item"):
        return _normalize(value.item())
    if isinstance(value, (list, tuple, set, deque)):
        return [_normalize(item) for item in value]
    if isinstance(value, (dict, Counter)):
        if all(isinstance(key, (str, int)) for key in value):
            return {str(key): _normalize(value[key]) for key in sorted(value, key=lambda item: str(item))}
        entries = [
            {"key": _normalize(key), "value": _normalize(item)}
            for key, item in value.items()
        ]
        return {"entries": sorted(entries, key=canonical_json)}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Tipo no serializable en procedencia: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_records(records: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def hash_development_events(events: pd.DataFrame) -> str:
    work = events.copy()
    work["__order"] = range(len(work))
    timestamps = pd.to_datetime(work.get("timestamp_dt"), errors="coerce", utc=True)
    work["__timestamp"] = timestamps
    work = work.sort_values(["entity_id", "__timestamp", "__order"], kind="stable", na_position="last")
    def records():
        for row in work.to_dict("records"):
            yield {column: _normalize(row.get(column)) for column in EVENT_COLUMNS}
    return _hash_records(records())


def hash_development_entities(entities: pd.DataFrame, present_entities: set[str]) -> str:
    if entities.empty:
        return _hash_records(())
    work = entities[entities["entity_id"].astype(str).isin(present_entities)].copy()
    work = work.sort_values("entity_id", kind="stable")
    return _hash_records(
        ({column: _normalize(row.get(column)) for column in ENTITY_COLUMNS} for row in work.to_dict("records"))
    )


def implementation_sha256(app_dir: Path | None = None) -> str:
    root = app_dir or Path(__file__).resolve().parent.parent
    digest = hashlib.sha256()
    for relative in IMPLEMENTATION_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__, "focus_cpt": package_version("focus-cpt")}


def build_identity(events: pd.DataFrame, entities: pd.DataFrame, manual_overrides: dict[str, Any], *, implementation_hash: str | None = None, runtime: dict[str, str] | None = None) -> dict[str, Any]:
    present = set(events["entity_id"].astype(str)) if "entity_id" in events else set()
    effective_overrides = {key: value for key, value in manual_overrides.items() if key in present}
    components = {
        "schema_version": SCHEMA_VERSION,
        "development_events_sha256": hash_development_events(events),
        "development_entities_sha256": hash_development_entities(entities, present),
        "manual_overrides_sha256": sha256_canonical(effective_overrides),
        "implementation_sha256": implementation_hash or implementation_sha256(),
        "runtime_versions": runtime or runtime_versions(),
    }
    return {**components, "baseline_id": sha256_canonical(components)}


def effective_parameters(baseline: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy({key: value for key, value in baseline.items() if key not in {"last_emitted", "provenance"}})
    for kind in ("continuous", "binary", "multistate"):
        for cached in result.get(kind, {}).values():
            if isinstance(cached, dict):
                cached.pop("runtime", None)
                # Los estimadores pertenecen exclusivamente a memoria. La
                # procedencia conserva ``metadata`` y el manifiesto continuo,
                # nunca objetos de scikit-learn ni su representación textual.
                cached.pop("model", None)
                cached.pop("predictor", None)
                if "persistence_reference" in cached:
                    cached["persistence_reference"].pop("runtime", None)
    return _normalize(result)


def write_manifest(cache_dir: Path, identity: dict[str, Any], source: dict[str, Any], manual_overrides: dict[str, Any], parameters: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    parameters_hash = sha256_canonical(parameters)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "baseline_id": identity["baseline_id"],
        "identity": identity,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": _normalize(source),
        "manual_overrides": _normalize(manual_overrides),
        "effective_parameters": parameters,
        "effective_parameters_sha256": parameters_hash,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"baseline_{identity['baseline_id']}.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Manifiesto de base corrupto: {path}") from exc
        if existing.get("identity") != identity or existing.get("effective_parameters_sha256") != parameters_hash:
            raise RuntimeError(f"Inconsistencia del manifiesto para baseline_id {identity['baseline_id']}")
        return path, existing
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise RuntimeError(f"No se pudo persistir el manifiesto de base en {path}") from exc
    return path, manifest
