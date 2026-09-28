"""Cache local del detector preparado; nunca deserializar archivos subidos.

El pickle conserva estimadores, deques y DataFrames sin perder sus tipos.
Solo se lee la ruta interna del volumen de cache, tras validar compatibilidad.
"""
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import pickle
import tempfile

import pandas as pd

from .Procedencia_Normalidad import runtime_versions, sha256_canonical


logger = logging.getLogger(__name__)
VERSION = 1


def file_digest(path: Path):
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_key(identity, history, contextual_config, map_path):
    root = Path(__file__).resolve().parent
    # Incluye dependencias locales indirectas, no solo la rutina de preparación.
    implementation = {
        path.relative_to(root).as_posix(): file_digest(path)
        for path in sorted(root.rglob("*.py"))
    }
    return sha256_canonical({
        "version": VERSION,
        "baseline": identity,
        "source": file_digest(Path(history.metadatos.get("source_path") or "")),
        "entities": history.entidades.to_json(orient="split", date_format="iso"),
        "contextual_config": asdict(contextual_config),
        "contextual_map": file_digest(map_path),
        "implementation": implementation,
        "runtime": runtime_versions(),
    })


def load(path: Path, key: str):
    try:
        with path.open("rb") as stream:
            header = json.loads(stream.readline(4096))
            if header != {"version": VERSION, "key": key}:
                return None
            payload = pickle.load(stream)
        if not isinstance(payload, dict):
            return None
        if not (isinstance(payload.get("baseline"), dict)
                and isinstance(payload.get("context_map"), dict)
                and isinstance(payload.get("context_events"), pd.DataFrame)
                and isinstance(payload.get("prepared_at"), str)):
            return None
        return payload
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("No se pudo recuperar la preparación causal; se recalculará", exc_info=True)
        return None


def save(path: Path, key: str, payload: dict) -> bool:
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write((json.dumps({"version": VERSION, "key": key}) + "\n").encode("utf-8"))
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except Exception:
        logger.warning("No se pudo persistir la preparación causal", exc_info=True)
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
