"""Rutas compartidas. ARTIFACT_ROOT permite ejecutar desde cualquier directorio."""
import os
from pathlib import Path

ARTIFACT_ROOT = Path(os.environ.get("ARTIFACT_ROOT", Path(__file__).resolve().parents[2])).resolve()
def path_setting(name, relative):
    return Path(os.environ.get(name, ARTIFACT_ROOT / relative))
DATA_DIR = path_setting("DATA_DIR", "data")
STORAGE_DIR = path_setting("STORAGE_DIR", "storage")
DATABASE_DIR = path_setting("DATABASE_DIR", "storage/database")
MODELS_DIR = path_setting("MODELS_DIR", "models/gat_lstm")
REVIEW_DATA_DIR = path_setting("REVIEW_DATA_DIR", "data/review")
HISTORICAL_DATA_DIR = path_setting("HISTORICAL_DATA_DIR", "data/history")
CACHE_DIR = path_setting("CACHE_DIR", "storage/cache")
JOBS_DIR = path_setting("JOBS_DIR", "storage/jobs")
CORS_ORIGINS = [s.strip() for s in os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if s.strip()]

def database_path(name, directory=None):
    # Isolated workbenches used by tests/embedders retain their own storage.
    base = DATABASE_DIR if directory is None or Path(directory).resolve() == REVIEW_DATA_DIR.resolve() else Path(directory) / "database"
    base.mkdir(parents=True, exist_ok=True)
    return base / name

def artifact_directory(directory, folder):
    directory = Path(directory)
    if directory.resolve() == MODELS_DIR.resolve():
        if folder in {"datasets", "sequences"}:
            return DATA_DIR / "gat_lstm" / folder
        if folder == "jobs":
            return JOBS_DIR / "diagnostics"
    return directory / folder

def artifact_file(directory, relative):
    relative = Path(relative)
    if len(relative.parts) > 1:
        return artifact_directory(directory, relative.parts[0]).joinpath(*relative.parts[1:])
    return Path(directory) / relative

# Defaults live here; read overrides on demand for isolated tests and worker processes.
ENVIRONMENT_DEFAULTS = {
    'ANALYSIS_CACHE_PATH': CACHE_DIR / "real_history_analysis.json",
    'CONTEXT_CALIBRATION_WORKERS': '1',
    'CONTEXT_CONTINUOUS_MAX_AGE_BY_ENTITY': "{}",
    'CONTEXT_CONTINUOUS_MAX_AGE_SECONDS': "3600",
    'CONTEXT_MAP_CACHE_PATH': CACHE_DIR / "contextual_map.json",
    'CONTINUOUS_CONF_PATH': CACHE_DIR / "conf.txt",
    'FLOW_LAB_DB_PATH': DATABASE_DIR / "flow_lab.sqlite3",
    'GAT_ATTENTION_BACKEND': 'dense',
    'GAT_SEQUENCE_CACHE_DIR': None,
    'HISTORICAL_DATA_DIR': HISTORICAL_DATA_DIR,
    'HISTORY_LOAD_SECONDS_PER_MIB': "0.22",
    'HISTORY_TRAINING_WORKERS': "1",
    'MKL_NUM_THREADS': "unset",
    'OMP_NUM_THREADS': "unset",
    'REAL_HISTORY_PATH': HISTORICAL_DATA_DIR / "cargados/history.csv",
    'REVIEW_DATA_DIR': REVIEW_DATA_DIR,
    'STATIC_HISTORY_DIR': DATA_DIR / "static_history",
    'VALIDATION_HISTORY_PATH': HISTORICAL_DATA_DIR / "cargados/historico_validacion.csv",
    'VALIDATION_REPLAY_MEMORY_RESULTS': "500",
    'VALIDATION_REPLAY_STATUS_RESULTS': "25",
}

def setting(name):
    return os.environ.get(name, ENVIRONMENT_DEFAULTS[name])
