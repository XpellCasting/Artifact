"""Contrato de configuración y particiones temporales del experimento."""
from __future__ import annotations
from app.config import setting

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math
import os
import re
from pathlib import Path
from pathlib import PureWindowsPath
import json

from app.modulos.clasificador import SCHEMA_VERSION


@dataclass(frozen=True)
class Config:
    method: str = "gat_lstm"
    schema_version: str = SCHEMA_VERSION
    task: str = "diagnostic_3class"
    architecture_version: str = "gat_lstm_v3_segments_multiscale"
    grid_seconds: int = 60
    sequence_steps: int = 60
    lookback_seconds: int | None = None
    feature_window_seconds: int = 900
    labelled_stride_seconds: int = 300
    pretrain_stride_seconds: int = 900
    pretrain_max_samples: int = 4096
    graph_top_k: int = 4
    max_nodes: int = 32
    entropy_bins: int = 16
    dtw_reference_count: int = 3
    dtw_band_fraction: float = .1
    gat_heads: int = 4
    gat_head_dim: int = 8
    lstm_hidden: int = 32
    dropout: float = .1
    batch_size: int = 16
    learning_rate: float = .001
    max_epochs: int = 30
    patience: int = 5
    seed: int = 42
    device: str = "cpu"
    pretrain_enabled: bool = False
    use_entropy: bool = True
    use_dtw: bool = True
    segment_threshold: float = .5
    local_timezone: str = "America/Santiago"
    entity_value_modes: dict[str, str] | None = None
    segment_encoder: str = "gat"
    active_scales: tuple[bool, bool, bool] = (True, True, True)

    def __post_init__(self):
        expected_schema = SCHEMA_VERSION
        if self.method != "gat_lstm" or self.schema_version != expected_schema or self.task != "diagnostic_3class":
            raise ValueError("Método, tarea o versión de esquema incompatibles")
        if self.architecture_version not in {"gat_lstm_v3_segments_multiscale"}:
            raise ValueError("Versión de arquitectura GAT-LSTM incompatible")
        if self.pretrain_enabled:
            raise ValueError("El preentrenamiento predictivo aún no está disponible en GAT-LSTM por segmentos")
        expected_lookback = (self.sequence_steps - 1) * self.grid_seconds
        if self.lookback_seconds is None:
            object.__setattr__(self, "lookback_seconds", expected_lookback)
        elif self.lookback_seconds != expected_lookback:
            raise ValueError(f"lookback_seconds se deriva de pasos y grilla; esperado {expected_lookback}")
        for key in ("grid_seconds", "sequence_steps", "feature_window_seconds",
                    "labelled_stride_seconds", "pretrain_stride_seconds", "pretrain_max_samples",
                    "max_nodes", "entropy_bins", "batch_size", "max_epochs"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} debe ser positivo")
        if (self.gat_heads <= 0 or self.gat_head_dim <= 0 or self.gat_heads * self.gat_head_dim != 32
                or self.lstm_hidden <= 0 or self.patience <= 0 or self.graph_top_k < 0
                or self.graph_top_k >= self.max_nodes or not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0 or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1
                or not 0 <= self.dtw_band_fraction <= 1 or self.dtw_reference_count <= 0):
            raise ValueError("Dimensiones de GAT, tasa, paciencia, vecinos o parámetros DTW inválidos")
        if self.max_nodes > 32:
            raise ValueError("El máximo de entidades no puede superar 32")
        if not isinstance(self.device, str) or (self.device not in {"cpu", "auto"}
                and not re.fullmatch(r"cuda(?::[0-9]+)?", self.device)):
            raise ValueError("Dispositivo inválido")

        if not math.isfinite(self.segment_threshold) or not 0 < self.segment_threshold < 1:
            raise ValueError("Umbral DTW inválido")
        from zoneinfo import ZoneInfo
        ZoneInfo(self.local_timezone)
        if self.segment_encoder not in {"gat", "lstm_only"}:
            raise ValueError("Encoder de segmentos: gat o lstm_only")
        if len(self.active_scales) != 3 or any(type(v) is not bool for v in self.active_scales) or not any(self.active_scales):
            raise ValueError("Se requieren tres máscaras de escala y al menos una activa")
        if self.entity_value_modes is not None and (not isinstance(self.entity_value_modes, dict) or
                any(mode not in {"numeric", "categorical"} for mode in self.entity_value_modes.values())):
            raise ValueError("Modos de entidad: numeric o categorical")

    @property
    def context_seconds(self):
        return 259200

    @classmethod
    def from_json(cls, path: Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Parámetros desconocidos: {sorted(unknown)}")
        return cls(**data)

    def as_dict(self):
        return asdict(self)


def parse_utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Fecha sin zona: {value}")
    return result.astimezone(timezone.utc)


def partition_bounds(manifest: dict) -> dict[str, tuple[datetime, datetime | None]]:
    boundaries = [parse_utc(value) for value in manifest["boundaries"]]
    if boundaries != sorted(boundaries) or len(boundaries) != 4:
        raise ValueError("Límites de partición inválidos")
    fit_end = (parse_utc(manifest["train_fit_end"]) if manifest.get("train_fit_end")
               else boundaries[0] + (boundaries[1] - boundaries[0]) * .8)
    if not boundaries[0] < fit_end < boundaries[1]:
        raise ValueError("train_fit_end debe quedar dentro de Desarrollo")
    return {
        "train_fit": (boundaries[0], fit_end),
        "dev_tune": (fit_end, boundaries[1]),
        "validation": (boundaries[1], boundaries[2]),
        "test": (boundaries[2], boundaries[3]),
        "reserve": (boundaries[3], None),
    }


def actual_partition(when: datetime, bounds: dict) -> str | None:
    for name, (start, end) in bounds.items():
        if when >= start and (end is None or when < end):
            return name
    return None


def required_context_seconds(config) -> int:
    return config.context_seconds


def historical_manifest_path() -> Path:
    from app.config import HISTORICAL_DATA_DIR
    return Path(setting('HISTORICAL_DATA_DIR')) / "particiones_seleccionadas_resumen.json"


def resolve_manifest_source(manifest_path: Path, source_path: str) -> Path:
    path = Path(source_path)
    if path.is_file():
        return path
    # Legacy manifests store Windows paths; Docker mounts the same files
    # under the manifest directory, so resolve by the exact recorded basename.
    candidate = Path(manifest_path).parent / PureWindowsPath(str(source_path)).name
    return candidate if candidate.is_file() else path
