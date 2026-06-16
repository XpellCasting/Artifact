from __future__ import annotations

from typing import Any

import pandas as pd


def select_clean_segments(
    normalized: pd.DataFrame,
    ground_truth: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Selecciona la referencia limpia que alimenta la base de normalidad.

    Version inicial: cuando existe ground truth sintetico, se excluyen segmentos
    etiquetados como anomalos. En datos reales no habra esas etiquetas; ahi esta
    funcion debe evolucionar a reglas de confiabilidad temporal: baja proporcion
    de estados invalidos, continuidad minima, ausencia de saltos extremos,
    cobertura horaria suficiente, minimo de dias por entidad y estabilidad
    estadistica.
    """
    if normalized.empty:
        return normalized.copy()

    working = normalized.copy()
    if ground_truth:
        clean_mask = working.apply(
            lambda row: is_clean_truth(ground_truth, str(row["timestamp_iso"]), str(row["entity_id"])),
            axis=1,
        )
        working = working[clean_mask].copy()
    else:
        working = select_unlabeled_reliable_rows(working)

    return working.sort_values(["entity_id", "timestamp_dt"]).reset_index(drop=True)


def select_unlabeled_reliable_rows(normalized: pd.DataFrame) -> pd.DataFrame:
    """Selecciona referencia cuando no existe ground truth.

    En historicos reales no se debe asumir que todas las lecturas son normales.
    Esta version inicial conserva los datos de estado y recorta extremos fuertes
    en continuas usando IQR amplio y desviacion robusta. No diagnostica fallas;
    solo evita que valores muy alejados dominen la base de normalidad.
    """
    selected = []
    for _, group in normalized.groupby("entity_id", sort=True):
        kind = str(group["sensor_kind"].iloc[0])
        if kind != "continuous":
            selected.append(group)
            continue

        values = pd.to_numeric(group["value"], errors="coerce")
        valid = group[values.notna()].copy()
        if len(valid) < 30:
            selected.append(valid)
            continue

        numeric = pd.to_numeric(valid["value"], errors="coerce")
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        median = float(numeric.median())
        mad_value = float((numeric - median).abs().median())
        lower = q1 - 3.0 * iqr
        upper = q3 + 3.0 * iqr
        robust_z = (numeric - median).abs() / (1.4826 * mad_value + 1e-9)
        mask = numeric.between(lower, upper) & (robust_z <= 8.0)
        filtered = valid[mask].copy()
        selected.append(filtered if len(filtered) >= 30 else valid)

    if not selected:
        return normalized.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def is_clean_truth(
    ground_truth: dict[tuple[str, str], dict[str, Any]],
    timestamp: str,
    entity_id: str,
) -> bool:
    truth = ground_truth.get((timestamp, entity_id))
    return not truth or str(truth.get("label", "normal")) == "normal"


def clean_segment_record(entity_id: str, training: pd.DataFrame) -> dict[str, Any] | None:
    if training.empty:
        return None
    return {
        "entity_id": entity_id,
        "start": str(training["timestamp_iso"].iloc[0]),
        "end": str(training["timestamp_iso"].iloc[-1]),
        "rows": int(len(training)),
        "accepted": True,
    }
