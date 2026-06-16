from __future__ import annotations

from typing import Any

import pandas as pd

from .simulacion import frame_to_csv


ENTITY_EXPORT_COLUMNS = [
    "entity_id",
    "domain",
    "device_class",
    "state_class",
    "unit",
    "area_id",
    "device_id",
    "integration",
    "labels",
]


def export_artifact_csv(
    name: str,
    states: pd.DataFrame,
    entities: dict[str, dict[str, Any]],
    ground_truth: pd.DataFrame,
) -> str:
    if name == "states":
        return frame_to_csv(states)
    if name == "entities":
        return frame_to_csv(complete_entity_export_columns(pd.DataFrame(entities.values())))
    if name == "ground_truth":
        return frame_to_csv(ground_truth)
    raise ValueError(f"Unknown export: {name}")


def complete_entity_export_columns(entities: pd.DataFrame) -> pd.DataFrame:
    working = entities.copy()
    for column in ENTITY_EXPORT_COLUMNS:
        if column not in working.columns:
            working[column] = ""
    return working[ENTITY_EXPORT_COLUMNS]
