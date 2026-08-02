"""Reprocesa el mes real y resume el puntaje continuo v2.

No carga ni crea el historial canonico/simulado. La ejecucion usa
``data/datosreales/history.csv`` y actualiza la cache normal de la aplicacion.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARTIFACT_ROOT / "backend"))

from app.main import state  # noqa: E402


def construir_resumen(
    status: dict[str, object],
    result: dict[str, object],
) -> dict[str, object]:
    rows = list(result.get("all_results") or result.get("results") or [])
    continuous = [
        row for row in rows if row.get("sensor_kind") == "continuous"
    ]
    scores = [float(row["suspicion_score"]) for row in continuous]
    component_scores = [
        float(value)
        for row in continuous
        for value in ((row.get("reading") or {}).get("components") or {}).values()
    ]
    contextual = (result.get("contextual") or {}).get("summary") or {}
    classification = (result.get("classification") or {}).get("summary") or {}

    return {
        "data_policy": {
            "source_type": status.get("source_type"),
            "source_path": status.get("source_path"),
            "uses_simulated_dataset": False,
        },
        "history": {
            "rows": status.get("rows"),
            "entities": status.get("entity_count"),
            "start": status.get("start"),
            "end": status.get("end"),
        },
        "cache_hit": (result.get("summary") or {}).get("cache_hit"),
        "candidates": {
            "total": len(rows),
            "continuous": len(continuous),
            "binary": sum(
                row.get("sensor_kind") == "binary" for row in rows
            ),
        },
        "continuous_scores": {
            "minimum": min(scores) if scores else None,
            "median": median(scores) if scores else None,
            "maximum": max(scores) if scores else None,
            "unique_6_decimals": len({round(value, 6) for value in scores}),
            "equal_0_714286": sum(
                abs(value - 0.714286) < 0.0000005 for value in scores
            ),
            "equal_1": sum(value == 1.0 for value in scores),
            "components_equal_1": sum(
                value == 1.0 for value in component_scores
            ),
            "hour_context_scores_equal_1": sum(
                float(
                    (row.get("temporal_context") or {}).get(
                        "hour_of_day_score",
                        0.0,
                    )
                )
                == 1.0
                for row in continuous
            ),
            "evidence_score_mismatches": sum(
                float(row["evidence_score"])
                != float(row["suspicion_score"])
                for row in continuous
            ),
            "versions": dict(
                Counter(row.get("score_version") for row in continuous)
            ),
        },
        "contextual_outcomes": dict(contextual.get("by_outcome") or {}),
        "classifier_classes": dict(classification.get("by_class") or {}),
    }


def main() -> None:
    status = state.status()
    if status.get("source_type") != "real":
        raise RuntimeError("La fuente activa no corresponde al historial real.")
    result = state.analizar_historial(0.6)
    print(
        json.dumps(
            construir_resumen(status, result),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
