"""Resultado explícito cuando no hay clasificador disponible."""
from copy import deepcopy

CLASSIFIER_MODEL_REQUIRED = (
    "No se puede iniciar la evaluación: el filtro clasificador no tiene un modelo disponible. "
    "Selecciona un modelo válido en el inspector del flujo o desactiva ese filtro para continuar."
)

def unavailable_results(results):
    enriched = deepcopy(results)
    for item in enriched:
        if item.get("routing", {}).get("terminal"):
            item.pop("diagnosis", None)
            item.pop("diagnosis_mode", None)
            continue
        item["diagnosis_mode"] = "unavailable"
        item["diagnosis"] = {
            "status": "indeterminate", "method": "gat_lstm", "predicted_label": None,
            "fault_type": None, "confidence": 0.0, "class_scores": {},
            "needs_review": True, "reasons": ["classifier_unavailable"],
            "entity_id": item.get("entity_id"), "timestamp": item.get("timestamp"),
        }
    return enriched
