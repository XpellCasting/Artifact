"""Declared coverage requirements; never move examples out of evaluation."""
from app.modulos.clasificador import LABELS
from app.modulos.clasificador.model_constants import FAULT_TYPES


def training_coverage(dataset):
    missing_classes = {part:[label for label in LABELS
        if dataset.get("label_counts",{}).get(part,{}).get(label,0)<1
        or dataset.get("episode_counts",{}).get(part,{}).get(label,0)<1]
        for part in ("train_fit","dev_tune")}
    inventory = dataset.get("fault_type_inventory",{})
    train_types = inventory.get("train_fit",{})
    missing_types = [kind for kind in FAULT_TYPES
        if train_types.get(kind,{}).get("windows",0)<1
        or train_types.get(kind,{}).get("independent_episodes",0)<1]
    weak_types = {part:[kind for kind in FAULT_TYPES
        if inventory.get(part,{}).get(kind,{}).get("independent_episodes",0)<3]
        for part in ("train_fit","dev_tune")}
    return {"ready":not missing_types and not any(missing_classes.values()),
            "missing_classes":missing_classes,"missing_fault_types_train_fit":missing_types,
            "fault_types_with_fewer_than_three_episodes":weak_types,
            "evaluation_rows_used":False,"require_all_fault_types":True}


def require_training_coverage(dataset):
    report = training_coverage(dataset)
    if not report["ready"]:
        raise ValueError("Cobertura de ajuste insuficiente. Clases ausentes: "
            f"{report['missing_classes']}; tipos de falla ausentes de train_fit: "
            f"{report['missing_fault_types_train_fit']}. No se trasladarán ejemplos de evaluación al entrenamiento.")
    return report
