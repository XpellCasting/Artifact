"""Exporta evidencia PNG/CSV de un informe de escenarios, sin reentrenar."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns

CLASSES = ["normal", "environmental_anomaly", "sensor_fault"]
TYPES = ["bias", "drift", "noise", "spike", "state_replace", "stuck", "unavailable"]
LABELS = {
    "normal": "Normal", "environmental_anomaly": "Anomalía ambiental",
    "sensor_fault": "Falla", "abstain": "Abstención",
    "bias": "Desplazamiento", "drift": "Deriva", "noise": "Ruido",
    "spike": "Pico", "state_replace": "Estado sustituido", "stuck": "Congelación",
    "unavailable": "Indisponibilidad (unavailable)",
    "missed_or_unpublished_type": "Omitida / tipo no publicado",
    "precision": "Precisión", "recall": "Recall", "f1": "F1",
}


def label(value):
    return LABELS.get(value, value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Figures:
    def __init__(self, output, run, dpi):
        self.output, self.run, self.dpi = output, run, dpi
        self.items = []
        output.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
        plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                             "axes.spines.right": False})

    def save(self, name, fig, table, title, note):
        fig.text(.04, .965, title, ha="left", va="top", fontsize=17, fontweight="bold")
        fig.text(.04, .90, self.run, ha="left", va="top", color="#475569", fontsize=10)
        footer = "\n".join(textwrap.fill(line, 115) for line in note.splitlines())
        fig.text(.04, .025, footer, ha="left", va="bottom", fontsize=9, color="#475569")
        fig.tight_layout(rect=(.02, .12, .98, .87))
        path = self.output / (name + ".png")
        fig.savefig(path, dpi=self.dpi, facecolor="white")
        plt.close(fig)
        table.to_csv(self.output / (name + ".csv"), index=True, encoding="utf-8-sig")
        self.items.append({"png": path.name, "csv": name + ".csv", "title": title,
                           "note": note, "sha256": sha256(path)})

    def bars(self, name, values, title, note, percent=False):
        series = pd.Series(values, dtype=float)
        fig, ax = plt.subplots(figsize=(11, max(5.5, len(series)*.48+2.5)))
        sns.barplot(x=series.values, y=series.index, color="#287e8e", ax=ax)
        if percent:
            ax.set_xlim(0, 1.12)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_xlim(0, max(1., series.max())*1.22)
        for i, value in enumerate(series):
            if pd.notna(value):
                text = f"{value:.2%}" if percent else f"{value:,.2f}".rstrip("0").rstrip(".")
                ax.text(value + ax.get_xlim()[1]*.012, i, text, va="center", fontsize=10)
            else:
                ax.text(0, i, "Sin datos", va="center")
        ax.set(xlabel="Proporción" if percent else "Valor", ylabel="")
        self.save(name, fig, series.to_frame("valor"), title, note)

    def heatmap(self, name, frame, title, note, normalized=False):
        display = frame.copy()
        if normalized:
            display = display.div(display.sum(axis=1).replace(0, np.nan), axis=0)
        fig, ax = plt.subplots(figsize=(12, max(5.8, len(frame)*.48+2.5)))
        sns.heatmap(display, annot=True, fmt=".1%" if normalized else ".0f",
                    cmap="Blues", vmin=0, vmax=1 if normalized else None,
                    linewidths=.6, ax=ax, cbar_kws={"label": "Proporción por fila" if normalized else "Casos"})
        ax.set(xlabel="Predicción", ylabel="Ground truth")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
        self.save(name, fig, display, title, note)


def plot_report(report, run, figures, entity_ids=None):
    result = report["results"][run]
    classes = result["classes"]
    matrix = np.asarray(classes["matrix"], dtype=int)
    total = classes["support"]
    if matrix.shape != (3, 4) or matrix.sum() != total:
        raise ValueError("Matriz de clases incompatible con el total de evaluación")
    if classes["accepted"] != int(matrix[:, :3].sum()):
        raise ValueError("Cobertura inconsistente con la matriz de clases")
    note = f"N = {total:,} objetivos evaluados. Las ventanas y réplicas no son independientes."
    frame = pd.DataFrame(matrix, index=[label(k) for k in CLASSES],
                         columns=[label(k) for k in classes["columns"]])
    for normalized in (False, True):
        figures.heatmap("clases_matriz_" + ("porcentajes" if normalized else "conteos"),
                        frame, "Clasificación principal · matriz de confusión", note, normalized)
    figures.bars("clases_soporte", {label(k): classes["per_class"][k]["support"] for k in CLASSES},
                 "Distribución del ground truth", note)
    for metric in ("precision", "recall", "f1"):
        figures.bars("clases_"+metric,
                     {label(k): classes["per_class"][k][metric] for k in CLASSES},
                     label(metric) + " por clase", "Recall y F1 incluyen las abstenciones como casos no resueltos.", True)
        figures.bars("tipos_"+metric,
                     {label(k): result["published"]["per_type"][k][metric] for k in TYPES},
                     label(metric) + " por tipo de falla · salida publicada",
                     "unavailable es un tipo de indisponibilidad; no significa ausencia de histórico.", True)
    accepted = classes["accepted"]
    correct = int(np.trace(matrix[:, :3]))
    figures.bars("cobertura_decision", {"Aceptados": accepted/total if total else None,
                  "Abstenciones": (total-accepted)/total if total else None},
                 "Cobertura de decisión", "No equivale a cobertura de entrada. El agregado no separa los motivos de abstención.", True)
    figures.bars("acierto", {"Acierto / total": correct/total if total else None,
                  "Acierto / aceptados": correct/accepted if accepted else None},
                 "Acierto global y entre diagnósticos aceptados",
                 f"Correctos: {correct:,}. Total: {total:,}. Aceptados: {accepted:,}.", True)
    figures.bars("f1_macro", {"Tres clases": classes["macro_f1"],
                  "Siete tipos publicados": result["published"]["macro_f1"]},
                 "F1 macro · tareas diferentes", "El F1 de tipos no sustituye al F1 de las tres clases principales.", True)
    figures.bars("falsas_fallas", {"Normal → falla": matrix[0, 2]/matrix[0].sum() if matrix[0].sum() else None,
                  "Ambiental → falla": matrix[1, 2]/matrix[1].sum() if matrix[1].sum() else None},
                 "Falsas detecciones de falla", "Denominadores: objetivos normales y ambientales, respectivamente.", True)
    breakdown = result.get("breakdown", {})
    for key, title in (("internal_type_matrix", "Tipos candidatos"), ("published_type_matrix", "Tipos publicados")):
        if key in breakdown:
            data = breakdown[key]
            frame = pd.DataFrame(data["matrix"], index=[label(k) for k in data["rows"]],
                                 columns=[label(k) for k in data["columns"]])
            figures.heatmap(key, frame, title + " · matriz de confusión",
                            "Solo objetivos con tipo real conocido; última columna incluye fallas omitidas o sin tipo.")
    per_type = result["published"]["per_type"]
    figures.bars("tipos_soporte", {label(k): per_type[k]["tp"]+per_type[k]["fn"] for k in TYPES},
                 "Soporte por tipo de falla", "Se cuentan objetivos, no episodios independientes.")
    sensors = breakdown.get("target_entity_id", {})
    if sensors:
        frame = pd.DataFrame({sensor: {"F1 macro tipos": m["macro_f1"], "Recall falla": m["fault_recall"],
                            "Normal → falla": m["false_alarm_rate"]} for sensor, m in sensors.items()}).T
        if entity_ids is not None:
            outside = [sensor for sensor in frame.index if sensor not in entity_ids]
            frame.loc[outside, :] = np.nan
            frame = frame.rename(index={sensor: sensor + " [fuera del catálogo]" for sensor in outside})
        fig, ax = plt.subplots(figsize=(13, max(7, len(frame)*.42+3)))
        sns.heatmap(frame, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax)
        ax.tick_params(axis="y", rotation=0)
        figures.save("metricas_por_sensor", fig, frame, "Resultados por sensor",
                     "Normal → falla: menor es mejor. Fuera del catálogo: sin métrica representada.\nF1 macro heredado: incluye tipos sin soporte como cero; no comparar sin considerar los casos de cada sensor.")
    counts = result.get("flow", {}).get("counts", {})
    faults = int(matrix[2].sum())
    if counts and faults:
        figures.bars("recall_modelo_vs_flujo", {"Modelo aislado": matrix[2, 2]/faults,
                      "Flujo completo": counts["faults_detected_final"]/faults},
                     "Recall de falla · modelo frente al flujo", f"Mismo denominador: {faults} objetivos de falla.", True)
        figures.bars("flujo_fallas", {"Fallas reales": faults,
                      "Llegan al clasificador": faults-counts["faults_not_routed_to_classifier"],
                      "Detectadas al final": counts["faults_detected_final"]},
                     "Recorrido de las fallas en el flujo", "Los filtros pueden impedir que una falla llegue al clasificador.")
    alerts = breakdown.get("false_alerts_normal")
    if alerts:
        figures.bars("falsas_fallas_por_dia_sensor", {"Objetivos falsos / día de sensor":
                     alerts["false_fault_targets_per_observed_sensor_day"]},
                     "Falsas detecciones en objetivos normales base",
                     f"{alerts['false_fault_targets']} objetivos falsos / {alerts['observed_normal_sensor_days']} días de sensor.\nMuestreo base sin réplicas; no equivale a alertas operativas deduplicadas.")
    episodes = pd.DataFrame(breakdown.get("episodes", []))
    for field, title in (("detection_delay_seconds", "Demora de detección"),
                         ("classification_delay_seconds", "Demora de identificación del tipo")):
        if field not in episodes:
            continue
        values = episodes[field].dropna()/60
        if len(values):
            fig, ax = plt.subplots(figsize=(11, 6))
            sns.ecdfplot(x=values, ax=ax)
            ax.set(xlabel="Minutos desde la intervención", ylabel="Fracción acumulada de episodios con demora conocida")
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            figures.save(field, fig, episodes[["scenario_id", "intervention_id", field]], title,
                         f"Demora conocida: {len(values)} de {len(episodes)} episodios.\nLos restantes no se dibujan como demora cero; pueden no haberse detectado o carecer de fecha inicial.")
    return result


def plot_training(path, figures, run):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.startswith("models/") and n.endswith(".json")]
            if len(names) != 1:
                raise ValueError("El ZIP debe contener exactamente un manifiesto en models/")
            model = json.loads(archive.read(names[0]))
    else:
        model = read_json(path)
    metrics = model.get("metrics", {})
    seed = model.get("config", {}).get("seed")
    if seed is not None and run.split(":")[0].rsplit("-", 1)[-1] != str(seed):
        raise ValueError("La semilla del modelo no coincide con la ejecución elegida")
    experiment = metrics.get("experiment")
    if isinstance(experiment, str) and experiment != run.split("-", 1)[0]:
        raise ValueError("El experimento del modelo no coincide con la ejecución elegida")
    history = pd.DataFrame(metrics.get("history", []))
    if history.empty:
        return {"status": "sin historial"}
    validation = metrics.get("fault_type_validation", {})
    for key, values, title in (
        ("respaldo_entrenamiento", metrics.get("fault_type_train_episode_support", {}), "Respaldo de tipos en entrenamiento"),
        ("respaldo_validacion", validation.get("independent_episodes_by_type", {}), "Respaldo independiente de tipos en validación"),
        ("recall_tipos_validacion", validation.get("recall_by_type", {}), "Recall de tipos registrado en validación"),
    ):
        if values:
            figures.bars(key, {label(k): values.get(k) for k in TYPES}, title,
                         "Publicar el tipo exige al menos 3 contextos de entrenamiento, 3 de validación y recall ≥ 50 %.",
                         percent=key == "recall_tipos_validacion")
    for field, title in (("train_loss", "Pérdida de clasificación en entrenamiento"),
                         ("train_type_loss", "Pérdida de tipos en entrenamiento"),
                         ("dev_loss", "Pérdida de clasificación en validación"),
                         ("dev_macro_f1", "F1 macro de clases en validación")):
        if field not in history:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.lineplot(data=history, x="epoch", y=field, marker="o", color="#287e8e", ax=ax)
        selected = metrics.get("selected_epoch")
        if selected is not None:
            ax.axvline(selected, ls="--", color="#be563d", label=f"Época seleccionada: {selected}")
            ax.legend()
        ax.set(xlabel="Época", ylabel="F1" if field == "dev_macro_f1" else "Pérdida registrada")
        if field == "dev_macro_f1":
            ax.set_ylim(0, 1)
        figures.save(field, fig, history[["epoch", field]], title,
                     "Pérdidas de entrenamiento y validación pueden usar ponderaciones diferentes.\nEstas curvas no demuestran por sí solas sobreajuste ni su causa.")
    return {"model_id": model.get("model_id"), "seed": seed, "sha256": sha256(path),
            "entity_ids": model.get("entity_ids")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON producido por la evaluación de escenarios")
    parser.add_argument("--run", default="C-20261030:none", help="Clave exacta en results; añadir :robustness para robustez")
    parser.add_argument("--model", type=Path, help="ZIP exportado o manifiesto JSON, para curvas de entrenamiento")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    report = read_json(args.report)
    if args.run not in report.get("results", {}):
        parser.error("Ejecución inexistente. Disponibles: " + ", ".join(report.get("results", {})))
    figures = Figures(args.output, args.run, args.dpi)
    training = plot_training(args.model, figures, args.run) if args.model else None
    result = plot_report(report, args.run, figures, training.get("entity_ids") if training else None)
    missing = ["Cobertura de entrada por escala y causas de abstención: requiere registros individuales.",
               "Calibración y curva error/cobertura: requiere probabilidades por observación.",
               "No se infiere que unavailable signifique falta de histórico."]
    manifest = {"run": args.run, "report_sha256": sha256(args.report), "training": training,
                "scope": "Visualización de resultados guardados; no ejecuta nueva inferencia ni entrenamiento.",
                "libraries": {"seaborn": sns.__version__, "matplotlib": matplotlib.__version__, "pandas": pd.__version__},
                "missing_or_limits": missing, "source_limitations": report.get("limitations", []),
                "figures": figures.items}
    (args.output/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output/"metricas_fuente.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Evidencia visual del modelo", "", f"Ejecución: `{args.run}`.", "", manifest["scope"], "",
             "Cada PNG incluye un CSV con sus valores. Las proporciones se guardan entre 0 y 1.", "",
             "No se suman semillas ni se inventan intervalos de confianza para ventanas dependientes.", ""]
    lines.extend("- " + item for item in missing)
    for item in figures.items:
        lines.extend(["", "## " + item["title"], "", item["note"], "", f"![{item['title']}]({item['png']})"])
    (args.output/"LEEME.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"png_count": len(figures.items), "output": str(args.output.resolve()), "run": args.run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
