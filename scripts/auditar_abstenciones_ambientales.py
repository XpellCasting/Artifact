"""Audita puntuaciones guardadas; no cambia umbrales, pesos ni etiquetas."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import zipfile


NAMES = ["Normal", "Anomalía ambiental", "Falla"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", default="C-20261030:none")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    report = json.loads(args.report.read_text(encoding="utf-8-sig"))
    original_report = args.records.parent / "report.json"
    if digest(original_report) != digest(args.report):
        raise ValueError("El informe de origen no coincide con la copia local")
    with zipfile.ZipFile(args.model) as archive:
        names = [n for n in archive.namelist() if n.startswith("models/") and n.endswith(".json")]
        if len(names) != 1:
            raise ValueError("Se requiere un único manifiesto de modelo")
        model = json.loads(archive.read(names[0]))
    run_path = args.records.parent.parent / "runs" / (args.run.split(":")[0] + ".json")
    run = json.loads(run_path.read_text())
    if run["model_id"] != model["model_id"]:
        raise ValueError("El modelo exportado no es el de esta ejecución")
    policy = model["rejection_policy"]
    matrix = [[0]*4 for _ in range(3)]
    for row in rows:
        scores = row["scores"]
        if row["evaluable"]:
            if len(scores) != 3 or any(not 0 <= s <= 1 for s in scores) or abs(sum(scores)-1) > 1e-5:
                raise ValueError("Puntuaciones inválidas")
            order = sorted(range(3), key=lambda i: (-scores[i], i))
            selected = order[0] if scores[order[0]] >= policy["threshold"] and scores[order[0]]-scores[order[1]] >= policy["margin"] else None
        else:
            if scores != [0, 0, 0]:
                raise ValueError("Caso no evaluable con puntuaciones inesperadas")
            selected = None
        if selected != row["predicted_label"]:
            raise ValueError("La política del modelo no reproduce las decisiones guardadas")
        matrix[row["true_label"]][selected if selected is not None else 3] += 1
    expected = report["results"][args.run]["classes"]
    if matrix != expected["matrix"] or len(rows) != expected["support"]:
        raise ValueError("Las predicciones no reproducen la matriz del informe")
    environmental = [row for row in rows if row["true_label"] == 1]
    abstentions = [row for row in environmental if row["predicted_label"] is None]
    detail = []
    for index, row in enumerate(abstentions, 1):
        valid = row["evaluable"]
        scores = row["scores"]
        winner = max(range(3), key=lambda k: scores[k]) if valid else None
        ranked = sorted(scores, reverse=True)
        outside = row["target_entity_id"] not in model["entity_ids"]
        detail.append({"id": index, "sample_id": row["sample_id"], "sensor": row["target_entity_id"],
                       "instante_utc": row["context_end"], "escenario": row["scenario_id"],
                       "origen_etiqueta": row["label_origin"], "certeza_etiqueta": row["label_certainty"],
                       "evaluable": valid, "fuera_catalogo": outside,
                       "p_normal": scores[0] if valid else None,
                       "p_ambiental": scores[1] if valid else None,
                       "p_falla": scores[2] if valid else None,
                       "clase_mayor_puntuacion": NAMES[winner] if valid else "No evaluable",
                       "p_maxima": max(scores) if valid else None,
                       "margen_observado": ranked[0]-ranked[1] if valid else None,
                       "umbral": policy["threshold"], "margen_exigido": policy["margin"],
                       "motivo": "no_evaluable_fuera_catalogo" if not valid and outside else
                                 "no_evaluable" if not valid else "puntuacion_bajo_umbral" if max(scores)<policy["threshold"] else "margen_insuficiente"})
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output/"detalle_abstenciones.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detail[0]))
        writer.writeheader()
        writer.writerows(detail)
    (output/"registros_ambientales_originales.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False)+"\n" for r in environmental), encoding="utf-8")
    counts = Counter(row["clase_mayor_puntuacion"] for row in detail)
    summary = {"run": args.run, "model_id": model["model_id"], "policy": policy,
               "observations_verified": len(rows), "environmental_total": len(environmental),
               "abstentions": len(detail), "counts": dict(counts),
               "reasons": dict(Counter(row["motivo"] for row in detail)),
               "unevaluable_sensors": dict(Counter(row["sensor"] for row in detail if not row["evaluable"])),
               "label_origin": dict(Counter(row["origen_etiqueta"] for row in detail)),
               "verified_class_matrix": matrix,
               "records_source": str(args.records), "records_sha256": digest(args.records),
               "report_sha256": digest(args.report), "model_zip_sha256": digest(args.model),
               "limits": ["Análisis de registros guardados; no es una nueva inferencia.",
                          "Los casos no evaluables tienen ceros de relleno, no probabilidades.",
                          "Son objetivos con ventanas y escenarios dependientes, no episodios independientes."]}
    (output/"resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    order = ["No evaluable", *NAMES]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=[counts[k] for k in order], y=order, hue=order, legend=False,
                palette=["#94a3b8", "#287e8e", "#369b70", "#be563d"], ax=ax)
    for i, key in enumerate(order):
        ax.text(counts[key]+.5, i, f"{counts[key]} ({counts[key]/len(detail):.1%})", va="center")
    ax.set_xlim(0, max(counts.values())*1.3)
    ax.set(xlabel="Casos", ylabel="", title="Anomalías ambientales: ¿qué había antes del rechazo?")
    fig.text(.03, .03, f"{args.run} · {len(detail)} abstenciones · Umbral: {policy['threshold']:.0%}\n"
             "Los no evaluables no tienen una clase preferida. Porcentajes sobre todas las abstenciones.", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, 1))
    fig.savefig(output/"abstenciones_desglose.png", dpi=180)
    plt.close(fig)
    valid = pd.DataFrame([r for r in detail if r["evaluable"]]).sort_values(["clase_mayor_puntuacion", "p_maxima"])
    frame = valid.set_index("id")[["p_normal", "p_ambiental", "p_falla"]]
    frame.columns = NAMES
    fig, ax = plt.subplots(figsize=(9, 13))
    sns.heatmap(frame, annot=True, fmt=".1%", cmap="Blues", vmin=0, vmax=1, ax=ax)
    ax.set(xlabel="Clase", ylabel="ID en detalle_abstenciones.csv", title=f"Puntuaciones de las {len(valid)} abstenciones evaluables")
    fig.text(.03, .02, "Los ceros de relleno de los casos no evaluables están excluidos.\nPuntuaciones softmax guardadas; no implican probabilidades calibradas.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output/"puntuaciones_abstenciones_evaluables.png", dpi=180)
    plt.close(fig)
    text = ["# Auditoría de abstenciones ambientales", "", f"Ejecución: {args.run}.", "",
            f"Se reprodujeron las decisiones y la matriz de los {len(rows)} registros usando la política exportada.", "",
            "| Resultado anterior al rechazo | Casos |", "|---|---:|"]
    text.extend(f"| {key} | {counts[key]} |" for key in order)
    text.extend(["", f"Casos no evaluables fuera del catálogo: {sum(not r['evaluable'] and r['fuera_catalogo'] for r in detail)}. Sus ceros son relleno.",
                 f"Rechazados por puntuación máxima bajo umbral: {summary['reasons'].get('puntuacion_bajo_umbral', 0)}. Umbral: {policy['threshold']}; margen exigido: {policy['margin']}.",
                 "No se modificaron umbrales ni se reentrenó el modelo. No inferir preferencia normal a partir de [0,0,0].", "",
                 "![Desglose](abstenciones_desglose.png)", "", "![Puntuaciones](puntuaciones_abstenciones_evaluables.png)", "",
                 "El CSV conserva sensor, instante, escenario, origen y certeza de etiquetas, puntuaciones y motivos."])
    (output/"LEEME.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
