"""Compara variantes con predicciones ya guardadas sobre los mismos objetivos.

Cada --variant usa NOMBRE=RUTA_JSON. Los JSON contienen `predictions` con
sample_id, entity_id y scores [normal, ambiental, falla]; threshold y margin
opcionales congelan su política. Los objetivos omitidos se cuentan como abstención.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.GAT_LSTM.comparison import compare_prediction_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_json", type=Path, help="Manifiesto persistido devuelto por el constructor de dataset")
    parser.add_argument("--partition", choices=("dev_tune", "validation", "test"), default="validation")
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=JSON",
                        help="Repetible; por ejemplo B3=gat-evaluation.json")
    parser.add_argument("--output", type=Path, help="Ruta JSON de salida; stdout si se omite")
    args = parser.parse_args()
    variants = {}
    for value in args.variant:
        if "=" not in value:
            parser.error(f"Variante inválida: {value}")
        name, path = value.split("=", 1)
        if not name or name in variants:
            parser.error(f"Nombre de variante vacío o duplicado: {name}")
        variants[name] = Path(path)
    result = compare_prediction_files(json.loads(args.dataset_json.read_text(encoding="utf-8")),
                                      args.partition, variants)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__": main()
