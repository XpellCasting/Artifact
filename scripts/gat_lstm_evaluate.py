"""Evalúa un candidato congelado en Validación o Prueba."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.Diagnostico_Trabajo import DiagnosticWorkbench


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("dataset_id")
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "data" / "review")
    args = parser.parse_args()
    result = DiagnosticWorkbench(args.review_dir).evaluate_gat(args.model_id, args.dataset_id,
        args.partition, lambda progress, message: print(json.dumps({"progress": progress, "message": message}, ensure_ascii=False)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
