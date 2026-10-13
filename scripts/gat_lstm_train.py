"""Entrena sin activar un candidato GAT-LSTM persistido."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.diagnostico import DiagnosticWorkbench
from app.modulos.clasificador.config import Config


def report(*args): print(json.dumps(args[0] if len(args) == 1 else args, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--config", type=Path, help="JSON opcional compatible con gat-lstm-segments-v1")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "data" / "review")
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else None
    result = DiagnosticWorkbench(args.review_dir).train_gat(args.dataset_id, None,
                                                            config.as_dict() if config else None,
                                                            report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
