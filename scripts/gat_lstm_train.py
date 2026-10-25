"""Entrena sin activar un candidato GAT-LSTM persistido."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.diagnostico import DiagnosticWorkbench
from app.modulos.clasificador.config import Config


def report(percent, message, **details):
    print(json.dumps({"percent": percent, "message": message, **details}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--config", type=Path, help="JSON opcional compatible con gat-lstm-segments-v1")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "data" / "review")
    parser.add_argument("--device", help="cpu, cuda o cuda:N; AMD/ROCm también usa cuda")
    parser.add_argument("--batch-size", type=int, help="Tamaño de lote; comenzar con 16 en RX 7800 XT")
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else None
    workbench = DiagnosticWorkbench(args.review_dir)
    if args.device is not None or args.batch_size is not None:
        # Preserve the dataset's preprocessing settings when overriding runtime options.
        config = config or Config(**workbench._read("datasets", args.dataset_id)["config"])
        overrides = {}
        if args.device is not None:
            overrides["device"] = args.device
        if args.batch_size is not None:
            overrides["batch_size"] = args.batch_size
        config = replace(config, **overrides)
    result = workbench.train_gat(args.dataset_id, None,
                                 config.as_dict() if config else None, report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
