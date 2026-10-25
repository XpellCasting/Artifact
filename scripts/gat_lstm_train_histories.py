"""Entrena el paquete verificado en el equipo del investigador, sin activar."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    from gat_lstm_verify_histories import verify
    report = verify(args.manifest, args.prepared)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if args.verify_only:
        return
    from dataclasses import replace
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.history_preparation import save_json
    from app.services.diagnostico import DiagnosticWorkbench
    workbench = DiagnosticWorkbench(args.prepared.resolve() / "review")
    dataset = workbench._read("datasets", report["dataset_id"])
    config = replace(Config(**dataset["config"]), device=args.device, batch_size=args.batch_size)
    def progress(percent, message, **details):
        print(json.dumps({"percent": percent, "message": message, **details}, ensure_ascii=False), flush=True)
    result = workbench.train_gat(report["dataset_id"], None, config.as_dict(), progress)
    save_json(args.prepared / "trained_model.json", result)
    # Se entrega el mismo formato ZIP que admite la interfaz de Artifact.
    import zipfile
    model_id = result["model_id"]
    output = args.prepared / f"modelo_7tipos_15sensores_{model_id}.zip"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder, suffix in (("weights", ".pt"), ("models", ".json"),
                               ("graphs", ".json"), ("preprocessors", ".json")):
            archive.write(workbench.directory / folder / f"{model_id}{suffix}", f"{folder}/{model_id}{suffix}")
    print(json.dumps({"model_id": model_id, "zip": str(output), "activated": False}), flush=True)


if __name__ == "__main__":
    main()
