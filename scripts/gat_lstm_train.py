"""Entrena un candidato GAT-LSTM independiente en CPU/CUDA con checkpoints."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

def report(value, message, **details):
    print(json.dumps({"progress": value, "message": message, **details},
                     ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--config", type=Path, help="JSON con cambios sobre la configuración del dataset")
    parser.add_argument("--review-dir", type=Path, help="Espacio de datos alternativo; por defecto el de la aplicación")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda o cuda:N; por defecto auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, help="Por defecto storage/checkpoints/DATASET_ID")
    parser.add_argument("--resume", action="store_true", help="Continúa desde la última época completa")
    parser.add_argument("--result", type=Path, help="Guarda el resumen JSON del candidato")
    args = parser.parse_args(argv)

    # --help funciona incluso antes de instalar las dependencias del backend.
    from app.config import REVIEW_DATA_DIR, STORAGE_DIR
    from app.services.diagnostico import DiagnosticWorkbench
    from app.modulos.clasificador.config import Config

    review_dir = args.review_dir or REVIEW_DATA_DIR
    workbench = DiagnosticWorkbench(review_dir)
    dataset = workbench._read("datasets", args.dataset_id)
    overrides = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--config debe contener un objeto JSON")
    parameters = {**dataset["config"], **overrides}
    parameters["device"] = args.device or overrides.get("device", "auto")
    for name in ("batch_size", "max_epochs", "patience"):
        value = getattr(args, name)
        if value is not None:
            parameters[name] = value
    try:
        config = Config(**parameters)
    except ValueError as exc:
        raise ValueError(f"Dataset/configuración incompatible: {exc}. Prepara un dataset nuevo con la arquitectura de segmentos actual en la aplicación.") from exc
    default_checkpoints = (STORAGE_DIR if review_dir.resolve() == REVIEW_DATA_DIR.resolve()
                           else review_dir) / "checkpoints" / args.dataset_id
    result = workbench.train_gat(args.dataset_id, gat_config=config.as_dict(), progress=report,
        checkpoint_dir=args.checkpoint_dir or default_checkpoints, resume=args.resume)
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result.with_name(args.result.name + ".tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Entrenamiento interrumpido. Usa el mismo comando con --resume para continuar desde la última época guardada.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
