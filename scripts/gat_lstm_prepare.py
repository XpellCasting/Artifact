"""Prepara un dataset GAT-LSTM con uno o varios CSV etiquetados de Desarrollo."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def report(percent, message):
    print(json.dumps({"progress": percent, "message": message}, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, action="append", required=True, help="Repite --csv para cada dataset de entrada")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--manifest", type=Path, help="Manifiesto de particiones ya existente")
    parser.add_argument("--development-start", help="Inicio inclusivo ISO con zona horaria")
    parser.add_argument("--development-end", help="Fin exclusivo ISO con zona horaria")
    parser.add_argument("--inspect-only", action="store_true", help="Verifica los CSV sin generar secuencias")
    parser.add_argument("--result", type=Path, help="Guarda el informe o manifiesto con dataset_id")
    args = parser.parse_args(argv)
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.preparation import prepare_csv_dataset
    result = prepare_csv_dataset(args.csv, config=Config.from_json(args.config) if args.config else None,
        review_dir=args.review_dir, manifest_path=args.manifest,
        development_start=args.development_start, development_end=args.development_end,
        inspect_only=args.inspect_only, progress=report)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result.with_name(args.result.name + ".tmp")
        temporary.write_text(output, encoding="utf-8")
        temporary.replace(args.result)
    print(output, end="", flush=True)
    return 0 if args.inspect_only or result["ready_for_training"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Preparación interrumpida; vuelve a ejecutar el comando para generar un dataset completo.", file=sys.stderr)
        raise SystemExit(130)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
