"""Exporta/importa datasets y candidatos GAT-LSTM para otro equipo, sin activarlos."""
import argparse
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Empaqueta un dataset y, opcionalmente, su candidato")
    export.add_argument("--dataset-id", required=True)
    export.add_argument("--model-id")
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--review-dir", type=Path)
    restore = commands.add_parser("import", help="Verifica e importa el paquete; conserva la activación local")
    restore.add_argument("archive", type=Path)
    restore.add_argument("--review-dir", type=Path)
    args = parser.parse_args()
    try:
        from app.modulos.clasificador.transfer import export_bundle, import_bundle
        if args.command == "export":
            result = export_bundle(args.dataset_id, args.output, model_id=args.model_id, review_dir=args.review_dir)
        else:
            result = import_bundle(args.archive, review_dir=args.review_dir)
    except (ValueError, OSError, tarfile.TarError, ImportError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
