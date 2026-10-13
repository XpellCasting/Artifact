"""Audita las particiones GAT-LSTM sin escribir en fuentes ni bases operativas."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.services.auditoria import audit_sources
from app.modulos.clasificador.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON de configuración GAT-LSTM")
    parser.add_argument("--manifest", type=Path, help="Manifiesto de particiones canónicas")
    parser.add_argument("--review-dir", type=Path, help="Directorio de revisión existente")
    parser.add_argument("--output", type=Path, required=True, help="JSON de salida fuera de las fuentes")
    args = parser.parse_args()
    config = Config.from_json(args.config) if args.config else Config()
    report = audit_sources(config, manifest_path=args.manifest, review_dir=args.review_dir).as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "aptitude": report["aptitude"],
                      "issues": len(report["issues"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
