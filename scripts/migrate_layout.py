"""Inventariar o migrar Artifact: ejecutar con la aplicación detenida."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.database.migration import migrate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    result = migrate(args.root, apply=args.apply, offline=args.offline)
    print(json.dumps(result, ensure_ascii=False, indent=2))
