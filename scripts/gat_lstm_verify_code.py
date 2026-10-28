"""Verify the delivered code bundle before preparing any history."""
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"code_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path=(root/item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise SystemExit("Falta un archivo: "+item["path"])
        with path.open("rb") as stream:
            checksum=hashlib.file_digest(stream,"sha256").hexdigest()
        if checksum!=item["sha256"]:
            raise SystemExit("El archivo cambió: "+item["path"])
    print(f"Código íntegro: {len(manifest['files'])} archivos verificados")


if __name__=="__main__":
    main()
