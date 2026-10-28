"""Export just the four candidate artifacts; never duplicate prepared windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile


def export_candidate(model_id, review_dir, output):
    if not re.fullmatch(r"[0-9a-f]{32}",model_id):
        raise ValueError("ID de modelo inválido")
    directory = Path(review_dir)/"diagnostics"
    manifest = json.loads((directory/"models"/f"{model_id}.json").read_text(encoding="utf-8"))
    weights = directory/"weights"/f"{model_id}.pt"
    with weights.open("rb") as stream:
        checksum = hashlib.file_digest(stream,"sha256").hexdigest()
    if manifest["model_id"] != model_id or checksum != manifest["weights_sha256"]:
        raise ValueError("Los pesos no coinciden con el manifiesto")
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix(output.suffix+".tmp")
    try:
        with zipfile.ZipFile(temporary,"w",compression=zipfile.ZIP_DEFLATED) as archive:
            for folder,suffix in (("models",".json"),("weights",".pt"),("preprocessors",".json"),("graphs",".json")):
                relative = Path(folder)/f"{model_id}{suffix}"
                archive.write(directory/relative,relative.as_posix())
        os.replace(temporary,output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"model_id":model_id,"zip":str(output),"activated":False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id")
    parser.add_argument("--review-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(export_candidate(args.model_id,args.review_dir,args.output)))


if __name__ == "__main__":
    main()
