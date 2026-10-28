"""Verify a complete held-out intervention set before training or evaluating."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))


def verify(experiment):
    from app.modulos.clasificador.digests import sha256_file
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    experiment=Path(experiment)
    dataset=json.loads((experiment/"dataset.json").read_text(encoding="utf-8"))
    if not dataset["ready_for_training"]:
        raise ValueError("La cobertura de ajuste es insuficiente; revisa dataset.json")
    report=json.loads((experiment/"evaluation_injections.json").read_text(encoding="utf-8"))
    if any(report["episode_counts"].get(kind,0)<1 for kind in (*FAULT_TYPES,"environmental_anomaly")):
        raise ValueError("La evaluación no cubre todos los tipos y la anomalía ambiental")
    for filename,key in (("evaluation_injected.csv","output_sha256"),("evaluation_interventions.csv","audit_sha256")):
        if sha256_file(experiment/filename)!=report[key]:
            raise ValueError("La evaluación o su auditoría cambió: "+filename)
    preprocessor=json.loads((Path(dataset["sequence_dir"])/"preprocessor.json").read_text(encoding="utf-8"))
    checksum=hashlib.sha256(json.dumps(preprocessor,sort_keys=True).encode()).hexdigest()
    if checksum!=report["preprocessor_sha256"]:
        raise ValueError("Las inyecciones corresponden a otro preprocesador de ajuste")
    return {"status":"VERIFIED","dataset_id":dataset["dataset_id"],"episode_counts":report["episode_counts"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",type=Path)
    args=parser.parse_args()
    try:
        print(json.dumps(verify(args.experiment),indent=2))
    except (ValueError,KeyError,OSError,TypeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        raise SystemExit(1)
