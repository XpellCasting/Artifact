"""Package code and already prepared data; never prepare or train on import."""
import argparse
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))


def package(data,output):
    from app.modulos.clasificador.scenarios import verify
    from app.modulos.clasificador.digests import sha256_file
    data,output=Path(data).resolve(),Path(output).resolve()
    verify(data)
    if (data/"runs").exists() or (data/"candidates").exists():
        raise ValueError("Este paquete es una entrega previa al entrenamiento; para pesos usa exportar_candidato.sh")
    if output.exists(): raise ValueError("El paquete ya existe")
    files={}
    for subtree in (ROOT/"backend"/"app", ROOT/"tests"/"backend", ROOT/"scripts", ROOT/"configs"):
        for p in subtree.rglob("*"):
            if p.is_file() and p.suffix in {".py",".sh",".json"} and "__pycache__" not in p.parts:
                files["Artifact/"+p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/"backend").glob("requirements*.txt"):
        files["Artifact/backend/"+p.name]=p
    for name in ("ENTRENAMIENTO_ESCENARIOS.md","preparar_escenarios.sh","entrenar_experimento.sh","evaluar_experimento.sh","exportar_candidato.sh","limpiar_entrenamientos.sh"):
        files["Artifact/"+name]=ROOT/name
    for p in data.rglob("*"):
        if p.is_file(): files[f"Artifact/training/{data.name}/"+p.relative_to(data).as_posix()]=p
    inventory={name:{"sha256":sha256_file(p),"bytes":p.stat().st_size} for name,p in sorted(files.items())}
    manifest=json.dumps({"version":"scenario-training-delivery-v1","training_executed":False,"files":inventory},sort_keys=True,indent=2).encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+".tmp")
    try:
        with tarfile.open(temporary,"w:gz",compresslevel=6) as archive:
            for name,p in sorted(files.items()):
                info=archive.gettarinfo(str(p),arcname=name)
                info.mode=0o755 if p.suffix==".sh" else 0o644
                with p.open("rb") as stream: archive.addfile(info,stream)
            info=tarfile.TarInfo("PACKAGE_MANIFEST.json");info.size=len(manifest);info.mode=0o644
            archive.addfile(info,BytesIO(manifest))
        # Independently read the completed archive and verify payload hashes.
        import hashlib
        with tarfile.open(temporary,"r:gz") as archive:
            for member in archive:
                if member.name not in inventory: continue
                h=hashlib.sha256()
                with archive.extractfile(member) as stream:
                    while chunk:=stream.read(1024*1024): h.update(chunk)
                if h.hexdigest()!=inventory[member.name]["sha256"]: raise ValueError(f"Paquete corrupto: {member.name}")
        os.replace(temporary,output)
    finally: temporary.unlink(missing_ok=True)
    checksum=sha256_file(output)
    output.with_suffix(output.suffix+".sha256").write_text(checksum+"  "+output.name+"\n",encoding="utf-8")
    return {"path":str(output),"sha256":checksum,"bytes":output.stat().st_size,"files":len(files),"training_executed":False}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"training/escenarios_20261029_v1")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    print(json.dumps(package(args.data,args.output),indent=2))
