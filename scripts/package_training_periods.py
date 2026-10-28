"""Build the small Ubuntu code bundle; never include histories or models."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    name="Entrenamiento_2meses_balanceado_20261024"
    output=ROOT.parent/"outputs"/name
    output.mkdir(parents=True,exist_ok=True)
    files={str(path.relative_to(ROOT)).replace("\\","/"):path
           for path in (ROOT/"backend/app").rglob("*.py")}
    files.update({f"backend/{path.name}":path for path in (ROOT/"backend").glob("requirements*.txt")})
    files.update({f"scripts/{path.name}":path for path in (ROOT/"scripts").glob("gat_lstm_*.py")
                  if path.name!="gat_lstm_prepare_2plus2.py"})
    files.update({f"tests/backend/{path.name}":path for path in (ROOT/"tests/backend").glob("test_gat*.py")})
    fixture=ROOT/"tests/backend/test_segment_multiscale.py"
    if fixture.exists(): files["tests/backend/test_segment_multiscale.py"]=fixture
    for relative in ("preparar_2mas2.sh","entrenar_2mas2.sh","configs/evaluation_injections.json"):
        files[relative]=ROOT/relative
    files["README.md"]=ROOT/"ENTRENAMIENTO_2MAS2.md"
    files["training/README.md"]=ROOT/"ENTRENAMIENTO_2MAS2.md"
    inventory=[]
    for relative,source in sorted(files.items()):
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        with target.open("rb") as stream: checksum=hashlib.file_digest(stream,"sha256").hexdigest()
        inventory.append(dict(path=relative,bytes=target.stat().st_size,sha256=checksum))
    manifest=dict(version="balanced_period_training_v1",inputs="two_explicit_nonoverlapping_two_month_CSVs",
                  user_data_included=False,training_executed=False,files=inventory)
    (output/"code_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    archive=output.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as bundle:
        for relative in [*[item["path"] for item in inventory],"code_manifest.json"]:
            bundle.write(output/relative,f"{name}/{relative}")
    with archive.open("rb") as stream:checksum=hashlib.file_digest(stream,"sha256").hexdigest()
    archive.with_suffix(".sha256").write_text(f"{checksum}  {archive.name}\n",encoding="ascii")
    print(json.dumps(dict(zip=str(archive),bytes=archive.stat().st_size,files=len(inventory),sha256=checksum)))


if __name__=="__main__":
    main()
