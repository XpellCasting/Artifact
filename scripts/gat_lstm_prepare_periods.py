"""Prepare a known Development CSV and independent later evaluation interventions."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))


def inspect_period(path, *, development):
    from app.modulos.clasificador.temporal_holdout import _rows, month_start
    from app.modulos.clasificador.evaluation_injections import evaluation_rows
    from app.modulos.clasificador.digests import sha256_file
    first=last=None;months=set();labels=Counter();faults=Counter();rows=0
    iterator=_rows(path) if development else evaluation_rows(path)
    for item in iterator:
        _,_,row,at,label=item[:5]
        first=min(first,at) if first else at;last=max(last,at) if last else at
        months.add((at.year,at.month));rows+=1
        labels[label["label"]]+=1
        if label["label"]=="sensor_fault" and label["fault_type"]:
            faults[label["fault_type"]]+=1
    if first is None:
        raise ValueError(f"{path.name}: CSV vacío")
    expected={(month_start(first,offset).year,month_start(first,offset).month) for offset in range(2)}
    if months!=expected:
        raise ValueError(f"{path.name}: se esperan dos meses calendario consecutivos")
    return dict(path=str(path.resolve()),sha256=sha256_file(path),rows=rows,
                start=first.isoformat(),end=last.isoformat(),
                period_start=month_start(first).isoformat(),period_end=month_start(first,2).isoformat(),
                raw_label_counts=dict(labels),raw_fault_counts=dict(faults))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_csv",type=Path)
    parser.add_argument("evaluation_csv",type=Path)
    parser.add_argument("--output",type=Path,default=ROOT/"local/entrenamiento_2mas2")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--injection-plan",type=Path,default=ROOT/"configs/evaluation_injections.json")
    parser.add_argument("--tune-days",type=int,default=14)
    parser.add_argument("--seed",type=int,default=20261024)
    args=parser.parse_args(argv)
    from datetime import timedelta
    from app.modulos.clasificador.config import Config,parse_utc
    from app.modulos.clasificador.digests import sha256_file
    from app.modulos.clasificador.preparation import prepare_csv_dataset
    from app.modulos.clasificador.training_coverage import training_coverage
    from app.modulos.clasificador.evaluation_injections import generate_evaluation
    if not 4<=args.tune_days<=30:
        raise ValueError("--tune-days debe estar entre 4 y 30")
    development=inspect_period(args.development_csv,development=True)
    evaluation=inspect_period(args.evaluation_csv,development=False)
    if parse_utc(evaluation["period_start"])<parse_utc(development["period_end"]):
        raise ValueError("La evaluación debe ser posterior y no solaparse con Desarrollo")
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    absent=[kind for kind in FAULT_TYPES if not development["raw_fault_counts"].get(kind)]
    if absent:
        raise ValueError(f"El CSV de Desarrollo no contiene todos los tipos: {absent}")
    plan=json.loads(args.injection_plan.read_text(encoding="utf-8"))
    output=args.output.resolve()
    if output.exists():
        raise ValueError("La carpeta de salida ya existe; usa una carpeta nueva")
    output.mkdir(parents=True)
    tune_start=parse_utc(development["period_end"])-timedelta(days=args.tune_days)
    manifest=dict(boundaries=[development["period_start"],development["period_end"],evaluation["period_end"],evaluation["period_end"]],
                  train_fit_end=tune_start.isoformat(),partitions={})
    split_path=output/"split_manifest.json"
    split_path.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    split_report=dict(version="independent_2month_periods_v1",development=development,evaluation=evaluation,
                      train_fit_end=tune_start.isoformat(),evaluation_used_for_training=False)
    (output/"split_report.json").write_text(json.dumps(split_report,indent=2)+"\n",encoding="utf-8")
    config=Config.from_json(args.config) if args.config else Config()
    dataset=prepare_csv_dataset([args.development_csv],config=config,review_dir=output/"review",manifest_path=split_path,
        progress=lambda value,message:print(json.dumps(dict(progress=value,message=message)),flush=True))
    dataset["temporal_holdout"]={"version":split_report["version"],
        "development_start":development["period_start"],"development_end":development["period_end"],
        "train_fit_end":manifest["train_fit_end"],"evaluation_start":evaluation["period_start"],"evaluation_end":evaluation["period_end"]}
    dataset["coverage_requirements"]={"all_fault_types_in_train_fit":True}
    dataset["training_coverage"]=training_coverage(dataset)
    dataset["ready_for_training"] &= dataset["training_coverage"]["ready"]
    from app.services.diagnostico import DiagnosticWorkbench,_write
    workbench=DiagnosticWorkbench(output/"review")
    _write(workbench._path("datasets",dataset["dataset_id"]),dataset)
    (output/"dataset.json").write_text(json.dumps(dataset,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if not dataset["ready_for_training"]:
        print(json.dumps(dataset["training_coverage"],indent=2),flush=True)
        print("No se entrenará: revisa la cobertura dentro de train_fit y dev_tune. Los ejemplos de evaluación permanecen reservados.",file=sys.stderr)
        return 1
    if sha256_file(args.development_csv)!=development["sha256"] or sha256_file(args.evaluation_csv)!=evaluation["sha256"]:
        raise ValueError("Un CSV cambió durante la preparación")
    preprocessor=json.loads((Path(dataset["sequence_dir"])/"preprocessor.json").read_text(encoding="utf-8"))
    injections=generate_evaluation(args.evaluation_csv,preprocessor,output,plan,seed=args.seed)
    print(json.dumps(dict(dataset_id=dataset["dataset_id"],ready_for_training=True,
                         evaluation_injections=injections["episode_counts"],output=str(output)),indent=2),flush=True)
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        raise SystemExit(1)
