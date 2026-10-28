"""Prepare a two-calendar-month Development split; reserve the following two."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT/"local/entrenamiento_2mas2")
    parser.add_argument("--config",type=Path)
    parser.add_argument("--tune-days",type=int,default=14)
    parser.add_argument("--split-only",action="store_true")
    args = parser.parse_args(argv)
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.temporal_holdout import split_four_month_csv
    from app.modulos.clasificador.preparation import prepare_csv_dataset
    config = Config.from_json(args.config) if args.config else Config()
    output = args.output.resolve()
    report = split_four_month_csv(args.csv,output,tune_days=args.tune_days,
        episode_gap_seconds=max(config.labelled_stride_seconds*3, config.feature_window_seconds))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    if args.split_only:
        return 0
    dataset = prepare_csv_dataset([output/"development.csv"],config=config,review_dir=output/"review",
        manifest_path=output/"split_manifest.json",
        progress=lambda value,message:print(json.dumps(dict(progress=value,message=message)),flush=True))
    dataset["temporal_holdout"] = {key:report[key] for key in
        ("version","development_start","development_end","train_fit_end","evaluation_start","evaluation_end")}
    from app.modulos.clasificador.training_coverage import training_coverage
    dataset["training_coverage"] = training_coverage(dataset)
    dataset["coverage_requirements"] = {"all_fault_types_in_train_fit":True}
    dataset["ready_for_training"] = dataset["ready_for_training"] and dataset["training_coverage"]["ready"]
    from app.services.diagnostico import DiagnosticWorkbench, _write
    workbench = DiagnosticWorkbench(output/"review")
    _write(workbench._path("datasets",dataset["dataset_id"]),dataset)
    (output/"dataset.json").write_text(json.dumps(dataset,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(dict(dataset_id=dataset["dataset_id"],ready_for_training=dataset["ready_for_training"],
                         label_support=dataset["label_support"],coverage=dataset["training_coverage"],
                         evaluation_csv=str(output/"evaluation.csv")),indent=2),flush=True)
    return 0 if dataset["ready_for_training"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(f"Error: {exc}",file=sys.stderr)
        raise SystemExit(1)
