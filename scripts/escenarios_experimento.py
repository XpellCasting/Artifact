"""Prepare data or explicitly run a frozen synthetic-scenario experiment."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    prepare=commands.add_parser("prepare",help="Generar datos compactos; nunca entrena")
    prepare.add_argument("--source",type=Path,default=ROOT/"data/history/en_uso/experimento_v2/4mesesDesarrollo_15etiquetado.csv")
    prepare.add_argument("--audit",type=Path,default=ROOT/"data/history/en_uso/experimento_v2/4mesesDesarrollo_auditoria.csv")
    prepare.add_argument("--config",type=Path,default=ROOT/"configs/escenarios_20261029.json")
    prepare.add_argument("--output",type=Path,default=ROOT/"training/escenarios_20261029_v1")
    prepare.add_argument("--sequences",action="store_true",help="Materializar ventanas y reproducir los filtros reales")
    prepare.add_argument("--pilot",action="store_true",help="Piloto de preparación: dos objetivos por tipo y escenario, sin entrenar")
    prepare.add_argument("--test",action="store_true",help="Solo mayo; requiere selección congelada")
    for command in ("verify","train","freeze","evaluate","export","cleanup","materialize"):
        sub=commands.add_parser(command)
        sub.add_argument("--data",type=Path,default=ROOT/"training/escenarios_20261029_v1")
        if command=="train":
            sub.add_argument("--experiments",nargs="+",choices=list("ABCD"),default=list("ABCD"))
            sub.add_argument("--seeds",nargs="+",type=int)
            sub.add_argument("--device",default="cuda")
            sub.add_argument("--max-epochs",type=int)
            sub.add_argument("--resume",action="store_true")
            sub.add_argument("--freeze",action="store_true",help="Congelar después de completar los doce candidatos")
        elif command=="export":
            sub.add_argument("--run",required=True)
            sub.add_argument("--output",type=Path,required=True)
        elif command=="cleanup":
            sub.add_argument("--run",action="append",default=[])
            sub.add_argument("--apply",action="store_true")
        elif command=="materialize":
            sub.add_argument("--scenario",required=True)
            sub.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=="prepare" and sum((args.sequences,args.pilot,args.test))>1:
        raise ValueError("Usa solo una opción entre --sequences, --pilot y --test")
    from app.modulos.clasificador.scenarios import prepare as prepare_data,verify,materialize
    if args.command=="prepare":
        settings=json.loads(args.config.read_text(encoding="utf-8"))
        if args.output.exists():
            result=verify(args.output)
            from app.modulos.clasificador.scenarios import digest
            if result["config_sha256"]!=digest(settings): raise ValueError("La configuración difiere de los datos ya preparados")
        else: result=prepare_data(args.source,args.audit,args.output,settings)
        if args.test:
            from app.modulos.clasificador.experiment import verify_freeze
            verify_freeze(args.output)
        if args.sequences or args.pilot or args.test:
            from app.modulos.clasificador.scenario_sequences import prepare_sequences
            target=args.output/("pilot_sequences" if args.pilot else "test_sequences" if args.test else "sequences")
            if target.exists():
                from app.modulos.clasificador.experiment import verify_sequences
                result=verify_sequences(target,allow_pilot=args.pilot)
                print(f"Secuencias existentes verificadas: {target}")
            else:
                result=prepare_sequences(args.output,include_test=args.test,pilot=args.pilot)
    elif args.command=="verify":
        result=verify(args.data)
        print(json.dumps({"version":result["version"],"status":result["status"],"files_verified":len(result["files"]),"training_executed":False},ensure_ascii=False));return 0
    elif args.command=="materialize":
        result=materialize(args.data,args.scenario,args.output)
    else:
        from app.modulos.clasificador import experiment
        if args.command=="train":
            result=experiment.run_training(args.data,experiments=args.experiments,seeds=args.seeds,device=args.device,max_epochs=args.max_epochs,resume=args.resume)
            if args.freeze: experiment.freeze(args.data)
        elif args.command=="freeze": result=experiment.freeze(args.data)
        elif args.command=="evaluate": result=experiment.evaluate(args.data)
        elif args.command=="export": result=experiment.export_candidate(args.data,args.run,args.output)
        elif args.command=="cleanup": result=experiment.cleanup(args.data,args.run,args.apply)
    print(json.dumps(result,ensure_ascii=False,default=str,indent=2))
    return 0


if __name__=="__main__":
    try: raise SystemExit(main())
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr);raise SystemExit(1)
