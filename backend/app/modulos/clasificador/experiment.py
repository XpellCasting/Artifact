"""Sequential A–D orchestration. Importing this module never starts training."""
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import shutil
import tarfile
import tempfile
from time import monotonic

from app.modulos.clasificador.scenarios import verify, write_json, digest
from app.modulos.clasificador.digests import sha256_file


class JoinedPartitions(Sequence):
    def __init__(self, partitions):
        self.partitions=partitions
        self.ends=[]; total=0
        for part in partitions:
            total+=len(part); self.ends.append(total)
    def __len__(self): return self.ends[-1] if self.ends else 0
    def __getitem__(self,index):
        if isinstance(index,slice): return [self[i] for i in range(*index.indices(len(self)))]
        if index<0: index+=len(self)
        if not 0<=index<len(self): raise IndexError(index)
        part=bisect_right(self.ends,index)
        return self.partitions[part][index-(self.ends[part-1] if part else 0)]
    def summaries(self):
        for part in self.partitions: yield from part.summaries()


@contextmanager
def partitions(directory,part,augmented=True,robust=False):
    from app.modulos.clasificador.storage import DiskPartition
    directory=Path(directory)
    names=[f"base-{part}"]
    if augmented: names += [p.name.removesuffix("_metadata.jsonl") for p in sorted(directory.glob(f"{part}-*_metadata.jsonl"))]
    if robust: names=[f"robustness-{part}"]
    with ExitStack() as stack:
        parts=[stack.enter_context(DiskPartition(directory,name)) for name in names if (directory/f"{name}_metadata.jsonl").exists()]
        yield JoinedPartitions(parts)


def verify_sequences(directory, *, allow_pilot=False):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("pilot") and not allow_pilot:
        raise ValueError("El piloto no es un dataset completo de entrenamiento")
    for name,expected in manifest["files"].items():
        if sha256_file(directory/name)!=expected: raise ValueError(f"Secuencia alterada: {name}")
    return manifest


def records_for(model,samples,config,*,mode="both",device="cpu"):
    import torch
    from app.modulos.clasificador.training import _tensor_batch, forward_with_evidence, _batch_groups
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    model.to(device).eval(); result=[]
    with torch.no_grad():
        for indices in _batch_groups(list(range(len(samples))),config.batch_size):
            batch=[samples[i] for i in indices]
            x,states,present,y,mask,adjacency=_tensor_batch(batch,device)
            logits,types=forward_with_evidence(model,batch,x,states,adjacency,present,model.kind_ids,mode)
            scores=logits.softmax(-1).cpu().tolist(); types=types.softmax(-1).cpu().tolist()
            for i,sample in enumerate(batch):
                meta=sample["metadata"]
                active_nodes=[node for node,active in enumerate(sample["label_mask"]) if active]
                for node in active_nodes or [None]:
                    subtype=meta.get("fault_subtype")
                    label=sample["y"][node] if node is not None else meta["target_label"]
                    from app.modulos.clasificador.evidence_features import encode_evidence
                    encoded=encode_evidence(meta.get("filter_evidence",[]),mode=mode)
                    evidence_mode=encoded["mode"]
                    result.append({"sample_id":meta["sample_id"],"true_label":label,"scores":scores[i][node] if node is not None else [0.,0.,0.],"type_scores":types[i][node] if node is not None else [0.]*7,
                        "evaluable":node is not None,"evidence_mode":evidence_mode,"consumed_evidence_ids":encoded["consumed_ids"],"true_type":FAULT_TYPES.index(subtype) if subtype in FAULT_TYPES and label==2 else None,
                        "episode_id":meta.get("original_context_id",meta["parent_episode_id"]),
                        **{k:meta.get(k) for k in ("source_row","source_block_id","scenario_id","intervention_id","original_episode_id","label_certainty","baseline_certainty","label_origin","target_entity_id","context_start","context_end","intensity","duration_seconds","start","identifiable_from")}})
    return result


def mode_policy(records,train_support,mode=None):
    from app.modulos.clasificador.context_support import disjoint_context_count
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    from app.modulos.clasificador.type_metrics import select_type_policy, diagnostic_metrics
    if mode is not None:
        records=[r for r in records if r.get("evidence_mode")==mode]
    policy,score,supported=select_type_policy(records,train_support)
    internal=diagnostic_metrics(records,policy["threshold"],policy["margin"])
    return {"rejection_policy":policy,"diagnostic_metrics":score,"supported_types":supported,"available_observations":len(records),
        "fault_type_validation":{"independent_episodes_by_type":{kind:disjoint_context_count([r for r in records if r["true_type"]==i and r.get("evaluable",True)]) for i,kind in enumerate(FAULT_TYPES)},
                                 "recall_by_type":{kind:internal["per_type"][kind]["recall"] for kind in FAULT_TYPES}}}


def run_training(directory,*,experiments=("A","B","C","D"),seeds=None,device="cuda",max_epochs=None,resume=False,progress=print):
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.training import train_classifier,save_candidate,load_candidate
    directory=Path(directory);manifest=verify(directory)
    sequence_manifest=verify_sequences(directory/"sequences")
    if (directory/"freeze.json").exists(): raise ValueError("El experimento está congelado; utiliza otra carpeta para una nueva selección")
    settings=json.loads((directory/"config.json").read_text(encoding="utf-8"))
    prep=json.loads((directory/"preprocessor.json").read_text(encoding="utf-8"))
    graph=json.loads((directory/"graph.json").read_text(encoding="utf-8"))
    seeds=seeds or settings["training_seeds"]
    if any(s not in settings["training_seeds"] for s in seeds): raise ValueError("Semilla no predefinida")
    runs=directory/"runs";runs.mkdir(exist_ok=True)
    lock=runs/"active.lock"
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(descriptor,str(os.getpid()).encode());os.close(descriptor)
    try:
        results=[]
        for seed in seeds:
            for name in experiments:
                if name not in {"A","B","C","D"}: raise ValueError("Experimento desconocido")
                run_id=f"{name}-{seed}";result_path=runs/f"{run_id}.json"
                if result_path.exists():
                    if not resume: raise ValueError(f"La ejecución {run_id} ya terminó; usa --resume")
                    results.append(json.loads(result_path.read_text(encoding="utf-8")));continue
                cfg=Config(**{**settings.get("model",{}),"seed":seed,"device":device,"require_all_classes":False,"frozen_sensor_catalog":True,
                    "checkpoint_selection":"class_macro_f1" if name=="A" else "type_macro_f1",
                    "architecture_version":"gat_lstm_v4_optional_evidence" if name=="D" else "gat_lstm_v3_segments_multiscale"})
                if max_epochs: cfg=replace(cfg,max_epochs=max_epochs)
                backbone=None
                if name=="D":
                    prior=runs/f"C-{seed}.json"
                    if not prior.exists(): raise ValueError(f"D requiere C-{seed} terminado")
                    backbone=load_candidate(json.loads(prior.read_text(encoding="utf-8"))["model_id"],directory/"candidates")[0]
                progress(f"Entrenando {run_id}")
                started=monotonic()
                with partitions(directory/"sequences","train_fit",name in {"C","D"}) as train,partitions(directory/"sequences","dev_tune",name in {"C","D"}) as tune:
                    if name in {"C","D"}:
                        from app.modulos.clasificador.model_constants import FAULT_TYPES
                        observed={m["metadata"].get("fault_subtype") for m in train.summaries() if any(m["label_mask"])}
                        missing=set(FAULT_TYPES)-observed
                        labels={label for s in train.summaries() for label,active in zip(s["y"],s["label_mask"]) if active}
                        if missing or labels!={0,1,2}: raise ValueError(f"Cobertura insuficiente de C/D: tipos ausentes {sorted(missing)}, clases presentes {sorted(labels)}; no se moverán objetivos de abril/mayo al ajuste")
                    kinds=[prep["entities"][e]["kind"] for e in prep["entity_ids"]]
                    model,metrics=train_classifier(train,tune,cfg,[],kinds,
                        progress=lambda row: progress(json.dumps({"run":run_id,**row})),
                        checkpoint_dir=runs/run_id,resume=resume and (runs/run_id/"run.json").exists(),
                        checkpoint_identity={"experiment":run_id,"dataset":digest(manifest),"sequences":digest(sequence_manifest),
                                             "backbone_sha256":sha256_file(directory/"candidates"/"weights"/(json.loads(prior.read_text(encoding="utf-8"))["model_id"]+".pt")) if backbone is not None else None},backbone_model=backbone)
                    if name=="D":
                        metrics["evidence_mode_policies"]={mode:mode_policy(records_for(model,tune,cfg,mode=mode,device=device),metrics["fault_type_train_episode_support"],mode)
                            for mode in ("none","statistical","contextual","both")}
                        # None uses C's own frozen publication/rejection policy.
                        prior_manifest=load_candidate(json.loads(prior.read_text(encoding="utf-8"))["model_id"],directory/"candidates")[3]
                        metrics["evidence_mode_policies"]["none"]={"rejection_policy":prior_manifest["rejection_policy"],"fault_type_validation":prior_manifest["metrics"]["fault_type_validation"]}
                    metrics.update(experiment=name,seed=seed,dataset_contract=manifest["version"],elapsed_seconds=monotonic()-started,
                                   evidence_contract="optional-filter-evidence-v1",source_hashes=manifest["inputs"],
                                   training_class_limitations=[k for k,v in metrics["balancing"]["episode_support"].items() if v==0])
                    from app.modulos.clasificador.scenarios import resource_usage
                    metrics["resources"]=resource_usage()
                    candidate=save_candidate(model,prep,graph,cfg,directory/"candidates",metrics)
                    metadata=[s["metadata"] for group in (train,tune) for s in group.summaries()]
                    candidate.update(dataset_id=digest(manifest)[:32],source_hashes=manifest["inputs"],
                        entity_ids=prep["entity_ids"],entity_kinds=kinds,classes=["normal","environmental_anomaly","sensor_fault"],
                        training_end=max((m.get("context_end","") for m in metadata),default=None),
                        dataset_contract=manifest["version"],evidence_contract="optional-filter-evidence-v1",
                        backbone_model_id=json.loads(prior.read_text(encoding="utf-8"))["model_id"] if name=="D" else None)
                    write_json(directory/"candidates"/"models"/f"{candidate['model_id']}.json",candidate)
                result={"run_id":run_id,"model_id":candidate["model_id"],"experiment":name,"seed":seed,"elapsed_seconds":metrics["elapsed_seconds"],"active":False}
                write_json(result_path,result); results.append(result)
        return results
    finally:
        lock.unlink(missing_ok=True)


def freeze(directory):
    directory=Path(directory);verify(directory);verify_sequences(directory/"sequences")
    if (directory/"runs"/"active.lock").exists(): raise ValueError("Hay un entrenamiento activo")
    if (directory/"freeze.json").exists(): raise ValueError("El manifiesto de selección ya está congelado")
    runs=sorted((directory/"runs").glob("*.json"))
    if not runs: raise ValueError("No hay candidatos entrenados")
    settings=json.loads((directory/"config.json").read_text(encoding="utf-8"))
    expected={f"{name}-{seed}" for seed in settings["training_seeds"] for name in "ABCD"}
    if {p.stem for p in runs} != expected: raise ValueError("Congela después de completar A–D con las tres semillas predefinidas")
    files=[directory/"manifest.json",directory/"sequences"/"manifest.json",*runs]
    files += [p for p in (directory/"candidates").rglob("*") if p.is_file()]
    value={"version":"frozen-experiment-v1","files":{p.relative_to(directory).as_posix():sha256_file(p) for p in files},"selection_partition":"April","test_partition":"May","automatic_activation":False,
           "code_sha256":{p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
           "flow_code_sha256":{name:sha256_file(Path(__file__).parents[2]/name) for name in
                ("flow/evidence.py","flow/runner.py","flow/definitions.py","nodos/handlers.py","services/diagnostico.py")}}
    write_json(directory/"freeze.json",value);return value


def verify_freeze(directory):
    directory=Path(directory)
    frozen=json.loads((directory/"freeze.json").read_text(encoding="utf-8"))
    for name,expected in frozen["files"].items():
        if sha256_file(directory/name)!=expected: raise ValueError(f"Selección modificada tras congelar: {name}")
    for name,expected in frozen.get("code_sha256",{}).items():
        if sha256_file(Path(__file__).parent/name)!=expected: raise ValueError(f"Código modificado tras congelar: {name}")
    for name,expected in frozen.get("flow_code_sha256",{}).items():
        if sha256_file(Path(__file__).parents[2]/name)!=expected: raise ValueError(f"Flujo modificado tras congelar: {name}")
    return frozen


def evaluate(directory,progress=print):
    from app.modulos.clasificador.training import load_candidate
    from app.modulos.clasificador.config import Config
    from app.modulos.clasificador.evaluation import metrics as class_metrics,decide
    from app.modulos.clasificador.type_metrics import diagnostic_metrics
    directory=Path(directory);verify(directory);frozen=verify_freeze(directory)
    verify_sequences(directory/"test_sequences")
    output=directory/"evaluation"
    if output.exists(): raise ValueError("La evaluación final ya existe; no se selecciona de nuevo sobre mayo")
    reports={};all_records={}
    with tempfile.TemporaryDirectory(prefix=".evaluation-",dir=directory) as temp:
        temp=Path(temp)
        for path in sorted((directory/"runs").glob("*.json")):
            run=json.loads(path.read_text(encoding="utf-8"));identifier=run["model_id"]
            model,_,_,manifest=load_candidate(identifier,directory/"candidates")
            cfg=Config(**manifest["config"])
            modes=("none","statistical","contextual","both") if run["experiment"]=="D" else ("none",)
            for mode in modes:
                for robust in (False,True):
                    key=f"{run['run_id']}:{mode}"+(":robustness" if robust else "")
                    progress(f"Evaluando {key}")
                    inference_started=monotonic()
                    with partitions(directory/"test_sequences","test",True,robust) as samples:
                        records=records_for(model,samples,cfg,mode=mode)
                        metadata=[s["metadata"] for s in samples.summaries()]
                    policy=manifest["metrics"].get("evidence_mode_policies",{}).get(mode,{})
                    reject=policy.get("rejection_policy",manifest["rejection_policy"])
                    validation=policy.get("fault_type_validation",manifest["metrics"]["fault_type_validation"])
                    supported=[kind for kind,count in manifest["metrics"]["fault_type_train_episode_support"].items() if count>=3
                        and validation["independent_episodes_by_type"].get(kind,0)>=3 and (validation["recall_by_type"].get(kind) or 0)>=.5]
                    if run["experiment"]=="D":
                        for row in records:
                            actual=manifest["metrics"]["evidence_mode_policies"][row["evidence_mode"]]
                            row["rejection_policy"]=actual["rejection_policy"]
                            v=actual["fault_type_validation"]
                            row["supported_types"]=[kind for kind,count in manifest["metrics"]["fault_type_train_episode_support"].items() if count>=3 and v["independent_episodes_by_type"].get(kind,0)>=3 and (v["recall_by_type"].get(kind) or 0)>=.5]
                    internal=diagnostic_metrics(records,reject["threshold"],reject["margin"])
                    public=diagnostic_metrics(records,reject["threshold"],reject["margin"],supported)
                    reports[key]={"internal":internal,"published":public,"classes":class_metrics(records,reject["threshold"],reject["margin"]),"supported_types":supported,
                                  "inference_seconds":monotonic()-inference_started,
                                  "by_certainty":{certainty:diagnostic_metrics([r for r in records if r["label_certainty"]==certainty],reject["threshold"],reject["margin"],supported)
                                                  for certainty in sorted({r["label_certainty"] for r in records})}}
                    for row in records:
                        selected_policy=row.get("rejection_policy",reject)
                        choice=decide(row["scores"],selected_policy["threshold"],selected_policy["margin"])
                        row["predicted_label"]=choice
                        candidate=max(range(7),key=row["type_scores"].__getitem__) if row["evaluable"] else None
                        from app.modulos.clasificador.model_constants import FAULT_TYPES
                        row["candidate_type"]=candidate
                        row["published_type"]=candidate if choice==2 and candidate is not None and FAULT_TYPES[candidate] in row.get("supported_types",supported) else None
                        name=FAULT_TYPES[candidate] if candidate is not None else None
                        row["candidate_type_name"]=name
                        row["published_type_name"]=FAULT_TYPES[row["published_type"]] if row["published_type"] is not None else None
                        v=manifest["metrics"].get("evidence_mode_policies",{}).get(row["evidence_mode"],{}).get("fault_type_validation",validation)
                        reasons=[]
                        if not row["evaluable"]: reasons.append("insufficient_input_coverage")
                        elif choice!=2: reasons.append("general_class_not_sensor_fault")
                        else:
                            if manifest["metrics"]["fault_type_train_episode_support"].get(name,0)<3: reasons.append("fewer_than_three_original_train_contexts")
                            if v["independent_episodes_by_type"].get(name,0)<3: reasons.append("fewer_than_three_original_validation_contexts")
                            if (v["recall_by_type"].get(name) or 0)<.5: reasons.append("validation_recall_below_0.5")
                        row["fault_type_block_reasons"]=reasons
                    from app.modulos.clasificador.experiment_reporting import breakdown,replay_gates
                    reports[key]["breakdown"]=breakdown(records,reject,supported)
                    reports[key]["flow"]=replay_gates(records,metadata,run["experiment"]=="D")
                    all_records[key]=records
                    with (temp/f"{key.replace(':','_')}.jsonl").open("w",encoding="utf-8") as stream:
                        for row in records: stream.write(json.dumps(row)+"\n")
        from statistics import mean,pstdev
        seed_variation={}
        for name in "ABCD":
            for mode in (("none","statistical","contextual","both") if name=="D" else ("none",)):
                values=[value["published"]["macro_f1"] for key,value in reports.items() if key.startswith(name+"-") and key.endswith(":"+mode)]
                if values: seed_variation[f"{name}:{mode}"]={"macro_f1_by_seed":values,"mean":mean(values),"std":pstdev(values)}
        write_json(temp/"report.json",{"freeze_sha256":digest(frozen),"results":reports,"comparisons":cluster_comparisons(all_records),"seed_variation":seed_variation,
            "limitations":["La agrupación por contextos solapados puede dejar pocos grupos: no se infiere independencia de las réplicas.","No existe presupuesto operativo de falsas alarmas; ningún resultado activa un modelo."]})
        os.rename(temp,output)
    return reports


def cluster_comparisons(records):
    """Paired cluster bootstrap, keeping seeds and scenario replicas together."""
    from app.modulos.clasificador.type_metrics import diagnostic_metrics
    pairs=(("A","B","none"),("B","C","none"),*( ("C","D",mode) for mode in ("none","statistical","contextual","both")))
    results={};rng=random.Random(20261029)
    for left,right,right_mode in pairs:
        grouped=defaultdict(lambda:[[],[]])
        for side,exp in enumerate((left,right)):
            for key,rows in records.items():
                selected_mode="none" if side==0 else right_mode
                if not key.startswith(exp+"-") or not key.endswith(":"+selected_mode): continue
                for row in rows: grouped[row["episode_id"]][side].append(row)
        ids=sorted(grouped)
        def score(rows):
            counts=[[0,0,0] for _ in range(7)]
            for row in rows:
                true=row["true_type"];pred=row["published_type"]
                for k in range(7):
                    counts[k][0]+=true==k and pred==k
                    counts[k][1]+=pred==k and (true!=k if true is not None else row["true_label"]!=2)
                    counts[k][2]+=true==k and pred!=k
            return sum(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0. for tp,fp,fn in counts)/7
        delta=score([r for g in grouped.values() for r in g[1]])-score([r for g in grouped.values() for r in g[0]])
        differences=[]
        if len(ids)>=2:
            for _ in range(1000):
                sample=rng.choices(ids,k=len(ids))
                differences.append(score([r for k in sample for r in grouped[k][1]])-score([r for k in sample for r in grouped[k][0]]))
            differences.sort();ci=[differences[25],differences[974]]
        else: ci=None
        results[f"{left}_to_{right}:{right_mode}"]={"delta_macro_f1":delta,"cluster_count":len(ids),"ci95":ci,
            "result":"sin mejora" if delta<=0 else "mejora demostrada" if ci and ci[0]>0 else "resultado inconcluso"}
    return results


def export_candidate(directory,run_id,output):
    directory,output=Path(directory),Path(output);verify_freeze(directory)
    if output.exists(): raise ValueError("El paquete de salida ya existe")
    run=json.loads((directory/"runs"/f"{run_id}.json").read_text(encoding="utf-8"))
    identifier=run["model_id"]
    from app.modulos.clasificador.training import load_candidate
    load_candidate(identifier,directory/"candidates") # Verify CPU loading before packaging.
    from app.modulos.clasificador.evidence_features import VERSION,FEATURE_NAMES
    with tempfile.TemporaryDirectory(prefix=".candidate-",dir=directory) as temp:
        temp=Path(temp)
        for folder,suffix in (("models","json"),("weights","pt"),("preprocessors","json"),("graphs","json")):
            target=temp/folder;target.mkdir()
            shutil.copy2(directory/"candidates"/folder/f"{identifier}.{suffix}",target)
        for name in ("config.json","manifest.json","freeze.json"): shutil.copy2(directory/name,temp/name)
        write_json(temp/"evidence_contract.json",{"version":VERSION,"features":FEATURE_NAMES})
        write_json(temp/"candidate_manifest.json",{"run_id":run_id,"model_id":identifier,"active":False,
            "files":{p.relative_to(temp).as_posix():sha256_file(p) for p in temp.rglob("*") if p.is_file()}})
        output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix(output.suffix+".tmp")
        with tarfile.open(temporary,"w:gz") as archive:
            for p in sorted(temp.rglob("*")):
                if p.is_file(): archive.add(p,arcname=p.relative_to(temp).as_posix())
        os.replace(temporary,output)
    return {"path":str(output),"sha256":sha256_file(output),"active":False}


def cleanup(directory,run_ids=(),apply=False):
    directory=Path(directory).resolve();verify(directory)
    runs=directory/"runs";inventory=[]
    if apply and not run_ids: raise ValueError("--apply requiere identificadores explícitos --run")
    if (runs/"active.lock").exists(): raise ValueError("Hay una ejecución activa o un bloqueo pendiente de revisar")
    if (directory/"freeze.json").exists() and apply:
        # Checkpoints are reproducible and not part of the frozen model package.
        verify_freeze(directory)
    requested=run_ids or [p.name for p in runs.iterdir() if p.is_dir()] if runs.exists() else []
    import re
    for name in requested:
        if not re.fullmatch(r"[ABCD]-202610(?:29|30|31)",name): raise ValueError("Identificador de ejecución inválido")
        target=(runs/name).resolve()
        if target.parent!=runs or (runs/name).is_symlink(): raise ValueError("Ruta de limpieza fuera del directorio de ejecuciones")
        if not (runs/f"{name}.json").exists(): raise ValueError("Se protege un entrenamiento incompleto; reanúdalo antes de limpiar")
        size=sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.exists() else 0
        inventory.append({"run_id":name,"path":str(target),"reclaimable_bytes":size,"dependencies":"Solo checkpoints de ejecución terminada; candidato, datos, manifiestos y resultados se conservan.","applied":apply})
        if apply and target.exists(): shutil.rmtree(target)
    return inventory
