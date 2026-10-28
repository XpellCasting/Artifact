"""Reproducible held-out interventions; no model predictions or fitting involved."""
from collections import Counter, defaultdict
import csv
from datetime import timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random

from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.services.csv_etiquetado import labelled_columns, read_label
from app.services.estados_validos import valid_state

VERSION="heldout_interventions_v1"


def evaluation_rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        reader=csv.DictReader(stream)
        names=[name.strip().lower() for name in reader.fieldnames or []]
        if len(set(names))!=len(names) or not {"entity_id","state","last_changed"}.issubset(names):
            raise ValueError("La evaluación requiere entity_id, state y last_changed")
        labelled=labelled_columns(names) is not None
        for number,raw in enumerate(reader,2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Evaluación, fila {number}: columnas inválidas")
            row=dict(zip(names,raw.values()))
            at=parse_utc(row["last_changed"]).astimezone(timezone.utc)
            if not row["entity_id"].strip():
                raise ValueError(f"Evaluación, fila {number}: entidad vacía")
            if not labelled:
                row.update(es_sintetica="false",tipo="NULL",tipo_falla="NULL")
            label=read_label([row[key] for key in ("es_sintetica","tipo","tipo_falla")],[0,1,2])
            yield number,list(dict.fromkeys([*names,"es_sintetica","tipo","tipo_falla"])),row,at,label,labelled


def _bounded(entity,value):
    if any(token in entity.lower() for token in ("illuminance","power","energy")):
        return max(0.,value)
    if any(token in entity.lower() for token in ("humidity","battery")):
        return min(100.,max(0.,value))
    return value


def generate_evaluation(csv_path,preprocessor,output,plan,*,seed=20261024):
    """Plan on clean/valid source blocks, then stream interventions to a copy.

    Unlabelled source rows remain NULL outside interventions: normality is never
    invented. Candidate normality from an unlabelled baseline is explicitly an
    assumption in the audit. Six-hour blocks and 72h guards avoid overlap.
    """
    path,output=Path(csv_path).resolve(),Path(output)
    if (output/"evaluation_injected.csv").exists():
        raise ValueError("La evaluación inyectada ya existe; usa una carpeta nueva")
    source_hash=sha256_file(path)
    entities=preprocessor["entities"]
    blocks={}; previous={}; first=last=None; labelled_source=None
    for number,names,row,at,label,labelled in evaluation_rows(path):
        first=min(first,at) if first else at;last=max(last,at) if last else at
        labelled_source=labelled
        entity=row["entity_id"]
        if entity in previous and at<previous[entity]:
            raise ValueError(f"Evaluación: fechas desordenadas de {entity}")
        previous[entity]=at
        if entity not in entities:
            continue
        bucket=int(at.timestamp())//21600
        key=(entity,bucket)
        if key not in blocks:
            blocks[key]=dict(entity=entity,bucket=bucket,start=at,end=at,count=0,eligible=True,
                             first_state=row["state"],varied=False,points=[])
        block=blocks[key];block["end"]=at;block["count"]+=1
        if entities[entity]["kind"]=="continuous":
            try:
                block["varied"] |= float(row["state"])!=float(block["first_state"])
            except ValueError:
                pass
        else:
            block["varied"] |= row["state"]!=block["first_state"]
        if len(block["points"])<3:
            block["points"].append((number,at))
        clean=label["label"]=="normal" if labelled else label["label"]=="uncertain"
        block["eligible"] &= clean and valid_state(row["state"])
        if entities[entity]["kind"]=="continuous":
            try:
                block["eligible"] &= math.isfinite(float(row["state"]))
            except ValueError:
                block["eligible"]=False
    if first is None:
        raise ValueError("CSV de evaluación vacío")
    eligible={key:b for key,b in blocks.items() if b["eligible"] and b["count"]>=3
              and b["start"]>=first+timedelta(hours=72) and b["end"]>b["start"]}
    rng=random.Random(seed); reserved=defaultdict(list); episodes=[]
    guard=timedelta(hours=72)

    def available(block):
        return not any(block["start"]<end+guard and block["end"]>start-guard
                       for start,end in reserved[block["entity"]])

    def add_episode(kind,chosen,amounts,replica):
        identity=hashlib.sha256(f"{source_hash}|{seed}|{kind}|{replica}|".encode()+
            "|".join(f"{b['entity']}:{b['bucket']}" for b in chosen).encode()).hexdigest()[:32]
        for block in chosen:
            reserved[block["entity"]].append((block["start"],block["end"]))
        episode=dict(episode_id=identity,kind=kind,entities=[b["entity"] for b in chosen],
            start=min(b["start"] for b in chosen).isoformat(),end=max(b["end"] for b in chosen).isoformat(),
            blocks={b["entity"]:dict(start=b["start"].isoformat(),end=b["end"].isoformat(),
                first_state=b["first_state"],spike_row=b["points"][1][0]) for b in chosen},
            amounts=amounts,seed=rng.randrange(2**31),rows=0,changed_rows=0,
            baseline_normality_basis="source_label_normal" if labelled_source else "unverified_valid_observations",
            evidence_origin="controlled_injection",real_fault_confirmed=False)
        episodes.append(episode)

    repetitions=int(plan.get("episodes_per_type",3))
    if repetitions<1:
        raise ValueError("episodes_per_type debe ser positivo")
    environmental_repetitions=int(plan.get("environmental_episodes",3))
    if environmental_repetitions<1:
        raise ValueError("environmental_episodes debe ser positivo")
    groups=plan.get("environmental_groups",[])
    if not groups:
        raise ValueError("Configura al menos un grupo físico para las anomalías ambientales")
    for group in groups:
        members=group["entities"]
        amount=float(group["amount_native_units"])
        if len(set(members))<2 or not math.isfinite(amount) or amount==0:
            raise ValueError("Una anomalía ambiental requiere al menos dos sensores relacionados y una magnitud finita no nula")
        if any(entities.get(e,{}).get("kind")!="continuous" for e in members):
            raise ValueError(f"Grupo ambiental {group.get('name')}: faltan sensores continuos compatibles en el ajuste")
        candidates=sorted({bucket for entity,bucket in eligible if entity==members[0]})
        rng.shuffle(candidates)
        for replica in range(environmental_repetitions):
            chosen=next(([eligible[e,bucket] for e in members] for bucket in candidates
                if all((e,bucket) in eligible and available(eligible[e,bucket]) for e in members)),None)
            if not chosen:
                raise ValueError(f"No hay suficientes bloques independientes para el grupo ambiental {group.get('name')}")
            add_episode("environmental_anomaly",chosen,dict.fromkeys(members,amount),replica)
    factors={"spike":6.,"bias":3.,"drift":3.,"noise":1.,**plan.get("scale_factors",{})}
    for kind in FAULT_TYPES:
        candidates=[]
        for (entity,bucket),block in eligible.items():
            info=entities[entity]
            if kind in {"spike","bias","drift","noise"} and info["kind"]!="continuous":
                continue
            if kind=="state_replace" and (info["kind"]=="continuous" or len(info.get("vocab",{}))<2):
                continue
            if kind=="stuck" and not block["varied"]:
                continue
            if kind=="spike" and block["points"][2][1]<=block["points"][1][1]:
                continue
            candidates.append(block)
        rng.shuffle(candidates)
        for replica in range(repetitions):
            block=next((b for b in candidates if available(b)),None)
            if block is None:
                raise ValueError(f"No hay {repetitions} bloques compatibles e independientes para {kind}; no se generó una evaluación incompleta")
            info=entities[block["entity"]]
            amount=float(factors.get(kind,1.))*float(info.get("scale",1.))
            if not math.isfinite(amount) or amount<=0:
                raise ValueError("La magnitud calibrada debe ser positiva y finita")
            add_episode(kind,[block],{block["entity"]:amount},replica)
    output.mkdir(parents=True,exist_ok=True)
    target=output/"evaluation_injected.csv";audit=output/"evaluation_interventions.csv"
    by_entity=defaultdict(list);randomizers={}
    for episode in episodes:
        randomizers[episode["episode_id"]]=random.Random(episode["seed"])
        for entity,block in episode["blocks"].items():
            by_entity[entity].append((parse_utc(block["start"]),parse_utc(block["end"]),episode,block))
    try:
        with target.open("w",encoding="utf-8",newline="") as destination,audit.open("w",encoding="utf-8",newline="") as trail:
            writer=None
            audit_writer=csv.DictWriter(trail,fieldnames=["episode_id","source_row","entity_id","last_changed","kind","original_state","injected_state","state_changed"])
            audit_writer.writeheader()
            for number,names,row,at,label,labelled in evaluation_rows(path):
                if writer is None:
                    writer=csv.DictWriter(destination,fieldnames=names);writer.writeheader()
                entity=row["entity_id"]
                active=next(((start,end,e,b) for start,end,e,b in by_entity[entity] if start<=at<=end),None)
                if active:
                    start,end,episode,block=active;kind=episode["kind"]
                    if kind!="spike" or number==block["spike_row"]:
                        before=row["state"];amount=episode["amounts"][entity]
                        progress=(at-start).total_seconds()/max(1.,(end-start).total_seconds())
                        if kind=="unavailable": after="unavailable"
                        elif kind=="stuck": after=block["first_state"]
                        elif kind=="state_replace":
                            categories=sorted(entities[entity]["vocab"])
                            after=next(value for value in categories if value!=before)
                        else:
                            delta=(randomizers[episode["episode_id"]].gauss(0,amount) if kind=="noise"
                                   else amount*progress if kind=="drift"
                                   else amount*(.25+.75*math.sin(math.pi*progress)) if kind=="environmental_anomaly" else amount)
                            value=_bounded(entity,float(before)+delta)
                            if not math.isfinite(value):
                                raise ValueError("La intervención produjo un valor no finito")
                            after=repr(value) if value!=float(before) else before
                        if entities[entity]["kind"]=="continuous" and valid_state(after) and float(after)==float(before):
                            after=before
                        row.update(state=after,es_sintetica="true",tipo="anomalia" if kind=="environmental_anomaly" else "Falla",
                                   tipo_falla="NULL" if kind=="environmental_anomaly" else kind)
                        episode["rows"]+=1;episode["changed_rows"]+=after!=before
                        audit_writer.writerow(dict(episode_id=episode["episode_id"],source_row=number,entity_id=entity,
                            last_changed=row["last_changed"],kind=kind,original_state=before,injected_state=after,state_changed=after!=before))
                writer.writerow(row)
        if any(e["changed_rows"]<1 for e in episodes) or sha256_file(path)!=source_hash:
            raise ValueError("Intervención sin cambios o CSV modificado durante la generación")
        report=dict(version=VERSION,seed=seed,source_sha256=source_hash,plan_config=plan,
            preprocessor_sha256=hashlib.sha256(json.dumps(preprocessor,sort_keys=True).encode()).hexdigest(),
            calibration="train_fit_preprocessor_or_prespecified_physical_scenario",model_predictions_used=False,
            source_labels_available=labelled_source,unlabelled_unchanged_rows_remain_NULL=not labelled_source,
            episode_counts=dict(Counter(e["kind"] for e in episodes)),episodes=episodes,
            output_sha256=sha256_file(target),audit_sha256=sha256_file(audit))
        (output/"evaluation_injections.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        return report
    except BaseException:
        target.unlink(missing_ok=True);audit.unlink(missing_ok=True)
        raise
