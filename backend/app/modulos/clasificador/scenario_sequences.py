"""Causal replay of production filters and disk-backed experiment sequences."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import tempfile
from time import monotonic

from app.modulos.clasificador import LABELS
from app.modulos.clasificador.config import Config, parse_utc
from app.modulos.clasificador.scenarios import connect, rows_for, verify, digest, write_json


def frame(rows, prep):
    """Strict observation whitelist. No synthetic flags, labels or episode IDs."""
    import pandas as pd
    columns=["entity_id","state","timestamp_dt","sensor_kind","value","_raw_index"]
    return pd.DataFrame([{"entity_id":r["entity"],"state":r["state"],"timestamp_dt":pd.Timestamp(r["t"]),
        "sensor_kind":prep["entities"][r["entity"]]["kind"],"value":float(r["state"]) if prep["entities"][r["entity"]]["kind"] == "continuous" and _number(r["state"]) else None,
        "_raw_index":r["row_id"]} for r in rows if r["entity"] in prep["entities"]],columns=columns)


def _number(value):
    from app.modulos.clasificador.features import _finite_number
    return _finite_number(value)


class FilterReplay:
    def __init__(self, directory, prep, config, progress=print):
        from app.services.reproduccion import MotorReproduccionValidacion
        from app.modulos.estadistico.filtro import FiltroEstadistico
        from app.modulos.contextual import FiltroContextualEspacioTemporal
        import pandas as pd
        self.prep=prep
        self.progress=progress
        with connect(directory,True) as db:
            reference=[dict(r) for r in db.execute("SELECT * FROM observations WHERE t>=? AND t<? ORDER BY t,row_id",(config["calibration_start"],config["calibration_end"]))]
        self.reference=frame(reference,prep)
        self.catalog=pd.DataFrame([{"entity_id":e,"sensor_kind":info["kind"],"unit_of_measurement":info.get("unit","")} for e,info in prep["entities"].items()])
        self.motor=MotorReproduccionValidacion(FiltroEstadistico())
        progress("Calibrando filtros reales únicamente con 1–14 febrero")
        self.baseline=self.motor.construir_baseline(self.reference,self.catalog,
            progress_callback=lambda phase,percent,message:progress(f"Referencia estadística {percent:.0f}%: {message}"))
        self.context=FiltroContextualEspacioTemporal()
        self.context_map=self.context.construir_mapa(self.reference,self.catalog,trigger="frozen_experiment_reference",
            progress_callback=lambda phase,percent,message:progress(f"Referencia contextual {percent:.0f}%: {message}"))
        # Runtime freshness must use the reference end, never wall-clock creation.
        self.context_map.setdefault("refresh",{})["reference_end"]=config["calibration_end"]
        # Hash stable calibration inputs/code, excluding generated timestamps/IDs.
        from app.modulos.clasificador.digests import sha256_file
        module_root=Path(__file__).resolve().parents[1]
        code={p.relative_to(module_root).as_posix():sha256_file(p) for family in ("contextual","estadistico") for p in (module_root/family).glob("*.py")}
        self.identity=digest([config["calibration_start"],config["calibration_end"],reference,code,
                              self.context.config.como_dict()])
        self.calibration={"sha256":self.identity,"reference_start":config["calibration_start"],"reference_end":config["calibration_end"],
                          "normality_basis":"Initial chronological observations; not externally confirmed normality. Labels were not supplied to the filters.",
                          "context_map":self.context_map,"statistical_models":self.baseline.get("continuous_model_manifest",{}),"code_sha256":code}

    def run(self, rows, targets, scenario_id):
        import pandas as pd
        from app.flow.evidence import append_evidence
        from app.modulos.clasificador.evidence_features import VERSION
        data=frame(rows,self.prep)
        baseline=deepcopy(self.baseline)
        context_state=self.context.preparar_eventos_causales(self.catalog,self.reference)
        context_state.bind_map(self.context_map)
        concurrence={}; output={}
        target_ids={r["row_id"] for r in targets}
        reference_end=self.calibration["reference_end"]
        data=data[data["timestamp_dt"]>=pd.Timestamp(reference_end)]
        for day_index,(_,day) in enumerate(data.groupby(data["timestamp_dt"].dt.floor("D"),sort=True)):
            if day_index%7==0: self.progress(f"Reproduciendo filtros {scenario_id}: {day['timestamp_dt'].min().date()}")
            # Reference period is not replayed a second time into the detector.
            if day.empty: continue
            stats=self.motor.analizar_lote(day,baseline,.6)
            detections={str(r.get("source_raw_index")):r for r in stats.get("detections",[])}
            temporal=baseline.get("last_observations",{})
            target_day=day[day["_raw_index"].isin(target_ids)]
            previous=day["timestamp_dt"].min()-pd.Timedelta(nanoseconds=1)
            for at,group in target_day.groupby("timestamp_dt",sort=True):
                observed=day[(day["timestamp_dt"]>previous)&(day["timestamp_dt"]<=at)]
                context_state.advance(observed,at)
                previous=at
                requests=[]; items={}
                for raw in group.to_dict("records"):
                    rid=raw["_raw_index"]; key=str(rid); entity=raw["entity_id"]
                    observation_id=f"{scenario_id}:{rid}"
                    item={"observation_id":observation_id,"entity_id":entity,"timestamp":at.isoformat(),"state":raw["state"],"value":raw["value"],"sensor_kind":raw["sensor_kind"],"evidence":[]}
                    detection=detections.get(key)
                    measurement=temporal.get(key,{})
                    evaluable=measurement.get("evaluable",entity in baseline.get(raw["sensor_kind"],{}))
                    positive=bool(measurement.get("detected_before_suppression",detection is not None))
                    measures={}
                    if "z" in measurement: measures["normalized_residual"]=measurement["z"]
                    measures.update(measurement.get("channels",{}))
                    if "joint_score" in measurement: measures["score_ratio"]=measurement["joint_score"]
                    details={**(detection or {}),"filter_features":{"version":VERSION,"measures":measures}}
                    details.update(measurement)
                    item=append_evidence(item,"statistical","statistical_focus","positive" if positive else "negative" if evaluable else "unknown","suspicion" if positive else "no_suspicion" if evaluable else "not_evaluable",details,"independent",scenario_id)
                    items[rid]=item
                    requests.append({**item,"is_suspicious":True,"passed_to_contextual_filter":True})
                contextual=self.context.analizar_causal(requests,day.iloc[:0],self.catalog,self.context_map,concurrence_state=concurrence,accept_all=True,prepared_state=context_state)
                by_id={r["observation_id"]:r for r in contextual.get("results",[])}
                for rid,item in items.items():
                    details=by_id.get(item["observation_id"],{})
                    outcome=details.get("context_outcome","not_evaluable")
                    signal="negative" if outcome=="supported" else "positive" if outcome=="contradicted" else "unknown"
                    item=append_evidence(item,"contextual","contextual",signal,outcome,details,"independent",scenario_id)
                    output[rid]=item["evidence"]
            tail=day[day["timestamp_dt"]>previous]
            if not tail.empty: context_state.advance(tail,tail["timestamp_dt"].max())
        return output


def target_rows(rows, config, part, *, scenario=None):
    earliest=parse_utc(config["partitions"][part][0])+timedelta(seconds=config["context_seconds"])
    if part=="train_fit": earliest=max(earliest,parse_utc(config["calibration_end"]))
    changes={r["row_id"]:r for r in (scenario or {}).get("overlays",[])}
    result=[]; last={}
    for row in rows:
        at=parse_utc(row["t"])
        if at<earliest: continue
        change=changes.get(row["row_id"])
        if scenario and not change:
            # Store affected past contexts too, with original targets. Never
            # duplicate clean normal windows which have no intervention in 72h.
            if not any(e["start"]<=row["t"] and parse_utc(e["end"])+timedelta(hours=72)>=at and row["entity"] in e["entities"] for e in scenario["episodes"]):
                continue
        label=change["label"] if change else row["label"]
        if label not in LABELS: continue
        episode=(change or {}).get("intervention_id") or row["episode"] or "unlinked"
        signature=(row["entity"],label,(change or {}).get("kind",row["fault"]),episode)
        stride=config["normal_stride_seconds"] if label=="normal" else config["target_stride_seconds"]
        if signature in last and (at-last[signature]).total_seconds()<stride: continue
        last[signature]=at
        result.append({**row,"state":change["after"] if change else row["state"],"label":label,
                       "fault":change["kind"] if change and label=="sensor_fault" else row["fault"]})
    return result


def original_contexts(rows, context_seconds=259200):
    """Connected components of intersecting contexts, shared across all sensors.

    Conservative by design: dense histories may form a single component. They
    cannot honestly certify three independent contexts through arbitrary bins.
    """
    mapping={}; end=None; identity=None
    for row in sorted(rows,key=lambda r:r["t"]):
        at=parse_utc(row["t"]); start=at-timedelta(seconds=context_seconds)
        if end is None or start>end:
            identity=digest(["overlap-component",row["part"],row["t"]])[:32]
        end=max(end,at) if end else at
        mapping[row["row_id"]]=identity
    return mapping


def prepare_sequences(directory, *, include_test=False, pilot=False, progress=print):
    """Materialize one scenario at a time. Test is a separate explicit action."""
    from app.modulos.clasificador.temporal import build_causal_sample
    from app.modulos.clasificador.storage import PartitionWriter
    directory=Path(directory); verify(directory)
    started=monotonic()
    source_history_id=verify_source(directory)
    config=json.loads((directory/"config.json").read_text(encoding="utf-8"))
    prep=json.loads((directory/"preprocessor.json").read_text(encoding="utf-8"))
    graph=json.loads((directory/"graph.json").read_text(encoding="utf-8"))
    cfg=Config(**config.get("model",{}))
    final=directory/("pilot_sequences" if pilot else "test_sequences" if include_test else "sequences")
    if final.exists(): raise ValueError(f"La salida ya existe: {final}")
    replay=FilterReplay(directory,prep,config,progress)
    if include_test:
        frozen_calibration=json.loads((directory/"sequences"/"calibration.json").read_text(encoding="utf-8"))
        def calibration_signature(value):
            context=value["context_map"]
            return digest([value["sha256"],value["statistical_models"],
                context.get("library_versions"),context.get("config"),context.get("edges"),context.get("decision_calibrations")])
        if calibration_signature(frozen_calibration)!=calibration_signature(replay.calibration):
            raise ValueError("La calibración auxiliar difiere de febrero–abril; conserva el mismo código y entorno. No se materializó mayo.")
    progress("Las secuencias se generan escenario por escenario; el manifiesto se publica al terminar")
    with tempfile.TemporaryDirectory(prefix=".sequences-",dir=directory) as temp:
        temp=Path(temp); inventory={}
        with connect(directory,True) as db:
            parts=("train_fit",) if pilot else ("test",) if include_test else ("train_fit","dev_tune")
            for part in parts:
                base=rows_for(db,part)
                scenarios=[None]+[json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory/"scenarios").glob(f"{part}-*.json"))]
                if pilot: scenarios=scenarios[1:2]
                if include_test: scenarios += [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory/"scenarios").glob("robustness-test.json"))]
                if not include_test and part=="dev_tune": scenarios += [json.loads((directory/"scenarios"/"robustness-dev_tune.json").read_text(encoding="utf-8"))]
                planned=[target_rows(base,config,part,scenario=s) for s in scenarios]
                contexts=original_contexts([r for targets in planned for r in targets],cfg.context_seconds)
                shape_bytes=36*len(prep["entity_ids"])*(103*4+8+1)
                estimate=sum(map(len,planned))*shape_bytes*2
                progress(f"{part}: {sum(map(len,planned))} ventanas; pico estimado {estimate/1024**3:.2f} GiB (arrays y temporales)")
                if shutil.disk_usage(temp).free<estimate+128*1024**2: raise OSError("Espacio insuficiente para materializar secuencias")
                for scenario,targets in zip(scenarios,planned):
                    name=scenario["scenario_id"] if scenario else f"base-{part}"
                    episode_parameters={e["intervention_id"]:e for e in (scenario or {}).get("episodes",[])}
                    if pilot:
                        by_kind=defaultdict(list)
                        for r in targets: by_kind[r["fault"] or r["label"]].append(r)
                        targets=[r for group in by_kind.values() for r in group[:2]]
                    overlays={r["row_id"]:r for r in (scenario or {}).get("overlays",[])}
                    events=[{**r,"state":overlays[r["row_id"]]["after"]} if r["row_id"] in overlays else r for r in base]
                    evidence=replay.run(events,targets,name)
                    grouped=defaultdict(list)
                    for r in events: grouped[r["entity"]].append(r)
                    timelines={e:[r["t"] for r in erows] for e,erows in grouped.items()}
                    def provider(start,end,entities):
                        values=[]
                        for entity in entities:
                            erows=grouped[entity];ts=timelines[entity]
                            left=bisect_left(ts,start.isoformat());right=bisect_right(ts,end.isoformat())
                            values.extend({"entity_id":entity,"state":r["state"],"timestamp":r["t"],"source_row":r["row_id"]} for r in erows[max(0,left-1):right])
                        return values
                    with PartitionWriter(temp,name) as writer:
                        for index,row in enumerate(targets):
                            def target_provider(a,b,entities): return provider(a,b,[row["entity"]])
                            sample=build_causal_sample(target_provider,row["t"],prep,graph,cfg,{row["entity"]:LABELS.index(row["label"])})
                            context=contexts[row["row_id"]]
                            sample["metadata"].update(sample_id=digest([name,row["row_id"]]),source_row=row["row_id"],source_history_id=source_history_id,source_id=source_history_id,scenario_id=name,
                                original_context_id=context,parent_episode_id=context,original_episode_id=row["episode"],
                                source_block_id=digest([part,int(parse_utc(row["t"]).timestamp())//259200])[:32],
                                intervention_id=overlays.get(row["row_id"],{}).get("intervention_id"),partition=part,
                                context_end=row["t"],target_entity_id=row["entity"],target_label=LABELS.index(row["label"]),fault_subtype=row["fault"],
                                label_certainty="synthetic_controlled" if row["row_id"] in overlays else row["certainty"],
                                baseline_certainty=row["certainty"],real_fault_confirmed=False,
                                label_origin="controlled_injection" if row["row_id"] in overlays else row["origin"],
                                evidence_origin="controlled_injection" if row["row_id"] in overlays else "historical_estimated",filter_evidence=evidence.get(row["row_id"],[]))
                            intervention=overlays.get(row["row_id"],{}).get("intervention_id")
                            sample["metadata"].update({key:episode_parameters.get(intervention,{}).get(key) for key in ("intensity","duration_seconds","start","identifiable_from")})
                            episode=episode_parameters.get(intervention,{})
                            block=int(parse_utc(episode.get("start",row["t"])).timestamp())//259200
                            sample["metadata"]["balance_episode_id"]=digest([source_history_id,part,
                                sorted(episode.get("entities",[row["entity"]])),block,
                                None if intervention else row["episode"]])[:32]
                            writer.append(sample)
                            if index%100==0: progress(f"{name}: {index}/{len(targets)}")
                        writer.finish()
                        inventory[name]={"windows":writer.count,"labels":dict(writer.labels),"original_contexts":{k:len(v) for k,v in writer.episodes.items()}}
                    del evidence,events,grouped
        write_json(temp/"graph.json",graph);write_json(temp/"preprocessor.json",prep)
        write_json(temp/"calibration.json",replay.calibration)
        from app.modulos.clasificador.digests import sha256_file
        from app.modulos.clasificador.scenarios import resource_usage
        write_json(temp/"manifest.json",{"version":"scenario-sequences-v1","pilot":pilot,"elapsed_seconds":monotonic()-started,"resources":resource_usage(),"filter_calibration_sha256":replay.identity,"inventory":inventory,
            "files":{p.name:sha256_file(p) for p in temp.iterdir() if p.is_file()}})
        os.rename(temp,final)
    return inventory


def verify_source(directory):
    return json.loads((Path(directory)/"manifest.json").read_text(encoding="utf-8"))["inputs"]["source"]
