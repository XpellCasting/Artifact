"""Atomic compact experiment: immutable observations, labels and sparse overlays.

This format intentionally does not make scenario names into sensor identifiers.
No detector or learned input receives the label/audit tables.
"""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sqlite3
import tempfile
from time import monotonic

from app.modulos.clasificador.config import Config, parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.features import fit_preprocessor, _finite_number
from app.modulos.clasificador.interventions import altered_value
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.services.csv_etiquetado import read_label
from app.services.estados_validos import valid_state

VERSION = "synthetic-scenarios-v1"
COLUMNS = ("entity_id", "state", "last_changed", "es_sintetica", "tipo", "tipo_falla")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    os.replace(temp, path)


def resource_usage():
    try:
        import resource
        return {"peak_resident_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024}
    except ImportError:
        return {"peak_resident_memory_bytes":None,"memory_measurement":"resource.getrusage unavailable on this platform"}


class ClosingConnection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect(directory, readonly=False):
    path = Path(directory)/"observations.sqlite3"
    db = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, factory=ClosingConnection) if readonly else sqlite3.connect(path, factory=ClosingConnection)
    db.row_factory = sqlite3.Row
    return db


def partition(at, config):
    return next((name for name, bounds in config["partitions"].items()
                 if parse_utc(bounds[0]) <= at < parse_utc(bounds[1])), None)


def validate_settings(config):
    if config.get("version")!=VERSION or config.get("context_seconds")!=259200:
        raise ValueError("Se requiere el contrato de escenarios v1 con contexto de 72 horas")
    parts=("train_fit","dev_tune","test")
    if set(config.get("partitions",{}))!=set(parts) or set(config.get("quotas",{}))!=set(parts):
        raise ValueError("Declara train_fit, dev_tune y test con sus cuotas")
    previous=None
    for name in parts:
        start,end=map(parse_utc,config["partitions"][name])
        if start>=end or previous is not None and start!=previous:
            raise ValueError("Particiones no contiguas, solapadas o vacías")
        previous=end
        if any(type(config["quotas"][name].get(key)) is not int or config["quotas"][name][key]<1 for key in ("scenarios","episodes_per_type")):
            raise ValueError("Las cuotas deben ser enteros positivos")
    a,b=parse_utc(config["calibration_start"]),parse_utc(config["calibration_end"])
    lo,hi=map(parse_utc,config["partitions"]["train_fit"])
    if not lo<=a<b<=hi or b-a!=timedelta(days=14):
        raise ValueError("La referencia auxiliar debe ocupar catorce días dentro de entrenamiento")
    for family in ("durations","robustness_durations"):
        if set(config.get(family,{}))!=set(FAULT_TYPES): raise ValueError(f"{family}: declara los siete tipos")
        for kind,values in config[family].items():
            if not values or any(not isinstance(v,(int,float)) or not math.isfinite(v) or (v<0 if kind=="spike" else v<=0) for v in values):
                raise ValueError("Duraciones inválidas")
    if not config.get("robustness_intensities") or any(not math.isfinite(v) or v<=0 for v in config["robustness_intensities"]):
        raise ValueError("Intensidades de robustez inválidas")
    for sensor in config.get("sensors",{}).values():
        lo,hi=sensor.get("limits",[None,None])
        if any(v is not None and not math.isfinite(v) for v in (lo,hi)) or lo is not None and hi is not None and lo>hi:
            raise ValueError("Límites físicos inválidos")
    Config(**config.get("model",{}))


def rows_for(db, part, entity=None):
    return [dict(row) for row in db.execute("SELECT o.*,l.label,l.fault,l.certainty,l.origin,l.episode,l.eligible FROM observations o JOIN labels l USING(row_id) WHERE part=?"+
        (" AND entity=?" if entity else "")+" ORDER BY t,row_id", (part, entity) if entity else (part,))]


def _import(source, audit, db, config):
    db.executescript("""
        CREATE TABLE observations(row_id INTEGER PRIMARY KEY,entity TEXT,t TEXT,state TEXT,part TEXT);
        CREATE TABLE labels(row_id INTEGER PRIMARY KEY,label TEXT,fault TEXT,synthetic INTEGER,
                            origin TEXT,certainty TEXT,episode TEXT,eligible INTEGER);
        CREATE TABLE evidence_sources(evidence_hash TEXT PRIMARY KEY,source_references TEXT);
        CREATE TABLE label_evidence(row_id INTEGER PRIMARY KEY,evidence_hash TEXT);
        CREATE TABLE audit(entity TEXT,t TEXT,state TEXT,label TEXT,fault TEXT,origin TEXT,certainty TEXT,episode TEXT,evidence TEXT);
        CREATE INDEX audit_key ON audit(entity,t,state);
    """)
    with Path(audit).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            db.execute("INSERT INTO audit VALUES(?,?,?,?,?,?,?,?,?)", (row["entity_id"],parse_utc(row["last_changed"]).isoformat(),row["state"],row["tipo"],row["tipo_falla"],row["origen"],row["certeza"],row["episode_id"],row["evidencia"]))
    db.commit()
    counts, unmatched, previous, synthetic = Counter(), 0, {}, 0
    with Path(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("El CSV base debe conservar sus seis columnas canónicas")
        for number, row in enumerate(reader, 2):
            at = parse_utc(row["last_changed"])
            part = partition(at, config)
            if part is None:
                raise ValueError(f"Fila {number}: fecha fuera de las particiones congeladas")
            entity = row["entity_id"]
            if entity in previous and at < previous[entity]:
                raise ValueError(f"Fechas desordenadas: {entity}")
            previous[entity] = at
            label = read_label([row[k] for k in COLUMNS[3:]], [0, 1, 2])
            matches = db.execute("SELECT * FROM audit WHERE entity=? AND t=? AND state=?", (entity, at.isoformat(), row["state"])).fetchall()
            if len(matches) > 1 and len({tuple(r)[3:] for r in matches}) > 1:
                raise ValueError(f"Auditoría ambigua en {entity}/{at}")
            match = matches[0] if matches else None
            if match and (match["label"] != row["tipo"] or match["fault"] != row["tipo_falla"]):
                raise ValueError(f"Etiqueta/auditoría incompatibles en fila {number}")
            unmatched += match is None
            is_synthetic = label["es_sintetica"] == "true"
            synthetic += is_synthetic
            # This baseline is pre-injection. A synthetic row without a reversible
            # row-level audit remains context, never a new intervention substrate.
            eligible = bool(match and not is_synthetic and label["label"] == "normal" and valid_state(row["state"]))
            db.execute("INSERT INTO observations VALUES(?,?,?,?,?)", (number, entity, at.isoformat(), row["state"], part))
            db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?,?)", (number, label["label"],label["fault_type"],int(is_synthetic),match["origin"] if match else "unmatched",match["certainty"] if match else "unknown",match["episode"] if match else None,int(eligible)))
            if match and match["evidence"]:
                evidence_hash=digest(match["evidence"])
                db.execute("INSERT OR IGNORE INTO evidence_sources VALUES(?,?)",(evidence_hash,match["evidence"]))
                db.execute("INSERT INTO label_evidence VALUES(?,?)",(number,evidence_hash))
            counts[f"{part}:{label['label']}"] += 1
    db.executescript("CREATE INDEX observation_time ON observations(part,entity,t); DROP TABLE audit;")
    db.commit()
    db.execute("VACUUM")
    return {"label_counts":dict(counts),"unmatched_audit":unmatched,"preexisting_synthetic_rows_excluded":synthetic,
            "reconstruction":"base anterior a las inyecciones de inyecciones/; no se inventaron valores anteriores",
            "label_dependency":"Las exportaciones heredadas abarcan fechas posteriores a los cortes. Su revisión pudo usar contexto posterior; certeza estimada, no evaluación prospectiva confirmada."}


def _fit(db, config):
    events = []
    kinds, values = {}, defaultdict(set)
    for row in db.execute("SELECT o.* FROM observations o JOIN labels l USING(row_id) WHERE part='train_fit' AND eligible=1"):
        event = {"entity_id":row["entity"],"timestamp":row["t"],"state":row["state"]}
        events.append(event); values[row["entity"]].add(row["state"])
    for entity, states in values.items():
        declared = config.get("sensors", {}).get(entity, {}).get("kind")
        kinds[entity] = declared or ("continuous" if all(_finite_number(v) for v in states) else "multistate")
    cfg = Config(**config.get("model", {}))
    prep = fit_preprocessor(events, [], cfg, kinds)
    from app.modulos.clasificador.segments import fit_spec
    bounds = tuple(parse_utc(v) for v in config["partitions"]["train_fit"])
    graph = fit_spec(events, prep, cfg, bounds)
    for entity, info in prep["entities"].items():
        info["unit"] = config.get("sensors", {}).get(entity, {}).get("unit", "unverified")
    return prep, graph


def _candidates(rows, kind, duration, info, earliest, context_seconds):
    times = [parse_utc(r["t"]) for r in rows]
    bad_prefix = [0]
    for row in rows:
        bad_prefix.append(bad_prefix[-1]+int(not row["eligible"]))
    result = []
    last_bucket = None
    for i, (at, row) in enumerate(zip(times, rows)):
        bucket = int(at.timestamp())//21600
        if bucket == last_bucket or at < earliest or at-times[0] < timedelta(seconds=context_seconds):
            continue
        end = at+timedelta(seconds=duration)
        j = bisect_right(times, end)
        context_left = bisect_left(times, at-timedelta(seconds=context_seconds))
        if i == j or j >= len(rows) or bad_prefix[j]-bad_prefix[context_left]:
            continue
        selected = rows[i:j]
        if kind == "spike":
            if i+1 >= len(rows) or times[i+1] <= at or not rows[i+1]["eligible"]:
                continue
            selected = rows[i:i+1]
        elif len(selected) < 3 or (times[j-1]-at).total_seconds() < duration*.5:
            continue
        if kind == "stuck" and len({r["state"] for r in selected}) < 2:
            continue
        last_bucket = bucket
        result.append((i,j))
    return result


def generate_scenario(db, part, name, config, prep, *, robust=False, candidate_cache=None, grouped=None):
    seed = int(digest([config["seed"],part,name])[:16], 16)
    rng = random.Random(seed)
    entities = prep["entities"]
    grouped = grouped or {e:rows_for(db,part,e) for e in entities}
    candidate_cache = candidate_cache if candidate_cache is not None else {}
    earliest = parse_utc(config["partitions"][part][0])+timedelta(seconds=config["context_seconds"])
    if part == "train_fit":
        earliest = max(earliest,parse_utc(config["calibration_end"]))
    reserved, usage, episodes, overlays, deficits = defaultdict(list), Counter(), [], [], []
    quota = config["quotas"][part]["episodes_per_type"]
    durations = config["robustness_durations"] if robust else config["durations"]
    def available(entity,a,b):
        return not any(a < right+timedelta(hours=72) and b > left-timedelta(hours=72) for left,right in reserved[entity])
    for kind in ("environmental_anomaly", *FAULT_TYPES):
        target = config.get("environmental_episodes",2) if kind == "environmental_anomaly" else quota
        generated = 0
        for repetition in range(target):
            duration = durations.get(kind,[3600])[repetition % len(durations.get(kind,[3600]))]
            candidates = []
            members = config.get("environmental_group", {}).get("entities", []) if kind == "environmental_anomaly" else list(entities)
            if kind == "environmental_anomaly" and (len(members) < 2 or any(e not in entities or entities[e]["kind"] != "continuous" or entities[e].get("unit") != config["environmental_group"].get("unit") for e in members)):
                break
            for entity in members:
                info = entities[entity]
                if kind in {"spike","bias","drift","noise","environmental_anomaly"} and info["kind"] != "continuous":
                    continue
                if kind == "state_replace" and (info["kind"] == "continuous" or len(info.get("vocab",{})) < 2):
                    continue
                rows = grouped[entity]
                cache_key=(part,entity,kind,duration)
                if cache_key not in candidate_cache:
                    candidate_cache[cache_key]=_candidates(rows,kind,duration,info,earliest,config["context_seconds"]) if rows else []
                for i,j in candidate_cache[cache_key]:
                    a,b = parse_utc(rows[i]["t"]),parse_utc(rows[j-1]["t"])
                    if available(entity,a,b):
                        candidates.append((usage[entity],rng.random(),entity,i,j))
            if not candidates:
                continue
            if kind == "environmental_anomaly":
                feasible=[]
                for candidate in candidates:
                    _,_,entity,i,j=candidate
                    a=parse_utc(grouped[entity][i]["t"]); b=a+timedelta(seconds=duration)
                    okay=True
                    for e in members:
                        erows=grouped[e]; ts=[parse_utc(r["t"]) for r in erows]
                        left,right=bisect_left(ts,a),bisect_right(ts,b)
                        context_left=bisect_left(ts,a-timedelta(hours=72))
                        if not ts or a-ts[0]<timedelta(hours=72) or right-left<3 or not available(e,a,b) or any(not r["eligible"] for r in erows[context_left:right]):
                            okay=False;break
                    if okay: feasible.append(candidate)
                candidates=feasible
                if not candidates: continue
            _,_,entity,i,j = min(candidates)
            chosen = [(entity,i,j)]
            rows = grouped[entity]; a = parse_utc(rows[i]["t"]); b = a+timedelta(seconds=duration)
            if kind == "environmental_anomaly":
                chosen = []
                for e in members:
                    erows = grouped[e]; ts = [parse_utc(r["t"]) for r in erows]
                    left,right = bisect_left(ts,a),bisect_right(ts,b)
                    if right-left < 3 or not available(e,a,b) or any(not r["eligible"] for r in erows[max(0,bisect_left(ts,a-timedelta(hours=72))):right]):
                        chosen = []; break
                    chosen.append((e,left,right))
                if not chosen:
                    continue
            intensity = config["robustness_intensities"][(repetition)%len(config["robustness_intensities"])] if robust else (.5,1.,1.5)[repetition%3]
            sign = -1 if repetition%2 else 1
            episode_seed = int(digest([seed,kind,repetition,entity,rows[i]["row_id"]])[:16],16)
            episode_id = digest([part,name,kind,repetition,episode_seed])[:32]
            changes = []
            for entity,left,right in chosen:
                info = entities[entity]; selected = grouped[entity][left:right]
                if kind == "spike": selected = selected[:1]
                amount = config.get("factors",{}).get(kind,1.)*info.get("scale",1.)*intensity*sign
                if kind == "environmental_anomaly": amount = config["environmental_group"]["amount_native_units"]*intensity*sign
                randomizer = random.Random(episode_seed)
                sensor = config.get("sensors",{}).get(entity,{})
                for row in selected:
                    progress = (parse_utc(row["t"])-a).total_seconds()/max(1,duration)
                    after = altered_value(kind,row["state"],amount=amount,progress=progress,initial=selected[0]["state"],categories=info.get("vocab",{}),rng=randomizer,limits=sensor.get("limits",[None,None]),shape=("quadratic" if kind == "drift" else "uniform") if robust else "linear")
                    if after != row["state"]:
                        changes.append({"row_id":row["row_id"],"entity":entity,"timestamp":row["t"],"before":row["state"],"after":after,"kind":kind,"intervention_id":episode_id,"label":"environmental_anomaly" if kind == "environmental_anomaly" else "sensor_fault","certainty":"synthetic_controlled","baseline_certainty":row["certainty"],"real_fault_confirmed":False})
            if not changes or any(not any(c["entity"] == e for c in changes) for e,_,_ in chosen):
                continue
            for e,_,_ in chosen:
                reserved[e].append((a,b)); usage[e] += 1
            overlays.extend(changes)
            episodes.append({"intervention_id":episode_id,"kind":kind,"entities":[e for e,_,_ in chosen],"start":a.isoformat(),"end":b.isoformat(),"seed":episode_seed,"intensity":intensity,"sign":sign,"duration_seconds":duration,"changed_rows":len(changes),"identifiable_from":None,"identifiability_reason":"No se estableció una regla causal verificable; no se usa la recuperación futura como entrada."})
            generated += 1
        if generated < target:
            deficits.append({"kind":kind,"requested":target,"generated":generated,"missing":target-generated,"reason":"insufficient_eligible_72h_normal_context_or_sensor_compatibility"})
    return {"scenario_id":name,"partition":part,"seed":seed,"robustness":robust,"episodes":episodes,"deficits":deficits,"overlays":overlays}


def prepare(source, audit, output, config, progress=print):
    started=monotonic()
    validate_settings(config)
    source,audit,output = Path(source),Path(audit),Path(output)
    if output.exists():
        raise ValueError("La salida ya existe; elige un directorio nuevo")
    expected = config.get("input_hashes",{})
    inputs = {"source":sha256_file(source),"audit":sha256_file(audit)}
    for key, actual in inputs.items():
        if expected.get(key) and expected[key] != actual:
            raise ValueError(f"Hash de entrada incompatible: {key}")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-",dir=output.parent) as temp:
        temp = Path(temp)
        with connect(temp) as db:
            progress("Verificando observaciones, etiquetas y procedencia")
            report = _import(source,audit,db,config)
            progress("Ajustando preprocesador exclusivamente sobre normalidad elegible de febrero-marzo")
            prep,graph = _fit(db,config)
            write_json(temp/"preprocessor.json",prep); write_json(temp/"graph.json",graph)
            scenarios=[]
            for part,quota in config["quotas"].items():
                grouped={e:rows_for(db,part,e) for e in prep["entities"]}
                candidate_cache={}
                for index in range(quota["scenarios"]):
                    name=f"{part}-{index+1:02}"
                    progress(f"Generando {name}")
                    scenario=generate_scenario(db,part,name,config,prep,candidate_cache=candidate_cache,grouped=grouped)
                    write_json(temp/"scenarios"/f"{name}.json",scenario)
                    scenarios.append({k:scenario[k] for k in ("scenario_id","partition","deficits")})
            for part in ("dev_tune","test"):
                name=f"robustness-{part}"
                scenario=generate_scenario(db,part,name,config,prep,robust=True)
                write_json(temp/"scenarios"/f"{name}.json",scenario)
                scenarios.append({k:scenario[k] for k in ("scenario_id","partition","deficits")})
        write_json(temp/"config.json",config)
        report.update(elapsed_seconds=monotonic()-started,resources=resource_usage())
        write_json(temp/"report.json",report)
        if sha256_file(source) != inputs["source"] or sha256_file(audit) != inputs["audit"]:
            raise ValueError("Las entradas cambiaron durante la preparación")
        files={p.relative_to(temp).as_posix():sha256_file(p) for p in temp.rglob("*") if p.is_file()}
        manifest={"version":VERSION,"status":"prepared_compact","inputs":inputs,"source_name":source.name,"audit_name":audit.name,
                  "files":files,"scenarios":scenarios,"config_sha256":digest(config),"training_executed":False,
                  "test_policy":"Mayo reservado para esta evaluación temporal; pudo usarse en trabajos previos.",
                  "next_step":"prepare-sequences: materializar ventanas y reproducir filtros reales en el entorno ROCm"}
        write_json(temp/"manifest.json",manifest)
        os.rename(temp,output)
    return manifest


def verify(directory):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        path=(directory/name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256_file(path) != expected:
            raise ValueError(f"Artefacto alterado: {name}")
    return manifest


def materialize(directory, scenario_name, output):
    directory,output=Path(directory),Path(output)
    verify(directory)
    if scenario_name not in {p.stem for p in (directory/"scenarios").glob("*.json")}:
        raise ValueError("Escenario no registrado")
    if output.exists(): raise ValueError("El CSV de salida ya existe")
    scenario=json.loads((directory/"scenarios"/f"{scenario_name}.json").read_text(encoding="utf-8"))
    overlays={r["row_id"]:r for r in scenario["overlays"]}
    with connect(directory,True) as db,output.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(COLUMNS)
        for row in rows_for(db,scenario["partition"]):
            change=overlays.get(row["row_id"])
            label=change["label"] if change else row["label"]
            writer.writerow([row["entity"],change["after"] if change else row["state"],row["t"],"true" if change else "false",{"normal":"normal","sensor_fault":"Falla","environmental_anomaly":"anomalia"}.get(label,"NULL"),change["kind"] if change and label=="sensor_fault" else row["fault"] or "NULL"])
