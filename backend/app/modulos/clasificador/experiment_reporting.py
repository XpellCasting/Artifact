"""Auditable per-observation reports and production graph gate replay."""
from collections import Counter,defaultdict

from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.type_metrics import diagnostic_metrics


def breakdown(records,policy,supported):
    from app.modulos.clasificador.evaluation import decide
    report={}
    for field in ("target_entity_id","label_certainty","baseline_certainty","label_origin","intensity","duration_seconds","scenario_id"):
        groups=defaultdict(list)
        for row in records: groups[str(row.get(field))].append(row)
        report[field]={key:diagnostic_metrics(rows,policy["threshold"],policy["margin"],supported) for key,rows in groups.items()}
    normal=[r for r in records if r["scenario_id"].startswith("base-") and r["true_label"]==0]
    exposure={(r["target_entity_id"],r["context_end"][:10]) for r in normal}
    false=sum(r.get("predicted_label")==2 for r in normal)
    report["false_alerts_normal"]={"false_fault_targets":false,"observed_normal_sensor_days":len(exposure),
        "false_fault_targets_per_observed_sensor_day":false/len(exposure) if exposure else None,
        "scope":"Solo objetivos normales base sin réplicas; muestreo normal_stride_seconds, no tasa sobre todas las lecturas."}
    episodes=defaultdict(list)
    for row in records:
        if row.get("intervention_id"): episodes[row["scenario_id"],row["intervention_id"]].append(row)
    delays=[]
    for (scenario,episode),rows in episodes.items():
        ordered=sorted(rows,key=lambda r:r["context_end"])
        detected=[r for r in ordered if r["predicted_label"]==2]
        correct=[r for r in ordered if r["published_type"] is not None and r["published_type"]==r["true_type"]]
        start=next((r["start"] for r in ordered if r.get("start")),None)
        if ordered[0]["true_label"]!=2: continue
        delays.append({"scenario_id":scenario,"intervention_id":episode,"evaluated_rows":len(ordered),
            "detected_rows":len(detected),"correct_type_rows":len(correct),
            "detection_delay_seconds":(parse_utc(detected[0]["context_end"])-parse_utc(start)).total_seconds() if detected and start else None,
            "classification_delay_seconds":(parse_utc(correct[0]["context_end"])-parse_utc(start)).total_seconds() if correct and start else None})
    report["episodes"]=delays
    report["internal_type_matrix"]=_type_matrix(records,"candidate_type")
    report["published_type_matrix"]=_type_matrix(records,"published_type")
    report["faults_without_published_type"]=sum(r["predicted_label"]==2 and r["published_type"] is None for r in records)
    return report


def _type_matrix(records,key):
    from app.modulos.clasificador.model_constants import FAULT_TYPES
    matrix=[[0]*8 for _ in range(7)]
    for row in records:
        if row["true_type"] is not None:
            candidate=row.get(key) if row["predicted_label"]==2 else None
            matrix[row["true_type"]][candidate if candidate is not None else 7]+=1
    return {"matrix":matrix,"rows":list(FAULT_TYPES),"columns":[*FAULT_TYPES,"missed_or_unpublished_type"]}


def replay_gates(records,metadata,integrated):
    """Run the actual default FlowRunner graph with precomputed causal results.

    The filter outputs have already been produced chronologically by the real
    detectors. Only routing is replayed here; labels never enter the runner.
    """
    import pandas as pd
    from app.flow.runner import FlowRunner
    from app.flow.definitions import default_flow
    from app.flow.evidence import append_evidence
    row_lookup={r["sample_id"]:r for r in records}
    evidence_lookup={m["sample_id"]:m.get("filter_evidence",[]) for m in metadata}
    observations=[];identity={}
    for index,row in enumerate(records):
        observations.append({"entity_id":row["target_entity_id"],"timestamp_dt":pd.Timestamp(row["context_end"]),"state":"observed","_raw_index":index})
        identity[index]=row
    if not observations: return {"scope":"default_flow","count":0}
    class CachedRunner(FlowRunner):
        def _statistical(self,node_id,incoming,*args): return self._cached_filter(node_id,incoming,"statistical_focus")
        def _contextual(self,node_id,incoming,*args): return self._cached_filter(node_id,incoming,"contextual")
        def _cached_filter(self,node_id,incoming,family):
            output=[]
            for item in incoming:
                row=identity[item["source_raw_index"]]
                entries=[e for e in evidence_lookup[row["sample_id"]] if e.get("module_type")==family]
                if not entries: output.append(self._with_evidence(item,node_id,"unknown","not_evaluable",{}));continue
                current=item
                for entry in entries:
                    current=self._with_evidence(current,node_id,entry.get("signal","unknown"),entry.get("status","not_evaluable"),entry.get("data",{}))
                output.append(current)
            return output
        def _classifier(self,node_id,incoming,*args):
            output=[]
            for item in incoming:
                row=identity[item["source_raw_index"]];choice=row["predicted_label"]
                status="compatible" if choice==2 else "environmental_anomaly" if choice==1 else "no_fault_evidence" if choice==0 else "indeterminate"
                signal="positive" if choice==2 else "negative" if choice in (0,1) else "unknown"
                data={"predicted_label":("normal","environmental_anomaly","sensor_fault")[choice] if choice is not None else None,
                      "status":status,"fault_type":row["published_type"],"decision_policy":"integrated_classifier_v1" if integrated else "legacy",
                      "classifier_evaluable":row.get("evaluable",True)}
                output.append(self._with_evidence(item,node_id,signal,status,data))
            return output
    runner=CachedRunner(None,None,None)
    runner.execution_id="gates-replay"
    frame=pd.DataFrame(observations)
    # Same timestamp/sensor may occur in separate scenarios. Run each scenario
    # independently so parallel replicas never share a route or observation ID.
    outputs=[];counts=Counter()
    by_scenario=defaultdict(list)
    for index,row in identity.items(): by_scenario[row["scenario_id"]].append(index)
    for scenario,indices in by_scenario.items():
        values=frame[frame["_raw_index"].isin(indices)]
        _,result,_=runner.run(default_flow(),values,pd.DataFrame(),{},candidate_events=values,source_id=scenario,include_snapshot=False)
        for item in result["output"]:
            row=identity[item["source_raw_index"]]
            reached=any(e.get("module_type")=="classifier" for e in item["evidence"])
            counts["total"]+=1
            counts["reached_classifier"]+=reached
            counts["faults_not_routed_to_classifier"]+=row["true_label"]==2 and not reached
            counts["faults_detected_final"]+=row["true_label"]==2 and item["final_decision"]=="fault_compatible"
            counts[f"final:{item['final_decision']}"]+=1
    return {"scope":"production_default_flow_with_causal_filter_outputs","counts":dict(counts)}
