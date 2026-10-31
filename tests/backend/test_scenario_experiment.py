import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import random
import sqlite3

import numpy as np
import unittest
import tempfile

from app.flow.evidence import append_evidence
from app.modulos.clasificador.evidence_features import encode_evidence, FEATURE_NAMES
from app.modulos.clasificador.interventions import altered_value
from app.modulos.clasificador.scenarios import generate_scenario, connect, verify
from app.modulos.clasificador.scenario_sequences import original_contexts,target_rows
from app.modulos.clasificador.type_metrics import diagnostic_metrics,select_type_policy


def evidence(family="statistical_focus",signal="positive",status="suspicion",data=None,node="f1"):
    item={"observation_id":"point","timestamp":"2026-04-05T00:00:00+00:00"}
    return append_evidence(item,node,family,signal,status,data or {},"combine_v1","run")["evidence"][0]


class ScenarioExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)

    def test_evidence_whitelist_masks_dedup_and_order(self):
        a=evidence(data={"z":0.,"tipo_falla":"spike","label":2,"scenario_id":"secret"})
        b=evidence("contextual","negative","supported",{"context_outcome":"supported","map_quality":{"stale":False}},"f2")
        encoded=encode_evidence([a,b,a])
        assert encoded==encode_evidence([b,a])
        assert encoded["mode"]=="both"
        assert len(encoded["values"])==len(FEATURE_NAMES)
        assert "secret" not in json.dumps(encoded)
        assert encoded["values"][FEATURE_NAMES.index("statistical_focus:normalized_residual:mask")]==1
        assert len(encoded["consumed_ids"])==2
        assert encode_evidence([a],observation_id="another")["mode"]=="none"
        assert encode_evidence([a],timestamp="future")["mode"]=="none"


    def test_four_execution_states_are_distinct(self):
        values=[encode_evidence(rows) for rows in ([],[evidence(signal="negative",status="no_suspicion")],
            [evidence(signal="unknown",status="not_evaluable")],[evidence()])]
        assert len({tuple(r["values"]) for r in values})==4
        assert [r["usable"] for r in values]==[False,True,False,True]
        stale=evidence("contextual","positive","contradicted",{"map_quality":{"stale":True}})
        assert not encode_evidence([stale])["usable"]


    def test_integrated_classifier_abstention_is_authoritative(self):
        item={"observation_id":"point","timestamp":"time"}
        item=append_evidence(item,"f1","statistical_focus","positive","suspicion",{},"combine_v1","run")
        data={"decision_policy":"integrated_classifier_v1","classifier_evaluable":True,"predicted_label":None}
        final=append_evidence(item,"f3","classifier","unknown","indeterminate",data,"combine_v1","run")
        assert final["combined_evaluation"]["label"]=="indeterminate"
        assert final["combined_evaluation"]["policy"]=="integrated_classifier_v1"
        data["classifier_evaluable"]=False
        fallback=append_evidence(item,"f3","classifier","unknown","indeterminate",data,"combine_v1","run")
        assert fallback["combined_evaluation"]["label"]=="suspicion"


    def test_operations_only_change_effective_rows(self):
        assert altered_value("drift","5",amount=3,progress=0)=="5"
        assert altered_value("bias","5",amount=-10,limits=(0,None))=="0"
        assert altered_value("state_replace","on",categories=["off","on"])=="off"
        assert altered_value("noise","5",rng=random.Random(3))==altered_value("noise","5",rng=random.Random(3))


    def test_subtype_metrics_include_missed_faults_and_wrong_false_positives(self):
        rows=[{"true_label":2,"true_type":0,"scores":[1.,0.,0.],"type_scores":[1.,0.,0.,0.,0.,0.,0.]}]
        report=diagnostic_metrics(rows)
        assert report["per_type"]["spike"]["fn"]==1
        rows.append({"true_label":0,"true_type":None,"scores":[0.,0.,1.],"type_scores":[1.,0.,0.,0.,0.,0.,0.]})
        report=diagnostic_metrics(rows)
        assert report["per_type"]["spike"]["fp"]==1


    def test_original_contexts_not_multiplied_by_replicas(self):
        rows=[{"row_id":i,"t":f"2026-04-{day:02}T00:00:00+00:00","part":"dev_tune"} for i,day in enumerate((4,5,9))]
        groups=original_contexts(rows+rows)
        assert groups[0]==groups[1]
        assert groups[2]!=groups[1]


    def test_connection_closes_before_atomic_publication(self):
        tmp_path=self.root
        with connect(tmp_path) as db:
            db.execute("CREATE TABLE x(v)")
        (tmp_path/"observations.sqlite3").rename(tmp_path/"moved.sqlite3")


    def test_seven_types_reproducible_and_no_unchanged_overlay(self):
        tmp_path=self.root
        start=datetime(2026,2,1,tzinfo=timezone.utc)
        prep={"entities":{**{f"n{i}":{"kind":"continuous","scale":1.} for i in range(4)},
                          **{f"c{i}":{"kind":"multistate","vocab":{"on":3,"off":4}} for i in range(2)}}}
        config={"seed":1,"context_seconds":259200,"partitions":{"train_fit":[start.isoformat(),(start+timedelta(days=40)).isoformat()]},
            "calibration_end":(start+timedelta(days=14)).isoformat(),"quotas":{"train_fit":{"episodes_per_type":1}},
            "durations":{k:[3600] for k in ("bias","noise","stuck","unavailable","state_replace")},"environmental_episodes":0}
        config["durations"].update(spike=[0],drift=[10800])
        with connect(tmp_path) as db:
            db.executescript("CREATE TABLE observations(row_id INTEGER PRIMARY KEY,entity,t,state,part); CREATE TABLE labels(row_id INTEGER PRIMARY KEY,label,fault,certainty,origin,episode,eligible);")
            rid=0
            for step in range(40*24*4):
                at=start+timedelta(minutes=15*step)
                for entity in prep["entities"]:
                    rid+=1;state=str(10+step%5) if entity.startswith("n") else ("on" if step%2 else "off")
                    db.execute("INSERT INTO observations VALUES(?,?,?,?,?)",(rid,entity,at.isoformat(),state,"train_fit"))
                    db.execute("INSERT INTO labels VALUES(?,?,?,?,?,?,?)",(rid,"normal",None,"estimada","test","base",1))
            a=generate_scenario(db,"train_fit","s1",config,prep)
            b=generate_scenario(db,"train_fit","s1",config,prep)
            assert a==b
            assert {r["kind"] for r in a["overlays"]}=={"spike","bias","drift","noise","stuck","unavailable","state_replace"}
            assert all(r["before"]!=r["after"] for r in a["overlays"])
            assert all(r["timestamp"]>=config["calibration_end"] for r in a["overlays"])


    def test_optional_branch_exact_no_evidence_and_frozen_backbone(self):
        import torch
        import torch_geometric
        from app.modulos.clasificador.config import Config
        from app.modulos.clasificador.network_segments import SegmentGATLSTM,OptionalEvidenceGATLSTM
        from app.modulos.clasificador.segments import FEATURE_NAMES as NAMES
        backbone=SegmentGATLSTM(len(NAMES),["continuous"],Config()).eval()
        model=OptionalEvidenceGATLSTM(backbone)
        x=torch.zeros(1,36,1,len(NAMES));state=torch.zeros(1,36,1,dtype=torch.long)
        present=torch.ones(1,36,1,dtype=torch.bool);adj=torch.ones(1,1,dtype=torch.bool)
        model.train()
        assert not model.backbone.training
        assert not any(p.requires_grad for p in model.backbone.parameters())
        with torch.no_grad():
            reference=backbone(x,state,adj,present)
            absent=model(x,state,adj,present,evidence=torch.randn(1,1,len(FEATURE_NAMES)),evidence_valid=torch.zeros(1,1,dtype=torch.bool))
        assert all(torch.equal(a,b) for a,b in zip(reference,absent))

    def test_optional_candidate_cpu_round_trip_without_training(self):
        import torch
        from app.modulos.clasificador.config import Config
        from app.modulos.clasificador.network_segments import SegmentGATLSTM,OptionalEvidenceGATLSTM
        from app.modulos.clasificador.segments import FEATURE_NAMES as NAMES
        from app.modulos.clasificador.training import save_candidate,load_candidate
        cfg=Config(architecture_version="gat_lstm_v4_optional_evidence")
        model=OptionalEvidenceGATLSTM(SegmentGATLSTM(len(NAMES),["continuous"],cfg))
        prep={"entity_ids":["sensor.x"],"entities":{"sensor.x":{"kind":"continuous"}},"feature_names":NAMES}
        manifest=save_candidate(model,prep,{"graph_version":"fixture"},cfg,self.root,{})
        loaded,_,_,meta=load_candidate(manifest["model_id"],self.root)
        assert meta["active"] is False
        assert isinstance(loaded,OptionalEvidenceGATLSTM)
        assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())

    def test_disjoint_support_ignores_replicas_and_overlapping_windows(self):
        from app.modulos.clasificador.context_support import disjoint_context_count
        values=[{"context_end":f"2026-04-{day:02}T00:00:00+00:00"} for day in (4,5,8,12)]
        assert disjoint_context_count(values*10)==3

    def test_unevaluable_targets_remain_in_complete_diagnosis(self):
        import torch
        from app.modulos.clasificador.config import Config
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        from app.modulos.clasificador.segments import FEATURE_NAMES as NAMES
        from app.modulos.clasificador.experiment import records_for
        model=SegmentGATLSTM(len(NAMES),["continuous"],Config())
        sample={"X":np.zeros((36,1,len(NAMES)),np.float32),"state_ids":np.zeros((36,1),np.int64),
            "node_present":np.zeros((36,1),bool),"y":[2],"label_mask":[False],"adjacency":[[True]],
            "metadata":{"sample_id":"target","parent_episode_id":"e","target_label":2,"fault_subtype":"spike"}}
        records=records_for(model,[sample],Config())
        assert len(records)==1 and records[0]["evaluable"] is False
        assert diagnostic_metrics(records)["per_type"]["spike"]["fn"]==1

    def test_production_gates_distinguish_missed_route_and_integrated_abstention(self):
        from app.modulos.clasificador.experiment_reporting import replay_gates
        rows=[{"sample_id":str(i),"true_label":2,"predicted_label":None,"published_type":None,
            "target_entity_id":"sensor.x","context_end":f"2026-04-05T0{i}:00:00+00:00","scenario_id":"scene"} for i in (0,1)]
        metadata=[{"sample_id":"0","filter_evidence":[evidence(signal="negative",status="no_suspicion")]},
                  {"sample_id":"1","filter_evidence":[evidence(),evidence("contextual","positive","contradicted",{},"f2")]}]
        result=replay_gates(rows,metadata,True)
        assert result["counts"]["faults_not_routed_to_classifier"]==1
        assert result["counts"]["final:indeterminate"]==1

    def test_orchestrator_resume_and_subset_cannot_freeze(self):
        from unittest.mock import patch,Mock
        from app.modulos.clasificador.experiment import run_training,freeze
        from app.modulos.clasificador.scenarios import write_json
        from app.modulos.clasificador.storage import PartitionWriter
        from app.modulos.clasificador.model_constants import FAULT_TYPES
        from app.modulos.clasificador.segments import FEATURE_NAMES as NAMES
        write_json(self.root/"manifest.json",{"version":"synthetic-scenarios-v1","files":{},"inputs":{}})
        write_json(self.root/"config.json",{"training_seeds":[20261029]})
        write_json(self.root/"preprocessor.json",{"entity_ids":["x"],"entities":{"x":{"kind":"continuous"}}})
        write_json(self.root/"graph.json",{"adjacency":[[True]]})
        seq=self.root/"sequences";seq.mkdir()
        write_json(seq/"graph.json",{"adjacency":[[True]]})
        for part in ("train_fit","dev_tune"):
            with PartitionWriter(seq,f"base-{part}") as writer:
                for index,fault in enumerate((None,None,*FAULT_TYPES)):
                    label=min(index,2)
                    writer.append({"X":np.zeros((36,1,len(NAMES))),"state_ids":np.zeros((36,1)),"node_present":np.ones((36,1)),"y":[label],"label_mask":[True],
                        "metadata":{"sample_id":str(index),"parent_episode_id":str(index),"fault_subtype":fault,"target_label":label}})
                writer.finish()
        write_json(seq/"manifest.json",{"pilot":False,"files":{}})
        metrics={"balancing":{"episode_support":{"normal":1,"environmental_anomaly":1,"sensor_fault":1}}}
        with patch("app.modulos.clasificador.training.train_classifier",return_value=(Mock(),metrics)) as train,patch("app.modulos.clasificador.training.save_candidate",return_value={"model_id":"a"*32}):
            result=run_training(self.root,experiments=["C"],device="cpu",progress=lambda *_:None)
            assert result[0]["active"] is False
            assert not (self.root/"runs"/"active.lock").exists()
            run_training(self.root,experiments=["C"],device="cpu",resume=True,progress=lambda *_:None)
            assert train.call_count==1
        with self.assertRaisesRegex(ValueError,"A–D"):
            freeze(self.root)

    def test_auxiliary_optimizer_only_updates_residual_on_toy_samples(self):
        import torch
        from app.modulos.clasificador.config import Config
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        from app.modulos.clasificador.segments import FEATURE_NAMES as NAMES
        from app.modulos.clasificador.model_constants import FAULT_TYPES
        from app.modulos.clasificador.training import train_classifier
        torch.set_num_threads(1)
        backbone=SegmentGATLSTM(len(NAMES),["continuous"],Config(dropout=0)).eval()
        original={k:v.clone() for k,v in backbone.state_dict().items()}
        samples=[]
        for i,fault in enumerate((None,None,*FAULT_TYPES)):
            label=min(i,2)
            samples.append({"X":np.zeros((36,1,len(NAMES)),np.float32),"state_ids":np.zeros((36,1),np.int64),
                "node_present":np.ones((36,1),bool),"adjacency":[[True]],"y":[label],"label_mask":[True],
                "metadata":{"sample_id":str(i),"parent_episode_id":str(i),"target_label":label,"fault_subtype":fault,
                            "filter_evidence":[evidence(data={"z":1.})]}})
        cfg=Config(architecture_version="gat_lstm_v4_optional_evidence",checkpoint_selection="type_macro_f1",max_epochs=1,batch_size=2,dropout=0)
        trained,metrics=train_classifier(samples,samples,cfg,[],["continuous"],backbone_model=backbone)
        assert all(torch.equal(v,trained.backbone.state_dict()[k]) for k,v in original.items())
        assert trained.residual[-1].weight.abs().sum()>0
        assert metrics["checkpoint_selection"]["primary"]=="type_macro_f1"

    def test_copy_and_duration_do_not_multiply_episode_weight(self):
        from app.modulos.clasificador.balancing import EpisodeBalance
        samples=[]
        for episode,count in (("same_source_episode",10),("another_episode",5)):
            for index in range(count):
                samples.append({"y":[2],"label_mask":[True],"metadata":{"source_history_id":"history",
                    "original_context_id":"overlapping_contexts","balance_episode_id":episode,
                    "scenario_id":str(index),"sample_id":f"{episode}-{index}"}})
        balance=EpisodeBalance(samples,require_all=False)
        weights=balance.target_weights(samples)
        assert abs(float(weights[:10].sum())-float(weights[10:].sum()))<1e-6
        assert balance.episode_support[2]==2
