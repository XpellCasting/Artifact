import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.features import fit_preprocessor
from app.modulos.clasificador.pipeline import _provider
from app.modulos.clasificador.segments import ARCHITECTURE, SCHEMA, SCALES, FEATURE_NAMES, build_sample, fit_spec, intervals, categorical_dtw


def fixture():
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    config = Config(architecture_version=ARCHITECTURE, schema_version=SCHEMA, max_epochs=1, patience=1, dropout=0)
    rows = [{"entity_id": "a", "state": str(day+hour), "timestamp": (start+timedelta(days=day,hours=hour)).isoformat(), "source_row": day*24+hour}
            for day in range(16) for hour in range(9,19)]
    end = start+timedelta(days=16)
    prep = fit_preprocessor(rows, [], config, {"a": "continuous"})
    graph = fit_spec(rows, prep, config, (start,end))
    index = {"a": [{**r, "_time": datetime.fromisoformat(r["timestamp"])} for r in rows]}
    return config, prep, graph, _provider(index), start, end


class SegmentTests(unittest.TestCase):
    def test_night_keeps_last_state_without_inventing_records(self):
        config, prep, graph, provider, start, end = fixture()
        sample = build_sample(provider, end, prep, graph, config)
        self.assertEqual(len(sample["X"]), 36)
        self.assertEqual(len(sample["X"][0][0]), len(FEATURE_NAMES))
        self.assertTrue(all(c["usable"] for c in sample["metadata"]["scale_coverage"]["a"]))
        short = [r[0] for r in sample["X"][:12] if r[0][8]]
        self.assertTrue(short)
        self.assertTrue(all(r[6] == 0 for r in short))
        self.assertTrue(all(r[7] > 0 for r in short))
        self.assertTrue(prep["entities"]["a"]["hourly_profile"]["available"])
        self.assertEqual(prep["entities"]["a"]["hourly_profile"]["record_rate"][0], 0)

    def test_future_cannot_change_past_sample(self):
        config, prep, graph, provider, _, end = fixture()
        before = build_sample(provider,end,prep,graph,config)
        def contaminated(a,b,entities):
            return provider(a,b,entities)+[{"entity_id":"a", "timestamp":(end+timedelta(hours=1)).isoformat(),"state":"999999"}]
        after = build_sample(contaminated,end,prep,graph,config)
        self.assertEqual(before["X"], after["X"])

    def test_absence_invalid_and_stability_are_distinct(self):
        config, prep, graph, _, start, end = fixture()
        info = prep["entities"]["a"]
        stable = intervals([{"state":"1","timestamp":start.isoformat()}],end-timedelta(hours=1),end,60,info)
        invalid = intervals([{"state":"unavailable","timestamp":start.isoformat()}],end-timedelta(hours=1),end,60,info)
        unknown = intervals([],end-timedelta(hours=1),end,60,info)
        self.assertTrue(all(t["known"] == 1 and t["invalid"] == 0 for t in stable))
        self.assertTrue(all(t["known"] == 1 and t["invalid"] == 1 for t in invalid))
        self.assertTrue(all(t["known"] == 0 for t in unknown))

    def test_short_change_survives_hourly_aggregation(self):
        config,prep,_,_,start,_ = fixture()
        rows = [{"state":str(v),"timestamp":(start+timedelta(minutes=m)).isoformat()} for m,v in [(0,0),(10,100),(11,0)]]
        tick = intervals(rows,start,start+timedelta(hours=1),3600,prep["entities"]["a"])[0]
        self.assertEqual(tick["changes"],2)
        self.assertGreater(tick["max"],tick["numeric"])

    def test_categorical_distance_does_not_use_arbitrary_numeric_codes(self):
        self.assertEqual(categorical_dtw(["on","off"],["on","off"],.1),0)
        self.assertEqual(categorical_dtw(["on","off"],["open","closed"],.1),1)

    def test_model_gradients_entity_isolation_reload_and_runtime(self):
        import torch
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        from app.modulos.clasificador.training import _tensor_batch, save_candidate, load_candidate
        from app.modulos.clasificador.runtime import predict_at
        config,prep,graph,provider,_,end = fixture()
        sample = build_sample(provider,end,prep,graph,config,{"a":2})
        x,states,present,y,mask,adjacency = _tensor_batch([sample],"cpu")
        model = SegmentGATLSTM(len(FEATURE_NAMES),["continuous"],config)
        logits,types = model(x,states,adjacency,present)
        self.assertEqual(tuple(logits.shape),(1,1,3))
        logits.sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_candidate(model,prep,graph,config,Path(directory),{})
            snapshot = load_candidate(manifest["model_id"],Path(directory))
            self.assertFalse(manifest["active"])
            with torch.no_grad():
                self.assertTrue(torch.allclose(model(x,states,adjacency,present)[0],snapshot[0](x,states,adjacency,present)[0]))
            result = predict_at(snapshot,provider,"a",end.isoformat())
            self.assertEqual(len(result["input_coverage"]["scales"]),3)
            empty = predict_at(snapshot,lambda *args: [],"a",end.isoformat())
            self.assertEqual(empty["status"],"indeterminate")
            self.assertIn("insufficient_scale_coverage",empty["reasons"])

    def test_training_selects_threshold_and_keeps_candidate_inactive(self):
        from app.modulos.clasificador.training import train_classifier, save_candidate, load_candidate
        config,prep,graph,provider,_,end = fixture()
        rows = []
        for label in range(3):
            row = build_sample(provider,end,prep,graph,config,{"a":label})
            row["metadata"].update(sample_id=f"s{label}",parent_episode_id=f"e{label}",target_label=label,
                                   fault_subtype="spike" if label == 2 else None)
            rows.append(row)
        model, metrics = train_classifier(rows, rows, config, [3], ["continuous"])
        self.assertEqual(len(metrics["segment_threshold_search"]),3)
        self.assertIn(model.threshold,[.3,.5,.7])
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_candidate(model,prep,graph,config,Path(directory),metrics)
            loaded = load_candidate(manifest["model_id"],Path(directory))
            self.assertEqual(loaded[0].threshold,model.threshold)
            self.assertFalse(manifest["active"])

    def test_another_entity_cannot_change_own_prediction(self):
        import torch
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        config,prep,graph,provider,_,end = fixture()
        row = build_sample(provider,end,prep,graph,config)
        x = torch.tensor([[ [tick[0],tick[0]] for tick in row["X"]]],dtype=torch.float32)
        present = torch.tensor([[ [tick[0],tick[0]] for tick in row["node_present"]]])
        model = SegmentGATLSTM(len(FEATURE_NAMES),["continuous","continuous"],config).eval()
        with torch.no_grad():
            before = model(x,None,None,present)[0]
            changed = x.clone()
            changed[:,:,1,:19] += 100
            after = model(changed,None,None,present)[0]
        self.assertTrue(torch.allclose(before[:,0],after[:,0]))

    def test_dataset_pipeline_uses_no_f2_and_persists_segments(self):
        import csv
        import hashlib
        import json
        from unittest.mock import patch
        from app.modulos.clasificador.pipeline import build_sequence_dataset, load_partition
        config,_,_,_,start,_ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"history.csv"
            with source.open("w",newline="",encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["entity_id","state","last_changed"])
                for hour in range(20*24):
                    writer.writerow(["a",hour%9,(start+timedelta(hours=hour)).isoformat()])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"boundaries":[(start+timedelta(days=day)).isoformat() for day in (0,20,25,30)]}),encoding="utf-8")
            annotation = {"annotation_id":"one","entity_id":"a","start":(start+timedelta(days=8)).isoformat(),
                          "end":(start+timedelta(days=8)).isoformat(),"status":"reviewed","label":"normal",
                          "source_sha256":digest,"dataset_id":"source"}
            class Review:
                def annotation(self,key): return annotation
                def dataset(self,key): return {"entities":[]}
            item = {"source_id":"source","path":str(source),"sha256":digest,"partition":"development", "filename":source.name}
            with patch("app.modulos.clasificador.pipeline.development_sources",return_value=[item]), patch("app.modulos.clasificador.pipeline.build_graph",side_effect=AssertionError("F2 must not be called")):
                dataset = build_sequence_dataset(Review(),["source"],["one"],config,root/"diagnostics",manifest)
            self.assertEqual(dataset["schema_version"],SCHEMA)
            self.assertEqual(dataset["context_seconds"],259200)
            partition = load_partition(dataset,"train_fit")
            try:
                self.assertEqual(len(partition),1)
                self.assertEqual(partition[0]["X"].shape,(36,1,len(FEATURE_NAMES)))
            finally:
                partition.close()

    def test_lstm_baseline_and_single_scale_are_supported(self):
        from dataclasses import replace
        import torch
        from app.modulos.clasificador.network_segments import SegmentGATLSTM
        from app.modulos.clasificador.training import _tensor_batch
        config,prep,graph,provider,_,end = fixture()
        config = replace(config,segment_encoder="lstm_only",active_scales=(True,False,False))
        row = build_sample(provider,end,prep,graph,config)
        tensors = _tensor_batch([row],"cpu")
        self.assertEqual([s["usable"] for s in row["metadata"]["scale_coverage"]["a"]],[True,False,False])
        model = SegmentGATLSTM(len(FEATURE_NAMES),["continuous"],config).eval()
        with torch.no_grad():
            x,states,present,_,_,adjacency = tensors
            first = model(x,states,adjacency,present)[0]
            x[:,:,:,-12:] = 0
            second = model(x,states,adjacency,present)[0]
        self.assertTrue(torch.allclose(first,second))


if __name__ == "__main__":
    unittest.main()
