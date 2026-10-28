"""Two explicit CSVs: fit only Development, reserve and perturb future data."""
import csv
from datetime import datetime,timedelta,timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.storage import DiskPartition,sample_summaries
from app.modulos.clasificador.config import parse_utc
from app.modulos.clasificador.model_constants import FAULT_TYPES
import test_gat_lstm_evaluation_injections as injection_fixture

spec=importlib.util.spec_from_file_location("period_cli",Path(__file__).resolve().parents[2]/"scripts/gat_lstm_prepare_periods.py")
CLI=importlib.util.module_from_spec(spec);spec.loader.exec_module(CLI)


class PeriodWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture=injection_fixture.EvaluationInjectionTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root=self.fixture.root
        self.development=self.root/"2mesesDesarrollo.csv"
        self.plan=self.root/"injection_plan.json"
        self.plan.write_text(json.dumps(self.fixture.plan))
        self.config=self.root/"config.json"
        self.config.write_text(json.dumps(dict(use_dtw=False,use_entropy=False,max_epochs=1,dropout=0)))
        start=datetime(2026,2,1,tzinfo=timezone.utc)
        faults={6+index*2:kind for index,kind in enumerate(FAULT_TYPES)}
        rows=[]
        for day in range(59):
            at=start+timedelta(days=day,hours=12)
            for index,entity in enumerate(self.fixture.preprocessor["entities"]):
                numeric=entity in self.fixture.members
                state=str(20+index+(day%5)/10) if numeric else "on" if day%2 else "off"
                rows.append([entity,state,at.isoformat(),"false","normal","NULL"])
            if day in (20,52):
                rows.append([self.fixture.members[0],"26",(at+timedelta(hours=1)).isoformat(),"false","anomalia","NULL"])
            if day in faults or day==53:
                kind=faults.get(day,"spike")
                entity="binary_sensor.motion" if kind=="state_replace" else self.fixture.members[0]
                state="unavailable" if kind=="unavailable" else "off" if kind=="state_replace" else "40"
                rows.append([entity,state,(at+timedelta(hours=2)).isoformat(),"true","Falla",kind])
        rows.sort(key=lambda r:r[2])
        with self.development.open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["entity_id","state","last_changed","es_sintetica","tipo","tipo_falla"]);writer.writerows(rows)

    def test_preparation_uses_only_known_development_and_keeps_future_out(self):
        before=(sha256_file(self.development),sha256_file(self.fixture.source))
        output=self.root/"experiment"
        with patch("builtins.print"):
            result=CLI.main([str(self.development),str(self.fixture.source),"--config",str(self.config),
                             "--injection-plan",str(self.plan),"--output",str(output)])
        self.assertEqual(result,0)
        self.assertEqual(before,(sha256_file(self.development),sha256_file(self.fixture.source)))
        dataset=json.loads((output/"dataset.json").read_text())
        self.assertTrue(dataset["ready_for_training"])
        self.assertEqual(dataset["temporal_holdout"]["version"],"independent_2month_periods_v1")
        self.assertEqual(set(dataset["source_hashes"].values()),{before[0]})
        for part in ("train_fit","dev_tune"):
            with DiskPartition(dataset["sequence_dir"],part) as samples:
                self.assertTrue(all(parse_utc(row["metadata"]["context_end"])<parse_utc("2026-04-01T00:00:00Z")
                                    for row in sample_summaries(samples)))
        injections=json.loads((output/"evaluation_injections.json").read_text())
        self.assertEqual(set(injections["episode_counts"]),{*FAULT_TYPES,"environmental_anomaly"})
        self.assertFalse((output/"trained_model.json").exists())
        verify_spec=importlib.util.spec_from_file_location("verify_evaluation_cli",
            Path(__file__).resolve().parents[2]/"scripts/gat_lstm_verify_evaluation.py")
        verifier=importlib.util.module_from_spec(verify_spec);verify_spec.loader.exec_module(verifier)
        self.assertEqual(verifier.verify(output)["status"],"VERIFIED")
        with (output/"evaluation_injected.csv").open("a") as stream:
            stream.write("tampered\n")
        with self.assertRaisesRegex(ValueError,"cambió"):
            verifier.verify(output)

    def test_overlapping_periods_fail_before_preparation(self):
        with self.assertRaisesRegex(ValueError,"no solaparse"):
            CLI.main([str(self.development),str(self.development),"--output",str(self.root/"overlap")])
        self.assertFalse((self.root/"overlap").exists())


if __name__=="__main__":
    unittest.main()
