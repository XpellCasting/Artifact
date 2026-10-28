"""Independent controlled interventions: coverage, immutability and provenance."""
from collections import Counter
import csv
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest

from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.evaluation_injections import generate_evaluation
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.modulos.clasificador.config import parse_utc


class EvaluationInjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/"evaluation.csv"
        self.members=["sensor.room_a_temperature","sensor.room_b_temperature","sensor.room_c_temperature"]
        self.preprocessor={"entities":{e:{"kind":"continuous","scale":2.,"median":20.} for e in self.members}}
        self.preprocessor["entities"]["binary_sensor.motion"]={"kind":"binary","vocab":{"on":3,"off":4}}
        self.plan={"episodes_per_type":3,"environmental_episodes":3,
                   "environmental_groups":[dict(name="thermal",entities=self.members,amount_native_units=3.)]}
        self.write_source()

    def write_source(self,labelled=True):
        start=datetime(2026,4,1,tzinfo=timezone.utc)
        with self.source.open("w",newline="") as stream:
            writer=csv.writer(stream)
            writer.writerow(["entity_id","state","last_changed"]+(["es_sintetica","tipo","tipo_falla"] if labelled else []))
            for hour in range(61*24):
                stamp=(start+timedelta(hours=hour)).isoformat()
                for index,entity in enumerate(self.preprocessor["entities"]):
                    state=str(20+(hour%24)/10+index) if entity in self.members else "on" if hour%2 else "off"
                    writer.writerow([entity,state,stamp]+(["false","normal","NULL"] if labelled else []))

    def test_all_types_environmental_coherence_and_exact_reproducibility(self):
        original_hash=sha256_file(self.source)
        first=generate_evaluation(self.source,self.preprocessor,self.root/"first",self.plan)
        second=generate_evaluation(self.source,self.preprocessor,self.root/"second",self.plan)
        self.assertEqual(sha256_file(self.source),original_hash)
        self.assertEqual(first["output_sha256"],second["output_sha256"])
        self.assertEqual(first["episode_counts"],dict.fromkeys([*FAULT_TYPES,"environmental_anomaly"],3))
        self.assertFalse(first["model_predictions_used"])
        episodes=first["episodes"]
        for episode in episodes:
            self.assertGreater(episode["changed_rows"],0)
            if episode["kind"]=="spike":
                self.assertEqual(episode["rows"],1)
            if episode["kind"]=="environmental_anomaly":
                self.assertEqual(episode["entities"],self.members)
        # Same-entity interventions have a full 72h guard between source blocks.
        for entity in self.preprocessor["entities"]:
            intervals=sorted((parse_utc(e["blocks"][entity]["start"]),parse_utc(e["blocks"][entity]["end"]))
                             for e in episodes if entity in e["entities"])
            for previous,next_interval in zip(intervals,intervals[1:]):
                self.assertGreaterEqual(next_interval[0]-previous[1],timedelta(hours=72))

    def test_unchanged_unlabelled_rows_remain_unknown_and_never_become_normal(self):
        self.write_source(labelled=False)
        result=generate_evaluation(self.source,self.preprocessor,self.root/"raw",self.plan)
        self.assertTrue(result["unlabelled_unchanged_rows_remain_NULL"])
        with (self.root/"raw/evaluation_injected.csv").open() as stream:
            rows=list(csv.DictReader(stream))
        self.assertTrue(any(row["tipo"]=="NULL" for row in rows))
        self.assertFalse(any(row["tipo"]=="normal" for row in rows))
        self.assertEqual({e["baseline_normality_basis"] for e in result["episodes"]},{"unverified_valid_observations"})

    def test_only_audited_rows_change_and_original_truth_is_preserved_elsewhere(self):
        generate_evaluation(self.source,self.preprocessor,self.root/"output",self.plan)
        with self.source.open() as stream: original=list(csv.DictReader(stream))
        with (self.root/"output/evaluation_injected.csv").open() as stream: modified=list(csv.DictReader(stream))
        with (self.root/"output/evaluation_interventions.csv").open() as stream:
            audit={int(row["source_row"]):row for row in csv.DictReader(stream)}
        for number,(before,after) in enumerate(zip(original,modified),2):
            self.assertEqual(before["entity_id"],after["entity_id"])
            self.assertEqual(before["last_changed"],after["last_changed"])
            if number not in audit:
                self.assertEqual(before,after)
            else:
                self.assertEqual(audit[number]["original_state"],before["state"])
                self.assertEqual(audit[number]["injected_state"],after["state"])

    def test_missing_compatible_sensor_or_physical_group_fails_without_partial_csv(self):
        self.preprocessor["entities"].pop("binary_sensor.motion")
        with self.assertRaisesRegex(ValueError,"state_replace"):
            generate_evaluation(self.source,self.preprocessor,self.root/"invalid",self.plan)
        self.assertFalse((self.root/"invalid/evaluation_injected.csv").exists())
        self.plan["environmental_groups"][0]["entities"].append("sensor.unrelated_missing")
        with self.assertRaisesRegex(ValueError,"Grupo ambiental"):
            generate_evaluation(self.source,self.preprocessor,self.root/"invalid",self.plan)


if __name__=="__main__":
    unittest.main()
