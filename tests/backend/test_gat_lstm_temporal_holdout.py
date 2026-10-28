"""Calendar holdout must never enter fitting or alter future ground truth."""
import csv
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from app.modulos.clasificador.config import Config,partition_bounds,parse_utc
from app.modulos.clasificador.digests import sha256_file
from app.modulos.clasificador.temporal_holdout import split_four_month_csv,temporal_plan
from app.modulos.clasificador.preparation import prepare_csv_dataset
from app.modulos.clasificador.storage import DiskPartition, sample_summaries
from app.modulos.clasificador.model_constants import FAULT_TYPES
from app.modulos.clasificador.training_coverage import training_coverage,require_training_coverage


HEADERS=["entity_id","state","last_changed","es_sintetica","tipo","tipo_falla"]


def write_history(path, future_value="99", crossing=False):
    rows=[]
    for month,day in ((2,1),(2,5),(3,25),(4,5),(5,25)):
        for offset,(label,fault) in enumerate((("normal","NULL"),("anomalia","NULL"),("Falla","Pico"))):
            value=str(10+offset) if month<4 else future_value
            rows.append(["sensor.a",value,f"2026-{month:02d}-{day:02d}T{12+offset:02d}:00:00Z","false",label,fault])
    if crossing:
        rows.extend([["sensor.a","unavailable",at,"false","Falla","Indisponibilidad"]
            for at in ("2026-03-31T23:59:00Z","2026-04-01T00:00:00Z")])
    rows.sort(key=lambda row:row[2])
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(HEADERS);writer.writerows(rows)
    return rows


class TemporalHoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/"4mesesDesarrollo.csv"

    def test_exact_calendar_boundaries_and_explicit_internal_selection(self):
        plan=temporal_plan(datetime(2026,2,1,8,tzinfo=timezone.utc))
        self.assertEqual(plan["evaluation_start"],"2026-04-01T00:00:00+00:00")
        self.assertEqual(plan["evaluation_end"],"2026-06-01T00:00:00+00:00")
        self.assertEqual(plan["train_fit_end"],"2026-03-18T00:00:00+00:00")
        bounds=partition_bounds(dict(boundaries=[plan["development_start"],plan["development_end"],plan["evaluation_end"],plan["evaluation_end"]],train_fit_end=plan["train_fit_end"]))
        self.assertEqual(bounds["train_fit"][1],parse_utc(plan["train_fit_end"]))
        year_end=temporal_plan(datetime(2026,11,1,tzinfo=timezone.utc))
        self.assertEqual(year_end["evaluation_start"],"2027-01-01T00:00:00+00:00")

    def test_split_preserves_source_future_labels_and_masks_crossing_training_episode(self):
        original=write_history(self.source,crossing=True);checksum=sha256_file(self.source)
        report=split_four_month_csv(self.source,self.root/"split")
        self.assertEqual(sha256_file(self.source),checksum)
        with (self.root/"split/evaluation.csv").open() as stream:
            future=list(csv.reader(stream))[1:]
        self.assertEqual(future,[row for row in original if row[2]>="2026-04-01T00:00:00Z"])
        with (self.root/"split/development.csv").open() as stream:
            earlier=list(csv.DictReader(stream))
        self.assertTrue(all(parse_utc(row["last_changed"])<parse_utc(report["evaluation_start"]) for row in earlier))
        crossing=next(row for row in earlier if row["state"]=="unavailable")
        self.assertEqual((crossing["tipo"],crossing["tipo_falla"]),("NULL","NULL"))
        self.assertEqual(report["development_labels_excluded_at_holdout_boundary"],1)
        with self.assertRaisesRegex(ValueError,"ya existe"):
            split_four_month_csv(self.source,self.root/"split")

    def test_future_changes_cannot_change_preprocessor_or_train_development_samples(self):
        config=Config(use_dtw=False,use_entropy=False,max_epochs=1,dropout=0)
        datasets=[]
        for variant in ("original","changed"):
            write_history(self.source,future_value="99" if variant=="original" else "99999999")
            output=self.root/variant
            split_four_month_csv(self.source,output)
            datasets.append(prepare_csv_dataset([output/"development.csv"],config=config,
                review_dir=output/"review",manifest_path=output/"split_manifest.json"))
        for name in ("preprocessor.json","graph.json"):
            a=json.loads((Path(datasets[0]["sequence_dir"])/name).read_text())
            b=json.loads((Path(datasets[1]["sequence_dir"])/name).read_text())
            self.assertEqual(a,b)
        self.assertEqual(datasets[0]["label_counts"],datasets[1]["label_counts"])
        self.assertTrue(datasets[0]["ready_for_training"])
        for part in ("train_fit","dev_tune"):
            with DiskPartition(datasets[0]["sequence_dir"],part) as partition:
                self.assertTrue(all(parse_utc(row["metadata"]["context_end"])<parse_utc("2026-04-01T00:00:00Z")
                                    for row in sample_summaries(partition)))

    def test_missing_training_fault_type_cannot_be_supplied_by_holdout_or_selection(self):
        dataset={"label_counts":{part:dict(normal=3,environmental_anomaly=3,sensor_fault=3)
                                 for part in ("train_fit","dev_tune")},
                 "episode_counts":{part:dict(normal=3,environmental_anomaly=3,sensor_fault=3)
                                   for part in ("train_fit","dev_tune")},
                 "fault_type_inventory":{part:{kind:dict(windows=3,independent_episodes=3) for kind in FAULT_TYPES}
                                         for part in ("train_fit","dev_tune","validation")}}
        self.assertTrue(require_training_coverage(dataset)["ready"])
        del dataset["fault_type_inventory"]["train_fit"]["spike"]
        self.assertFalse(training_coverage(dataset)["ready"])
        with self.assertRaisesRegex(ValueError,"spike"):
            require_training_coverage(dataset)

    def test_unlabelled_or_missing_months_are_rejected_before_creating_output(self):
        self.source.write_text("entity_id,state,last_changed\nsensor.a,1,2026-02-01T00:00:00Z\n")
        with self.assertRaisesRegex(ValueError,"CSV etiquetado"):
            split_four_month_csv(self.source,self.root/"invalid")
        self.assertFalse((self.root/"invalid").exists())
        write_history(self.source)
        text=self.source.read_text().replace("2026-05-25","2026-04-25")
        self.source.write_text(text)
        with self.assertRaisesRegex(ValueError,"cuatro meses"):
            split_four_month_csv(self.source,self.root/"invalid")


if __name__ == "__main__":
    unittest.main()
