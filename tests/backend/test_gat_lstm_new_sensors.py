from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

from app.flow.evaluacion import CausalFlowRunner
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.features import fit_preprocessor
from app.modulos.clasificador.network_segments import SegmentGATLSTM
from app.modulos.clasificador.profiles import build_sensor_profile, profile_from_past
from app.modulos.clasificador.runtime import predict_at
from app.modulos.clasificador.segments import FEATURE_NAMES, fit_spec
from app.modulos.clasificador.temporal import CausalInputError
from app.modulos.clasificador.training import save_candidate
from app.services.diagnostico import DiagnosticWorkbench, _write


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
TARGET = START + timedelta(hours=4)
NEW_KINDS = {"sensor.prueba_temp": "continuous", "binary_sensor.prueba": "binary",
             "sensor.prueba_estado": "multistate"}


def history(entity, kind, count=16):
    states = {"continuous": ("20", "21", "22"), "binary": ("off", "on"),
              "multistate": ("idle", "running", "paused")}[kind]
    return [{"entity_id": entity, "timestamp": (START + timedelta(minutes=15*i)).isoformat(),
             "sensor_kind": kind, "state": states[i % len(states)], "source_row": i}
            for i in range(count)]


def provider(rows):
    return lambda start, end, entities: [row for row in rows
        if row["entity_id"] in entities and start <= datetime.fromisoformat(row["timestamp"]) <= end]


def frame(rows):
    result = pd.DataFrame(rows)
    result["timestamp_dt"] = pd.to_datetime(result["timestamp"], utc=True)
    result["_raw_index"] = range(len(result))
    return result


class NewSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(dropout=0, use_entropy=False, use_dtw=False)
        kinds = {f"trained.{kind}": kind for kind in NEW_KINDS.values()}
        rows = [row for entity, kind in kinds.items() for row in history(entity, kind)]
        cls.prep = fit_preprocessor(rows, [], cls.config, kinds)
        cls.graph = fit_spec(rows, cls.prep, cls.config, (START, TARGET))
        cls.model = SegmentGATLSTM(len(FEATURE_NAMES), [kinds[key] for key in cls.prep["entity_ids"]], cls.config).eval()
        cls.manifest = {"model_id": "a"*32, "config": cls.config.as_dict(),
                        "rejection_policy": {"threshold": 0, "margin": 0}}
        cls.snapshot = (cls.model, cls.prep, cls.graph, cls.manifest)

    def test_unseen_entities_of_all_three_kinds_reach_the_frozen_model(self):
        weights = {key: value.clone() for key, value in self.model.state_dict().items()}
        original_prep, original_graph = deepcopy(self.prep), deepcopy(self.graph)
        profiles = {}
        for entity, kind in NEW_KINDS.items():
            with self.subTest(kind=kind):
                result = predict_at(self.snapshot, provider(history(entity, kind)), entity,
                                    TARGET.isoformat(), profiles, sensor_kind=kind)
                self.assertEqual(len(result["class_scores"]), 3)
                self.assertNotEqual(result["status"], "indeterminate")
                self.assertEqual(result["onboarding"]["input_kind"], kind)
                self.assertFalse(result["onboarding"]["normality_verified"])
                self.assertEqual(result["onboarding"]["reference_origin"], "causal_warmup")
        self.assertEqual(set(profiles), set(NEW_KINDS))
        self.assertEqual(self.prep, original_prep)
        self.assertEqual(self.graph, original_graph)
        self.assertTrue(all(torch.equal(value, weights[key])
                            for key, value in self.model.state_dict().items()))

    def test_identity_and_numeric_units_do_not_change_normalized_prediction(self):
        original = history("sensor.prueba_temp", "continuous")
        renamed = [{**row, "entity_id": "sensor.otro", "state": str(float(row["state"])*10+100)}
                   for row in original]
        left = predict_at(self.snapshot, provider(original), "sensor.prueba_temp", TARGET.isoformat())
        right = predict_at(self.snapshot, provider(renamed), "sensor.otro", TARGET.isoformat())
        self.assertEqual(left["class_scores"], right["class_scores"])

    def test_future_and_current_hour_cannot_calibrate_the_reference(self):
        entity = "sensor.prueba_temp"
        rows = history(entity, "continuous")
        clean = profile_from_past(provider(rows), entity, TARGET, self.config, {"continuous"})
        contaminated = [{**row, "state": "999999"} if datetime.fromisoformat(row["timestamp"]) >= TARGET-timedelta(hours=1)
                        else row for row in rows]
        contaminated += [{"entity_id": entity, "timestamp": (TARGET+timedelta(hours=1)).isoformat(),
                          "state": "-999999", "sensor_kind": "continuous"}]
        # Even a provider that returns future data cannot leak it into calibration.
        dirty = profile_from_past(lambda *args: contaminated, entity, TARGET, self.config, {"continuous"})
        self.assertEqual(clean, dirty)
        before = predict_at(self.snapshot, provider(rows), entity, TARGET.isoformat())
        after = predict_at(self.snapshot, lambda *args: rows + contaminated[-1:], entity, TARGET.isoformat())
        self.assertEqual(before["class_scores"], after["class_scores"])

    def test_profile_stays_fixed_when_later_readings_change(self):
        entity = "sensor.prueba_temp"
        rows, profiles = history(entity, "continuous"), {}
        predict_at(self.snapshot, provider(rows), entity, TARGET.isoformat(), profiles)
        frozen = deepcopy(profiles)
        rows += [{"entity_id": entity, "timestamp": (TARGET+timedelta(minutes=i)).isoformat(),
                  "state": "1000", "sensor_kind": "continuous"} for i in range(180)]
        with patch("app.modulos.clasificador.profiles.profile_from_past", side_effect=AssertionError("must remain frozen")):
            result = predict_at(self.snapshot, provider(rows), entity,
                                (TARGET+timedelta(hours=3)).isoformat(), profiles)
        self.assertEqual(profiles, frozen)
        self.assertEqual(len(result["class_scores"]), 3)

    def test_short_invalid_and_duplicate_only_histories_remain_indeterminate(self):
        entity = "sensor.prueba_temp"
        cases = [[], history(entity, "continuous", 3),
                 [{**row, "state": "unavailable"} for row in history(entity, "continuous")],
                 [history(entity, "continuous")[0]]*20]
        for rows in cases:
            with self.subTest(rows=len(rows)):
                result = predict_at(self.snapshot, provider(rows), entity, TARGET.isoformat())
                self.assertEqual(result["status"], "indeterminate")
                self.assertEqual(result["reasons"], ["insufficient_sensor_reference"])
                self.assertEqual(result["class_scores"], {})

    def test_numeric_categories_and_binary_compatibility_use_categorical_input(self):
        for kind in ("binary", "multistate"):
            rows = [{**row, "state": str(i % 2)} for i, row in enumerate(history("sensor.codes", kind))]
            profile = build_sensor_profile(rows, "sensor.codes", TARGET, self.config,
                                           {"continuous", "multistate"})
            self.assertEqual(profile["entity"]["kind"], "multistate")
            self.assertEqual(set(profile["entity"]["vocab"]), {"0", "1"})
            self.assertEqual(profile["sensor_kind"], kind)

    def test_unsupported_kind_is_not_silently_sent_to_untrained_input(self):
        with self.assertRaises(CausalInputError) as error:
            build_sensor_profile(history("sensor.states", "multistate"), "sensor.states",
                                 TARGET, self.config, {"continuous"})
        self.assertEqual(error.exception.reason, "unsupported_sensor_kind")

    def test_binary_uses_the_categorical_branch_of_existing_snapshots(self):
        prep = deepcopy(self.prep)
        prep["entity_ids"].remove("trained.binary")
        prep["entities"].pop("trained.binary")
        entity = "binary_sensor.prueba"
        result = predict_at((self.model, prep, self.graph, self.manifest),
                            provider(history(entity, "binary")), entity, TARGET.isoformat())
        self.assertEqual(result["onboarding"]["sensor_kind"], "binary")
        self.assertEqual(result["onboarding"]["input_kind"], "multistate")
        self.assertEqual(len(result["class_scores"]), 3)

    def test_reviewed_reference_has_priority_and_must_be_prior(self):
        entity = "sensor.prueba_temp"
        rows = history(entity, "continuous")
        profile = build_sensor_profile(rows, entity, TARGET, self.config, {"continuous"})
        profile.update(onboarding_policy="reviewed_normal_reference_v1", normality_verified=True)
        with patch("app.modulos.clasificador.profiles.profile_from_past", side_effect=AssertionError("manual reference wins")):
            result = predict_at(self.snapshot, provider(rows), entity, TARGET.isoformat(), {entity: profile})
            self.assertTrue(result["onboarding"]["normality_verified"])
            rejected = predict_at(self.snapshot, provider(rows), entity, profile["reference_end"], {entity: profile})
        self.assertEqual(rejected["reasons"], ["reference_not_prior_to_target"])

    def test_known_entities_keep_the_saved_preprocessor(self):
        entity = "trained.continuous"
        with patch("app.modulos.clasificador.profiles.profile_from_past", side_effect=AssertionError("known entity")):
            result = predict_at(self.snapshot, provider(history(entity, "continuous")), entity, TARGET.isoformat())
        self.assertNotIn("onboarding", result)
        self.assertEqual(len(result["class_scores"]), 3)

    def test_saved_profile_cannot_supply_later_exposure_to_an_earlier_target(self):
        entity = "sensor.prueba_temp"
        rows = history(entity, "continuous")
        profile = build_sensor_profile(rows, entity, TARGET+timedelta(days=20), self.config, {"continuous"})
        result = predict_at(self.snapshot, provider(rows), entity, TARGET.isoformat(), {entity: profile})
        self.assertEqual(result["reasons"], ["reference_not_prior_to_target"])

    def make_workbench(self, directory):
        workbench = DiagnosticWorkbench(Path(directory))
        manifest = save_candidate(self.model, self.prep, self.graph, self.config, workbench.directory,
                                  {"rejection_policy": {"threshold": 0, "margin": 0}})
        # Independent sensors can have older dates than the model's training home.
        manifest["training_end"] = (TARGET+timedelta(days=365)).isoformat()
        _write(workbench._path("models", manifest["model_id"]), manifest)
        _write(workbench.directory/"active.json", {"model_id": manifest["model_id"]})
        return workbench

    def test_selected_history_prepares_new_sensors_in_the_actual_flow(self):
        rows = [row for entity, kind in NEW_KINDS.items() for row in history(entity, kind)]
        future = frame([{**history(entity, kind)[-1], "timestamp": TARGET.isoformat()}
                        for entity, kind in NEW_KINDS.items()])
        flow = {"name": "unseen sensors", "nodes": [
            {"id": name, "type": name, "position": {}, "config": {"evidence_policy": "independent"}}
            for name in ("input", "classifier", "output")], "edges": [
            {"id": "in-classifier", "source": "input", "target": "classifier"},
            {"id": "classifier-out", "source": "classifier", "target": "output"}]}
        with tempfile.TemporaryDirectory() as tmp:
            workbench = self.make_workbench(tmp)
            runner = CausalFlowRunner(None, None, workbench)
            _, outputs, _ = runner.evaluate(flow, frame(rows), future, pd.DataFrame(), {}, {})
            self.assertEqual(len(outputs["classifier"]), 3)
            for item in outputs["classifier"]:
                diagnosis = item["evidence"][-1]["data"]
                self.assertEqual(len(diagnosis["class_scores"]), 3)
                self.assertEqual(diagnosis["onboarding"]["reference_origin"], "historical_baseline")
                self.assertEqual(diagnosis["onboarding"]["reference_rows"], 16)
            self.assertEqual(workbench._list("sensors"), [])
            profiles = runner.diagnostic_sessions["classifier"].sensor_profiles
            self.assertEqual(set(profiles), set(NEW_KINDS))

    def test_reference_initialization_filters_target_future_and_isolates_runs(self):
        entity = "sensor.prueba_temp"
        with tempfile.TemporaryDirectory() as tmp:
            workbench = self.make_workbench(tmp)
            first = workbench.create_run_session("test", "one")
            rows = history(entity, "continuous")
            rows += [{**rows[-1], "timestamp": at.isoformat(), "state": "9999"}
                     for at in (TARGET, TARGET+timedelta(days=1))]
            first.prepare_sensor_profiles(frame(rows), TARGET)
            self.assertEqual(first.sensor_profiles[entity]["reference_rows"], 16)
            self.assertLess(first.sensor_profiles[entity]["entity"]["median"], 25)
            second = workbench.create_run_session("test", "two")
            self.assertEqual(second.sensor_profiles, {})

    def test_batch_history_is_causal_even_when_candidates_are_reversed(self):
        entity = "sensor.prueba_temp"
        rows = history(entity, "continuous", 32)
        contexts = [{"entity_id": entity, "timestamp": at.isoformat(), "sensor_kind": "continuous"}
                    for at in (TARGET+timedelta(hours=2), TARGET)]
        with tempfile.TemporaryDirectory() as tmp:
            workbench = self.make_workbench(tmp)
            results = workbench.enrich_contextual([], contexts, frame(rows), "history", "reverse")
            self.assertEqual([item["timestamp"] for item in results], [item["timestamp"] for item in contexts])
            self.assertTrue(all(len(item["diagnosis"]["class_scores"]) == 3 for item in results))
            self.assertEqual(results[0]["diagnosis"]["onboarding"], results[1]["diagnosis"]["onboarding"])

    def test_separate_batches_and_restart_reuse_the_same_reference(self):
        entity = "sensor.prueba_temp"
        rows = history(entity, "continuous")
        with tempfile.TemporaryDirectory() as tmp:
            workbench = self.make_workbench(tmp)
            context = {"entity_id": entity, "timestamp": TARGET.isoformat()}
            first = workbench.enrich_contextual([], [context], frame(rows), "history", "frozen")
            frozen = workbench.sensor_profiles_for_run("history:frozen")
            later = TARGET+timedelta(hours=4)
            rows += [{**rows[-1], "timestamp": (TARGET+timedelta(minutes=i)).isoformat(), "state": "9999"}
                     for i in range(240)]
            restarted = DiagnosticWorkbench(Path(tmp))
            with patch("app.modulos.clasificador.profiles.profile_from_past", side_effect=AssertionError("reference already saved")):
                second = restarted.enrich_contextual([], [{**context, "timestamp": later.isoformat()}],
                                                     frame(rows), "history", "frozen")
            self.assertEqual(first[0]["diagnosis"]["onboarding"], second[0]["diagnosis"]["onboarding"])
            self.assertEqual(restarted.sensor_profiles_for_run("history:frozen"), frozen)


if __name__ == "__main__":
    unittest.main()
