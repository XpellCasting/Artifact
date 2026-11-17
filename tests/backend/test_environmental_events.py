import copy
from datetime import datetime, timedelta, timezone
import unittest

from app.modulos.clasificador.scenarios import generate_scenario
from app.modulos.clasificador.environmental_events import validate_events, response_value


def fixture():
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    prep = {"entities": {"temperature": {"kind": "continuous", "scale": 1},
                         "motion": {"kind": "binary", "vocab": {"off": 3, "on": 4}},
                         "occupancy": {"kind": "multistate", "vocab": {"empty": 3, "low": 4, "high": 5}}}}
    config = {"seed": 17, "context_seconds": 259200,
              "partitions": {"train_fit": [start.isoformat(), (start+timedelta(days=25)).isoformat()]},
              "calibration_end": (start+timedelta(days=14)).isoformat(),
              "quotas": {"train_fit": {"episodes_per_type": 0}},
              "durations": {}, "environmental_episodes": 0,
              "environmental_events": [{"name": "occupancy_demo", "relation_basis": "Fixture: misma habitación ficticia",
                  "episodes": 2, "durations_seconds": [3600, 7200], "intensities": [1, 1.5],
                  "responses": [
                      {"entity": "temperature", "kind": "continuous", "pattern": [[0, 0], [.5, 3], [1, 0]]},
                      {"entity": "motion", "kind": "binary", "delay_seconds": 900, "pattern": [[0, "on"], [1, "off"]]},
                      {"entity": "occupancy", "kind": "multistate", "pattern": [[0, "low"], [.5, "high"], [1, "empty"]]}]}]}
    grouped = {e: [] for e in prep["entities"]}
    for step in range(25*24*4):
        for index, entity in enumerate(grouped):
            grouped[entity].append({"row_id": step*3+index, "t": (start+timedelta(minutes=15*step)).isoformat(),
                                   "state": {"temperature": "20", "motion": "off", "occupancy": "empty"}[entity],
                                   "eligible": True, "certainty": "fixture", "entity": entity})
    return config, prep, grouped


class MixedEventTests(unittest.TestCase):
    def test_related_types_delays_provenance_reproducibility_and_immutability(self):
        config, prep, grouped = fixture()
        original = copy.deepcopy(grouped)
        run = lambda: generate_scenario(None, "train_fit", "demo", config, prep, grouped=grouped)
        result = run()
        self.assertEqual(result, run())
        self.assertEqual(grouped, original)
        self.assertEqual(len(result["episodes"]), 2)
        self.assertFalse(result["deficits"])
        self.assertEqual(len({r["row_id"] for r in result["overlays"]}), len(result["overlays"]))
        for episode in result["episodes"]:
            rows = [r for r in result["overlays"] if r["intervention_id"] == episode["intervention_id"]]
            self.assertEqual({r["entity"] for r in rows}, set(grouped))
            self.assertTrue(all(r["before"] != r["after"] and r["certainty"] == "synthetic_controlled" for r in rows))
            start = datetime.fromisoformat(episode["start"])
            self.assertTrue(all(datetime.fromisoformat(r["timestamp"]) >= start+timedelta(minutes=15) for r in rows if r["entity"] == "motion"))
            self.assertGreaterEqual(episode["start"], config["calibration_end"])
            self.assertLess(episode["end"], config["partitions"]["train_fit"][1])

    def test_missing_context_is_a_deficit_not_partial_episode(self):
        config, prep, grouped = fixture()
        grouped["motion"] = []
        result = generate_scenario(None, "train_fit", "demo", config, prep, grouped=grouped)
        self.assertFalse(result["overlays"])
        self.assertEqual(result["deficits"][0]["missing"], 2)

    def test_invalid_context_never_receives_an_episode(self):
        config, prep, grouped = fixture()
        for row in grouped["occupancy"]:
            row["eligible"] = False
        result = generate_scenario(None, "train_fit", "demo", config, prep, grouped=grouped)
        self.assertFalse(result["episodes"])

    def test_unknown_states_types_and_undocumented_relations_rejected(self):
        config, prep, _ = fixture()
        for mutation in (lambda e: e.update(relation_basis=""),
                         lambda e: e["responses"][1].update(kind="multistate"),
                         lambda e: e["responses"][1].update(pattern=[[0, "ALERT"], [1, "off"]]),
                         lambda e: e["responses"][0].update(delay_seconds=-1)):
            events = copy.deepcopy(config["environmental_events"])
            mutation(events[0])
            with self.assertRaises(ValueError):
                validate_events(events, prep["entities"])

    def test_interpolation_limits_and_categorical_steps(self):
        config, _, _ = fixture()
        continuous, binary, multi = config["environmental_events"][0]["responses"]
        self.assertEqual(response_value(continuous, "20", .25, 2, [0, 22]), "22")
        self.assertEqual(response_value(continuous, "20", 1, 2, [0, 22]), "20")
        self.assertEqual(response_value(binary, "off", .5, 100, [None, None]), "on")
        self.assertEqual(response_value(multi, "empty", .75, 1, [None, None]), "high")


if __name__ == "__main__":
    unittest.main()
