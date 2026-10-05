import unittest
from datetime import datetime, timezone, timedelta

from app.modulos.clasificador.config import Config
from app.modulos.clasificador.config import partition_bounds
from app.modulos.clasificador.dataset import index_labelled_targets
from app.modulos.clasificador.features import fit_preprocessor
from app.modulos.clasificador.features import INVALID
from app.modulos.clasificador.features import OOV
from app.modulos.clasificador.features import state_id
from app.modulos.clasificador.temporal import build_causal_sample
from app.modulos.clasificador.segments import fit_spec, FEATURE_NAMES


class CausalDataTests(unittest.TestCase):
    def setUp(self):
        self.config = Config(sequence_steps=3, grid_seconds=60, feature_window_seconds=120,
                             )
        self.events = [
            {"entity_id": "sensor.a", "state": "10", "timestamp": "2026-02-02T00:00:00Z", "source_row": 2},
            {"entity_id": "sensor.a", "state": "unavailable", "timestamp": "2026-02-02T00:01:00Z", "source_row": 3},
            {"entity_id": "sensor.b", "state": "NONE", "timestamp": "2026-02-02T00:01:00Z", "source_row": 4},
            {"entity_id": "sensor.a", "state": "12", "timestamp": "2026-02-02T00:02:00Z", "source_row": 5},
        ]
        self.prep = fit_preprocessor(self.events, [], self.config,
                                     {"sensor.b": "multistate", "sensor.a": "continuous"})
        self.graph = fit_spec(self.events, self.prep, self.config,
            (datetime(2026, 1, 29, tzinfo=timezone.utc), datetime(2026, 2, 3, tzinfo=timezone.utc)))

    def provider(self, start, end, entities):
        # Deliberately return all rows, including future rows: constructor must filter.
        return self.events

    def test_future_invariant_and_invalid_presence(self):
        target = "2026-02-02T00:01:30Z"
        first = build_causal_sample(self.provider, target, self.prep, self.graph, self.config)
        self.events.append({"entity_id": "sensor.a", "state": "9999",
                            "timestamp": "2026-02-02T00:05:00Z", "source_row": 6})
        second = build_causal_sample(self.provider, target, self.prep, self.graph, self.config)
        self.assertEqual(first, second)
        self.assertEqual(first["metadata"]["entity_ids"], ["sensor.a", "sensor.b"])
        known = FEATURE_NAMES.index("known_fraction")
        invalid = FEATURE_NAMES.index("invalid_fraction")
        self.assertTrue(any(row[0][known] > 0 for row in first["X"]))
        self.assertTrue(any(row[0][invalid] > 0 for row in first["X"]))
        self.assertFalse(any(first["label_mask"]))

    def test_vocab_and_none(self):
        info = self.prep["entities"]["sensor.b"]
        self.assertGreaterEqual(state_id("NONE", info), 3)
        self.assertEqual(state_id("none", info), INVALID)
        self.assertEqual(state_id("UNSEEN", info), OOV)

    def test_point_exact_and_interval_end_inclusive(self):
        bounds = {"train_fit": (datetime(2026, 1, 29, tzinfo=timezone.utc),
                                datetime(2026, 2, 3, tzinfo=timezone.utc))}
        point = {"annotation_id": "p", "entity_id": "sensor.a", "label": "sensor_fault",
                 "status": "reviewed", "start": "2026-02-02T00:01:00.123Z",
                 "end": "2026-02-02T00:01:00.123Z", "source_row": 7}
        interval = {"annotation_id": "i", "entity_id": "sensor.a", "label": "normal",
                    "status": "reviewed", "start": "2026-02-02T00:00:00Z",
                    "end": "2026-02-02T00:02:00Z"}
        rows, errors = index_labelled_targets([point, interval], self.provider, bounds, self.config)
        self.assertFalse(errors)
        self.assertEqual(sum(row.annotation_id == "p" for row in rows), 1)
        self.assertIn("2026-02-02T00:01:00.123000+00:00", [r.timestamp for r in rows])
        self.assertIn("2026-02-02T00:00:00+00:00", [r.timestamp for r in rows])

    def test_manifest_boundaries(self):
        manifest = {"boundaries": ["2026-02-01T00:00:00Z", "2026-05-21T00:00:00Z",
                                   "2026-06-26T00:00:00Z", "2026-08-01T00:00:00Z"]}
        bounds = partition_bounds(manifest)
        self.assertLess(bounds["train_fit"][1], bounds["dev_tune"][1])

    def test_conflicts_never_reintroduced_and_order_independent(self):
        bounds = {"train_fit": (datetime(2026, 1, 29, tzinfo=timezone.utc),
                                datetime(2026, 2, 3, tzinfo=timezone.utc))}
        annotations = []
        for identifier, label in (("a", "normal"), ("b", "sensor_fault"), ("c", "normal")):
            annotations.append({"annotation_id": identifier, "entity_id": "sensor.a", "label": label,
                                "status": "reviewed", "start": "2026-02-02T00:01:00Z",
                                "end": "2026-02-02T00:01:00Z", "source_row": 7})
        first, _ = index_labelled_targets(annotations, self.provider, bounds, self.config)
        second, _ = index_labelled_targets(list(reversed(annotations)), self.provider, bounds, self.config)
        self.assertEqual(first, second)
        self.assertEqual(first, [])


if __name__ == "__main__":
    unittest.main()
