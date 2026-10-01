import unittest

from app.modulos.clasificador.features import dtw_distance
from app.modulos.clasificador.features import dtw_summary
from app.modulos.clasificador.features import entropy
from app.modulos.clasificador.features import histogram_id
from app.modulos.clasificador.config import Config
from app.modulos.clasificador.temporal import build_causal_sample
from app.modulos.clasificador.features import fit_preprocessor


class FeatureTests(unittest.TestCase):
    def test_entropy(self):
        self.assertEqual(entropy([0, 0, 0], 1), (0.0, True))
        self.assertAlmostEqual(entropy([0, 0, 1, 1], 2)[0], 1.0)
        self.assertFalse(entropy([0, 1], 2)[1])
        self.assertEqual(histogram_id(100, [0, 1]), 2)

    def test_dtw(self):
        self.assertEqual(dtw_distance([1, 2, 3], [1, 2, 3]), 0)
        self.assertGreater(dtw_distance([1, 2, 3], [2, 3, 4]), 0)
        self.assertIsNone(dtw_distance([1, 2], [1, 2, 3, 4], .1))
        self.assertEqual(dtw_summary([1, 2], []), (0.0, 0.0, 0.0, False))

    def test_dtw_does_not_bridge_invalid_report(self):
        config = Config(sequence_steps=2, grid_seconds=60, feature_window_seconds=120,
                        use_dtw=True, use_entropy=False)
        events = [{"entity_id": "sensor.a", "state": "1", "timestamp": "2026-02-02T00:00:00Z"},
                  {"entity_id": "sensor.a", "state": "unavailable", "timestamp": "2026-02-02T00:00:30Z"},
                  {"entity_id": "sensor.a", "state": "3", "timestamp": "2026-02-02T00:01:00Z"}]
        prep = fit_preprocessor(events, [], config, {"sensor.a": "continuous"})
        prep["entities"]["sensor.a"]["normal_references"] = [[0.0, 1.0, 2.0]]
        sample = build_causal_sample(lambda *_: events, "2026-02-02T00:01:00Z", prep,
                                     {"adjacency": [[True]]}, config)
        self.assertEqual(sample["X"][-1][0][18], 0.0)


if __name__ == "__main__":
    unittest.main()
