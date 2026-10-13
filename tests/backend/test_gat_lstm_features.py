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

    def test_invalid_reports_remain_visible_in_segment_features(self):
        from app.modulos.clasificador.segments import intervals
        from datetime import datetime, timezone, timedelta
        start = datetime(2026, 2, 2, tzinfo=timezone.utc)
        config = Config()
        events = [{"entity_id": "sensor.a", "state": state,
                   "timestamp": (start + timedelta(seconds=seconds)).isoformat()}
                  for seconds, state in [(0, "1"), (30, "unavailable"), (60, "3")]]
        prep = fit_preprocessor(events, [], config, {"sensor.a": "continuous"})
        ticks = intervals(events, start, start + timedelta(seconds=90), 30,
                          prep["entities"]["sensor.a"])
        self.assertEqual([tick["invalid"] for tick in ticks], [0, 1, 0])


if __name__ == "__main__":
    unittest.main()
