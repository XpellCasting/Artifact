import unittest

from app.modulos.clasificador.evaluation import choose_rejection_policy
from app.modulos.clasificador.evaluation import metrics


class EvaluationTests(unittest.TestCase):
    def test_matrix_with_abstention_and_missing_class(self):
        rows = [
            {"true_label": 0, "scores": [.9, .05, .05], "episode_id": "n"},
            {"true_label": 1, "scores": [.1, .2, .7], "episode_id": "a"},
            {"true_label": 2, "scores": [.4, .35, .25], "episode_id": "f"},
        ]
        result = metrics(rows, .5, .1)
        self.assertEqual(result["matrix"], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(result["per_class"]["sensor_fault"]["fn"], 1)
        self.assertEqual(result["coverage"], 2 / 3)
        self.assertEqual(result["false_alarms_environmental"], 1)
        empty = metrics(rows[:1])
        self.assertIsNone(empty["per_class"]["sensor_fault"]["recall"])
        self.assertIsNone(empty["macro_f1"])
        self.assertFalse(choose_rejection_policy(rows[:1])["validated"])

    def test_zero_f1_is_estimable(self):
        rows = [{"true_label": 2, "scores": [.9, .05, .05]}]
        result = metrics(rows, .4, 0)
        self.assertEqual(result["per_class"]["sensor_fault"]["f1"], 0.0)
        self.assertEqual(result["per_class"]["normal"]["f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
