import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.modulos.clasificador.comparison import compare_prediction_files


class ComparisonTests(unittest.TestCase):
    def test_structural_miss_stays_in_fixed_target_denominator_as_abstention(self):
        samples = []
        for label, class_id in (("normal", 0), ("environmental_anomaly", 1), ("sensor_fault", 2)):
            samples.append({"y": [class_id], "label_mask": [True], "metadata": {
                "sample_id": label, "entity_ids": ["sensor.a"], "parent_episode_id": label,
                "evidence_origin": "historical_review"}})
        with tempfile.TemporaryDirectory() as tmp:
            predictions = Path(tmp) / "predictions.json"
            predictions.write_text(json.dumps({"threshold": .4, "margin": 0.0, "predictions": [
                {"sample_id": "normal", "entity_id": "sensor.a", "scores": [.9, .05, .05]},
                {"sample_id": "environmental_anomaly", "entity_id": "sensor.a", "scores": [.1, .8, .1]},
            ]}), encoding="utf-8")
            with patch("app.modulos.clasificador.comparison.load_partition", return_value=samples):
                result = compare_prediction_files({"dataset_id": "fixture"}, "validation", {"B0": predictions})
        variant = result["variants"]["B0"]
        self.assertEqual(result["target_count"], 3)
        self.assertEqual(result["class_support"]["sensor_fault"], 1)
        self.assertEqual(variant["structural_abstentions"], 1)
        self.assertEqual(variant["matrix"][2][3], 1)
        self.assertEqual(variant["coverage"], 2 / 3)


if __name__ == "__main__": unittest.main()
