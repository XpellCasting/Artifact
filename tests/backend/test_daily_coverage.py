from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from app.flow.evaluacion import labelled_report
from app.modulos.estadistico.cobertura import coverage, daily_partitions, reporting_policy
from app.modulos.estadistico.continuos_temporales import FiltroContinuoTemporal, causal_features, FEATURE_WIDTH
from app.modulos.estadistico.persistencia import prepare_persistence, evaluate_persistence
from app.modulos.estadistico.residuos_secuenciales import prepare_sensor, RESIDUAL_FOCUS_METHOD
from app.services.historial.cargar_historial import CargadorHistorial


class DailyCoverageTests(unittest.TestCase):
    def frame(self, timestamps, values=None, mode="on_change"):
        if values is None:
            values = np.arange(len(timestamps), dtype=float)
        return pd.DataFrame({"entity_id": "sensor.uv", "timestamp_dt": timestamps,
            "value": values, "state": np.asarray(values).astype(str), "sensor_kind": "continuous",
            "is_valid_value": True, "reporting_mode": mode})

    def test_daily_boundaries_ignore_report_density_and_keep_partial_tail(self):
        start = pd.Timestamp("2026-01-01T08:37:00Z")
        dense = list(pd.date_range(start, periods=500, freq="min"))
        last = start + pd.Timedelta(days=60, hours=5, minutes=12)
        _, _, stop, info = daily_partitions(dense + [last])
        self.assertEqual(stop, len(dense) + 1)
        self.assertEqual(info["complete_blocks"], 60)
        self.assertAlmostEqual(info["final_partial_block_hours"], 5.2)
        self.assertEqual(info["last_observation_at"], last.isoformat())
        self.assertEqual(pd.Timestamp(info["threshold_start"]), start + pd.Timedelta(days=51))
        self.assertAlmostEqual(info["threshold_calendar_days"], 9 + 5.2 / 24)

    def test_night_without_uv_changes_counts_as_elapsed_exposure(self):
        at = pd.to_datetime(["2026-01-01T18:00Z", "2026-01-01T18:01Z", "2026-01-02T06:00Z"], utc=True)
        frame = self.frame(at, [1., 0., 1.])
        result = coverage(frame, at[0], at[-1], reporting_policy(frame))
        self.assertEqual(result["exposure_days"], .5)
        self.assertFalse(result["connectivity_verified"])
        self.assertIn("complete_change_log", result["coverage_basis"])

    def test_unavailable_breaks_coverage_and_periodic_gap_expires(self):
        at = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
        frame = self.frame(at)
        frame.loc[1, "is_valid_value"] = False
        result = coverage(frame, at[0], at[-1], reporting_policy(frame))
        self.assertAlmostEqual(result["exposure_days"], 2 / 24)
        frame["reporting_mode"] = "periodic"
        frame["max_report_gap_seconds"] = 600.
        result = coverage(frame, at[0], at[-1], reporting_policy(frame))
        self.assertAlmostEqual(result["exposure_days"], 1200 / 86400)

    def test_real_loader_records_change_semantics_and_declaration_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text("entity_id,state,last_changed\nsensor.uv,0,2026-01-01T18:00Z\nsensor.uv,1,2026-01-02T06:00Z\n")
            loaded = CargadorHistorial().cargar_real(path)
            self.assertEqual(loaded.entidades.iloc[0].reporting_mode, "on_change")
            path.write_text("entity_id,state,last_changed,reporting_mode,max_report_gap_seconds\nsensor.uv,0,2026-01-01T18:00Z,periodic,600\n")
            loaded = CargadorHistorial().cargar_real(path)
            self.assertEqual(reporting_policy(loaded.eventos)["max_report_gap_seconds"], 600.)

    def test_temporal_features_record_duration_and_reset_after_invalid_report(self):
        at = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
        frame = self.frame(at, np.zeros(30))
        vector = causal_features(list(zip(at[:12], frame.value[:12])), at[12])
        self.assertEqual(len(vector), FEATURE_WIDTH)
        self.assertAlmostEqual(vector[30], np.log1p(12 * 3600))
        frame.loc[14, "is_valid_value"] = False
        _, _, times = FiltroContinuoTemporal()._samples(frame)
        self.assertNotIn(at[15], times)
        self.assertIn(at[27], times)

    def test_uv_night_is_quiet_but_prolonged_zero_is_suspicious(self):
        at, values = [], []
        start = pd.Timestamp("2026-01-01", tz="UTC")
        for day in range(20):
            for hour, value in [(6, 1.), (12, 5.), (18, 0.)]:
                at.append(start + pd.Timedelta(days=day, hours=hour)); values.append(value)
        reference = prepare_persistence(self.frame(at, values))
        night = {"timestamp_dt": start + pd.Timedelta(days=20, hours=5), "value": 0., "is_valid_value": True}
        self.assertFalse(evaluate_persistence(reference, night)["detected"])
        delayed = {**night, "timestamp_dt": start + pd.Timedelta(days=20, hours=12)}
        self.assertTrue(evaluate_persistence(reference, delayed)["detected"])
        unavailable = {**delayed, "timestamp_dt": start + pd.Timedelta(days=20, hours=13), "is_valid_value": False}
        evaluate_persistence(reference, unavailable)
        self.assertFalse(evaluate_persistence(reference, {**delayed, "timestamp_dt": start + pd.Timedelta(days=21)})["detected"])

    def test_two_months_calibrate_without_dropping_final_value(self):
        at = pd.date_range("2026-01-01T08:37Z", periods=60 * 48 + 8, freq="30min")
        values = np.sin(np.arange(len(at)) / 48 * 2 * np.pi) + np.random.default_rng(42).normal(0, .1, len(at))
        with threadpool_limits(limits=1):
            model = prepare_sensor(FiltroContinuoTemporal(), self.frame(at, values), "sensor.uv", RESIDUAL_FOCUS_METHOD)
        self.assertTrue(model["eligible"], model.get("reason"))
        self.assertGreaterEqual(model["coverage"]["exposure_days"], 7.)
        self.assertEqual(model["last_observation_at"], at[-1].isoformat())
        self.assertGreater(model["partitions"]["final_partial_block_hours"], 0.)

    def test_report_separates_missed_from_unassessed(self):
        truth = [{"raw_index": i, "entity": "sensor.a", "tipo": "Falla", "synthetic": "true", "fault_type": "bias"} for i in range(3)]
        items = [{"source_raw_index": i, "final_decision": decision} for i, decision in enumerate(["suspicion", "no_alert", "indeterminate"])]
        report = labelled_report(items, truth)
        row = report["groups"][0]
        self.assertEqual((row["detected"], row["missed_evaluable"], row["non_evaluable"]), (1, 1, 1))
        self.assertEqual(items[-1]["evaluation"]["outcome"], "no_evaluable")
