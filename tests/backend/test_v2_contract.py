from pathlib import Path
import json
import tempfile
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from app.api.diagnostico import crear_router_diagnostico
from app.modulos.clasificador.config import Config
from app.services.diagnostico import DiagnosticWorkbench
from app.database.review import ReviewError


class SegmentContractTests(unittest.TestCase):
    def test_saved_run_restores_its_model_after_restart_with_active_or_explicit_selection(self):
        for source in ("history", "validation_replay", "flow_evaluation"):
            for explicit in (False, True):
                with self.subTest(source=source, explicit=explicit), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    service = DiagnosticWorkbench(root)
                    identifier = "a" * 32
                    model = {"model_id": identifier, "method": "gat_lstm", "training_count": 10,
                             "config": {"architecture_version": "gat_lstm_v3_segments_multiscale"}}
                    service._path("models", identifier).write_text(json.dumps(model), encoding="utf-8")
                    (service.directory / "weights" / f"{identifier}.pt").write_bytes(b"fixture")
                    (service.directory / "active.json").write_text(json.dumps({"model_id": identifier}), encoding="utf-8")
                    model_id = identifier if explicit else None
                    snapshot = service._run_snapshot_for_model(source, "saved-run", model_id)

                    # Saved runs retain their original model even if local metadata changes.
                    service._path("models", identifier).write_text(json.dumps({**model, "training_count": 99}), encoding="utf-8")
                    restarted = DiagnosticWorkbench(root)
                    self.assertEqual(restarted.window_seconds_for_run(source, "saved-run", model_id), 259200)
                    restored = restarted._run_snapshot_for_model(source, "saved-run", model_id)
                    self.assertEqual(restored, snapshot)
                    self.assertEqual(restored["model"]["training_count"], 10)
                    with restarted._connect() as connection:
                        self.assertEqual(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)

    def test_saved_run_rejects_retired_models_with_the_current_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            for architecture in (None, "gat_lstm_v1", "gat_lstm_v2_shared_type"):
                with self.subTest(architecture=architecture):
                    run_id = str(architecture)
                    config = {} if architecture is None else {"architecture_version": architecture}
                    snapshot = {"model": {"method": "gat_lstm", "config": config}, "references": []}
                    with service._connect() as connection:
                        connection.execute("INSERT INTO runs VALUES (?, ?)",
                                           (f"history:{run_id}", json.dumps(snapshot)))
                    with self.assertRaises(ReviewError) as failure:
                        service.window_seconds_for_run("history", run_id)
                    self.assertEqual(failure.exception.status_code, 409)

    def test_retired_artifacts_cannot_be_selected_or_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            identifier = "a" * 32
            for architecture in (None, "gat_lstm_v1", "gat_lstm_v1_entity", "gat_lstm_v2_shared_type"):
                config = {} if architecture is None else {"architecture_version": architecture}
                for folder in ("models", "datasets", "configurations"):
                    service._path(folder, identifier).write_text(json.dumps({"method": "gat_lstm", "config": config}))
                    with self.assertRaises(ReviewError):
                        service._read(folder, identifier)
                    self.assertEqual(service._list(folder), [])
                (Path(tmp) / "active.json").write_text(json.dumps({"model_id": identifier}))
                self.assertIsNone(service._active())

    def test_reject_legacy_and_predictive_architectures(self):
        self.assertEqual(Config().architecture_version, "gat_lstm_v3_segments_multiscale")
        with self.assertRaises(ValueError):
            Config(architecture_version="gat_lstm_v2_shared_type")
        with self.assertRaises(ValueError):
            Config(architecture_version="gat_lstm_v1_entity")
        with self.assertRaises(ValueError):
            Config(pretrain_enabled=True)

    def test_no_model_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            context = [{"entity_id": "sensor.a", "timestamp": "2026-01-01T00:00:00Z", "routing": {"terminal": False}},
                       {"entity_id": "sensor.b", "routing": {"terminal": True}}]
            result = service.enrich_contextual([], context)
            self.assertEqual(result[0]["diagnosis_mode"], "unavailable")
            self.assertEqual(result[0]["diagnosis"]["reasons"], ["classifier_unavailable"])
            self.assertNotIn("diagnosis", result[1])
            self.assertNotIn("diagnosis", context[0])
            self.assertEqual(service.status()["mode"], "unavailable")

    def test_http_rejects_retired_method_and_pretraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = DiagnosticWorkbench(Path(tmp))
            api = FastAPI(); api.include_router(crear_router_diagnostico(service))
            with TestClient(api) as client:
                self.assertEqual(client.post("/diagnostics/build", json={"method": "robust_nearest_prototype"}).status_code, 422)
                self.assertEqual(client.post("/diagnostics/train", json={"dataset_id": "a" * 32, "method": "robust_nearest_prototype"}).status_code, 400)
                self.assertEqual(client.post("/diagnostics/pretrain", json={}).status_code, 404)
                self.assertEqual(client.get("/diagnostics/sources?method=robust_nearest_prototype").status_code, 422)
