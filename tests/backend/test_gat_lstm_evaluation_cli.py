import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError


spec = importlib.util.spec_from_file_location(
    "gat_evaluation_cli", Path(__file__).resolve().parents[2] / "scripts/gat_lstm_evaluate.py")
CLI = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CLI)


class EvaluationCliTests(unittest.TestCase):
    def test_returns_the_persisted_flow_report_and_keeps_progress_out_of_json(self):
        flow = {"nodes": [{"type": "classifier", "config": {"model_id": "chosen-model"}}]}
        run = {"run_id": "saved-run", "summary": {"evaluation": {"labelled": True, "matched_rows": 144}},
               "snapshot": {"flow": flow, "execution_mode": "causal_future"}}
        responses = [flow, {"job_id": "evaluation", "status": "running"},
                     {"job_id": "evaluation", "status": "running", "processed": 50},
                     {"job_id": "evaluation", "status": "completed", "run": run}]
        with patch.object(CLI, "request_json", side_effect=responses) as request, \
                patch.object(CLI.time, "sleep"), \
                patch("sys.stdout", new_callable=io.StringIO) as output, \
                patch("sys.stderr", new_callable=io.StringIO) as progress:
            self.assertEqual(CLI.main(["--flow-id", "saved-flow", "--base", "http://localhost:8000/"]), 0)
        self.assertEqual(json.loads(output.getvalue()), run)
        self.assertIn("job_id=evaluation", progress.getvalue())
        mutations = [call for call in request.call_args_list if len(call.args) == 3]
        self.assertEqual(len(mutations), 1)
        self.assertEqual(mutations[0].args, ("http://localhost:8000", "/flows/saved-flow/runs", {}))

    def test_missing_or_disabled_classifier_does_not_start_an_evaluation(self):
        for nodes in ([], [{"type": "classifier", "config": {"enabled": False}}]):
            with self.subTest(nodes=nodes), \
                    patch.object(CLI, "request_json", return_value={"nodes": nodes}) as request:
                with self.assertRaisesRegex(ValueError, "clasificador GAT-LSTM habilitado"):
                    CLI.evaluate_flow("http://localhost:8000", "flow", 10)
                self.assertEqual(request.call_count, 1)

    def test_failed_job_reports_the_server_error(self):
        responses = [{"nodes": [{"type": "classifier"}]},
                     {"job_id": "evaluation", "status": "running"},
                     {"job_id": "evaluation", "status": "failed", "error": "modelo incompatible"}]
        with patch.object(CLI, "request_json", side_effect=responses), patch("sys.stderr", io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "evaluation.*modelo incompatible"):
                CLI.evaluate_flow("http://localhost:8000", "flow", 10)

    def test_timeout_preserves_the_job_identifier_without_starting_another_job(self):
        responses = [{"nodes": [{"type": "classifier"}]},
                     {"job_id": "evaluation", "status": "running"}]
        with patch.object(CLI, "request_json", side_effect=responses) as request, \
                patch.object(CLI.time, "monotonic", side_effect=[0, 2]), patch("sys.stderr", io.StringIO()):
            with self.assertRaisesRegex(TimeoutError, "/flow-evaluations/evaluation"):
                CLI.evaluate_flow("http://localhost:8000", "flow", 1)
        self.assertEqual(request.call_count, 2)

    def test_http_rejection_preserves_the_required_labelled_csv_message(self):
        error = HTTPError("http://localhost:8000/flows/flow/runs", 400, "Bad Request", {},
                          io.BytesIO(b'{"detail":"Carga un conjunto CSV etiquetado"}'))
        with patch.object(CLI, "urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "400.*CSV etiquetado"):
                CLI.request_json("http://localhost:8000", "/flows/flow/runs", {})

    def test_invalid_wait_is_rejected_before_any_request(self):
        with patch.object(CLI, "request_json") as request, patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                CLI.main(["--flow-id", "flow", "--wait-seconds", "0"])
        self.assertEqual(error.exception.code, 2)
        request.assert_not_called()
