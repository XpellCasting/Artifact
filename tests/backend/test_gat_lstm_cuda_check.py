import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("gat_lstm_check", ROOT / "scripts" / "gat_lstm_check.py")
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class CudaCheckTests(unittest.TestCase):
    def test_missing_dependency_reports_actionable_error(self):
        with patch.dict(sys.modules, {"torch": None}):
            result = CHECK.check_environment("cuda", require_cuda=True)
        self.assertFalse(result["ok"])
        self.assertIn("ModuleNotFoundError", result["error"])
        self.assertIn("training/README.md", result["help"])

    def test_cli_persists_failed_report_and_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "check.json"
            with patch.object(CHECK, "check_environment", return_value={"ok": False, "error": "missing GPU"}):
                with patch("builtins.print"):
                    code = CHECK.main(["--device", "cuda", "--require-cuda", "--output", str(path)])
            self.assertEqual(code, 1)
            self.assertIn("missing GPU", path.read_text(encoding="utf-8"))

    def test_real_cpu_network_backward_and_reload(self):
        try:
            import torch
            import torch_geometric  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch/PyG no instalados")
        result = CHECK.check_environment("cpu")
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["selected_device"], "cpu")
        self.assertTrue(result["smoke_test"]["fault_type_head"])
        self.assertTrue(result["smoke_test"]["cpu_reload"])

    def test_require_cuda_never_accepts_cpu(self):
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch/PyG no instalados")
        result = CHECK.check_environment("cpu", require_cuda=True)
        self.assertFalse(result["ok"])
        self.assertIn("Se requiere CUDA", result["error"])


if __name__ == "__main__":
    unittest.main()
