"""Run equation-based verification; exits nonzero when a defect is detected."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import sys
import unittest
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'tests/backend'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
        for name in ('test_gat_lstm_reference', 'test_segment_multiscale'))
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    files = ['backend/app/modulos/clasificador/network_segments.py',
             'backend/app/modulos/clasificador/segments.py',
             'tests/backend/test_gat_lstm_reference.py',
             'tests/backend/test_segment_multiscale.py']
    payload = {
        'verification_status': 'PASSED' if result.wasSuccessful() else 'DEFECT_DETECTED',
        'scope': 'documented equations and integration; no benchmark accuracy claim',
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'torch_geometric': version('torch-geometric'), 'device': 'cpu',
                        'reference_dtype': 'float64', 'seed': 42, 'threads': torch.get_num_threads()},
        'tolerance': {'atol': 1e-10, 'rtol': 1e-9},
        'tests_run': result.testsRun,
        'failures': [{'test': str(test), 'traceback': trace} for test, trace in result.failures],
        'errors': [{'test': str(test), 'traceback': trace} for test, trace in result.errors],
        'skipped': result.skipped,
        'sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        'sources': [
            'https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html',
            'https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html',
            'https://arxiv.org/html/2410.11923v1'],
        'log': stream.getvalue(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: payload[key] for key in ('verification_status', 'tests_run', 'environment')}, indent=2))
    print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
