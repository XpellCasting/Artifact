"""Ejecuta las pruebas desde cualquier directorio de trabajo."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-only", action="store_true")
    parser.add_argument("--frontend-only", action="store_true")
    args = parser.parse_args()
    env = {**os.environ, "PYTHONPATH": str(ROOT / "backend"), "PYTHONUTF8": "1",
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    if not args.frontend_only:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests/backend", "-v"], cwd=ROOT, env=env, check=True)
    if not args.backend_only:
        subprocess.run([shutil.which("node") or "node", "--test", "--test-concurrency=1",
                        *map(str, sorted((ROOT / "tests/frontend").glob("*.test.js")))], cwd=ROOT, env=env, check=True)
