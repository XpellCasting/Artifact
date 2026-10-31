#!/usr/bin/env bash
# Shared environment, sourced by the public entry points.
set -euo pipefail
EXPERIMENT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${ROCM_VENV:-}" ]]; then
  if [[ ! -f "$ROCM_VENV/bin/activate" ]]; then
    printf 'No existe el entorno ROCM_VENV: %s\n' "$ROCM_VENV" >&2
    exit 1
  fi
  source "$ROCM_VENV/bin/activate"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONUNBUFFERED=1
export ARTIFACT_ROOT="$EXPERIMENT_ROOT"
cd "$EXPERIMENT_ROOT"
