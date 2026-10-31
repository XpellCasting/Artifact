#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experimento_env.sh"
exec "$PYTHON_BIN" scripts/escenarios_experimento.py cleanup "$@"
