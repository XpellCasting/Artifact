#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/experimento_env.sh"
"$PYTHON_BIN" -c 'import torch; assert torch.version.hip and torch.cuda.is_available(), "Activa tu entorno PyTorch ROCm existente; no se reinstala automáticamente"; print(torch.cuda.get_device_name(0))'
exec "$PYTHON_BIN" scripts/escenarios_experimento.py train "$@"
