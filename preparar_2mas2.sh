#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Uso: bash preparar_2mas2.sh /ruta/2mesesDesarrollo_etiquetado.csv /ruta/2mesesEvaluacion.csv [carpeta_salida_nueva]' >&2
  exit 2
fi
output_path="${3:-local/entrenamiento_2mas2}"
python scripts/gat_lstm_prepare_periods.py "$1" "$2" --output "$output_path"
