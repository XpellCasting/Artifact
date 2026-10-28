#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
output_path="${1:-local/entrenamiento_2mas2}"
if [[ $# -gt 2 || ( $# -eq 2 && "$2" != '--resume' ) ]]; then
  echo 'Uso: bash entrenar_2mas2.sh [carpeta_preparada] [--resume]' >&2
  exit 2
fi
dataset_id="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_id"])' "$output_path/dataset.json")"
python scripts/gat_lstm_verify_evaluation.py "$output_path"
resume_args=()
if [[ "${2:-}" == '--resume' ]]; then resume_args+=(--resume); fi
python scripts/gat_lstm_train.py "$dataset_id" \
  --review-dir "$output_path/review" --device "${GAT_DEVICE:-cuda:0}" \
  --batch-size "${GAT_BATCH_SIZE:-16}" --checkpoint-dir "$output_path/checkpoints" \
  --result "$output_path/trained_model.json" "${resume_args[@]}"
python scripts/gat_lstm_export_candidate.py \
  "$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_id"])' "$output_path/trained_model.json")" \
  --review-dir "$output_path/review" --output "$output_path/modelo_2mas2.zip"
