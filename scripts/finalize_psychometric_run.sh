#!/bin/bash -l
set -euo pipefail

if [[ "$#" -eq 3 ]]; then
  RUN_ROOT="$1"
  DDM_ROOT="$2"
  CIRCUIT_ROOT="$3"
else
  : "${RUN_ROOT:?RUN_ROOT must be set}"
  : "${DDM_ROOT:?DDM_ROOT must be set}"
  : "${CIRCUIT_ROOT:?CIRCUIT_ROOT must be set}"
fi

if [[ -n "${SGE_O_WORKDIR:-}" ]]; then
  REPO_ROOT="$SGE_O_WORKDIR"
else
  REPO_ROOT="$(pwd -P)"
fi
cd "$REPO_ROOT"

echo "[$(date -Iseconds)] Starting psychometric finalizer"
echo "run_root=$RUN_ROOT"
echo "ddm_root=$DDM_ROOT"
echo "circuit_root=$CIRCUIT_ROOT"

conda run -n cann_ddm_v2 python "$REPO_ROOT/scripts/merge_circuit_psychometric_outputs.py" \
  "$CIRCUIT_ROOT"

conda run -n cann_ddm_v2 python "$REPO_ROOT/scripts/combine_psychometric_model_datasets.py" \
  --ddm-root "$DDM_ROOT" \
  --circuit-root "$CIRCUIT_ROOT" \
  --output-root "$RUN_ROOT" \
  --delete-intermediates

echo "[$(date -Iseconds)] Finished psychometric finalizer"
