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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"
CANN_DDM_PYTHON="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"
MERGE_SCRIPT="$REPO_ROOT/scripts/merge_circuit_psychometric_outputs.py"
COMBINE_SCRIPT="$REPO_ROOT/scripts/combine_psychometric_model_datasets.py"

echo "[$(date -Iseconds)] Starting psychometric finalizer"
echo "repo_root=$REPO_ROOT"
echo "run_root=$RUN_ROOT"
echo "ddm_root=$DDM_ROOT"
echo "circuit_root=$CIRCUIT_ROOT"
echo "merge_script=$MERGE_SCRIPT"
echo "combine_script=$COMBINE_SCRIPT"

if [[ ! -f "$MERGE_SCRIPT" ]]; then
  echo "Missing merge script: $MERGE_SCRIPT" >&2
  exit 1
fi
if [[ ! -f "$COMBINE_SCRIPT" ]]; then
  echo "Missing combine script: $COMBINE_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$DDM_ROOT" ]]; then
  echo "Missing DDM root: $DDM_ROOT" >&2
  exit 1
fi
if [[ ! -d "$CIRCUIT_ROOT" ]]; then
  echo "Missing circuit root: $CIRCUIT_ROOT" >&2
  exit 1
fi

"$CANN_DDM_PYTHON" "$MERGE_SCRIPT" \
  "$CIRCUIT_ROOT"

"$CANN_DDM_PYTHON" "$COMBINE_SCRIPT" \
  --ddm-root "$DDM_ROOT" \
  --circuit-root "$CIRCUIT_ROOT" \
  --output-root "$RUN_ROOT" \
  --delete-intermediates

echo "[$(date -Iseconds)] Finished psychometric finalizer"
