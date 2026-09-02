#!/bin/bash -l
set -euo pipefail

if [[ "$#" -eq 3 ]]; then
  COHERENCE="$1"
  NUM_TRIALS="$2"
  OUTPUT_DIR="$3"
else
  : "${COHERENCE:?COHERENCE must be set}"
  : "${NUM_TRIALS:?NUM_TRIALS must be set}"
  : "${OUTPUT_DIR:?OUTPUT_DIR must be set}"
fi

MODEL="${MODEL:-circuit}"
DRIFT_GAIN="${DRIFT_GAIN:-1}"
NOISE_SCALE="${NOISE_SCALE:-0.3}"
DT_DDM="${DT_DDM:-5.0}"
DT_MODEL="${DT_MODEL:-1.0}"
T_START="${T_START:-10}"
DUR="${DUR:-2000}"
X0="${X0:-0.5}"
BOUNDARY="${BOUNDARY:-1.0}"
SEED="${SEED:-201}"
CHUNK_MS="${CHUNK_MS:-1000}"
SAVE_TRAJ="${SAVE_TRAJ:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="${REPO_ROOT:-$DEFAULT_REPO_ROOT}"
cd "$REPO_ROOT"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-cpu}"
CANN_DDM_PYTHON="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"

echo "[$(date -Iseconds)] Starting circuit psychometric job"
echo "model=$MODEL"
echo "coherence=$COHERENCE"
echo "num_trials=$NUM_TRIALS"
echo "output_dir=$OUTPUT_DIR"

cmd=(
  "$CANN_DDM_PYTHON" "$REPO_ROOT/scripts/simulate_psychometric_data_cDDM.py"
  --model "$MODEL"
  --coherence-values="$COHERENCE"
  --drift-gain "$DRIFT_GAIN"
  --noise-scale "$NOISE_SCALE"
  --dt-ddm "$DT_DDM"
  --dt-model "$DT_MODEL"
  --t-start "$T_START"
  --dur "$DUR"
  --x0 "$X0"
  --boundary "$BOUNDARY"
  --num-trials "$NUM_TRIALS"
  --seed "$SEED"
  --chunk-ms "$CHUNK_MS"
  --resume
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SAVE_TRAJ" == "1" ]]; then
  cmd+=(--save-traj)
fi

"${cmd[@]}"

echo "[$(date -Iseconds)] Finished circuit psychometric job"
