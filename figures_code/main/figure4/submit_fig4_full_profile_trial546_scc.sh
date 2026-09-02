#!/bin/bash -l
set -euo pipefail

PROJECT="${PROJECT:-ecog-eeg}"
WALLTIME="${WALLTIME:-01:00:00}"
GPUS="${GPUS:-1}"
GPU_C="${GPU_C:-6.0}"
GPU_MEMORY="${GPU_MEMORY:-40G}"
PYTHON_BIN="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"
COHERENCE="${COHERENCE:-0.5}"
DUR="${DUR:-2000}"
BASE_SEED="${BASE_SEED:-2663447816}"
TRIAL_SEED="${TRIAL_SEED:-1940034869}"
TRIAL_LABEL="${TRIAL_LABEL:-seed_${TRIAL_SEED}}"
DATASET_FILENAME="${DATASET_FILENAME:-full_profile_${TRIAL_LABEL}.npz}"
JOB_NAME="${JOB_NAME:-fig4_full_profile_${TRIAL_LABEL}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/figure4/fig4_edge_bump_full_profile_${TRIAL_LABEL}}"
LOGS_ROOT="$OUTPUT_DIR/logs"
JOB_SCRIPT="$OUTPUT_DIR/run_fig4_full_profile_${TRIAL_LABEL}_job.sh"

mkdir -p "$OUTPUT_DIR" "$LOGS_ROOT"

cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$OUTPUT_DIR/.mplconfig")
echo "[\$(date -Iseconds)] Generating Fig.4 full-profile trial seed $(printf '%q' "$TRIAL_SEED")"
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/generate_fig4_single_trial_dynamics_dataset.py") \
  --coherence $(printf '%q' "$COHERENCE") \
  --num-trials 1 \
  --dur $(printf '%q' "$DUR") \
  --seed $(printf '%q' "$BASE_SEED") \
  --trial-seeds $(printf '%q' "$TRIAL_SEED") \
  --output-dir $(printf '%q' "$OUTPUT_DIR") \
  --dataset-filename $(printf '%q' "$DATASET_FILENAME") \
  --param-config $(printf '%q' "$REPO_ROOT/figures_code/main/figure4/fig4_single_trial_full_profile_config.json") \
  --batch-index 0
EOF
chmod +x "$JOB_SCRIPT"

job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$LOGS_ROOT" \
    -N "$JOB_NAME" \
    -l "h_rt=$WALLTIME" \
    -l "gpus=$GPUS" \
    -l "gpu_c=$GPU_C" \
    -l "gpu_memory=$GPU_MEMORY" \
    "$JOB_SCRIPT"
)"

echo "Submitted Fig.4 full-profile trial job"
echo "job_id: $job_id"
echo "trial_seed: $TRIAL_SEED"
echo "trial_label: $TRIAL_LABEL"
echo "output_dir: $OUTPUT_DIR"
echo "expected_dataset: $OUTPUT_DIR/$DATASET_FILENAME"
echo "job_script: $JOB_SCRIPT"
