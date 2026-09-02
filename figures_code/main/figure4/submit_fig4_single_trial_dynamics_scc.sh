#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_fig4_single_trial_dynamics_scc.sh [options]

Options:
  --project <name>                Default: ecog-eeg
  --walltime <HH:MM:SS>           Default: 04:00:00
  --gpus <int>                    Default: 1
  --gpu-c <value>                 Default: 6.0
  --gpu-memory <value>            Default: 40G
  --output-dir <path>             Default: results/figure4/fig4_single_trial_dynamics_tuned_bump
  --param-config <path>           Default: figures_code/main/figure4/fig4_single_trial_dynamics_config.json
  --coherence <float>             Single-coherence mode default: 0.5
  --coherences "<floats>"         Multi-coherence mode, e.g. "0.5 0.25 -0.25 -0.5"
  --num-trials <int>              Single-coherence total trials. Default: 100
  --trials-per-coherence <int>    Multi-coherence trials per condition
  --batch-trials <int>            Default: 100
  --noise-scale <float>           Decision-making noise scale. Default: 0.5
  --dur <int>                     Default: 2000
  --seed <int>                    Default: 201
  --dry-run                       Generate worker/merge scripts and print planned jobs without qsub
EOF
  exit 2
}

PROJECT="ecog-eeg"
WALLTIME="04:00:00"
GPUS="1"
GPU_C="6.0"
GPU_MEMORY="40G"
PYTHON_BIN="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"
OUTPUT_DIR="results/figure4/dv_receptive_fields"
PARAM_CONFIG="figures_code/main/figure4/fig4_dv_receptive_fields_config.json"
COHERENCES="0.5 0.25 -0.25 -0.5"
NUM_TRIALS="1000"
TRIALS_PER_COHERENCE="250"
BATCH_TRIALS="250"
NOISE_SCALE="0"
DUR="2000"
SEED="201"
DRY_RUN="0"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-c) GPU_C="$2"; shift 2 ;;
    --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --param-config) PARAM_CONFIG="$2"; shift 2 ;;
    --coherence) COHERENCE="$2"; shift 2 ;;
    --coherences) COHERENCES="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --trials-per-coherence) TRIALS_PER_COHERENCE="$2"; shift 2 ;;
    --batch-trials) BATCH_TRIALS="$2"; shift 2 ;;
    --noise-scale) NOISE_SCALE="$2"; shift 2 ;;
    --dur) DUR="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN="1"; shift 1 ;;
    *) usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "$OUTPUT_DIR" = /* ]]; then
  RUN_ROOT="$OUTPUT_DIR"
else
  RUN_ROOT="$REPO_ROOT/$OUTPUT_DIR"
fi
if [[ "$PARAM_CONFIG" = /* ]]; then
  PARAM_CONFIG_PATH="$PARAM_CONFIG"
else
  PARAM_CONFIG_PATH="$REPO_ROOT/$PARAM_CONFIG"
fi

LOGS_ROOT="$RUN_ROOT/logs"
MERGE_LOGS_ROOT="$RUN_ROOT/merge_logs"
BATCHES_ROOT="$RUN_ROOT/batches"
mkdir -p "$RUN_ROOT" "$LOGS_ROOT" "$MERGE_LOGS_ROOT" "$BATCHES_ROOT"

if ! [[ "$BATCH_TRIALS" =~ ^[0-9]+$ ]] || (( BATCH_TRIALS <= 0 )); then
  echo "batch-trials must be a positive integer" >&2
  exit 1
fi

coherence_label() {
  local value="$1"
  value="${value#+}"
  value="${value/-/m}"
  value="${value//./p}"
  echo "$value"
}

WORKER_SCRIPT="$RUN_ROOT/run_fig4_single_trial_dynamics_batch_job.sh"
MERGE_SCRIPT="$RUN_ROOT/run_fig4_single_trial_dynamics_merge_job.sh"
cat > "$WORKER_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$RUN_ROOT/.mplconfig")
echo "[\$(date -Iseconds)] Starting Fig.4 single-trial dynamics batch run"
echo "run_root=$(printf '%q' "$RUN_ROOT")"
echo "batch_index=\$1"
echo "batch_trials=\$2"
echo "batch_seed=\$3"
echo "batch_output_dir=\$4"
echo "coherence=\$5"
echo "noise_scale=$(printf '%q' "$NOISE_SCALE")"
echo "dur=$(printf '%q' "$DUR")"
echo "param_config=$(printf '%q' "$PARAM_CONFIG_PATH")"
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/generate_fig4_single_trial_dynamics_dataset.py") \
  --coherence "\$5" \
  --noise-scale $(printf '%q' "$NOISE_SCALE") \
  --num-trials "\$2" \
  --dur $(printf '%q' "$DUR") \
  --seed "\$3" \
  --output-dir "\$4" \
  --param-config $(printf '%q' "$PARAM_CONFIG_PATH") \
  --batch-index "\$1"
EOF
chmod +x "$WORKER_SCRIPT"

cat > "$MERGE_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$RUN_ROOT/.mplconfig")
echo "[\$(date -Iseconds)] Merging Fig.4 single-trial dynamics batches"
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/merge_fig4_single_trial_dynamics_batches.py") \
  $(printf '%q' "$RUN_ROOT")
EOF
chmod +x "$MERGE_SCRIPT"

coherence_values=()
trials_per_condition=()
if [[ -n "$COHERENCES" ]]; then
  read -r -a coherence_values <<< "$COHERENCES"
  if (( ${#coherence_values[@]} == 0 )); then
    echo "--coherences must include at least one value" >&2
    exit 1
  fi
  if [[ -n "$TRIALS_PER_COHERENCE" ]]; then
    if ! [[ "$TRIALS_PER_COHERENCE" =~ ^[0-9]+$ ]] || (( TRIALS_PER_COHERENCE <= 0 )); then
      echo "trials-per-coherence must be a positive integer" >&2
      exit 1
    fi
    for _coherence in "${coherence_values[@]}"; do
      trials_per_condition+=("$TRIALS_PER_COHERENCE")
    done
  else
    if ! [[ "$NUM_TRIALS" =~ ^[0-9]+$ ]] || (( NUM_TRIALS <= 0 )); then
      echo "num-trials must be a positive integer" >&2
      exit 1
    fi
    if (( NUM_TRIALS % ${#coherence_values[@]} != 0 )); then
      echo "num-trials must be divisible by the number of coherences when --trials-per-coherence is omitted" >&2
      exit 1
    fi
    per_condition=$(( NUM_TRIALS / ${#coherence_values[@]} ))
    for _coherence in "${coherence_values[@]}"; do
      trials_per_condition+=("$per_condition")
    done
  fi
else
  if ! [[ "$NUM_TRIALS" =~ ^[0-9]+$ ]] || (( NUM_TRIALS <= 0 )); then
    echo "num-trials must be a positive integer" >&2
    exit 1
  fi
  coherence_values=("$COHERENCE")
  trials_per_condition=("$NUM_TRIALS")
fi

total_trials=0
num_worker_jobs=0
for condition_index in "${!coherence_values[@]}"; do
  condition_trials="${trials_per_condition[$condition_index]}"
  total_trials=$(( total_trials + condition_trials ))
  num_worker_jobs=$(( num_worker_jobs + (condition_trials + BATCH_TRIALS - 1) / BATCH_TRIALS ))
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only"
  echo "run_root: $RUN_ROOT"
  echo "worker_script: $WORKER_SCRIPT"
  echo "merge_script: $MERGE_SCRIPT"
  echo "total_trials: $total_trials"
  echo "batch_trials: $BATCH_TRIALS"
  echo "noise_scale: $NOISE_SCALE"
  echo "num_worker_jobs: $num_worker_jobs"
  echo "coherences: ${coherence_values[*]}"
  echo "trials_per_condition: ${trials_per_condition[*]}"
  echo "repo_root: $REPO_ROOT"
  echo "python_bin: $PYTHON_BIN"
  echo "generator_script: $REPO_ROOT/scripts/generate_fig4_single_trial_dynamics_dataset.py"
  echo "merge_python_script: $REPO_ROOT/scripts/merge_fig4_single_trial_dynamics_batches.py"
  echo "param_config: $PARAM_CONFIG_PATH"
  exit 0
fi

worker_job_ids=()
global_batch_index=0
for condition_index in "${!coherence_values[@]}"; do
  condition_coherence="${coherence_values[$condition_index]}"
  condition_trials="${trials_per_condition[$condition_index]}"
  condition_remaining_trials="$condition_trials"
  condition_label="$(coherence_label "$condition_coherence")"
  condition_batches_root="$BATCHES_ROOT/coh_$condition_label"
  mkdir -p "$condition_batches_root"
  condition_batch_index=0
  while (( condition_remaining_trials > 0 )); do
    batch_trials="$BATCH_TRIALS"
    if (( condition_remaining_trials < batch_trials )); then
      batch_trials="$condition_remaining_trials"
    fi
    condition_remaining_trials=$(( condition_remaining_trials - batch_trials ))
    batch_output_dir="$condition_batches_root/$(printf 'batch_%03d' "$condition_batch_index")"
    mkdir -p "$batch_output_dir"
    batch_seed="$(
      "$PYTHON_BIN" -c "import numpy as np; print(int(np.random.SeedSequence(int('$SEED')).spawn($((global_batch_index + 1)))[$global_batch_index].generate_state(1)[0]))"
    )"
    job_name="fig4_${condition_label}_b$(printf '%03d' "$condition_batch_index")"
    echo "Submitting coherence=$condition_coherence condition_batch=$condition_batch_index global_batch=$global_batch_index batch_trials=$batch_trials seed=$batch_seed job_name=$job_name"
    job_id="$(
      qsub -terse \
        -P "$PROJECT" \
        -cwd \
        -V \
        -j y \
        -o "$LOGS_ROOT" \
        -N "$job_name" \
        -l "h_rt=$WALLTIME" \
        -l "gpus=$GPUS" \
        -l "gpu_c=$GPU_C" \
        -l "gpu_memory=$GPU_MEMORY" \
        "$WORKER_SCRIPT" \
        "$global_batch_index" \
        "$batch_trials" \
        "$batch_seed" \
        "$batch_output_dir" \
        "$condition_coherence"
    )"
    if [[ -z "$job_id" ]]; then
      echo "Failed to capture job id for coherence $condition_coherence batch $condition_batch_index" >&2
      exit 1
    fi
    worker_job_ids+=("$job_id")
    condition_batch_index=$(( condition_batch_index + 1 ))
    global_batch_index=$(( global_batch_index + 1 ))
  done
done

hold_jid="$(IFS=,; echo "${worker_job_ids[*]}")"
merge_job_name="fig4_single_trial_merge"
merge_job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$MERGE_LOGS_ROOT" \
    -N "$merge_job_name" \
    -hold_jid "$hold_jid" \
    -l "h_rt=02:00:00" \
    "$MERGE_SCRIPT"
)"

echo "Submitted Fig.4 single-trial dynamics job"
echo "run_root: $RUN_ROOT"
echo "worker_job_ids: $hold_jid"
echo "num_worker_jobs: ${#worker_job_ids[@]}"
echo "batch_trials: $BATCH_TRIALS"
echo "noise_scale: $NOISE_SCALE"
echo "coherences: ${coherence_values[*]}"
echo "trials_per_condition: ${trials_per_condition[*]}"
echo "merge_job_id: $merge_job_id"
echo "worker_script: $WORKER_SCRIPT"
echo "merge_script: $MERGE_SCRIPT"
echo "param_config: $PARAM_CONFIG_PATH"
echo "gpus: $GPUS"
echo "gpu_c: $GPU_C"
echo "gpu_memory: $GPU_MEMORY"
echo "expected outputs:"
echo "  $RUN_ROOT/single_condition_dynamics.npz"
echo "  $RUN_ROOT/trial_summary.csv"
echo "  $RUN_ROOT/config.json"
