#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_fig3_psychometric_scc.sh <run_name> [options]

Options:
  --project <name>          Default: ecog-eeg
  --walltime <HH:MM:SS>     Default: 04:00:00
  --gpus <int>              Default: 1
  --gpu-c <value>           Default: 6.0
  --gpu-memory <value>      Default: 40G
  --output-root <path>      Default: results/figure3
  --conditions <csv>        Default: -1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0
  --num-trials <int>        Default: 1000
  --dur <int>               Default: 4000
EOF
  exit 2
}

if [[ "$#" -lt 1 ]]; then
  usage
fi

RUN_NAME="$1"
shift

PROJECT="ecog-eeg"
WALLTIME="04:00:00"
GPUS="1"
GPU_C="6.0"
GPU_MEMORY="40G"
PYTHON_BIN="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"
OUTPUT_ROOT="results/figure3"
CONDITIONS="-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"
NUM_TRIALS="10000"
DUR="4000"
BATCH_TRIALS="1000"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --gpu-c) GPU_C="$2"; shift 2 ;;
    --gpu-memory) GPU_MEMORY="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --conditions) CONDITIONS="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --dur) DUR="$2"; shift 2 ;;
    *) usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$OUTPUT_ROOT" = /* ]]; then
  RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
else
  RUN_ROOT="$REPO_ROOT/$OUTPUT_ROOT/$RUN_NAME"
fi

CONDITIONS_ROOT="$RUN_ROOT/conditions"
WORKER_LOGS_ROOT="$RUN_ROOT/logs"
MERGE_LOGS_ROOT="$RUN_ROOT/merge_logs"
mkdir -p "$CONDITIONS_ROOT" "$WORKER_LOGS_ROOT" "$MERGE_LOGS_ROOT"
CALIBRATION_FILE="$RUN_ROOT/shared_calibration.json"

"$PYTHON_BIN" "$REPO_ROOT/scripts/generate_fig3_psychometric_dataset.py" \
  prepare-run \
  --run-root "$RUN_ROOT" \
  --dur "$DUR"

WORKER_SCRIPT="$RUN_ROOT/run_fig3_psychometric_condition_job.sh"
MERGE_SCRIPT="$RUN_ROOT/run_fig3_psychometric_merge_job.sh"

if [[ "$NUM_TRIALS" =~ ^[0-9]+$ ]] && [[ "$BATCH_TRIALS" =~ ^[0-9]+$ ]]; then
  NUM_BATCHES=$(( (NUM_TRIALS + BATCH_TRIALS - 1) / BATCH_TRIALS ))
else
  echo "num-trials and batch size must be positive integers" >&2
  exit 1
fi

cat > "$WORKER_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$RUN_ROOT/.mplconfig")
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/generate_fig3_psychometric_dataset.py") \
  single-condition \
  --condition "\$1" \
  --condition-index "\$2" \
  --output-dir "\$3" \
  --batch-index "\$4" \
  --calibration-file $(printf '%q' "$CALIBRATION_FILE") \
  --num-trials "\$5" \
  --dur $(printf '%q' "$DUR")
EOF
chmod +x "$WORKER_SCRIPT"

cat > "$MERGE_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$RUN_ROOT/.mplconfig")
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/generate_fig3_psychometric_dataset.py") \
  merge \
  --run-root $(printf '%q' "$RUN_ROOT") \
  --conditions=$(printf '%q' "$CONDITIONS") \
  --num-trials $(printf '%q' "$NUM_TRIALS") \
  --dur $(printf '%q' "$DUR")
EOF
chmod +x "$MERGE_SCRIPT"

IFS=',' read -r -a condition_list <<< "$CONDITIONS"
if [[ "${#condition_list[@]}" -eq 0 ]]; then
  echo "No conditions provided" >&2
  exit 1
fi

worker_job_ids=()
for index in "${!condition_list[@]}"; do
  condition="${condition_list[$index]}"
  condition="${condition//[[:space:]]/}"
  if [[ -z "$condition" ]]; then
    continue
  fi
  condition_slug="$(
    PYTHONPATH="$REPO_ROOT" "$PYTHON_BIN" -c \
      "from scripts.generate_fig3_psychometric_dataset import drift_label; print(drift_label(float('$condition')))"
  )"
  condition_root="$CONDITIONS_ROOT/$condition_slug"
  mkdir -p "$condition_root"
  remaining_trials="$NUM_TRIALS"
  for ((batch_index=0; batch_index<NUM_BATCHES; batch_index++)); do
    if (( remaining_trials <= 0 )); then
      break
    fi
    batch_trials="$BATCH_TRIALS"
    if (( remaining_trials < batch_trials )); then
      batch_trials="$remaining_trials"
    fi
    remaining_trials=$(( remaining_trials - batch_trials ))
    if (( NUM_BATCHES > 1 )); then
      batch_output_dir="$condition_root/$(printf 'batch_%03d' "$batch_index")"
      job_name="fig3_psy_${RUN_NAME}_${condition_slug}_b$(printf '%03d' "$batch_index")"
    else
      batch_output_dir="$condition_root"
      job_name="fig3_psy_${RUN_NAME}_${condition_slug}"
    fi
    mkdir -p "$batch_output_dir"
    echo "Submitting condition=$condition index=$index batch=$batch_index batch_trials=$batch_trials job_name=$job_name"
    job_id="$(
      qsub -terse \
        -P "$PROJECT" \
        -cwd \
        -V \
        -j y \
        -o "$WORKER_LOGS_ROOT" \
        -N "$job_name" \
        -l "h_rt=$WALLTIME" \
        -l "gpus=$GPUS" \
        -l "gpu_c=$GPU_C" \
        -l "gpu_memory=$GPU_MEMORY" \
        "$WORKER_SCRIPT" \
        "$condition" \
        "$index" \
        "$batch_output_dir" \
        "$batch_index" \
        "$batch_trials"
    )"
    if [[ -z "$job_id" ]]; then
      echo "Failed to capture job id for condition $condition batch $batch_index" >&2
      exit 1
    fi
    worker_job_ids+=("$job_id")
  done
done

hold_jid="$(IFS=,; echo "${worker_job_ids[*]}")"
merge_job_name="fig3_psy_merge_${RUN_NAME}"
merge_job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$MERGE_LOGS_ROOT" \
    -N "$merge_job_name" \
    -hold_jid "$hold_jid" \
    -l "h_rt=01:00:00" \
    "$MERGE_SCRIPT"
)"

echo "Submitted Figure 3 psychometric jobs"
echo "run_name: $RUN_NAME"
echo "run_root: $RUN_ROOT"
echo "worker_job_ids: $hold_jid"
echo "num_batches_per_condition: $NUM_BATCHES"
echo "merge_job_id: $merge_job_id"
echo "calibration_file: $CALIBRATION_FILE"
echo "gpus: $GPUS"
echo "gpu_c: $GPU_C"
echo "gpu_memory: $GPU_MEMORY"
echo "expected outputs:"
echo "  $RUN_ROOT/dataset.npz"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/config.json"
