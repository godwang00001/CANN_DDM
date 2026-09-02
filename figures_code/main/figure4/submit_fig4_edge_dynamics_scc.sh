#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_fig4_edge_dynamics_scc.sh <run_name> [options]

Options:
  --project <name>          Default: ecog-eeg
  --walltime <HH:MM:SS>     Default: 04:00:00
  --gpus <int>              Default: 1
  --gpu-c <value>           Default: 6.0
  --gpu-memory <value>      Default: 40G
  --output-root <path>      Default: results/figure4
  --conditions <csv>        Default: -1,-0.5,-0.25,-0.125,0.125,0.25,0.5,1
  --num-trials <int>        Default: 100
  --dur <int>               Default: 2000
  --dry-run                 Generate the worker job script and exit without qsub
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
OUTPUT_ROOT="results/figure4"
CONDITIONS="-1,-0.5,-0.25,-0.125,0.125,0.25,0.5,1"
NUM_TRIALS="100"
DUR="2000"
DRY_RUN="0"

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
    --dry-run) DRY_RUN="1"; shift 1 ;;
    *) usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ "$OUTPUT_ROOT" = /* ]]; then
  RUN_ROOT="$OUTPUT_ROOT/$RUN_NAME"
else
  RUN_ROOT="$REPO_ROOT/$OUTPUT_ROOT/$RUN_NAME"
fi

LOGS_ROOT="$RUN_ROOT/logs"
mkdir -p "$RUN_ROOT" "$LOGS_ROOT"

JOB_SCRIPT="$RUN_ROOT/run_fig4_edge_dynamics_job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
export MPLCONFIGDIR=$(printf '%q' "$RUN_ROOT/.mplconfig")
echo "[\$(date -Iseconds)] Starting Fig.4 edge-dynamics dataset run"
echo "run_name=$(printf '%q' "$RUN_NAME")"
echo "run_root=$(printf '%q' "$RUN_ROOT")"
echo "conditions=$(printf '%q' "$CONDITIONS")"
echo "num_trials=$(printf '%q' "$NUM_TRIALS")"
echo "dur=$(printf '%q' "$DUR")"
exec $(printf '%q' "$PYTHON_BIN") $(printf '%q' "$REPO_ROOT/scripts/generate_fig3_psychometric_dataset.py") \
  all \
  --run-name $(printf '%q' "$RUN_NAME") \
  --output-root $(printf '%q' "$(dirname "$RUN_ROOT")") \
  --conditions=$(printf '%q' "$CONDITIONS") \
  --num-trials $(printf '%q' "$NUM_TRIALS") \
  --dur $(printf '%q' "$DUR") \
  --save-r-e
EOF
chmod +x "$JOB_SCRIPT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry run only"
  echo "run_name: $RUN_NAME"
  echo "run_root: $RUN_ROOT"
  echo "job_script: $JOB_SCRIPT"
  echo "repo_root: $REPO_ROOT"
  echo "python_bin: $PYTHON_BIN"
  echo "generator_script: $REPO_ROOT/scripts/generate_fig3_psychometric_dataset.py"
  exit 0
fi

job_name="fig4_dyn_${RUN_NAME}"
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
    "$JOB_SCRIPT"
)"

echo "Submitted Fig.4 edge-dynamics job"
echo "run_name: $RUN_NAME"
echo "run_root: $RUN_ROOT"
echo "job_id: $job_id"
echo "job_script: $JOB_SCRIPT"
echo "gpus: $GPUS"
echo "gpu_c: $GPU_C"
echo "gpu_memory: $GPU_MEMORY"
echo "expected outputs:"
echo "  $RUN_ROOT/dataset.npz"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/config.json"
