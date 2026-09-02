#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_fig3_cddm_two_condition_scc.sh <run_name> [options]

Options:
  --project <name>          Default: ecog-eeg
  --walltime <HH:MM:SS>     Default: 01:00:00
  --python-bin <path>       Default: /projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python
  --output-root <path>      Default: results/figure3
  --dt-ddm <float>          Default: 5.0
  --dt-model <float>        Default: 1.0
  --t-start <int>           Default: 10
  --dur <int>               Default: 2000
  --max-time <int>          Default: dur
  --x0 <float>              Default: 0.5
  --boundary <float>        Default: 1.0
  --seed <int>              Default: 201
  --save-traj               Save trajectories explicitly
  --no-save-traj            Disable trajectory saving
EOF
  exit 2
}

if [[ "$#" -lt 1 ]]; then
  usage
fi

RUN_NAME="$1"
shift

PROJECT="ecog-eeg"
WALLTIME="01:00:00"
PYTHON_BIN="/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python"
OUTPUT_ROOT="results/figure3"
DT_DDM="5.0"
DT_MODEL="1.0"
T_START="10"
DUR="2000"
MAX_TIME="2000"
X0="0.5"
BOUNDARY="1.0"
SEED="201"
SAVE_TRAJ="1"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --dt-ddm) DT_DDM="$2"; shift 2 ;;
    --dt-model) DT_MODEL="$2"; shift 2 ;;
    --t-start) T_START="$2"; shift 2 ;;
    --dur) DUR="$2"; shift 2 ;;
    --max-time) MAX_TIME="$2"; shift 2 ;;
    --x0) X0="$2"; shift 2 ;;
    --boundary) BOUNDARY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --save-traj) SAVE_TRAJ="1"; shift ;;
    --no-save-traj) SAVE_TRAJ="0"; shift ;;
    *) usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$REPO_ROOT/$OUTPUT_ROOT/$RUN_NAME"
LOGS_ROOT="$RUN_ROOT/logs"
mkdir -p "$LOGS_ROOT"

job_name="fig3_cddm_${RUN_NAME}"
cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/scripts/generate_fig3_cddm_two_condition_dataset.py"
  --run-name "$RUN_NAME"
  --output-root "$OUTPUT_ROOT"
  --dt-ddm "$DT_DDM"
  --dt-model "$DT_MODEL"
  --t-start "$T_START"
  --dur "$DUR"
  --x0 "$X0"
  --boundary "$BOUNDARY"
  --seed "$SEED"
)

if [[ -n "$MAX_TIME" ]]; then
  cmd+=(--max-time "$MAX_TIME")
fi

if [[ "$SAVE_TRAJ" == "1" ]]; then
  cmd+=(--save-traj)
else
  cmd+=(--no-save-traj)
fi

printf -v qsub_command '%q ' "${cmd[@]}"
qsub_command="${qsub_command% }"

JOB_SCRIPT="$RUN_ROOT/run_fig3_cddm_two_condition_job.sh"
cat > "$JOB_SCRIPT" <<EOF
#!/bin/bash -l
set -euo pipefail
cd $(printf '%q' "$REPO_ROOT")
exec $qsub_command
EOF
chmod +x "$JOB_SCRIPT"

job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$LOGS_ROOT" \
    -N "$job_name" \
    -l "h_rt=$WALLTIME" \
    "$JOB_SCRIPT"
)"

echo "Submitted Figure 3 cDDM job"
echo "run_name: $RUN_NAME"
echo "run_root: $RUN_ROOT"
echo "job_name: $job_name"
echo "job_id: $job_id"
echo "expected outputs:"
echo "  $RUN_ROOT/dataset.npz"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/config.json"
