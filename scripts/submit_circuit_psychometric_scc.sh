#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_circuit_psychometric_scc.sh <run_name> [options]

Scheduler options:
  --project <name>                Default: ecog-eeg
  --walltime <HH:MM:SS>           Default: 01:00:00

Workflow options:
  --num-trials <int>              Default: 200
  --num-batches <int>             Default: 1
  --circuit-batch-trials <int>    Default: --num-trials
  --coherence-values <csv>        Default: -1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0
  --drift-gain <float>            Default: 1
  --noise-scale <float>           Default: 0.5
  --dt-ddm <float>                Default: 5.0
  --dt-model <float>              Default: 1.0
  --t-start <int>                 Default: 10
  --dur <int>                     Default: 2000
  --x0 <float>                    Default: 0.5
  --boundary <float>              Default: 1.0
  --seed <int>                    Default: 201
  --chunk-ms <int>                Default: 1000
  --output-root <path>            Default: results/psychometric
  --python-bin <path>             Default: CANN_DDM_PYTHON or cann_ddm_v2 python
  --save-traj                     Default: off
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
NUM_TRIALS="200"
NUM_BATCHES="1"
CIRCUIT_BATCH_TRIALS=""
COHERENCE_VALUES="-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"
DRIFT_GAIN="1.0"
NOISE_SCALE="0.5"
DT_DDM="5.0"
DT_MODEL="1.0"
T_START="10"
DUR="2000"
X0="0.5"
BOUNDARY="1.0"
SEED="201"
CHUNK_MS="1000"
OUTPUT_ROOT="results/psychometric"
SAVE_TRAJ="0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_DDM_PYTHON="${CANN_DDM_PYTHON:-/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --num-batches) NUM_BATCHES="$2"; shift 2 ;;
    --circuit-batch-trials) CIRCUIT_BATCH_TRIALS="$2"; shift 2 ;;
    --coherence-values) COHERENCE_VALUES="$2"; shift 2 ;;
    --drift-gain) DRIFT_GAIN="$2"; shift 2 ;;
    --noise-scale) NOISE_SCALE="$2"; shift 2 ;;
    --dt-ddm) DT_DDM="$2"; shift 2 ;;
    --dt-model) DT_MODEL="$2"; shift 2 ;;
    --t-start) T_START="$2"; shift 2 ;;
    --dur) DUR="$2"; shift 2 ;;
    --x0) X0="$2"; shift 2 ;;
    --boundary) BOUNDARY="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --chunk-ms) CHUNK_MS="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --python-bin) CANN_DDM_PYTHON="$2"; shift 2 ;;
    --save-traj) SAVE_TRAJ="1"; shift ;;
    *) usage ;;
  esac
done

export REPO_ROOT
export CANN_DDM_PYTHON

prepare_cmd=(
  "$CANN_DDM_PYTHON" "$REPO_ROOT/scripts/run_psychometric_workflow.py" prepare "$RUN_NAME"
  --num-trials "$NUM_TRIALS"
  --num-batches "$NUM_BATCHES"
  "--coherence-values=$COHERENCE_VALUES"
  --drift-gain "$DRIFT_GAIN"
  --noise-scale "$NOISE_SCALE"
  --dt-ddm "$DT_DDM"
  --dt-model "$DT_MODEL"
  --t-start "$T_START"
  --dur "$DUR"
  --x0 "$X0"
  --boundary "$BOUNDARY"
  --seed "$SEED"
  --chunk-ms "$CHUNK_MS"
  --output-root "$OUTPUT_ROOT"
  --python-bin "$CANN_DDM_PYTHON"
)
if [[ -n "$CIRCUIT_BATCH_TRIALS" ]]; then
  prepare_cmd+=(--circuit-batch-trials "$CIRCUIT_BATCH_TRIALS")
fi
if [[ "$SAVE_TRAJ" == "1" ]]; then
  prepare_cmd+=(--save-traj)
fi
"${prepare_cmd[@]}"

if [[ "$OUTPUT_ROOT" = /* ]]; then
  RUN_BASE="$OUTPUT_ROOT"
else
  RUN_BASE="$REPO_ROOT/$OUTPUT_ROOT"
fi
RUN_ROOT="$RUN_BASE/$RUN_NAME"
CIRCUIT_ROOT="$RUN_ROOT/circuit"
MANIFEST_PATH="$CIRCUIT_ROOT/submission_manifest.tsv"
METADATA_PATH="$RUN_ROOT/finalizer_metadata.env"
WORKER_JOB_SCRIPT="$RUN_ROOT/run_psychometric_worker_job.sh"
FINALIZER_JOB_SCRIPT="$RUN_ROOT/run_psychometric_finalizer_job.sh"
LOGS_ROOT="$CIRCUIT_ROOT/logs"
FINALIZER_LOGS_ROOT="$RUN_ROOT/finalizer_logs"
FINALIZER_NAME="psweep_finalize_${RUN_NAME}"

mapfile -t manifest_rows < <(tail -n +2 "$MANIFEST_PATH")
if [[ "${#manifest_rows[@]}" -eq 0 ]]; then
  echo "No worker rows found in manifest: $MANIFEST_PATH" >&2
  exit 1
fi

worker_job_ids=()
for row in "${manifest_rows[@]}"; do
  IFS=$'\t' read -r coherence slug batch_index batch_num_trials seed job_name _job_id output_dir <<< "$row"
  cmd=(
    qsub -terse
    -P "$PROJECT"
    -cwd
    -V
    -j y
    -o "$LOGS_ROOT"
    -N "$job_name"
    -l "h_rt=$WALLTIME"
    "$WORKER_JOB_SCRIPT"
    --coherence "$coherence"
    --num-trials "$batch_num_trials"
    --output-dir "$output_dir"
    --model circuit
    --drift-gain "$DRIFT_GAIN"
    --noise-scale "$NOISE_SCALE"
    --dt-ddm "$DT_DDM"
    --dt-model "$DT_MODEL"
    --t-start "$T_START"
    --dur "$DUR"
    --x0 "$X0"
    --boundary "$BOUNDARY"
    --seed "$seed"
    --chunk-ms "$CHUNK_MS"
  )
  if [[ "$SAVE_TRAJ" == "1" ]]; then
    cmd+=(--save-traj)
  fi
  echo "Submitting coherence=$coherence batch=$batch_index job_name=$job_name"
  job_id="$("${cmd[@]}")"
  if [[ -z "$job_id" ]]; then
    echo "Failed to capture worker job id for $job_name" >&2
    exit 1
  fi
  worker_job_ids+=("$job_id")
done

hold_jid="$(IFS=,; echo "${worker_job_ids[*]}")"
finalizer_job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$FINALIZER_LOGS_ROOT" \
    -N "$FINALIZER_NAME" \
    -hold_jid "$hold_jid" \
    -l "h_rt=$WALLTIME" \
    "$FINALIZER_JOB_SCRIPT" \
    --run-root "$RUN_ROOT" \
    --ddm-root "$RUN_ROOT/ddm" \
    --circuit-root "$CIRCUIT_ROOT"
)"
if [[ -z "$finalizer_job_id" ]]; then
  echo "Failed to capture finalizer job id" >&2
  exit 1
fi

"$CANN_DDM_PYTHON" "$REPO_ROOT/scripts/run_psychometric_workflow.py" record-submissions \
  --manifest-path "$MANIFEST_PATH" \
  --metadata-path "$METADATA_PATH" \
  --worker-job-ids "$hold_jid" \
  --finalizer-job-id "$finalizer_job_id"

echo "Run root: $RUN_ROOT"
echo "Worker job ids: $hold_jid"
echo "Finalizer job id: $finalizer_job_id"
echo "Circuit worker shard outputs:"
echo "  \$RUN_ROOT/circuit/jobs/coh_*/batch_*/conditions/*.npz"
echo "  \$RUN_ROOT/circuit/jobs/coh_*/batch_*/summary.csv"
echo "  \$RUN_ROOT/circuit/jobs/coh_*/batch_*/config.json"
echo "Merged circuit sweep outputs after the finalizer merge step:"
echo "  $CIRCUIT_ROOT/dataset.npz"
echo "  $CIRCUIT_ROOT/summary.csv"
echo "  $CIRCUIT_ROOT/config.json"
echo "Final combined outputs after successful merge + combine:"
echo "  $RUN_ROOT/dataset.npz"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/config.json"
