#!/bin/bash -l
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage: submit_circuit_psychometric_scc.sh <run_name> [options]

Options:
  --num-trials <int>              Default: 200
  --coherence-values <csv>        Default: -1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0
  --drift-gain <float>            Default: 1
  --noise-scale <float>           Default: 0.3
  --dt-ddm <float>                Default: 5.0
  --dt-model <float>              Default: 1.0
  --t-start <int>                 Default: 10
  --dur <int>                     Default: 2000
  --x0 <float>                    Default: 0.5
  --boundary <float>              Default: 1.0
  --seed <int>                    Default: 201
  --chunk-ms <int>                Default: 1000
  --save-traj                     Default: off
  --project <name>                Default: ecog-eeg
  --walltime <HH:MM:SS>           Default: 01:00:00
EOF
  exit 2
}

if [[ "$#" -lt 1 ]]; then
  usage
fi

RUN_NAME="$1"
shift

NUM_TRIALS="200"
COHERENCE_VALUES_RAW="-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"
DRIFT_GAIN="1"
NOISE_SCALE="0.3"
DT_DDM="5.0"
DT_MODEL="1.0"
T_START="10"
DUR="2000"
X0="0.5"
BOUNDARY="1.0"
SEED="201"
CHUNK_MS="1000"
SAVE_TRAJ="0"
PROJECT="ecog-eeg"
WALLTIME="01:00:00"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --num-trials) NUM_TRIALS="$2"; shift 2 ;;
    --coherence-values) COHERENCE_VALUES_RAW="$2"; shift 2 ;;
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
    --project) PROJECT="$2"; shift 2 ;;
    --walltime) WALLTIME="$2"; shift 2 ;;
    --save-traj) SAVE_TRAJ="1"; shift ;;
    *) usage ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="$REPO_ROOT/results/psychometric/$RUN_NAME"
DDM_ROOT="$RUN_ROOT/ddm"
CIRCUIT_ROOT="$RUN_ROOT/circuit"
JOBS_ROOT="$CIRCUIT_ROOT/jobs"
LOGS_ROOT="$CIRCUIT_ROOT/logs"
MANIFEST_PATH="$CIRCUIT_ROOT/submission_manifest.tsv"
FINALIZER_LOGS_ROOT="/tmp/cann_ddm_psychometric_finalizer_logs"

mkdir -p "$RUN_ROOT" "$JOBS_ROOT" "$LOGS_ROOT"
mkdir -p "$FINALIZER_LOGS_ROOT"
printf "coherence\tcoherence_slug\tjob_name\tjob_id\toutput_dir\n" > "$MANIFEST_PATH"

echo "Generating DDM dataset locally under $DDM_ROOT"
ddm_cmd=(
  python "$REPO_ROOT/scripts/generate_ddm_psychometric_dataset.py"
  --run-name "ddm"
  --output-root "$RUN_ROOT"
  "--coherence-values=$COHERENCE_VALUES_RAW"
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
)
if [[ "$SAVE_TRAJ" == "1" ]]; then
  ddm_cmd+=(--save-traj)
fi
"${ddm_cmd[@]}"

IFS=',' read -r -a coherences <<< "$COHERENCE_VALUES_RAW"

coherence_slug() {
  python - "$1" <<'PY'
import sys
x = float(sys.argv[1])
sign = "p" if x >= 0 else "m"
magnitude = f"{abs(x):.6f}".rstrip("0").rstrip(".") or "0"
print(f"{sign}{magnitude.replace('.', 'p')}")
PY
}

job_ids=()
for coherence in "${coherences[@]}"; do
  slug="$(coherence_slug "$coherence")"
  job_name="psweep_${slug}_n${NUM_TRIALS}"
  output_dir="$JOBS_ROOT/coh_${slug}"
  mkdir -p "$output_dir"
  echo "Submitting coherence=$coherence job_name=$job_name"
  job_id="$(
    qsub -terse \
      -P "$PROJECT" \
      -cwd \
      -V \
      -j y \
      -o "$LOGS_ROOT" \
      -N "$job_name" \
      -v "MODEL=circuit,COHERENCE=$coherence,NUM_TRIALS=$NUM_TRIALS,OUTPUT_DIR=$output_dir,DRIFT_GAIN=$DRIFT_GAIN,NOISE_SCALE=$NOISE_SCALE,DT_DDM=$DT_DDM,DT_MODEL=$DT_MODEL,T_START=$T_START,DUR=$DUR,X0=$X0,BOUNDARY=$BOUNDARY,SEED=$SEED,CHUNK_MS=$CHUNK_MS,SAVE_TRAJ=$SAVE_TRAJ" \
      -l "h_rt=$WALLTIME" \
      "$REPO_ROOT/scripts/run_circuit_psychometric_one_coherence.sh"
  )"
  job_ids+=("$job_id")
  printf "%s\t%s\t%s\t%s\t%s\n" "$coherence" "$slug" "$job_name" "$job_id" "$output_dir" >> "$MANIFEST_PATH"
done

hold_jid="$(IFS=,; echo "${job_ids[*]}")"
finalizer_name="psweep_finalize_${RUN_NAME}"
finalizer_job_id="$(
  qsub -terse \
    -P "$PROJECT" \
    -cwd \
    -V \
    -j y \
    -o "$FINALIZER_LOGS_ROOT" \
    -N "$finalizer_name" \
    -hold_jid "$hold_jid" \
    -v "RUN_ROOT=$RUN_ROOT,DDM_ROOT=$DDM_ROOT,CIRCUIT_ROOT=$CIRCUIT_ROOT" \
    -l "h_rt=$WALLTIME" \
    "$REPO_ROOT/scripts/finalize_psychometric_run.sh"
)"

echo "Run root: $RUN_ROOT"
echo "Worker job ids: $hold_jid"
echo "Finalizer job id: $finalizer_job_id"
echo "Final outputs after successful merge:"
echo "  $RUN_ROOT/dataset.npz"
echo "  $RUN_ROOT/summary.csv"
echo "  $RUN_ROOT/config.json"
