#!/bin/bash -l
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SGE_ROOT_DEFAULT="/usr/local/ogs-ge2011.11.p1/sge_root"
SGE_SETTINGS_DEFAULT="$SGE_ROOT_DEFAULT/default/common/settings.sh"
PYTHON_DEFAULT="/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python"

SGE_ROOT="${SGE_ROOT:-$SGE_ROOT_DEFAULT}"
SGE_SETTINGS="${SGE_SETTINGS:-$SGE_SETTINGS_DEFAULT}"
CANN_DDM_PYTHON="${CANN_DDM_PYTHON:-$PYTHON_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT:/projectnb/ecog-eeg/cyw6/CANN_DDM_rate_model/results/
  psychometric_test}"
WALLTIME="${WALLTIME:-00:30:00}"
RUN_NAME="${RUN_NAME:-test_psychometric_n1_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$SGE_SETTINGS" ]]; then
  echo "Missing SGE settings file: $SGE_SETTINGS" >&2
  exit 1
fi

export SGE_ROOT
export CANN_DDM_PYTHON
source "$SGE_SETTINGS"

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available after sourcing $SGE_SETTINGS" >&2
  exit 1
fi

echo "Submitting test psychometric run"
echo "run_name: $RUN_NAME"
echo "output_root: $OUTPUT_ROOT"
echo "python: $CANN_DDM_PYTHON"
echo "qsub: $(command -v qsub)"

cd "$REPO_ROOT"

bash ./scripts/submit_circuit_psychometric_scc.sh "$RUN_NAME" \
  --output-root "$OUTPUT_ROOT" \
  --num-trials 1 \
  --num-batches 1 \
  --circuit-batch-trials 1 \
  --walltime "$WALLTIME"

echo
echo "Run submitted or prepared under:"
echo "  $OUTPUT_ROOT/$RUN_NAME"
