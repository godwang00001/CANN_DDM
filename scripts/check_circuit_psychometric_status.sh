#!/bin/bash -l
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 2
fi

RUN_DIR="$1"
MANIFEST_PATH="$RUN_DIR/submission_manifest.tsv"
FINALIZER_METADATA_PATH="$(dirname "$RUN_DIR")/finalizer_metadata.env"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Missing manifest: $MANIFEST_PATH" >&2
  exit 1
fi

QSTAT_OUTPUT="$(qstat -u "${USER:-cyw6}" || true)"

job_state() {
  local job_id="$1"
  local state
  state="$(awk -v job_id="$job_id" '$1 == job_id {print $5}' <<<"$QSTAT_OUTPUT")"
  if [[ -z "$state" ]]; then
    printf "not_seen"
  else
    printf "%s" "$state"
  fi
}

printf "worker_status\n"
printf "coherence\tjob_id\tstate\toutput_present\tsummary_present\n"
tail -n +2 "$MANIFEST_PATH" | while IFS=$'\t' read -r coherence slug batch_index batch_num_trials seed job_name job_id output_dir; do
  summary_present=false
  output_present=false
  [[ -f "$output_dir/summary.csv" ]] && summary_present=true
  compgen -G "$output_dir/conditions/*.npz" > /dev/null && output_present=true
  if [[ "$summary_present" == true ]]; then
    state="done"
  else
    state="$(job_state "$job_id")"
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" "$coherence" "$job_id" "$state" "$output_present" "$summary_present"
done

printf "\nfinalizer_status\n"
printf "item\tvalue\n"

root_dataset=false
root_summary=false
root_config=false
circuit_dataset=false
circuit_summary=false
circuit_config=false

[[ -f "$(dirname "$RUN_DIR")/dataset.npz" ]] && root_dataset=true
[[ -f "$(dirname "$RUN_DIR")/summary.csv" ]] && root_summary=true
[[ -f "$(dirname "$RUN_DIR")/config.json" ]] && root_config=true
[[ -f "$RUN_DIR/dataset.npz" ]] && circuit_dataset=true
[[ -f "$RUN_DIR/summary.csv" ]] && circuit_summary=true
[[ -f "$RUN_DIR/config.json" ]] && circuit_config=true

printf "circuit_dataset_present\t%s\n" "$circuit_dataset"
printf "circuit_summary_present\t%s\n" "$circuit_summary"
printf "circuit_config_present\t%s\n" "$circuit_config"
printf "root_dataset_present\t%s\n" "$root_dataset"
printf "root_summary_present\t%s\n" "$root_summary"
printf "root_config_present\t%s\n" "$root_config"

if [[ -f "$FINALIZER_METADATA_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$FINALIZER_METADATA_PATH"
  finalizer_state="$(job_state "${finalizer_job_id:-}")"
  finalizer_log_present=false
  if compgen -G "${finalizer_logs_root:-}/"* > /dev/null; then
    finalizer_log_present=true
  fi
  printf "finalizer_job_id\t%s\n" "${finalizer_job_id:-}"
  printf "finalizer_name\t%s\n" "${finalizer_name:-}"
  printf "finalizer_state\t%s\n" "$finalizer_state"
  printf "finalizer_logs_root\t%s\n" "${finalizer_logs_root:-}"
  printf "finalizer_log_present\t%s\n" "$finalizer_log_present"
else
  printf "finalizer_metadata_present\tfalse\n"
fi
