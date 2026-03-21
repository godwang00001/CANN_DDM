#!/usr/bin/env python3
"""Merge per-coherence psychometric job outputs into one sweep-level dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.accumulator_simulation import (
    AccumulatorSimulationSweep,
    load_simulation_result_npz,
    save_simulation_sweep_npz,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--delete-intermediates", action="store_true")
    return parser


def read_one_row_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _job_dirs_from_manifest(run_dir: Path, manifest_rows: list[dict[str, str]]) -> list[Path]:
    return [Path(row["output_dir"]) for row in manifest_rows]


def main() -> int:
    args = make_parser().parse_args()
    run_dir = args.run_dir.resolve()
    jobs_root = run_dir / "jobs"
    logs_root = run_dir / "logs"
    manifest_path = run_dir / "submission_manifest.tsv"

    manifest_rows = read_manifest(manifest_path) if manifest_path.exists() else []
    if manifest_rows:
        job_dirs = _job_dirs_from_manifest(run_dir, manifest_rows)
    else:
        job_dirs = sorted(path for path in jobs_root.glob("coh_*") if path.is_dir())

    if not job_dirs:
        raise ValueError(f"No job directories found under {jobs_root}")

    summary_rows: list[dict[str, str]] = []
    condition_results = []
    condition_metadata: list[dict] = []
    config_template: dict | None = None

    for job_dir in job_dirs:
        summary_path = job_dir / "summary.csv"
        config_path = job_dir / "config.json"
        condition_paths = sorted((job_dir / "conditions").glob("*.npz"))
        if not summary_path.exists():
            raise ValueError(f"Missing summary file: {summary_path}")
        if not config_path.exists():
            raise ValueError(f"Missing config file: {config_path}")
        if len(condition_paths) != 1:
            raise ValueError(
                f"Expected exactly one condition archive in {job_dir / 'conditions'}, found {len(condition_paths)}"
            )

        row = read_one_row_csv(summary_path)
        summary_rows.append(row)

        result = load_simulation_result_npz(condition_paths[0])
        condition_results.append(result)
        condition_metadata.append(dict(result.metadata))

        if config_template is None:
            config_template = json.loads(config_path.read_text())

    sort_order = np.argsort([float(row["coherence"]) for row in summary_rows])
    summary_rows = [summary_rows[idx] for idx in sort_order]
    condition_results = [condition_results[idx] for idx in sort_order]
    condition_metadata = [condition_metadata[idx] for idx in sort_order]

    coherence_values = np.asarray([float(row["coherence"]) for row in summary_rows], dtype=float)
    time_ms = np.asarray(condition_results[0].time_ms, dtype=float)
    have_traj = condition_results[0].x_traj is not None

    for result in condition_results[1:]:
        if not np.array_equal(np.asarray(result.time_ms, dtype=float), time_ms):
            raise ValueError("All condition results must share the same time_ms grid")
        if (result.x_traj is not None) != have_traj:
            raise ValueError("All condition results must either all include x_traj or all omit it")

    choice = np.stack([np.asarray(result.choice) for result in condition_results], axis=0)
    hit_boundary = np.stack([np.asarray(result.hit_boundary) for result in condition_results], axis=0)
    rt_ms = np.stack([np.asarray(result.rt_ms, dtype=float) for result in condition_results], axis=0)
    final_x = np.stack([np.asarray(result.final_x, dtype=float) for result in condition_results], axis=0)
    x_traj = (
        np.stack([np.asarray(result.x_traj) for result in condition_results], axis=0)
        if have_traj
        else None
    )

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)
    for row in summary_rows:
        row["result_file"] = dataset_name

    sweep = AccumulatorSimulationSweep(
        coherence_values=coherence_values,
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        time_ms=time_ms,
        x_traj=x_traj,
        metadata={
            "model_type": summary_rows[0]["model"],
            "num_conditions": int(len(summary_rows)),
            "num_trials": int(choice.shape[1]),
            "dataset_file": dataset_name,
            "summary_file": summary_name,
            "config_file": config_name,
            "condition_metadata": condition_metadata,
            "submission_manifest": manifest_rows,
        },
    )
    save_simulation_sweep_npz(run_dir / dataset_name, sweep)

    summary_path = run_dir / summary_name
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    combined_config = dict(config_template or {})
    combined_config["coherence_values"] = coherence_values.tolist()
    combined_config["dataset_file"] = dataset_name
    combined_config["summary_file"] = summary_name
    combined_config["submission_manifest"] = manifest_rows
    combined_config["num_conditions"] = int(len(summary_rows))
    combined_config["num_trials"] = int(choice.shape[1])
    combined_config.pop("result_dir", None)
    (run_dir / config_name).write_text(json.dumps(combined_config, indent=2))

    if args.delete_intermediates:
        if jobs_root.exists():
            shutil.rmtree(jobs_root)
        if logs_root.exists():
            shutil.rmtree(logs_root)
        if manifest_path.exists():
            manifest_path.unlink()

    print(f"Merged {len(summary_rows)} coherence jobs into {run_dir / dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
