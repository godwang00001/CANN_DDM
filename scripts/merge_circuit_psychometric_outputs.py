#!/usr/bin/env python3
"""Merge per-coherence psychometric job outputs into one sweep-level dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections.abc import Sequence
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


def _summary_fieldnames(rows: list[dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def _group_job_dirs(
    run_dir: Path,
    manifest_rows: list[dict[str, str]],
    jobs_root: Path,
) -> list[tuple[float, list[Path], list[dict[str, str]]]]:
    if manifest_rows:
        grouped: dict[float, list[tuple[Path, dict[str, str]]]] = {}
        for row in manifest_rows:
            coherence = float(row["coherence"])
            grouped.setdefault(coherence, []).append((Path(row["output_dir"]), row))
        grouped_items: list[tuple[float, list[Path], list[dict[str, str]]]] = []
        for coherence in sorted(grouped):
            shard_entries = sorted(
                grouped[coherence],
                key=lambda item: int(item[1].get("batch_index", "0")),
            )
            grouped_items.append(
                (
                    coherence,
                    [entry[0] for entry in shard_entries],
                    [entry[1] for entry in shard_entries],
                )
            )
        return grouped_items

    job_dirs = sorted(path for path in jobs_root.glob("coh_*") if path.is_dir())
    grouped_items = []
    for job_dir in job_dirs:
        shard_dirs = sorted(
            path for path in job_dir.glob("batch_*") if path.is_dir()
        )
        if shard_dirs:
            summary_rows = [read_one_row_csv(path / "summary.csv") for path in shard_dirs]
            coherence = float(summary_rows[0]["coherence"])
            grouped_items.append((coherence, shard_dirs, []))
        else:
            row = read_one_row_csv(job_dir / "summary.csv")
            grouped_items.append((float(row["coherence"]), [job_dir], []))
    grouped_items.sort(key=lambda item: item[0])
    return grouped_items


def _concatenate_condition_results(results):
    first = results[0]
    time_ms = np.asarray(first.time_ms, dtype=float)
    have_traj = first.x_traj is not None
    for result in results[1:]:
        if not np.array_equal(np.asarray(result.time_ms, dtype=float), time_ms):
            raise ValueError("All shard results for a coherence must share the same time_ms grid")
        if (result.x_traj is not None) != have_traj:
            raise ValueError("All shard results for a coherence must either all include x_traj or all omit it")

    choice = np.concatenate([np.asarray(result.choice) for result in results], axis=0)
    hit_boundary = np.concatenate([np.asarray(result.hit_boundary, dtype=bool) for result in results], axis=0)
    rt_ms = np.concatenate([np.asarray(result.rt_ms, dtype=float) for result in results], axis=0)
    final_x = np.concatenate([np.asarray(result.final_x, dtype=float) for result in results], axis=0)
    x_traj = (
        np.concatenate([np.asarray(result.x_traj) for result in results], axis=0)
        if have_traj
        else None
    )
    return choice, hit_boundary, rt_ms, final_x, time_ms, x_traj


def _merged_summary_row(
    *,
    summary_rows: list[dict[str, str]],
    coherence: float,
    dataset_name: str,
    choice: np.ndarray,
    hit_boundary: np.ndarray,
    rt_ms: np.ndarray,
) -> dict[str, object]:
    first_row = dict(summary_rows[0])
    hit_mask = np.asarray(hit_boundary, dtype=bool)
    num_hit = int(np.sum(hit_mask))
    p_right = float(np.mean(np.asarray(choice)[hit_mask] == 1)) if num_hit > 0 else float("nan")
    hit_fraction = float(np.mean(hit_mask))
    mean_rt_ms = float(np.nanmean(np.asarray(rt_ms, dtype=float))) if num_hit > 0 else float("nan")
    merged_row: dict[str, object] = {
        "model": first_row["model"],
        "coherence": float(coherence),
        "drift_rate": float(first_row["drift_rate"]),
        "p_right": p_right,
        "num_hit": num_hit,
        "miss_fraction": 1.0 - hit_fraction,
        "hit_fraction": hit_fraction,
        "mean_rt_ms": mean_rt_ms,
        "ci_half_width": 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_hit)) if num_hit > 0 else float("nan"),
        "num_trials": int(np.asarray(choice).shape[0]),
        "seed": first_row["seed"],
        "result_file": dataset_name,
        "batch_count": int(len(summary_rows)),
        "batch_num_trials": ",".join(row["num_trials"] for row in summary_rows),
        "batch_seeds": ",".join(row["seed"] for row in summary_rows),
    }
    return merged_row


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(list(argv) if argv is not None else None)
    run_dir = args.run_dir.resolve()
    jobs_root = run_dir / "jobs"
    logs_root = run_dir / "logs"
    manifest_path = run_dir / "submission_manifest.tsv"

    manifest_rows = read_manifest(manifest_path) if manifest_path.exists() else []
    grouped_job_dirs = _group_job_dirs(run_dir, manifest_rows, jobs_root)

    if not grouped_job_dirs:
        raise ValueError(f"No job directories found under {jobs_root}")

    summary_rows: list[dict[str, str]] = []
    condition_results = []
    condition_metadata: list[dict] = []
    config_template: dict | None = None

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)

    coherence_values_list: list[float] = []
    choice_list = []
    hit_boundary_list = []
    rt_ms_list = []
    final_x_list = []
    x_traj_list = []
    have_traj: bool | None = None
    time_ms: np.ndarray | None = None

    for coherence, shard_dirs, coherence_manifest_rows in grouped_job_dirs:
        shard_summary_rows: list[dict[str, str]] = []
        shard_results = []
        shard_metadata = []

        for shard_dir in shard_dirs:
            summary_path = shard_dir / "summary.csv"
            config_path = shard_dir / "config.json"
            condition_paths = sorted((shard_dir / "conditions").glob("*.npz"))
            if not summary_path.exists():
                raise ValueError(f"Missing summary file: {summary_path}")
            if not config_path.exists():
                raise ValueError(f"Missing config file: {config_path}")
            if len(condition_paths) != 1:
                raise ValueError(
                    f"Expected exactly one condition archive in {shard_dir / 'conditions'}, found {len(condition_paths)}"
                )

            shard_summary_rows.append(read_one_row_csv(summary_path))
            result = load_simulation_result_npz(condition_paths[0])
            shard_results.append(result)
            shard_metadata.append(dict(result.metadata))

            if config_template is None:
                config_template = json.loads(config_path.read_text())

        merged_choice, merged_hit_boundary, merged_rt_ms, merged_final_x, merged_time_ms, merged_x_traj = _concatenate_condition_results(shard_results)
        if time_ms is None:
            time_ms = np.asarray(merged_time_ms, dtype=float)
            have_traj = merged_x_traj is not None
        else:
            if not np.array_equal(np.asarray(merged_time_ms, dtype=float), time_ms):
                raise ValueError("All merged coherence results must share the same time_ms grid")
            if (merged_x_traj is not None) != have_traj:
                raise ValueError("All merged coherence results must either all include x_traj or all omit it")

        summary_rows.append(
            _merged_summary_row(
                summary_rows=shard_summary_rows,
                coherence=coherence,
                dataset_name=dataset_name,
                choice=merged_choice,
                hit_boundary=merged_hit_boundary,
                rt_ms=merged_rt_ms,
            )
        )
        coherence_values_list.append(float(coherence))
        choice_list.append(merged_choice)
        hit_boundary_list.append(merged_hit_boundary)
        rt_ms_list.append(merged_rt_ms)
        final_x_list.append(merged_final_x)
        if merged_x_traj is not None:
            x_traj_list.append(merged_x_traj)
        condition_metadata.append(
            {
                "coherence": float(coherence),
                "num_batches": int(len(shard_results)),
                "num_trials": int(merged_choice.shape[0]),
                "batch_metadata": shard_metadata,
                "submission_rows": coherence_manifest_rows,
            }
        )

    coherence_values = np.asarray(coherence_values_list, dtype=float)
    choice = np.stack(choice_list, axis=0)
    hit_boundary = np.stack(hit_boundary_list, axis=0)
    rt_ms = np.stack(rt_ms_list, axis=0)
    final_x = np.stack(final_x_list, axis=0)
    x_traj = np.stack(x_traj_list, axis=0) if have_traj else None

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
        writer = csv.DictWriter(handle, fieldnames=_summary_fieldnames(summary_rows))
        writer.writeheader()
        writer.writerows(summary_rows)

    combined_config = dict(config_template or {})
    combined_config["coherence_values"] = coherence_values.tolist()
    combined_config["dataset_file"] = dataset_name
    combined_config["summary_file"] = summary_name
    combined_config["submission_manifest"] = manifest_rows
    combined_config["num_conditions"] = int(len(summary_rows))
    combined_config["num_trials"] = int(choice.shape[1])
    combined_config["num_batches"] = int(max(len(item[1]) for item in grouped_job_dirs))
    combined_config["circuit_batch_trials"] = int(choice.shape[1] // max(len(item[1]) for item in grouped_job_dirs))
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
