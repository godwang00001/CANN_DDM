#!/usr/bin/env python3
"""Merge batched Figure 4 single-trial dynamics outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASET_NAME = "single_condition_dynamics.npz"
SUMMARY_NAME = "trial_summary.csv"
CONFIG_NAME = "config.json"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _load_batch(batch_dir: Path) -> dict[str, Any]:
    dataset_path = batch_dir / DATASET_NAME
    config_path = batch_dir / CONFIG_NAME
    summary_path = batch_dir / SUMMARY_NAME
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Missing batch dataset: {dataset_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing batch config: {config_path}")
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing batch summary: {summary_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files if name != "metadata_json"}
        metadata = json.loads(str(data["metadata_json"].item()))
    with summary_path.open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    config = json.loads(config_path.read_text())
    return {
        "batch_dir": batch_dir,
        "arrays": arrays,
        "metadata": metadata,
        "config": config,
        "summary_rows": summary_rows,
    }


def _validate_compatible(batches: list[dict[str, Any]]) -> None:
    first = batches[0]
    first_arrays = first["arrays"]
    for batch in batches[1:]:
        arrays = batch["arrays"]
        if not np.array_equal(arrays["time_ms"], first_arrays["time_ms"]):
            raise ValueError(f"time_ms mismatch in {batch['batch_dir']}")
        meta = batch["metadata"]
        first_meta = first["metadata"]
        for key in (
            "drift_gain",
            "noise_scale",
            "dt_ddm",
            "dt_model",
            "t_start",
            "dur",
            "num_units",
            "num_selected_units",
            "selected_neuron_indices",
            "param_overrides",
            "kappa",
        ):
            if meta.get(key) != first_meta.get(key):
                raise ValueError(f"metadata key '{key}' mismatch in {batch['batch_dir']}")


def _merged_summary_rows(batches: list[dict[str, Any]]) -> list[dict[str, object]]:
    merged_rows: list[dict[str, object]] = []
    trial_offset = 0
    for batch_index, batch in enumerate(batches):
        rows = batch["summary_rows"]
        coherence = float(batch["metadata"]["coherence"])
        for batch_trial_id, row in enumerate(rows):
            merged = dict(row)
            merged["batch_index"] = int(batch_index)
            merged["batch_trial_id"] = int(row.get("trial_id", batch_trial_id))
            merged["trial_id"] = int(trial_offset + batch_trial_id)
            merged.setdefault("coherence", coherence)
            merged_rows.append(merged)
        trial_offset += len(rows)
    return merged_rows


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_batches(run_root: Path) -> None:
    batch_root = run_root / "batches"
    batch_dirs = sorted(path for path in batch_root.rglob("batch_*") if path.is_dir())
    if not batch_dirs:
        raise ValueError(f"No batch directories found under {batch_root}")
    batches = sorted(
        (_load_batch(batch_dir) for batch_dir in batch_dirs),
        key=lambda batch: int(batch["metadata"].get("batch_index", 0)),
    )
    _validate_compatible(batches)

    arrays = [batch["arrays"] for batch in batches]
    coherence_by_trial_parts = []
    coherence_values: list[float] = []
    coherence_counts: dict[str, int] = {}
    for batch, item in zip(batches, arrays, strict=True):
        coherence = float(batch["metadata"]["coherence"])
        if coherence not in coherence_values:
            coherence_values.append(coherence)
        if "coherence_by_trial" in item:
            batch_coherence_by_trial = np.asarray(item["coherence_by_trial"], dtype=np.float32)
        else:
            batch_coherence_by_trial = np.full(item["choice"].shape[0], coherence, dtype=np.float32)
        coherence_by_trial_parts.append(batch_coherence_by_trial)
        key = str(coherence)
        coherence_counts[key] = coherence_counts.get(key, 0) + int(item["choice"].shape[0])

    merged = {
        "coherence_values": np.asarray(coherence_values, dtype=float),
        "coherence_by_trial": np.concatenate(coherence_by_trial_parts, axis=0),
        "choice": np.concatenate([item["choice"] for item in arrays], axis=0),
        "hit_boundary": np.concatenate([item["hit_boundary"] for item in arrays], axis=0),
        "rt_ms": np.concatenate([item["rt_ms"] for item in arrays], axis=0),
        "final_x": np.concatenate([item["final_x"] for item in arrays], axis=0),
        "time_ms": arrays[0]["time_ms"],
        "x_E": np.concatenate([item["x_E"] for item in arrays], axis=0),
        "r_E": np.concatenate([item["r_E"] for item in arrays], axis=0),
        "r_B": np.concatenate([item["r_B"] for item in arrays], axis=0),
    }

    base_metadata = dict(batches[0]["metadata"])
    batch_metadata = []
    trial_seeds: list[int] = []
    for batch_index, batch in enumerate(batches):
        meta = batch["metadata"]
        batch_metadata.append(
            {
                "batch_index": int(meta.get("batch_index", batch_index)),
                "batch_num_trials": int(meta.get("num_trials", len(batch["summary_rows"]))),
                "seed": int(meta.get("seed")),
                "output_dir": str(batch["batch_dir"]),
                "dataset_file": str(batch["batch_dir"] / DATASET_NAME),
            }
        )
        trial_seeds.extend(int(seed) for seed in meta.get("trial_seeds", []))

    total_trials = int(merged["choice"].shape[0])
    base_metadata.update(
        {
            "dataset": "fig4_single_trial_dynamics",
            "mixed_coherence": bool(len(coherence_values) > 1),
            "coherence_values": coherence_values,
            "coherence_counts": coherence_counts,
            "num_trials": total_trials,
            "num_batches": int(len(batches)),
            "batch_metadata": batch_metadata,
            "trial_seeds": trial_seeds,
            "dataset_file": DATASET_NAME,
            "summary_file": SUMMARY_NAME,
            "config_file": CONFIG_NAME,
            "merged_from_batches": True,
        }
    )

    merged["metadata_json"] = np.asarray(json.dumps(_json_ready(base_metadata)))
    np.savez_compressed(run_root / DATASET_NAME, **merged)

    summary_rows = _merged_summary_rows(batches)
    _write_summary(run_root / SUMMARY_NAME, summary_rows)
    (run_root / CONFIG_NAME).write_text(json.dumps(_json_ready(base_metadata), indent=2))

    print(f"merged_batches {len(batches)}")
    print(f"num_trials {total_trials}")
    print(f"saved_dataset {run_root / DATASET_NAME}")
    print(f"saved_summary {run_root / SUMMARY_NAME}")
    print(f"saved_config {run_root / CONFIG_NAME}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    merge_batches(args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
