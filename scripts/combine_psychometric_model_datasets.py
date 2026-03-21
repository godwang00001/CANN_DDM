#!/usr/bin/env python3
"""Combine sweep-level DDM and circuit datasets into one shared output bundle."""

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
    load_simulation_sweep_npz,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddm-root", type=Path, required=True)
    parser.add_argument("--circuit-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--delete-intermediates", action="store_true")
    return parser


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_bundle(root: Path) -> tuple[dict, list[dict[str, str]], object]:
    config = json.loads((root / "config.json").read_text())
    summary_rows = read_summary(root / "summary.csv")
    sweep = load_simulation_sweep_npz(root / "dataset.npz")
    return config, summary_rows, sweep


def main() -> int:
    args = make_parser().parse_args()
    ddm_root = args.ddm_root.resolve()
    circuit_root = args.circuit_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ddm_config, ddm_summary_rows, ddm_sweep = _load_bundle(ddm_root)
    circuit_config, circuit_summary_rows, circuit_sweep = _load_bundle(circuit_root)

    if not np.array_equal(ddm_sweep.coherence_values, circuit_sweep.coherence_values):
        raise ValueError("DDM and circuit coherence grids do not match")
    if ddm_sweep.choice.shape[1] != circuit_sweep.choice.shape[1]:
        raise ValueError("DDM and circuit num_trials do not match")

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)

    model_names = np.asarray(["ddm", "circuit"])
    choice = np.stack([np.asarray(ddm_sweep.choice), np.asarray(circuit_sweep.choice)], axis=0)
    hit_boundary = np.stack([np.asarray(ddm_sweep.hit_boundary), np.asarray(circuit_sweep.hit_boundary)], axis=0)
    rt_ms = np.stack([np.asarray(ddm_sweep.rt_ms, dtype=float), np.asarray(circuit_sweep.rt_ms, dtype=float)], axis=0)
    final_x = np.stack([np.asarray(ddm_sweep.final_x, dtype=float), np.asarray(circuit_sweep.final_x, dtype=float)], axis=0)
    time_ms = np.stack([np.asarray(ddm_sweep.time_ms, dtype=float), np.asarray(circuit_sweep.time_ms, dtype=float)], axis=0)

    payload = {
        "model_names": model_names,
        "coherence_values": np.asarray(ddm_sweep.coherence_values, dtype=float),
        "choice": choice,
        "hit_boundary": hit_boundary,
        "rt_ms": rt_ms,
        "final_x": final_x,
        "time_ms": time_ms,
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "dataset_file": dataset_name,
                    "summary_file": summary_name,
                    "config_file": config_name,
                    "num_models": 2,
                    "num_conditions": int(choice.shape[1]),
                    "num_trials": int(choice.shape[2]),
                    "ddm_config": ddm_config,
                    "circuit_config": circuit_config,
                    "ddm_metadata": ddm_sweep.metadata,
                    "circuit_metadata": circuit_sweep.metadata,
                }
            )
        ),
    }
    if ddm_sweep.x_traj is not None and circuit_sweep.x_traj is not None:
        payload["x_traj"] = np.stack([np.asarray(ddm_sweep.x_traj), np.asarray(circuit_sweep.x_traj)], axis=0)
    np.savez_compressed(output_root / dataset_name, **payload)

    summary_rows = []
    for rows in (ddm_summary_rows, circuit_summary_rows):
        for row in rows:
            row = dict(row)
            row["result_file"] = dataset_name
            summary_rows.append(row)
    summary_rows.sort(key=lambda row: (row["model"], float(row["coherence"])))
    with (output_root / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    output_config = {
        "models": ["ddm", "circuit"],
        "coherence_values": np.asarray(ddm_sweep.coherence_values, dtype=float).tolist(),
        "drift_gain": float(ddm_config["drift_gain"]),
        "noise_scale": float(ddm_config["noise_scale"]),
        "dt_ddm": float(ddm_config["dt_ddm"]),
        "dt_model": float(circuit_config["dt_model"]),
        "t_start": int(ddm_config["t_start"]),
        "dur": int(ddm_config["dur"]),
        "x0": float(ddm_config["x0"]),
        "boundary": float(ddm_config["boundary"]),
        "num_trials": int(ddm_config["num_trials"]),
        "seed": int(ddm_config["seed"]),
        "save_traj": bool(ddm_config.get("save_traj", False) and circuit_config.get("save_traj", False)),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_models": 2,
        "num_conditions": int(choice.shape[1]),
        "ddm_root": str(ddm_root),
        "circuit_root": str(circuit_root),
    }
    (output_root / config_name).write_text(json.dumps(output_config, indent=2))

    if args.delete_intermediates:
        if ddm_root.exists():
            shutil.rmtree(ddm_root)
        if circuit_root.exists():
            shutil.rmtree(circuit_root)

    print(f"Combined DDM and circuit datasets into {output_root / dataset_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
