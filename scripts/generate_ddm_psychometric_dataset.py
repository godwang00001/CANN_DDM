#!/usr/bin/env python3
"""Generate one sweep-level DDM psychometric dataset with notebook-matching defaults."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.accumulator_simulation import (
    AccumulatorSimulationSweep,
    save_simulation_sweep_npz,
    simulate_ddm_trials,
)


DEFAULT_COHERENCE_VALUES = "-1.0,-0.5,-0.25,-0.125,0.0,0.125,0.25,0.5,1.0"


def parse_coherence_values(raw: str) -> np.ndarray:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("coherence-values must contain at least one numeric value")
    return np.asarray(values, dtype=float)


def binomial_half_width(p_right: float, num_hit: int) -> float:
    if int(num_hit) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_hit))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="ddm_psychometric_n200")
    parser.add_argument("--coherence-values", default=DEFAULT_COHERENCE_VALUES)
    parser.add_argument("--drift-gain", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=0.3)
    parser.add_argument("--dt-ddm", type=float, default=5.0)
    parser.add_argument("--dt-model", type=float, default=1.0)
    parser.add_argument("--t-start", type=int, default=10)
    parser.add_argument("--dur", type=int, default=2000)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output-root", type=Path, default=Path("results/psychometric"))
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--save-traj", action="store_true")
    return parser


def summarize_condition(*, coherence: float, drift_gain: float, num_trials: int, seed: int, dataset_name: str, result) -> dict[str, object]:
    hit_mask = np.asarray(result.hit_boundary, dtype=bool)
    num_hit = int(np.sum(hit_mask))
    p_right = float(np.mean(result.choice[hit_mask] == 1)) if num_hit > 0 else float("nan")
    hit_fraction = float(np.mean(result.hit_boundary))
    mean_rt_ms = float(np.nanmean(result.rt_ms)) if np.any(result.hit_boundary) else float("nan")
    return {
        "model": "ddm",
        "coherence": float(coherence),
        "drift_rate": float(drift_gain) * float(coherence),
        "p_right": p_right,
        "num_hit": int(num_hit),
        "miss_fraction": 1.0 - hit_fraction,
        "hit_fraction": hit_fraction,
        "mean_rt_ms": mean_rt_ms,
        "ci_half_width": binomial_half_width(p_right, int(num_hit)),
        "num_trials": int(num_trials),
        "seed": int(seed),
        "result_file": str(dataset_name),
    }


def main() -> int:
    args = make_parser().parse_args()
    coherence_values = parse_coherence_values(args.coherence_values)
    run_root = Path(args.output_root) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(coherence_values))

    summary_rows: list[dict[str, object]] = []
    condition_results = []

    for coherence, child in zip(coherence_values, child_sequences):
        condition_seed = int(child.generate_state(1)[0])
        result = simulate_ddm_trials(
            drift_rate=float(args.drift_gain) * float(coherence),
            noise_scale=float(args.noise_scale),
            dt_DDM=float(args.dt_ddm),
            dur=int(args.dur),
            t_start=int(args.t_start),
            x0=float(args.x0),
            boundary=float(args.boundary),
            num_trials=int(args.num_trials),
            seed=condition_seed,
            return_traj=bool(args.save_traj),
        )
        condition_results.append(result)
        summary_rows.append(
            summarize_condition(
                coherence=float(coherence),
                drift_gain=float(args.drift_gain),
                num_trials=int(args.num_trials),
                seed=condition_seed,
                dataset_name=dataset_name,
                result=result,
            )
        )

    time_ms = np.asarray(condition_results[0].time_ms, dtype=float)
    choice = np.stack([np.asarray(result.choice) for result in condition_results], axis=0)
    hit_boundary = np.stack([np.asarray(result.hit_boundary) for result in condition_results], axis=0)
    rt_ms = np.stack([np.asarray(result.rt_ms, dtype=float) for result in condition_results], axis=0)
    final_x = np.stack([np.asarray(result.final_x, dtype=float) for result in condition_results], axis=0)
    x_traj = (
        np.stack([np.asarray(result.x_traj) for result in condition_results], axis=0)
        if bool(args.save_traj)
        else None
    )

    sweep = AccumulatorSimulationSweep(
        coherence_values=np.asarray(coherence_values, dtype=float),
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        time_ms=time_ms,
        x_traj=x_traj,
        metadata={
            "model_type": "ddm",
            "num_conditions": int(len(coherence_values)),
            "num_trials": int(args.num_trials),
            "dataset_file": dataset_name,
            "summary_file": summary_name,
            "config_file": config_name,
            "condition_seeds": [int(row["seed"]) for row in summary_rows],
        },
    )
    save_simulation_sweep_npz(run_root / dataset_name, sweep)

    summary_path = run_root / summary_name
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "model": "ddm",
        "coherence_values": coherence_values.tolist(),
        "drift_gain": float(args.drift_gain),
        "noise_scale": float(args.noise_scale),
        "dt_ddm": float(args.dt_ddm),
        "dt_model": float(args.dt_model),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "save_traj": bool(args.save_traj),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_conditions": int(len(coherence_values)),
    }
    (run_root / config_name).write_text(json.dumps(config, indent=2))

    print("DDM psychometric dataset")
    print(f"run_root: {run_root}")
    print(f"dataset: {run_root / dataset_name}")
    print(f"summary: {summary_path}")
    print(f"num_conditions: {len(coherence_values)}")
    print(f"num_trials: {int(args.num_trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
