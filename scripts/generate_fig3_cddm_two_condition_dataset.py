#!/usr/bin/env python3
"""Generate the Figure 3 two-condition continuous-DDM dataset."""

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
    ContinuousDDMConfig,
    save_simulation_sweep_npz,
    simulate_ddm_trials,
)


FIG3_CONDITIONS = (
    {"label": "slow", "drift_rate": 0.3, "noise_scale": .5, "num_trials": 3000},
    {"label": "fast", "drift_rate": 0.9, "noise_scale": .5, "num_trials": 3000},
)


def binomial_half_width(p_right: float, num_hit: int) -> float:
    if int(num_hit) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_hit))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="fig3_cddm_two_condition_n1000")
    parser.add_argument("--dt-ddm", type=float, default=5.0)
    parser.add_argument("--dt-model", type=float, default=1.0)
    parser.add_argument("--t-start", type=int, default=10)
    parser.add_argument("--dur", type=int, default=1000)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--decision-paradigm", choices=("free_response", "interrogation"), default="free_response")
    parser.add_argument("--output-root", type=Path, default=Path("results/figure3"))
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--save-traj", dest="save_traj", action="store_true")
    parser.add_argument("--no-save-traj", dest="save_traj", action="store_false")
    parser.set_defaults(save_traj=True)
    return parser


def summarize_condition(
    *,
    condition_label: str,
    drift_rate: float,
    noise_scale: float,
    num_trials: int,
    seed: int,
    dataset_name: str,
    result,
) -> dict[str, object]:
    hit_mask = np.asarray(result.hit_boundary, dtype=bool)
    num_hit = int(np.sum(hit_mask))
    p_right = float(np.mean(result.choice[hit_mask] == 1)) if num_hit > 0 else float("nan")
    hit_fraction = float(np.mean(result.hit_boundary))
    mean_rt_ms = float(np.nanmean(result.rt_ms)) if np.any(result.hit_boundary) else float("nan")
    return {
        "model": "ddm",
        "decision_mode": "continuous",
        "condition_label": str(condition_label),
        "drift_rate": float(drift_rate),
        "noise_scale": float(noise_scale),
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
    conditions = list(FIG3_CONDITIONS)
    run_root = Path(args.output_root) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(conditions))

    summary_rows: list[dict[str, object]] = []
    condition_results = []
    resolved_conditions = []

    for condition, child in zip(conditions, child_sequences):
        condition_seed = int(child.generate_state(1)[0])
        result = simulate_ddm_trials(
            decision_mode="continuous",
            config=ContinuousDDMConfig(
                drift_rate=float(condition["drift_rate"]),
                noise_scale=float(condition["noise_scale"]),
            ),
            dt_DDM=float(args.dt_ddm),
            dur=int(args.dur),
            max_time=None if args.max_time is None else int(args.max_time),
            t_start=int(args.t_start),
            x0=float(args.x0),
            boundary=float(args.boundary),
            num_trials=int(condition["num_trials"]),
            decision_paradigm=str(args.decision_paradigm),
            seed=condition_seed,
            return_traj=bool(args.save_traj),
        )
        condition_results.append(result)
        resolved_conditions.append(
            {
                "label": str(condition["label"]),
                "drift_rate": float(condition["drift_rate"]),
                "noise_scale": float(condition["noise_scale"]),
                "num_trials": int(condition["num_trials"]),
                "seed": condition_seed,
            }
        )
        summary_rows.append(
            summarize_condition(
                condition_label=str(condition["label"]),
                drift_rate=float(condition["drift_rate"]),
                noise_scale=float(condition["noise_scale"]),
                num_trials=int(condition["num_trials"]),
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
        coherence_values=np.asarray([condition["drift_rate"] for condition in resolved_conditions], dtype=float),
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        time_ms=time_ms,
        x_traj=x_traj,
        metadata={
            "model_type": "ddm",
            "decision_mode": "continuous",
            "decision_paradigm": str(args.decision_paradigm),
            "condition_axis": "drift_rate",
            "condition_labels": [condition["label"] for condition in resolved_conditions],
            "condition_drift_rates": [condition["drift_rate"] for condition in resolved_conditions],
            "condition_noise_scales": [condition["noise_scale"] for condition in resolved_conditions],
            "condition_num_trials": [condition["num_trials"] for condition in resolved_conditions],
            "num_conditions": int(len(resolved_conditions)),
            "num_trials_per_condition": [condition["num_trials"] for condition in resolved_conditions],
            "total_trials": int(sum(condition["num_trials"] for condition in resolved_conditions)),
            "dataset_file": dataset_name,
            "summary_file": summary_name,
            "config_file": config_name,
            "condition_seeds": [condition["seed"] for condition in resolved_conditions],
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
        "decision_mode": "continuous",
        "decision_paradigm": str(args.decision_paradigm),
        "conditions": resolved_conditions,
        "dt_ddm": float(args.dt_ddm),
        "dt_model": float(args.dt_model),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "max_time": int(args.max_time) if args.max_time is not None else int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "seed": int(args.seed),
        "save_traj": bool(args.save_traj),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_conditions": int(len(resolved_conditions)),
        "num_trials_per_condition": [condition["num_trials"] for condition in resolved_conditions],
        "total_trials": int(sum(condition["num_trials"] for condition in resolved_conditions)),
    }
    (run_root / config_name).write_text(json.dumps(config, indent=2))

    print("Figure 3 continuous DDM two-condition dataset")
    print(f"run_root: {run_root}")
    print(f"dataset: {run_root / dataset_name}")
    print(f"summary: {summary_path}")
    print(f"num_conditions: {len(resolved_conditions)}")
    print(f"total_trials: {int(sum(condition['num_trials'] for condition in resolved_conditions))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
