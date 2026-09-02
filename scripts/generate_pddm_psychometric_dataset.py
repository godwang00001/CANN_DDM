#!/usr/bin/env python3
"""Generate one sweep-level pulse-DDM psychometric dataset with fixed click-rate conditions."""

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
    PulseDDMConfig,
    save_simulation_sweep_npz,
    simulate_ddm_trials,
)


LAMBDA_RIGHT_HIGH = 0.03
LAMBDA_LEFT_LOW = 0.01
NOISE_SCALE_DEFAULT = 0.2
DELTA_CLICK_X_DEFAULT = 0.05


def fixed_conditions() -> list[dict[str, float | str]]:
    return [
        {
            "condition": "right_biased",
            "condition_value": 1.0,
            "lambda_click_L": LAMBDA_LEFT_LOW,
            "lambda_click_R": LAMBDA_RIGHT_HIGH,
        },
        {
            "condition": "left_biased",
            "condition_value": -1.0,
            "lambda_click_L": LAMBDA_RIGHT_HIGH,
            "lambda_click_R": LAMBDA_LEFT_LOW,
        },
    ]


def binomial_half_width(p_right: float, num_hit: int) -> float:
    if int(num_hit) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_hit))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="pddm_psychometric_n200")
    parser.add_argument("--dt-ddm", type=float, default=10.0)
    parser.add_argument("--dt-model", type=float, default=1.0)
    parser.add_argument("--t-start", type=int, default=10)
    parser.add_argument("--dur", type=int, default=2000)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--num-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--decision-paradigm", choices=("free_response", "interrogation"), default="free_response")
    parser.add_argument("--output-root", type=Path, default=Path("results/psychometric"))
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--save-traj", action="store_true")
    return parser


def summarize_condition(
    *,
    condition: str,
    lambda_click_L: float,
    lambda_click_R: float,
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
        "decision_mode": "discrete",
        "condition": str(condition),
        "lambda_click_L": float(lambda_click_L),
        "lambda_click_R": float(lambda_click_R),
        "delta_click_x": float(DELTA_CLICK_X_DEFAULT),
        "noise_scale": float(NOISE_SCALE_DEFAULT),
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
    conditions = fixed_conditions()
    run_root = Path(args.output_root) / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    dataset_name = str(args.dataset_name)
    summary_name = str(args.summary_name)
    config_name = str(args.config_name)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(conditions))

    summary_rows: list[dict[str, object]] = []
    condition_results = []

    for condition, child in zip(conditions, child_sequences):
        condition_seed = int(child.generate_state(1)[0])
        result = simulate_ddm_trials(
            decision_mode="discrete",
            config=PulseDDMConfig(
                lambda_click_L=float(condition["lambda_click_L"]),
                lambda_click_R=float(condition["lambda_click_R"]),
                delta_click_x=float(DELTA_CLICK_X_DEFAULT),
                noise_scale=float(NOISE_SCALE_DEFAULT),
            ),
            dt_DDM=float(args.dt_ddm),
            dur=int(args.dur),
            max_time=None if args.max_time is None else int(args.max_time),
            t_start=int(args.t_start),
            x0=float(args.x0),
            boundary=float(args.boundary),
            num_trials=int(args.num_trials),
            decision_paradigm=str(args.decision_paradigm),
            seed=condition_seed,
            return_traj=bool(args.save_traj),
        )
        condition_results.append(result)
        summary_rows.append(
            summarize_condition(
                condition=str(condition["condition"]),
                lambda_click_L=float(condition["lambda_click_L"]),
                lambda_click_R=float(condition["lambda_click_R"]),
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
        coherence_values=np.asarray([float(condition["condition_value"]) for condition in conditions], dtype=float),
        choice=choice,
        hit_boundary=hit_boundary,
        rt_ms=rt_ms,
        final_x=final_x,
        time_ms=time_ms,
        x_traj=x_traj,
        metadata={
            "model_type": "ddm",
            "decision_mode": "discrete",
            "decision_paradigm": str(args.decision_paradigm),
            "condition_labels": [str(condition["condition"]) for condition in conditions],
            "lambda_click_L_values": [float(condition["lambda_click_L"]) for condition in conditions],
            "lambda_click_R_values": [float(condition["lambda_click_R"]) for condition in conditions],
            "delta_click_x": float(DELTA_CLICK_X_DEFAULT),
            "noise_scale": float(NOISE_SCALE_DEFAULT),
            "num_conditions": int(len(conditions)),
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
        "decision_mode": "discrete",
        "decision_paradigm": str(args.decision_paradigm),
        "conditions": conditions,
        "noise_scale": float(NOISE_SCALE_DEFAULT),
        "delta_click_x": float(DELTA_CLICK_X_DEFAULT),
        "dt_ddm": float(args.dt_ddm),
        "dt_model": float(args.dt_model),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "max_time": int(args.max_time) if args.max_time is not None else int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "save_traj": bool(args.save_traj),
        "dataset_file": dataset_name,
        "summary_file": summary_name,
        "num_conditions": int(len(conditions)),
    }
    (run_root / config_name).write_text(json.dumps(config, indent=2))

    print("Pulse DDM psychometric dataset")
    print(f"run_root: {run_root}")
    print(f"dataset: {run_root / dataset_name}")
    print(f"summary: {summary_path}")
    print(f"num_conditions: {len(conditions)}")
    print(f"num_trials: {int(args.num_trials)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
