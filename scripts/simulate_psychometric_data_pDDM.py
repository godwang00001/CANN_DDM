#!/usr/bin/env python3
"""Generate psychometric-condition simulation data for a fixed two-condition pulse-based DDM sweep."""

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
    PulseDDMConfig,
    load_simulation_result_npz,
    save_simulation_result_npz,
    simulate_ddm_trials,
)


LAMBDA_RIGHT_HIGH = 0.03
LAMBDA_LEFT_LOW = 0.01
NOISE_SCALE_DEFAULT = 0.2
DELTA_CLICK_X_DEFAULT = 0.05


def binomial_half_width(p_right: float, num_trials: int) -> float:
    if int(num_trials) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_trials))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-ddm", type=float, required=True)
    parser.add_argument("--t-start", type=int, required=True)
    parser.add_argument("--dur", type=int, required=True)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--num-trials", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--decision-paradigm", choices=("free_response", "interrogation"), default="free_response")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-traj", action="store_true")
    return parser


def fixed_conditions() -> list[dict[str, float | str]]:
    return [
        {
            "condition": "right_biased",
            "lambda_click_L": LAMBDA_LEFT_LOW,
            "lambda_click_R": LAMBDA_RIGHT_HIGH,
        },
        {
            "condition": "left_biased",
            "lambda_click_L": LAMBDA_RIGHT_HIGH,
            "lambda_click_R": LAMBDA_LEFT_LOW,
        },
    ]


def simulate_one_condition(
    *,
    lambda_click_L: float,
    lambda_click_R: float,
    dt_ddm: float,
    t_start: int,
    dur: int,
    max_time: int | None,
    x0: float,
    boundary: float,
    num_trials: int,
    seed: int,
    decision_paradigm: str,
    save_traj: bool,
):
    return simulate_ddm_trials(
        decision_mode="discrete",
        config=PulseDDMConfig(
            lambda_click_L=float(lambda_click_L),
            lambda_click_R=float(lambda_click_R),
            delta_click_x=float(DELTA_CLICK_X_DEFAULT),
            noise_scale=float(NOISE_SCALE_DEFAULT),
        ),
        dt_DDM=float(dt_ddm),
        dur=int(dur),
        max_time=None if max_time is None else int(max_time),
        t_start=int(t_start),
        x0=float(x0),
        boundary=float(boundary),
        num_trials=int(num_trials),
        decision_paradigm=str(decision_paradigm),
        seed=int(seed),
        return_traj=bool(save_traj),
    )


def summarize_result(
    *,
    condition: str,
    lambda_click_L: float,
    lambda_click_R: float,
    num_trials: int,
    seed: int,
    result_file: str,
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
        "result_file": str(result_file),
    }


def main() -> int:
    args = make_parser().parse_args()
    conditions = fixed_conditions()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "conditions"
    results_dir.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(conditions))

    summary_rows: list[dict[str, object]] = []

    for condition, child in zip(conditions, child_sequences):
        trial_seed = int(child.generate_state(1)[0])
        result_path = results_dir / f"ddm_{condition['condition']}.npz"

        if args.resume and result_path.exists():
            result = load_simulation_result_npz(result_path)
        else:
            result = simulate_one_condition(
                lambda_click_L=float(condition["lambda_click_L"]),
                lambda_click_R=float(condition["lambda_click_R"]),
                dt_ddm=float(args.dt_ddm),
                t_start=int(args.t_start),
                dur=int(args.dur),
                max_time=None if args.max_time is None else int(args.max_time),
                x0=float(args.x0),
                boundary=float(args.boundary),
                num_trials=int(args.num_trials),
                seed=trial_seed,
                decision_paradigm=str(args.decision_paradigm),
                save_traj=bool(args.save_traj),
            )
            save_simulation_result_npz(result_path, result)

        summary_rows.append(
            summarize_result(
                condition=str(condition["condition"]),
                lambda_click_L=float(condition["lambda_click_L"]),
                lambda_click_R=float(condition["lambda_click_R"]),
                num_trials=int(args.num_trials),
                seed=trial_seed,
                result_file=str(result_path.relative_to(output_dir)),
                result=result,
            )
        )

    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    config_path = output_dir / "config.json"
    config_payload = {
        "model": "ddm",
        "decision_mode": "discrete",
        "decision_paradigm": str(args.decision_paradigm),
        "conditions": conditions,
        "noise_scale": float(NOISE_SCALE_DEFAULT),
        "delta_click_x": float(DELTA_CLICK_X_DEFAULT),
        "dt_ddm": float(args.dt_ddm),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "max_time": int(args.max_time) if args.max_time is not None else int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "num_trials": int(args.num_trials),
        "seed": int(args.seed),
        "resume": bool(args.resume),
        "save_traj": bool(args.save_traj),
        "summary_file": str(summary_path.name),
        "result_dir": str(results_dir.name),
    }
    config_path.write_text(json.dumps(config_payload, indent=2))

    print("Psychometric data summary")
    print("model: ddm")
    print("decision_mode: discrete")
    print(f"output_dir: {output_dir}")
    print(f"conditions: {len(summary_rows)}")
    print(f"summary_file: {summary_path}")
    print(f"max_miss_fraction: {max(float(row['miss_fraction']) for row in summary_rows):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
