#!/usr/bin/env python3
"""Simulate fixed-condition pulse DDM trials and plot p(right) against net click count (#R - #L)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.accumulator_simulation import PulseDDMConfig, simulate_ddm_trials


LAMBDA_RIGHT_HIGH = 0.03
LAMBDA_LEFT_LOW = 0.01
NOISE_SCALE_DEFAULT = 0.3
DELTA_CLICK_X_DEFAULT = 0.05
LAMBDA_COMMIT_DEFAULT = 0.0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials-per-condition", type=int, default=500)
    parser.add_argument("--lambda-right", type=float, default=LAMBDA_RIGHT_HIGH)
    parser.add_argument("--lambda-left", type=float, default=LAMBDA_LEFT_LOW)
    parser.add_argument("--dt-ddm", type=float, default=10.0)
    parser.add_argument("--t-start", type=int, default=10)
    parser.add_argument("--dur", type=int, default=2000)
    parser.add_argument("--max-time", type=int)
    parser.add_argument("--x0", type=float, default=0.5)
    parser.add_argument("--boundary", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=NOISE_SCALE_DEFAULT)
    parser.add_argument("--lambda-commit", type=float, default=LAMBDA_COMMIT_DEFAULT)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--decision-paradigm", choices=("free_response", "interrogation"), default="free_response")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pddm_net_click_psychometric"))
    parser.add_argument("--figure-name", default="psychometric_by_net_click.png")
    parser.add_argument("--summary-name", default="psychometric_by_net_click.csv")
    parser.add_argument("--config-name", default="config.json")
    return parser


def fixed_conditions(*, lambda_right: float, lambda_left: float) -> list[dict[str, float | str]]:
    return [
        {
            "condition": "right_biased",
            "lambda_click_L": float(lambda_left),
            "lambda_click_R": float(lambda_right),
        },
        {
            "condition": "left_biased",
            "lambda_click_L": float(lambda_right),
            "lambda_click_R": float(lambda_left),
        },
    ]


def binomial_half_width(p_right: float, num_trials: int) -> float:
    if int(num_trials) <= 0:
        return float("nan")
    return 1.96 * np.sqrt(float(p_right) * (1.0 - float(p_right)) / float(num_trials))


def simulate_condition(
    *,
    condition: dict[str, float | str],
    num_trials: int,
    dt_ddm: float,
    t_start: int,
    dur: int,
    max_time: int | None,
    x0: float,
    boundary: float,
    noise_scale: float,
    lambda_commit: float,
    decision_paradigm: str,
    seed: int,
):
    return simulate_ddm_trials(
        decision_mode="discrete",
        config=PulseDDMConfig(
            lambda_click_L=float(condition["lambda_click_L"]),
            lambda_click_R=float(condition["lambda_click_R"]),
            delta_click_x=float(DELTA_CLICK_X_DEFAULT),
            noise_scale=float(noise_scale),
            lambda_commit=float(lambda_commit),
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
        return_traj=False,
    )


def aggregate_by_net_click(results: list[tuple[str, object]]) -> list[dict[str, float | int]]:
    pooled_net_clicks: list[np.ndarray] = []
    pooled_choices: list[np.ndarray] = []
    pooled_hit_boundary: list[np.ndarray] = []

    for _, result in results:
        net_click_count = np.asarray(result.metadata["net_click_count"], dtype=int)
        hit_boundary = np.asarray(result.hit_boundary, dtype=bool)
        choice_right = (np.asarray(result.choice) == 1).astype(float)
        pooled_net_clicks.append(net_click_count)
        pooled_choices.append(choice_right)
        pooled_hit_boundary.append(hit_boundary)

    net_clicks = np.concatenate(pooled_net_clicks, axis=0)
    choices = np.concatenate(pooled_choices, axis=0)
    hit_boundary = np.concatenate(pooled_hit_boundary, axis=0)

    rows: list[dict[str, float | int]] = []
    for value in np.unique(net_clicks):
        mask = net_clicks == int(value)
        hit_mask = mask & hit_boundary
        num_trials = int(np.sum(mask))
        num_hit = int(np.sum(hit_mask))
        p_right = float(np.mean(choices[hit_mask])) if num_hit > 0 else float("nan")
        rows.append(
            {
                "net_click_count": int(value),
                "num_trials": num_trials,
                "num_hit": num_hit,
                "miss_fraction": 1.0 - (float(num_hit) / float(num_trials)),
                "p_right": p_right,
                "ci_half_width": binomial_half_width(p_right, num_hit),
            }
        )
    return rows


def main() -> int:
    args = make_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = fixed_conditions(
        lambda_right=float(args.lambda_right),
        lambda_left=float(args.lambda_left),
    )
    seed_sequence = np.random.SeedSequence(int(args.seed))
    child_sequences = seed_sequence.spawn(len(conditions))

    results: list[tuple[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    for condition, child in zip(conditions, child_sequences):
        condition_seed = int(child.generate_state(1)[0])
        result = simulate_condition(
            condition=condition,
            num_trials=int(args.num_trials_per_condition),
            dt_ddm=float(args.dt_ddm),
            t_start=int(args.t_start),
            dur=int(args.dur),
            max_time=None if args.max_time is None else int(args.max_time),
            x0=float(args.x0),
            boundary=float(args.boundary),
            noise_scale=float(args.noise_scale),
            lambda_commit=float(args.lambda_commit),
            decision_paradigm=str(args.decision_paradigm),
            seed=condition_seed,
        )
        results.append((str(condition["condition"]), result))
        condition_rows.append(
            {
                "condition": str(condition["condition"]),
                "seed": condition_seed,
                "lambda_click_L": float(condition["lambda_click_L"]),
                "lambda_click_R": float(condition["lambda_click_R"]),
                "num_trials": int(args.num_trials_per_condition),
                "hit_fraction": float(np.mean(result.hit_boundary)),
                "p_right": (
                    float(np.mean(np.asarray(result.choice)[np.asarray(result.hit_boundary, dtype=bool)] == 1))
                    if np.any(result.hit_boundary)
                    else float("nan")
                ),
            }
        )

    summary_rows = aggregate_by_net_click(results)
    summary_path = output_dir / args.summary_name
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    x = np.asarray([row["net_click_count"] for row in summary_rows], dtype=float)
    y = np.asarray([row["p_right"] for row in summary_rows], dtype=float)
    err = np.asarray([row["ci_half_width"] for row in summary_rows], dtype=float)
    n = np.asarray([row["num_trials"] for row in summary_rows], dtype=float)
    n_hit = np.asarray([row["num_hit"] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(x, y, yerr=err, fmt="o-", color="#1f4e79", lw=1.5, ms=4, capsize=3)
    ax.axhline(0.5, color="0.7", lw=1.0, ls="--")
    ax.axvline(0.0, color="0.8", lw=1.0, ls=":")
    ax.set_xlabel("#R - #L")
    ax.set_ylabel("P(choose right)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Pulse DDM Psychometric Curve by Full-Window Net Click Count")

    ax2 = ax.twinx()
    ax2.bar(x, n, width=0.8, color="#9ecae1", alpha=0.25)
    ax2.plot(x, n_hit, color="#4d4d4d", lw=1.2, alpha=0.85)
    ax2.set_ylabel("Trials per net-click bin")
    ax2.set_ylim(0, max(1.0, 1.15 * float(np.max(n))))

    fig.tight_layout()
    figure_path = output_dir / args.figure_name
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    config_path = output_dir / args.config_name
    config = {
        "decision_mode": "discrete",
        "decision_paradigm": str(args.decision_paradigm),
        "num_trials_per_condition": int(args.num_trials_per_condition),
        "num_conditions": int(len(conditions)),
        "total_trials": int(args.num_trials_per_condition) * int(len(conditions)),
        "conditions": conditions,
        "lambda_right": float(args.lambda_right),
        "lambda_left": float(args.lambda_left),
        "noise_scale": float(args.noise_scale),
        "lambda_commit": float(args.lambda_commit),
        "delta_click_x": float(DELTA_CLICK_X_DEFAULT),
        "dt_ddm": float(args.dt_ddm),
        "t_start": int(args.t_start),
        "dur": int(args.dur),
        "max_time": int(args.max_time) if args.max_time is not None else int(args.dur),
        "x0": float(args.x0),
        "boundary": float(args.boundary),
        "seed": int(args.seed),
        "click_count_window": "full_stimulus",
        "figure_file": str(figure_path.name),
        "summary_file": str(summary_path.name),
        "condition_rows": condition_rows,
    }
    config_path.write_text(json.dumps(config, indent=2))

    print("Pulse DDM psychometric by net click count")
    print(f"output_dir: {output_dir}")
    print(f"figure: {figure_path}")
    print(f"summary: {summary_path}")
    print(f"total_trials: {config['total_trials']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
