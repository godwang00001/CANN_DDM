#!/usr/bin/env python3
"""Search for an edge-kernel offset that minimizes pre-onset edge drift.

This calibrates the `edge_pop.offset` parameter under the current supplemental
constant-drive baseline by minimizing

    max_t |theta_E(t) - theta_E(0)|

over a short no-cue, no-noise trial that is chosen to avoid boundary hits.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


def build_baseline_params(*, x0: float, offset: float, tau_e: float, search_duration: int) -> dict:
    geometry = {
        "num_units": 1024,
        "coding_limit": float(np.pi / 2),
        "coding_frac": 0.3,
        "clamp_frac": 0.1,
    }
    edge_pop = {
        "tau_E": float(tau_e),
        "c_EB": 5.0,
        "alpha_E": 1.0,
        "gamma_E": 20.0,
        "edge_type": "tanh",
        "offset": float(offset),
        "noise_scale_edge": 0.0,
    }
    bump_pop = {
        "tau_B": 0.3,
        "c_BE": 0.0,
        "beta_B": 6.0,
        "c_BE_params": {"mode": "const"},
        "sigma_B": 0.05,
        "noise_scale_bump": 0.0,
    }
    decision_space_params = {
        "t_start": int(search_duration),
        "boundary": 1.0,
        "drift_rate": 1.0,
        # Must stay positive for the current constructor, then cues are zeroed below.
        "noise_scale": 0.1,
        "dt_DDM": 1.0,
        "x0": float(x0),
        "dur": int(search_duration),
        "seed": 4,
    }
    return {
        "geometry": geometry,
        "edge_pop": edge_pop,
        "bump_pop": bump_pop,
        "decision_space_params": decision_space_params,
    }


def summarize_run(*, x0: float, offset: float, tau_e: float, search_duration: int) -> dict:
    model = CANN_DDM_model(
        CANN_params=build_baseline_params(
            x0=x0,
            offset=offset,
            tau_e=tau_e,
            search_duration=search_duration,
        )
    )
    model.cue_R_all[:] = 0.0
    model.cue_L_all[:] = 0.0
    runner = model.run_simulation(
        mon_vars=["theta_E", "x_E", "hit_boundary"],
        progress_bar=False,
        dt=1.0,
        get_RT=False,
    )

    theta_e = np.asarray(runner.mon.theta_E).reshape(-1)
    x_e = np.asarray(runner.mon.x_E).reshape(-1)
    hit = np.asarray(runner.mon.hit_boundary).reshape(-1)
    eval_theta = theta_e
    eval_x = x_e

    return {
        "offset": float(offset),
        "tau_E": float(tau_e),
        "x0": float(x0),
        "search_duration": int(search_duration),
        "theta_E_init": float(eval_theta[0]),
        "theta_E_final": float(eval_theta[-1]),
        "theta_E_max_shift": float(np.max(np.abs(eval_theta - eval_theta[0]))),
        "x_E_init": float(eval_x[0]),
        "x_E_final": float(eval_x[-1]),
        "x_E_max_shift": float(np.max(np.abs(eval_x - eval_x[0]))),
        "hit_boundary": bool(np.any(hit)),
        "all_finite": bool(np.isfinite(theta_e).all() and np.isfinite(x_e).all()),
    }


def objective(*, x0: float, offset: float, tau_e: float, search_duration: int) -> float:
    summary = summarize_run(x0=x0, offset=offset, tau_e=tau_e, search_duration=search_duration)
    penalty = 0.0
    if not summary["all_finite"]:
        penalty += 1e6
    if summary["hit_boundary"]:
        penalty += 1e3
    return summary["theta_E_max_shift"] + penalty


def golden_section_search(
    *,
    x0: float,
    tau_e: float,
    search_duration: int,
    left: float,
    right: float,
    tol: float,
    max_iter: int,
) -> tuple[float, float]:
    phi = (1 + math.sqrt(5)) / 2
    invphi = 1 / phi

    c = right - (right - left) * invphi
    d = left + (right - left) * invphi
    fc = objective(x0=x0, offset=c, tau_e=tau_e, search_duration=search_duration)
    fd = objective(x0=x0, offset=d, tau_e=tau_e, search_duration=search_duration)
    it = 0

    while (right - left) > tol and it < max_iter:
        if fc < fd:
            right = d
            d = c
            fd = fc
            c = right - (right - left) * invphi
            fc = objective(x0=x0, offset=c, tau_e=tau_e, search_duration=search_duration)
        else:
            left = c
            c = d
            fc = fd
            d = left + (right - left) * invphi
            fd = objective(x0=x0, offset=d, tau_e=tau_e, search_duration=search_duration)
        it += 1

    best_offset = 0.5 * (left + right)
    best_objective = objective(
        x0=x0,
        offset=best_offset,
        tau_e=tau_e,
        search_duration=search_duration,
    )
    return best_offset, best_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x0", type=float, default=0.5, help="Initial decision variable.")
    parser.add_argument("--tau-e", type=float, default=1.0, help="Edge population time constant.")
    parser.add_argument(
        "--search-duration",
        type=int,
        default=50,
        help="Short no-interaction trial length used for each offset evaluation.",
    )
    parser.add_argument("--offset-min", type=float, default=-0.1, help="Lower search bound.")
    parser.add_argument("--offset-max", type=float, default=0.1, help="Upper search bound.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Golden-search interval tolerance.")
    parser.add_argument("--max-iter", type=int, default=50, help="Maximum golden-search iterations.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.offset_min >= args.offset_max:
        raise ValueError("offset-min must be smaller than offset-max")
    if args.search_duration <= 1:
        raise ValueError("search-duration must be greater than 1")

    baseline_summary = summarize_run(
        x0=args.x0,
        offset=0.0,
        tau_e=args.tau_e,
        search_duration=args.search_duration,
    )
    best_offset, best_objective = golden_section_search(
        x0=args.x0,
        tau_e=args.tau_e,
        search_duration=args.search_duration,
        left=args.offset_min,
        right=args.offset_max,
        tol=args.tol,
        max_iter=args.max_iter,
    )
    best_summary = summarize_run(
        x0=args.x0,
        offset=best_offset,
        tau_e=args.tau_e,
        search_duration=args.search_duration,
    )

    print("Edge offset search summary")
    print(f"x0: {args.x0}")
    print(f"tau_E: {args.tau_e}")
    print(f"search_duration: {args.search_duration}")
    print(f"search_range: [{args.offset_min}, {args.offset_max}]")
    print(f"tol: {args.tol}")
    print(f"max_iter: {args.max_iter}")

    print("\nBaseline (offset=0.0)")
    for key, value in baseline_summary.items():
        print(f"{key}: {value}")

    print("\nBest candidate")
    print(f"best_offset: {best_offset}")
    print(f"best_objective: {best_objective}")
    for key, value in best_summary.items():
        print(f"{key}: {value}")

    print("\nImprovement")
    print(
        "theta_E_max_shift_delta: "
        f"{best_summary['theta_E_max_shift'] - baseline_summary['theta_E_max_shift']}"
    )
    print(
        "x_E_max_shift_delta: "
        f"{best_summary['x_E_max_shift'] - baseline_summary['x_E_max_shift']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
