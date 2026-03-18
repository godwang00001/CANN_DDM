#!/usr/bin/env python3
"""Scan bump-population parameters for stable live-r_B coupling regimes.

This study keeps the main model code unchanged and evaluates a live-r_B
definition of I_BE against the canonical shifted-bump reference under the
validated Figure 2 cue schedule.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


class CanonicalShiftedBumpIBEModel(CANN_DDM_model):
    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        center = self.find_current_edge_location(self.r_E)
        canonical_bump = self.bump_states(
            self.num_B,
            self.sigma_B,
            self.bump_geometry,
            center_pos=center,
        )
        return c_BE * (cue_R * canonical_bump + cue_L * (-canonical_bump))


class LiveBumpIBEModel(CANN_DDM_model):
    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        return c_BE * (cue_R * r_B + cue_L * (-r_B))


def parse_grid(spec: str, cast=float) -> list:
    values = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(cast(piece))
    return values


def build_params(
    *,
    c_be: float,
    c_eb: float,
    beta_b: float,
    sigma_b: float,
    tau_b: float,
    kernel_mode: str = "gaussian_cann",
    kernel_gain: float = 2.0,
    kernel_sigma: float | None = None,
    kernel_normed: bool = True,
) -> dict:
    return {
        "edge_pop": {
            "tau_E": 2,
            "c_EB": c_eb,
            "alpha_E": 1,
            "gamma_E": 10,
            "edge_type": "tanh",
            "offset": 0.0,
            "noise_scale_edge": 0.0,
        },
        "bump_pop": {
            "tau_B": tau_b,
            "c_BE": c_be,
            "beta_B": beta_b,
            "c_BE_params": {"mode": "const"},
            "sigma_B": sigma_b,
            "noise_scale_bump": 0.0,
            "kernel_mode": kernel_mode,
            "kernel_gain": kernel_gain,
            "kernel_sigma": kernel_sigma,
            "kernel_normed": kernel_normed,
        },
        "decision_space_params": {
            "t_start": 200,
            "boundary": 1,
            "drift_rate": 1,
            "noise_scale": 0.1,
            "dt_DDM": 1.0,
            "x0": 0.5,
            "dur1": 100,
            "dur2": 1000,
            "seed": 4,
        },
        "geometry": {
            "coding_limit": float(np.pi / 2),
            "num_units": 1024,
            "coding_frac": 0.9,
            "clamp_frac": 0.1,
        },
    }


def apply_figure_cues(model: CANN_DDM_model) -> None:
    t_start = int(model.t_start)
    t1 = 200
    t2 = 400
    model.cue_R_all[t_start:t_start + t1] = 1
    model.cue_R_all[t_start + t1:t_start + t2] = 0
    model.cue_L_all[t_start:t_start + t1] = 0
    model.cue_L_all[t_start + t1:t_start + t2] = 1


def run_variant(model_cls: type[CANN_DDM_model], params: dict) -> dict:
    model = model_cls(CANN_params=params)
    apply_figure_cues(model)
    runner = model.run_simulation(
        mon_vars=["theta_E", "theta_B", "x_E", "x_B", "I_BE", "I_EB", "hit_boundary", "r_B", "r_E"],
        progress_bar=False,
        dt=1.0,
        get_RT=False,
    )

    theta_e = np.asarray(runner.mon.theta_E).reshape(-1)
    theta_b = np.asarray(runner.mon.theta_B).reshape(-1)
    x_e = np.asarray(runner.mon.x_E).reshape(-1)
    x_b = np.asarray(runner.mon.x_B).reshape(-1)
    i_be = np.asarray(runner.mon.I_BE)
    i_eb = np.asarray(runner.mon.I_EB)
    hit = np.asarray(runner.mon.hit_boundary).reshape(-1)
    r_b = np.asarray(runner.mon.r_B)
    r_e = np.asarray(runner.mon.r_E)

    return {
        "theta_E_final": float(theta_e[-1]),
        "theta_B_final": float(theta_b[-1]),
        "x_E_final": float(x_e[-1]),
        "x_B_final": float(x_b[-1]),
        "theta_E_path": theta_e,
        "theta_B_path": theta_b,
        "x_E_path": x_e,
        "x_B_path": x_b,
        "I_BE_max_abs": float(np.max(np.abs(i_be))),
        "I_EB_max_abs": float(np.max(np.abs(i_eb))),
        "hit_any": bool(np.any(hit)),
        "all_finite": bool(
            np.all(np.isfinite(theta_e))
            and np.all(np.isfinite(theta_b))
            and np.all(np.isfinite(x_e))
            and np.all(np.isfinite(x_b))
            and np.all(np.isfinite(i_be))
            and np.all(np.isfinite(i_eb))
            and np.all(np.isfinite(r_b))
            and np.all(np.isfinite(r_e))
        ),
    }


def score_candidate(reference: dict, candidate: dict) -> dict:
    theta_e_rmse = float(np.sqrt(np.mean((candidate["theta_E_path"] - reference["theta_E_path"]) ** 2)))
    theta_b_rmse = float(np.sqrt(np.mean((candidate["theta_B_path"] - reference["theta_B_path"]) ** 2)))
    x_e_rmse = float(np.sqrt(np.mean((candidate["x_E_path"] - reference["x_E_path"]) ** 2)))
    x_b_rmse = float(np.sqrt(np.mean((candidate["x_B_path"] - reference["x_B_path"]) ** 2)))
    return {
        "theta_E_rmse": theta_e_rmse,
        "theta_B_rmse": theta_b_rmse,
        "x_E_rmse": x_e_rmse,
        "x_B_rmse": x_b_rmse,
        "trajectory_score": theta_e_rmse + theta_b_rmse + x_e_rmse + x_b_rmse,
    }


def is_stable(result: dict, *, max_i_be: float, max_i_eb: float) -> bool:
    return (
        result["all_finite"]
        and not result["hit_any"]
        and result["I_BE_max_abs"] <= max_i_be
        and result["I_EB_max_abs"] <= max_i_eb
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c-be", type=float, default=0.3)
    parser.add_argument("--c-be-grid", default="")
    parser.add_argument("--c-eb-grid", default="0.2,0.3,0.4")
    parser.add_argument("--beta-b-grid", default="4,6,8,10,12,16,20")
    parser.add_argument("--sigma-b-grid", default="0.05,0.075,0.1,0.125,0.15,0.2")
    parser.add_argument("--tau-b-grid", default="0.5")
    parser.add_argument("--kernel-mode", default="gaussian_cann")
    parser.add_argument("--kernel-gain-grid", default="")
    parser.add_argument("--kernel-sigma-grid", default="")
    parser.add_argument("--kernel-normed", default="true")
    parser.add_argument("--max-i-be", type=float, default=1.0)
    parser.add_argument("--max-i-eb", type=float, default=3.0)
    parser.add_argument("--csv-out", default="")
    args = parser.parse_args()

    c_be_grid = parse_grid(args.c_be_grid, float) if args.c_be_grid else [args.c_be]
    c_eb_grid = parse_grid(args.c_eb_grid, float)
    beta_b_grid = parse_grid(args.beta_b_grid, float)
    sigma_b_grid = parse_grid(args.sigma_b_grid, float)
    tau_b_grid = parse_grid(args.tau_b_grid, float)
    kernel_gain_grid = parse_grid(args.kernel_gain_grid, float) if args.kernel_gain_grid else [2.0]
    kernel_sigma_grid = parse_grid(args.kernel_sigma_grid, float) if args.kernel_sigma_grid else [None]
    kernel_normed = args.kernel_normed.lower() in {"1", "true", "yes", "y"}

    reference_c_be = c_be_grid[0]
    reference_params = build_params(
        c_be=reference_c_be,
        c_eb=0.3,
        beta_b=4.0,
        sigma_b=0.1,
        tau_b=0.5,
        kernel_mode="gaussian_cann",
        kernel_gain=2.0,
        kernel_sigma=None,
        kernel_normed=True,
    )
    reference = run_variant(CanonicalShiftedBumpIBEModel, reference_params)

    rows = []
    for c_be in c_be_grid:
        for c_eb in c_eb_grid:
            for beta_b in beta_b_grid:
                for sigma_b in sigma_b_grid:
                    for tau_b in tau_b_grid:
                        for kernel_gain in kernel_gain_grid:
                            for kernel_sigma in kernel_sigma_grid:
                                params = build_params(
                                    c_be=c_be,
                                    c_eb=c_eb,
                                    beta_b=beta_b,
                                    sigma_b=sigma_b,
                                    tau_b=tau_b,
                                    kernel_mode=args.kernel_mode,
                                    kernel_gain=kernel_gain,
                                    kernel_sigma=kernel_sigma,
                                    kernel_normed=kernel_normed,
                                )
                                candidate = run_variant(LiveBumpIBEModel, params)
                                score = score_candidate(reference, candidate)
                                stable = is_stable(candidate, max_i_be=args.max_i_be, max_i_eb=args.max_i_eb)
                                rows.append(
                                    {
                                        "c_BE": c_be,
                                        "c_EB": c_eb,
                                        "beta_B": beta_b,
                                        "sigma_B": sigma_b,
                                        "tau_B": tau_b,
                                        "kernel_mode": args.kernel_mode,
                                        "kernel_gain": kernel_gain,
                                        "kernel_sigma": kernel_sigma,
                                        "kernel_normed": kernel_normed,
                                        "stable": stable,
                                        "all_finite": candidate["all_finite"],
                                        "hit_any": candidate["hit_any"],
                                        "I_BE_max_abs": candidate["I_BE_max_abs"],
                                        "I_EB_max_abs": candidate["I_EB_max_abs"],
                                        "theta_E_final": candidate["theta_E_final"],
                                        "theta_B_final": candidate["theta_B_final"],
                                        "x_E_final": candidate["x_E_final"],
                                        "x_B_final": candidate["x_B_final"],
                                        "theta_E_rmse": score["theta_E_rmse"],
                                        "theta_B_rmse": score["theta_B_rmse"],
                                        "x_E_rmse": score["x_E_rmse"],
                                        "x_B_rmse": score["x_B_rmse"],
                                        "trajectory_score": score["trajectory_score"],
                                    }
                                )

    rows.sort(
        key=lambda row: (
            not row["stable"],
            row["trajectory_score"],
            row["I_EB_max_abs"],
            row["I_BE_max_abs"],
        )
    )

    print("Reference canonical baseline")
    print(f"c_BE: {reference_c_be}")
    print(f"theta_E_final: {reference['theta_E_final']}")
    print(f"theta_B_final: {reference['theta_B_final']}")
    print(f"I_BE_max_abs: {reference['I_BE_max_abs']}")
    print(f"I_EB_max_abs: {reference['I_EB_max_abs']}")

    print("\nTop candidates")
    for row in rows[:10]:
        print(
            "stable={stable} c_EB={c_EB} beta_B={beta_B} sigma_B={sigma_B} tau_B={tau_B} "
            "traj={trajectory_score:.6f} I_BE={I_BE_max_abs:.6f} I_EB={I_EB_max_abs:.6f} "
            "theta_E={theta_E_final:.6f} theta_B={theta_B_final:.6f}".format(**row)
        )

    stable_rows = [row for row in rows if row["stable"]]
    print(f"\nStable candidates: {len(stable_rows)} / {len(rows)}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        fieldnames = list(rows[0].keys()) if rows else [
            "c_BE", "c_EB", "beta_B", "sigma_B", "tau_B", "kernel_mode", "kernel_gain",
            "kernel_sigma", "kernel_normed", "stable", "all_finite",
            "hit_any", "I_BE_max_abs", "I_EB_max_abs", "theta_E_final", "theta_B_final",
            "x_E_final", "x_B_final", "theta_E_rmse", "theta_B_rmse", "x_E_rmse",
            "x_B_rmse", "trajectory_score",
        ]
        write_csv(csv_path, rows, fieldnames)
        print(f"Saved CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
