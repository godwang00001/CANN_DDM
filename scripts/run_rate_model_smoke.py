#!/usr/bin/env python3
"""Deterministic smoke test for the coupled CANN/DDM rate model.

This script uses the parameter configuration from
`figures_code/fig2_micro_dyn_scheme.ipynb` as the baseline because those
settings are treated as user-validated. It then forces a no-cue, no-population-
noise condition and checks that the bump and edge attractors remain stable.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


@dataclass(frozen=True)
class SmokeThresholds:
    theta_e_max_shift: float = 1e-3
    theta_b_max_shift: float = 5e-3
    x_e_max_shift: float = 1e-7
    x_b_max_shift: float = 1e-7


def build_smoke_test_params() -> dict:
    """Return the validated fig2 parameter set for the no-cue smoke test."""
    geometry = {
        "num_units": 1024,
        "coding_limit": np.pi / 2,
        "coding_frac": 0.9,
        "clamp_frac": 0.1,
    }
    edge_pop = {
        "tau_E": 2,
        "c_EB": 0.3,
        "alpha_E": 1,
        "gamma_E": 10,
        "edge_type": "tanh",
        "noise_scale_edge": 0.0,
    }
    bump_pop = {
        "tau_B": 0.5,
        "c_BE": 0.3,
        "beta_B": 4,
        "c_BE_params": {"mode": "const"},
        "sigma_B": 0.1,
        "noise_scale_bump": 0.0,
    }
    decision_space_params = {
        "t_start": 200,
        "boundary": 1,
        "drift_rate": 1,
        # Keep this positive so the current constructor can initialize cue
        # generation, then overwrite the generated cues with zeros below.
        "noise_scale": 0.1,
        "dt_DDM": 1.0,
        "x0": 0.5,
        "dur1": 100,
        "dur2": 1000,
        "seed": 4,
    }
    return {
        "geometry": geometry,
        "edge_pop": edge_pop,
        "bump_pop": bump_pop,
        "decision_space_params": decision_space_params,
    }


def summarize_run(model: CANN_DDM_model) -> dict:
    model.cue_R_all[:] = 0
    model.cue_L_all[:] = 0
    runner = model.run_simulation(
        mon_vars=["theta_E", "theta_B", "x_E", "x_B", "hit_boundary"],
        progress_bar=False,
        dt=1.0,
        get_RT=False,
    )

    theta_e = np.asarray(runner.mon.theta_E).reshape(-1)
    theta_b = np.asarray(runner.mon.theta_B).reshape(-1)
    x_e = np.asarray(runner.mon.x_E).reshape(-1)
    x_b = np.asarray(runner.mon.x_B).reshape(-1)
    hit_boundary = np.asarray(runner.mon.hit_boundary).reshape(-1)

    return {
        "theta_E_init": float(theta_e[0]),
        "theta_E_final": float(theta_e[-1]),
        "theta_E_max_shift": float(np.max(np.abs(theta_e - theta_e[0]))),
        "theta_B_init": float(theta_b[0]),
        "theta_B_final": float(theta_b[-1]),
        "theta_B_max_shift": float(np.max(np.abs(theta_b - theta_b[0]))),
        "x_E_init": float(x_e[0]),
        "x_E_final": float(x_e[-1]),
        "x_E_max_shift": float(np.max(np.abs(x_e - x_e[0]))),
        "x_B_init": float(x_b[0]),
        "x_B_final": float(x_b[-1]),
        "x_B_max_shift": float(np.max(np.abs(x_b - x_b[0]))),
        "hit_boundary_any": bool(np.any(hit_boundary)),
        "all_finite": bool(
            np.isfinite(theta_e).all()
            and np.isfinite(theta_b).all()
            and np.isfinite(x_e).all()
            and np.isfinite(x_b).all()
        ),
    }


def evaluate_summary(summary: dict, thresholds: SmokeThresholds) -> list[str]:
    failures: list[str] = []
    if not summary["all_finite"]:
        failures.append("monitored values contain NaN or inf")
    if summary["hit_boundary_any"]:
        failures.append("boundary was hit during no-cue stability run")
    if summary["theta_E_max_shift"] > thresholds.theta_e_max_shift:
        failures.append(
            f"theta_E drift {summary['theta_E_max_shift']:.6g} exceeds "
            f"{thresholds.theta_e_max_shift:.6g}"
        )
    if summary["theta_B_max_shift"] > thresholds.theta_b_max_shift:
        failures.append(
            f"theta_B drift {summary['theta_B_max_shift']:.6g} exceeds "
            f"{thresholds.theta_b_max_shift:.6g}"
        )
    if summary["x_E_max_shift"] > thresholds.x_e_max_shift:
        failures.append(
            f"x_E drift {summary['x_E_max_shift']:.6g} exceeds "
            f"{thresholds.x_e_max_shift:.6g}"
        )
    if summary["x_B_max_shift"] > thresholds.x_b_max_shift:
        failures.append(
            f"x_B drift {summary['x_B_max_shift']:.6g} exceeds "
            f"{thresholds.x_b_max_shift:.6g}"
        )
    return failures


def main() -> int:
    params = build_smoke_test_params()
    thresholds = SmokeThresholds()
    model = CANN_DDM_model(CANN_params=params)
    summary = summarize_run(model)
    failures = evaluate_summary(summary, thresholds)

    print("Smoke test summary")
    for key, value in summary.items():
        print(f"{key}: {value}")

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
