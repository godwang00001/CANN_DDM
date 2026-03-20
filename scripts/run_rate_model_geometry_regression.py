#!/usr/bin/env python3
"""Regression check for the geometry-config refactor.

This compares the legacy index-space-facing configuration path against the new
shared `geometry` config path under the cue-driven figure-2 microdynamics setup.
The two runs should remain numerically identical to within tight tolerances.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


def build_legacy_params() -> dict:
    edge_pop = {
        "num_E": 1024,
        "tau_E": 2,
        "c_EB": 0.3,
        "alpha_E": 1,
        "gamma_E": 10,
        "clamp_frac_E": 0.1,
        "edge_type": "tanh",
        "offset": 0.0,
        "noise_scale_edge": 0.0,
    }
    bump_pop = {
        "num_B": 1024,
        "tau_B": 0.5,
        "c_BE": 0.3,
        "beta_B": 4,
        "c_BE_params": {"mode": "const"},
        "sigma_B": 0.1,
        "clamp_frac_B": 0.1,
        "noise_scale_bump": 0.0,
    }
    decision_space_params = {
        "decision_mode": "discrete",
        "t_start": 200,
        "boundary": 1,
        "drift_rate": 1,
        "noise_scale": 0.1,
        "dt_DDM": 1.0,
        "x0": 0.5,
        "dur": 1100,
        "seed": 4,
    }
    return {
        "edge_pop": edge_pop,
        "bump_pop": bump_pop,
        "decision_space_params": decision_space_params,
    }


def build_geometry_params() -> dict:
    params = build_legacy_params()
    params["edge_pop"].pop("num_E", None)
    params["edge_pop"].pop("clamp_frac_E", None)
    params["bump_pop"].pop("num_B", None)
    params["bump_pop"].pop("clamp_frac_B", None)
    params["geometry"] = {
        "coding_limit": float(np.pi / 2),
        "num_units": 1024,
        "coding_frac": 0.3,
        "clamp_frac": 0.1,
    }
    return params


def apply_figure_cues(model: CANN_DDM_model) -> None:
    t_start = int(model.t_start)
    t1 = 200
    t2 = 400
    model.cue_R_all[t_start:t_start + t1] = 1
    model.cue_R_all[t_start + t1:t_start + t2] = 0
    model.cue_L_all[t_start:t_start + t1] = 0
    model.cue_L_all[t_start + t1:t_start + t2] = 1


def summarize_run(model: CANN_DDM_model) -> dict:
    apply_figure_cues(model)
    runner = model.run_simulation(
        mon_vars=["theta_E", "theta_B", "x_E", "x_B", "I_BE", "I_EB", "hit_boundary"],
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
    return {
        "theta_E_final": float(theta_e[-1]),
        "theta_B_final": float(theta_b[-1]),
        "theta_E_max": float(np.max(theta_e)),
        "theta_E_min": float(np.min(theta_e)),
        "theta_B_max": float(np.max(theta_b)),
        "theta_B_min": float(np.min(theta_b)),
        "x_E_final": float(x_e[-1]),
        "x_B_final": float(x_b[-1]),
        "I_BE_max_abs": float(np.max(np.abs(i_be))),
        "I_EB_max_abs": float(np.max(np.abs(i_eb))),
        "hit_any": bool(np.any(hit)),
    }


def compare_summaries(reference: dict, candidate: dict, atol: float = 1e-8) -> list[str]:
    failures: list[str] = []
    for key, ref_value in reference.items():
        cand_value = candidate[key]
        if isinstance(ref_value, bool):
            if ref_value != cand_value:
                failures.append(f"{key} mismatch: {ref_value} vs {cand_value}")
        else:
            if abs(ref_value - cand_value) > atol:
                failures.append(
                    f"{key} mismatch: {ref_value:.12g} vs {cand_value:.12g} "
                    f"(abs diff {abs(ref_value - cand_value):.3g})"
                )
    return failures


def main() -> int:
    legacy_summary = summarize_run(CANN_DDM_model(CANN_params=build_legacy_params()))
    geometry_summary = summarize_run(CANN_DDM_model(CANN_params=build_geometry_params()))
    failures = compare_summaries(legacy_summary, geometry_summary)

    print("Legacy summary")
    for key, value in legacy_summary.items():
        print(f"{key}: {value}")

    print("\nGeometry summary")
    for key, value in geometry_summary.items():
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
