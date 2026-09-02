#!/usr/bin/env python3
"""Compare I_EB constructions on the saved unstable edge sample."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model
from rate_model_core.default_params import build_stable_default_params


def summarize_response(theta: np.ndarray, response: np.ndarray) -> dict:
    response = np.asarray(response, dtype=float)
    return {
        "max_abs": float(np.max(np.abs(response))),
        "argmax": int(np.argmax(response)),
        "peak_pos": float(theta[int(np.argmax(response))]),
        "tv": float(np.sum(np.abs(np.diff(response)))),
        "hf": float(np.sum(np.abs(np.diff(response, n=2)))),
        "local_peaks": int(
            np.sum((response[1:-1] > response[:-2]) & (response[1:-1] > response[2:]))
        ),
    }


def build_model(mode: str) -> CANN_DDM_model:
    params = build_stable_default_params()
    params["edge_pop"].update(
        {
            "eb_kernel_mode": mode,
            "eb_kernel_sigma": 5.0,
            "eb_kernel_gain": 100.0,
        }
    )
    return CANN_DDM_model(CANN_params=params)


def main() -> int:
    sample = np.load("figures_code/supp/r_E_instability.npy")
    modes = ["simple", "smoothed_derivative", "edge_readout_bump"]

    for mode in modes:
        model = build_model(mode)
        theta = np.linspace(model.theta_min, model.theta_max, model.num_E)
        i_eb = np.asarray(model.get_current_I_EB(sample, model.c_EB))
        theta_readout = float(model.find_current_edge_location(sample))

        print(mode)
        print(f"theta_readout: {theta_readout}")
        for key, value in summarize_response(theta, i_eb).items():
            print(f"{key}: {value}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
