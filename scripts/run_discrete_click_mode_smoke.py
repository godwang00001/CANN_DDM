#!/usr/bin/env python3
"""Smoke checks for the discrete 1 ms click-mode preprocessing path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


def build_params(**decision_overrides) -> dict:
    return {
        "edge_pop": {
            "tau_E": 2.0,
            "c_EB": 0.3,
            "alpha_E": 1.0,
            "gamma_E": 10.0,
            "edge_type": "tanh",
            "offset": 0.0,
            "noise_scale_edge": 0.0,
        },
        "bump_pop": {
            "tau_B": 0.5,
            "c_BE": 0.3,
            "beta_B": 4.0,
            "c_BE_params": {"mode": "const"},
            "sigma_B": 0.1,
            "noise_scale_bump": 0.0,
        },
        "decision_space_params": {
            "decision_mode": "discrete",
            "t_start": 20,
            "boundary": 1.0,
            "drift_rate": 0.0,
            "noise_scale": 0.1,
            "dt_DDM": 5.0,
            "lambda_click_L": 0.2,
            "lambda_click_R": 0.6,
            "delta_click_x": 0.05,
            "x0": 0.5,
            "dur": 120,
            "seed": 7,
        } | decision_overrides,
        "geometry": {
            "coding_limit": float(np.pi / 2),
            "num_units": 256,
            "coding_frac": 0.3,
            "clamp_frac": 0.1,
        },
    }


def assert_close(name: str, lhs, rhs, atol: float = 1e-10) -> None:
    if not np.allclose(lhs, rhs, atol=atol, rtol=0.0):
        raise AssertionError(f"{name} mismatch")


def check_preprocessing() -> None:
    model = CANN_DDM_model(CANN_params=build_params())
    assert model.click_L_all.shape == (model.max_time,)
    assert model.click_R_all.shape == (model.max_time,)
    assert np.all(model.click_L_all[:model.t_start] == 0.0)
    assert np.all(model.click_R_all[:model.t_start] == 0.0)
    assert np.all(np.isin(model.click_L_all, [0.0, 1.0]))
    assert np.all(np.isin(model.click_R_all, [0.0, 1.0]))
    assert_close("v_drive_all", model.v_drive_all, model.v_drift_all + model.v_noise_all)


def check_windowed_click_drift() -> None:
    model = CANN_DDM_model(
        CANN_params=build_params(
            noise_scale=0.0,
            dt_DDM=10.0,
            lambda_click_L=0.0,
            lambda_click_R=0.0,
            delta_click_x=0.1,
        )
    )
    click_R_all = np.zeros(model.dur, dtype=float)
    click_L_all = np.zeros(model.dur, dtype=float)
    click_R_all[model.t_start] = 1.0
    v_drift = model.build_discrete_click_drift(
        click_R_all,
        click_L_all,
        model.t_start,
        model.dt_DDM,
        model.delta_click_x,
        model.drive_x_speed_unit,
    )
    held = model.delta_click_x / (model.drive_x_speed_unit * model.dt_DDM_ms)
    expected = np.zeros(model.dur, dtype=float)
    expected[model.t_start:model.t_start + model.dt_DDM_ms] = held
    assert_close("windowed v_drift", v_drift, expected)
    if abs(np.sum(v_drift) - model.delta_click_x / model.drive_x_speed_unit) > 1e-10:
        raise AssertionError("windowed v_drift should preserve the total click displacement")


def check_zero_motion() -> None:
    model = CANN_DDM_model(
        CANN_params=build_params(
            noise_scale=0.0,
            lambda_click_L=0.0,
            lambda_click_R=0.0,
            delta_click_x=0.05,
        )
    )
    assert_close("zero-motion x_traj", model.x_traj, np.full(model.dur, model.x0))


def check_one_sided_click_motion() -> None:
    model = CANN_DDM_model(
        CANN_params=build_params(
            noise_scale=0.0,
            lambda_click_L=0.0,
            lambda_click_R=0.0,
            delta_click_x=0.1,
        )
    )
    click_R_all = np.zeros(model.dur, dtype=float)
    click_L_all = np.zeros(model.dur, dtype=float)
    click_R_all[model.t_start:model.t_start + 4] = 1.0
    x_traj = model.get_x_traj_discrete(
        model.t_start,
        model.dur,
        model.delta_click_x,
        model.x0,
        click_R_all,
        click_L_all,
        np.zeros(model.dur, dtype=float),
        0.0,
        model.boundary,
    )
    diffs = np.diff(x_traj[model.t_start:model.t_start + 5])
    if not np.all(diffs >= -1e-10):
        raise AssertionError("one-sided clicks should not decrease x_traj")


def check_same_ms_cancellation() -> None:
    model = CANN_DDM_model(
        CANN_params=build_params(
            noise_scale=0.0,
            lambda_click_L=0.0,
            lambda_click_R=0.0,
            delta_click_x=0.1,
        )
    )
    click_R_all = np.zeros(model.dur, dtype=float)
    click_L_all = np.zeros(model.dur, dtype=float)
    click_R_all[model.t_start] = 1.0
    click_L_all[model.t_start] = 1.0
    x_traj = model.get_x_traj_discrete(
        model.t_start,
        model.dur,
        model.delta_click_x,
        model.x0,
        click_R_all,
        click_L_all,
        np.zeros(model.dur, dtype=float),
        0.0,
        model.boundary,
    )
    if abs(x_traj[model.t_start] - model.x0) > 1e-10:
        raise AssertionError("same-ms opposite clicks should cancel")


def check_extended_horizon_padding() -> None:
    model = CANN_DDM_model(
        CANN_params=build_params(
            dur=120,
            max_time=160,
            noise_scale=0.0,
            lambda_click_L=0.0,
            lambda_click_R=0.0,
        )
    )
    if model.click_L_all.shape != (model.max_time,):
        raise AssertionError("click_L_all should span max_time")
    if model.click_R_all.shape != (model.max_time,):
        raise AssertionError("click_R_all should span max_time")
    if not np.all(model.click_L_all[model.dur:] == 0.0):
        raise AssertionError("left clicks should be zero after dur")
    if not np.all(model.click_R_all[model.dur:] == 0.0):
        raise AssertionError("right clicks should be zero after dur")
    if not np.all(model.v_drift_all[model.dur:] == 0.0):
        raise AssertionError("v_drift should be zero after dur")


def main() -> int:
    check_preprocessing()
    check_windowed_click_drift()
    check_zero_motion()
    check_one_sided_click_motion()
    check_same_ms_cancellation()
    check_extended_horizon_padding()
    print("PASS discrete click-mode smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
