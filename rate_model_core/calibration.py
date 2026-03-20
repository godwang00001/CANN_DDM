from __future__ import annotations

from copy import deepcopy

import numpy as np


def _model_params(model) -> dict:
    return {
        "geometry": {
            "num_units": int(model.geometry.num_units),
            "coding_limit": float(model.geometry.coding_limit),
            "coding_frac": float(model.geometry.coding_frac),
            "clamp_frac": float(model.geometry.clamp_frac),
        },
        "edge_pop": {
            "tau_E": float(model.tau_E),
            "alpha_E": float(model.alpha_E),
            "gamma_E": float(model.gamma_E),
            "noise_scale_edge": float(model.noise_scale_edge),
            "edge_type": model.edge_type,
            "c_EB": float(model.c_EB),
            "offset": float(model.offset),
            "eb_kernel_mode": model.eb_kernel_mode,
            "eb_kernel_sigma": float(model.eb_kernel_sigma),
            "eb_kernel_shift": float(model.eb_kernel_shift),
            "eb_kernel_gain": float(model.eb_kernel_gain),
        },
        "bump_pop": {
            "tau_B": float(model.tau_B),
            "c_BE": float(model.c_BE),
            "c_BE_params": dict(model.c_BE_params),
            "noise_scale_bump": float(model.noise_scale_bump),
            "sigma_B": float(model.sigma_B),
            "sigma_I_BE": None if model.sigma_I_BE is None else float(model.sigma_I_BE),
            "beta_B": float(model.beta_B),
            "kernel_mode": model.bump_kernel_mode,
            "kernel_gain": float(model.bump_kernel_gain),
            "kernel_sigma": None if model.bump_kernel_sigma is None else float(model.bump_kernel_sigma),
            "kernel_normed": bool(model.bump_kernel_normed),
            "be_kernel_mode": model.be_kernel_mode,
            "be_kernel_sigma": float(model.be_kernel_sigma),
            "be_kernel_gain": float(model.be_kernel_gain),
        },
        "decision_space_params": {
            "decision_mode": model.decision_mode,
            "t_start": int(model.t_start),
            "dur": int(model.dur),
            "boundary": float(model.boundary),
            "drift_rate": float(model.drift_rate),
            "noise_scale": float(model.noise_scale),
            "dt_DDM": float(model.dt_DDM),
            "x0": float(model.x0),
            "seed": model.seed,
        },
    }


def _estimate_slope(times: np.ndarray, values: np.ndarray, window: tuple[float, float]) -> float:
    start, stop = window
    mask = (times >= start) & (times <= stop)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(times[mask], values[mask], deg=1)
    return float(slope)


def _accumulation_fit_window(times: np.ndarray, *, t_start: float, boundary_index: int | None) -> tuple[float, float]:
    stop = float(times[boundary_index]) if boundary_index is not None else float(times[-1])
    return float(t_start), stop


def _normalized_dx_dtheta(theta, *, boundary: float, theta_min: float, theta_max: float, gamma: float):
    interval = theta_max - theta_min
    normalization = 1.0 - np.exp(-gamma * interval)
    return boundary * gamma * np.exp(-gamma * (theta - theta_min)) / normalization


def _constant_drive_model(model, *, c_be: float, x0: float):
    params = deepcopy(_model_params(model))
    params["bump_pop"]["c_BE"] = float(c_be)
    params["bump_pop"]["c_BE_params"] = {"mode": "const"}
    params["bump_pop"]["noise_scale_bump"] = 0.0
    params["edge_pop"]["noise_scale_edge"] = 0.0
    params["decision_space_params"]["x0"] = float(x0)
    params["decision_space_params"]["drift_rate"] = 0.0
    params["decision_space_params"]["noise_scale"] = 0.0

    class ConstantDriveCalibrationModel(type(model)):
        def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
            filtered_r_B = self.W_BE @ r_B
            return c_BE * filtered_r_B

    calibration_model = ConstantDriveCalibrationModel(CANN_params=params)
    calibration_model.cue_R_all[:] = 0.0
    calibration_model.cue_L_all[:] = 0.0
    calibration_model.v_drift_all[:] = 0.0
    calibration_model.v_noise_all[:] = 0.0
    calibration_model.v_drive_all[:] = 0.0
    return calibration_model


def _run_constant_drive_trial(
    model,
    *,
    c_be: float,
    x0: float,
    min_accumulation_samples: int,
    mean_abs_tol: float,
    std_abs_tol: float,
) -> dict:
    calibration_model = _constant_drive_model(model, c_be=c_be, x0=x0)
    runner = calibration_model.run_simulation(
        mon_vars=["theta_E", "theta_B", "x_E", "hit_boundary"],
        progress_bar=False,
        dt=1.0,
        get_RT=False,
    )

    times = np.arange(len(runner.mon.theta_E), dtype=float)
    theta_e = np.asarray(runner.mon.theta_E).reshape(-1)
    theta_b = np.asarray(runner.mon.theta_B).reshape(-1)
    x_e = np.asarray(runner.mon.x_E).reshape(-1)
    hit = np.asarray(runner.mon.hit_boundary).reshape(-1)
    hit_indices = np.flatnonzero(hit)
    boundary_index = int(hit_indices[0]) if hit_indices.size else None
    fit_window = _accumulation_fit_window(
        times,
        t_start=float(calibration_model.t_start),
        boundary_index=boundary_index,
    )
    accumulation_sample_count = int(np.count_nonzero((times >= fit_window[0]) & (times <= fit_window[1])))

    abs_delta = np.abs(theta_b - theta_e)
    abs_delta_pre_boundary = abs_delta[:boundary_index] if boundary_index is not None else abs_delta
    alignment_mean_abs = float(np.mean(abs_delta_pre_boundary)) if abs_delta_pre_boundary.size else float("nan")
    alignment_std_abs = float(np.std(abs_delta_pre_boundary)) if abs_delta_pre_boundary.size else float("nan")

    valid = (
        accumulation_sample_count >= min_accumulation_samples
        and np.isfinite(alignment_mean_abs)
        and np.isfinite(alignment_std_abs)
        and alignment_mean_abs <= mean_abs_tol
        and alignment_std_abs <= std_abs_tol
    )
    return {
        "c_BE": float(c_be),
        "v_theta_E": _estimate_slope(times, theta_e, fit_window),
        "alignment_mean_abs": alignment_mean_abs,
        "alignment_std_abs": alignment_std_abs,
        "valid": bool(valid),
        "x_E": x_e,
    }


def _fit_kappa_from_sweep(
    model,
    *,
    c_be_sweep: np.ndarray,
    x0: float,
    min_accumulation_samples: int,
    mean_abs_tol: float,
    std_abs_tol: float,
) -> dict:
    sweep_results = [
        _run_constant_drive_trial(
            model,
            c_be=float(c_be),
            x0=x0,
            min_accumulation_samples=min_accumulation_samples,
            mean_abs_tol=float(mean_abs_tol),
            std_abs_tol=float(std_abs_tol),
        )
        for c_be in c_be_sweep
    ]

    c_be_vals = np.array([row["c_BE"] for row in sweep_results], dtype=float)
    v_theta_e_vals = np.array([row["v_theta_E"] for row in sweep_results], dtype=float)
    valid_mask = np.array([row["valid"] for row in sweep_results], dtype=bool)

    if np.count_nonzero(valid_mask) < 2:
        raise ValueError("Not enough valid c_BE sweep points to calibrate kappa.")

    kappa, _ = np.polyfit(c_be_vals[valid_mask], v_theta_e_vals[valid_mask], deg=1)
    kappa = float(kappa)
    if np.isclose(kappa, 0.0):
        raise ValueError("Calibrated kappa is zero; cannot build target_diffusion c_BE(theta).")

    valid_c_be_max = float(np.max(np.abs(c_be_vals[valid_mask])))
    sweep_c_be_max = float(np.max(np.abs(c_be_vals)))
    return {
        "kappa": kappa,
        "valid_c_be_max": valid_c_be_max,
        "sweep_c_be_max": sweep_c_be_max,
        "sweep_results": sweep_results,
    }


def _target_diffusion_profile(model, *, kappa: float, theta_margin: float, drive_x_speed_unit: float) -> dict:
    theta_min = float(model.geometry.coding_theta_min)
    theta_max = float(model.geometry.coding_theta_max)
    theta_grid = np.linspace(theta_min + theta_margin, theta_max - theta_margin, 500)
    dx_dtheta = _normalized_dx_dtheta(
        theta_grid,
        boundary=float(model.boundary),
        theta_min=theta_min,
        theta_max=theta_max,
        gamma=float(model.gamma_E),
    )
    c_be_theta = float(drive_x_speed_unit) / (float(kappa) * dx_dtheta)
    return {
        "theta_grid": theta_grid,
        "dx_dtheta": dx_dtheta,
        "c_be_theta": c_be_theta,
        "c_be_theta_max": float(np.max(np.abs(c_be_theta))),
    }


def _max_abs_v_drive(model) -> float:
    if hasattr(model, "v_drive_all"):
        return float(np.max(np.abs(np.asarray(model.v_drive_all, dtype=float))))
    return 1.0


def calibrate_target_diffusion_profile(
    model,
    *,
    calibration_x0: float | None = None,
    c_be_sweep=None,
    min_accumulation_samples: int = 200,
    mean_abs_tol: float | None = None,
    std_abs_tol: float | None = None,
    theta_margin: float = 0.02,
    sweep_expand_factor: float = 1.1,
    kappa_tol: float = 0.1,
) -> dict:
    x0 = float(model.x0 if calibration_x0 is None else calibration_x0)
    drive_x_speed_unit = float(model.c_BE_params.get("drive_x_speed_unit", 3.0e-4))
    if drive_x_speed_unit <= 0:
        raise ValueError("c_BE_params['drive_x_speed_unit'] must be positive for target_diffusion calibration")
    theta_spacing = float((model.theta_max - model.theta_min) / (model.num_E - 1))
    if mean_abs_tol is None:
        mean_abs_tol = 3.0 * theta_spacing
    if std_abs_tol is None:
        std_abs_tol = 2.0 * theta_spacing
    if c_be_sweep is None:
        c_be_sweep = np.linspace(-0.2, 0.2, 10)
    c_be_sweep = np.asarray(c_be_sweep, dtype=float)

    first_fit = _fit_kappa_from_sweep(
        model,
        c_be_sweep=c_be_sweep,
        x0=x0,
        min_accumulation_samples=min_accumulation_samples,
        mean_abs_tol=float(mean_abs_tol),
        std_abs_tol=float(std_abs_tol),
    )

    first_profile = _target_diffusion_profile(
        model,
        kappa=float(first_fit["kappa"]),
        theta_margin=float(theta_margin),
        drive_x_speed_unit=drive_x_speed_unit,
    )
    max_abs_v_drive = _max_abs_v_drive(model)
    first_effective_c_be_max = float(first_profile["c_be_theta_max"] * max_abs_v_drive)

    final_fit = first_fit
    final_profile = first_profile
    kappa_rel_error = 0.0
    if first_effective_c_be_max > first_fit["sweep_c_be_max"]:
        expanded_abs_max = max(
            first_fit["sweep_c_be_max"],
            float(sweep_expand_factor) * first_effective_c_be_max,
        )
        expanded_sweep = np.linspace(-expanded_abs_max, expanded_abs_max, len(c_be_sweep))
        try:
            final_fit = _fit_kappa_from_sweep(
                model,
                c_be_sweep=expanded_sweep,
                x0=x0,
                min_accumulation_samples=min_accumulation_samples,
                mean_abs_tol=float(mean_abs_tol),
                std_abs_tol=float(std_abs_tol),
            )
            final_profile = _target_diffusion_profile(
                model,
                kappa=float(final_fit["kappa"]),
                theta_margin=float(theta_margin),
                drive_x_speed_unit=drive_x_speed_unit,
            )
            kappa_rel_error = float(abs(final_fit["kappa"] - first_fit["kappa"]) / abs(first_fit["kappa"]))
        except ValueError:
            final_fit = first_fit
            final_profile = first_profile
            kappa_rel_error = float("inf")

    effective_c_be_theta = np.abs(final_profile["c_be_theta"]) * max_abs_v_drive
    effective_c_be_max = float(np.max(effective_c_be_theta))
    trusted_theta_mask = effective_c_be_theta <= float(final_fit["valid_c_be_max"])
    certificate_passed = bool(
        np.all(trusted_theta_mask)
        and effective_c_be_max <= float(final_fit["valid_c_be_max"])
        and kappa_rel_error <= float(kappa_tol)
    )
    return {
        "kappa": float(final_fit["kappa"]),
        "c_be_theta_max": float(final_profile["c_be_theta_max"]),
        "effective_c_be_max": effective_c_be_max,
        "valid_c_be_max": float(final_fit["valid_c_be_max"]),
        "max_abs_v_drive": max_abs_v_drive,
        "kappa_rel_error": kappa_rel_error,
        "certificate_passed": certificate_passed,
        "trusted_theta_mask": trusted_theta_mask,
        "theta_grid": final_profile["theta_grid"],
        "c_be_theta": final_profile["c_be_theta"],
        "sweep_results": final_fit["sweep_results"],
    }
