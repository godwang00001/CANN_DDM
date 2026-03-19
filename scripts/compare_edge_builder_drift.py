#!/usr/bin/env python3
"""Compare structural J_EE variants on zero-input edge drift.

This script isolates the edge subsystem by turning off both coupling pathways,
all cue input, and all noise. It reports:

- static readout bias for translated canonical edge profiles
- dynamic drift of theta_E and x_E under zero input
- Goldstone-mode fixed-point residual versus translated edge center
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model
from rate_model_core.connectivity import (
    DoG_kernel,
    EDGE_KERNEL_BASE_EXC_SIGMA,
    EDGE_KERNEL_INHIBITION_WIDTH_RATIO,
    make_conn_mat_from_kernel,
)
from rate_model_core.default_params import build_stable_default_params

VARIANT_SPECS = (
    ("normalized", {"normed": True, "support_mode": "default", "boundary_mode": "truncate"}),
    ("normalized_reflect", {"normed": True, "support_mode": "default", "boundary_mode": "reflect"}),
    ("unnormalized", {"normed": False, "support_mode": "default", "boundary_mode": "truncate"}),
    ("normalized_full_support", {"normed": True, "support_mode": "full", "boundary_mode": "truncate"}),
    ("normalized_full_support_reflect", {"normed": True, "support_mode": "full", "boundary_mode": "reflect"}),
)


def build_params(*, x0: float, dur: int) -> dict:
    params = build_stable_default_params()
    params["edge_pop"]["offset"] = 0.0
    params["edge_pop"]["noise_scale_edge"] = 0.0
    params["edge_pop"]["c_EB"] = 0.0
    params["bump_pop"]["c_BE"] = 0.0
    params["bump_pop"]["noise_scale_bump"] = 0.0
    params["decision_space_params"]["dur"] = int(dur)
    params["decision_space_params"]["t_start"] = int(dur)
    params["decision_space_params"]["x0"] = float(x0)
    return params


def make_model(*, x0: float, dur: int) -> CANN_DDM_model:
    model = CANN_DDM_model(CANN_params=build_params(x0=x0, dur=dur))
    model.cue_R_all[:] = 0.0
    model.cue_L_all[:] = 0.0
    return model


def clamp_current(model: CANN_DDM_model) -> np.ndarray:
    return np.asarray(model.I_clamp_E.value, dtype=float)


def calibrate_legacy_scalars(model: CANN_DDM_model, base_matrix: np.ndarray) -> tuple[float, float]:
    center_idx = model.num_E // 2
    theta = np.linspace(float(model.theta_min), float(model.theta_max), model.num_E)
    real_r = canonical_edge_profile(model, 0.0)
    drive = base_matrix @ real_r
    th = theta[center_idx - 10:center_idx + 10]
    y = drive[center_idx - 10:center_idx + 10]
    slope, _ = np.polyfit(th, y, deg=1)
    gamma_effective = 4.0 * float(model.gamma_E) / np.e if model.edge_type == "tanh" else float(model.gamma_E)
    J0 = -gamma_effective / float(slope)
    beta = -J0 / 2.0
    return float(J0), float(beta)


def build_edge_operator_variant(model: CANN_DDM_model, variant_name: str) -> tuple[np.ndarray, float]:
    variant_lookup = dict(VARIANT_SPECS)
    if variant_name not in variant_lookup:
        raise ValueError(f"Unknown variant '{variant_name}'. Supported variants: {tuple(variant_lookup)}")

    spec = variant_lookup[variant_name]
    clamp_frac = float(model.edge_geometry.clamp_frac)
    if spec["support_mode"] == "default":
        kernel_len = int(model.num_E * (1.0 - clamp_frac))
    elif spec["support_mode"] == "full":
        kernel_len = int(model.num_E)
    else:
        raise ValueError(f"Unknown support_mode '{spec['support_mode']}'")
    if kernel_len % 2 == 0:
        kernel_len -= 1

    kernel = DoG_kernel(
        kernel_len,
        EDGE_KERNEL_BASE_EXC_SIGMA,
        EDGE_KERNEL_INHIBITION_WIDTH_RATIO,
        offset=0.0,
    )
    base_matrix = np.asarray(
        make_conn_mat_from_kernel(
            model.num_E,
            kernel,
            normed=bool(spec["normed"]),
            boundary_mode=spec["boundary_mode"],
        ),
        dtype=float,
    )
    j0, beta = calibrate_legacy_scalars(model, base_matrix)
    return j0 * base_matrix, beta


def static_readout_bias(model: CANN_DDM_model, *, num_positions: int) -> dict:
    geometry = model.edge_geometry
    theta_min = float(geometry.coding_theta_min + 0.15 * (geometry.coding_theta_max - geometry.coding_theta_min))
    theta_max = float(geometry.coding_theta_max - 0.15 * (geometry.coding_theta_max - geometry.coding_theta_min))
    positions = np.linspace(theta_min, theta_max, num_positions)
    recovered = []
    biases = []

    for pos in positions:
        profile = model.edge_states(model.num_E, model.gamma_E, geometry, model.edge_type, center_pos=float(pos))
        theta_hat = float(np.asarray(model.find_current_edge_location(profile)))
        recovered.append(theta_hat)
        biases.append(theta_hat - float(pos))

    bias_arr = np.asarray(biases, dtype=float)
    return {
        "positions": positions,
        "recovered_positions": np.asarray(recovered, dtype=float),
        "biases": bias_arr,
        "static_readout_bias_max_abs": float(np.max(np.abs(bias_arr))),
    }


def canonical_edge_profile(model: CANN_DDM_model, theta0: float) -> np.ndarray:
    profile = model.edge_states(
        model.num_E,
        model.gamma_E,
        model.edge_geometry,
        model.edge_type,
        center_pos=float(theta0),
    )
    return np.asarray(profile, dtype=float)


def fixed_point_response(model: CANN_DDM_model, profile: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    drive = np.asarray(model.J_EE, dtype=float) @ profile
    response = model.phi_E(bm.asarray(drive + clamp_current(model)))
    return np.asarray(response, dtype=float)


def goldstone_projection_curve(
    model: CANN_DDM_model,
    *,
    num_theta: int,
    theta_margin_frac: float,
) -> dict:
    geometry = model.edge_geometry
    theta_spacing = float((model.theta_max - model.theta_min) / (model.num_E - 1))
    coding_width = float(geometry.coding_theta_max - geometry.coding_theta_min)
    lower = float(geometry.coding_theta_min + theta_margin_frac * coding_width)
    upper = float(geometry.coding_theta_max - theta_margin_frac * coding_width)
    theta0_values = np.linspace(lower, upper, num_theta)

    residual_projections = []
    residual_norms = []
    for theta0 in theta0_values:
        profile = canonical_edge_profile(model, theta0)
        response = fixed_point_response(model, profile)
        residual = response - profile

        profile_plus = canonical_edge_profile(model, theta0 + theta_spacing)
        profile_minus = canonical_edge_profile(model, theta0 - theta_spacing)
        translation_mode = (profile_plus - profile_minus) / (2.0 * theta_spacing)

        denom = float(np.dot(translation_mode, translation_mode))
        if denom <= 1e-16:
            raise ValueError("translation-mode norm vanished while computing Goldstone projection")

        projection = float(np.dot(residual, translation_mode) / denom)
        residual_projections.append(projection)
        residual_norms.append(float(np.linalg.norm(residual)))

    projections = np.asarray(residual_projections, dtype=float)
    norms = np.asarray(residual_norms, dtype=float)
    sign_pattern = "".join(
        "+" if value > 0 else "-" if value < 0 else "0"
        for value in projections
    )
    return {
        "theta0_values": theta0_values,
        "goldstone_projection": projections,
        "residual_norm": norms,
        "max_abs_goldstone_projection": float(np.max(np.abs(projections))),
        "mean_abs_goldstone_projection": float(np.mean(np.abs(projections))),
        "projection_sign_pattern": sign_pattern,
        "theta_spacing": theta_spacing,
    }


def dynamic_drift(*, builder_mode: str, x0: float, dur: int) -> dict:
    model = make_model(x0=x0, dur=dur)
    J_EE, beta_E = build_edge_operator_variant(model, builder_mode)
    model.J_EE = bm.asarray(J_EE)
    model.beta_E = float(beta_E)
    runner = model.run_simulation(
        mon_vars=["theta_E", "x_E", "hit_boundary"],
        progress_bar=False,
        dt=1.0,
        get_RT=False,
    )
    theta_e = np.asarray(runner.mon.theta_E).reshape(-1)
    x_e = np.asarray(runner.mon.x_E).reshape(-1)
    hit = np.asarray(runner.mon.hit_boundary).reshape(-1)
    static_summary = static_readout_bias(model, num_positions=9)
    return {
        "builder_mode": builder_mode,
        "x0": float(x0),
        "dur": int(dur),
        "theta_E_init": float(theta_e[0]),
        "theta_E_final": float(theta_e[-1]),
        "theta_E_max_shift": float(np.max(np.abs(theta_e - theta_e[0]))),
        "x_E_init": float(x_e[0]),
        "x_E_final": float(x_e[-1]),
        "x_E_max_shift": float(np.max(np.abs(x_e - x_e[0]))),
        "hit_boundary": bool(np.any(hit)),
        "all_finite": bool(np.isfinite(theta_e).all() and np.isfinite(x_e).all()),
        **static_summary,
    }


def plot_goldstone_projection(
    summaries: dict[str, dict],
    *,
    save_path: str | None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {
        "normalized": "#1d52a1",
        "normalized_reflect": "#0e7c86",
        "unnormalized": "#b94e48",
        "normalized_full_support": "#2a7b4f",
        "normalized_full_support_reflect": "#6b5fc7",
    }
    labels = {
        "normalized": "normalized",
        "normalized_reflect": "normalized reflect",
        "unnormalized": "unnormalized",
        "normalized_full_support": "normalized full support",
        "normalized_full_support_reflect": "normalized full support reflect",
    }

    for builder_mode, summary in summaries.items():
        ax.plot(
            summary["theta0_values"],
            summary["goldstone_projection"],
            "o-",
            linewidth=2,
            markersize=4,
            color=colors.get(builder_mode, None),
            label=labels.get(builder_mode, builder_mode),
        )

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\langle R_{\theta_0}, g_{\theta_0}\rangle / \langle g_{\theta_0}, g_{\theta_0}\rangle$")
    ax.set_title(r"Goldstone-mode residual versus translated edge center $\theta_0$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved figure to: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dur", type=int, default=300, help="Zero-input simulation duration in ms.")
    parser.add_argument(
        "--x0-values",
        type=float,
        nargs="+",
        default=[0.2, 0.5, 0.8],
        help="Initial evidence values to test.",
    )
    parser.add_argument(
        "--num-theta",
        type=int,
        default=21,
        help="Number of translated edge centers used in the Goldstone projection curve.",
    )
    parser.add_argument(
        "--theta-margin-frac",
        type=float,
        default=0.15,
        help="Fractional margin removed from both ends of the coding region for theta_0 evaluation.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Optional output path for the residual-vs-theta_0 figure.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dur <= 1:
        raise ValueError("--dur must be greater than 1")
    if args.num_theta < 3:
        raise ValueError("--num-theta must be at least 3")
    if not (0.0 <= args.theta_margin_frac < 0.5):
        raise ValueError("--theta-margin-frac must lie in [0, 0.5)")

    residual_summaries: dict[str, dict] = {}
    for variant_name, _ in VARIANT_SPECS:
        model = make_model(x0=0.5, dur=args.dur)
        J_EE, beta_E = build_edge_operator_variant(model, variant_name)
        model.J_EE = bm.asarray(J_EE)
        if hasattr(model, "J_EE_full"):
            model.J_EE_full = bm.asarray(J_EE)
        model.beta_E = float(beta_E)
        residual_summary = goldstone_projection_curve(
            model,
            num_theta=args.num_theta,
            theta_margin_frac=args.theta_margin_frac,
        )
        residual_summaries[variant_name] = residual_summary

    print("Goldstone projection summary")
    for variant_name, summary in residual_summaries.items():
        print(f"\n[{variant_name}]")
        for key in (
            "max_abs_goldstone_projection",
            "mean_abs_goldstone_projection",
            "projection_sign_pattern",
        ):
            print(f"{key}: {summary[key]}")

    plot_goldstone_projection(residual_summaries, save_path=args.save_path)

    for x0 in args.x0_values:
        print(f"\n=== x0={x0:.3f} ===")
        for variant_name, _ in VARIANT_SPECS:
            summary = dynamic_drift(builder_mode=variant_name, x0=float(x0), dur=args.dur)
            print(f"\n[{variant_name}]")
            for key in (
                "theta_E_init",
                "theta_E_final",
                "theta_E_max_shift",
                "x_E_init",
                "x_E_final",
                "x_E_max_shift",
                "static_readout_bias_max_abs",
                "hit_boundary",
                "all_finite",
            ):
                print(f"{key}: {summary[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
