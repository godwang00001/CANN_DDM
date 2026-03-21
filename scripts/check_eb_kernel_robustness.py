#!/usr/bin/env python3
"""Check clean-edge fidelity and jitter rejection for W_EB kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rate_model_core.config import build_geometry, parse_bump_config, parse_edge_config, parse_geometry_config
from rate_model_core.connectivity import make_edge_to_bump_conn_mat
from rate_model_core.default_params import build_stable_default_params
from rate_model_core.math import edge_states, theta_grid


def build_geometry_and_edge_config():
    params = build_stable_default_params()
    edge_config = parse_edge_config(params["edge_pop"])
    bump_config = parse_bump_config(params["bump_pop"])
    geometry_config = parse_geometry_config(params["geometry"], edge_config, bump_config)
    geometry = build_geometry(geometry_config)
    return geometry, edge_config


def response_center(theta: np.ndarray, response: np.ndarray) -> float:
    return float(theta[int(np.argmax(response))])


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if x_norm == 0.0 or y_norm == 0.0:
        return 0.0
    return float(np.dot(x, y) / (x_norm * y_norm))


def build_local_jitter(theta: np.ndarray, center: float, amplitude: float, envelope_sigma: float) -> np.ndarray:
    alternating = (-1.0) ** np.arange(theta.size)
    envelope = np.exp(-0.5 * ((theta - center) / envelope_sigma) ** 2)
    return amplitude * alternating * envelope


def evaluate_clean_edge_fidelity(theta: np.ndarray, centers: np.ndarray, clean_edge_fn, w_simple, w_robust):
    center_diffs = []
    clean_cosines = []
    robust_center_errors = []
    for center in centers:
        clean_edge = clean_edge_fn(center)
        simple_response = np.asarray(w_simple @ clean_edge)
        robust_response = np.asarray(w_robust @ clean_edge)
        center_diffs.append(
            abs(response_center(theta, robust_response) - response_center(theta, simple_response))
        )
        robust_center_errors.append(abs(response_center(theta, robust_response) - float(center)))
        clean_cosines.append(cosine_similarity(simple_response, robust_response))
    return {
        "max_center_diff_vs_simple": float(np.max(center_diffs)),
        "max_center_error_vs_truth": float(np.max(robust_center_errors)),
        "mean_cosine_vs_simple": float(np.mean(clean_cosines)),
        "min_cosine_vs_simple": float(np.min(clean_cosines)),
    }


def evaluate_jitter_rejection(
    theta: np.ndarray,
    centers: np.ndarray,
    clean_edge_fn,
    w_simple,
    w_robust,
    jitter_amp: float,
    envelope_sigma: float,
):
    simple_errors = []
    robust_errors = []
    for center in centers:
        clean_edge = clean_edge_fn(center)
        jitter = build_local_jitter(theta, center, amplitude=jitter_amp, envelope_sigma=envelope_sigma)
        jittered_edge = np.clip(clean_edge + jitter, 0.0, 1.0)
        simple_clean = np.asarray(w_simple @ clean_edge)
        robust_clean = np.asarray(w_robust @ clean_edge)
        simple_jittered = np.asarray(w_simple @ jittered_edge)
        robust_jittered = np.asarray(w_robust @ jittered_edge)
        simple_errors.append(float(np.linalg.norm(simple_jittered - simple_clean) / np.linalg.norm(simple_clean)))
        robust_errors.append(float(np.linalg.norm(robust_jittered - robust_clean) / np.linalg.norm(robust_clean)))
    return {
        "mean_simple_rel_error": float(np.mean(simple_errors)),
        "mean_robust_rel_error": float(np.mean(robust_errors)),
        "max_simple_rel_error": float(np.max(simple_errors)),
        "max_robust_rel_error": float(np.max(robust_errors)),
        "improvement_ratio": float(np.mean(simple_errors) / np.mean(robust_errors)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust-sigma", type=float, default=1.0)
    parser.add_argument("--robust-gain", type=float, default=100.0)
    parser.add_argument("--jitter-amp", type=float, default=0.02)
    parser.add_argument("--num-centers", type=int, default=5)
    parser.add_argument("--max-center-diff", type=float, default=0.03)
    parser.add_argument("--min-cosine", type=float, default=0.9)
    parser.add_argument("--min-improvement-ratio", type=float, default=1.5)
    args = parser.parse_args()

    geometry, edge_config = build_geometry_and_edge_config()
    theta = np.asarray(theta_grid(geometry.num_units, geometry))
    centers = np.linspace(
        float(geometry.coding_theta_min) * 0.75,
        float(geometry.coding_theta_max) * 0.75,
        args.num_centers,
    )
    clean_edge_fn = lambda center: np.asarray(
        edge_states(
            geometry.num_units,
            edge_config.gamma_E,
            geometry,
            edge_type=edge_config.edge_type,
            center_pos=float(center),
        )
    )

    w_simple = np.asarray(
        make_edge_to_bump_conn_mat(
            geometry.num_units,
            kernel_mode="simple",
            kernel_gain=edge_config.eb_kernel_gain,
        )
    )
    w_robust = np.asarray(
        make_edge_to_bump_conn_mat(
            geometry.num_units,
            kernel_mode="smoothed_derivative",
            kernel_sigma=args.robust_sigma,
            kernel_gain=args.robust_gain,
        )
    )

    clean_metrics = evaluate_clean_edge_fidelity(theta, centers, clean_edge_fn, w_simple, w_robust)
    jitter_metrics = evaluate_jitter_rejection(
        theta,
        centers,
        clean_edge_fn,
        w_simple,
        w_robust,
        jitter_amp=args.jitter_amp,
        envelope_sigma=max(2.0 * args.robust_sigma, 0.1),
    )

    print("Clean-edge fidelity")
    for key, value in clean_metrics.items():
        print(f"{key}: {value}")

    print("\nJitter rejection")
    for key, value in jitter_metrics.items():
        print(f"{key}: {value}")

    failures = []
    if clean_metrics["max_center_diff_vs_simple"] > args.max_center_diff:
        failures.append(
            f"robust center shift {clean_metrics['max_center_diff_vs_simple']:.6g} exceeds {args.max_center_diff:.6g}"
        )
    if clean_metrics["min_cosine_vs_simple"] < args.min_cosine:
        failures.append(
            f"clean-response cosine {clean_metrics['min_cosine_vs_simple']:.6g} is below {args.min_cosine:.6g}"
        )
    if jitter_metrics["improvement_ratio"] < args.min_improvement_ratio:
        failures.append(
            f"jitter improvement ratio {jitter_metrics['improvement_ratio']:.6g} is below {args.min_improvement_ratio:.6g}"
        )

    if failures:
        print("\nFAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
