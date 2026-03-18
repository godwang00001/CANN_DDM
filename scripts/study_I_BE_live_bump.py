#!/usr/bin/env python3
"""Study the impact of using the live bump state directly in I_BE.

This script compares two model variants under the validated Figure 2
microdynamics regime:

1. Current behavior: construct a canonical bump profile aligned to the current
   edge location before feeding it into I_BE.
2. Proposed behavior: feed the live bump population state r_B directly into I_BE.

The goal is to quantify whether the two definitions remain close, and how much
the full trajectory changes if the live bump state is used directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import brainpy.math as bm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CANN_DDM_model_rate_based import CANN_DDM_model


class CanonicalShiftedBumpIBEModel(CANN_DDM_model):
    """Reference variant matching the original canonical shifted-bump I_BE."""

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
    """Variant that uses the current live bump state directly in I_BE."""

    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        return c_BE * (cue_R * r_B + cue_L * (-r_B))


class PeakNormalizedLiveBumpIBEModel(CANN_DDM_model):
    """Variant that rescales the live bump state to the canonical bump peak."""

    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        center = self.find_current_edge_location(self.r_E)
        canonical_bump = self.bump_states(
            self.num_B,
            self.sigma_B,
            self.bump_geometry,
            center_pos=center,
        )
        live_peak = bm.max(r_B)
        canonical_peak = bm.max(canonical_bump)
        scale = bm.where(live_peak > 0, canonical_peak / live_peak, 0.0)
        normalized_r_B = scale * r_B
        return c_BE * (cue_R * normalized_r_B + cue_L * (-normalized_r_B))


class AlignedPeakNormalizedLiveBumpIBEModel(CANN_DDM_model):
    """Variant that aligns live r_B to the reference center and matches canonical peak."""

    def get_current_I_BE(self, cue_R, cue_L, r_B, c_BE):
        edge_center_pos = self.find_current_edge_location(self.r_E)
        edge_center = self.pos_to_idx(edge_center_pos)
        bump_center = self.pos_to_idx(self.find_current_bump_location(r_B))
        shift = edge_center - bump_center
        aligned_r_B = bm.roll(r_B, shift)

        canonical_bump = self.bump_states(
            self.num_B,
            self.sigma_B,
            self.bump_geometry,
            center_pos=edge_center_pos,
        )
        live_peak = bm.max(aligned_r_B)
        canonical_peak = bm.max(canonical_bump)
        scale = bm.where(live_peak > 0, canonical_peak / live_peak, 0.0)
        normalized_aligned_r_B = scale * aligned_r_B
        return c_BE * (cue_R * normalized_aligned_r_B + cue_L * (-normalized_aligned_r_B))


def build_geometry_params() -> dict:
    edge_pop = {
        "tau_E": 2,
        "c_EB": 0.3,
        "alpha_E": 1,
        "gamma_E": 10,
        "edge_type": "tanh",
        "offset": 0.0,
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
        "noise_scale": 0.1,
        "dt_DDM": 1.0,
        "x0": 0.5,
        "dur1": 100,
        "dur2": 1000,
        "seed": 4,
    }
    return {
        "edge_pop": edge_pop,
        "bump_pop": bump_pop,
        "decision_space_params": decision_space_params,
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


def canonical_bump_from_edge(model: CANN_DDM_model, r_E: np.ndarray) -> np.ndarray:
    center = model.find_current_edge_location(r_E)
    return np.asarray(
        model.bump_states(model.num_B, model.sigma_B, model.bump_geometry, center_pos=center)
    )


def summarize_variant(model_cls: type[CANN_DDM_model], label: str) -> dict:
    model = model_cls(CANN_params=build_geometry_params())
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

    cosine_sims = []
    l2_norm_ratios = []
    for edge_state, bump_state in zip(r_e, r_b):
        canonical_bump = canonical_bump_from_edge(model, edge_state)
        denom = np.linalg.norm(canonical_bump) * np.linalg.norm(bump_state)
        cosine = 1.0 if denom == 0 else float(np.dot(canonical_bump, bump_state) / denom)
        cosine_sims.append(cosine)
        l2_norm_ratios.append(float(np.linalg.norm(bump_state - canonical_bump) / np.linalg.norm(canonical_bump)))

    return {
        "label": label,
        "theta_E_final": float(theta_e[-1]),
        "theta_B_final": float(theta_b[-1]),
        "x_E_final": float(x_e[-1]),
        "x_B_final": float(x_b[-1]),
        "I_BE_max_abs": float(np.max(np.abs(i_be))),
        "I_EB_max_abs": float(np.max(np.abs(i_eb))),
        "hit_any": bool(np.any(hit)),
        "mean_bump_cosine_vs_canonical": float(np.mean(cosine_sims)),
        "min_bump_cosine_vs_canonical": float(np.min(cosine_sims)),
        "mean_rel_l2_live_vs_canonical": float(np.mean(l2_norm_ratios)),
        "max_rel_l2_live_vs_canonical": float(np.max(l2_norm_ratios)),
    }


def summarize_difference(reference: dict, candidate: dict) -> dict:
    diffs = {}
    for key, ref_value in reference.items():
        if key == "label":
            continue
        cand_value = candidate[key]
        if isinstance(ref_value, bool):
            diffs[key] = cand_value != ref_value
        else:
            diffs[key] = float(cand_value - ref_value)
    return diffs


def print_summary(summary: dict) -> None:
    print(summary["label"])
    for key, value in summary.items():
        if key == "label":
            continue
        print(f"{key}: {value}")


def main() -> int:
    current = summarize_variant(
        CanonicalShiftedBumpIBEModel,
        "Reference I_BE: canonical bump aligned to edge",
    )
    live = summarize_variant(LiveBumpIBEModel, "Proposed I_BE: live bump state")
    peak_normalized = summarize_variant(
        PeakNormalizedLiveBumpIBEModel,
        "Proposed I_BE: live bump state normalized to canonical peak",
    )
    aligned_peak_normalized = summarize_variant(
        AlignedPeakNormalizedLiveBumpIBEModel,
        "Proposed I_BE: live bump aligned to edge center and normalized to canonical peak",
    )
    live_diffs = summarize_difference(current, live)
    peak_normalized_diffs = summarize_difference(current, peak_normalized)
    aligned_peak_normalized_diffs = summarize_difference(current, aligned_peak_normalized)

    print_summary(current)
    print()
    print_summary(live)
    print()
    print_summary(peak_normalized)
    print()
    print_summary(aligned_peak_normalized)

    print("\nDelta live - current")
    for key, value in live_diffs.items():
        print(f"{key}: {value}")

    print("\nDelta peak-normalized live - current")
    for key, value in peak_normalized_diffs.items():
        print(f"{key}: {value}")

    print("\nDelta aligned+peak-normalized live - current")
    for key, value in aligned_peak_normalized_diffs.items():
        print(f"{key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
