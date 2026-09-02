"""Canonical tested parameter presets for the CANN/DDM rate model."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


def build_stable_default_params() -> dict[str, Any]:
    """Return the tested stable default parameter block.

    This mirrors the validated baseline currently recorded in
    `figures_code/main/fig2_micro_dyn_scheme.ipynb` before later experiment
    cells apply cue-specific or coupling-specific overrides. The intent is
    broader than that single figure: this is the repository's tested stable
    default configuration unless the user specifies otherwise.
    """

    geometry = {
        "num_units": 1024,
        "coding_limit": float(np.pi / 2),
        "coding_frac": 0.2,
        "clamp_frac": 0.1,
    }
    edge_pop = {
        "tau_E": 1,
        "alpha_E": 1.0,
        "gamma_E": 1.1,
        "edge_type": "tanh",
        "offset": 0.0,
        'c_EB': 0.1,
        "optimize_offset": False,
    }
    bump_pop = {
        "tau_B": 0.15,
        "beta_B": 6.0,
        "c_BE_params": {"mode": "const"},
        "sigma_B": 0.2,
    }
    decision_space_params = {
        "decision_mode": "continuous",
        "decision_paradigm": "free_response",
        "t_start": 10,
        "boundary": 1.0,
        "drift_rate": 1.0,
        "noise_scale": 0.1,
        "dt_DDM": 1.0,
        "x0": 0.5,
        "dur": 1000,
        "seed": 4,
        "mar": 0.01
    }
    return deepcopy(
        {
            "geometry": geometry,
            "edge_pop": edge_pop,
            "bump_pop": bump_pop,
            "decision_space_params": decision_space_params,
        }
    )
