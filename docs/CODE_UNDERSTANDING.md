# Code Understanding

## Status
This file records the current implementation-level understanding of the rate-model code. It should track the live code, not the older pre-refactor structure.

## Main Implementation Files
The active simulation framework is now organized as:

- `CANN_DDM_model_rate_based.py`
- `rate_model_core/config.py`
- `rate_model_core/connectivity.py`
- `rate_model_core/math.py`
- `rate_model_core/utils.py`

The root model file owns model state, dynamics, and simulation entrypoints. The `rate_model_core/` package now owns pure configuration, connectivity, math-profile, and utility helpers.

## Public Setup Path
The public constructor still takes `CANN_params`, but the intended active setup path now uses a shared top-level `geometry` block:

- `num_units`
- `coding_limit`
- `coding_frac`
- `clamp_frac`

Internally this is converted into a shared derived geometry with:

- `k0`
- `k1`
- `k2`
- `theta_min`
- `theta_max`

`k0`, `k1`, and `k2` are now internal discretization variables, not user-facing configuration inputs.

## Population Setup
### Edge Population
`_init_edge_pop()` reads `EdgePopConfig` and builds:

- `J_EE` with `make_edge_conn_mat()`
- `W_EB` with `make_edge_to_bump_conn_mat()`
- canonical edge state `r_E0`

The edge readout is:

- `theta_E`: inferred edge location
- `x_E = pos_to_evidence(theta_E, sigma_E)`

The active edge nonlinearity is still:

$$
\phi_E(x) = \sigma(\alpha_E x + \beta_E)
$$

### Bump Population
`_init_bump_pop()` reads `BumpPopConfig` and builds:

- `J_BB` with `make_bump_conn_mat()`
- `W_BE` with `make_bump_to_edge_conn_mat()`
- canonical bump state `r_B0`

The bump readout is:

- `theta_B`: inferred bump location
- `x_B = pos_to_evidence(theta_B, sigma_E)`

The active bump nonlinearity is still the normalized quadratic CANN form:

$$
\phi_B(x) = \frac{x^2}{\beta_B \rho_B (1 + \sum x^2)}
$$

## Connectivity Structure
### `J_EE`
`make_edge_conn_mat()` in `rate_model_core/connectivity.py` still builds the edge recurrent matrix from a DoG kernel and rescales it to match the desired local drive around the edge center.

The current implementation still contains fixed edge-kernel construction constants:

- `EDGE_KERNEL_BASE_EXC_SIGMA = 1.0`
- `EDGE_KERNEL_INHIBITION_WIDTH_RATIO = 1.2`

These are implementation constants for constructing `J_EE`, not the public edge readout/profile parameter `sigma_E`.

### `J_BB`
`make_bump_conn_mat()` now takes explicit bump-kernel controls:

- `kernel_mode`
- `kernel_gain`
- `kernel_sigma`
- `kernel_normed`

The active default path is `kernel_mode='gaussian_cann'`, which preserves the original Gaussian bump-kernel behavior while making the recurrent kernel parameters explicit.

## Coupling Operators
### `W_EB`
`I_EB` is no longer built from `abs(diff(r_E))`. It is now defined as an explicit kernel operator:

$$
I_{EB} = c_{EB} \, (W_{EB} r_E)
$$

The active baseline is:

- `eb_kernel_mode='simple'`
- `eb_kernel_gain=100.0`

`simple` is the restored safe baseline that exactly reproduces the old local three-point operator:

- centered interior stencil
- one-sided boundary rows
- old scale absorbed into the kernel gain

There is also an exploratory smooth option:

- `eb_kernel_mode='smooth_asymmetric'`

which builds a short-range antisymmetric kernel with vanishing discrete zeroth moment.

### `W_BE`
`I_BE` is also now defined as an explicit kernel operator:

$$
I_{BE} = c_{BE}\left(\text{cue}_R\, W_{BE} r_B - \text{cue}_L\, W_{BE} r_B\right)
$$

The active baseline is:

- `be_kernel_mode='simple'`
- `be_kernel_gain=1.0`

`simple` is just the identity operator, which makes the baseline `I_BE` path equivalent to directly using the current live bump state `r_B`, but written as an explicit kernel operator for consistency with the theory notation.

There is also an exploratory smooth option:

- `be_kernel_mode='smooth_symmetric'`

which builds a short-range symmetric Gaussian kernel.

## Canonical Profile Helpers
Canonical population profiles now live in `rate_model_core/math.py`:

- `sigmoid()`
- `edge_states()`
- `bump_states()`
- `bump_states_at_idx()`

The model file keeps thin wrappers so the class API stays stable while the formula source of truth lives in the shared math module.

## Geometry / Readout Helpers
Pure geometry and utility helpers now live in `rate_model_core/utils.py`:

- `idx_to_pos()`
- `pos_to_idx()`
- `pos_to_evidence()`
- `evidence_to_pos()`
- `generate_cues_input()`
- `get_x_traj()`
- `get_RT()`

So the main model file is now more focused on state initialization, coupling, and runtime dynamics.

## Decision Inputs And Simulation
`decision_space_params` still controls:

- cue timing
- reference drift-diffusion trajectory
- DDM-style cue-generation noise

Important current implementation fact:

- `cue_R_all` and `cue_L_all` drive the neural simulation
- `x_traj` is still only a reference trajectory
- the neural circuit is not directly forced to follow `x_traj`

## Regression Guardrails
The current structure-preserving guardrails are:

- `scripts/run_rate_model_smoke.py`
- `scripts/run_rate_model_geometry_regression.py`

The smoke test checks a deterministic no-cue, no-population-noise stability condition. The geometry regression compares the legacy config path against the new shared-geometry path under a fixed-seed Figure 2 microdynamics condition.

## Current Safe Baseline
The current safe baseline for the coupling operators is:

- `W_EB`: `simple`
- `W_BE`: `simple`

This baseline is the one currently trusted for Figure 2-style microdynamics work. The exploratory smooth kernel modes are available, but they are not yet the trusted default.

## Current Open Questions
The main code-level questions still open are:

- whether the remaining legacy geometry compatibility path should now be removed
- whether the smooth `W_EB` and `W_BE` options can be tuned into a stable and theory-faithful regime
- whether the fixed `J_EE` kernel constants should stay as implementation constants or be promoted into explicit parameters
- how the code-level noise terms should ultimately map onto the manuscript-level stochastic formulation
