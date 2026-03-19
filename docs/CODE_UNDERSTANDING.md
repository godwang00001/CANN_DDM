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

- `k1`
- `k2`
- `theta_min`
- `theta_max`
- `coding_theta_min`
- `coding_theta_max`

`k1` and `k2` remain internal discretization markers for the coding-region boundaries. The canonical profile center is no longer stored as a geometry index; profile construction is now centered directly in theta space.

## Population Setup
### Edge Population
`_init_edge_pop()` reads `EdgePopConfig` and builds:

- `J_EE` with `make_edge_conn_mat()`
- `W_EB` with `make_edge_to_bump_conn_mat()`
- canonical edge state `r_E0`

The edge readout is:

- `theta_E`: inferred edge location
- `x_E = pos_to_evidence(theta_E, gamma_E)`

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
- `x_B = pos_to_evidence(theta_B, gamma_E)`

The active bump nonlinearity is still the normalized quadratic CANN form:

$$
\phi_B(x) = \frac{x^2}{\beta_B \rho_B (1 + \sum x^2)}
$$

## Connectivity Structure
### `J_EE`
`make_edge_conn_mat()` in `rate_model_core/connectivity.py` still builds the edge recurrent matrix from a DoG kernel and rescales it to match the desired local drive around the edge center.

The current runtime default now uses a reflected-boundary construction for the finite connectivity matrix rather than the older truncated one. Conceptually, missing kernel mass beyond the left/right boundary is folded back into the valid domain with an edge-inclusive mirror rule, so boundary rows no longer lose support simply because the domain is finite.

The current implementation still contains fixed edge-kernel construction constants:

- `EDGE_KERNEL_BASE_EXC_SIGMA = 1.0`
- `EDGE_KERNEL_INHIBITION_WIDTH_RATIO = 1.2`

These are implementation constants for constructing `J_EE`, not the public edge readout/profile parameter `gamma_E`.

My current understanding is that this reflected-boundary builder is now the active no-hidden-state fix for the edge-drift problem. An earlier experimental direction added hidden edge guard units and full-domain state variables, but that mechanism has been removed in favor of the simpler reflected operator.

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

Both `edge_states()` and `bump_states()` are now theta-space constructors with an optional `center_pos` argument. The model file keeps thin wrappers so the class API stays stable while the formula source of truth lives in the shared math module.

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

The current implementation readout map is now a finite-interval normalized exponential map over the coding window. In other words, the helper pair

- `pos_to_evidence()`
- `evidence_to_pos()`

is no longer the old unnormalized `x \propto e^{\gamma \theta}` convention. It now explicitly maps the coding-window endpoints onto the evidence-domain endpoints:

$$
x(\theta_{\min}) = 0,
\qquad
x(\theta_{\max}) = \text{boundary}.
$$

My current understanding is that this is an implementation correction for bounded finite simulations, not a statement that the live code now exactly matches the manuscript's analytic $x \leftrightarrow \theta$ convention.

## Decision Inputs And Simulation
`decision_space_params` still controls:

- cue timing
- reference drift-diffusion trajectory
- DDM-style cue-generation noise

The active decision-duration parameter is now a single

- `dur`

rather than the older split `dur1` / `dur2` convention. As the current code stands, only the total duration mattered, so the separate variables have been removed from the active config path.

Important current implementation fact:

- `cue_R_all` and `cue_L_all` drive the neural simulation
- `x_traj` is still only a reference trajectory
- the neural circuit is not directly forced to follow `x_traj`
- `run_simulation()` now initializes the bump and edge states directly at `theta^* = evidence_to_pos(x0, gamma_E)` and no longer uses external initialization inputs `I1` / `I2`

## Default Parameter Baseline
The repository now has a shared tested default-parameter helper:

- `rate_model_core.default_params.build_stable_default_params()`

My current understanding is that this should be treated as the canonical stable baseline unless an experiment notebook or script applies explicit local overrides. The supplement notebooks for `c_{BE}` analysis now derive their baseline config from this helper rather than duplicating the full parameter block inline.

## Regression Guardrails
The current structure-preserving guardrails are:

- `scripts/run_rate_model_smoke.py`
- `scripts/run_rate_model_geometry_regression.py`

The smoke test and geometry regression remain the main structure-preserving checks, but their old assumptions should now be interpreted carefully because the model initialization has changed: the circuit is now started directly at the `theta^*` implied by `x0` rather than being nudged there by temporary external inputs.

## Current Safe Baseline
The current safe baseline for the coupling operators is:

- `W_EB`: `simple`
- `W_BE`: `simple`

For the edge recurrent operator, the active runtime baseline is now the reflected-boundary `J_EE` builder rather than the old truncated one.

## Current Open Questions
The main code-level questions still open are:

- whether the remaining legacy geometry compatibility path should now be removed
- whether the smooth `W_EB` and `W_BE` options can be tuned into a stable and theory-faithful regime
- whether the fixed `J_EE` kernel constants and reflected-boundary rule should stay as implementation constants or be promoted into explicit parameters
- how much cue-driven and manuscript-facing behavior changes under the reflected-boundary runtime default relative to the old truncated builder
- how the normalized finite-interval implementation map should ultimately be reconciled with the manuscript's analytic $x(\theta)$ convention
- how the code-level noise terms should ultimately map onto the manuscript-level stochastic formulation
