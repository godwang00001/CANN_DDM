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

There is now also a robust filtered-derivative option:

- `eb_kernel_mode='smoothed_derivative'`

which builds an antisymmetric derivative-of-Gaussian style kernel with:

- exact vanishing discrete zeroth moment
- normalized first moment
- reflected-boundary matrix construction

My current understanding is that this mode is the intended jitter-resistant `I_EB` path: it preserves the edge-centered bump readout for clean translated edges more faithfully than an arbitrary smooth kernel, while attenuating short-wavelength noise in `r_E` before it reaches the bump population.

There is also now a readout-and-reconstruct option:

- `eb_kernel_mode='edge_readout_bump'`

This mode does not differentiate the live edge profile at all. Instead it:

- reads out the current edge position `theta_E` from `r_E`
- constructs a canonical bump centered at that `theta_E`
- scales the bump to match the current clean-edge `simple`-operator peak

My current understanding is that this is the most robust `I_EB` option when the live edge profile itself becomes locally unstable, because it projects the noisy edge activity onto the low-dimensional edge-position manifold before generating the bump-shaped drive.

### `W_BE`
`I_BE` is also now defined as an explicit kernel operator:

$$
I_{BE} = c_{BE}(\theta_E)\, v_{\text{drive}}(t)\, (W_{BE} r_B)
$$

with the current runtime convention:

- in `decision_mode='discrete'`, `v_drive = v_drift + v_noise`, where `v_drift` is built from net click count over each `dt_DDM` window
- in `decision_mode='continuous'`, `v_drive = v_drift + v_noise`

The active baseline is:

- `be_kernel_mode='simple'`
- `be_kernel_gain=1.0`

`simple` is just the identity operator, which makes the baseline `I_BE` path equivalent to directly using the current live bump state `r_B`, but written as an explicit kernel operator for consistency with the theory notation.

There is also an exploratory smooth option:

- `be_kernel_mode='smooth_symmetric'`

which builds a short-range symmetric Gaussian kernel.

On top of the old analytic `c_BE_params` modes, the model now also supports a calibration-backed runtime mode:

- `c_BE_params['mode'] = 'target_diffusion'`

This mode depends on a numerical calibration of the current parameter configuration and then constructs a geometry-corrected `c_BE(\theta)` profile for the continuous DDM input path.

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
- `generate_click_inputs()`
- `build_discrete_click_drift()`
- `get_x_traj()`
- `get_x_traj_discrete()`
- `get_RT()`

So the main model file is now more focused on state initialization, coupling, and runtime dynamics.

The model file now also exposes a reusable runner builder:

- `build_runner()`

My current understanding is that this exists so task-level code can drive the same circuit trial in chunks, inspect `hit_boundary`, and stop early without rebuilding the full BrainPy runner every time.

There is now also a shared task-level accumulator helper in:

- `rate_model_core/accumulator_simulation.py`

My current understanding is that this module is meant to keep task-level Monte Carlo simulations and shared result serialization separate from the neural circuit implementation in `CANN_DDM_model_rate_based.py`. It uses the same continuous-task timing convention as the model:

- integer-millisecond `dt_DDM`
- pre-stimulus hold at `x0` until `t_start`
- `dur` as evidence duration
- `max_time` as total trial horizon

The task semantics are now explicit:

- `decision_mode` chooses the evidence process:
  - `continuous`: drift + diffusion
  - `discrete`: click / pulse evidence with optional noise
- `decision_paradigm` chooses the task rule:
  - `free_response`: stop on first bound hit, or timeout at `max_time`
  - `interrogation`: run through `max_time` and classify by terminal DV

Within this setup:

- free-response returns RT measured relative to `t_start`
- interrogation sets `rt_ms = NaN`
- by default, `decision_paradigm='free_response'` and `max_time = dur`
- if `max_time > dur`, evidence turns off after `dur` but the state can keep evolving until `max_time`
- task-level `choice` is encoded as:
  - `+1` for upper / right bound classification
  - `-1` for lower / left bound classification

The new helper is the clean path for pure DDM psychometric and RT analyses, and it now also defines the shared result shape and reusable circuit-condition simulation path used before those task-level statistics are compared to the full circuit. The intended public entrypoints are now:

- `simulate_ddm_trials()`
- `simulate_circuit_trials()`

There is now also a dedicated Figure 3 DDM-only workflow on top of this shared accumulator layer:

- `scripts/generate_fig3_cddm_two_condition_dataset.py`
- `scripts/submit_fig3_cddm_two_condition_scc.sh`
- `figures_code/main/fig3_ddm_rt_traj_two_condition.ipynb`

My current understanding is that this path is intentionally figure-specific rather than a general psychometric sweep API: it generates one fixed two-condition dataset with saved trajectories so the notebook can render the RT-distribution-plus-example-trajectory panel directly.

There is now also a Figure 3 one-condition wrapper built on top of the unified psychometric workflow:

- `scripts/submit_fig3_one_condition_scc.sh`

My current understanding is that this is the intended combined-model Figure 3 data path going forward. It does not define a new dataset format; it simply fixes the psychometric workflow to one Figure 3 condition:

- `coherence = 0.3`
- `drift_gain = 1.0`
- `noise_scale = 0.5`
- `num_trials = 3000`
- `save_traj = on`

so the final output remains one merged DDM+circuit bundle with the same `ddm/`, `circuit/`, and top-level combined layout used by the broader psychometric pipeline.

Both return the same `AccumulatorSimulationResult` object. Calibration details needed for the circuit path are stored in `result.metadata` rather than returned as a second object.

The circuit path now has two important runtime optimizations:

- optional sweep-level calibration reuse through a provided calibration object
- chunked early-stop execution for circuit trials

The free-response early-stop path is task-level, not model-level: the circuit still evolves exactly as before until the first detected boundary hit, but the shared simulator stops advancing the runner after that point and treats the saved task-level trajectory as absorbing for storage purposes. Under interrogation, the same runner continues to `max_time` and the saved choice is derived from the terminal decoded DV instead of the first hit.

At the script layer, psychometric figure data generation is now separated from notebook plotting through:

- `scripts/simulate_psychometric_data_cDDM.py`
- `scripts/simulate_psychometric_data_pDDM.py`

My current understanding is that these scripts are the intended condition-level batch paths for notebook-facing psychometric generation. They drive the shared DDM/circuit simulators under the explicit `decision_mode` + `decision_paradigm` API and write NPZ results plus summary/config sidecars.

There is now also a higher-level unified psychometric workflow centered on:

- `scripts/submit_circuit_psychometric_scc.sh`

My current understanding is that this shell entrypoint is now the intended public path when a matched DDM-vs-circuit psychometric dataset is needed under one shared task configuration. The workflow is:

1. generate the DDM sweep locally with `scripts/generate_cddm_psychometric_dataset.py` or `scripts/generate_pddm_psychometric_dataset.py`, depending on the evidence task
2. submit one SCC worker job per circuit coherence
3. submit a dependent finalizer job
4. merge the circuit worker outputs
5. combine the final DDM and circuit sweep outputs into one top-level bundle

The local DDM sweep generators are now:

- `scripts/generate_cddm_psychometric_dataset.py`
- `scripts/generate_pddm_psychometric_dataset.py`

The DDM local generator now writes one sweep-level dataset directly:

- `dataset.npz`
- `summary.csv`
- `config.json`

and uses the same parameter names and semantics as the circuit sweep:

- `coherence_values`
- `drift_gain`
- `noise_scale`
- `dt_ddm`
- `dt_model`
- `t_start`
- `dur`
- `max_time`
- `decision_paradigm`
- `x0`
- `boundary`
- `num_trials`
- `seed`

The current combined-model bundler is:

- `scripts/combine_psychometric_model_datasets.py`

My current understanding is that this script is intentionally result-preserving: it does not resimulate anything. It reads the already-generated DDM sweep bundle and the already-finalized circuit sweep bundle, then writes one shared top-level bundle containing both models.

The current combined dataset format is a plain NPZ archive with:

- `model_names`
- `coherence_values`
- `choice`
- `hit_boundary`
- `rt_ms`
- `final_x`
- `time_ms`
- `metadata_json`

where the main task-level arrays are currently shaped as:

- `choice.shape = (num_models, num_coherences, num_trials)`
- `hit_boundary.shape = (num_models, num_coherences, num_trials)`
- `rt_ms.shape = (num_models, num_coherences, num_trials)`

and the shared top-level `summary.csv` contains one row per `(model, coherence)` pair rather than one row per coherence only.

The current notebook-facing model order in the combined dataset is:

- `model_names = ['ddm', 'circuit']`

The current comparison notebook:

- `figures_code/supp/ddm_circuit_psychometric_curve.ipynb`

no longer loads two separate run folders. It now reads one combined run folder, splits the shared bundle by `model_names`, reconstructs the per-model psychometric summaries inside the notebook, and then fits/plots the two curves on shared axes.

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
`decision_space_params` now controls:

- `decision_mode` (`continuous` or `discrete`)
- `decision_paradigm` (`free_response` or `interrogation`)
- cue timing / task horizon
- reference drift-diffusion trajectory
- continuous DDM drift and diffusion parameters

The active timing parameters are now:

- `t_start`
- `dur`
- `max_time`

with the current semantics:

- `dur`: duration of nonzero external evidence
- `max_time`: total trial horizon
- default: `max_time = dur`

Important current implementation fact:

- `x_traj` is still only a reference trajectory
- the neural circuit is not directly forced to follow `x_traj`
- `run_simulation()` now initializes the bump and edge states directly at `theta^* = evidence_to_pos(x0, gamma_E)` and no longer uses external initialization inputs `I1` / `I2`
- in `decision_mode='continuous'`, the model builds `v_drift_all`, `v_noise_all`, and `v_drive_all`
- in `decision_mode='discrete'`, the model builds click streams plus a windowed `v_drift_all`
- these decision-space arrays now span `max_time`, but are zero-padded after `dur`

The current time-step split is:

- `decision_space_params['dt_DDM']`: decision/noise update interval used to generate `dW` and the reference DDM path
- `run_simulation(dt=...)`: neural simulation / monitor sampling interval

These two time scales are independent in the current implementation. In particular, the Brownian supplement notebook now uses separate notebook variables for them (`DT_DDM` and `DT_MODEL`) to avoid shape mismatches when the DDM update interval is coarser than the neural simulation step.

For the shared stable default baseline, `x0` is now set to `0.5`, so the model starts from the interior of the coding range unless a notebook or script overrides it.

## Default Parameter Baseline
The repository now has a shared tested default-parameter helper:

- `rate_model_core.default_params.build_stable_default_params()`

My current understanding is that this should be treated as the canonical stable baseline unless an experiment notebook or script applies explicit local overrides. The supplement notebooks for `c_{BE}` analysis now derive their baseline config from this helper rather than duplicating the full parameter block inline.

## `target_diffusion` Calibration Path
The current code now has a model-integrated path for constructing a position-dependent `c_{BE}(\theta)` for the continuous DDM input:

- helper: `rate_model_core.calibration.calibrate_target_diffusion_profile()`
- model method: `prepare_target_diffusion_mode()`

The intended flow is:

1. set `c_BE_params['mode'] = 'target_diffusion'`
2. optionally provide calibration overrides such as `theta_margin` or `c_be_sweep`
3. call `prepare_target_diffusion_mode()` explicitly
4. run the simulation in `decision_mode='continuous'`

The helper calibrates a local gain `\kappa` from a constant-`c_{BE}` sweep, constructs the geometry-corrected `c_{BE}(\theta)` profile, and checks whether the implied effective coupling remains inside the numerically trusted regime.

Important implementation detail:

- if the first sweep is too small to cover the implied effective coupling range, the helper automatically expands the sweep, refits `\kappa`, and checks that the relative change in `\kappa` remains below tolerance

The returned calibration result is intentionally minimal:

- `kappa`
- `c_be_theta_max`
- `effective_c_be_max`
- `valid_c_be_max`
- `kappa_rel_error`
- `certificate_passed`

My current understanding is that this is an implementation-level certification of the `target_diffusion` construction, not a proof that the full nonlinear task dynamics are always safe.

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
The new `W_EB='smoothed_derivative'` path is available as an opt-in robust alternative, but the repo still keeps `simple` as the default for exact legacy reproducibility.
For severe local edge instability, the new `W_EB='edge_readout_bump'` mode is the strongest stabilizing option because it bypasses direct differentiation of the noisy profile.

## Current Open Questions
The main code-level questions still open are:

- whether the remaining legacy geometry compatibility path should now be removed
- whether the smooth `W_EB` and `W_BE` options can be tuned into a stable and theory-faithful regime
- what the best default `eb_kernel_sigma` / `eb_kernel_gain` pair is for the new `smoothed_derivative` mode across the main figure regimes
- whether `edge_readout_bump` should remain an opt-in fallback or become the recommended default in high-noise continuous-DDM regimes
- whether the fixed `J_EE` kernel constants and reflected-boundary rule should stay as implementation constants or be promoted into explicit parameters
- how much cue-driven and manuscript-facing behavior changes under the reflected-boundary runtime default relative to the old truncated builder
- how the normalized finite-interval implementation map should ultimately be reconciled with the manuscript's analytic $x(\theta)$ convention
- how robust the calibrated `target_diffusion` / `c_{BE}(\theta)` construction remains once the task leaves the weak-input aligned regime
- how the code-level noise terms should ultimately map onto the manuscript-level stochastic formulation
