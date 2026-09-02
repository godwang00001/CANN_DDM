# Progress

This file records completed modeling, validation, and figure-generation work.

## 2026-04-03

### Completed
- Added `scripts/submit_fig3_one_condition_scc.sh` as the new Figure 3 submission wrapper built directly on top of `scripts/submit_circuit_psychometric_scc.sh`.
- Fixed the intended Figure 3 combined-model run to the one-condition cDDM/circuit setting:
  - `coherence = 0.3`
  - `drift_gain = 1.0`
  - `noise_scale = 0.5`
  - `num_trials = 3000`
  - `save_traj = on` by default
- Standardized the intended Figure 3 data layout to the same psychometric pipeline structure already used for DDM-vs-circuit sweeps:
  - local DDM output under `.../ddm/`
  - SCC circuit output under `.../circuit/`
  - one merged top-level `dataset.npz`, `summary.csv`, and `config.json`
- Chose CPU-mode circuit workers for the new Figure 3 wrapper to stay consistent with `scripts/run_circuit_psychometric_one_coherence.sh`, which currently exports `JAX_PLATFORMS=cpu`.

### Validation
- `bash -n scripts/submit_fig3_one_condition_scc.sh` passed after the wrapper was added.

### Checkpoint
- The intended Figure 3 combined-model data source is now the one-condition psychometric-style bundle launched through `scripts/submit_fig3_one_condition_scc.sh`, not the older standalone Figure 3 circuit-only path.

## 2026-04-02

### Completed
- Added a dedicated Figure 3 continuous-DDM dataset generator, `scripts/generate_fig3_cddm_two_condition_dataset.py`, for a fixed two-condition macroscopic comparison panel.
- Added the matching SCC submit wrapper, `scripts/submit_fig3_cddm_two_condition_scc.sh`, so the Figure 3 DDM-only run can be launched as one SCC job rather than through the larger psychometric sweep workflow.
- Finalized the current Figure 3 DDM-only dataset under `results/figure3/fig3_cddm_two_condition_n3000/` with:
  - `slow`: `v=0.3`, `c=0.5`, `n=3000`
  - `fast`: `v=0.9`, `c=0.5`, `n=3000`
  - `decision_paradigm='free_response'`
  - `dt_ddm=5 ms`, `t_start=10 ms`, `dur=max_time=2000 ms`
- Kept trajectory saving enabled in the Figure 3 generator so the saved `dataset.npz` includes `x_traj` for example-trial visualization.
- Added `figures_code/main/fig3_ddm_rt_traj_two_condition.ipynb` as the current DDM-only Figure 3 notebook for the macroscopic panel.
- Updated that notebook so it now:
  - loads the Figure 3 two-condition dataset directly
  - plots the correct-trial RT histogram on top
  - plots exactly two example correct trajectories in the middle, one per condition
  - plots the error-trial RT histogram downward on the bottom
  - distinguishes the two conditions primarily by alpha rather than by unrelated colors
- Fixed the notebook-side task-output interpretation after confirming that the shared accumulator uses `choice=+1` for upper/right hits and `choice=-1` for lower/left hits.

### Validation
- `bash -n scripts/submit_fig3_cddm_two_condition_scc.sh` passed after the SCC-wrapper fixes.
- Executed the code cells from `figures_code/main/fig3_ddm_rt_traj_two_condition.ipynb` through a local JSON-driven validation pass with the project conda interpreter; the notebook completed successfully against the live `fig3_cddm_two_condition_n3000` dataset.
- The validated notebook run confirmed:
  - `choice.shape = (2, 3000)`
  - `x_traj.shape = (2, 3000, 2000)`
  - slow condition: `2706` hits = `2076` correct + `630` error + `294` miss
  - fast condition: `2971` hits = `2908` correct + `63` error + `29` miss

### Known Issues
- The Figure 3 generator still has a stale default `--run-name` string, `fig3_cddm_two_condition_n1000`, even though the current hard-coded condition block uses `3000` trials per condition and the active finished run is `fig3_cddm_two_condition_n3000`.
- The current Figure 3 notebook is DDM-only; the corresponding circuit-side macroscopic comparison panel still needs a follow-up implementation pass.

### Checkpoint
- The repo now has a dedicated DDM-only Figure 3 workflow: one fixed two-condition dataset generator, one SCC submit wrapper, one finished `n3000` run folder, and one notebook that renders the RT-plus-example-trajectory panel directly from that dataset.

## 2026-03-31

### Completed
- Split task semantics from evidence semantics in the decision module:
  - `decision_mode` now means evidence type (`continuous` or `discrete`)
  - `decision_paradigm` now means task rule (`free_response` or `interrogation`)
- Made `free_response` the default task paradigm across the shared DDM/circuit accumulator APIs.
- Split timing semantics in the active task API:
  - `dur` now means evidence duration
  - `max_time` now means total trial horizon
  - default behavior is `max_time = dur`
- Updated the circuit preprocessing path so decision-space arrays span `max_time` but become zero after `dur`, allowing post-stimulus evolution in free-response runs when `max_time > dur`.
- Updated the shared DDM simulators so free-response stops on first bound hit while interrogation runs through `max_time` and classifies from the terminal DV.
- Exposed `--decision-paradigm` and `--max-time` in the current psychometric data-generation scripts.
- Updated `docs/STATES.md` and `docs/CODE_UNDERSTANDING.md` to reflect the new task semantics and the current script names.

### Validation
- `python -m py_compile CANN_DDM_model_rate_based.py rate_model_core/config.py rate_model_core/default_params.py rate_model_core/accumulator_simulation.py rate_model_core/calibration.py scripts/run_discrete_click_mode_smoke.py scripts/simulate_psychometric_data_cDDM.py scripts/simulate_psychometric_data_pDDM.py scripts/generate_cddm_psychometric_dataset.py scripts/generate_pddm_psychometric_dataset.py scripts/plot_pddm_psychometric_by_net_clicks.py` passed after the task-semantics refactor.
- A direct Python check of `simulate_ddm_trials()` confirmed the new free-response and interrogation semantics, including `max_time > dur` behavior and `rt_ms = NaN` for interrogation.
- `env CONDA_PKGS_DIRS=/tmp/conda-pkgs CONDA_ENVS_PATH=/tmp/conda-envs conda run -n cann_ddm_v2 python scripts/run_discrete_click_mode_smoke.py` passed after adding the extended-horizon preprocessing check.

### Checkpoint
- The active decision module now explicitly supports both interrogation and free-response paradigms while keeping free-response as the default.
- In the current codebase, `dur` should be read as evidence duration and `max_time` as the hard trial cutoff.

## 2026-03-21

### Completed
- Unified the psychometric sweep workflow around one public SCC entrypoint, `scripts/submit_circuit_psychometric_scc.sh`.
- Updated that entrypoint so it now:
  - generates the DDM sweep locally with the same task parameters
  - submits one SCC worker job per circuit coherence
  - submits one dependent finalizer job
  - leaves one top-level `dataset.npz`, `summary.csv`, and `config.json`
- Added `scripts/generate_ddm_psychometric_dataset.py` as the local sweep generator for the pure DDM path.
- Added `scripts/combine_psychometric_model_datasets.py` so the finalizer can merge the completed DDM and circuit sweep outputs into one shared dataset bundle.
- Updated `scripts/finalize_psychometric_run.sh` so it now merges the circuit sweep first, then combines DDM and circuit outputs into the final top-level bundle.
- Fixed the SCC worker-launch path so the per-coherence circuit jobs no longer depend on the SGE spool copy for resolving repository paths.
- Forced the SCC circuit jobs onto CPU with `JAX_PLATFORMS=cpu`, avoiding the earlier node-dependent CUDA backend failures.
- Updated `figures_code/supp/ddm_circuit_psychometric_curve.ipynb` so it now loads one combined DDM+circuit dataset rather than two separate run folders.
- Fixed the notebook root-path logic so it searches upward for the real repo root instead of assuming `Path.cwd()` already points at it.

### Validation
- `bash -n scripts/submit_circuit_psychometric_scc.sh scripts/finalize_psychometric_run.sh scripts/run_circuit_psychometric_one_coherence.sh` passed after the unified workflow changes.
- `python -m py_compile scripts/generate_ddm_psychometric_dataset.py scripts/combine_psychometric_model_datasets.py scripts/merge_circuit_psychometric_outputs.py rate_model_core/accumulator_simulation.py rate_model_core/__init__.py` passed after the new helpers were added.
- `python scripts/generate_ddm_psychometric_dataset.py --run-name ddm_psychometric_n200` completed locally and produced a valid 9-condition, 200-trial DDM sweep bundle.
- A local smoke merge with `scripts/combine_psychometric_model_datasets.py` successfully combined the existing DDM and circuit sweep bundles into one dataset with:
  - `model_names = ['ddm', 'circuit']`
  - `choice.shape = (2, 9, 200)`
  - `rt_ms.shape = (2, 9, 200)`
- The live combined SCC run `results/psychometric/9_coh_200_trials_each/` completed and produced:
  - one `dataset.npz`
  - one `summary.csv`
  - one `config.json`
- Executed the code cells from `figures_code/supp/ddm_circuit_psychometric_curve.ipynb` through a local JSON-driven validation pass after the combined-dataset refactor; the notebook completed successfully and saved `figures/supp/ddm_circuit_psychometric_curve.png`.

### Known Issues
- The Codex nested-shell environment still cannot reliably invoke `qsub` through a child shell even though direct top-level `qsub` works; the unified submit script is intended to be launched from a normal SCC login shell.
- The current combined dataset format is notebook-facing and ad hoc; it is not yet exposed through a shared load helper in `rate_model_core/`.

### Checkpoint
- The active psychometric workflow now produces one combined DDM+circuit dataset bundle per run instead of separate top-level DDM and circuit result folders.
- The DDM-vs-circuit psychometric notebook now expects one combined run folder as its source of truth.

## 2026-03-20

### Completed
- Added `rate_model_core/accumulator_simulation.py` as the shared task-level accumulator helper for both pure DDM outputs and decoded circuit outputs.
- Kept `rate_model_core/ddm.py` as a compatibility wrapper so older imports still resolve.
- Implemented vectorized absorbed multi-trial simulation with:
  - shared scalar `drift_rate`
  - fixed `noise_scale`
  - integer-millisecond `dt_DDM`
  - optional dense trajectory return
  - RT measured relative to `t_start` in milliseconds
- Added NPZ save/load helpers for the shared accumulator result format.
- Added `scripts/simulate_circuit_condition.py` to simulate one calibrated circuit-model coherence condition and save a DDM-compatible core result archive.
- Folded the reusable circuit-condition simulation logic into `rate_model_core/accumulator_simulation.py` so both abstract DDM and decoded circuit simulations share one task-level API; kept `scripts/simulate_circuit_condition.py` as a thin CLI wrapper.
- Tightened the shared public API names to `simulate_ddm_trials()` and `simulate_circuit_trials()`, with both returning only `AccumulatorSimulationResult`; circuit calibration fields now live in result metadata.
- Added `scripts/simulate_psychometric_data.py` as the sweep-level data generator for psychometric figures; it saves one result archive per coherence condition plus a sweep summary/config for notebook import.
- Added a reusable `build_runner()` entrypoint in `CANN_DDM_model_rate_based.py` so circuit trials can be advanced in chunks without forcing one-shot full-duration runs.
- Updated the shared circuit simulator to reuse a provided calibration, stop trials early after the first boundary hit, and store absorbing post-hit task-level trajectories when trajectory saving is enabled.
- Updated the sweep script to reuse one circuit calibration across a sweep, add `--resume`, and avoid recalibration when only existing condition archives are being reused.
- Exposed the shared accumulator helpers through `rate_model_core/__init__.py`.
- Added `figures_code/supp/ddm_psychometric_curve.ipynb` to generate a pure-DDM psychometric curve from a coherence-to-drift sweep, fit a sigmoid, and save the resulting figure.
- Updated `docs/CODE_UNDERSTANDING.md` and `docs/STATES.md` to record the new pure-DDM entrypoints.

### Validation
- `python -m py_compile rate_model_core/accumulator_simulation.py rate_model_core/ddm.py rate_model_core/__init__.py scripts/simulate_circuit_condition.py` passed after the shared accumulator refactor and circuit script were added.
- Executed the code cells from `figures_code/supp/ddm_psychometric_curve.ipynb` in order through a local JSON-driven validation pass; the notebook completed successfully, fit the psychometric sigmoid, and saved `figures/supp/ddm_psychometric_curve.png`.
- Under the current notebook defaults (`noise_scale=0.3`, `drift_gain=1.5`, `dur=10000`, `num_trials=100`), the worst miss fraction across the coherence sweep stayed low (`≈ 0.0300`), and the fitted midpoint stayed near zero (`bias ≈ -0.00437`).
- Verified DDM NPZ save/load round-trip through the shared accumulator format: choices, hit flags, RTs, trajectories, and metadata all survived reload without shape or value drift.
- Ran a one-trial circuit smoke test through `scripts/simulate_circuit_condition.py` using the notebook-style task semantics (`coherence=0`, `drift_gain=1.5`, `noise_scale=0.3`, `dt_ddm=1 ms`, `dt_model=1 ms`, `t_start=10 ms`, `dur=10000 ms`, `seed=7`); the script completed, calibrated `kappa ≈ 0.0150`, saved an NPZ archive, and the saved result reloaded cleanly with `model_type='circuit'`.
- `python -m py_compile scripts/simulate_psychometric_data.py` passed after adding the sweep-level data generator.
- `python scripts/simulate_psychometric_data.py --model ddm ...` completed under the current notebook-style DDM defaults and wrote a nine-condition sweep with `max_miss_fraction = 0.03`.
- `conda run -n cann_ddm_v2 python -c "import runpy, sys; ... runpy.run_path('scripts/simulate_psychometric_data.py', run_name='__main__')"` completed for a one-condition circuit smoke sweep and wrote the same summary/config layout used by the DDM path.
- `python -m py_compile CANN_DDM_model_rate_based.py rate_model_core/accumulator_simulation.py scripts/simulate_psychometric_data.py rate_model_core/__init__.py rate_model_core/ddm.py` passed after the runtime-fix implementation.
- Fixed-seed comparisons against the pre-optimization saved circuit smoke results preserved `choice`, `hit_boundary`, and `rt_ms` exactly for both `dur=1000` and `dur=10000`; `final_x` now becomes explicitly absorbing on hit trials by design.
- On the `coherence=0`, `dt_ddm=5 ms`, `dt_model=1 ms`, `num_trials=10` circuit smoke benchmark:
  - old `dur=10000` path: `elapsed_s ≈ 136`
  - new early-stop path: `elapsed_s ≈ 70`
- On the already-computed one-condition circuit sweep, the new `--resume` path completed in `elapsed_s ≈ 0.267`, confirming that it now avoids both recalibration and recomputation when condition archives already exist.

### Known Issues
- The new pure-DDM notebook currently only generates the DDM curve; circuit-vs-DDM comparison still needs a follow-up implementation pass.
- The coherence-to-drift mapping remains an external figure-layer decision rather than a model-layer invariant.

### Checkpoint
- The repository now has a shared task-level accumulator result format used by both pure DDM simulations and saved circuit-condition runs, plus a script entrypoint for generating calibrated circuit-condition archives for later notebook analysis.

## 2026-03-19

### Completed
- Added an opt-in robust `W_EB` mode, `eb_kernel_mode='smoothed_derivative'`, in `rate_model_core/connectivity.py`.
- Implemented the new mode as a reflected-boundary derivative-of-Gaussian style operator with zero discrete sum and normalized first moment so it remains derivative-like on clean edges while suppressing short-wavelength jitter in `r_E`.
- Added `scripts/check_eb_kernel_robustness.py` to compare clean-edge fidelity and jitter rejection between the legacy `simple` operator and the new robust operator.
- Added an opt-in severe-instability fallback, `eb_kernel_mode='edge_readout_bump'`, in `CANN_DDM_model_rate_based.py`.
- Implemented `edge_readout_bump` as a readout-and-reconstruct path: infer `theta_E` from the current edge state, then generate a canonical bump centered at that `theta_E` and scaled to the clean-edge `simple` operator peak.
- Added `scripts/check_eb_instability_sample.py` to evaluate `simple`, `smoothed_derivative`, and `edge_readout_bump` directly on `figures_code/supp/r_E_instability.npy`.
- Updated `docs/CODE_UNDERSTANDING.md` and `docs/STATES.md` to record the new `W_EB` option and its intended use.

### Validation
- `python -m py_compile rate_model_core/connectivity.py scripts/check_eb_kernel_robustness.py` passed after the `W_EB` update.
- `python -m py_compile CANN_DDM_model_rate_based.py rate_model_core/config.py rate_model_core/default_params.py` still passed after wiring in the new mode.
- `conda run -n cann_ddm_v2 python -c "import runpy; runpy.run_path('scripts/check_eb_kernel_robustness.py', run_name='__main__')"` passed with the new `smoothed_derivative` operator.
- In a quick no-cue noisy comparison using the stable default parameter block with `noise_scale_edge=0.02`, the offline `W_EB r_E` readout under `smoothed_derivative` had lower mean spatial total variation and lower second-difference roughness than `simple` while keeping `hit_boundary=False` in both runs.
- On the saved failure sample `figures_code/supp/r_E_instability.npy`, `edge_readout_bump` produced a single-peaked smooth bump centered at the live edge readout (`theta_E ≈ 1.328`) with much lower roughness than `simple`:
  - `simple`: `tv ≈ 32.02`, `hf ≈ 38.96`, `local_peaks = 32`
  - `edge_readout_bump`: `tv ≈ 1.24`, `hf ≈ 0.12`, `local_peaks = 1`

### Known Issues
- `smoothed_derivative` is intentionally opt-in and is not yet the repository default; figure notebooks and calibration workflows still need explicit retuning if they want to use it.
- The current robustness check is static and operator-level; it does not yet certify every full closed-loop noisy regime.
- `edge_readout_bump` intentionally projects away edge-shape deviations, so it trades local profile fidelity for position-level robustness.

### Checkpoint
- The codebase now has a dedicated jitter-resistant `I_EB` path that preserves the legacy `simple` operator for exact backward compatibility.
- The codebase also now has a position-readout fallback for severe edge-profile instability where direct differentiation of `r_E` is no longer trustworthy.

## 2026-03-09

### Completed
- Read `docs/paper.pdf` from scratch to form an initial model-level understanding before relying on code.
- Held a theory clarification round with the user and corrected the understanding of neural-coordinate range, aligned bump-edge function, and the roles of $I_{BE}$ and $I_{EB}$.
- Wrote the first substantive version of `docs/THEORY_UNDERSTANDING.md`.

### Validation
- Cross-checked the current understanding against the March 9, 2026 manuscript revision in `docs/paper.pdf`.
- Confirmed the revised summary with the user before updating `docs/THEORY_UNDERSTANDING.md`.

### Known Issues
- The circuit-level placement of stochastic noise is still unresolved in the manuscript and simulation framework.

### Checkpoint
- `docs/THEORY_UNDERSTANDING.md` now reflects a user-confirmed baseline interpretation of the theory as of March 9, 2026.

## 2026-03-09

### Completed
- Added a free manuscript-sync workflow based on a separate local LaTeX repo rather than paid platform integrations.
- Added `scripts/sync_paper.sh` to pull, build, and mirror the manuscript PDF into `docs/paper.pdf`.
- Added `docs/MANUSCRIPT_SYNC.md` to document the expected source-repo layout and sync procedure.

### Validation
- Verified the sync workflow design against the current repository structure and kept it isolated from unrelated local code changes.

### Known Issues
- The external manuscript source repo `/projectnb/ecog-eeg/cyw6/CANN_DDM_paper` has not been created or cloned yet in this workspace.

### Checkpoint
- This repo now has a documented path for keeping `docs/paper.pdf` synced from a separate LaTeX source repository.

## 2026-03-09

### Completed
- Added explicit project rules for when `./scripts/sync_paper.sh` should and should not be run.
- Updated the manuscript-sync documentation and long-term state notes to reflect those trigger conditions.

### Validation
- Confirmed that manuscript sync should be tied to theory- and manuscript-grounded tasks rather than unrelated code-only work.

### Known Issues
- The mirrored `docs/paper.pdf` may still change outside these rule updates and should be committed separately when desired.

### Checkpoint
- The repo policy now states when manuscript refresh is expected before theory-grounded work.

## 2026-03-10

### Completed
- Added `docs/CODE_UNDERSTANDING.md` as a companion to `docs/THEORY_UNDERSTANDING.md`.
- Recorded a first-pass implementation-level understanding of how the edge-bump model is realized in BrainPy, including practical differences from the manuscript description.
- Updated the project policy so the code-understanding document is treated as a living checkpoint that should be revised when understanding improves.

### Validation
- Grounded the initial code-understanding summary in `CANN_DDM_model_rate_based.py` and `make_conn_mat_updated.py`.

### Known Issues
- Several implementation-versus-theory differences are intentionally left as open questions until the user confirms intended behavior.

### Checkpoint
- The repo now has separate persistent records for theory understanding and code understanding.

## 2026-03-11

### Completed
- Added `scripts/run_rate_model_smoke.py` as a deterministic no-cue, no-population-noise regression check for the refactor workflow.
- Reorganized `CANN_DDM_model_rate_based.py` without changing the public constructor or notebook-facing simulation entrypoints.
- Introduced a shared geometry/config architecture so the public setup path now centers on `geometry` with `num_units`, `coding_limit`, `coding_frac`, and `clamp_frac`.
- Moved pure config and geometry parsing into `rate_model_config.py`.
- Updated `make_conn_mat_updated.py` so its active interface consumes shared derived geometry instead of the old user-facing discretization arguments.
- Added `scripts/run_rate_model_geometry_regression.py` to compare the legacy configuration path against the new shared-geometry path under the fixed-seed cue-driven Figure 2 microdynamics condition.
- Migrated `figures_code/fig2_micro_dyn_scheme.ipynb` to the new top-level `geometry` and `num_units` setup.

### Validation
- Repeatedly ran `python -m py_compile CANN_DDM_model_rate_based.py make_conn_mat_updated.py rate_model_config.py scripts/run_rate_model_smoke.py scripts/run_rate_model_geometry_regression.py` after each major refactor batch.
- Verified that `conda run -n cann_ddm_v2 python scripts/run_rate_model_smoke.py` remained stable throughout the reorganization.
- Verified that `conda run -n cann_ddm_v2 python scripts/run_rate_model_geometry_regression.py` preserved identical cue-driven summaries between the legacy and new shared-geometry configuration paths.

### Known Issues
- `make_conn_mat_updated.py` still contains hard-coded kernel-shape constants in the edge connectivity builder, so the geometry/config cleanup has not yet exposed all recurrent-kernel assumptions as named model parameters.
- The constructor still keeps legacy compatibility for old `num_E`, `num_B`, and clamp-fraction inputs at the model boundary even though the active workflow now uses shared geometry.

### Checkpoint
- The active model and current Figure 2 notebook workflow now use a shared geometry/config layer with `num_units`.
- The no-cue smoke test and the cue-driven legacy-vs-geometry regression are the current guardrails for structure-preserving refactors.

## 2026-03-12

### Completed
- Split the old helper/config code into the `rate_model_core/` package:
  - `config.py`
  - `connectivity.py`
  - `math.py`
  - `utils.py`
- Removed the old root-level `make_conn_mat_updated.py` path in favor of `rate_model_core/connectivity.py`.
- Refactored the model so both coupling paths are now explicit kernel operators:
  - `I_EB = c_EB (W_EB r_E)`
  - `I_BE = c_BE (\mathrm{cue}_R W_BE r_B - \mathrm{cue}_L W_BE r_B)`
- Restored the old trusted edge-to-bump local operator as the `W_EB` safe baseline:
  - `eb_kernel_mode='simple'`
  - old three-point stencil with one-sided boundaries
  - old scale absorbed into `eb_kernel_gain=100.0`
- Added explicit bump-to-edge operator support with a matching safe baseline:
  - `be_kernel_mode='simple'`
  - identity operator with `be_kernel_gain=1.0`
- Kept exploratory smooth coupling-kernel options available:
  - `W_EB`: `smooth_asymmetric`
  - `W_BE`: `smooth_symmetric`
- Added study/scan scripts to explore live-bump coupling stability and parameter sensitivity:
  - `scripts/study_I_BE_live_bump.py`
  - `scripts/scan_live_bump_stability.py`

### Validation
- Repeatedly ran `python -m py_compile` after the refactor batches.
- Verified `conda run -n cann_ddm_v2 python scripts/run_rate_model_smoke.py` still passes after the explicit `W_EB` / `W_BE` conversion.
- Verified `conda run -n cann_ddm_v2 python scripts/run_rate_model_geometry_regression.py` still passes after the module split and coupling-operator refactor.
- Rechecked the trusted Figure 2-style microdynamics regime with the restored `simple` baseline and confirmed the expected current scale:
  - `max I_EB ≈ 0.1734`
  - `max I_BE ≈ 0.1715`

### Known Issues
- The smooth `W_EB` and `W_BE` options are exploratory and are not yet trusted replacements for the restored `simple` baseline.
- `parse_geometry_config()` still keeps a legacy compatibility path for old `num_E`, `num_B`, and clamp-fraction inputs.
- `J_EE` still uses fixed edge-kernel construction constants in `rate_model_core/connectivity.py`.

### Checkpoint
- The codebase now has a cleaner package layout centered on `rate_model_core/`.
- Both cross-population coupling paths are now expressed as explicit kernels, with a stable trusted baseline still available for Figure 2 work.

## 2026-03-18

### Completed
- Investigated the no-input edge-drift problem with explicit residual-field diagnostics instead of only trajectory inspection.
- Added `scripts/compare_edge_builder_drift.py` to compare edge-operator variants through Goldstone-mode residuals, static readout bias, and zero-input drift.
- Tested structural no-hidden-state fixes for the edge recurrent operator and found that a reflected-boundary DoG construction removes most of the drift without introducing hidden guard-state variables.
- Removed the temporary `edge_guard_frac` / hidden-full-edge implementation after confirming the reflected-boundary operator was the cleaner fix.
- Promoted the reflected-boundary `J_EE` construction to the runtime default in `rate_model_core/connectivity.py`.
- Saved the reflected-vs-truncated comparison figure to `figures/figure2/edge_operator_goldstone_projection_reflect.png`.

### Validation
- `python -m py_compile CANN_DDM_model_rate_based.py rate_model_core/config.py rate_model_core/connectivity.py scripts/compare_edge_builder_drift.py scripts/run_rate_model_geometry_regression.py` passed after the reflected-boundary runtime update.
- `conda run -n cann_ddm_v2 python scripts/run_rate_model_geometry_regression.py` still passed after making reflected boundaries the runtime default.
- On the zero-input edge-drift diagnostic at `dur=120`, the reflected boundary operator strongly outperformed the old truncated one:
  - truncated `normalized`: `max_abs_goldstone_projection ≈ 9.29e-3`, `mean_abs_goldstone_projection ≈ 3.64e-3`, `theta_E_max_shift ≈ 0.1833`
  - reflected `normalized_reflect`: `max_abs_goldstone_projection ≈ 8.08e-7`, `mean_abs_goldstone_projection ≈ 2.32e-7`, `theta_E_max_shift ≈ 0.00156`

### Known Issues
- The reflected-boundary runtime builder materially changes cue-driven trajectories relative to the old truncated builder, so figure-level outputs that depend on `J_EE` should be revalidated rather than assumed unchanged.
- The runtime smoke/geometry checks remain structural guardrails; they do not by themselves prove full theory-faithful behavior of the new edge operator across all parameter regimes.

### Checkpoint
- The active runtime edge builder now uses reflected boundaries instead of truncation.
- The repo no longer contains the experimental hidden-edge `edge_guard_frac` mechanism.
