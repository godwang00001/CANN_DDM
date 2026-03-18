# Progress

This file records completed modeling, validation, and figure-generation work.

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
