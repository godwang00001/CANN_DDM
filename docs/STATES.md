# States

This file records validated long-term project state only.

- The active Git branch for this project is `CANN_DDM_rate_model`.
- `docs/paper.pdf` is currently the primary theory reference and should be read before making nontrivial model edits.
- `docs/paper.pdf` should be treated as a mirrored artifact from an external LaTeX manuscript repo, not as the editing source of truth.
- The default manuscript source repo path for local syncing is `/projectnb/ecog-eeg/cyw6/CANN_DDM_paper`.
- `./scripts/sync_paper.sh` should be run before theory- or manuscript-grounded tasks when the mirrored PDF may be stale.
- `docs/THEORY_UNDERSTANDING.md` is the working theory-checkpoint document for the agent's user-confirmed conceptual understanding.
- `docs/CODE_UNDERSTANDING.md` is the working implementation-checkpoint document for the agent's current understanding of how the model is realized in code.
- The current theory understanding includes the aligned edge-bump interpretation and explicitly tracks circuit-level noise as an open question.
- The active public configuration path now uses a shared top-level `geometry` block with `num_units`, `coding_limit`, `coding_frac`, and `clamp_frac`.
- The support modules now live under `rate_model_core/`:
  - `rate_model_core/config.py`
  - `rate_model_core/connectivity.py`
  - `rate_model_core/math.py`
  - `rate_model_core/utils.py`
- `CANN_DDM_model_rate_based.py` remains the main model/dynamics entrypoint used by the notebooks.
- The trusted current coupling baseline uses explicit kernel operators with:
  - `W_EB`: `eb_kernel_mode='simple'`
  - `W_BE`: `be_kernel_mode='simple'`
- `scripts/run_rate_model_smoke.py` is the current deterministic no-cue regression guard for structure-preserving refactors.
- `scripts/run_rate_model_geometry_regression.py` is the current fixed-seed cue-driven regression guard for checking that the new shared-geometry path reproduces the legacy configuration behavior.
- `scripts/study_I_BE_live_bump.py` and `scripts/scan_live_bump_stability.py` are the current exploratory study scripts for coupling-mechanism and stability work.
- `figures_code/fig2_micro_dyn_scheme.ipynb` has been migrated to the shared `geometry` / `num_units` setup.
