# TODO

This file records pending modeling tasks and reproducible next steps.

- Revisit `docs/THEORY_UNDERSTANDING.md` after the manuscript and simulation include an explicit circuit-level noise term.
- Revise `docs/CODE_UNDERSTANDING.md` whenever implementation-level understanding improves or when the user clarifies code-level intent.
- Create or clone the separate LaTeX manuscript repo at `/projectnb/ecog-eeg/cyw6/CANN_DDM_paper` and use `./scripts/sync_paper.sh` to refresh `docs/paper.pdf`.
- Keep `docs/paper.pdf` refreshed with `./scripts/sync_paper.sh` before manuscript-grounded reasoning tasks when the paper may have changed.
- Decide when to remove the remaining legacy config-compatibility inputs at the model boundary (`num_E`, `num_B`, legacy clamp-fraction path) now that the active workflow uses shared geometry.
- Revisit the exploratory smooth coupling kernels:
  - `W_EB`: `smooth_asymmetric`
  - `W_BE`: `smooth_symmetric`
  and determine whether they can be tuned into a stable regime that is both theory-faithful and consistent with the Figure 2 microdynamics.
- Decide whether the bump recurrent kernel (`J_BB`) should be tuned through the newly exposed kernel parameters or kept fixed while the cross-population coupling kernels are studied first.
- Review whether the fixed `J_EE` kernel construction constants in `rate_model_core/connectivity.py` should remain implementation constants or be promoted into explicit model/config parameters.
