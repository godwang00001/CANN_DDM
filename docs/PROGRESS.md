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
