# Manuscript Sync

## Goal
Keep `docs/paper.pdf` in this repository synced from a separate LaTeX manuscript repository without relying on paid platform integrations.

## Recommended Source Layout
- Manuscript source repo: `/projectnb/ecog-eeg/cyw6/CANN_DDM_paper`
- Simulation repo: `/projectnb/ecog-eeg/cyw6/CANN_DDM_rate_model`

The manuscript repo should be the source of truth for LaTeX files. This repo should only mirror the compiled PDF at `docs/paper.pdf`.

## One-Time Setup
1. Create or clone the manuscript repo at `/projectnb/ecog-eeg/cyw6/CANN_DDM_paper`.
2. Make sure the manuscript can be built locally with `latexmk`.
3. If the root TeX file is not `main.tex`, note its filename for the sync command.

## Sync Command
Default usage:

```bash
./scripts/sync_paper.sh
```

If the manuscript root file is not `main.tex`, override it:

```bash
PAPER_MAIN_TEX=paper.tex ./scripts/sync_paper.sh
```

If the manuscript repo lives elsewhere, override the path:

```bash
PAPER_REPO=/path/to/CANN_DDM_paper ./scripts/sync_paper.sh
```

## What The Script Does
1. `git pull --ff-only` in the manuscript repo
2. build the PDF locally with `latexmk -pdf`
3. copy the built PDF into `docs/paper.pdf`

## Notes
- Do not edit `docs/paper.pdf` directly.
- Keep the LaTeX manuscript repo separate from this simulation repo; do not nest one Git repo inside the other.
- If you later move the manuscript source to another repo path, update the environment variable or script defaults.
