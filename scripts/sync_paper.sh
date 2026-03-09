#!/usr/bin/env bash
set -euo pipefail

# Sync the latest compiled manuscript PDF from a sibling LaTeX repo into docs/paper.pdf.
# Defaults can be overridden with environment variables.
#
# Example:
#   PAPER_REPO=/projectnb/ecog-eeg/cyw6/CANN_DDM_paper \
#   PAPER_MAIN_TEX=paper.tex \
#   ./scripts/sync_paper.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PAPER_REPO="${PAPER_REPO:-/projectnb/ecog-eeg/cyw6/CANN_DDM_paper}"
PAPER_MAIN_TEX="${PAPER_MAIN_TEX:-main.tex}"
PAPER_OUTPUT_PDF="${PAPER_OUTPUT_PDF:-${PAPER_MAIN_TEX%.tex}.pdf}"
DEST_PDF="${DEST_PDF:-${REPO_ROOT}/docs/paper.pdf}"

if [[ ! -d "${PAPER_REPO}" ]]; then
  echo "Paper repo not found: ${PAPER_REPO}" >&2
  echo "Clone or create the LaTeX manuscript repo first, or override PAPER_REPO." >&2
  exit 1
fi

if [[ ! -f "${PAPER_REPO}/${PAPER_MAIN_TEX}" ]]; then
  echo "Main TeX file not found: ${PAPER_REPO}/${PAPER_MAIN_TEX}" >&2
  echo "Override PAPER_MAIN_TEX if the manuscript root file has a different name." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required but not available in PATH." >&2
  exit 1
fi

if ! command -v latexmk >/dev/null 2>&1; then
  echo "latexmk is required to build the manuscript PDF locally." >&2
  exit 1
fi

echo "[sync-paper] Pulling latest manuscript source from ${PAPER_REPO}"
git -C "${PAPER_REPO}" pull --ff-only

echo "[sync-paper] Building ${PAPER_MAIN_TEX}"
latexmk -pdf -cd "${PAPER_REPO}/${PAPER_MAIN_TEX}"

if [[ ! -f "${PAPER_REPO}/${PAPER_OUTPUT_PDF}" ]]; then
  echo "Expected output PDF not found: ${PAPER_REPO}/${PAPER_OUTPUT_PDF}" >&2
  echo "Override PAPER_OUTPUT_PDF if the build produces a different PDF name." >&2
  exit 1
fi

mkdir -p "$(dirname "${DEST_PDF}")"
cp "${PAPER_REPO}/${PAPER_OUTPUT_PDF}" "${DEST_PDF}"

echo "[sync-paper] Updated ${DEST_PDF}"
