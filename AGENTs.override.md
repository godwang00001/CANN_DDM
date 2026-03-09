# Repository Guidelines

## Project Scope
This repository contains simulation code for a theoretical neuroscience paper on decision-making dynamics. The model is intended to explain electrophysiological observations from decision-making tasks. Treat scientific correctness, reproducibility, and interpretability as higher priority than broad refactoring.

This is not a greenfield AI-generated project. The theory, model design, and core simulation framework already exist and have been verified by the user. The default task is therefore to modify, update, and polish the current repository without casually reinterpreting or replacing its scientific logic.

## Project Structure & Module Organization
Primary model code lives in `CANN_DDM_model_rate_based.py`. This is the main implementation of the coupled CANN/DDM rate model and should be treated as the authoritative simulation entry point unless the user says otherwise. Connectivity construction helpers live in `make_conn_mat_updated.py`; `make_conn_mat.py` should be treated as legacy or comparison code unless the user explicitly asks to modify it.

Notebook-based figure and analysis workflows live in `figures_code/`. Generated figure assets live in `figures/figure1/` and `figures/figure2/`. The notebook `mexican_hat_demo.ipynb` is exploratory/reference material and should be edited cautiously. Keep cache files such as `__pycache__/` out of commits.

Use `docs/` for project coordination, validated modeling state, and experiment tracking.

Recommended `docs/` layout:
- `docs/PROGRESS.md`: append-only record of completed model changes, parameter sweeps, figure regeneration, validation runs, and result-impacting observations.
- `docs/TODO.md`: current pending modeling tasks, unresolved numerical issues, figure updates, and reproducible next steps. Prune completed items after recording outcomes in `PROGRESS.md`.
- `docs/STATES.md`: bounded summary of validated long-term state only, such as authoritative files, active branch, trusted workflows, and currently accepted figure-generation paths.
- `docs/EXPERIMENTS.md`: simulation experiment logs, parameter comparisons, coupling variants, and interpretation notes that should not be mixed directly into source files.
- `docs/THEORY_UNDERSTANDING.md`: a Markdown document written from scratch that records the agent's current understanding of the user's theory and simulation framework. This file is not code documentation; it is a theory-understanding checkpoint for the user to review and correct.

## Environment & Core Framework
The conda environment for running project code is `CANN_DDM_V2`. Prefer reproducible commands in the form `conda run -n CANN_DDM_V2 ...` when executing scripts or checks.

This project is built around `brainpy`, a framework for computational neuroscience and brain-inspired computation with JIT-compiled numerical workflows. Treat BrainPy state variables, update rules, integrators, and simulation semantics as scientifically sensitive components.

## Build, Test, and Development Commands
This repo is script- and notebook-driven rather than package-driven. Typical commands:
- `conda run -n CANN_DDM_V2 python -m py_compile CANN_DDM_model_rate_based.py make_conn_mat_updated.py`
- `conda run -n CANN_DDM_V2 python -i CANN_DDM_model_rate_based.py`
- `conda run -n CANN_DDM_V2 jupyter notebook`
- `tail -n 80 docs/PROGRESS.md`
- `tail -n 80 docs/TODO.md`

Use syntax checks before committing. Use targeted short simulations or notebook cell reruns for behavior checks when feasible.

## Working Rules
Be cautious by default.

- Read and understand the existing theory implementation before editing nontrivial code.
- Treat the current simulation framework as trusted unless the user explicitly asks to revisit a modeling assumption.
- Do not change model equations, parameter semantics, coupling definitions, or numerical update rules unless the user explicitly asks for that.
- Prefer small, local edits over structural rewrites.
- Preserve scientific meaning over software-style cleanup when the two are in tension.
- When behavior may change, call out the likely modeling consequence clearly.
- Distinguish scientific changes from engineering or organizational changes when summarizing work.
- Trace local consistency after edits: if a variable, state update, coupling term, or figure path changes, inspect where else it is used.
- Preserve figure output locations and existing notebook workflows unless there is a strong reason to change them.
- Avoid silently editing both the active and legacy connectivity files; change only the file relevant to the task.
- If the authoritative source is ambiguous, determine whether the source of truth is the theory, the main implementation, or the figure notebooks before proceeding.

## Testing Guidelines
This repo does not currently have a formal automated test suite. For code edits, use lightweight validation appropriate to the change:
- `conda run -n CANN_DDM_V2 python -m py_compile CANN_DDM_model_rate_based.py make_conn_mat_updated.py`
- targeted imports or short simulation checks when feasible
- notebook re-execution only for cells directly affected by the change

If full scientific validation is not possible in the current turn, state that explicitly.

## Commit & Pull Request Guidelines
The active branch for this project is `CANN_DDM_rate_model`, which tracks `origin/CANN_DDM_rate_model`. When asked to keep GitHub updated, work in this branch and push to the same remote branch over SSH. Do not rewrite remote history unless the user explicitly requests it.

Use concise imperative commit messages, for example: `refine bump-edge coupling logic` or `document figure regeneration workflow`. Pull requests should summarize whether a change is mathematical, numerical, organizational, or figure-related, and should explicitly mention any possible impact on simulation results.

## Operational Rules
If interrupted by a new request, answer quickly when possible; otherwise record the deferred work in `docs/TODO.md` before resuming. Keep `docs/PROGRESS.md` append-only. Keep `docs/STATES.md` limited to validated current state rather than speculative notes.

Before making substantial model changes, build enough local context to understand:
- what theoretical component is being implemented
- which file is authoritative for that component
- whether the requested change is expected to alter numerical behavior or paper-facing outputs

When reporting completed work, make provenance visible: state what changed, why it changed, what was validated, and whether simulation results or figure interpretation may shift.

Maintain `docs/THEORY_UNDERSTANDING.md` as a living checkpoint of the agent's understanding of the user's theory and framework. Write it from scratch rather than copying code comments. Use it to summarize the conceptual model, component roles, intended behaviors, and open uncertainties. Revise it when the user's explanations deepen or correct the agent's interpretation.

## Manuscript Sync Rules
Treat `docs/paper.pdf` as a mirrored artifact built from the external LaTeX manuscript repo, not as the editing source of truth.

- Run `./scripts/sync_paper.sh` before theory- or manuscript-grounded tasks when `docs/paper.pdf` may be stale.
- Run `./scripts/sync_paper.sh` when the user explicitly asks to refresh or reread the manuscript.
- Do not run `./scripts/sync_paper.sh` for unrelated code-only tasks that do not depend on manuscript content.
- If theory, equations, or manuscript-grounded interpretation are central to the task, prefer syncing the paper first when feasible.

## NotebookLM References
Use NotebookLM as the first lookup path for BrainPy, package behavior, framework semantics, or other project-specific documentation questions. Prefer querying this notebook before relying on memory:

- BrainPy / project reference notebook: `https://notebooklm.google.com/notebook/e6340252-320f-4ff9-acc0-10a8456267d9`

Use the local `notebooklm` skill workflow when querying this notebook. If NotebookLM does not fully answer the question, state that explicitly and then fall back to local repository context or official primary documentation.

## Markdown Writing Rules
When writing Markdown notes intended for local reading apps such as Obsidian, use Markdown math syntax for equations instead of fenced code blocks.

- Use inline math like `$x = y$` for short expressions.
- Use display math blocks like `$$ ... $$` for standalone equations.
- Do not use triple-backtick code fences for equations unless the user explicitly asks for code-style formatting.

## Progress And TODO Workflow Rules
When updating project tracking docs, follow this structure and checkpoint behavior.

- `docs/PROGRESS.md` updates should be grouped by date, then by short subsections:
  - `Completed`
  - `Validation`
  - `Known Issues`
  - `Checkpoint`
- Keep `docs/PROGRESS.md` append-only. Do not rewrite or delete older entries; append clarified summaries if readability is needed.
- Ensure `docs/TODO.md` exists. Record pending next actions there with reproducible next steps, and remove or revise stale entries once outcomes are captured in `PROGRESS.md`.
- After each major milestone, explicitly confirm the next major action with the user before starting it.
- If multiple next actions are possible, record them in `docs/TODO.md` and ask the user to choose the next one.
