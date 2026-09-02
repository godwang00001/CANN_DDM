#!/usr/bin/env python3
"""Finalize the Figure 3 v0p3 RT-distribution dataset bundle."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = REPO_ROOT / "results" / "figure3" / "fig3_v0p3_RT_dist"
DEFAULT_PYTHON_BIN = Path(
    os.environ.get(
        "CANN_DDM_PYTHON",
        "/projectnb/ecog-eeg/cyw6/.conda/envs/cann_ddm_v2/bin/python",
    )
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_root",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Root directory containing ddm/ and circuit/ for the v0p3 run",
    )
    parser.add_argument("--ddm-subdir", default="ddm")
    parser.add_argument("--circuit-subdir", default="circuit")
    parser.add_argument("--dataset-name", default="dataset.npz")
    parser.add_argument("--summary-name", default="summary.csv")
    parser.add_argument("--config-name", default="config.json")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--skip-circuit-merge",
        action="store_true",
        help="Assume the circuit sweep-level dataset already exists",
    )
    parser.add_argument(
        "--delete-intermediates",
        action="store_true",
        help="Delete ddm/ and circuit/ after building the combined bundle",
    )
    return parser


def run_step(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    args = make_parser().parse_args()

    run_root = args.run_root.resolve()
    ddm_root = (run_root / args.ddm_subdir).resolve()
    circuit_root = (run_root / args.circuit_subdir).resolve()

    if not ddm_root.exists():
        raise FileNotFoundError(f"Missing DDM root: {ddm_root}")
    if not circuit_root.exists():
        raise FileNotFoundError(f"Missing circuit root: {circuit_root}")

    merge_script = REPO_ROOT / "scripts" / "merge_circuit_psychometric_outputs.py"
    combine_script = REPO_ROOT / "scripts" / "combine_psychometric_model_datasets.py"
    python_bin = str(args.python_bin.resolve())

    circuit_dataset_path = circuit_root / args.dataset_name
    circuit_summary_path = circuit_root / args.summary_name
    circuit_config_path = circuit_root / args.config_name

    if not args.skip_circuit_merge:
        run_step(
            [
                python_bin,
                str(merge_script),
                str(circuit_root),
                "--dataset-name",
                args.dataset_name,
                "--summary-name",
                args.summary_name,
                "--config-name",
                args.config_name,
            ]
        )
    elif not (
        circuit_dataset_path.exists()
        and circuit_summary_path.exists()
        and circuit_config_path.exists()
    ):
        raise FileNotFoundError(
            "Circuit merge was skipped but the sweep-level circuit bundle is incomplete: "
            f"{circuit_dataset_path}, {circuit_summary_path}, {circuit_config_path}"
        )

    combine_cmd = [
        python_bin,
        str(combine_script),
        "--ddm-root",
        str(ddm_root),
        "--circuit-root",
        str(circuit_root),
        "--output-root",
        str(run_root),
        "--dataset-name",
        args.dataset_name,
        "--summary-name",
        args.summary_name,
        "--config-name",
        args.config_name,
    ]
    if args.delete_intermediates:
        combine_cmd.append("--delete-intermediates")
    run_step(combine_cmd)

    print(f"Finished Figure 3 v0p3 merge under {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
